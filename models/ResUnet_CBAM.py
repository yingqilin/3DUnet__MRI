import torch
import torch.nn as nn
import torch.nn.functional as F


# Reference: https://github.com/Jongchan/attention-module

class Flatten(nn.Module):
    # input: (batch_size, C, 1, 1, 1)
    # output: (batch_size, C)
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.gate_channels = gate_channels
        self.ratio = reduction_ratio
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(self.gate_channels, self.gate_channels // self.ratio),
            nn.ReLU(),
            nn.Linear(self.gate_channels // self.ratio, self.gate_channels)
        )

    def forward(self, x):
        avg_pool = F.avg_pool3d(x, kernel_size=(x.size(2), x.size(3), x.size(4)),
                                stride=(x.size(2), x.size(3), x.size(4)))
        channel_att_raw = self.mlp(avg_pool)
        channel_att_sum = channel_att_raw
        max_pool = F.max_pool3d(x, kernel_size=(x.size(2), x.size(3), x.size(4)),
                                stride=(x.size(2), x.size(3), x.size(4)))
        channel_att_raw = self.mlp(max_pool)
        channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)

        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2),
            nn.BatchNorm3d(1, eps=1e-5, momentum=0.01, affine=True)
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, no_spatial=False):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.no_spatial = no_spatial
        if not self.no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)

        return x_out


class FirstLayer(nn.Module):
    """
    3*3 Convolution -> BN -> ReLU -> 3*3 Convolution -> CBAM -> Addition
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            CBAM(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        return self.conv_block(x) + self.shortcut(x)


class ResBlock(nn.Module):
    """
    BN -> ReLU -> 3*3 Convolution -> BN -> ReLU -> 3*3 Convolution -> CBAM -> Addition
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            CBAM(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        return self.conv_block(x) + self.shortcut(x)


class UpBlock(nn.Module):
    """
    up-sampling -> res_block (with CBAM)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sampling = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.up_conv_block = nn.Sequential(
            nn.BatchNorm3d(in_channels + out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels + out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            CBAM(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels + out_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x1, x2):
        x1 = self.up_sampling(x1)

        diffC = x2.shape[2] - x1.shape[2]
        diffH = x2.shape[3] - x1.shape[3]
        diffW = x2.shape[4] - x1.shape[4]

        x1 = F.pad(x1, (diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2, diffC // 2, diffC - diffC // 2))
        x = torch.cat([x2, x1], dim=1)

        return self.up_conv_block(x) + self.shortcut(x)


class OutLayer(nn.Module):
    """
    1*1*1 convolution -> softmax
    """

    def __init__(self, in_channels, num_labels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=num_labels, kernel_size=1, stride=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.conv(x)


class ResUnet_CBAM(nn.Module):
    def __init__(self, in_channels, num_labels, training):
        super(ResUnet_CBAM, self).__init__()
        self.in_channels = in_channels
        self.num_labels = num_labels
        self.training = training

        ''' encoder '''
        self.enc1 = FirstLayer(in_channels, 64)
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)

        ''' bridge '''
        self.bridge = ResBlock(256, 512)

        ''' decoder '''
        self.dec1 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec3 = UpBlock(128, 64)
        self.out = OutLayer(64, self.num_labels)

    def forward(self, x):
        x1 = self.enc1(x)
        # print('The shape of x1: ', x1.shape)
        x2 = self.enc2(x1)
        # print('The shape of x2: ', x2.shape)
        x3 = self.enc3(x2)
        # print('The shape of x3: ', x3.shape)
        x4 = self.bridge(x3)
        # print('The shape of x4: ', x4.shape)
        x = self.dec1(x4, x3)
        # print(x.shape)
        x = self.dec2(x, x2)
        # print(x.shape)
        x = self.dec3(x, x1)
        # print(x.shape)
        output = self.out(x)
        # print('The shape of output: ', output.shape)
        return output