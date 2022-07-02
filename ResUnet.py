import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstLayer(nn.Module):
    """
    3*3 Convolution -> BN -> ReLU -> 3*3 Convolution -> Addition
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        return self.conv_block(x) + self.shortcut(x)


class ResBlock(nn.Module):
    """
    BN -> ReLU -> 3*3 Convolution -> BN -> ReLU -> 3*3 Convolution -> Addition
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        return self.conv_block(x) + self.shortcut(x)


class UpBlock(nn.Module):
    """
    up-sampling -> res_block
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
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
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


class ResUnet3d(nn.Module):
    def __init__(self, in_channels, num_labels, training):
        super(ResUnet3d, self).__init__()
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