import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    3*3*3 convolution -> BN -> ReLu   (twice)
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    """
    2*2*2 max pooling with strides of 2 -> double convolution
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, mid_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    """
    up-conv -> double convolution
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels+mid_channels, mid_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffC = x2.shape[2] - x1.shape[2]
        diffH = x2.shape[3] - x1.shape[3]
        diffW = x2.shape[4] - x1.shape[4]

        x1 = F.pad(x1, (diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2, diffC // 2, diffC - diffC // 2))
        x = torch.cat([x2, x1], dim=1)

        return self.conv.forward(x)


class OutLayer(nn.Module):
    """
    1*1*1 convolution
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=2, kernel_size=1, stride=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_labels, training):
        super(UNet3d, self).__init__()
        self.in_channels = in_channels
        self.n_labels = n_labels
        self.training = training

        ''' encoder '''
        self.conv1 = DoubleConv(in_channels, 32, 64)
        self.conv2 = DownSample(64, 64, 128)
        self.conv3 = DownSample(128, 128, 256)
        self.conv4 = DownSample(256, 256, 512)

        ''' decoder '''
        self.up1 = UpSample(512, 256, 256)
        self.up2 = UpSample(256, 128, 128)
        self.up3 = UpSample(128, 64, 64)
        self.out = OutLayer(64)

    def forward(self, x):
        x1 = self.conv1(x)
        # print('The shape of x1: ', x1.shape)
        x2 = self.conv2(x1)
        # print('The shape of x2: ', x2.shape)
        x3 = self.conv3(x2)
        # print('The shape of x3: ', x3.shape)
        x4 = self.conv4(x3)
        # print('The shape of x4: ', x4.shape)
        x = self.up1(x4, x3)
        # print(x.shape)
        x = self.up2(x, x2)
        # print(x.shape)
        x = self.up3(x, x1)
        # print(x.shape)
        output = self.out(x)
        return output