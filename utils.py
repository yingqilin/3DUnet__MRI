import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import cv2


def init_weights(net):
    """
    Initialize model
    :param net: network
    """
    if isinstance(net, nn.Conv3d):
        nn.init.kaiming_normal_(net.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(net.bias.data, 0.0)
    elif isinstance(net, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(net.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(net.bias.data, 0.0)
    elif isinstance(net, nn.BatchNorm3d):
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0.0)
        # net.weight.data.fill_(1)
        # net.bias.data.zero_()
    elif isinstance(net, nn.Linear):
        nn.init.kaiming_normal_(net.weight.data, a=0, mode='fan_in')


def print_net(net):
    """
    Print the structure of network
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print('The structure of the network:')
    # print(net)
    print('Total number of parameters in network: ', num_params)


def Lr_adjustment(optim, epoch, args):
    """
    Adjust learning rate
    The learning rate decays by 10 every 10 epochs
    """
    lr = args.learning_rate * (0.1 ** (epoch // 40))
    for param_group in optim.param_groups:
        param_group['lr'] = lr
    return lr


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label):
        '''
        pred: tensor of size (N, 2, #slices, height, width)
        label: mask of size (N, 1, #slices, height, width)
        '''

        num = pred.size(0)
        smooth = 1

        pred = pred[:, 1]

        # flatten
        pred = pred.view(num, -1)
        label = label.view(num, -1)

        intersection = pred * label
        score = (2. * intersection.sum(dim=1) + smooth) / (pred.sum(dim=1) + label.sum(dim=1) + smooth)
        dice = score.sum() / num
        loss = 1 - dice
        return loss


class TverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label):
        smooth = 1
        alpha = 0.3  # control penalty for FP
        beta = 1 - alpha  # FN
        num = pred.size(0)

        pred = pred[:, 1]

        # flatten
        pred = pred.view(num, -1)
        label = label.view(num, -1)

        true_pos = pred * label
        true_pos = true_pos.sum(dim=1)
        false_neg = (1 - pred) * label
        false_neg = false_neg.sum(dim=1)
        false_pos = pred * (1 - label)
        false_pos = false_pos.sum(dim=1)

        tversky = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
        loss = 1 - tversky.sum() / num

        return loss


class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        '''
        pred: tensor of size (N, 2, #slices, height, width)
        label: mask of size (N, 1, #slices, height, width)
        '''
        loss = DiceLoss()
        dice_loss = loss(pred, target)

        num = pred.size(0)
        pred = pred[:, 1]

        pred = pred.view(num, -1)
        target = target.view(num, -1)

        BCE = F.binary_cross_entropy(pred, target, reduction='mean')
        BCE_exp = torch.exp(-BCE)

        focal_loss = self.alpha * (1 - BCE_exp) ** self.gamma * BCE

        return dice_loss + focal_loss


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        '''
        pred: tensor of size (N, 2, #slices, height, width)
        label: mask of size (N, 1, #slices, height, width)
        '''
        num = pred.size(0)
        smooth = 1

        pred = pred[:, 1]

        pred = pred.view(num, -1)
        target = target.view(num, -1)

        intersection = (pred * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        BCE = F.binary_cross_entropy(pred, target, reduction='mean')

        Dice_BCE = BCE + dice_loss

        return Dice_BCE


def Dice(pred, target):
    '''
    pred: tensor of size (N, #slices, height, width)
    target: tensor of size (N, #slices, height, width)
    '''

    smooth = 1
    num = pred.size(0)
    dice = 0

    for i in range(num):
        inter = torch.sum(pred[i, :, :, :] * target[i, :, :, :])
        union = torch.sum(pred[i, :, :, :]) + torch.sum(target[i, :, :, :])
        dice += (2. * inter + smooth) / (union + smooth)

    dice = dice / num

    return dice


def random_rotate(image, degree):
    scale = 1
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, scale)
    re = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = np.reshape(re, image.shape)
    return out
