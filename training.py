from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.optim as optim
import os
import numpy as np
from train_dataset import MRI_Dataset
from unet3d import UNet3d
from ResUnet import ResUnet3d
from ResUnet_CBAM import ResUnet_CBAM
from logger import get_Train_Logger, get_Test_Logger
from params import args
from utils import Dice, DiceLoss, DiceFocalLoss, DiceBCELoss, init_weights, print_net

# Author: Yingqi Lin

def train(net, X_train_Loader, optimizer, loss_func):
    net.train()
    train_loss = 0
    dice0 = 0
    dice1 = 0

    for i, (X_train, y_train) in tqdm(enumerate(X_train_Loader), total=len(X_train_Loader)):
        X_train = X_train.float()
        y_train = y_train.float()
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        y_pred = net(X_train)

        loss = loss_func(y_pred, y_train)
        train_loss += loss.item()

        # back-prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metric
        pred = torch.argmax(y_pred, dim=1)
        d1 = Dice(pred, y_train[:, 0])
        dice1 += d1.item()
        d0 = Dice(1 - pred, 1 - y_train[:, 0])
        dice0 += d0.item()

        torch.cuda.empty_cache()

    train_loss_avg = train_loss / len(X_train_Loader)
    dice0_avg = dice0 / len(X_train_Loader)
    dice1_avg = dice1 / len(X_train_Loader)

    return train_loss_avg, dice0_avg, dice1_avg


def val(net, val_loader, loss_func):
    net.eval()
    val_loss = 0
    dice0 = 0
    dice1 = 0
    dice_arr = np.zeros((len(val_loader), 1))
    k = 0

    with torch.no_grad():
        for i, (X_eval, y_eval) in tqdm(enumerate(val_loader), total=len(val_loader)):
            X_eval = X_eval.float()
            y_eval = y_eval.float()
            X_eval = X_eval.to(device)
            y_eval = y_eval.to(device)

            y = net(X_eval)
            loss = loss_func(y, y_eval)
            val_loss += loss.item()

            logit = torch.argmax(y, dim=1)
            dice = Dice(logit, y_eval[:, 0])
            dice1 += dice.item()
            dice_arr[k] = dice.item()
            k += 1
            dice = Dice(1 - logit, 1 - y_eval[:, 0])
            dice0 += dice.item()

            torch.cuda.empty_cache()

    val_loss_avg = val_loss / len(val_loader)
    dice0_avg = dice0 / len(val_loader)
    dice1_avg = dice1 / len(val_loader)

    return val_loss_avg, dice0_avg, dice1_avg, dice_arr


if __name__ == '__main__':
    new_train_path = os.path.join(args.folder_path, 'train_dataset_processed')
    new_val_path = os.path.join(args.folder_path, 'val_dataset_processed')
    data_path = new_train_path
    val_path = new_val_path
    save_path = args.save_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(device)
    print('Cuda is available: ', torch.cuda.is_available())

    # Load dataset
    train_loader = DataLoader(dataset=MRI_Dataset(data_path, do_augmentation=True, training=True),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=MRI_Dataset(val_path, do_augmentation=False, training=False), batch_size=1,
                            shuffle=False)

    # model
    net = UNet3d(in_channels=1, n_labels=2, training=True).to(device)
    net.apply(init_weights)
    print_net(net)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.001)

    loss = DiceLoss()

    save_checkpoint = True
    best_loss = 100
    logger = get_Train_Logger()
    logger.info('======== Start training ========')

    loss_train_arr = np.ones((args.epochs, 1))
    dice_train_arr = np.ones((args.epochs, 1))
    loss_val_arr = np.ones((args.epochs, 1))
    dice_val_arr = np.ones((args.epochs, 1))

    for epoch in range(1, args.epochs + 1):
        # temp_lr = utils.Lr_adjustment(optimizer, epoch, args)
        train_loss, train_dice0, train_dice1 = train(net, train_loader, optimizer, loss)
        logger.info('Epoch:{} / {}'.format(epoch, args.epochs))
        logger.info('Training Loss:{} || Dice0:{} || Dice1:{}'.format(train_loss, train_dice0, train_dice1))
        print('Epoch:{} / {}'.format(epoch, args.epochs))
        print('Training Loss:{} || Dice0:{} || Dice1:{}'.format(train_loss, train_dice0, train_dice1))
        loss_train_arr[epoch - 1] = train_loss
        dice_train_arr[epoch - 1] = train_dice1

        val_loss, val_dice0, val_dice1, arr = val(net, val_loader, loss)
        logger.info('Validation Loss:{} || Dice0:{} || Dice1:{}'.format(val_loss, val_dice0, val_dice1))
        print('Validation Loss:{} || Dice0:{} || Dice1:{}'.format(val_loss, val_dice0, val_dice1))
        print(arr)
        loss_val_arr[epoch - 1] = val_loss
        dice_val_arr[epoch - 1] = val_dice1

        torch.save(net, os.path.join(save_path, 'latest_model.pth'))

        if save_checkpoint & (val_loss <= best_loss):
            best_loss = val_loss
            torch.save(net, os.path.join(save_path, 'best_model.pth'))
            logger.info('Checkpoint saved!')
            print('Checkpoint saved!')

        torch.cuda.empty_cache()

    logger.info('======== Finish training ========')



