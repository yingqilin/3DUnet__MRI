import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import SimpleITK as sitk
import torch
import os
from test_dataset import Test_Dataset
from unet3d import UNet3d, DoubleConv, DownSample, OutLayer, UpSample
from logger import get_Test_Logger
from params import args
from utils import Dice
from metrics import F1


def predict(model, one_img_dataset, args):
    test_loader = DataLoader(dataset=one_img_dataset, batch_size=1, shuffle=False)
    model.eval()

    with torch.no_grad():
        for image, mask in tqdm(test_loader, total=len(test_loader)):
            image = image.float()
            mask = mask.long()  # 1, 1, depth, height, width
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)
            logit = torch.argmax(output, dim=1)  # 1, depth, height, width
            logit = one_img_dataset.recover(logit)

        logit = logit.detach().cpu()
        mask = mask.detach().cpu()
        dice1 = Dice(logit, mask[:, 0]).item()
        dice0 = Dice(1 - logit, 1 - mask[:, 0]).item()
        precision, recall, f1 = F1(logit, mask)
        # iou = IOU(logit.unsqueeze(0), target)

        logit = logit.squeeze(0)  # depth, height, width
        logit = logit.numpy()
        pred_img = logit.astype(np.uint8)
        assert len(pred_img.shape) == 3

    img = sitk.GetImageFromArray(pred_img)
    img.SetOrigin(one_img_dataset.Origin)
    img.SetSpacing(one_img_dataset.Spacing)
    img.SetDirection(one_img_dataset.Direction)

    return dice0, dice1, precision, recall, f1, img


if __name__ == '__main__':
    # args = params.args
    new_test_path = os.path.join(args.folder_path, 'test_dataset_processed')
    save_path = args.save_path
    test_path = new_test_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Cuda is available: ', torch.cuda.is_available())

    # Load model
    model = torch.load(os.path.join(save_path, 'best_model.pth'), map_location=device)
    print('Model is loaded!')

    labels_save_path = os.path.join(save_path, 'prediction')
    if not os.path.exists(labels_save_path):
        os.mkdir(labels_save_path)

    test_logger = get_Test_Logger()
    test_logger.info('======== Start testing ========')

    test_list = os.listdir(test_path)
    num_test_list = len(test_list)
    # print('The number of testing samples:{}'.format(num_test_list))

    for i in range(num_test_list):
        public_path = os.path.join(test_path, test_list[i])

        image_path = os.path.join(public_path, 'imaging.nii.gz')
        label_path = os.path.join(public_path, 'segmentation.nii.gz')

        print('Start testing image: {}'.format(test_list[i]))
        test_logger.info('Evaluate image: {}'.format(test_list[i]))

        one_img_dataset = Test_Dataset(image_path, label_path, args)
        dice0, dice1, precision, recall, f1, pred_img = predict(model, one_img_dataset, args)

        test_logger.info(
            'Dice0:{} || Dice1:{} || precision:{} || recall:{} || F1 score:{}'.format(dice0, dice1, precision, recall,
                                                                                      f1))
        print('Dice0:{} || Dice1:{} || precision:{} || recall:{} || F1 score:{}'.format(dice0, dice1, precision, recall,
                                                                                        f1))

        sitk.WriteImage(pred_img, os.path.join(labels_save_path, '{}-labels.nii.gz'.format(test_list[i])))
        print('Result is stored!')

        torch.cuda.empty_cache()

    test_logger.info('======== Finish testing ========')
