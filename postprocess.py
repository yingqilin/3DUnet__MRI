import os
import torch
import numpy as np
import SimpleITK as sitk
from params import args
from utils import Dice
from metrics import F1

# Yingqi Lin

save_path = args.save_path
patch_path = os.path.join(save_path, 'prediction')

final_label_path = os.path.join(save_path, 'labels')
if not os.path.exists(final_label_path):
    os.mkdir(final_label_path)

patch_list = os.listdir(patch_path)
patch_list.sort(key=lambda x: int(x[:2]))
num = len(patch_list)

k = 1

for i in range(0, num, 2):
    assert patch_list[i][:2] == patch_list[i + 1][:2]
    path1 = os.path.join(patch_path, patch_list[i])
    path2 = os.path.join(patch_path, patch_list[i + 1])

    label1 = sitk.ReadImage(path1, sitk.sitkInt8)
    label2 = sitk.ReadImage(path2, sitk.sitkInt8)
    label1_arr = sitk.GetArrayFromImage(label1)
    label2_arr = sitk.GetArrayFromImage(label2)

    x0, y0, z0 = label1_arr.shape
    _, _, z1 = label2_arr.shape

    full_label = np.zeros((x0, y0, z0 + z1))
    full_label[:, :, :z0] = label1_arr
    full_label[:, :, z0:] = label2_arr

    full_label = sitk.GetImageFromArray(full_label)

    full_label.SetDirection(label1.GetDirection())
    full_label.SetOrigin(label1.GetOrigin())
    full_label.SetSpacing(label1.GetSpacing())

    folder_path = os.path.join(final_label_path, patch_list[i][:2])
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    sitk.WriteImage(full_label, os.path.join(folder_path, 'segmentation.nii.gz'))

    print('Generate segmentation {} consisting of patches {} and {}'.format(patch_list[i][:2], patch_list[i], patch_list[i + 1]))

# evaluation
final_label_path = os.path.join(args.save_path, 'labels')
test_path = args.test_path

pred_list = os.listdir(final_label_path)
seg_list = os.listdir(test_path)
num = len(pred_list)

avg_dice = 0
avg_pre = 0
avg_re = 0
avg_f1 = 0

for i in range(0, num):
    pred_path = os.path.join(final_label_path, pred_list[i])
    pred_path = os.path.join(pred_path, 'segmentation.nii.gz')

    for j in range(0, num):
        if pred_list[i] == seg_list[j]:
            seg_path = os.path.join(test_path, seg_list[j])
            break

    seg_path = os.path.join(seg_path, 'segmentation.nii.gz')
    assert pred_list[i] == seg_list[j]

    pred = sitk.ReadImage(pred_path, sitk.sitkInt8)
    pred = sitk.GetArrayFromImage(pred)
    pred = pred.astype(float)

    mask = sitk.ReadImage(seg_path, sitk.sitkInt8)
    mask = sitk.GetArrayFromImage(mask)
    mask = mask.astype(float)

    pred = torch.FloatTensor(pred).unsqueeze(0)
    mask = torch.FloatTensor(mask).unsqueeze(0)
    # print(pred.shape, mask.shape)

    dice1 = Dice(pred, mask)
    dice0 = Dice(1 - pred, 1 - mask)
    precision, recall, f1 = F1(pred, mask)

    print('Segmentation {} -- Prediction{}'.format(seg_list[j], pred_list[i]))
    print('Dice0:{} || Dice1:{} || precision:{} || recall:{} || F1 score:{}'.format(dice0, dice1, precision, recall, f1))

    avg_dice += dice1
    avg_pre += precision
    avg_re += recall
    avg_f1 += f1

avg_dice = avg_dice / num
avg_pre = avg_pre / num
avg_re = avg_re / num
avg_f1 = avg_f1 / num
print('Average Metrics:')
print('Dice1:{} || precision:{} || recall:{} || F1 score:{}'.format(avg_dice, avg_pre, avg_re, avg_f1))