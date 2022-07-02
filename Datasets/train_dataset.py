import os
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from preprocess import process


class MRI_Dataset(Dataset):
    def __init__(self, data_dir, do_augmentation, training):
        self.data_dir = data_dir
        self.file_list = os.listdir(self.data_dir)
        self.num = len(self.file_list)
        self.do_augmentation = do_augmentation
        self.training = training

    def __len__(self):
        """
        return the number of samples.
        """
        return self.num

    def __getitem__(self, item):
        """
        Read the corresponding image and label
        :param item: index for training data
        :return: image tensor of size (C, D, H, W) = (1, #slices, H, W)
             label tensor of size (#slices, H, W)
        """
        public_path = os.path.join(self.data_dir, self.file_list[item])

        if self.training:
            image_path = os.path.join(public_path, 'image.nii.gz')  # training data
            label_path = os.path.join(public_path, 'labels.nii.gz')
        else:
            image_path = os.path.join(public_path, 'imaging.nii.gz')
            label_path = os.path.join(public_path, 'segmentation.nii.gz')  # validation data

        image = sitk.ReadImage(image_path, sitk.sitkInt16)
        label = sitk.ReadImage(label_path, sitk.sitkInt8)
        data_image = sitk.GetArrayFromImage(image)
        data_label = sitk.GetArrayFromImage(label)

        image_arr = data_image.astype(float)
        label_arr = data_label.astype(float)

        image_proc, label_proc = process(image_arr, label_arr, self.do_augmentation, self.training)

        # add one dimension (C)
        image_tensor = torch.FloatTensor(image_proc).unsqueeze(0)
        label_tensor = torch.FloatTensor(label_proc).unsqueeze(0)

        # print('The shape of image is: ', image_tensor.shape)
        # print('The shape of label is: ', label_tensor.shape)

        return image_tensor, label_tensor