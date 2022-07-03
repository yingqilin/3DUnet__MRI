import os
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
from params import args


class Test_Dataset(Dataset):
    def __init__(self, image_path, label_path, args):

        self.x_size = args.x_roi_size
        self.y_size = args.y_roi_size
        self.z_size = args.z_roi_size
        self.upper = args.upper

        self.image = sitk.ReadImage(image_path)
        self.label = sitk.ReadImage(label_path)
        self.image_arr = sitk.GetArrayFromImage(self.image)
        self.label_arr = sitk.GetArrayFromImage(self.label)

        self.image_arr = self.image_arr.astype(float)
        self.label_arr = self.label_arr.astype(float)

        self.image_arr[self.image_arr > self.upper] = self.upper
        self.image_arr = self.image_arr / self.upper
        self.shape_ori = self.image_arr.shape

        self.image_cropped, self.label_cropped, self.pos = self.crop_volume_allDim_test(self.image_arr, self.label_arr,
                                                                                        self.x_size, self.y_size,
                                                                                        self.z_size)

        self.Origin = self.label.GetOrigin()
        self.Spacing = self.label.GetSpacing()
        self.Direction = self.label.GetDirection()

    def __len__(self):

        return 1

    def __getitem__(self, item):
        """
        return the cropped roi and the original label
        tensor of size (1, depth, height, width)
        """
        self.image_cropped = torch.FloatTensor(self.image_cropped).unsqueeze(0)
        self.label_arr = torch.FloatTensor(self.label_arr).unsqueeze(0)

        return self.image_cropped, self.label_arr

    def crop_volume_allDim_test(self, image, label, x, y, z):

        coords = np.argwhere(label > 0)
        if coords.size == 0:  # do not contain target
            return self.center_crop(image, label, x, y, z)

        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1

        if (x1 - x0) < x:
            expand_x = (x - (x1 - x0)) // 2
            x0 = max(0, x0 - expand_x)
            x1 = x0 + x

        if (y1 - y0) < y:
            expand_y = (y - (y1 - y0)) // 2
            y0 = max(0, y0 - expand_y)
            y1 = y0 + y

        if (z1 - z0) < z:
            expand_z = (z - (z1 - z0)) // 2
            z0 = max(0, z0 - expand_z)
            z1 = z0 + z

        cropped_label = label[x0:x1, y0:y1, z0:z1]
        cropped_image = image[x0:x1, y0:y1, z0:z1]

        coord = np.array([x0, x1, y0, y1, z0, z1])

        return cropped_image, cropped_label, coord

    def center_crop(self, image, label, x, y, z):

        s, h, w = image.shape

        x0 = (s - x) // 2
        y0 = (h - y) // 2
        z0 = (w - z) // 2

        x1 = x0 + x
        y1 = y0 + y
        z1 = z0 + z

        image_cropped = image[x0:x1, y0:y1, z0:z1]
        label_cropped = label[x0:x1, y0:y1, z0:z1]

        coords = np.array([x0, x1, y0, y1, z0, z1])

        return image_cropped, label_cropped, coords

    def recover(self, logit):
        """
        logit: prediction of roi                    (1, shape of roi)
        :return: prediction corresponding to orginal image size   (1, depth, height, width)
        """
        depth, height, width = self.shape_ori
        x0, x1, y0, y1, z0, z1 = self.pos

        full_pred = torch.zeros((1, depth, height, width))

        full_pred[:, x0:x1, y0:y1, z0:z1] = logit[0, :, :, :]

        return full_pred