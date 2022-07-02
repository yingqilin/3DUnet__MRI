import os
import SimpleITK as sitk
from params import args

# Author: Yingqi Lin

# split training data

file_list = os.listdir(args.data_path)
n = len(file_list)
k = 1

# folder to store new data
new_train_path = os.path.join(args.save_path, 'train_dataset_processed')
if not os.path.exists(new_train_path):
    os.mkdir(new_train_path)

for i in range(n):
    public_path = os.path.join(args.data_path, file_list[i])
    image_path = os.path.join(public_path, 'image.nii.gz')
    label_path = os.path.join(public_path, 'labels.nii.gz')

    # read original images and masks
    image = sitk.ReadImage(image_path, sitk.sitkInt16)
    label = sitk.ReadImage(label_path, sitk.sitkInt8)
    image_arr = sitk.GetArrayFromImage(image)
    label_arr = sitk.GetArrayFromImage(label)

    # separate into 2 parts
    s, h, w = image_arr.shape
    middle = w // 2
    image1 = image_arr[:, :, 0:middle]
    label1 = label_arr[:, :, 0:middle]
    image2 = image_arr[:, :, middle:]
    label2 = label_arr[:, :, middle:]

    # store MRI
    new1 = os.path.join(new_train_path, 'new_' + str(k))
    if not os.path.exists(new1):
        os.mkdir(new1)
    k += 1

    new2 = os.path.join(new_train_path, 'new_' + str(k))
    if not os.path.exists(new2):
        os.mkdir(new2)
    k += 1

    # -----------------------------------------------------------------
    new_image = sitk.GetImageFromArray(image1)
    new_label = sitk.GetImageFromArray(label1)

    new_image.SetDirection(image.GetDirection())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing(image.GetSpacing())

    new_label.SetDirection(label.GetDirection())
    new_label.SetOrigin(label.GetOrigin())
    new_label.SetSpacing(label.GetSpacing())

    sitk.WriteImage(new_image, os.path.join(new1, 'image.nii.gz'))
    sitk.WriteImage(new_label, os.path.join(new1, 'labels.nii.gz'))

    # -----------------------------------------------------------------
    new_image = sitk.GetImageFromArray(image2)
    new_label = sitk.GetImageFromArray(label2)

    new_image.SetDirection(image.GetDirection())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing(image.GetSpacing())

    new_label.SetDirection(label.GetDirection())
    new_label.SetOrigin(label.GetOrigin())
    new_label.SetSpacing(label.GetSpacing())

    sitk.WriteImage(new_image, os.path.join(new2, 'image.nii.gz'))
    sitk.WriteImage(new_label, os.path.join(new2, 'labels.nii.gz'))


# split validation data
file_list = os.listdir(args.val_path)
n = len(file_list)
k = 1

# folder to store new data
new_val_path = os.path.join(args.save_path, 'val_dataset_processed')
if not os.path.exists(new_val_path):
    os.mkdir(new_val_path)

for i in range(n):
    public_path = os.path.join(args.val_path, file_list[i])
    image_path = os.path.join(public_path, 'imaging.nii.gz')
    label_path = os.path.join(public_path, 'segmentation.nii.gz')

    # read original images and masks
    image = sitk.ReadImage(image_path, sitk.sitkInt16)
    label = sitk.ReadImage(label_path, sitk.sitkInt8)
    image_arr = sitk.GetArrayFromImage(image)
    label_arr = sitk.GetArrayFromImage(label)

    # separate into 2 parts
    s, h, w = image_arr.shape
    middle = w // 2
    image1 = image_arr[:, :, 0:middle]
    label1 = label_arr[:, :, 0:middle]
    image2 = image_arr[:, :, middle:]
    label2 = label_arr[:, :, middle:]

    # store MRI
    new1 = os.path.join(new_val_path, 'new_' + str(k))
    if not os.path.exists(new1):
        os.mkdir(new1)
    k += 1

    new2 = os.path.join(new_val_path, 'new_' + str(k))
    if not os.path.exists(new2):
        os.mkdir(new2)
    k += 1

    # -----------------------------------------------------------------
    new_image = sitk.GetImageFromArray(image1)
    new_label = sitk.GetImageFromArray(label1)

    new_image.SetDirection(image.GetDirection())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing(image.GetSpacing())

    new_label.SetDirection(label.GetDirection())
    new_label.SetOrigin(label.GetOrigin())
    new_label.SetSpacing(label.GetSpacing())

    sitk.WriteImage(new_image, os.path.join(new1, 'imaging.nii.gz'))
    sitk.WriteImage(new_label, os.path.join(new1, 'segmentation.nii.gz'))

    # -----------------------------------------------------------------
    new_image = sitk.GetImageFromArray(image2)
    new_label = sitk.GetImageFromArray(label2)

    new_image.SetDirection(image.GetDirection())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing(image.GetSpacing())

    new_label.SetDirection(label.GetDirection())
    new_label.SetOrigin(label.GetOrigin())
    new_label.SetSpacing(label.GetSpacing())

    sitk.WriteImage(new_image, os.path.join(new2, 'imaging.nii.gz'))
    sitk.WriteImage(new_label, os.path.join(new2, 'segmentation.nii.gz'))


# split test data
file_list = os.listdir(args.test_path)
n = len(file_list)

# folder to store new data
new_test_path = os.path.join(args.save_path, 'test_dataset_processed')
if not os.path.exists(new_test_path):
    os.mkdir(new_test_path)

for i in range(n):
    public_path = os.path.join(args.test_path, file_list[i])
    image_path = os.path.join(public_path, 'imaging.nii.gz')
    label_path = os.path.join(public_path, 'segmentation.nii.gz')

    # read original images and masks
    image = sitk.ReadImage(image_path, sitk.sitkInt16)
    label = sitk.ReadImage(label_path, sitk.sitkInt8)
    image_arr = sitk.GetArrayFromImage(image)
    label_arr = sitk.GetArrayFromImage(label)

    # separate into 2 parts
    s, h, w = image_arr.shape
    middle = w // 2
    image1 = image_arr[:, :, 0:middle]
    label1 = label_arr[:, :, 0:middle]
    image2 = image_arr[:, :, middle:]
    label2 = label_arr[:, :, middle:]

    # store MRI
    new1 = os.path.join(new_test_path, file_list[i] + '-new1')
    if not os.path.exists(new1):
        os.mkdir(new1)

    new2 = os.path.join(new_test_path, file_list[i] + '-new2')
    if not os.path.exists(new2):
        os.mkdir(new2)

    # -----------------------------------------------------------------
    new_image = sitk.GetImageFromArray(image1)
    new_label = sitk.GetImageFromArray(label1)

    new_image.SetDirection(image.GetDirection())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing(image.GetSpacing())

    new_label.SetDirection(label.GetDirection())
    new_label.SetOrigin(label.GetOrigin())
    new_label.SetSpacing(label.GetSpacing())

    sitk.WriteImage(new_image, os.path.join(new1, 'imaging.nii.gz'))
    sitk.WriteImage(new_label, os.path.join(new1, 'segmentation.nii.gz'))

    # -----------------------------------------------------------------
    new_image = sitk.GetImageFromArray(image2)
    new_label = sitk.GetImageFromArray(label2)

    new_image.SetDirection(image.GetDirection())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing(image.GetSpacing())

    new_label.SetDirection(label.GetDirection())
    new_label.SetOrigin(label.GetOrigin())
    new_label.SetSpacing(label.GetSpacing())

    sitk.WriteImage(new_image, os.path.join(new2, 'imaging.nii.gz'))
    sitk.WriteImage(new_label, os.path.join(new2, 'segmentation.nii.gz'))
