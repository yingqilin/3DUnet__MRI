import random
import numpy as np
from params import args
from utils import random_rotate


def crop_volume_allDim(image, label, x, y, z):
    """
    Strip away the zeros on the edges of the three dimensions of the image
    Idea: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132934
    Reference: https://github.com/RobinBruegger/PartiallyReversibleUnet/blob/master/dataProcessing/brats18_data_loader.py
    """

    '''
    Do padding if ROI is too small. The size after padding is: (x, y, z)
    '''
    coords = np.argwhere(label > 0)
    if coords.size == 0:  # do not crop roi
        return image, label, True

    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1
    # print(x0, y0, z0, x1, y1, z1)

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

    return cropped_image, cropped_label, False


def random_crop_3d(img, label, x_size, y_size, z_size):
    """
    Random crop
    :param x_size: desired #slices
    :param y_size: desired height
    :param z_size: desired width
    :return: numpy array of size (x_size, y_size, z_size)
    """
    x, y, z = img.shape

    x_max = x - x_size
    y_max = y - y_size
    z_max = z - z_size

    # random start point in original image
    start_x = random.randint(0, x_max)
    start_y = random.randint(0, y_max)
    start_z = random.randint(0, z_max)

    # copy to resized image
    img_crop = img[start_x:start_x + x_size, start_y:start_y + y_size, start_z:start_z + z_size]
    label_crop = label[start_x:start_x + x_size, start_y:start_y + y_size, start_z:start_z + z_size]

    return img_crop, label_crop


def augmentation(img, label, do_rotate, do_flip, rotDegree):
    """
    Reference: https://github.com/RobinBruegger/PartiallyReversibleUnet/blob/master/dataProcessing/augmentation.py
    """
    # flip
    if do_flip:
        ''' upside down '''
        ran = random.uniform(0, 1)
        if ran <= 0.5:
            img = np.flip(img, axis=2)
            label = np.flip(label, axis=2)
        ''' Left-right '''
        ran = random.uniform(0, 1)
        if ran <= 0.5:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=1)

    # rotate
    if do_rotate:
        image = img.copy()
        mask = label.copy()
        random_degree = random.uniform(-rotDegree, rotDegree)

        for i in range(image.shape[0]):
            image[i, :, :] = random_rotate(image[i, :, :], random_degree)
            mask[i, :, :] = random_rotate(mask[i, :, :], random_degree)

    return image, mask


def normalization(img):
    """
    Only consider non-zero values in image
    """
    # For value == 0, replace it with nan; otherwise, keep the original value
    # Compute mean and standard deviation only for non-nan value
    m = np.nanmean(np.where(img == 0, np.nan, img)).astype(float)
    sigma = np.nanstd(np.where(img == 0, np.nan, img)).astype(float)
    normalized = np.divide((img - m), sigma)

    # Replace non-zero value with normalized value
    img = np.where(img == 0, 0, normalized)
    return img


def process(image_arr, label_arr, aug, train):
    """
    Main function for preprocessing.
    :param image_arr: numpy array of size (#slices, H, W)
    :param label_arr: numpy array of size (#slices, H, W)
    :return: image and label (numpy array) of size (#slices, H, W)
    """
    # args = params.args
    upper = args.upper
    x_size = args.x_roi_size
    y_size = args.y_roi_size
    z_size = args.z_roi_size
    x_crop = args.x_padding
    y_crop = args.y_padding
    z_crop = args.z_padding
    do_flip = args.do_flip
    do_rotate = args.do_rotate
    rotDegree = args.rotDegree

    assert len(image_arr.shape) == 3
    assert len(label_arr.shape) == 3

    image_arr[image_arr > upper] = upper
    image_arr = image_arr / upper

    if train:
        image_arr, label_arr, is_none = crop_volume_allDim(image_arr, label_arr, x_crop, y_crop, z_crop)
        image_roi, label_roi = random_crop_3d(image_arr, label_arr, x_size, y_size, z_size)
    else:
        image_roi, label_roi, is_none = crop_volume_allDim(image_arr, label_arr, x_size, y_size, z_size)  # validation
        if is_none:
            image_roi, label_roi = random_crop_3d(image_arr, label_arr, x_size, y_size, z_size)
        # image_roi, label_roi = random_divide(image_arr, label_arr)

    if aug:
        image_aug, label_aug = augmentation(image_roi, label_roi, do_rotate, do_flip, rotDegree)
        # image_aug = normalization(image_aug)
        return image_aug, label_aug
    else:
        # image_roi = normalization(image_roi)
        return image_roi, label_roi