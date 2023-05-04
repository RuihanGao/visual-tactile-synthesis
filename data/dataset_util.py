import numpy as np
from PIL import Image
import random

def touch_data_loader(path, convert2im=True,verbose=False, return_mask=True):
    """
    Load data of touch patches from npz files and parse them into data loaders.

    Args:
        path: path to the npz file
        convert2im: option to convert gx, gy to PIL grayscale image to facilitate data processing, assume [-1,1] -> (0,255)
        verbose: option to print out logging information
        return_mask: option to return touch_mask and touch_center_mask. 
        
    The ROI represents the location of the whole GelSight (rectangle) sensing area in the visual image. The masks represent the (irregular) contact area within the sensing area and they can be used for sampling patches with sufficient contact for model training.

    Example npz file format: np.savez("{}/{}_{}_tactile.npz".format(material_cropped_output_dir, material, serial_no), gx_raw=touch_gx_cropped_output, gy_raw=touch_gy_cropped_output, vision_mask_x=vision_mask_x, vision_mask_y=vision_mask_y, vision_mask_h=vision_mask_h, vision_mask_w=vision_mask_w, touch_thresh=touch_thresh_cropped_output, touch_center_thresh=touch_thresh_center_cropped_output) 

    Returns:
        gx: array / PIL grayscale image of gx
        gy: array / PIL grayscale image of gy
        ROI_x: x offset of the rectangle area of GelSight sensor
        ROI_y: y offset of the rectangle area of GelSight sensor
        ROI_h: height of the rectangle area of GelSight sensor
        ROI_w: width of the rectangle area of GelSight sensor
        touch_mask: array of touch mask
        touch_center_mask: array of touch center mask
    """
    npz_data = np.load(path)

    # rectangle sensing of GelSight sensor
    ROI_x = npz_data["vision_mask_x"]
    ROI_y = npz_data["vision_mask_y"]
    ROI_h = npz_data["vision_mask_h"]
    ROI_w = npz_data["vision_mask_w"]
    gx = npz_data["gx_raw"]
    gy = npz_data["gy_raw"]

    if convert2im:
        # convert gx, gy to PIL grayscale image to facilitate data processing, assume [-1,1] -> (0,255)
        gx = Image.fromarray(np.uint8((gx+1)/2*255), 'L')
        gy = Image.fromarray(np.uint8((gy+1)/2*255), 'L')

    if verbose:
        print("Load touch data from", path)
        print("ROI_offset_x {} ROI_offset_y {}, ROI_h {} ROI_w {}".format(ROI_x, ROI_y, ROI_h, ROI_w))
        print("gx {} gy {}".format(gx.shape, gy.shape))

    if return_mask:
        assert 'touch_thresh' in npz_data.files, "touch_thresh not found in npz_data"
        assert 'touch_center_thresh' in npz_data.files, "touch_center_thresh not found in npz_data"
        touch_mask = npz_data["touch_thresh"]; touch_center_mask = npz_data["touch_center_thresh"]
        # normalize touch_mask and touch_center_mask to [0,1] if the max value is 255
        if np.max(touch_mask) > 1:
            touch_mask = touch_mask / 255
        if np.max(touch_center_mask) > 1:
            touch_center_mask = touch_center_mask / 255
    else:
        touch_mask = None; touch_center_mask = None
    return gx, gy, ROI_x, ROI_y, ROI_h, ROI_w, touch_mask, touch_center_mask


### The following functions are for data preprocessing, adapted from VisGel/utils.py ###
def resize(img, size, interpolation=Image.LANCZOS):
    """
    Data preprocessing function, adapted from VisGel/utils.py
    """
    # resize to square shape
    if isinstance(size, int):
        # preserve the aspect ratio, and resize so that the shorter side length matches the predefined size
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        new_h, new_w = size
        # return img.resize(size[::-1], interpolation)
        return img.resize((new_w, new_h), interpolation)

def crop(img, i, j, h, w):
    return img.crop((j, i, j + w, i + h))

def array_crop(arr, i, j, h, w):
    return arr[i:i+h, j:j+w]


def get_crop_params(phase, img, crop_size):
    w, h = img.size
    if isinstance(crop_size, int):
        th, tw = crop_size, crop_size
    else:
        th, tw = crop_size

    if phase == 'train':
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w-tw)

    else:
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

    return i, j, th, tw


def get_crop_params_array(phase, arr, crop_size):
    h, w = arr.shape
    if isinstance(crop_size, int):
        th, tw = crop_size, crop_size
    else:
        th, tw = crop_size

    if phase == 'train':
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w-tw)

    else:
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

    return i, j, th, tw


def resize_and_crop(phase, src, scale_size, crop_size, verbose=False):
    # resize the images
    src = resize(src, scale_size)
    crop_params = get_crop_params(phase, src, crop_size)

    # crop the images
    src = crop(src, crop_params[0], crop_params[1], crop_params[2], crop_params[3])

    return src, crop_params


""" 
For visual-tactile synthesis, we re-implement the data augmentation (zoom and patch) here to accommodate the processing of ROI params, regardless of the transform list in base_dataset.py
"""
def zoom_find_coords(ROI_x, ROI_y, ROI_h, ROI_w, scale_factor_h=1, scale_factor_w=1):
    new_ROI_x = ROI_x*scale_factor_w
    new_ROI_w = ROI_w*scale_factor_w
    new_ROI_y = ROI_y*scale_factor_h
    new_ROI_h = ROI_h*scale_factor_h
    return new_ROI_x, new_ROI_y, new_ROI_h, new_ROI_w

def zoom_img(img, scale_factor_h=1, scale_factor_w=1, method=Image.BICUBIC):
    ow, oh = img.size
    nw, nh = ow*scale_factor_w, oh*scale_factor_h  
    new_img = img.resize((int(round(nw)), int(round(nh))), method)
    return new_img

def get_params(size, crop_size_h=512, crop_size_w=512, center_w=0, center_h=0, center_crop=False):
    w, h = size
    assert w >= crop_size_w and h >= crop_size_h, "The image is smaller than crop_size. Cannot perform get_params for cropping"
    assert crop_size_h >= center_h and crop_size_w >= center_w, "crop_size h {} w {} cannot cover the center region h {} w {}".format(crop_size_h, crop_size_w, center_h, center_w)
    if center_crop:
        # crop from the center
        x = (w - crop_size_w)//2
        y = (h - crop_size_h)//2
    else:
        # random choose starting point (x,y)
        if center_w > 0 or center_h > 0:
            buffer = min(np.maximum(0, (w -center_w)//2), np.maximum(0, (h -center_h)//2), h-crop_size_h, w-crop_size_w)
            x = random.randint(0, buffer)
            y = random.randint(0, buffer)
        else:
            x = random.randint(0, np.maximum(0, w - crop_size_w))
            y = random.randint(0, np.maximum(0, h - crop_size_h))

    return (x,y)


def crop_img(img, crop_size_h, crop_size_w, method=Image.BICUBIC, resize_ratio=None,crop_pos_x=None, crop_pos_y=None, center_w=0,center_h=0, center_crop=False):
    w, h = img.size
    if resize_ratio is None:
        if w >= crop_size_w and h >= crop_size_h:
                resize_ratio = 1
        else:
            resize_ratio = max(crop_size_w/w, crop_size_h/h)
    img = img.resize((int(round(w*resize_ratio)),int(round(h*resize_ratio))), method)

    if crop_pos_x is None and crop_pos_y is None:
        crop_pos_x, crop_pos_y = get_params(img.size, crop_size_h=crop_size_h, crop_size_w=crop_size_w, center_w=center_w, center_h=center_h, center_crop=center_crop)

    new_img = img.crop((crop_pos_x, crop_pos_y, crop_pos_x+crop_size_w, crop_pos_y+crop_size_h))
    return new_img, resize_ratio, crop_pos_x, crop_pos_y


def crop_find_coords(ROI_x, ROI_y, ROI_h, ROI_w, crop_size_h, crop_size_w, resize_ratio, crop_pos_x, crop_pos_y):
    # Define the valid as falling within the center_h  x center_w region
    ROI_x = ROI_x*resize_ratio; ROI_y = ROI_y*resize_ratio
    ROI_h = ROI_h*resize_ratio; ROI_w = ROI_w*resize_ratio

    new_ROI_x = ROI_x-crop_pos_x
    new_ROI_y = ROI_y - crop_pos_y
    # ROI_h and ROI_w remain the same
    if new_ROI_x < 0 or new_ROI_x+ROI_w > crop_size_w:
        return False, new_ROI_x, new_ROI_y, ROI_h, ROI_w 
    elif new_ROI_y < 0 or new_ROI_y+ROI_h > crop_size_h:
        return False, new_ROI_x, new_ROI_y, ROI_h, ROI_w
    else:
        return True, new_ROI_x, new_ROI_y, ROI_h, ROI_w


def make_power_2_img(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        resize_ratio_w = 1
        resizs_ratio_h = 1
        return img, resize_ratio_w, resizs_ratio_h
    else:
        resize_ratio_w = w/ow
        resizs_ratio_h = h/oh
        return img.resize((w, h), method), resize_ratio_w, resizs_ratio_h

def make_power_2_find_coords(ROI_x, ROI_y, ROI_h, ROI_w, resize_ratio_w, resize_ratio_h):
    new_ROI_x = ROI_x*resize_ratio_w
    new_ROI_w = ROI_w*resize_ratio_w
    new_ROI_y = ROI_y*resize_ratio_h
    new_ROI_h = ROI_h*resize_ratio_h
    return new_ROI_x, new_ROI_y, new_ROI_h, new_ROI_w

def global_padding_find_coords(ROI_x, ROI_y, ROI_h, ROI_w, org_w=1280, org_h=960, padded_size=1600):
    new_ROI_x = ROI_x + (padded_size-org_w)//2
    new_ROI_y = ROI_y + (padded_size-org_h)//2
    return new_ROI_x, new_ROI_y, ROI_h, ROI_w