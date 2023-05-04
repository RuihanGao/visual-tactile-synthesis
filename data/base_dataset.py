"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
from matplotlib.pyplot import grid
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        self.current_epoch = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt,params=None, grayscale=False, method=Image.BICUBIC, convert=True, magnification=1, verbose=False, normalize=True, load_size=None, crop_size=None):
    """
    Return a list of transformations for image preprocessing
    For visual-tactile synthesis project, we add a mangification factor which can scale up/down the image size for different kinds of input.
    """
    transform_list = []
    if load_size is None:
        load_size = opt.load_size * magnification
    if crop_size is None:
        crop_size = opt.crop_size * magnification

    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'fixsize' in opt.preprocess:
        transform_list.append(transforms.Resize(params["size"], method))
    if 'resize' in opt.preprocess:
        osize = [load_size, load_size]
        if "gta2cityscapes" in opt.dataroot:
            osize[0] = load_size // 2
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, crop_size, method)))
    elif 'scale_shortside' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, load_size, crop_size, method)))

    # Available operation for paired data, zoom, crop, patch
    if 'zoom' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, load_size, crop_size, method)))
        else:
            transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, load_size, crop_size, method, factor=params["scale_factor"])))

    if 'crop' in opt.preprocess:
        if params is None or 'crop_pos' not in params:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

    if 'patch' in opt.preprocess:
        assert 'patch_index' in params, 'patch_index is required for patch operation to aviod repetitive patches'
        if 'patch_startxy' not in params:
            transform_list.append(transforms.Lambda(lambda img: __patch(img, params['patch_index'], crop_size)))
        else:
            transform_list.append(transforms.Lambda(lambda img: __patch(img, params['patch_index'],  crop_size, startxy=params['patch_startxy'])))

    if 'trim' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __trim(img, crop_size)))

    # if opt.preprocess == 'none':
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
    

    if not opt.no_flip:
        if params is None or 'flip' not in params:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif 'flip' in params:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if normalize:
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if verbose:
        print("transform_list")
        print(transform_list)
    return transforms.Compose(transform_list)

def get_none_preprocess_transform(grayscale=False, method=Image.BICUBIC, convert=True):
    """
    Reproduce 
    """
    transform_list = []
    # if opt.preprocess == 'none':
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    return img.resize((w, h), method)

def __make_power_2_mask(img, base, ow, oh, method=Image.BICUBIC):
    '''
    gridx and gridy are sizes of vision patches. Not directly available through img.size, so pass as parameters here.
    '''
    mask_w, mask_h = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    new_w = int(round(w/ow*mask_w))
    new_h = int(round(h/oh*mask_h))
    return img.resize((new_w, new_h), method)


def __random_zoom(img, target_width, crop_width, method=Image.BICUBIC, factor=None):
    if factor is None:
        zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    else:
        zoom_level = (factor[0], factor[1])
    iw, ih = img.size
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    print("In random_zoom, zoom_level({}, {}), iw {}, ih {}, zoomw {} zoomh {}".format(zoom_level[0], zoom_level[1], iw, ih, zoomw, zoomh))
    img = img.resize((int(round(zoomw)), int(round(zoomh))), method)
    return img



def __scale_shortside(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    shortside = min(ow, oh)
    if shortside >= target_width:
        return img
    else:
        scale = target_width / shortside
        return img.resize((round(ow * scale), round(oh * scale)), method)


def __trim(img, trim_width):
    ow, oh = img.size
    if ow > trim_width:
        xstart = np.random.randint(ow - trim_width)
        xend = xstart + trim_width
    else:
        xstart = 0
        xend = ow
    if oh > trim_width:
        ystart = np.random.randint(oh - trim_width)
        yend = ystart + trim_width
    else:
        ystart = 0
        yend = oh
    return img.crop((xstart, ystart, xend, yend))


def __scale_width(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width and oh >= crop_width:
        return img
    w = target_width
    h = int(max(target_width * oh / ow, crop_width))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __crop_mask(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    print("in __crop_mask, ow {} oh {} x1 {} y1 {} tw {} th {}".format(ow, oh, x1, y1, tw, th))
    if (ow > tw or oh > th):
        # For mask, assign all the values beyond the cropping area to zero
        img_arr = np.asarray(img)
        print("img_arr", img_arr.shape, np.min(img_arr), np.max(img_arr)) #should be 0 and 1
        img_arr[:y1] = 0
        img_arr[y1+th:] = 0
        img_arr[:, :x1] = 0
        img_arr[:, x1+tw:] = 0
        img_crop_mask = Image.fromarray(img_arr)
        return img_crop_mask
    else:
        return img


def __patch(img, index, size, startxy=None):
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    if startxy is None:
        startx = np.random.randint(int(roomx) + 1)
        starty = np.random.randint(int(roomy) + 1)
    else:
        startx, starty = startxy

    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    return img.crop((gridx, gridy, gridx + size, gridy + size))


def __patch_mask(img, index, size, startxy=None):
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    if startxy is None:
        startx = np.random.randint(int(roomx) + 1)
        starty = np.random.randint(int(roomy) + 1)
    else:
        startx, starty = startxy

    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    
    img_arr = np.asarray(img)
    print("img_arr in patch_mask", img_arr.shape, np.min(img_arr), np.max(img_arr)) #should be 0 and 1

    img_arr[:gridy] = 0
    img_arr[gridy+size:] = 0
    img_arr[:, :gridx] = 0
    img_arr[:, gridx+size:] = 0
    img_crop_mask = Image.fromarray(img_arr)
    return [img_crop_mask, gridx, gridy]


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def compute_startxy_per_patch(img, size):
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    startx = np.random.randint(int(roomx) + 1)
    starty = np.random.randint(int(roomy) + 1)
    return startx, starty


def get_transformation_for_mask(img, opt,params=None, method=Image.BICUBIC, magnification=1, verbose=False, load_size=None, crop_size=None):
    """
    Given the transform options, modify the masked area but does not crop the whole image.
    Return PIL image as new mask
    """
    transform_list = []
    if load_size is None:
        # Add magnification factor. magnification = 1 for sketch and image,  magnification = 4 for tactile image
        load_size = opt.load_size * magnification
    if crop_size is None:
        crop_size = opt.crop_size * magnification

    if 'fixsize' in opt.preprocess or 'resize' in opt.preprocess or 'scale_width' in opt.preprocess or 'scale_shortside' in opt.preprocess or 'trim' in opt.preprocess:
        raise NotImplementedError("opt.preprocess %s is not implemented for get_transform_mask"%opt.preprocess)

    # Available operation for paired data, zoom, crop, patch
    if 'zoom' in opt.preprocess:
        if params is None:
            raise ValueError('params[scale_factor] is required for zoom operation in get_transform_mask')
        else:
            transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, load_size, crop_size, method, factor=params["scale_factor"])))

    if 'crop' in opt.preprocess:
        if params is None or 'crop_pos' not in params:
            raise ValueError('params[crop_pos] is required for crop operation in get_transform_mask')
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop_mask(img, params['crop_pos'], crop_size)))


    if not opt.no_flip:
        raise ValueError("No_flip is required to generate the correct mask")

    transform = transforms.Compose(transform_list)
    img = transform(img)
    img, gridx, gridy = __patch_mask(img, params['patch_index'],  crop_size, startxy=params['patch_startxy'])
    img = __make_power_2_mask(img, base=4, ow=gridx, oh=gridy, method=method)

    return img
