"""This module contains simple helper functions """
from __future__ import print_function
from operator import ne
import torch
import numpy as np
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision
import matplotlib.pyplot as plt
from datetime import datetime
import cv2


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8, convert_gray_imgs=False, colormap='viridis'):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    # print('In tensor2im', input_image.shape)

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            if len(input_image.shape) == 2: input_image = torch.unsqueeze(torch.unsqueeze(input_image, 0),0) # for T data, unsqueeze in channel dimension
            if len(input_image.shape) == 3: input_image = torch.unsqueeze(input_image, 0) 
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        image_numpy = (image_numpy + 1)/ 2.0 # scale (-1, 1) to (0, 1)
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            if convert_gray_imgs:
                # use matplotlib cmap to convert, can define colormap
                cmap = plt.get_cmap(colormap)
                rgba_img = cmap(np.squeeze(image_numpy)) # take 2D arr as input (H, W)
                image_numpy = np.delete(rgba_img, 3, 2)
            else:
                # simplest conversion, tile 1 channel to 3 channels, output gray image
                image_numpy = np.transpose(np.tile(image_numpy, (3, 1, 1)), (1, 2, 0))
        else: 
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
        image_numpy = image_numpy*255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



def tensor2arr(input_image, imtype=np.float64, convert2RGB=False, verbose=False):
    """
    Modified from tensor2im. Convert tensor to numpy array to save gx and gy in raw floating point format. It's convenient for 3D reconstruction.
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            input_image = torch.squeeze(input_image)
            if len(input_image.shape) == 2: input_image = torch.unsqueeze(torch.unsqueeze(input_image, 0),0) # for T data, unsqueeze in channel dimension
            if len(input_image.shape) == 3: input_image = torch.unsqueeze(input_image, 0) 
            image_tensor = input_image.data
            if verbose: print("original tensor min {} max {}, clamp to -1, 1".format(torch.min(image_tensor), torch.max(image_tensor)))
            image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        else:
            raise NotImplementedError("unknown type of input_iamge", type(input_image))
    else:
        image_numpy = input_image
    if convert2RGB and image_numpy.shape[0] == 1:  # grayscale to RGB
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    if len(image_numpy.shape) == 3: h, w, _ = image_numpy.shape
    else:
        assert len(image_numpy.shape) == 2
        h, w = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    # print("save pil image", image_path, image_pil.width, image_pil.height)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def plot_grad_flow(named_parameters, net='net'):
    ave_grads = []
    layers = []
    # print('named_parameters')
    for n, p in named_parameters:
        # print('name:', n)
        # print('parameter', type(p))
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            # grad = p.grad.view(-1)
            # print('save grad', type(grad), grad.shape)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
        else:
            # print('skip')
            pass
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow for net %s"%net)
    plt.grid(True)
    plt.tight_layout()
    today = datetime.today().strftime('%Y-%m-%d')
    log_dir = 'logs/%s'%today
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    plt.savefig('%s/sample_grad_plot.png'%(log_dir))
    plt.close()


def variance_of_laplacian(image, ref=None):
    if ref is None:
        ref = np.ones_like(image)*127
    image = image-ref
    return cv2.Laplacian(image, cv2.CV_64F).var()