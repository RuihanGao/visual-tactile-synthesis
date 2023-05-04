#!/usr/bin/env python3
"""Calculates ***Single Image*** Frechet Inception Distance (SIFID) to evalulate Single-Image-GANs
Code was adapted from:
https://github.com/mseitzer/pytorch-fid.git
Which was adapted from the TensorFlow implementation of:
                                 
https://github.com/bioinf-jku/TTUR
The FID metric calculates the distance between two distributions of images.
The SIFID calculates the distance between the distribution of deep features of a single real image and a single fake image.
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.                                                               
"""

import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
#from scipy.misc import imread
from matplotlib.pyplot import imread
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from .inception import InceptionV3
import torchvision
import numpy
import scipy
import pickle


def get_activations(batch, model, batch_size=1, dims=64,
                    device=None, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : input tensor of shape (N, C, H, W), range (0, 1)
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    if device is not None:
        model.to(device)
    model.eval()

    if len(batch) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(batch):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(batch)

    n_batches = len(batch) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        if device is not None:
            batch = batch.to(device)
        pred = model(batch)[0]

        pred_arr = pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(batch_size*pred.shape[2]*pred.shape[3],-1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(images, model, batch_size=1,
                                    dims=64, device=None, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : image array of shape (N, H, W, C)
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the inception model.
    -- sigma : The covariance matrix of the activations of the inception model.
    """
    act = get_activations(images, model, batch_size, dims, device=device, verbose=verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma



def convert2tensor(images, normalize=True, vmin=-1, vmax=1):

    # Convert the format, convert the numpy array to tensors of shape (N x C x H x W)
    if isinstance(images, np.ndarray):
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        else: 
            assert(len(images.shape) == 4), f"arr1 must be of shape (N, H, W, C) or (H, W, C), current shape is {images.shape}"
        images = images[:,:,:,0:3]
        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        #images = images[0,:,:,:]
        images /= 255
        images = torch.from_numpy(images).type(torch.FloatTensor)

    else:
        assert(isinstance(images, torch.Tensor)) and len(images.shape) == 4, f"arr1 must be of type np.ndarray or torch.Tensor, current type is {type(images)}, shape is {images.shape}"
        assert images.shape[1] == 3, f"images tensor must be of shape (N, 3, H, W), current shape is {images.shape}"
        # convert to torch float tensor
        images = images.type(torch.FloatTensor)
        # normalize to [0,1]
        if normalize: images = (images - vmin) / (vmax - vmin)
    
    return images

def calculate_sifid_given_arrays(arr1, arr2, batch_size=1, dims=64, device=None, normalize=True):
    """
    Calculates the SIFID of two paths
    arr1, arr2: can be image arrays of shape (N, H, W, C) or (H, W, C) OR torch tensor of shape (N, C, H, W), C=3; range should be (0,1)
    Args:
    normalize: if True, normalize the input images to [0,1]
    """

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if device is not None:
        if device == -1:
            device = torch.device('cpu')
        model.to(device)

    fid_values = []

    arr1 = convert2tensor(arr1, normalize=normalize)
    arr2 = convert2tensor(arr2, normalize=normalize)
        

    for i in range(len(arr1)):
        m1, s1 = calculate_activation_statistics(torch.unsqueeze(arr1[i], 0), model, batch_size, dims, device=device)
        m2, s2 = calculate_activation_statistics(torch.unsqueeze(arr2[i],0), model, batch_size, dims, device=device)
        fid_values.append(calculate_frechet_distance(m1, s1, m2, s2))
        # for a single image, should return a list of length 1
    return fid_values

