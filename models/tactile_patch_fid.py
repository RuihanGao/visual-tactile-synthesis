import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F

"""
Helper function to compute FID and LPIPS for tactile output.
Since the tactile output is 2-channel, 1 for gx and 1 for gy, we implement a few tweaks, e.g. tiling the single channel to 3 channels, to make the FID and LPIPS work.
Adapted from https://github.com/GaParmar/clean-fid/blob/9c9dded6758fc4b575c27c4958dbc87b9065ec6e/cleanfid/fid.py 
"""

def compute_normal(T_concat, scale_nz=0, verbose=False):
    gx = torch.unsqueeze(T_concat[:,0,:,:],1)
    gy = torch.unsqueeze(T_concat[:,1,:,:],1)
    normal = torch.cat([gx, gy, scale_nz*torch.ones_like(gx)], dim=1)
    normal = F.normalize(normal, dim=1)
    return normal


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
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

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def multichannel_conv2d(x, weight, bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H, W, output_dim)
    '''
    # Prep
    h, w, _ = x.shape
    out_d, in_d, filt_h, filt_w = weight.shape
    p_h, p_w = (filt_h - 1) // 2, (filt_w - 1) // 2
    im = np.pad(x, ((p_h,p_h), (p_w,p_w), (0,0)), 'constant')
    
    # Im2col
    im_h, im_w, _ = im.shape
    im = np.transpose(im, (2, 0, 1))
    im_idx = np.arange(im_w - filt_w + 1) + np.arange(im_h - filt_h + 1)[:, None] * im_w
    hf_idx = np.arange(filt_w) + np.arange(filt_h)[:, None] * im_w
    channel_idx = np.arange(in_d) * im_h * im_w
    im2col_idx = im_idx.reshape(1, 1, -1) + hf_idx.reshape(1, -1, 1) + channel_idx.reshape(-1, 1, 1)
    im2col = np.take(im, im2col_idx)
    im2col = im2col.reshape(-1, im2col.shape[2])
    
    # Convolve
    weight = weight.reshape(out_d, -1)
    feat = weight.dot(im2col).reshape(out_d, h, w)
    return np.transpose(feat, (1, 2, 0)) + bias[None, None, :]


def Im2col(im, filt_w=3, filt_h=3):
    '''
    input shape: (H, W, C)
    '''
    im_h, im_w, in_d = im.shape
    im = np.transpose(im, (2, 0, 1))
    im_idx = np.arange(im_w - filt_w + 1) + np.arange(im_h - filt_h + 1)[:, None] * im_w
    hf_idx = np.arange(filt_w) + np.arange(filt_h)[:, None] * im_w
    channel_idx = np.arange(in_d) * im_h * im_w
    im2col_idx = im_idx.reshape(1, 1, -1) + hf_idx.reshape(1, -1, 1) + channel_idx.reshape(-1, 1, 1)
    im2col = np.take(im, im2col_idx)
    im2col = im2col.reshape(-1, im2col.shape[2])
    return im2col


def calculate_frechet_distance(grp1_im2col, grp2_im2col):
    # transpose to (sample_size, num_feature)
    grp1_im2col = grp1_im2col.T
    grp2_im2col = grp2_im2col.T

    # compute mean and cov
    mu1 = np.mean(grp1_im2col, axis=0)
    sigma1 = np.cov(grp1_im2col, rowvar=False)
    mu2 = np.mean(grp2_im2col, axis=0)
    sigma2 = np.cov(grp2_im2col, rowvar=False)

    # compute fid 
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def TactilePatchFID(real_T_concat, fake_T_concat, verbose=False, reduction='none'):
    """
    Compute FID for tactile patches (sometimes denoted as T-FID in the code base)
    Note: to compuate T-FID for each pair of patches, set reduction='mean'

    Args:
        real_T_concat: real tactile patches, shape (N, C, H, W)
        fake_T_concat: fake tactile patches, shape (N, C, H, W)
        verbose: option to print additional logging information
        reduction: option to reduce the FID across all patches, 'none' or 'mean'
    
    Returns:
        fid: FID between real and fake tactile patches
    """
    # crop all patches in each group to 3x3 grid and compute the mean and std
    grp1_real_T_concat_arr = np.transpose(real_T_concat.detach().cpu().numpy(), (0, 2, 3, 1)) # (N, C, H, W) -> (N, H, W, C)
    grp2_real_T_concat_arr = np.transpose(fake_T_concat.detach().cpu().numpy(), (0, 2, 3, 1)) # (N, C, H, W) -> (N, H, W, C)

    if reduction == 'none':
        # stack all crops of all patches together
        grp1_im2col = np.hstack([Im2col(x) for x in grp1_real_T_concat_arr])
        grp2_im2col = np.hstack([Im2col(x) for x in grp2_real_T_concat_arr])
        return calculate_frechet_distance(grp1_im2col, grp2_im2col)
    elif reduction == 'mean':
        # for each paired patch, compute the fid and then average the fid across all patches
        fid_sum = 0
        assert len(grp1_real_T_concat_arr) == len(grp2_real_T_concat_arr), "The number of images in the two groups are not equal"
        for i in range(len(grp1_real_T_concat_arr)):
            grp1_im2col = Im2col(grp1_real_T_concat_arr[i])
            grp2_im2col = Im2col(grp2_real_T_concat_arr[i])
            fid_sum += calculate_frechet_distance(grp1_im2col, grp2_im2col)
        if verbose: print(f"grp1_im2col {grp1_im2col.shape}, grp2_im2col {grp2_im2col.shape}")
        fid = fid_sum / len(grp1_real_T_concat_arr)
        return fid
    else:
        raise NotImplementedError(f"reduction {reduction} is not implemented")


def compute_touch_lpips_loss(criterionLPIPS, fake_T_concat, real_T_concat, lambda_lpips=1, separate_xy=False, batch_size_G2=None, verbose=False, return_lpips_res=False):
    """
    Compute LPIPS for tactile patches. Tile gx and gy of 2-channel tactile patches individually to 3-channel images and then compute LPIPS.

    Args:
        criterionLPIPS: LPIPS loss function
        fake_T_concat (tensor): fake tactile patches, shape (N, C, H, W)
        real_T_concat (tensor): real tactile patches, shape (N, C, H, W)
        lambda_lpips (float): weight for LPIPS loss
        separate_xy (bool): option to compute LPIPS for gx and gy separately
        batch_size_G2 (int): batch size for the second generator
        verbose (bool): option to print additional logging information
        return_lpips_res (bool): option to return the LPIPS results
    
    Returns:
        loss_lpips_gx: LPIPS loss for gx
        loss_lpips_gy: LPIPS loss for gy
        loss_lpips: LPIPS loss for gx and gy
    """
    loss_lpips_gx_res = criterionLPIPS(torch.unsqueeze(fake_T_concat[:,0,:,:],1), torch.unsqueeze(real_T_concat[:,0,:,:],1))
    loss_lpips_gy_res = criterionLPIPS(torch.unsqueeze(fake_T_concat[:,1,:,:],1), torch.unsqueeze(real_T_concat[:,1,:,:],1))

    if verbose: print(f"In compute_touch_lpips_loss loss_lpips_gx {loss_lpips_gx_res.shape}, loss_lpips_gy {loss_lpips_gy_res.shape}") # [NxNT, 1, 1, 1]
    N, C, H, W = loss_lpips_gx_res.shape
    if batch_size_G2 is None: 
        batch_size_G2 = N
    loss_lpips_gx = loss_lpips_gx_res.view(-1, batch_size_G2, C, H, W)
    loss_lpips_gx = loss_lpips_gx.mean()
    loss_lpips_gy = loss_lpips_gy_res.view(-1, batch_size_G2, C, H, W)
    loss_lpips_gy = loss_lpips_gy.mean()

    if separate_xy:
        if return_lpips_res:
            return (loss_lpips_gx*lambda_lpips), (loss_lpips_gy*lambda_lpips), loss_lpips_gx_res, loss_lpips_gy_res
        else:
            return (loss_lpips_gx*lambda_lpips), (loss_lpips_gy*lambda_lpips)
    else:
        loss_lpips = lambda_lpips*(loss_lpips_gx+loss_lpips_gy)
        if return_lpips_res:
            return loss_lpips, loss_lpips_gx_res, loss_lpips_gy_res
        else:
            return loss_lpips
            