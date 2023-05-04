import torch
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F

import util.util as util
import time
import cv2
from collections import OrderedDict

from models.sifid import *
from models.tactile_patch_fid import *
from models.normal_losses import *
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import structural_similarity_index_measure as SSIM

### Followiing functions are helper functions to locate patches given the input image and coordinates ###


def find_coords_for_patch(coords, scale_multiplier=1, verbose=False):
    """
    Given the coordinates of the contact region in low-res visual image, return the corresponding offset and cutout size for tactile patches in tensor.

    Args:
        coords (array): T_coords shape (1, N, 8) - (new_ROI_x, new_ROI_y, new_ROI_h, new_ROI_w, patch_crop_size, resize_ratio, crop_pos_x, crop_pos_y)
        scale_multiplier (int): scale multiplier for the input image
        verbose (bool): option to print log info

    Returns:
        offset_x (tensor): (N, 1, 1) offset in x direction
        offset_y (tensor): (N, 1, 1) offset in y direction
        cutout_size (tensor): (N, 1, 1) size of the cutout patch
    """
    coords = np.squeeze(coords)  # squeeze (1, 16, 8) to (16, 8)
    # elementwise round and convert to int for indexing
    offset_x = np.round((coords[..., 0] + coords[..., -2] / coords[..., -3]) * scale_multiplier)
    offset_y = np.round((coords[..., 1] + coords[..., -1] / coords[..., -3]) * scale_multiplier)
    cutout_size = np.round(coords[..., -4] / coords[..., -3] * scale_multiplier)

    if verbose:
        print(
            "check offset_x, offset_y, cutout_size in find_coords_for_patch",
            offset_x.shape,
            offset_y.shape,
            cutout_size.shape,
        )  # (16,) (16,) (16,)
        print(
            f"check cutout_size values, should be a constant \n {cutout_size}"
        )  # [64. 64. 64. 64. 64. 64. 64. 64. 64. 64. 64. 64. 64. 64. 64. 64.]

    # cast to int tensor
    offset_x = torch.FloatTensor(offset_x).to(torch.int32).reshape(-1, 1, 1)
    offset_y = torch.FloatTensor(offset_y).to(torch.int32).reshape(-1, 1, 1)
    cutout_size = torch.FloatTensor(cutout_size).to(torch.int32).reshape(-1, 1, 1)

    if verbose:
        print(
            "check tensor size",
            "offset_x",
            offset_x.shape,
            "offset_y",
            offset_y.shape,
            "cutout_size",
            cutout_size.shape,
        )  # [16, 1, 1], [16, 1, 1], [16, 1, 1]
    return offset_x, offset_y, cutout_size


def get_patch_in_input(
    input_image,
    coords=None,
    sample_size=None,
    method="bicubic",
    scale_multiplier=1,
    patch_size=32,
    verbose=False,
    real_gx_concat=None,
    return_offset=False,
    offset_x=None,
    offset_y=None,
    cutout_size=None,
    center_h=None,
    center_w=None,
    augmentation_params=None,
    device="cuda",
    M=None,
):
    """
    Given the input image and coordinates, return the corresponding patch in the input image.

    Args:
        input_image (tensor): input image tensor, shape (1, 3, H, W)
        coords (array): If not None, "coords" stores coordinate of patches in low-res visual image (scale_multiplier=1). T_coords shape (1, N, 8) - (new_ROI_x, new_ROI_y, new_ROI_h, new_ROI_w, patch_crop_size, resize_ratio, crop_pos_x, crop_pos_y). If coords is None, we randomly sample the patch coordinates.
        sample_size (int): sample size for random cutout
        method (str): interpolation method for resize
        scale_multiplier (int): scale multiplier for the input image, used when tactile output and visual output are different sizes. Default is 1.
        patch_size (int): size of the patch to be cutout
        verbose (bool): option to print log info
        real_gx_concat (tensor): real tactile patches, used for plotting an example.

        return_offset (bool): option to return offset_x, offset_y, cutout_size, used for more fake T samples
        offset_x (tensor): (N, 1, 1) offset in x direction
        offset_y (tensor): (N, 1, 1) offset in y direction
        cutout_size (tensor): (N, 1, 1) size of the cutout patch
        center_h (int): center height of the patch
        center_w (int): center width of the patch
        augmentation_params (dict): dictionary of augmentation parameters
        device (str): device to run the code
        M (tensor): object mask input

    Returns:
        samples (tensor): patch tensor
        offset_x (tensor): (N, 1, 1) offset in x direction
        offset_y (tensor): (N, 1, 1) offset in y direction
        cutout_size (tensor): (N, 1, 1) size of the cutout patch

    Usage:
    1. known coords (real train data), fetch the offset and cutout grid directly
    2. For fake T samples, random sample patch in T, return the offsets
    3. To find corresponding S and I patches for fake T samples, use offset_x, offset_y, cutout_size inputs
    """
    # Make sure that the coords are on the same device as the input image
    device = input_image.get_device()
    device = "cpu" if device == -1 else device
    # assume patch_size considers scale_multiplier
    patch_size = patch_size * scale_multiplier
    if center_h is not None:
        center_h = center_h * scale_multiplier
    if center_w is not None:
        center_w = center_w * scale_multiplier
    image_size = input_image.shape[-2:]  # H, W

    if coords is None:
        # random cutout
        assert sample_size is not None, "need sample_size to create random offset grid"
        if offset_x is None and offset_y is None:
            if verbose:
                print("use random cutout")
            if center_h is None or center_w is None:
                if verbose:
                    print(f"center_h is None? {center_h is None}, center_w is None? {center_w is None}")
                bb_xmin = 0
                bb_xmax = image_size[0] + (1 - patch_size % 2)
                bb_ymin = 0
                bb_ymax = image_size[1] + (1 - patch_size % 2)
            else:
                if verbose:
                    print(f"setting constraint for more fake T samples, given center_h {center_h}, center_w {center_w}")
                assert (
                    augmentation_params is not None
                ), "augmentation_params is required for setting fake T sample constraints"
                assert (
                    augmentation_params["scale_factor_h"] == 1 and augmentation_params["scale_factor_w"] == 1
                ), "bb_xy constrains assume scaling factor is 1, i.e. no zoom augmentation"
                if verbose:
                    print("set constrains for more fake_T samples")
                # define the cutout region based on data augmentation parameters
                h = augmentation_params["H"].numpy()[0] * scale_multiplier
                w = augmentation_params["W"].numpy()[0] * scale_multiplier
                center_crop_x = augmentation_params["crop_pos_x"].numpy()[0]
                center_crop_y = augmentation_params["crop_pos_y"].numpy()[0]
                resize_ratio_w = augmentation_params["resize_ratio_w"].numpy()[0]
                resize_ratio_h = augmentation_params["resize_ratio_h"].numpy()[0]
                if verbose:
                    print(
                        "h",
                        h,
                        "w",
                        w,
                        "center_h",
                        center_h,
                        "center_w",
                        center_w,
                        "center_crop_x",
                        center_crop_x,
                        "center_crop_y",
                        center_crop_y,
                        "resize_ratio_w",
                        resize_ratio_w,
                        "resize_ratio_h",
                        resize_ratio_h,
                    )
                bb_xmin = int(max(0, round(((w - center_w) / 2 - center_crop_x) / resize_ratio_w)))
                bb_xmax = int(
                    min(
                        image_size[0] + (1 - patch_size % 2),
                        round(((w + center_w) / 2 - center_crop_x) / resize_ratio_w - patch_size),
                    )
                )
                bb_ymin = int(max(0, round(((h - center_h) / 2 - center_crop_y) / resize_ratio_h)))
                bb_ymax = int(
                    min(
                        image_size[1] + (1 - patch_size % 2),
                        round(((h + center_h) / 2 - center_crop_y) / resize_ratio_h - patch_size),
                    )
                )
            if verbose:
                print("bb_xmin", bb_xmin, "bb_xmax", bb_xmax, "bb_ymin", bb_ymin, "bb_ymax", bb_ymax)

            cutout_size = torch.tile(torch.tensor(patch_size), (sample_size, 1, 1)).to(torch.int32).to(device)

            # Random sample the coordinates until a desired number of samples are found within the object mask
            # M is the object mask. create a mask to sample indexes where M = 1
            if verbose:
                print(
                    f"check M for masked sampling, M.shape {M.shape}, M.sum() {M.sum()} max {torch.max(M)} min {torch.min(M)}"
                )
            # erode the mask so that the sampled fake T patches stay within the object mask
            kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(np.ones((17, 17), dtype=np.float32), 0), 0)).to(
                device
            )  # size: (1, 1, 3, 3)
            eroded_M = torch.clamp(torch.nn.functional.conv2d(M, kernel_tensor, padding=(1, 1)), 0, 1)
            masked_indexes = torch.nonzero(eroded_M, as_tuple=False)
            sampled_indexes = random.sample(range(masked_indexes.shape[0]), sample_size)
            sampled_indexes = masked_indexes[sampled_indexes][:, -2:]
            if verbose:
                print(f"sampled_indexes.shape {sampled_indexes.shape} example {sampled_indexes[0]}")  # Nx2
            offset_x = sampled_indexes[:, 1].view(sample_size, 1, 1).to(torch.int32)
            offset_y = sampled_indexes[:, 0].view(sample_size, 1, 1).to(torch.int32)
            if verbose:
                print(
                    f"offset_x.shape {offset_x.shape} dtype {offset_x.dtype} device {offset_x.get_device()}, offset_y.shape {offset_y.shape}"
                )
        else:
            if verbose:
                print("use input offset_x, offset_y, cutout_size")

    else:
        # use predefined coordinates
        if verbose:
            print("use predefined coords", coords.shape)
        assert coords.shape[0] == 1, "coords should have batch size of 1"
        sample_size = coords.shape[1]
        offset_x, offset_y, cutout_size = find_coords_for_patch(coords, scale_multiplier, verbose=verbose)
        offset_x = offset_x.to(device)
        offset_y = offset_y.to(device)
        cutout_size = cutout_size.to(device)

    if verbose:
        if coords is not None:
            print(f"coords len {len(coords)}")
        else:
            print("coords is None")
        print(
            "After obtaining the coords, offset_x {}, offset_y {}, cutout_size {}".format(
                offset_x.shape, offset_y.shape, cutout_size.shape
            )
        )

    # repeat the input_image for batch operation for each cutou
    input_image_repeat = input_image.repeat(sample_size, 1, 1, 1)

    # create the grid
    (
        grid_batch,
        grid_y,
        grid_x,
    ) = torch.meshgrid(
        torch.arange(start=0, end=sample_size, dtype=torch.int32),
        torch.arange(start=0, end=torch.max(cutout_size), dtype=torch.int32),
        torch.arange(start=0, end=torch.max(cutout_size), dtype=torch.int32),
        indexing="ij",
    )
    grid_batch = grid_batch.to(device)
    grid_y = grid_y.to(device)
    grid_x = grid_x.to(device)
    if verbose:
        print(
            "check shape grid_batch {}, grid_x {}, offset_x {}, cutout_size {}".format(
                grid_batch.shape, grid_x.shape, offset_x.shape, cutout_size.shape
            )
        )  # check shape grid_batch torch.Size([16, 64, 64]), grid_x torch.Size([16, 64, 64]), offset_x torch.Size([16, 1, 1]), cutout_size torch.Size([16, 1, 1])
    if verbose:
        print(
            f"check shape grid_batch {grid_batch.shape} {grid_batch.get_device()},  grid_x {grid_x.shape} {grid_x.get_device()},  grid_y {grid_y.shape} {grid_y.get_device()}, offset_x {offset_x.shape} {offset_x.get_device()}, offset_y {offset_y.shape} {offset_y.get_device()}"
        )

    # confine the cutout region within the mask
    cutout_grid = torch.stack([grid_batch, grid_y + offset_y, grid_x + offset_x], axis=-1)
    mask_shape = torch.Tensor([sample_size, image_size[0], image_size[1]]).to(device)
    cutout_grid = torch.maximum(cutout_grid, torch.zeros_like(cutout_grid))  # elementwise maximum
    cutout_grid = torch.minimum(cutout_grid, torch.reshape(mask_shape - 1, [1, 1, 1, 3]))
    if verbose:
        print("check cutout_grid", cutout_grid.shape)  # [16, 64, 64, 3]
    if verbose:
        print("check cutout_grid [0,0,:,2]", cutout_grid[0, 0, :, 2].shape)
        # print(cutout_grid[0,0,:,2])
        x_min, _ = torch.min(cutout_grid[:, 0, :, 2], dim=-1)
        x_max, _ = torch.max(cutout_grid[:, 0, :, 2], dim=-1)
        patch_size_x = x_max - x_min

        print("check cutout_grid [0,:,0,1]", cutout_grid[0, :, 0, 1].shape)
        # print(cutout_grid[0,:,0,1])
        y_min, _ = torch.min(cutout_grid[:, :, 0, 1], dim=-1)
        y_max, _ = torch.max(cutout_grid[:, :, 0, 1], dim=-1)
        patch_size_y = y_max - y_min

        print("offset_x", np.squeeze(offset_x.numpy()))
        print("cutout x", np.squeeze(x_min.numpy()))

        print("offset_y", np.squeeze(offset_y.numpy()))
        print("cutout y", np.squeeze(y_min.numpy()))

        print("patch_size_x", patch_size_x)
        print("patch_size_y", patch_size_y)
    if verbose:
        print("mask_shape", mask_shape)  # [  16., 3072., 3072.]

    N, H, W, M = cutout_grid.shape
    cutout_grid_reshape = torch.reshape(cutout_grid, (-1, M)).to(torch.long)
    if verbose:
        print(
            "cutout_grid_reshape", cutout_grid_reshape.shape, cutout_grid_reshape.dtype
        )  # torch.Size([65536, 3]) torch.int64
    if verbose:
        print("sample cutout_grid_reshape")
        print(cutout_grid_reshape[:5])
        print(
            "N max {} min {}, H max {} min {}, W max {} min {}".format(
                torch.max(cutout_grid_reshape[:, 0]),
                torch.min(cutout_grid_reshape[:, 0]),
                torch.max(cutout_grid_reshape[:, 1]),
                torch.min(cutout_grid_reshape[:, 1]),
                torch.max(cutout_grid_reshape[:, 2]),
                torch.min(cutout_grid_reshape[:, 2]),
            )
        )

    # cut out the samples given the cutout_grid coordinates
    samples = input_image_repeat[cutout_grid_reshape[:, 0], :, cutout_grid_reshape[:, 1], cutout_grid_reshape[:, 2]]
    samples = torch.reshape(samples, (N, H, W, -1))  # reshape from [65536, 2] to [16, 64, 64, 2]
    samples = samples.permute(0, 3, 1, 2)  # permute (N, H, W, C) to (N, C, H, W)

    if H < patch_size or W < patch_size:
        print("scale small samples to patch_size")
        samples = F.interpolate(
            samples, size=(patch_size, patch_size), mode=method, align_corners=False, antialias=True
        )
    assert len(samples.shape) == 4, "samples should have shape (N, C, H, W)"

    if verbose:
        print("get_patch_in_input, return samples shape ", samples.shape)  # more_fake_T [16, 2, 32, 32]
        # plot the samples by 1. cropping the PIL image and 2. using tensor cutout_grid
        offset_x = np.squeeze(offset_x.numpy())
        offset_y = np.squeeze(offset_y.numpy())
        print("for plotting")

        if real_gx_concat is not None:
            print("plot visual and tactile patches")

            input_image_arr = util.tensor2im(torch.squeeze(input_image))
            input_image_im = Image.fromarray(input_image_arr)
            print(
                "check input_image",
                type(input_image),
                input_image.shape,
                "input_image_arr",
                type(input_image_arr),
                input_image_arr.shape,
                "input_image_im",
                input_image_im.size,
            )

            for i in range(len(samples)):
                fig, axes = plt.subplots(nrows=2, ncols=2)
                axes[0, 0].imshow(input_image_im)
                rect_1 = patches.Rectangle(
                    (offset_x[i], offset_y[i]), patch_size, patch_size, linewidth=1, facecolor="none", edgecolor="r"
                )
                axes[0, 0].add_patch(rect_1)
                axes[0, 0].set_title("im")

                image_patch = input_image_im.crop(
                    (offset_x[i], offset_y[i], offset_x[i] + patch_size, offset_y[i] + patch_size)
                )
                axes[0, 1].imshow(image_patch)
                axes[0, 1].set_title("square image_patch")

                gx_im = util.tensor2im(real_gx_concat[i])
                axes[1, 0].imshow(gx_im, cmap="gray")
                axes[1, 0].set_title("gx_im")

                sample_im = util.tensor2im(samples[i])
                axes[1, 1].imshow(sample_im)
                axes[1, 1].set_title("sample_im")

                plt.tight_layout()
                plt.savefig("logs/2022-08-19/Find_patch_comparison_%d.png" % i, dpi=900)
                plt.close()
                print("save data for patch %d" % i)
        # convert offset back to tensor
        offset_x = torch.Tensor(offset_x).to(torch.int32)
        offset_y = torch.Tensor(offset_y).to(torch.int32)

    if return_offset:
        if verbose:
            print(
                f"return offset offset_x type {type(offset_x)}, offset_y type {type(offset_y)}, cutout_size type {type(cutout_size)}, scale_multiplier type {type(scale_multiplier)}"
            )
        return samples, offset_x / scale_multiplier, offset_y / scale_multiplier, cutout_size / scale_multiplier
    else:
        return samples


def compute_normal(T_concat, scale_nz=0, verbose=False):
    """
    Compute the normal map from the surface gradient map (gx, gy)
    Args:
        T_concat (tensor): (gx, gy) patches. Shape: [NxNT, 2, H, W]
        scale_nz (int, optional): scale factor to multiply gz. Correct conversion should have scale_nz=1, but in implementation, we may scale down to see more significant difference in gx and gy. Defaults to 0.
        verbose (bool, optional): optiion ot print more logs. Defaults to False.
    Returns:
        normal: normal map tensor. Shape: [NxNT, 3, H, W]
    """
    gx = torch.unsqueeze(T_concat[:, 0, :, :], 1)
    gy = torch.unsqueeze(T_concat[:, 1, :, :], 1)
    normal = torch.cat([gx, gy, scale_nz * torch.ones_like(gx)], dim=1)
    if verbose:
        print(
            "gx shape", gx.shape, "gy shape", gy.shape, "normal shape", normal.shape
        )  # [127, 1, 32, 32], [127, 1, 32, 32], [127, 3, 32, 32]
    normal = F.normalize(normal, dim=1)
    if verbose:
        print("after normalization", normal.shape)
    return normal


def compute_evaluation_metric(
    model_names,
    real_I,
    fake_I,
    real_T_concat=None,
    fake_T_concat=None,
    eval_metrics=[],
    eval_LPIPS=None,
    opt=None,
    device=None,
    verbose=False,
    timing=False,
    prefix="",
):
    """
    Compute evaluation metrics for the current batch of data

    Args:
        model_names (list): list of names of the modules
        real_I (tensor): real visual image
        fake_I (tensor): fake visual image
        fake_T_concat (tensor, optional): concatenated fake tactile images. Defaults to None.
        real_T_concat (tensor, optional): concatenated real tactile images. Defaults to None.
        eval_metrics (list, optional): list of evaluation metrics. Defaults to [].
        eval_LPIPS (tensor, optional): LPIPS model. Defaults to None.

        verbose (bool, optional): option to print logs for debugging. Defaults to False.
        timing (bool, optional): option to print computation time for each step. Defaults to False.
        prefix (str, optional): prefix for the metric name. Defaults to "". can be "train_" or "test_".

    Returns:
    metric_dict: Set the loss items as dictionary keys.

    Note:
    1. The input to lpips should be normalized to [-1,1]. Ref: https://github.com/richzhang/PerceptualSimilarity
        The input to PSNR and SSIM should be normalized to [0,1].
    """
    metric_dict = {}

    if verbose:
        print(f"model names: {model_names}, eval metrics: {eval_metrics}, prefix: {prefix}")
    if timing:
        start_time = time.time()

    # compare real_I and fake_I, range [-1,1]
    if "I_LPIPS" in eval_metrics:
        lpips_I = eval_LPIPS(real_I, fake_I).mean().cpu().detach().numpy()  # shape (1, 1, 1, 1)
        metric_dict["metric_%sI_LPIPS" % (prefix)] = np.squeeze(lpips_I)[
            ()
        ]  # squeeze to 0d array then take the first element

    # transform I from [-1,1] to [0,1] using min-max of real images
    I_min = real_I.min()
    I_max = real_I.max()
    real_I = (real_I - I_min) / (I_max - I_min)
    fake_I = (fake_I - I_min) / (I_max - I_min)
    fake_I = torch.clamp(fake_I, 0, 1)

    if "I_SIFID" in eval_metrics:
        # real_I & fake_I shape [1, 3, 1536, 1536] range [-1, 1]
        sifid_I = calculate_sifid_given_arrays(real_I, fake_I, normalize=False, device=fake_I.get_device())
        # return the first item is len == 1, otherwise return the mean of the list
        metric_dict["metric_%sI_SIFID" % (prefix)] = sifid_I[0] if len(sifid_I) == 1 else np.mean(sifid_I)

    if "I_PSNR" in eval_metrics:
        metric_dict["metric_%sI_PSNR" % (prefix)] = PSNR(real_I, fake_I, data_range=1).cpu().detach().numpy()

    if "I_SSIM" in eval_metrics:
        metric_dict["metric_%sI_SSIM" % (prefix)] = SSIM(real_I, fake_I, data_range=1).cpu().detach().numpy()

    if timing:
        print("eval metric for visual output takes time ", time.time() - start_time)
        start_time = time.time()

    # compare real_T_concat and fake_T_concat

    # Save torch tensor to file
    if hasattr(opt, "save_T_concat_tensor") and opt.save_T_concat_tensor:
        T_concat = {"real_T_concat": real_T_concat, "fake_T_concat": fake_T_concat}
        tensor_path = os.path.join(
            opt.checkpoints_dir, opt.name, f"T_concat_{opt.epoch}.pth"
        )  # set postfix to be the number of epoch
        print("Save T_concat to {}".format(tensor_path))
        torch.save(T_concat, tensor_path)

    N, C, H, W = real_T_concat.shape
    fake_T_concat = torch.clamp(fake_T_concat, 0, 1)

    if "T_LPIPS" in eval_metrics:
        resize_size = 224
        real_T_concat_resize = F.interpolate(real_T_concat, (resize_size, resize_size))
        fake_T_concat_resize = F.interpolate(fake_T_concat, (resize_size, resize_size))
        lpips = compute_touch_lpips_loss(eval_LPIPS, real_T_concat_resize, fake_T_concat_resize)
        T_lpips = lpips.detach().cpu().numpy()
        metric_dict["metric_%sT_LPIPS" % (prefix)] = T_lpips

    if "T_AE" in eval_metrics:
        fake_normals = compute_normal(fake_T_concat, scale_nz=1)
        real_normals = compute_normal(real_T_concat, scale_nz=1)
        T_AE = compute_surface_normal_angle_error(real_normals, fake_normals, mode="evaluate")
        T_AE = T_AE.mean().cpu().detach().numpy()
        metric_dict["metric_%sT_AE" % (prefix)] = T_AE

    if "T_FID" in eval_metrics:
        T_fid = TactilePatchFID(real_T_concat, fake_T_concat)
        metric_dict["metric_%sT_FID" % (prefix)] = T_fid

    if "T_SIFID" in eval_metrics:
        batch_size = 1
        dims = 64
        resize_size = 299
        real_T_concat_resize = F.interpolate(real_T_concat, (resize_size, resize_size))
        fake_T_concat_resize = F.interpolate(fake_T_concat, (resize_size, resize_size))
        T_gx_grp1 = torch.tile(torch.unsqueeze(real_T_concat_resize[:, 0, :, :], 1), (1, 3, 1, 1))
        T_gy_grp1 = torch.tile(torch.unsqueeze(real_T_concat_resize[:, 1, :, :], 1), (1, 3, 1, 1))
        T_gx_grp2 = torch.tile(torch.unsqueeze(fake_T_concat_resize[:, 0, :, :], 1), (1, 3, 1, 1))
        T_gy_grp2 = torch.tile(torch.unsqueeze(fake_T_concat_resize[:, 1, :, :], 1), (1, 3, 1, 1))
        gx_sifid = calculate_sifid_given_arrays(T_gx_grp1, T_gx_grp2, batch_size=batch_size, dims=dims, device=device)
        gy_sifid = calculate_sifid_given_arrays(T_gy_grp1, T_gy_grp2, batch_size=batch_size, dims=dims, device=device)
        T_sifid = np.mean((np.array(gx_sifid) + np.array(gy_sifid)) / 2)
        metric_dict["metric_%sT_SIFID" % (prefix)] = T_sifid

    if "T_MSE" in eval_metrics:
        T_MSE = torch.mean((real_T_concat - fake_T_concat) ** 2).cpu().detach().numpy()
        metric_dict["metric_%sT_MSE" % (prefix)] = T_MSE

    if timing:
        print("eval metric for G2 takes time ", time.time() - start_time)
        start_time = time.time()

    return metric_dict


def compute_additional_visuals(
    real_S,
    real_I,
    fake_I,
    I_mask_concat,
    real_T_concat,
    fake_T,
    T_coords,
    scale_nz=1,
    T_resolution_multiplier=1,
    phase="test",
    test_edit_S=False,
    use_more_fakeT=False,
    return_patch=False,
    draw_bounding_box=True,
    full_T_coords=None,
    num_patch_for_logging=10,
    collage_patch_as_array=True,
    colormap="gray",
    save_S_patch=False,
    fake_sample_offset_x=None,
    fake_sample_offset_y=None,
    fake_sample_cutout_size=None,
    fake_S_samples=None,
):
    """
    Compute additional visuals (patches of S, I, I', T(gx), T'(gx), T(gy), T'(gy) for both train & eval) that are not self attributes of the model. Return the images as an OrderedDict.
    Args:
        real_S (tensor): real sketch
        real_I (tensor): real visual image
        fake_I (tensor): fake visual image
        I_mask_concat (tensor): mask for the visual image
        real_T_concat (tensor): real tactile image
        fake_T (tensor): fake tactile image
        T_coords (tensor): coordinates of the tactile image
        scale_nz (int, optional): scale the normal vector by this factor. Defaults to 1.
        T_resolution_multiplier (int, optional): resolution multiplier for the tactile image. Defaults to 1.
        phase (str, optional): phase of the model. Defaults to "test".
        test_edit_S (bool, optional): indicator of whether the dataset is edited dataset. If True, we don't have ground truth for visual and tactile data for visualization. Defaults to False.
        use_more_fakeT (bool, optional): indicator of whether to use more fake tactile images for visualization. If True, we visualize sampled random patches. Defaults to False.
        return_patch (bool, optional): indicator of whether to return the patches. Defaults to False.
        draw_bounding_box (bool, optional): option to draw bounding box of the patchces on visual and tactile data. Defaults to True.
        full_T_coords (tensor, optional): full coordinates of the tactile image. Defaults to None.
        num_patch_for_logging (int, optional): number of patches to visualize. Defaults to 10.
        collage_patch_as_array (bool, optional): option to concatenate the (S, I, T) patches as one image in a numpy array. Defaults to True.
        colormap (str, optional): colormap for the tactile patches (single channel). Defaults to "gray".
        save_S_patch (bool, optional): option to save the S patches. Defaults to False.
        fake_sample_offset_x (int, optional): x offset of the sampled random patch if use_more_fakeT is True. Defaults to None.
        fake_sample_offset_y (int, optional): y offset of the sampled random patch if use_more_fakeT is True. Defaults to None.
        fake_sample_cutout_size (int, optional): size of the sampled random patch if use_more_fakeT is True. Defaults to None.
        fake_S_samples (tensor, optional): sampled S patches if use_more_fakeT is True. Defaults to None.
    Returns:
        OrderedDict: a dictionary of additional visuals
    """
    # Create a dictionary to store the additional visuals (key-value pairs)
    additional_visual_names = OrderedDict()
    fake_gx = fake_T[:, 0, :, :]

    with torch.no_grad():
        if not test_edit_S:  # only visualize patches when the ground truth is available
            # make a copy to draw the bounding boxes on the generated visual and tactile images
            I = util.tensor2im(fake_I).astype(np.uint8).copy()
            gx = util.tensor2im(fake_gx).astype(np.uint8).copy()

            # gather patched data
            real_gx_concat = torch.unsqueeze(real_T_concat[:, 0, :, :], 1)
            real_gy_concat = torch.unsqueeze(real_T_concat[:, 1, :, :], 1)
            real_normal_concat = compute_normal(real_T_concat[:, :2, :, :], scale_nz=scale_nz)

            if real_I is not None:
                if T_coords is not None:
                    # if the model is running on the whole image, retrieve patches from the whole image
                    S_concat = get_patch_in_input(real_S, T_coords)
                    fake_I_concat = get_patch_in_input(fake_I, T_coords)
                    real_I_concat, offset_x, offset_y, cutout_size = get_patch_in_input(
                        real_I, T_coords, return_offset=True
                    )
                    fake_T_concat = get_patch_in_input(fake_T, T_coords, scale_multiplier=T_resolution_multiplier)
                else:
                    # if the model is running on patches, directly use the patches
                    S_concat = real_S
                    fake_I_concat = fake_I
                    real_I_concat = real_I
                    fake_T_concat = fake_T

                fake_gx_concat = torch.unsqueeze(fake_T_concat[:, 0, :, :], 1)
                fake_gy_concat = torch.unsqueeze(fake_T_concat[:, 1, :, :], 1)
                fake_normal_concat = compute_normal(fake_T_concat[:, :2, :, :], scale_nz=scale_nz)
                H, W = real_gx_concat.shape[-2:]

                if (not return_patch) and draw_bounding_box:
                    # draw bounding box on the visual and tactile images to indicate the location
                    offset_x = offset_x.detach().cpu().numpy().astype(np.int32)
                    offset_y = offset_y.detach().cpu().numpy().astype(np.int32)
                    cutout_size = cutout_size.detach().cpu().numpy().astype(np.int32)
                    c = (255, 0, 0) if phase == "train" else (0, 255, 0)
                    I_phase = I.copy()
                    gx_phase = gx.copy()
                    for i in range(len(offset_x)):
                        I_phase = cv2.rectangle(
                            I_phase,
                            (int(offset_x[i]), int(offset_y[i])),
                            (int(offset_x[i] + cutout_size[i]), int(offset_y[i] + cutout_size[i])),
                            c,
                            2,
                        )
                        gx_phase = cv2.rectangle(
                            gx_phase,
                            (
                                int(offset_x[i] * T_resolution_multiplier),
                                int(offset_y[i] * T_resolution_multiplier),
                            ),
                            (
                                int((offset_x[i] + cutout_size[i]) * T_resolution_multiplier),
                                int((offset_y[i] + cutout_size[i]) * T_resolution_multiplier),
                            ),
                            c,
                            2,
                        )  # current epoch

                    if full_T_coords is not None:
                        # if the coodinates of all T patches are available, draw all bounding boxes on the full tactile image
                        for i in range(len(full_T_coords)):
                            new_ROI_x, new_ROI_y, new_ROI_h, new_ROI_w = full_T_coords[i]
                            new_ROI_x = new_ROI_x.cpu().numpy()[0]
                            new_ROI_y = new_ROI_y.cpu().numpy()[0]
                            new_ROI_h = new_ROI_h.cpu().numpy()[0]
                            new_ROI_w = new_ROI_w.cpu().numpy()[0]
                            gx_phase = cv2.rectangle(
                                gx_phase,
                                (
                                    int(new_ROI_x * T_resolution_multiplier),
                                    int(new_ROI_y * T_resolution_multiplier),
                                ),
                                (
                                    int((new_ROI_x + new_ROI_w) * T_resolution_multiplier),
                                    int((new_ROI_y + new_ROI_h) * T_resolution_multiplier),
                                ),
                                (0, 255, 0),
                                2,
                            )
                    # save the bounding box images
                    additional_visual_names[
                        "%s_I_bb" % phase
                    ] = I_phase  # draw bounding boxes for train, total number of bbox 16 # draw bounding boxes for val, total number of bbox 30
                    additional_visual_names[
                        "%s_gx_bb" % phase
                    ] = gx_phase  # draw bounding boxes for train, total number of bbox 16 # draw bounding boxes for val, total number of bbox 30
                    
                    # save the coordinates of the bounding boxes
                    coords = coords.squeeze()  # [1, 60, 8] -> [60, 8
                    additional_visual_names["%s_patch_coords" % phase] = coords
                    # elementwise round and convert to int for indexing
                    offset_x = np.round(
                        (coords[:, 0] + coords[:, -2] / coords[:, -3]) * T_resolution_multiplier
                    )
                    offset_y = np.round((coords[:, 1] + coords[:, -1] / coords[:, -3]) * T_resolution_multiplier)
                    cutout_size = np.round(coords[:, -4] / coords[:, -3] * T_resolution_multiplier)
                    additional_visual_names["%s_patch_coords" % phase] = np.vstack(
                        [offset_x + cutout_size // 2, offset_y + cutout_size // 2]
                    ).T  # find the center

                # collect the patches for visualization 
                for (
                    i,
                    S_patch,
                    I_mask_patch,
                    real_I_patch,
                    fake_I_patch,
                    real_gx_patch,
                    fake_gx_patch,
                    real_gy_patch,
                    fake_gy_patch,
                    real_normal_patch,
                    fake_normal_patch,
                ) in zip(
                    range(len(S_concat)),
                    S_concat,
                    I_mask_concat,
                    real_I_concat,
                    fake_I_concat,
                    real_gx_concat,
                    fake_gx_concat,
                    real_gy_concat,
                    fake_gy_concat,
                    real_normal_concat,
                    fake_normal_concat,
                ):
                    if i == num_patch_for_logging:
                        break
                    if collage_patch_as_array:
                        # For each sample, combine S, I, T patches as one large numpy array image
                        # convert each tensor to im indivisually in order to not mess up the channel and range
                        # resize vision patches to the same size as touch patches

                        S_patch_arr = util.tensor2im(
                            torch.squeeze(F.interpolate(torch.unsqueeze(S_patch, 0), (H, W)), dim=0)
                        )
                        real_I_patch_tensor = torch.squeeze(
                            F.interpolate(torch.unsqueeze(real_I_patch, dim=0), (H, W)), dim=0
                        )
                        real_I_patch_arr = util.tensor2im(real_I_patch_tensor)
                        fake_I_patch_arr = util.tensor2im(
                            torch.squeeze(F.interpolate(torch.unsqueeze(fake_I_patch, dim=0), (H, W)), dim=0)
                        )
                        real_gx_patch_arr = util.tensor2im(real_gx_patch, convert_gray_imgs=True, colormap=colormap)
                        fake_gx_patch_arr = util.tensor2im(fake_gx_patch, convert_gray_imgs=True, colormap=colormap)
                        real_gy_patch_arr = util.tensor2im(real_gy_patch, convert_gray_imgs=True, colormap=colormap)
                        fake_gy_patch_arr = util.tensor2im(fake_gy_patch, convert_gray_imgs=True, colormap=colormap)
                        real_normal_patch_arr = util.tensor2im(real_normal_patch)
                        fake_normal_patch_arr = util.tensor2im(fake_normal_patch)

                        place_holder_S = I_mask_patch.cpu().numpy().astype(np.uint8) * 255
                        place_holder_S = np.stack([place_holder_S, place_holder_S, place_holder_S], axis=0)
                        place_holder_S = np.transpose(np.squeeze(place_holder_S), (1, 2, 0))
                        collage_img = np.hstack(
                            (
                                np.vstack((S_patch_arr, place_holder_S)),
                                np.vstack((real_I_patch_arr, fake_I_patch_arr)),
                                np.vstack((real_gx_patch_arr, fake_gx_patch_arr)),
                                np.vstack((real_gy_patch_arr, fake_gy_patch_arr)),
                                np.vstack((real_normal_patch_arr, fake_normal_patch_arr)),
                            )
                        )  # upper row is real, lower row is fake, each column is one output, shape (128, 256, 3)
                        additional_visual_names["%s_patch_%d" % (phase, i)] = collage_img

                    # else: # change else condtion to "always" case. save individual patch for raw array output
                    # pass each individual patch as a visual item
                    additional_visual_names["%s_real_gx_patch_%d" % (phase, i)] = real_gx_patch
                    additional_visual_names["%s_fake_gx_patch_%d" % (phase, i)] = fake_gx_patch
                    additional_visual_names["%s_real_gy_patch_%d" % (phase, i)] = real_gy_patch
                    additional_visual_names["%s_fake_gy_patch_%d" % (phase, i)] = fake_gy_patch
                    additional_visual_names["%s_real_normal_patch_%d" % (phase, i)] = real_normal_patch
                    additional_visual_names["%s_fake_normal_patch_%d" % (phase, i)] = fake_normal_patch
                    if save_S_patch:
                        additional_visual_names["%s_S_patch_%d" % (phase, i)] = S_patch
                        additional_visual_names["%s_I_patch_%d" % (phase, i)] = real_I_patch
                        additional_visual_names["%s_I_mask_patch_%d" % (phase, i)] = I_mask_patch
                        additional_visual_names["%s_fake_I_patch_%d" % (phase, i)] = fake_I_patch

            else:  
                # only S is available, draw S, generated I, T
                if T_coords is not None:
                    # run on the whole image
                    S_concat = get_patch_in_input(real_S, coords)
                    fake_I_concat = get_patch_in_input(fake_I, coords)
                    real_I_concat, offset_x, offset_y, cutout_size = get_patch_in_input(
                        real_I, coords, return_offset=True
                    )
                    fake_T_concat = get_patch_in_input(fake_T, coords, scale_multiplier=T_resolution_multiplier)
                else:
                    # run on patches
                    S_concat = real_S
                    fake_I_concat = fake_I
                    real_I_concat = real_I
                    fake_T_concat = fake_T
                    S_concat = get_patch_in_input(real_S, coords)
                    fake_I_concat = get_patch_in_input(fake_I, coords)
                    fake_T_concat = get_patch_in_input(fake_T, coords, scale_multiplier=T_resolution_multiplier)

                fake_gx_concat = torch.unsqueeze(fake_T_concat[:, 0, :, :], 1)
                fake_gy_concat = torch.unsqueeze(fake_T_concat[:, 1, :, :], 1)
                fake_normal_concat = compute_normal(fake_T_concat[:, :2, :, :], scale_nz=scale_nz)
                H, W = fake_gx_concat.shape[-2:]

                additional_visual_names[
                    "%s_I_bb" % phase
                ] = (
                    I.copy()
                )  # draw bounding boxes for train, total number of bbox 16 # draw bounding boxes for val, total number of bbox 30
                additional_visual_names[
                    "%s_gx_bb" % phase
                ] = (
                    gx.copy()
                )  # draw bounding boxes for train, total number of bbox 16 # draw bounding boxes for val, total number of bbox 30

                for i, S_patch, fake_I_patch, fake_gx_patch, fake_gy_patch, fake_normal_patch in zip(
                    range(len(S_concat)),
                    S_concat,
                    fake_I_concat,
                    fake_gx_concat,
                    fake_gy_concat,
                    fake_normal_concat,
                ):
                    if i == num_patch_for_logging:
                        break
                    if collage_patch_as_array:
                        # For each sample, combine S, I, T patches as one large numpy array image
                        # convert each tensor to im indivisually in order to not mess up the channel and range
                        # print("collage_patch_as_array") # resize vision patches to the same size as touch patches
                        S_patch_arr = util.tensor2im(
                            torch.squeeze(F.interpolate(torch.unsqueeze(S_patch, 0), (H, W)), dim=0)
                        )
                        fake_I_patch_arr = util.tensor2im(
                            torch.squeeze(F.interpolate(torch.unsqueeze(fake_I_patch, dim=0), (H, W)), dim=0)
                        )
                        fake_gx_patch_arr = util.tensor2im(fake_gx_patch, convert_gray_imgs=True, colormap=colormap)
                        fake_gy_patch_arr = util.tensor2im(fake_gy_patch, convert_gray_imgs=True, colormap=colormap)
                        fake_normal_patch_arr = util.tensor2im(fake_normal_patch)
                        place_holder_S = I_mask_patch.cpu().numpy().astype(np.uint8) * 255
                        place_holder_S = np.stack([place_holder_S, place_holder_S, place_holder_S], axis=0)
                        place_holder_S = np.transpose(np.squeeze(place_holder_S), (1, 2, 0))

                        collage_img = np.hstack(
                            (
                                S_patch_arr,
                                fake_I_patch_arr,
                                fake_gx_patch_arr,
                                fake_gy_patch_arr,
                                fake_normal_patch_arr,
                            )
                        )
                        additional_visual_names["%s_patch_%d" % (phase, i)] = collage_img

                    # save each individual patch as a visual item
                    additional_visual_names["%s_fake_gx_patch_%d" % (phase, i)] = fake_gx_patch
                    additional_visual_names["%s_fake_gy_patch_%d" % (phase, i)] = fake_gy_patch

            # draw bounding boxes for more fake T samples during training to verify their positions
            if draw_bounding_box and use_more_fakeT:
                assert (
                    fake_sample_offset_x is not None
                    and fake_sample_offset_y is not None
                    and fake_sample_cutout_size is not None
                ), "fake_sample_offset_x, fake_sample_offset_y, fake_sample_cutout_size should not be None"

                # Draw bounding boxes
                I_phase = I.copy()
                for i in range(len(fake_sample_offset_x)):
                    I_phase = cv2.rectangle(
                        I_phase,
                        (int(fake_sample_offset_x[i]), int(fake_sample_offset_y[i])),
                        (
                            int(fake_sample_offset_x[i] + cutout_size[i]),
                            int(fake_sample_offset_y[i] + fake_sample_cutout_size[i]),
                        ),
                        (255, 0, 0),
                        2,
                    )
                    fake_S_sample_arr = util.tensor2im(
                        torch.squeeze(F.interpolate(torch.unsqueeze(fake_S_samples[i], 0), (H, W)), dim=0)
                    )
                    # print(f"check fake_S_sample_arr shape {fake_S_sample_arr.shape}") # (32, 32, 3)
                    additional_visual_names["fake_S_sample_%d" % (i)] = fake_S_sample_arr
                additional_visual_names[
                    "morefakeT_I_bb"
                ] = I_phase  # draw bounding boxes for train, total number of bbox 16 # draw bounding boxes for val, total number of bbox 30

    return additional_visual_names
