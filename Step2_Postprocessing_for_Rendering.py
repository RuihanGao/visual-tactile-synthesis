import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import filters

import util.util as util
import myutils

"""
This sript takes the tactile output (gx, gy) from the trianed model and computes the friction map for haptic rendering.
It also takes the visual output (fake_I) and the mask (M), resizes them together with the friction map to fit the size of TanvasTouch screen.
"""


def postprocess_gz(
    fake_I,
    M,
    gx,
    gy,
    Tanvas_width=1280,
    Tanvas_height=800,
    verbose=False,
    use_raw_arr=False,
    thresholding=False,
    threshold_quantile=0.9,
    method="equalize",
    compute_gz=True,
    gz=None,
    change_bg_color=False,
    bg_color=(255, 255, 255),
):
    """
    Given tactile output from the trained model, compute the friction map for haptic rendering.
    Optionally, change the visual output's background color.
    Resize the (visual, haptic) pair to fit the TanvasTouch screen.

    Args:
        fake_I (image, array): visual output from the trained model, range (0,255)
        M (image, array): mask from the trained model, range (0,255)
        gx (image, array): tactile output from the trained model, range (0,255)
        gy (image, array): tactile output from the trained model, range (0,255)
        Tanvas_width (int): width of the TanvasTouch screen
        Tanvas_height (int): height of the TanvasTouch screen
        verbose (bool): print the shape and range of the input and output images
        use_raw_arr (bool): if True, the input gx and gy are loaded from .npy file, range (-1,1); otherwise, they are loaded from .png file, range (0,255)
        thresholding (bool): option to threshold the friction map by a quantile ratio to remove outliers, used for non-linear mapping
        threshold_quantile (float): the quantile used for thresholding
        method (str): the method used for non-linear mapping
        gz (image, array): the friction map computed from gx and gy, range (0,255)
        change_bg_color (bool): option to change the background color of fake_I to bg_color
        bg_color (tuple): the background color of fake_I

    Returns:
        gz_im (image, array): the friction map computed from gx and gy, range (0,255)
        fake_I_im (image, array): visual output from the trained model, range (0,255)
        gz_postprocess_im (image, array): the friction map after postprocessing, range (0,255)
        gz_im_Tanvas (image, array): gz_im resized to fit the TanvasTouch screen, range (0,255)
        fake_I_im_Tanvas (image, array): fake_I_im resized to fit the TanvasTouch screen, range (0,255)
        gz_postprocess_im_Tanvas (image, array): gz_postprocess_im resized to fit the TanvasTouch screen, range (0,255)
    """
    if compute_gz:
        # Load the gx and gy
        if not use_raw_arr:
            # gx and gy are loaded from .png file, range (0,255),
            gx = gx / 255.0 * 2.0 - 1
            gy = gy / 255.0 * 2.0 - 1

        gz = gx**2 + gy**2
    else:
        assert gz is not None, "gz is None, please set compute_gz to True or provide gx and gy to compute gz"

    if thresholding:
        gz_threshold = np.quantile(gz, threshold_quantile)
        # threshold and truncate the higher values
        gz[gz > gz_threshold] = gz_threshold
    gz = (gz - np.min(gz)) / (np.max(gz) - np.min(gz))

    if verbose:
        print("gz", gz.shape, np.min(gz), np.max(gz))
    if len(gz.shape) == 2:
        gz = np.expand_dims(gz, axis=2)
        gz = np.tile(gz, (1, 1, 3))

    print("using postprocessing method: ", method)
    if method == "equalize":
        # Enhance the contrast, need input gz to have 3 channels
        gz_equalize = myutils.equalize_this(gz, with_plot=False, clipLimit=4.0, tileGridLength=4)
        gz_equalize = (gz_equalize - np.min(gz_equalize)) / (np.max(gz_equalize) - np.min(gz_equalize))
        gz_postprocess = gz_equalize

    elif method == "dilation":
        gz_edges = filters.sobel(gz_equalize)  # [0,1]
        gz_edges_normalized = (gz_edges - np.min(gz_edges)) / (np.max(gz_edges) - np.min(gz_edges)) * 255
        gz_edges_normalized = np.array(gz_edges_normalized).astype(np.uint8)
        # add on erosion and dilation
        erosion_kernel = np.ones((1, 1), np.uint8)
        erosion = cv2.erode(gz_edges_normalized, erosion_kernel, iterations=1)
        dk1 = 5
        dk2 = 3
        # dilation wider for the 1st time to avoid obstrusion
        dilation_kernel_1 = np.ones((dk1, dk1), np.uint8)
        dilation = cv2.dilate(erosion, dilation_kernel_1, iterations=1)  # build the inner line
        # print('dilation', type(dilation), dilation.shape, np.min(dilation), np.max(dilation)) # array, (3840, 5120, 3) 0 255
        gz_postprocess = dilation

    elif method == "log10":
        # map [0,1] to [1,10] so that the log value is in [0,1]
        gz_log10 = np.log10(gz * 9.0 + 1.0)
        gz_postprocess = gz_log10

    elif method == "exp2":
        # map [0,1] to [-3, 0]
        gz_exp2 = np.exp2(gz * 3.0 - 3.0)
        gz_postprocess = gz_exp2

    else:
        raise NotImplementedError(f"method {method} for non-linear mapping  is not implemented")

    # normalize the output_img
    gz_postprocess = (gz_postprocess - gz_postprocess.min()) / (np.max(gz_postprocess) - gz_postprocess.min())
    print(f"check gz_postprocess range min {gz_postprocess.min()} max {gz_postprocess.max()}")

    gz_im = np.uint8(gz * 255)
    fake_I_im = np.uint8(fake_I)
    # apply mask and change the background color of fake_I
    if change_bg_color:
        fake_I_im[M < 255] = bg_color  # mask min=127, max=255

    gz_postprocess_im = np.uint8(gz_postprocess * 255)

    gz_im_Tanvas = np.array(Image.fromarray(gz_im).resize((Tanvas_width, Tanvas_height)))
    fake_I_im_Tanvas = np.array(Image.fromarray(fake_I_im).resize((Tanvas_width, Tanvas_height)))
    gz_postprocess_im_Tanvas = np.array(Image.fromarray(gz_postprocess_im).resize((Tanvas_width, Tanvas_height)))

    return gz_im, fake_I_im, gz_postprocess_im, gz_im_Tanvas, fake_I_im_Tanvas, gz_postprocess_im_Tanvas


def generate_Tanvas_images(
    exp_base_name="_sinskitG_baseline_ours",
    train_material="FlowerShorts",
    test_material=None,
    test_edit_data=False,
    edit_index=0,
    verbose=False,
    crop_mask=True,
    output_dir=None,
    thresholding=False,
    threshold_quantile=0.9,
    method="equalize",
    plot_vis=False,
    save_postprocess_im=False,
    add_test_material_prefix=False,
    use_short_exp_name=False,
    short_exp_name=None,
    change_bg_color=False,
    bg_color=(255, 255, 255),
):
    """
    Given the model name and material name, load visual (fake_I) and tactile (gx, gy) outputs and generate Tanvas images.

    Args:
        exp_base_name (str): the base name of the experiment, e.g. "_sinskitG_baseline_ours"
        train_material (str): the material name used for training, e.g. "FlowerShorts"
        test_material (str): the material name used for testing, e.g. "FlowerShorts"
        test_edit_data (bool): option to use the edited data for testing. If True, edit_index must be specified.
        edit_index (int): the index of the edited data to use for testing, used for renaming the output images.
        verbose (bool): option to print out log information.
        crop_mask (bool): option to crop the square outputs to 960x1280 (original camera resolution before padding).
        output_dir (str): the output directory to save the generated images.
        thresholding (bool): option to apply thresholding to the tactile outputs to remove outliers, used for non-linear mapping.
        threshold_quantile (float): the quantile value used for thresholding.
        method (str): the non-linear mapping method to use, e.g. "equalize", "dilation", "log10", "exp2".
        plot_vis (bool): option to plot the intermediate results for debugging.
        save_postprocess_im (bool): option to save the post-processed outputs before resizing them to Tanvas resolution.
        add_test_material_prefix (bool): option to add the test material name as prefix to the output images.
        use_short_exp_name (bool): option to use the short experiment name, e.g. "FlowerShorts_baseline_ours".
        short_exp_name (str): the short experiment name, e.g. "FlowerShorts_baseline_ours".
        change_bg_color (bool): option to change the background color of the visual outputs.
        bg_color (tuple): the RGB color tuple used for changing the background color.
    """

    # Step 1. Load data
    results_parent_dir = "results"
    exp_name = f"{train_material}{exp_base_name}"
    if test_material is None:
        test_material = train_material
    if test_edit_data:
        edit_postfix = f"_edit{edit_index}"
    else:
        edit_postfix = ""

    # filter the test_epoch
    phase = "test"
    if any("best" in s for s in os.listdir(os.path.join(results_parent_dir, exp_name))):
        epoch = "best"
    else:
        assert any(
            "400" in s for s in os.listdir(os.path.join(results_parent_dir, exp_name))
        ), f"no suitable checkpoint exists {exp_name}"
        epoch = 400  # use 400 epoch. some are trained to 400, some are trained to 500. take the minimum
    test_epoch = f"{phase}_{epoch}"

    if verbose:
        print(f"test_epoch is {test_epoch}")
    result_dir = os.path.join(results_parent_dir, exp_name, test_epoch, "images")

    png_name = f"{test_material}_test_0_padded_1800{edit_postfix}_edge.png"
    fake_I = cv2.imread(os.path.join(result_dir, "fake_I", png_name))
    M = cv2.imread(os.path.join(result_dir, "M", png_name), cv2.IMREAD_GRAYSCALE)

    # Method 1. Load gx and gy from .png file
    # gx = cv2.imread(os.path.join(result_dir, 'fake_gx', png_name))
    # gy = cv2.imread(os.path.join(result_dir, 'fake_gy', png_name))

    # Method 2. Load gx and gy from .npy file
    npy_name = f"{test_material}_test_0_padded_1800{edit_postfix}_edge.npy"
    gx = np.load(os.path.join(result_dir, "fake_gx", npy_name))
    gy = np.load(os.path.join(result_dir, "fake_gy", npy_name))

    if crop_mask:
        # crop the mask to save space
        center_h = 960
        center_w = 1280  # image size of camera frame
        # the network resized 1800x1800 to 1536x1536
        H, W, C = fake_I.shape
        crop_pox_y = (H - center_h) // 2
        crop_pox_x = (W - center_w) // 2
        fake_I = fake_I[crop_pox_y : crop_pox_y + center_h, crop_pox_x : crop_pox_x + center_w, :]
        M = M[crop_pox_y : crop_pox_y + center_h, crop_pox_x : crop_pox_x + center_w]
        gx = gx[crop_pox_y : crop_pox_y + center_h, crop_pox_x : crop_pox_x + center_w]
        gy = gy[crop_pox_y : crop_pox_y + center_h, crop_pox_x : crop_pox_x + center_w]

    # visualize fake_I, gx, gy
    if plot_vis:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, img, title in zip(axes, [fake_I, gx, gy], ["fake_I", "gx", "gy"]):
            if "I" in title:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
            ax.set_title(title)
            ax.axis("off")
        plt.suptitle(f"Load data {exp_name}")
        plt.tight_layout()
        plt.show()

    if verbose:
        # check range of gx, gy
        print("gx", np.min(gx), np.max(gx), gx.shape, gx.dtype)
        print("gy", np.min(gy), np.max(gy), gy.shape, gy.dtype)

    # Step 2. Generate images for Tanvas
    gz_im, fake_I_im, gz_postprocess_im, gz_im_Tanvas, fake_I_im_Tanvas, gz_postprocess_im_Tanvas = postprocess_gz(
        fake_I,
        M,
        gx,
        gy,
        verbose=False,
        use_raw_arr=True,
        thresholding=thresholding,
        threshold_quantile=threshold_quantile,
        method=method,
        change_bg_color=change_bg_color,
        bg_color=bg_color,
    )

    # Step 3. Visualize and save images for Tanvas
    if verbose:
        # check shape of all images
        print("gz_im", gz_im.shape, np.min(gz_im), np.max(gz_im))
        print("fake_I_im", fake_I_im.shape, np.min(fake_I_im), np.max(fake_I_im))
        print("gz_postprocess_im", gz_postprocess_im.shape, np.min(gz_postprocess_im), np.max(gz_postprocess_im))
        print("gz_im_Tanvas", gz_im_Tanvas.shape, np.min(gz_im_Tanvas), np.max(gz_im_Tanvas))
        print("fake_I_im_Tanvas", fake_I_im_Tanvas.shape, np.min(fake_I_im_Tanvas), np.max(fake_I_im_Tanvas))
        print(
            "gz_postprocess_im_Tanvas",
            gz_postprocess_im_Tanvas.shape,
            np.min(gz_postprocess_im_Tanvas),
            np.max(gz_postprocess_im_Tanvas),
        )

    # Visualize
    if plot_vis:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title in zip(
            axes, [gz_im, fake_I_im, gz_postprocess_im], ["gz_im", "fake_I_im", "gz_postprocess_im"]
        ):
            if "I" in title:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    # Visualize Tanvas img
    if plot_vis:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title in zip(
            axes,
            [gz_im_Tanvas, fake_I_im_Tanvas, gz_postprocess_im_Tanvas],
            ["gz_im_Tanvas", "fake_I_im_Tanvas", "gz_postprocess_im_Tanvas"],
        ):
            if "I" in title:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    # Save Tanvas images
    if output_dir is None:
        output_dir = myutils.create_log_dir_by_date(parent_dir=".", log_dir="results")
    print(f"Save output Tanvas maps to {output_dir}")
    if not use_short_exp_name:
        save_name = exp_name
    else:
        save_name = short_exp_name
    if add_test_material_prefix:
        save_name = f"{save_name}_{test_material}"
    if save_postprocess_im:
        cv2.imwrite(os.path.join(output_dir, f"{save_name}_gz_im.png"), gz_im)
        cv2.imwrite(os.path.join(output_dir, f"{save_name}_fake_I_im.png"), fake_I_im)
        cv2.imwrite(os.path.join(output_dir, f"{save_name}_gz_postprocess_im.png"), gz_postprocess_im)
    cv2.imwrite(os.path.join(output_dir, f"{save_name}_gz_im_Tanvas.png"), gz_im_Tanvas)
    cv2.imwrite(os.path.join(output_dir, f"{save_name}_fake_I_im_Tanvas.png"), fake_I_im_Tanvas)
    cv2.imwrite(os.path.join(output_dir, f"{save_name}_gz_postprocess_im_Tanvas.png"), gz_postprocess_im_Tanvas)


if __name__ == "__main__":
    output_folder = "Tanvas_maps"
    output_dir = os.path.join(myutils.create_log_dir_by_date(parent_dir=".", log_dir="results"), output_folder)
    os.makedirs(output_dir, exist_ok=True)
        materials = ["BlackJeans", "BluePants", "BlueSports", "BrownVest", "ColorPants", "ColorSweater", "DenimShirt",
        "FlowerJeans", "FlowerShorts", "GrayPants", "GreenShirt", "GreenSkirt", "GreenSweater", "GreenTee",
        "NavyHoodie", "PinkShorts", "PurplePants", "RedShirt", "WhiteTshirt", "WhiteVest"]

    model_type = "sinskitG"  # "skitG"

    if model_type == "sinskitG":
        # Generate Tanvas images for single-object model
        exp_base_names = [
            "_sinskitG_baseline_ours",
        ]  # "_pix2pix_baseline", "_pix2pixHD_baseline", "_spade_baseline" "_sinskitG_abl_allGAN", "_sinskitG_abl_allrec"


        train_materials = ["GrayPants"]

        test_material = "BluePants"  # None
        test_edit_data = True  # False
        add_test_material_prefix = (True,)  # False

        for train_material in train_materials:
            for exp_base_name in exp_base_names:
                generate_Tanvas_images(
                    exp_base_name=exp_base_name,
                    train_material=train_material,
                    test_material=test_material,
                    test_edit_data=test_edit_data,
                    add_test_material_prefix=add_test_material_prefix,
                    edit_index=0,
                    verbose=False,
                    crop_mask=True,
                    output_dir=output_dir,
                    thresholding=True,
                    threshold_quantile=0.98,
                    method="log10",
                    plot_vis=False,
                    save_postprocess_im=True,
                    change_bg_color=True,
                )
            #     break # exit the loop for exp_base_name
            # break # exit the loop for train_material

    elif model_type == "skitG":
        # Generate Tanvas images for cross-object model
        exp_base_name = "OneForAll_mat_BlackJeans_BluePants_BlueSports_ColorPants_FlowerJeans_FlowerShorts_GrayPants_GreenSkirt_PinkShorts_PurplePants_style_True_adain_NL_4_mapping_project_len_20_multiscale_debug"
        short_exp_name = "OneForAll_10_pants"
        train_material = ""  # for cross-object model, train_material is empty, no prefix
        test_material = "BluePants"
        for test_material in materials:
            generate_Tanvas_images(
                exp_base_name=exp_base_name,
                train_material=train_material,
                test_material=test_material,
                test_edit_data=False,
                edit_index=0,
                verbose=False,
                crop_mask=True,
                output_dir=output_dir,
                thresholding=True,
                threshold_quantile=0.98,
                method="log10",
                plot_vis=False,
                save_postprocess_im=True,
                add_test_material_prefix=True,
                use_short_exp_name=True,
                short_exp_name=short_exp_name,
            )
            # break # exit the loop for test_material
