import ntpath
import os.path
import random
import time

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms

import util.util as util
from data.base_dataset import BaseDataset
from data.dataset_util import *
from data.image_folder import make_dataset, make_touch_image_dataset

from .singleskit_dataset import SingleSkitDataset

"""
Custom dataset class for skitG model. Given a list of materials, create a dataset to load vision and touch data of multiple materials
The difference between sinskitG and skitG is that sinskitG trains one model for one object, while skitG trains one model for all objects.
Therefore, in terms of dataset, singleskit dataset takes a dataroot that leads to the directory of a certain object/material, while skitG dataset takes a list of objects/materials and iterate through them to gather all data. 
Note: we use object/material interchangeably here to refer to one object that we collect data from.
"""


class SkitDataset(SingleSkitDataset):
    """
    This dataset is customized for visual-tactile synthesis from sketch. More specifically, it is based on singleskit dataset but used for skitG model that learns one generalized model on multiple objects.
    Each item contains one sketch image, one object mask, one visual image, multiple tactile patches and their coordinates of the patches in the sketch/visual image.

    It requires three directories to host training images from sketch domain '/path/to/data/trainA' to visual image domain '/path/to/data/trainB',
    and then to tactile image domain '/path/to/data/trainC' . '/path/to/data/trainC' will also contains coordinates of low-res visual image to crop the square patches.

    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, during test time,
    you need to prepare at least '/path/to/data/testA' to run the model,
    and optionally '/path/to/data/testB' and '/path/to/data/testC' to compute any loss.

    The difference between sinskitG and skitG is that sinskitG trains one model for one object, while skitG trains one model for all objects.
    Therefore, in terms of dataset, singleskit dataset takes a dataroot that leads to the directory of a certain object/material, while skitG dataset takes a list of objects/materials and iterate through them to gather all data.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Add new dataset-specific options, and rewrite default values for existing options.

        Args:
            parser: original option parser
            is_train (bool): option to set training phase or test phase. use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        Note: "is_train" option is set by "base option setter" in options/base_options.py. It applies to all base model & dataset
        """
        parser.add_argument("--subdir_S", type=str, default="trainS", help="subdirectory for S input")
        parser.add_argument("--subdir_I", type=str, default="trainI", help="subdirectory for I input")
        parser.add_argument("--subdir_T", type=str, default="trainT", help="subdirectory for T input")
        parser.add_argument("--subdir_M", type=str, default="trainM", help="subdirectory for mask input")
        parser.add_argument("--subdir_valT", type=str, default="valT", help="subdirectory for T input for validation")
        parser.add_argument(
            "--is_train", type=util.str2bool, default=True, help="whether the model is in training mode"
        )

        if is_train:
            parser.set_defaults(
                subdir_S="trainS",
                subdir_I="trainI",
                subdir_T="trainT",
                subdir_M="trainM",
                subdir_valT="valT",
                is_train=True,
            )
        else:
            parser.set_defaults(
                subdir_S="testS",
                subdir_I="testI",
                subdir_T="testT",
                subdir_M="testM",
                subdir_valT=None,
                is_train=False,
            )

        return parser

    def __init__(self, opt, verbose=False, default_len=1000):
        """
        Initialize this dataset class.
        We modify the __init__ function of SingleSkitDataset to load data from multiple materials.

        Args:
            opt (Option class): stores all the experiment flags; needs to be a subclass of BaseOptions
            verbose (bool): option to print out more information
            default_len (int): default length of the dataset

        Returns:
            The last step "preprocess_data" save the data_dict to `self.data_dict` attribute.
        """
        BaseDataset.__init__(self, opt)  # in __init__ we set self.opt = opt
        self.verbose = verbose
        self.data_dict = {}
        self.data_len = opt.data_len if hasattr(opt, "data_len") else default_len

        ### Load data ###

        # Set the paths for multiple materials
        if hasattr(self.opt, "material_list"):
            print("material_list is {}".format(self.opt.material_list))

        # For multiple materials, load S, I, M for each material and combine into a list
        self.S_paths = []
        self.I_paths = []
        self.M_paths = []
        self.T_paths = []
        self.T_sizes = []
        self.val_T_paths = []
        self.val_T_sizes = []

        if hasattr(self.opt, "use_external_test_input") and self.opt.use_external_test_input:
            self.style_I_paths = []
            self.style_M_paths = []
            # Use external test input
            print("Use external test input")
            assert hasattr(self.opt, "test_sketch_material"), "test_sketch_material is not defined"
            external_sketch_dataroot = f"./datasets/singleskit_{self.opt.test_sketch_material}_padded_{opt.padded_size}_x{opt.T_resolution_multiplier}_edit0/"

            self.S_paths.extend(
                sorted(make_dataset(os.path.join(external_sketch_dataroot, self.opt.subdir_S), opt.max_dataset_size))
            )
            self.M_paths.extend(
                sorted(make_dataset(os.path.join(external_sketch_dataroot, self.opt.subdir_M), opt.max_dataset_size))
            )

            external_style_dataroot = f"./datasets/singleskit_{self.opt.test_style_material}_padded_{opt.padded_size}_x{opt.T_resolution_multiplier}_edit0/"
            self.style_I_paths.extend(
                sorted(make_dataset(os.path.join(external_style_dataroot, self.opt.subdir_I), opt.max_dataset_size))
            )
            self.style_M_paths.extend(
                sorted(make_dataset(os.path.join(external_style_dataroot, self.opt.subdir_M), opt.max_dataset_size))
            )

            # Note: for testing, we don't need to load val_T_paths

        else:
            print("Iterate over material_list")
            for material in self.opt.material_list:
                dataroot = f"./datasets/singleskit_{material}_padded_{opt.padded_size}_x{opt.T_resolution_multiplier}/"
                dir_S = os.path.join(dataroot, self.opt.subdir_S)
                dir_I = os.path.join(dataroot, self.opt.subdir_I)
                dir_T = os.path.join(dataroot, self.opt.subdir_T)
                dir_M = os.path.join(dataroot, self.opt.subdir_M)
                dir_valT = os.path.join(dataroot, self.opt.subdir_valT) if self.opt.subdir_valT is not None else None

                assert (
                    os.path.exists(dir_S) and os.path.exists(dir_I) and os.path.exists(dir_T) and os.path.exists(dir_M)
                ), "datasets directories are invalid, \n dir_S {} \n dir_I {} \n dir_T {} \n dir_M {}".format(
                    dir_S, dir_I, dir_T, dir_M
                )
                self.S_paths.extend(sorted(make_dataset(dir_S, opt.max_dataset_size)))
                self.I_paths.extend(sorted(make_dataset(dir_I, opt.max_dataset_size)))
                self.M_paths.extend(sorted(make_dataset(dir_M, opt.max_dataset_size)))
                T_paths = make_touch_image_dataset(dir_T, opt.max_dataset_size)
                self.T_paths.append(T_paths)  # For T images, we want a list of list
                self.T_sizes.append(len(T_paths))
                if dir_valT is not None:
                    val_T_paths = make_touch_image_dataset(dir_valT, opt.max_dataset_size)
                else:
                    val_T_paths = []
                self.val_T_paths.append(val_T_paths)
                self.val_T_sizes.append(len(val_T_paths))

        # Set the images
        if opt.sketch_nc == 1:
            self.S_imgs = [ImageOps.grayscale(Image.open(path)) for path in self.S_paths]
        else:
            assert opt.sketch_nc == 3, "Load sketch either in grayscale or RGB"
            self.S_imgs = [Image.open(path).convert("RGB") for path in self.S_paths]
        assert opt.image_nc == 3, "Visual image should have RGB 3 channels"
        if len(self.I_paths) > 0:
            self.I_imgs = [Image.open(path).convert("RGB") for path in self.I_paths]
        else:
            self.I_imgs = None
        if self.opt.use_bg_mask is True:
            self.M_imgs = [ImageOps.grayscale(Image.open(path)) for path in self.M_paths]
        else:
            self.M_imgs = None

        if hasattr(self.opt, "use_external_test_input") and self.opt.use_external_test_input:
            self.style_I_imgs = [Image.open(path).convert("RGB") for path in self.style_I_paths]
            self.style_M_imgs = [ImageOps.grayscale(Image.open(path)) for path in self.style_M_paths]

        ### Data augmentation ###

        if self.opt.is_train:
            # In single-image translation, we augment the data loader by applying
            # random scaling. Still, we design the data loader such that the
            # amount of scaling is the same within a minibatch. To do this,
            # we precompute the random scaling values, and repeat them by |batch_size|.
            A_zoom = 1 / self.opt.random_scale_max
        else:
            A_zoom = 1

        zoom_levels_A = np.random.uniform(A_zoom, 1.0, size=(len(self) // opt.batch_size + 1, 1, 2))
        self.zoom_levels_A = np.reshape(np.tile(zoom_levels_A, (1, opt.batch_size, 1)), [-1, 2])
        if hasattr(opt, "separate_val_set") and opt.separate_val_set:
            separate_val_set = opt.separate_val_set
        else:
            separate_val_set = False
        self.preprocess_data(timing=False, verbose=False, separate_val_set=separate_val_set)

    def preprocess_data(self, timing=False, verbose=False, separate_val_set=False):
        """
        Preprocess the data and save in cache.
        The function is modified from preprocess_data in SingleSkitDataset to load data from multiple materials. We create a for loop and iterate though each material_index.

        Args:
            timing (bool, optional): option to print out the timing for each step. Defaults to False.
            verbose (bool, optional): option to print out additional logging information. Defaults to False.
            separate_val_set (bool, optional): option to create a separate validation set.  Defaults to False.
            For SkitDataset, training and validation sets share the same sketch and visual image. For PatchSkitDataset, the training and validation sets are totally separated.

        Returns:
            After preprocessing, data are cached in `self.data_dict` attribute.
        """

        print(f"Preproccess data for skit_dataset and save them in cache, len {len(self)}...")
        preprocess_start_time = time.time()

        # load parameter settings
        if "padded" in self.opt.dataroot:
            self.padded_size = int(self.opt.dataroot.split("padded_")[1].split("/")[0].split("_")[0])

        # Similar to singleskit_dataset, for skit_dataset, we only have one sketch/visual image per material
        # Thus we perform different augmentation to make a dataset. len(dataset) = # of augmentation
        for index in range(len(self)):
            if timing:
                last_time = time.time()

            # load parameter settings
            material_index = index % len(self.opt.material_list)
            print(f"index {index} material_list {len(self.opt.material_list)} material_index is {material_index}")

            S_path = self.S_paths[material_index]
            S_img = self.S_imgs[material_index]
            if self.I_imgs is not None:
                I_img = self.I_imgs[material_index]
            else:
                I_img = None
            if self.opt.use_bg_mask:
                M_path = self.M_paths[material_index]
                M_img = self.M_imgs[material_index]
            else:
                M_path = None
                M_img = None
            method = Image.LANCZOS
            patch_crop_size = 32
            # load style image
            if hasattr(self.opt, "use_external_test_input") and self.opt.use_external_test_input:
                use_external_test_input = True
                style_I_img = self.style_I_imgs[material_index]
                style_M_img = self.style_M_imgs[material_index]
            else:
                use_external_test_input = False
                style_I_img = None
                style_M_img = None

            # Step 1. Transformation 1. zoom
            if "zoom" in self.opt.preprocess:
                # Step 1. zoom - transformation
                scale_factor_h, scale_factor_w = self.zoom_levels_A[index]
                S1 = zoom_img(S_img, scale_factor_h=scale_factor_h, scale_factor_w=scale_factor_w, method=method)
                if I_img is not None:
                    I1 = zoom_img(I_img, scale_factor_h=scale_factor_h, scale_factor_w=scale_factor_w, method=method)
                if M_img is not None:
                    M1 = zoom_img(M_img, scale_factor_h=scale_factor_h, scale_factor_w=scale_factor_w, method=method)
                if use_external_test_input:
                    style_I1 = zoom_img(
                        style_I_img, scale_factor_h=scale_factor_h, scale_factor_w=scale_factor_w, method=method
                    )
                    style_M1 = zoom_img(
                        style_M_img, scale_factor_h=scale_factor_h, scale_factor_w=scale_factor_w, method=method
                    )
                if verbose:
                    print("after zoom transformation", S1.size)
            else:
                S1 = S_img
                I1 = I_img
                M1 = M_img
                scale_factor_h = 1
                scale_factor_w = 1
                if use_external_test_input:
                    style_I1 = style_I_img
                    style_M1 = style_M_img
            H, W = S_img.size[:2]

            if verbose:
                print(
                    f"after zoom transformatio {S1.size}, scale_factor_h {scale_factor_h}, scale_factor_w {scale_factor_w}"
                )
            if timing:
                print("zoom takes", time.time() - last_time)
                last_time = time.time()

            # Step 2. Transformation 2. crop
            if "crop" in self.opt.preprocess:
                crop_size_h = self.opt.crop_size
                crop_size_w = self.opt.crop_size
                crop_pos_x = None
                crop_pos_y = None
                resize_ratio = None
                center_crop = False  # set to None to allow random cropping
            else:
                # no augmentation for testing, crop the padded data from the center to keep the same size as training data
                crop_size_h = self.opt.crop_size
                crop_size_w = self.opt.crop_size
                crop_pos_x = None
                crop_pos_y = None
                resize_ratio = None
                center_crop = True
            S2, resize_ratio, crop_pos_x, crop_pos_y = crop_img(
                S1,
                crop_size_h,
                crop_size_w,
                method,
                resize_ratio,
                crop_pos_x,
                crop_pos_y,
                self.opt.center_w,
                self.opt.center_h,
                center_crop=center_crop,
            )
            if I_img is not None:
                I2, _, _, _ = crop_img(I1, crop_size_h, crop_size_w, method, resize_ratio, crop_pos_x, crop_pos_y)
            if M_img is not None:
                M2, _, _, _ = crop_img(M1, crop_size_h, crop_size_w, method, resize_ratio, crop_pos_x, crop_pos_y)
            if use_external_test_input:
                style_I2, _, _, _ = crop_img(
                    style_I1, crop_size_h, crop_size_w, method, resize_ratio, crop_pos_x, crop_pos_y
                )
                style_M2, _, _, _ = crop_img(
                    style_M1, crop_size_h, crop_size_w, method, resize_ratio, crop_pos_x, crop_pos_y
                )

            if verbose:
                print(
                    f"after crop transformation {S2.size}, resize_ratio {resize_ratio}, crop_pos_x {crop_pos_x}, crop_pos_y {crop_pos_y}"
                )
            if timing:
                print("crop takes", time.time() - last_time)
                last_time = time.time()

            # Step 3. Transformation 3. make power 2
            S3, resize_ratio_w, resize_ratio_h = make_power_2_img(S2, 256, method)
            if I_img is not None:
                I3, resize_ratio_w, resize_ratio_h = make_power_2_img(I2, 256, method)
            if M_img is not None:
                M3, resize_ratio_w, resize_ratio_h = make_power_2_img(M2, 256, method)
            if use_external_test_input:
                style_I3, _, _ = make_power_2_img(style_I2, 256, method)
                style_M3, _, _ = make_power_2_img(style_M2, 256, method)

            if verbose:
                print(
                    "after make power 2 transformation",
                    S3.size,
                    "resize_ratio_w {}, resize_ratio_h {}".format(resize_ratio_w, resize_ratio_h),
                )

            if timing:
                print("power 2 takes", time.time() - last_time)
                last_time = time.time()

            # Step 4. Convert S, I, M to tensor
            S_tf_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            S_tf = transforms.Compose(S_tf_list)
            S_tensor = S_tf(S3)
            I_tf_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            I_tf = transforms.Compose(I_tf_list)
            if I_img is not None:
                assert self.opt.image_nc == 3
                I_tensor = I_tf(I3)
            M_tf_list = [transforms.ToTensor()]
            M_tf = transforms.Compose(M_tf_list)
            if M_img is not None:
                M_tensor = M_tf(M3)  # range [0, 1]
            if use_external_test_input:
                style_I_tensor = I_tf(style_I3)
                style_M_tensor = M_tf(style_M3)

            # Step 5. Iterate through all coordinates of touch patches, filter those fall within the transformed area
            augmentation_params = {
                "H": H,
                "W": W,  # original size
                "scale_factor_h": scale_factor_h,
                "scale_factor_w": scale_factor_w,
                "crop_size_h": crop_size_h,
                "crop_size_w": crop_size_w,
                "resize_ratio": resize_ratio,
                "crop_pos_x": crop_pos_x,
                "crop_pos_y": crop_pos_y,
                "resize_ratio_w": resize_ratio_w,
                "resize_ratio_h": resize_ratio_h,
                "patch_crop_size": patch_crop_size,
            }

            if I_img is not None:
                # If tactile data is available, based on the cooridnates of touch patches, find the corresponding sketch and visual patches
                if self.T_sizes[material_index] > 0:
                    is_train = self.opt.is_train
                    T_images, T_coords, full_T_coords, I_masks, _, _, _ = self.find_validate_touch_patches_and_coords(
                        self.T_sizes[material_index],
                        self.T_paths[material_index],
                        augmentation_params,
                        I1,
                        I2,
                        I3,
                        S3,
                        M3,
                        is_train=is_train,
                        is_val=False,
                        method=method,
                        save_compare_vision_touch_plot=False,
                        load_contact_mask=self.opt.load_contact_mask,
                        verbose=verbose,
                        compute_SIM_patches=False,
                        S_tf=S_tf,
                        M_tf=M_tf,
                        I_tf=I_tf,
                        material_index=material_index,
                    )

                # If tactile data is available for validation data, also retrieve corresponding data based on the coordinates of validation patches
                val_T_images = []
                val_T_coords = []
                val_full_T_coords = []
                val_I_masks = []

                if self.val_T_sizes[material_index] > 0:
                    (
                        val_T_images,
                        val_T_coords,
                        val_full_T_coords,
                        val_I_masks,
                        _,
                        _,
                        _,
                    ) = self.find_validate_touch_patches_and_coords(
                        self.val_T_sizes[material_index],
                        self.val_T_paths[material_index],
                        augmentation_params,
                        I1,
                        I2,
                        I3,
                        S3,
                        M3,
                        is_train=is_train,
                        is_val=True,
                        method=method,
                        verbose=verbose,
                        compute_SIM_patches=False,
                        S_tf=S_tf,
                        M_tf=M_tf,
                        I_tf=I_tf,
                        material_index=material_index,
                    )
            short_path = ntpath.basename(S_path)
            name = os.path.splitext(short_path)[0]

            # Step 6. Save data in cache
            if I_img is not None:
                self.data_dict[index] = {
                    "S": S_tensor,
                    "I": I_tensor,
                    "name": name,
                    "I_masks": I_masks,
                    "val_I_masks": val_I_masks,
                    "T_images": T_images,
                    "T_coords": T_coords,
                    "S_paths": self.S_paths[0],
                    "augmentation_params": augmentation_params,
                    "full_T_coords": full_T_coords,
                    "val_T_images": val_T_images,
                    "val_T_coords": val_T_coords,
                    "val_full_T_coords": val_full_T_coords,
                }
            else:
                self.data_dict[index] = {
                    "S": S_tensor,
                    "name": name,
                    "S_paths": self.S_paths[0],
                    "T_images": [],
                    "augmentation_params": augmentation_params,
                }
            if M_img is not None:
                self.data_dict[index].update({"M": M_tensor, "M_paths": M_path})

            if use_external_test_input:
                self.data_dict[index].update({"style_I": style_I_tensor, "style_M": style_M_tensor})

        print("Finish preprocessing %d data, takes " % (len(self)), time.time() - preprocess_start_time)
