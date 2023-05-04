import numpy as np
import os.path

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset, make_touch_image_dataset
from PIL import Image, ImageOps
import random
import util.util as util
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time 
import ntpath
from utils import create_log_dir_by_date
from data.dataset_util import *
import cv2
from util.util import variance_of_laplacian

"""
Custom dataset class for skinskitG model
Returns sketch, mask, (visual, tactile)  image pairs. 
Bracket meeans if the sketch is a new sketch, we don't have ground truth visual and tactile images, so only return sketch and mask.
"""

class SingleSkitDataset(BaseDataset):
    """
    This dataset is customized for visual-tactile synthesis from sketch. 
    Each item contains one sketch image, one object mask, one visual image, multiple tactile patches and their coordinates of the patches in the sketch/visual image.

    It requires three directories to host training images from sketch domain '/path/to/data/trainA' to visual image domain '/path/to/data/trainB', 
    and then to tactile image domain '/path/to/data/trainC' . '/path/to/data/trainC' will also contains coordinates of low-res visual image to crop the square patches.

    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, during test time, 
    you need to prepare at least '/path/to/data/testA' to run the model, 
    and optionally '/path/to/data/testB' and '/path/to/data/testC' to compute any loss.
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
        parser.add_argument('--subdir_S', type=str, default='trainS', help='subdirectory for S input')
        parser.add_argument('--subdir_I', type=str, default='trainI', help='subdirectory for I input')
        parser.add_argument('--subdir_T', type=str, default='trainT', help='subdirectory for T input')
        parser.add_argument('--subdir_M', type=str, default='trainM', help='subdirectory for mask input')
        parser.add_argument('--subdir_valT', type=str, default='valT', help='subdirectory for T input for validation')
        parser.add_argument('--is_train', type=util.str2bool, default=True, help='whether the model is in training mode')

        if is_train:
            parser.set_defaults(subdir_S='trainS', 
            subdir_I='trainI', 
            subdir_T='trainT',
            subdir_M = 'trainM',
            subdir_valT='valT',
            is_train = True,
            )
        else:
            parser.set_defaults(subdir_S='testS', 
            subdir_I='testI', 
            subdir_T='testT',
            subdir_M='testM',
            subdir_valT=None,
            is_train = False,
            )

        return parser


    def __init__(self, opt, verbose=False, default_len=1000):
        """
        Initialize this dataset class.

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
        self.data_len = opt.data_len if hasattr(opt, 'data_len') else default_len

        ### Load data ###

        # Load paths to different data directories
        self.dir_S = os.path.join(opt.dataroot, opt.subdir_S)  # path '/path/to/data/trainS'
        self.dir_I = os.path.join(opt.dataroot, opt.subdir_I)  # path '/path/to/data/trainI'
        self.dir_T = os.path.join(opt.dataroot, opt.subdir_T)  # path '/path/to/data/trainT'
        self.dir_M = os.path.join(opt.dataroot, opt.subdir_M)  # path '/path/to/data/trainM'
        self.is_train = opt.is_train
        # Load validation tactile images if available
        if opt.subdir_valT is not None:
            self.dir_valT = os.path.join(opt.dataroot, opt.subdir_valT)  # path '/path/to/data/valT'
            assert os.path.exists(self.dir_valT), 'missing val T data for train datasets {}'.format(self.dir_valT)
        
        # Process S and M for both original and edited sketch
        assert os.path.exists(self.dir_S), 'missing S data for datasets {}'.format(self.dir_S)
        self.S_paths = sorted(make_dataset(self.dir_S, opt.max_dataset_size))   # load images from '/path/to/data/trainS'
        assert len(self.S_paths) == 1,  "SingleSkitDataset class should be used with one image in sketch S_paths {}".format(self.S_paths)
        if opt.sketch_nc == 1: 
            self.S_img = ImageOps.grayscale(Image.open(self.S_paths[0]))
        else:
            assert opt.sketch_nc == 3, "Load sketch either in grayscale or RGB"
            self.S_img = Image.open(self.S_paths[0]).convert('RGB')

        if self.opt.use_bg_mask is True:
            assert os.path.exists(self.dir_M), "Cannot find valid path for binary mask, %s"%self.dir_M
            self.M_paths = sorted(make_dataset(self.dir_M, opt.max_dataset_size))  
            assert len(self.M_paths) == 1, "SingleSkitDataset class should be used with one image for mask"
            self.M_img = ImageOps.grayscale(Image.open(self.M_paths[0])) 

        # Process I and T for original sketch. For edited sketch input,  we don't have ground truth I and T data.
        if not os.path.exists(self.dir_I):
            print("Warning: missing I data opt dataroot {}, opt subdir_I {}".format(opt.dataroot, opt.subdir_I))
            assert 'edit' in opt.dataroot, "I and T data are required for original sketches"
            self.I_paths = []
            self.I_img = None
            self.T_paths = []
            self.T_size = 0
            
        else:
            assert os.path.exists(self.dir_I) and os.path.exists(self.dir_T), 'datasets directories are invalid, \n dir_I {} \n dir_T {}'.format(self.dir_S, self.dir_I, self.dir_T)
            self.I_paths = sorted(make_dataset(self.dir_I, opt.max_dataset_size))    # load images from '/path/to/data/trainI'
            # Set the images
            assert len(self.I_paths) == 1, "SingleSkitDataset class should be used with one image in sketch and visual image, S_paths {}, I_paths {}".format(self.S_paths, self.I_paths)
            assert opt.image_nc == 3, "Visual image should have RGB 3 channels"
            self.I_img = Image.open(self.I_paths[0]).convert('RGB')

            self.T_paths = make_touch_image_dataset(self.dir_T, opt.max_dataset_size) # load tactile images and coordinates from '/path/to/data/trainT'
            self.T_size = len(self.T_paths) # get the size of dataset T
        
        # Create additional validation data for tactile modality if it is available. Validation shares the same S, M, I with training data for SingleSkitDataset
        if opt.subdir_valT is not None:
            self.val_T_paths = make_touch_image_dataset(self.dir_valT, opt.max_dataset_size) # load tactile images and coordinates from '/path/to/data/valT'
            self.val_T_size = len(self.val_T_paths) # get the size of dataset T
        else:
            self.val_T_paths = None
            self.val_T_size = 0

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
        if hasattr(opt, 'separate_val_set') and opt.separate_val_set:
            separate_val_set = opt.separate_val_set
        else:
            separate_val_set=False
        self.preprocess_data(timing=False, verbose=False, separate_val_set=separate_val_set)


    def preprocess_data(self, timing=False, verbose=False, separate_val_set=False):
        """
        Preprocess the data and save in cache.

        Args:
            timing (bool, optional): option to print out the timing for each step. Defaults to False.
            verbose (bool, optional): option to print out additional logging information. Defaults to False.
            separate_val_set (bool, optional): option to create a separate validation set.  Defaults to False.
            For SingleSkitDataset, training and validation sets share the same sketch and visual image. For PatchSkitDataset, the training and validation sets are totally separated.
        
        Returns:
            After preprocessing, data are cached in `self.data_dict` attribute.
        """

        print(f"Preproccess data for singleskit_dataset and save them in cache, len {len(self)}...")
        preprocess_start_time = time.time()

        # load parameter settings
        if 'padded' in self.opt.dataroot:
            self.padded_size = int(self.opt.dataroot.split('padded_')[1].split('/')[0].split('_')[0])
        
        # For singleskit_dataset, we only have one sketch/visual image. 
        # Thus we perform different augmentation to make a dataset. len(dataset) = # of augmentation
        for index in range(len(self)):
            if timing: last_time = time.time()

            # load parameter settings
            S_path = self.S_paths[0]
            S_img = self.S_img
            I_img = self.I_img
            if self.opt.use_bg_mask: 
                M_path = self.M_paths[0]
                M_img = self.M_img
            else:
                M_path = None
                M_img = None
            method = Image.LANCZOS
            patch_crop_size = 32

            # Step 1. Transformation 1. zoom
            if 'zoom' in self.opt.preprocess:
                # Step 1. zoom - transformation
                scale_factor_h, scale_factor_w = self.zoom_levels_A[0]
                S1 = zoom_img(S_img, scale_factor_h=scale_factor_h, scale_factor_w=scale_factor_w, method=method)
                if I_img is not None: I1 = zoom_img(I_img, scale_factor_h=scale_factor_h, scale_factor_w=scale_factor_w, method=method)
                if M_img is not None: M1 = zoom_img(M_img, scale_factor_h=scale_factor_h, scale_factor_w=scale_factor_w, method=method)
                if verbose: print("after zoom transformation", S1.size)
            else:
                S1 = S_img; I1 = I_img; M1 = M_img
                scale_factor_h = 1; scale_factor_w = 1
            H, W = S_img.size[:2]

            if verbose: 
                print(f"after zoom transformatio {S1.size}, scale_factor_h {scale_factor_h}, scale_factor_w {scale_factor_w}")
            if timing: 
                print('zoom takes', time.time()-last_time)
                last_time = time.time()
            
            # Step 2. Transformation 2. crop
            if 'crop' in self.opt.preprocess:
                crop_size_h = self.opt.crop_size
                crop_size_w = self.opt.crop_size
                crop_pos_x = None; crop_pos_y = None; resize_ratio = None; center_crop=False # set to None to allow random cropping
            else:
                # no augmentation for testing, crop the padded data from the center to keep the same size as training data
                crop_size_h = self.opt.crop_size; crop_size_w = self.opt.crop_size
                crop_pos_x = None; crop_pos_y = None; resize_ratio = None; center_crop = True
            S2, resize_ratio, crop_pos_x, crop_pos_y = crop_img(S1, crop_size_h, crop_size_w, method, resize_ratio, crop_pos_x, crop_pos_y, self.opt.center_w, self.opt.center_h, center_crop=center_crop)
            if I_img is not None: I2, _, _, _ = crop_img(I1, crop_size_h, crop_size_w, method, resize_ratio, crop_pos_x, crop_pos_y)
            if M_img is not None: M2, _, _, _ = crop_img(M1, crop_size_h, crop_size_w, method, resize_ratio, crop_pos_x, crop_pos_y)

            if verbose:
                print(f"after crop transformation {S2.size}, resize_ratio {resize_ratio}, crop_pos_x {crop_pos_x}, crop_pos_y {crop_pos_y}")
            if timing: 
                print('crop takes', time.time()-last_time)
                last_time = time.time()

            # Step 3. Transformation 3. make power 2
            S3, resize_ratio_w, resize_ratio_h = make_power_2_img(S2, 256, method)
            if I_img is not None: I3, resize_ratio_w, resize_ratio_h = make_power_2_img(I2, 256, method)
            if M_img is not None: M3, resize_ratio_w, resize_ratio_h = make_power_2_img(M2, 256, method)

            if verbose: 
                print(f"after make power 2 transformation {S3.size} resize_ratio_w {resize_ratio_w}, resize_ratio_h {resize_ratio_h}")
            if timing: 
                print('power 2 takes', time.time()-last_time) 
                last_time = time.time()

            # Step 4. Convert S, I, M to tensor
            S_tf_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            S_tf = transforms.Compose(S_tf_list); S_tensor = S_tf(S3)
            I_tf_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            I_tf = transforms.Compose(I_tf_list)
            if I_img is not None: 
                assert self.opt.image_nc == 3
                I_tensor = I_tf(I3)
            M_tf_list = [transforms.ToTensor()]
            M_tf = transforms.Compose(M_tf_list)
            if M_img is not None:
                M_tensor = M_tf(M3) # range [0, 1]
            
            # Step 5. Iterate through all coordinates of touch patches, filter those fall within the transformed area
            augmentation_params = {
                "H": H, "W": W, # original size
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

            # If tactile data is available, based on the cooridnates of touch patches, find the corresponding sketch and visual patches
            if self.T_size > 0:
                is_train = self.opt.is_train
                T_images, T_coords, full_T_coords, I_masks, _, _, _ = self.find_validate_touch_patches_and_coords(self.T_size, self.T_paths, augmentation_params, I1, I2, I3, S3, M3, is_train=is_train, is_val=False, method=method, save_compare_vision_touch_plot=False, verbose=verbose, compute_SIM_patches=False, S_tf=S_tf, M_tf=M_tf, I_tf=I_tf)
            # If tactile data is available for validation data, also retrieve corresponding data based on the coordinates of validation patches
            val_T_images = []; val_T_coords = []; val_full_T_coords = []; val_I_masks = []
            if self.val_T_size > 0:
                val_T_images, val_T_coords, val_full_T_coords, val_I_masks, _, _, _ = self.find_validate_touch_patches_and_coords(self.val_T_size, self.val_T_paths, augmentation_params, I1, I2, I3, S3, M3, is_train=is_train, is_val=True, method=method, verbose=verbose, compute_SIM_patches=False, S_tf=S_tf, M_tf=M_tf, I_tf=I_tf)
            short_path = ntpath.basename(S_path[0])
            name = os.path.splitext(short_path)[0]

            # Step 6. Save data in cache
            if I_img is not None: 
                self.data_dict[index] = {'S': S_tensor, 'I': I_tensor, 'name':name, 'I_masks': I_masks, 'val_I_masks': val_I_masks, 'T_images': T_images, 'T_coords': T_coords, 'S_paths': self.S_paths[0], 'augmentation_params': augmentation_params, 'full_T_coords': full_T_coords, 'val_T_images': val_T_images, 'val_T_coords': val_T_coords, 'val_full_T_coords': val_full_T_coords}
            else:
                self.data_dict[index] = {'S': S_tensor, 'name':name, 'S_paths': self.S_paths[0], 'T_images': [], 'augmentation_params': augmentation_params}
            if M_img is not None:
                self.data_dict[index].update({'M': M_tensor, 'M_paths': M_path})

        print("Finish preprocessing %d data, takes "%(len(self)), time.time()-preprocess_start_time)


    def find_validate_touch_patches_and_coords(self, T_size, T_paths, augmentation_params, I1, I2, I3, S3, M3, verbose=False, timing=False, is_train=False, is_val=False, method='bicubic', save_compare_vision_touch_plot=False, load_contact_mask=True, compute_SIM_patches=False, S_tf=None, I_tf=None, M_tf=None, material_index=0):
        """
        Load the coordinates of all tactile patches, filter those fall within the transformed area (after data augmentation), i.e. the valid patches.

        Args:
            T_size (int): number of touch patches
            T_paths (str): paths to tactile data
            augmentation_params (dict): parameters for data augmentation
            I1, I2, I3, S3, M3 (PIL images): Image, Sketch, Mask after augmentation. I3, S3, M3 are the final images after all transformations. Load I1, I2 here for logging purpose.
            verbose (bool, optional): option to print additional logging information
            timing (bool, optional): option to print timing for each step
            is_train (bool, optional): indicator of whether it is training phase
            is_val (bool, optional): indicator of whether it is validataion phase. 
                Note: The phase affects how and how many square patches are sampled. Trying to make the square crops deterministic for testing, but stochastic for training and validation.
            method (str, optional): interpolation method for resizing
            save_compare_vision_touch_plot (bool, optional): option to save a sample plot of visual and tactile patches, useful for debugging
            load_contact_mask (bool, optional): option to load contact mask. contact mask indicates the contact area within the GelSight rectangular sensing surface
            compute_SIM_patches (bool, optional): option to compute Sketch-Image-Mask patches
            S_tf, I_tf, M_tf (transforms): transforms for converting PIL images to tensors
            material_index (int, optional): index of material, used for skit_dataset to track the material index
        
        Returns:
            valid tactile patches, corresponding coordinates, and optionally corresponding patches of sketh, image, image masks
        """
        # Iterate all patches to find the valid ones first, save their indexes, and then perform image processing for valid patches
        valid_indexes_list = []
        valid_ROI1_list = []
        valid_ROI2_list = []
        valid_ROI3_list = []

        if verbose: print("In function find_validate_touch_patches_and_coords")
        last_time = time.time()
        
        for i in range(int(T_size)):          
            # Step 5.1 Load original coords
            touch_data_path = T_paths[i]
            if verbose: print(f'touch_data_path {i} {touch_data_path}')
            gx, gy, ROI_x, ROI_y, ROI_h, ROI_w, touch_mask, touch_center_mask = touch_data_loader(touch_data_path, convert2im=False, return_mask=load_contact_mask)
            if verbose: 
                print(f"Load original ROI_x {ROI_x}, ROI_y {ROI_y}, ROI_h {ROI_h}, ROI_w {ROI_w}")

            # Step 5.2 Update ROI coords based on transformations
            if 'padded' in self.opt.dataroot:
                ROI_x, ROI_y, ROI_h, ROI_w = global_padding_find_coords(ROI_x, ROI_y, ROI_h, ROI_w, padded_size=self.padded_size, org_h=self.opt.center_h, org_w=self.opt.center_w)
                if verbose: print(f"After padding ROI_x {ROI_x}, ROI_y {ROI_y}, ROI_h {ROI_h}, ROI_w {ROI_w}")

            new_ROI_x1, new_ROI_y1, new_ROI_h1, new_ROI_w1 = zoom_find_coords(ROI_x, ROI_y, ROI_h, ROI_w, scale_factor_h=augmentation_params["scale_factor_h"], scale_factor_w=augmentation_params["scale_factor_w"])
            if verbose: 
                print(f"After zoom_find_coords new_ROI_x {new_ROI_x1}, new_ROI_y {new_ROI_y1}, new_ROI_h {new_ROI_h1}, new_ROI_w {new_ROI_w1}")

            valid_patch, new_ROI_x2, new_ROI_y2, new_ROI_h2, new_ROI_w2 = crop_find_coords(new_ROI_x1, new_ROI_y1, new_ROI_h1, new_ROI_w1, augmentation_params["crop_size_h"], augmentation_params["crop_size_w"], augmentation_params["resize_ratio"], augmentation_params["crop_pos_x"], augmentation_params["crop_pos_y"])
            if verbose: 
                print(f"After crop_find_coords new_ROI_x {new_ROI_x2}, new_ROI_y {new_ROI_y2}, new_ROI_h {new_ROI_h2}, new_ROI_w {new_ROI_w2}")     

            new_ROI_x3, new_ROI_y3, new_ROI_h3, new_ROI_w3 = make_power_2_find_coords(new_ROI_x2, new_ROI_y2, new_ROI_h2, new_ROI_w2 , augmentation_params["resize_ratio_w"], augmentation_params["resize_ratio_h"])
            if verbose: 
                print(f"After make_power_2_find_coords new_ROI_x {new_ROI_x3}, new_ROI_y {new_ROI_y3}, new_ROI_h {new_ROI_h3}, new_ROI_w {new_ROI_w3}")    
            
            if valid_patch:
                valid_indexes_list.append(i)
                valid_ROI3_list.append([int(round(new_ROI_x3)), int(round(new_ROI_y3)), int(round(new_ROI_h3)), int(round(new_ROI_w3))])
                valid_ROI1_list.append([int(round(new_ROI_x1)), int(round(new_ROI_y1)), int(round(new_ROI_h1)), int(round(new_ROI_w1))])
                valid_ROI2_list.append([int(round(new_ROI_x2)), int(round(new_ROI_y2)), int(round(new_ROI_h2)), int(round(new_ROI_w2))])

        if timing: 
            print('find valid patch takes', time.time()-last_time)
            last_time = time.time()

        # Step 5.3 Preprocess all valid patches. Then select a subset of size max(batch_size_G2, len(T_images)) to avoid CUDA memory issue
        # assume we can itearate through the dataset given enough number of iterations
        calc_weight = True if hasattr(self.opt, 'w_resampling') and self.opt.w_resampling else False # option to weight differently for each patch based on edge info
        # valid_ROI3_list contains valid patches in the camera frame (center_h, center_w); valid_ROI3_list_update contains pathces in the object mask
        all_T_images, all_T_coords, all_I_masks, weights, valid_ROI3_list_update, all_S_images, all_I_images, all_M_images= self.process_all_valid_patches(valid_indexes_list, valid_ROI3_list, valid_ROI2_list, valid_ROI1_list, T_paths, load_contact_mask, augmentation_params, calc_weight=calc_weight, timing=timing, verbose=verbose, save_compare_vision_touch_plot=save_compare_vision_touch_plot, I1=I1, I2=I2, I3=I3, S3=S3, M3=M3, method=method, compute_SIM_patches=compute_SIM_patches, S_tf=S_tf, I_tf=I_tf, M_tf=M_tf, is_train=is_train, material_index=material_index)
        
        # Random select indexes from valid patches. Process the touch data
        total_num_valid_patch = len(all_T_images)
        if verbose: print("total_num_valid_patch %d"%total_num_valid_patch)
        if hasattr(self.opt, 'batch_size_G2') and self.opt.batch_size_G2 > 0:
            batch_size_G2 = min(self.opt.batch_size_G2, total_num_valid_patch)
        else:
            batch_size_G2 = total_num_valid_patch

        if hasattr(self.opt, 'batch_size_G2_val') and self.opt.batch_size_G2_val > 0:
            batch_size_G2_val = min(self.opt.batch_size_G2_val, total_num_valid_patch)
        else:
            batch_size_G2_val = total_num_valid_patch

        if is_train:
            if not is_val:
                # training set         
                if hasattr(self.opt, 'w_resampling') and self.opt.w_resampling:
                    # weighted resampling
                    selected_idxes = random.choices(range(len(all_T_coords)), weights=weights, k=batch_size_G2)
                else:
                    # random sampling
                    selected_idxes = random.sample(range(len(all_T_coords)), batch_size_G2)
            else:
                # validation set, select a larger batch size
                selected_idxes = random.sample(range(len(all_T_coords)), batch_size_G2_val)
        else:
            # test sets, select all patches
            print("test set, select all patches")
            selected_idxes = range(len(all_T_coords))

        T_images = all_T_images[selected_idxes]
        T_coords = all_T_coords[selected_idxes]
        I_masks = all_I_masks[selected_idxes]
        if compute_SIM_patches:
            S_images = all_S_images[selected_idxes]
            I_images = all_I_images[selected_idxes]
            M_images = all_M_images[selected_idxes]
        else:
            S_images = None; I_images = None; M_images = None
        full_T_coords = valid_ROI3_list_update

        return T_images, T_coords, full_T_coords, I_masks, S_images, I_images, M_images
        

    def process_all_valid_patches(self, valid_indexes_list, valid_ROI3_list, valid_ROI2_list, valid_ROI1_list, T_paths, load_contact_mask, augmentation_params, calc_weight=False, timing=False, verbose=False,save_compare_vision_touch_plot=False, I1=None, I2=None, I3=None, S3=None, M3=None, method=None, compute_SIM_patches=False, S_tf=None, I_tf=None, M_tf=None, is_train=False, save_sample_patches=False, material_index=0): 
        """
        Process all valid patches. Then select a subset of size max(batch_size_G2, len(T_images)) to avoid CUDA memory issue

        Args:
            valid_indexes_list (_type_): _description_
            valid_ROI3_list (_type_): _description_
            valid_ROI2_list (_type_): _description_
            valid_ROI1_list (_type_): _description_
            T_paths (_type_): _description_
            load_contact_mask (_type_): _description_
            augmentation_params (_type_): _description_
            calc_weight (bool, optional): _description_. Defaults to False.
            timing (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to False.
            save_compare_vision_touch_plot (bool, optional): _description_. Defaults to False.
            I1 (_type_, optional): _description_. Defaults to None.
            I2 (_type_, optional): _description_. Defaults to None.
            I3 (_type_, optional): _description_. Defaults to None.
            S3 (_type_, optional): _description_. Defaults to None.
            M3 (_type_, optional): _description_. Defaults to None.
            method (_type_, optional): _description_. Defaults to None.
            compute_SIM_patches (bool, optional): _description_. Defaults to False.
            S_tf (_type_, optional): _description_. Defaults to None.
            I_tf (_type_, optional): _description_. Defaults to None.
            M_tf (_type_, optional): _description_. Defaults to None.
            is_train (bool, optional): _description_. Defaults to False.
            save_sample_patches (bool, optional): option to save sample SIMT patches for logging. Defaults to False.

        Returns:
            _type_: _description_
        """
        last_time = time.time()
        T_tf_list = [transforms.ToTensor()]
        T_tf = transforms.Compose(T_tf_list)

        T_images = [] 
        T_coords = []
        I_masks = []
        gx_im_squares = []
        gy_im_squares = []
        if compute_SIM_patches:
            S_images = []
            I_images = []
            M_images = []   

        # For training purpose, since the batch size for tactile patch is limited, we sample a list of indexes, and save their coords
        weights = []
        if calc_weight:
            assert load_contact_mask == True, "We need contact mask to calculate weights for tactile patches"
        logs_date_dir = create_log_dir_by_date()
        patch_counter = 0
        valid_ROI3_list_update = []
        M3_arr = np.array(M3)
        for i in range(len(valid_indexes_list)):
            # Find the image patch
            patch_index = valid_indexes_list[i]
            new_ROI_x3, new_ROI_y3, new_ROI_h3, new_ROI_w3 = valid_ROI3_list[patch_index]
            # check if at least one pixel in the patch is in object mask M3
            if np.sum(M3_arr[new_ROI_y3:new_ROI_y3+new_ROI_h3, new_ROI_x3:new_ROI_x3+new_ROI_w3]) == 0:
                continue
            valid_ROI3_list_update.append(valid_ROI3_list[patch_index])
            new_ROI_x1, new_ROI_y1, new_ROI_h1, new_ROI_w1 = valid_ROI1_list[patch_index]
            new_ROI_x2, new_ROI_y2, new_ROI_h2, new_ROI_w2 = valid_ROI2_list[patch_index]
            new_ROI_x, new_ROI_y, new_ROI_h, new_ROI_w = new_ROI_x3, new_ROI_y3, new_ROI_h3, new_ROI_w3
            
            coords_index = valid_indexes_list[patch_index]
            touch_data_path = T_paths[coords_index]
            # Crop the touch patch into square shapes
            patch_crop_size_t = augmentation_params["patch_crop_size"]*self.opt.T_resolution_multiplier
            # Method 1.
            # Given the center mask, sample the square center within the mask
            if load_contact_mask:
                # Find the touch patch
                gx_im, gy_im, ROI_x, ROI_y, ROI_h, ROI_w, touch_mask, touch_center_mask = touch_data_loader(touch_data_path, convert2im=False, return_mask=load_contact_mask) # Load the tactile data, the ROI info is useless (coordinates of the tactile patch in vision iamge is handled by new_ROI)
                assert touch_mask is not None and touch_center_mask is not None, "Need valid touch mask and touch center mask"
                # verify whether the touch mask is in contact
                center_ys_contact, center_xs_contact = np.where(touch_center_mask > 0) 
                
                # verify whether the patch falls within object bondary
                center_xs = []; center_ys = []; square_masks = []
                resize_ratio_square = 1
                for center_x, center_y in zip(center_xs_contact, center_ys_contact):
                    square_mask = touch_mask[center_y-patch_crop_size_t//2:center_y+patch_crop_size_t//2, center_x-patch_crop_size_t//2:center_x+patch_crop_size_t//2]
                    crop_pos_x_t = center_x - patch_crop_size_t//2
                    crop_pos_y_t = center_y - patch_crop_size_t//2
                    # convert tactile crop_pos to visual crop_pos
                    crop_pos_x_square = int(crop_pos_x_t/self.opt.T_resolution_multiplier)
                    crop_pos_y_square = int(crop_pos_y_t/self.opt.T_resolution_multiplier)      
                    # Find the corresponding sketch & mask patch, similar to find_coords_for_patch
                    offset_x = np.round((new_ROI_x +crop_pos_x_square/resize_ratio_square)*self.opt.T_resolution_multiplier)
                    offset_y = np.round((new_ROI_y +crop_pos_y_square/resize_ratio_square)*self.opt.T_resolution_multiplier)
                    cutout_size = np.round(augmentation_params["patch_crop_size"]/resize_ratio_square*self.opt.T_resolution_multiplier)
                    # Add the following so that the touch mask not only represents the contact area of the patch, but also represents the object boundary. They could be different because during data collection, the pants are expanded to maintain single layer on the table top.
                    M_patch = np.array(M3.crop((offset_x, offset_y, offset_x+cutout_size, offset_y+cutout_size))) #  min 0, max 255
                    square_mask = square_mask*M_patch/255
                    if np.max(square_mask) >=1:
                        center_xs.append(center_x); center_ys.append(center_y); square_masks.append(square_mask)
                    else:
                        continue
                
                # determine the number of patches to sample
                num_sample_bbox = min(len(center_xs), self.opt.sample_bbox_per_patch)
                if is_train:
                    selected_square_idxes = random.sample(range(len(center_xs)), num_sample_bbox)
                else:
                    # set it deterministic for testing set
                    # # select the first one, which doesn't return most representative patches
                    # selected_square_idxes = np.arange(num_sample_bbox) 
                    # select the center one
                    selected_square_idxes = np.arange(len(center_xs)//2, len(center_xs)//2+num_sample_bbox)

                # sample the patches
                for square_idx in selected_square_idxes:
                    center_x, center_y, square_mask = center_xs[square_idx], center_ys[square_idx], square_masks[square_idx]
                    gx_im_square = gx_im[center_y-patch_crop_size_t//2:center_y+patch_crop_size_t//2, center_x-patch_crop_size_t//2:center_x+patch_crop_size_t//2]
                    gy_im_square = gy_im[center_y-patch_crop_size_t//2:center_y+patch_crop_size_t//2, center_x-patch_crop_size_t//2:center_x+patch_crop_size_t//2]
                    crop_pos_x_t = center_x - patch_crop_size_t//2
                    crop_pos_y_t = center_y - patch_crop_size_t//2
                    
                    # convert tactile crop_pos to visual crop_pos
                    crop_pos_x_square = int(crop_pos_x_t/self.opt.T_resolution_multiplier)
                    crop_pos_y_square = int(crop_pos_y_t/self.opt.T_resolution_multiplier)                
                    # convert to tensor
                    gx_tensor = T_tf(gx_im_square); gy_tensor = T_tf(gy_im_square)
                    # save the patch data, T_images,T_coords
                    gxy = torch.cat((gx_tensor, gy_tensor), 0)
                    assert gxy.shape == (2, patch_crop_size_t, patch_crop_size_t), "gxy shape %s, center_x %d, center_y %d, patch_crop_size_t %d gx_im.shape %s, gy_im.shape %s"%(str(gxy.shape), center_x, center_y, patch_crop_size_t, str(gx_im.shape), str(gy_im.shape))
                    T_images.append(gxy)
                    T_coords.append([new_ROI_x, new_ROI_y, new_ROI_h, new_ROI_w, augmentation_params["patch_crop_size"], resize_ratio_square, crop_pos_x_square, crop_pos_y_square])
                    I_masks.append(square_mask)
                    gx_im_squares.append(gx_im_square)
                    gy_im_squares.append(gy_im_square)

            else:
                # Method 2. 
                # crop to square patches
                # prev: we crop the image to a square patch and save the parameters to deal with the tactile patches. Now: crop the tactile patches directly and save the coordinates
                # In comparison, when passing to crop_img function, the resize_ratio should be the same for vision & tactile. the crop_pos and crop_size should scale accordingly
                gx_im, gy_im, _, _, _, _, _, _ = touch_data_loader(touch_data_path, convert2im=True,verbose=False)
                # Crop the touch patch to self.opt.T_resolution_multiplier*patch_crop_size
                gx_im_square, resize_ratio_square, crop_pos_x_t, crop_pos_y_t = crop_img(gx_im, patch_crop_size_t, patch_crop_size_t, method)
                assert resize_ratio_square == 1, "Maker sure that all patches are larger than 32 pixels"
                gy_im_square, _, _, _ = crop_img(gy_im, patch_crop_size_t, patch_crop_size_t, method, resize_ratio=resize_ratio_square, crop_pos_x=crop_pos_x_t, crop_pos_y=crop_pos_y_t)

                # convert tactile crop_pos to visual crop_pos
                crop_pos_x_square = crop_pos_x_t/self.opt.T_resolution_multiplier
                crop_pos_y_square = crop_pos_y_t/self.opt.T_resolution_multiplier

                # convert to tensor
                gx_tensor = T_tf(gx_im_square); gy_tensor = T_tf(gy_im_square)
                gxy = torch.cat((gx_tensor, gy_tensor), 0)
                # save the patch data, T_images, T_coords
                T_images.append(gxy)
                T_coords.append([new_ROI_x, new_ROI_y, new_ROI_h, new_ROI_w, augmentation_params["patch_crop_size"], resize_ratio_square, crop_pos_x_square, crop_pos_y_square])
                gx_im_squares.append(gx_im_square)
                gy_im_squares.append(gy_im_square)

            if save_compare_vision_touch_plot:
                # plot the sample data of visual and tactile patches, including the different steps of augmentation
                assert I1 is not None and I2 is not None and I3 is not None, "I1, I2, I3 are required for save_compare_vision_touch_plot"          
                logs_date_dir = create_log_dir_by_date()

                if hasattr(self, "I_imgs"):
                    # for skitG dataset, we have a list of materials, choose one to visualize
                    original_I = self.I_imgs[material_index]
                else:
                    # for singleskit dataset, we only have one image
                    original_I = self.I_img
                image_patch = I3.crop((new_ROI_x+crop_pos_x_square, new_ROI_y+crop_pos_y_square, new_ROI_x+crop_pos_x_square+augmentation_params["patch_crop_size"], new_ROI_y+crop_pos_y_square+augmentation_params["patch_crop_size"]))

                fig, axes = plt.subplots(nrows=3, ncols=3)

                axes[0,0].imshow(I1)
                rect_zoom = patches.Rectangle((new_ROI_x1,new_ROI_y1), new_ROI_w1, new_ROI_h1, linewidth=1,  facecolor='none', edgecolor='r')
                axes[0,0].add_patch(rect_zoom)
                axes[0,0].set_title('I1')

                axes[0,1].imshow(I2)
                rect_zoom_cropped = patches.Rectangle((new_ROI_x2,new_ROI_y2), new_ROI_w2, new_ROI_h2, linewidth=1,  facecolor='none', edgecolor='r')
                axes[0,1].add_patch(rect_zoom_cropped)
                axes[0,1].set_title('I2')

                axes[0,2].imshow(I3)
                rect_zoom_cropped = patches.Rectangle((new_ROI_x3,new_ROI_y3), new_ROI_w3, new_ROI_h3, linewidth=1,  facecolor='none', edgecolor='r')
                axes[0,2].add_patch(rect_zoom_cropped)
                axes[0,2].set_title('I3')

                axes[1,0].imshow(I3)
                rect = patches.Rectangle((new_ROI_x3, new_ROI_y3), new_ROI_w3, new_ROI_h3, linewidth=1,  facecolor='none', edgecolor='r')
                axes[1,0].add_patch(rect)
                axes[1,0].set_title('rect image_patch')
                axes[1,1].imshow(image_patch); axes[1,1].set_title('square image_patch')
                
                if load_contact_mask:
                    axes[1,2].imshow(square_mask[0])
                    axes[1,2].set_title('square_mask')
                else:
                    axes[1,2].axis('off')

                axes[2,0].imshow(original_I)
                rect_org = patches.Rectangle((ROI_x, ROI_y), ROI_w, ROI_h, linewidth=1,  facecolor='none', edgecolor='r')
                axes[2,0].add_patch(rect_org)
                axes[2,0].set_title('rect tactile patch')


                axes[2,1].imshow(gx_im); 
                rect_zoom = patches.Rectangle((crop_pos_x_t,crop_pos_y_t), patch_crop_size_t, patch_crop_size_t, linewidth=1,  facecolor='none', edgecolor='r')
                print("crop_pos_x_t", crop_pos_x_t, "crop_pos_y_t", crop_pos_y_t, "patch_crop_size_t", patch_crop_size_t)
                axes[2,1].add_patch(rect_zoom)
                axes[2,1].set_title('gx_im')
                axes[2,2].imshow(gx_im_square); axes[2,2].set_title('square gx_im')

                plt.tight_layout()

                plt.savefig(os.path.join(logs_date_dir, 'Compare_image_touch_patch_%d.png'%patch_index), dpi=900)
                plt.close()
                print("save data for valid patch %d"%patch_index)


        # compute other patches and weights
        for patch_counter, T_coord, gx_im_square, gy_im_square in zip(range(len(T_coords)), T_coords, gx_im_squares, gy_im_squares):
            new_ROI_x, new_ROI_y, new_ROI_h, new_ROI_w, patch_crop_size, resize_ratio_square, crop_pos_x_square, crop_pos_y_square = T_coord

            offset_x = np.round((new_ROI_x +crop_pos_x_square/resize_ratio_square)*self.opt.T_resolution_multiplier)
            offset_y = np.round((new_ROI_y +crop_pos_y_square/resize_ratio_square)*self.opt.T_resolution_multiplier)
            cutout_size = np.round(patch_crop_size/resize_ratio_square*self.opt.T_resolution_multiplier)
            S_patch = np.array(S3.crop((offset_x, offset_y, offset_x+cutout_size, offset_y+cutout_size))) # (left, upper, right, lower)

            if compute_SIM_patches:
                I_patch = np.array(I3.crop((offset_x, offset_y, offset_x+cutout_size, offset_y+cutout_size)))
                M_patch = np.array(M3.crop((offset_x, offset_y, offset_x+cutout_size, offset_y+cutout_size)))
                # convert to tensor
                assert S_tf is not None and I_tf is not None and M_tf is not None, "S_tf, I_tf, M_tf are required for compute_SIM_patches"
                S_images.append(S_tf(S_patch))
                I_images.append(I_tf(I_patch))
                M_images.append(M_tf(M_patch))

            else:
                I_patch = None; M_patch = None
            if save_sample_patches:
                # Save sample patch to confirm coorespondence
                print(f"check S_patch {type(S_patch)}, gx_im_square {type(gx_im_square)}, gy_im_square {type(gy_im_square)}")
                print(f"check S_patch {S_patch.size}, gx_im_square {gx_im_square.shape}, gy_im_square {gy_im_square.shape}")
                print("range of S_patch", np.min(S_patch), np.max(S_patch)) # 255 255
                print("range of gx_im_square", np.min(gx_im_square), np.max(gx_im_square)) #  -0.14376898 0.084498435
                print("range of gy_im_square", np.min(gy_im_square), np.max(gy_im_square)) # -0.117421955 0.29439673
                cv2.imwrite(os.path.join(logs_date_dir, f"S_patch_{patch_counter}.png"), S_patch)
                cv2.imwrite(os.path.join(logs_date_dir, f"gx_im_square_{patch_counter}.png"), ((gx_im_square+1)/2*255).astype(np.uint8))
                cv2.imwrite(os.path.join(logs_date_dir, f"gy_im_square_{patch_counter}.png"), ((gy_im_square+1)/2*255).astype(np.uint8))
                if I_patch is not None: cv2.imwrite(os.path.join(logs_date_dir, f"I_patch_{patch_counter}.png"), I_patch)
                if M_patch is not None: cv2.imwrite(os.path.join(logs_date_dir, f"M_patch_{patch_counter}.png"), M_patch)
                fig, axes = plt.subplots(1, 5,)
                for ax, img, title in zip(axes, [S_patch, I_patch, M_patch, gx_im_square, gy_im_square], ["S_patch", "I_patch", "M_patch", "gx_im_sq", "gy_im_sq"]):
                    ax.imshow(img)
                    ax.set_title(title); ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(logs_date_dir, f"Sample_patch_S_I_M_gx_gy_{patch_counter}.png"), dpi=900)
                plt.close()

            if calc_weight: 
                S_var = variance_of_laplacian(S_patch, ref=np.ones_like(S_patch)*255) # for S, the reference is white
                # set lower and upper bound for S_var. original value could be 0 (all white) or 4x10^5
                S_var = min(max(self.opt.resampling_w_min, S_var), self.opt.resampling_w_max)
                weights.append(S_var)

        if timing: 
            print('process touch patch takes', time.time()-last_time)
            last_time = time.time()

        # covnert tactile image to tensors
        if len(T_images) > 0:
            if len(T_images) > 1:
                T_images = torch.stack(T_images, dim=0)
                T_coords = np.stack(T_coords, axis=0)
                if len(I_masks) > 1:
                    I_masks = torch.from_numpy(np.array(I_masks))
            else:
                # only one non-zero patch
                if verbose:
                    print("only find one touch patch")
                    print("before stack len(T_images) {}, T_images[0]".format(len(T_images)), T_images[0].shape)
                T_images = T_images[0]
                T_images = torch.unsqueeze(T_images, 0)
                T_coords = np.array(T_coords)
                if len(I_masks) > 0:
                    I_masks = torch.unsqueeze(torch.from_numpy(I_masks[0]), 0)
        else: 
            # no available touch patches
            pass
        if compute_SIM_patches:
            assert len(S_images) == len(I_images) == len(M_images) and len(S_images)>1, "S_images, I_images, M_images should have the same length and > 1"
            S_images = torch.stack(S_images, dim=0)
            I_images = torch.stack(I_images, dim=0)
            M_images = torch.stack(M_images, dim=0)
        else:
            S_images = None; I_images = None; M_images = None
        if calc_weight:
            weights = np.array(weights)
            assert len(weights) == len(T_coords), "weights and T_coords should have the same length"
        else:
            weights = None

        if timing: 
            print('convert to tensor takes', time.time()-last_time)
            last_time = time.time()
        return T_images, T_coords, I_masks, weights, valid_ROI3_list_update, S_images, I_images, M_images


    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Args:
            index (int): a random integer for data indexing

        Returns:
            a dictionary that contains S, I, M, T, depending on the available keys
        """
        assert index in self.data_dict.keys(), 'Cannot find index %d in dataset'%(index)
        return self.data_dict[index]


    def __len__(self):
        """ 
        Return the total number of images in the dataset.
        """
        return self.data_len
