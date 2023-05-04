from .singleskit_dataset import SingleSkitDataset

import os.path


from PIL import Image

import util.util as util
import torchvision.transforms as transforms
import torch

import time 
import ntpath

from data.dataset_util import *


"""
PatchSkitDataset is adapted from singleskit dataset for baseline comparison (pix2pix, pix2pixHD, spade).
Since baseline methods need paired data, patchskit loads (sketch, visual, tactile) patches for training and the entire sketch image for testing. 
"""


class PatchSkitDataset(SingleSkitDataset):
    """
    This dataset class is adapted from SingleSkitDataset.
    Instead of returing a single sketch/visual image, it returns a set of sketch/visual patches. 
    The tactile patches are the same. 

    Each item contains a paired set of sketch, visual, tactile, and mask patches.
    Since we only consider patches, we don't need the translation augmentation. 

    It requires three directories to host training images from sketch domain '/path/to/data/trainA' to visual image domain '/path/to/data/trainB', 
    and then to tactile image domain '/path/to/data/trainC' . '/path/to/data/trainC' will also contains coordinates of low-res visual image to crop the square patches.

    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, during test time, 
    you need to prepare at least '/path/to/data/testA' to run the model, 
    and optionally '/path/to/data/testB' and '/path/to/data/testC' to compute any loss.

    More specifically for the implementation: PatchSkitDataset is different from SingleSkitDataset in the following ways:
    1. SingleSkitDataset returns 1 sketch + 1 mask + 1 visual + N tactile patches. PatchSkitDataset returns N sketch + N mask + N visual + N tactile patches.
    2. For getitem, SingleSkitDataset returns a variation of the data augmentaiton (currently being cropping from a larger canvas). PatchSkitDataset returns a SIMT patch among all available patches.
    """


    def __init__(self, opt, verbose=False,  default_len=1000, return_patch=True):
        """
        Initialize the dataset class.
        Compared to SingleSkitDataset, we add the "return_patch" parameter.  We return patchces for training and the entire image for testing.

        Args:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            verbose (bool) -- option to print out more log information
            default_len (int) -- default length of the dataset
            return_patch (bool) -- option to return a patch or the entire image       
        """
        self.return_patch = opt.return_patch if hasattr(opt, 'return_patch') else return_patch
        SingleSkitDataset.__init__(self, opt)  # in __init__ we set self.opt = opt


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

        print(f"Preprocess data for patchskit_dataset and save them in cache, len {len(self)}...")
        preprocess_start_time = time.time()
        
        # Preproccess the data. Given the fix amount of valid tactile patches (GelSight pad data), we control the length of dataset by opt.sample_bbox_per_patch
        if timing: last_time = time.time()

        # load parameter settings
        if 'padded' in self.opt.dataroot:
            self.padded_size = int(self.opt.dataroot.split('padded_')[1].split('/')[0].split('_')[0])
        S_path = self.S_paths[0]
        S_img = self.S_img
        I_img = self.I_img
        if self.opt.use_bg_mask:
            M_img = self.M_img
        else:
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
            if verbose: print("after zoom transformation", I1.size)
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
            if separate_val_set:
                # create a separate validation set. 
                # For SingleSkitDataset, the validataion set and training set shares the S and I; while for PatchSkitDataset, the validation set and training set have different S and I
                T_images, T_coords, full_T_coords, I_masks, S_images, I_images, M_images = self.find_validate_touch_patches_and_coords(self.val_T_size, self.val_T_paths, augmentation_params, I1, I2, I3, S3, M3, is_train=is_train, is_val=True, method=method, save_compare_vision_touch_plot=False, load_contact_mask=True, verbose=verbose, compute_SIM_patches=True, S_tf=S_tf, M_tf=M_tf, I_tf=I_tf)
            else:
                T_images, T_coords, full_T_coords, I_masks, S_images, I_images, M_images = self.find_validate_touch_patches_and_coords(self.T_size, self.T_paths, augmentation_params,I1, I2, I3, S3, M3, is_train=is_train, is_val=False, method=method, save_compare_vision_touch_plot=False, load_contact_mask=True, verbose=verbose, compute_SIM_patches=True,  S_tf=S_tf, M_tf=M_tf, I_tf=I_tf)
            # sample shape: S/I/M/T_images torch.Size([216, 1, 32, 32]), T_coords <class 'numpy.ndarray'> len 216,
        short_path = ntpath.basename(S_path[0])
        name = os.path.splitext(short_path)[0]

        ### Note: the following part is different from implementation in singleskit_dataset ###
        # There are two modes. 1. return S_patches, M_patches, (I_patches, T_patches) 2. return S, M, (I, T_patches)
        # Bracket meeans if the sketch is a new sketch, we only run the model to generate but don't have the ground truth.

        # Step 6. Save data in cache
        if self.return_patch:  
            # Mode 1. return S_patches, M_patches, (I_patches, T_patches)
            self.data_dict = {'S_images': S_images, 'name':[name for _ in range(len(S_images))], 'S_paths': [self.S_paths[0] for _ in range(len(S_images))], 'augmentation_params': [augmentation_params for _ in range(len(S_images))] }
            if I_img is not None: self.data_dict.update({'I_images': I_images, 'T_images': T_images, 'I_masks': I_masks,})
            if M_img is not None: self.data_dict.update({'M_images': M_images})
            self.data_len = len(self.data_dict['S_images'])
            # sample shape: all patches are [425, C, 32, 32]
        else:
            # Mode 2. return S, M, (I, T_patches)
            self.data_dict = {'S': torch.unsqueeze(S_tensor, 0), 'name': [name],  'augmentation_params': [augmentation_params], 'S_paths': [self.S_paths[0]], 'augmentation_params': [augmentation_params]}
            if I_img is not None: self.data_dict.update({'I': torch.unsqueeze(I_tensor,0), 'T_images': torch.unsqueeze(T_images, 0), 'T_coords': [T_coords], 'full_T_coords': [full_T_coords], 'I_masks': torch.unsqueeze(torch.unsqueeze(I_masks, 0), -2)}) # TODO: check I_masks shape, whether we need unsqueeze (-2)
            if M_img is not None: self.data_dict.update({'M': torch.unsqueeze(M_tensor,0)})
            self.data_len = len(self.data_dict['S'])
            # sample shape: S [1, 1, 1536, 1536],  I [1, 3, 1536, 1536], T_images [1, 60, 2, 32, 32]

        print("Finish preprocessing %d data, takes "%(len(self)), time.time()-preprocess_start_time)        


    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Args:
            index (int): a random integer for data indexing

        Returns:
            a dictionary that contains S, I, M, T, depending on the available keys
        """
        item_dict = {}
        for k, v in self.data_dict.items():
            item_dict[k] = v[index]
        return item_dict
 
    def __len__(self):
        """ 
        Return the total number of images in the dataset.
        """
        return self.data_len
