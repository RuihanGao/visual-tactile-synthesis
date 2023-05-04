import time
from collections import OrderedDict

import lpips
import numpy as np
import torch
import torch.nn.functional as F

import util.util as util
from models.model_utils import (
    compute_additional_visuals,
    compute_evaluation_metric,
    compute_normal,
)

from . import networks
from .base_model import BaseModel


class Pix2PixModel(BaseModel):
    """
    This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    The script is adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf

    We made the following modifications:
    1. Add options for SKIT project
    2. Change 1 Discriminator to 2 Discriminators for multi-modal output
    3. copy "set_input" from sinskitG_model.py

    The model training requires '--dataset_mode aligned' dataset.
    Originally, it uses a '--netG unet256' U-Net generator, but the patch (32, 32) is too small to use unet_256. Therefore, we use a '--netG resnet_9blocks' ResNet generator instead.

    Main difference from sinskitG_model.py:
    1. Different model architecture
    2. Use patchskit dataset instead of sinskit dataset (train on paired patches, and test on full-scale sketch image)
    3. Don't use positional encoding since we train on patches and test on full images
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """

        ##### Modification for SKIT project #####

        # configure conditional GAN and loss function
        parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")

        # configure network architecture
        parser.add_argument("--lr_G2", type=float, default=0.0005, help="lr for G2 model")

        # configure input and output channels
        parser.add_argument("--sketch_nc", type=int, default=1, help="input channel for sketch image")
        parser.add_argument("--image_nc", type=int, default=3, help="input channel for visual image")
        parser.add_argument("--touch_nc", type=int, default=2, help="input channel for tactile image")

        # configure data
        parser.add_argument(
            "--data_len", type=int, default=200, help="for single image dataset, do this amount of augmentation"
        )
        parser.add_argument(
            "--center_w",
            type=int,
            default=1280,
            help="ROI width of objects in the center of sketch image,so do not crop it",
        )
        parser.add_argument(
            "--center_h",
            type=int,
            default=960,
            help="ROI heigh of objects in the center of sketch image,so do not crop it",
        )
        parser.add_argument(
            "--num_touch_patch_for_logging", type=int, default=10, help="number of images saved for logging"
        )
        parser.add_argument(
            "--use_bg_mask", type=util.str2bool, default=True, help="use background mask to cover non-ROI"
        )
        parser.add_argument(
            "--T_resolution_multiplier",
            type=int,
            default=1,
            help="multipliction factor from visual resolution to tactile resolution, could be 1, 2, 4. 1 means same resolution as visual image (output 1K)",
        )
        parser.add_argument("--padded_size", type=int, default=1800, help="padded size for sketch image")
        parser.add_argument(
            "--sample_bbox_per_patch",
            type=int,
            default=2,
            help="option to load contact mask and center mask for data preprocessing",
        )

        # configure logs
        parser.add_argument(
            "--save_S_patch",
            type=util.str2bool,
            default=False,
            help="option to save S patch to compute the resampling weight",
        )
        parser.add_argument(
            "--save_T_concat_tensor",
            type=util.str2bool,
            default=False,
            help="option to save T_concat to inspect the metric",
        )
        parser.add_argument(
            "--save_raw_arr_vis",
            type=util.str2bool,
            default=False,
            help="option to save raw output to both .npy and .exr format for visualization",
        )
        parser.add_argument(
            "--scale_nz",
            type=float,
            default=0.25,
            help="adjust the scale of nz wrt nx and ny for better visualization of normal map",
        )

        ### special for patchskit dataset ###
        parser.add_argument(
            "--return_patch",
            type=util.str2bool,
            default=False,
            help="option to return a patch or a full image, it depends on the dataset format",
        )

        # Set the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(
            norm="batch",
            netG="resnet_9blocks",
            dataset_mode="aligned",
            dataset="patchskit",
            crop_size=1536,
        )
        verbose_freq = 320  # set it to be divisble by 32 and smaller than len(data_set)
        if is_train:
            parser.set_defaults(
                pool_size=0,
                gan_mode="vanilla",
                return_patch=True,
                batch_size=32,
                display_freq=verbose_freq,
                print_freq=verbose_freq,
                save_latest_freq=verbose_freq,
                validation_freq=verbose_freq,
                save_epoch_freq=50,
                display_id=0,  # disable displaying results on html
                save_raw_arr_vis=False,
            )
        else:
            parser.set_defaults(
                return_patch=False, batch_size=1, save_S_patch=True, save_raw_arr_vis=False, sample_bbox_per_patch=1
            )

        return parser

    def name(self):
        return "Pix2PixModel"

    def __init__(self, opt):
        """
        Initialize the pix2pix class.

        Args:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        ##### Modification for SKIT project #####

        # update configuration for logging
        self.num_patch_for_logging = self.opt.num_touch_patch_for_logging
        # for edited S, we don't expect ground truth for quantitative evaluation
        if "edit" in opt.dataroot:
            self.test_edit_S = True
        else:
            self.test_edit_S = False

        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D", "D2"]
        else:  # during test time, we only load G
            self.model_names = ["G"]

        # specify the images you want to log for visualization.
        # The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ["real_S", "M", "fake_I", "fake_gx", "fake_gy", "fake_N"]
        if not self.test_edit_S:
            self.visual_names.insert(2, "real_I")

        # specify the losses you want to print out for logging.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake", "D2_real", "D2_fake"]

        # specify      the metrics you want to print out for logging.
        # The training/test scripts will call <BaseModel.get_current_metrics>
        self.metric_names = []
        self.eval_metrics = ["I_SIFID", "I_LPIPS", "I_PSNR", "I_SSIM", "T_SIFID", "T_LPIPS", "T_AE", "T_MSE"]
        # For train.py, compute metric for eval samples and train samples. For test.py, compute metric for test samples
        prefix_list = [""]
        if hasattr(self.opt, "train_for_each_epoch") and self.opt.train_for_each_epoch:
            prefix_list.append("train_")

        if self.test_edit_S:
            print("Test on edited sketch, skip metrics")
            pass
        else:
            for prefix in prefix_list:
                for metric in self.eval_metrics:
                    self.metric_names.append(prefix + metric)

        # define loss functions
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionLPIPS_vgg = lpips.LPIPS(net="vgg").to(self.device)
        self.criterionFeat = torch.nn.L1Loss()
        if self.isTrain:
            self.eval_LPIPS = self.criterionLPIPS_vgg  # use the same backbone for LPIPS to aviod CUDA OOM issue
        else:
            self.eval_LPIPS = lpips.LPIPS(net="alex").to(self.device)

        # define networks (both generator and discriminator)
        input_nc = opt.sketch_nc
        output_nc = opt.image_nc + opt.touch_nc
        print("Define G network")
        self.netG = networks.define_G(
            input_nc,
            output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.no_antialias,
            opt.no_antialias_up,
            self.gpu_ids,
            opt,
        )
        if self.isTrain:
            print("Define D networks")
            self.netD = networks.define_D(
                opt.image_nc + opt.sketch_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                opt.no_antialias,
                gpu_ids=self.gpu_ids,
                opt=opt,
            )
            self.netD2 = networks.define_D(
                opt.touch_nc + opt.sketch_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                opt.no_antialias,
                gpu_ids=self.gpu_ids,
                opt=opt,
            )

        # define optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr_G2, betas=(opt.beta1, opt.beta2))
        self.optimizers.append(self.optimizer_D2)

    def set_input(self, input, phase="train", timing=False, verbose=False):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): input data from the dataloader
            phase (str, optional): indicator for train/test. Defaults to 'train'.
            timing (bool, optional): option to print computation time for each step. Defaults to False.
            verbose (bool, optional): option ot print logs for debugging Defaults to False.
        """
        if timing:
            start_time = time.time()
        if verbose:
            print("In set_input")
            print("check data content", [key for key in input.keys()])
            print("len(T_images)", len(input["T_images"]), input["T_images"].shape)

        self.data_phase = phase
        # Update the key for different cases. 1. run on patch data 2. run on full image data
        S_key = "S_images" if self.opt.return_patch else "S"
        M_key = "M_images" if self.opt.return_patch else "M"
        I_key = "I_images" if self.opt.return_patch else "I"

        self.real_S = input[S_key].to(self.device, non_blocking=True)  # equivalent to self.real_A
        self.name = input["name"]
        self.image_paths = input["S_paths"]  # used for save_image in test.py
        self.augmentation_params = input[
            "augmentation_params"
        ]  # the dictionary stores the params used for data augmentation

        # Load the masked version of sketch and images
        if self.opt.use_bg_mask:
            self.M = input[M_key].to(self.device, non_blocking=True)
            self.real_S = self.real_S * self.M  # elementwise multiplication of the feature map and the mask
            self.M_T = F.interpolate(self.M, size=self.M.shape[-1] * self.opt.T_resolution_multiplier).to(self.device)

        if self.test_edit_S:
            pass
        else:
            # For original sketch, where I and T are available
            self.real_I = input[I_key].to(self.device, non_blocking=True)  # equivalent to self.real_B
            if self.opt.use_bg_mask:
                self.real_I = self.real_I * self.M
                # create larger mask for tactile images
                if verbose:
                    print(
                        "check shape after multiplying with the mask, \n real_S {} real_I {} M {}  M_T {}".format(
                            self.real_S.shape, self.real_I.shape, self.M.shape, self.M_T.shape
                        )
                    )  # real_S torch.Size([1, 1, 1536, 1536]) real_I torch.Size([1, 3, 1536, 1536]) M torch.Size([1, 1, 1536, 1536])  M_T torch.Size([1, 1, 1536, 1536])

            # Note: real_T_concat and T_mask will automatically convert to batch_size = 1. Manually reshape here
            assert "T_images" in input.keys(), "T_images not in input keys"
            if self.opt.return_patch:
                self.T_coords = None
            else:
                assert "T_coords" in input.keys(), "T_coords is required for running full images not in input keys"
                self.T_coords = input["T_coords"]
                self.full_T_coords = input["full_T_coords"]
                self.train_T_coords = input["T_coords"].numpy()

            if verbose:
                print(f"check T_images shape. {input['T_images'].shape}")  # [32,2,32,32] / [1, 60, 2, 32, 32]
            if len(input["T_images"].shape) == 4:
                N, C, H, W = input["T_images"].shape
            else:
                assert len(input["T_images"].shape) == 5, "T_images should be 4D or 5D"
                N, NT, C, H, W = input["T_images"].shape
            self.real_T = input["T_images"].reshape(-1, C, H, W).to(torch.float32).to(self.device, non_blocking=True)
            if verbose:
                print("In set_input, check self.real_T", self.real_T.shape)

            # Multiply the real tactile image with the mask
            self.I_masks = input["I_masks"].reshape(-1, 1, H, W).to(torch.float32).to(self.device, non_blocking=True)
            self.real_T = self.real_T * self.I_masks
            # separate the tactile images to gx and gy channel
            self.real_gx = torch.unsqueeze(self.real_T[:, 0, :, :], 1)
            self.real_gy = torch.unsqueeze(self.real_T[:, 1, :, :], 1)
        if timing:
            print("set_input takes time", time.time() - start_time)

    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        # forward pass
        output = self.netG(self.real_S)
        # separate the channels for fake_I and fake_T
        self.fake_I = output[:, 0:3, :, :]
        self.fake_T = output[:, -2:, :, :]

        # print(f"check output shape {output.shape}, real_S {self.real_S.shape}, fake_I {self.fake_I.shape}, fake_T {self.fake_T.shape} M {self.M.shape}, M_T {self.M_T.shape}")  # check output shape torch.Size([32, 5, 32, 32]), real_S torch.Size([32, 1, 32, 32]), fake_I torch.Size([32, 3, 32, 32]), fake_T torch.Size([32, 2, 32, 32]) M torch.Size([32, 1, 32, 32]), M_T torch.Size([32, 1, 32, 32])

        # mask out the background
        if self.opt.use_bg_mask:
            self.fake_I = self.fake_I * self.M
            self.fake_T = self.fake_T * self.M_T

        # parse the tactile output
        self.fake_gx = torch.unsqueeze(self.fake_T[:, 0, :, :], 1)
        self.fake_gy = torch.unsqueeze(self.fake_T[:, 1, :, :], 1)
        self.fake_N = compute_normal(self.fake_T[:, :2, :, :].detach(), scale_nz=self.opt.scale_nz)

    def backward_D(self):
        """
        Compute loss and backpropgate for the discriminators.
        """
        # Fake; stop backprop to the generator by detaching fake_B

        # S -> I, conditional GAN loss
        # Fake
        fake_SI = torch.cat((self.real_S, self.fake_I), 1)
        pred_fake_I = self.netD(fake_SI.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake_I, False)
        # Real
        real_SI = torch.cat((self.real_S, self.real_I), 1)
        pred_real_I = self.netD(real_SI)
        self.loss_D_real = self.criterionGAN(pred_real_I, True)
        # combine loss and calculate gradients
        self.loss_D1 = (self.loss_D_fake + self.loss_D_real) * 0.5

        # S -> T, conditional GAN loss
        fake_ST = torch.cat((self.real_S, self.fake_T), 1)
        pred_fake_T = self.netD2(fake_ST.detach())
        self.loss_D2_fake = self.criterionGAN(pred_fake_T, False)
        # Real
        real_ST = torch.cat((self.real_S, self.real_T), 1)
        pred_real_T = self.netD2(real_ST)
        self.loss_D2_real = self.criterionGAN(pred_real_T, True)
        # combine loss and calculate gradients
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5

        # combine the loss from visual and tactile discriminators
        self.loss_D = self.loss_D1 + self.loss_D2
        self.loss_D.backward()

    def backward_G(self):
        """
        Compute loss and backpropgate for the generator.
        """
        #  GAN loss
        fake_SI = torch.cat((self.real_S, self.fake_I), 1)
        pred_fake_I = self.netD(fake_SI)
        self.loss_G_GAN_I = self.criterionGAN(pred_fake_I, True)

        fake_ST = torch.cat((self.real_S, self.fake_T), 1)
        pred_fake_T = self.netD2(fake_ST)
        self.loss_G_GAN_T = self.criterionGAN(pred_fake_T, True)

        self.loss_G_GAN = self.loss_G_GAN_I + self.loss_G_GAN_T

        # L1 loss
        self.loss_G_L1 = (
            self.criterionL1(self.fake_I, self.real_I) + self.criterionL1(self.fake_T, self.real_T)
        ) * self.opt.lambda_L1

        # combine GAN loss and L1 loss and backpropgate
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self, epoch=0):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration.

        Args:
            epoch (int, optional): index of the current epoch. Defaults to 0.
        """
        # forward pass
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad(set_to_none=True)  # set D's gradients to zero
        self.set_requires_grad(self.netD2, True)  # enable backprop for D
        self.optimizer_D2.zero_grad(set_to_none=True)  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        self.optimizer_D2.step()  # update D2's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD2, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad(set_to_none=True)  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def test(self, timing=False):
        """
        Overwrite the test function in base_model.py
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop.
        Args:
            timing (bool, optional): option to print out the time for forward pass. Defaults to False.
        """
        if timing:
            start_time = time.time()
        with torch.no_grad():
            self.forward()
            if timing:
                print("forward model takes time ", time.time() - start_time)
                start_time = time.time()

    def get_current_visuals(self):
        """
        Return visualization images. train.py will display these images with visdom/wandb, and save the images to a HTML
        This function overwrite the one in base_model.py, because for sinskitG_model, the visuals are divided into two parts:
        1. self-attribute visuals: real_S, real_I, fake_I, fake_gx_full, fake_gy_full
        2. non-self-attribute visuals (for efficient memory usage): patches of S, I, I', T(gx), T'(gx), T(gy), T'(gy) for both train & eval
        It also computes and updates the evaluation.
        """
        # compute attribute visuals and return the results as a dict (OrderedDict)
        self.compute_visuals()
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        # compute non-self-attribute visuals and return the results as a dict (OrderedDict)
        additional_visual_dict = self.compute_additional_visuals_and_metric()
        # merge two dictionaries
        visual_ret.update(additional_visual_dict)
        return visual_ret

    def compute_additional_visuals_and_metric(
        self,
    ):
        """
        Compute non-self-attribute visuals (for efficient memory usage): patches of S, I, I', T(gx), T'(gx), T(gy), T'(gy) for both train & eval
        It also computes and updates the evaluation.
        """
        
        I_mask_concat = (
            self.I_masks
            if hasattr(self, "I_masks")
            else torch.zeros(self.real_T.shape[0], 1, self.real_T.shape[1], self.real_T.shape[2]).to(self.device)
        )
        fake_I = self.fake_I.detach()
        real_T_concat = self.real_T.detach()
        real_I = self.real_I.detach()
        real_S = self.real_S.detach()
        T_coords = self.T_coords
        scale_nz = self.opt.scale_nz
        fake_T = self.fake_T.detach()
        T_resolution_multiplier = self.opt.T_resolution_multiplier
        use_more_fakeT = (
            hasattr(self.opt, "use_more_fakeT") and self.opt.use_more_fakeT and hasattr(self, "fake_sample_offset_x")
        )
        return_patch = self.opt.return_patch
        if hasattr(self, "full_T_coords"):
            full_T_coords = self.full_T_coords
        else:
            full_T_coords = None
        num_patch_for_logging = self.num_patch_for_logging
        save_S_patch = self.opt.save_S_patch
        if use_more_fakeT:
            fake_sample_offset_x = self.fake_sample_offset_x.detach().cpu().numpy().astype(np.int32)
            fake_sample_offset_y = self.fake_sample_offset_y.detach().cpu().numpy().astype(np.int32)
            fake_sample_cutout_size = self.fake_sample_cutout_size.detach().cpu().numpy().astype(np.int32)
            fake_S_samples = self.fake_S_samples
        else:
            fake_sample_offset_x = None
            fake_sample_offset_y = None
            fake_sample_cutout_size = None
            fake_S_samples = None

        additional_visuals = compute_additional_visuals(
            real_S,
            real_I,
            fake_I,
            I_mask_concat,
            real_T_concat,
            fake_T,
            T_coords,
            scale_nz,
            T_resolution_multiplier,
            self.data_phase,
            self.test_edit_S,
            use_more_fakeT,
            return_patch,
            full_T_coords=full_T_coords,
            num_patch_for_logging=num_patch_for_logging,
            save_S_patch=save_S_patch,
            fake_sample_offset_x=fake_sample_offset_x,
            fake_sample_offset_y=fake_sample_offset_y,
            fake_sample_cutout_size=fake_sample_cutout_size,
            fake_S_samples=fake_S_samples,
        )

        # compute evaluation metrics
        prefix = "train_" if self.data_phase == "train" else ""
        metric_dict = compute_evaluation_metric(
            model_names=self.model_names,
            real_I=self.real_I.detach(),
            fake_I=self.fake_I.detach(),
            real_T_concat=self.real_T.detach(),
            fake_T_concat=self.fake_T.detach(),
            eval_metrics=self.eval_metrics,
            eval_LPIPS=self.eval_LPIPS,
            opt=self.opt,
            device=self.device,
            verbose=False,
            timing=False,
            prefix=prefix,
        )
        # assign each key, value pair of metric_dict to self attributes
        for key, value in metric_dict.items():
            setattr(self, key, value)

        return additional_visuals
