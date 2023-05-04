import os
import time
from collections import OrderedDict

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import vision_aided_loss

# for metrics
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import structural_similarity_index_measure as SSIM

import util.util as util
from models.model_utils import (
    compute_evaluation_metric,
    compute_normal,
    get_patch_in_input,
)
from models.normal_losses import *
from models.sifid import *
from models.tactile_patch_fid import *
from thirdparty.DiffAugment import *
from thirdparty.mmgeneration.positional_encoding import CatersianGrid as CSG
from thirdparty.mmgeneration.positional_encoding import (
    SinusoidalPositionalEmbedding as SPE,
)

from . import networks
from .base_model import BaseModel


class SinSKITGModel(BaseModel):
    """
    This model implements SKIT model, image-to-image translation
    for sketch-image-touch.
    The code borrows heavily from the PyTorch implementation of Contrastive Unpaired Translation (CUT)
    https://github.com/taesungp/contrastive-unpaired-translation
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Configures options specific for SKIT model
        """

        # configure contiditonal GAN & loss function
        parser.add_argument(
            "--use_cGAN",
            type=util.str2bool,
            default=True,
            help="choice to replace GAN loss by conditional GAN loss",
        )
        parser.add_argument(
            "--lambda_G1_GAN",
            type=float,
            default=1.0,
            help="weight for GAN loss: GAN(G(X))",
        )
        parser.add_argument(
            "--lambda_G1_L1",
            type=float,
            default=100.0,
            help="weight for L1 loss: L1(G(X), X)",
        )
        parser.add_argument(
            "--lambda_G1_lpips",
            type=float,
            default=1.0,
            help="weight for perceptual loss: LPIPS(G(X))",
        )
        parser.add_argument(
            "--use_cGAN_G2",
            type=util.str2bool,
            default=True,
            help="choice to use conditional GAN for generating tactile images",
        )
        parser.add_argument(
            "--use_cGAN_G2_S",
            type=util.str2bool,
            default=True,
            help="choice to use conditional GAN for generating tactile images, conditionedl on ",
        )
        parser.add_argument(
            "--use_cGAN_G2_I",
            type=util.str2bool,
            default=True,
            help="choice to use conditional GAN for generating tactile images",
        )
        parser.add_argument(
            "--lambda_G2_GAN",
            type=float,
            default=5.0,
            help="weight for G2 GAN loss: GAN(G2(X))",
        )
        parser.add_argument(
            "--lambda_G2_L1",
            type=float,
            default=10.0,
            help="weight for G2 L1 loss: L1(G2(X), X)",
        )
        parser.add_argument(
            "--lambda_G2_lpips",
            type=float,
            default=10.0,
            help="weight for G2 perceptual loss: LPIPS(G2(X))",
        )
        parser.add_argument(
            "--lambda_G2_GAN_feat",
            type=float,
            default=1,
            help="weight for G2 GAN feature matching loss: L1(F(G2(X)), F(X))",
        )
        parser.add_argument(
            "--smooth_GAN_label",
            type=util.str2bool,
            nargs="?",
            const=False,
            default=True,
            help="smooth GAN label from 1 to 0.7",
        )
        # option to add vision-aided-loss for RGB image generation
        parser.add_argument(
            "--use_vision_aided_loss",
            type=util.str2bool,
            default=True,
            help="option to use vision-aided-loss for RGB image generation",
        )
        parser.add_argument(
            "--vision_aided_warmup_epoch",
            type=int,
            default=100,
            help="number of epochs to warm up for stable GAN training before adding vision-aided-loss",
        )

        # configure network architecture
        parser.add_argument("--lr_G2", type=float, default=0.0005, help="lr for G2 model")
        parser.add_argument(
            "--netD2",
            type=str,
            default="basic",
            help="network structure, [basic | n_layers | pixel | stylegan2], n_layers with n_layers_D2=3 is the same as basic",
        )
        parser.add_argument(
            "--n_layers_D2",
            type=int,
            default=3,
            help="only used if which_model_netD2==n_layers",
        )
        parser.add_argument(
            "--num_layer_separate",
            type=int,
            default=4,
            help="number of separate layers in the upsampling decoder, should be in range [0, num_downs], num_downs for Unet Generator, 8 for unet256, 7 for unet128",
        )
        parser.add_argument(
            "--num_D_D2",
            type=int,
            default=3,
            help="num of layer output to use in the multiple discriminators",
        )
        parser.add_argument(
            "--num_D_D1",
            type=int,
            default=3,
            help="num of layer output to use in the multiple discriminators",
        )
        parser.add_argument(
            "--model_phase",
            type=str,
            default="train",
            help="set model to train/eval for visual_names",
        )

        # configure input and output channels
        parser.add_argument("--sketch_nc", type=int, default=1, help="input channel for sketch image")
        parser.add_argument("--image_nc", type=int, default=3, help="input channel for visual image")
        parser.add_argument("--touch_nc", type=int, default=2, help="input channel for tactile image")
        # option to add positional encoding to the input of generator
        parser.add_argument(
            "--use_positional_encoding",
            type=util.str2bool,
            default=True,
            help="option to use positional encoding with sketch input",
        )
        parser.add_argument(
            "--positional_encoding_mode",
            type=str,
            default="spe",
            choices=["spe", "csg"],
            help="mode for positional encoding, spe for Sinusoidal Positional Encoding, csg for Cartesian Grid encoding",
        )
        parser.add_argument(
            "--positional_encoding_dim",
            type=int,
            default=4,
            help="dimension of positional encoding, constant 1 for csg, variable parameter for spe",
        )

        # configure data
        parser.add_argument(
            "--data_len",
            type=int,
            default=200,
            help="for single image dataset, do this amount of augmentation",
        )
        parser.add_argument("--batch_size_G2", type=int, default=64, help="batch size for G2 model")
        parser.add_argument(
            "--batch_size_G2_val",
            type=int,
            default=128,
            help="batch size for G2 model for validation",
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
            "--T_resolution_multiplier",
            type=int,
            default=1,
            help="multipliction factor from visual resolution to tactile resolution, could be 1, 2, 4. 1 means same resolution as visual image (output 1K)",
        )
        parser.add_argument("--padded_size", type=int, default=1800, help="padded size for sketch image")
        parser.add_argument(
            "--num_touch_patch_for_logging", type=int, default=10, help="number of images saved for logging"
        )
        parser.add_argument(
            "--use_bg_mask",
            type=util.str2bool,
            default=True,
            help="use background mask to cover non-ROI",
        )
        parser.add_argument(
            "--use_more_fakeT",
            type=util.str2bool,
            default=True,
            help="choice to use more fake T patches for discriminator D2",
        )
        parser.add_argument(
            "--add_fake_T_sample_size",
            type=int,
            default=32,
            help="sample size of creating more fake T patches for discriminator D2",
        )
        parser.add_argument(
            "--sample_bbox_per_patch",
            type=int,
            default=2,
            help="option to load contact mask and center mask for data preprocessing",
        )
        # option for DiffAug for discriminator augmentation
        parser.add_argument(
            "--use_diffaug",
            type=util.str2bool,
            default=True,
            help="option to use DiffAug for discriminator augmentation",
        )
        parser.add_argument(
            "--diffaugment",
            type=str,
            default="bs",
            help="DiffAugment parameters, b-brighness, s-saturation, c-contrast, t-translation, o-cutout",
        )
        # option to resample the touch patches based on calcualted weight
        parser.add_argument(
            "--w_resampling",
            type=util.str2bool,
            default=True,
            help="option to resample the touch patches based on calcualted weight",
        )
        parser.add_argument(
            "--resampling_w_min",
            type=int,
            default=1,
            help="minimum weight for resampling",
        )
        parser.add_argument(
            "--resampling_w_max",
            type=int,
            default=10,
            help="maximum weight for resampling",
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

        parser.set_defaults(
            model="sinskitG",
            dataset_mode="singleskit",
            netG="unet256_custom",
            netD="multiscale",
            netD2="multiscale",
            gan_mode="nonsaturating",
            ngf=10,
            ndf=8,
            lr=0.001,
            beta1=0.0,
            beta2=0.99,
            crop_size=1536,
            no_flip=True,
            dataroot="./datasets/singleskit_FlowerShorts_padded_1800_x1/",
        )

        verbose_freq = 3  # default is 100, set to 3 for debugging
        if is_train:
            parser.set_defaults(
                preprocess="crop",  # "zoom_and_crop"
                batch_size=1,  # Set to 1 to avoid incurring error for collocation of touch data
                display_freq=verbose_freq,  # for faster debugging
                print_freq=verbose_freq,
                save_latest_freq=verbose_freq,
                validation_freq=verbose_freq,
                save_epoch_freq=50,
                n_epochs=5,  # keep the same learning rate for the first <opt.n_epochs> epochs
                # For lr_policy='linear', linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
                n_epochs_decay=400,
                num_threads=0,  # set to 0 to disable multi-process data loading. otherwise, may incur errors
                batch_size_G2=64,  # set to 0 if you want to use all available touch patches, but may encounter memory issue when adding lpips loss
                val_for_each_epoch=True,  # 2022-06-09. Add testing after each epoch
                model_phase="train",
                display_id=0,  # try to disable displaying results on html
                save_raw_arr_vis=False,
            )
        else:
            parser.set_defaults(
                preprocess="none",  # load the whole image as it is
                batch_size=1,
                num_test=1,
                data_len=1,  # no augmentation required for testing
                epoch="latest",  # set to epoch number if not using the latest model
                # for testing dataset, only 13 test patches, set a lower limit for visualiza
                num_touch_patch_for_logging=100,
                batch_size_G2=100,
                model_phase="eval",
                display_id=0,  # try to disable displaying results on html
                save_S_patch=True,
                # True.  set to False so that we don't keep so many files in the server
                save_raw_arr_vis=False,
                sample_bbox_per_patch=1,
            )

        return parser

    def __init__(self, opt):
        """
        Initialize the SinSKIT class.

        Args:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.device = torch.device("cuda:{}".format(self.gpu_ids[0])) if self.gpu_ids else torch.device("cpu")

        # update configuration for logging
        self.num_patch_for_logging = min(self.opt.batch_size_G2, self.opt.num_touch_patch_for_logging)
        # for edited S, we don't expect ground truth for quantitative evaluation
        if "edit" in opt.dataroot:
            self.test_edit_S = True
        else:
            self.test_edit_S = False

        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = []
        if self.isTrain:
            self.model_names.append("G")
            if opt.lambda_G1_GAN > 0.0:
                # only add discriminator if GAN loss is used
                self.model_names.append("D")

            if opt.lambda_G2_GAN > 0.0:
                self.model_names.append("D2")
            else:
                if hasattr(opt, "lambda_G2_GAN_feat") and opt.lambda_G2_GAN_feat > 0.0:
                    opt.lambda_G2_GAN_feat = 0.0  # disable GAN feature loss if GAN loss is not used
        else:  # during test time, only load Generators
            self.model_names.append("G")

        # specify the images you want to log for visualization.
        # The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = []
        self.visual_names.extend(["real_S", "M", "fake_I", "fake_gx", "fake_gy", "fake_N"])
        if not self.test_edit_S:
            self.visual_names.insert(2, "real_I")
        if self.isTrain and opt.lambda_G1_GAN > 0:
            self.visual_names.extend(["pred_fake_I"])
        if self.isTrain and opt.lambda_G2_GAN > 0:
            self.visual_names.extend(["pred_fake_T_full"])
        if hasattr(opt, "use_diffaug") and opt.use_diffaug and not self.test_edit_S:  # no ground truth for edited S
            self.visual_names.extend(["aug_fake_I", "aug_real_I"])

        # specify the losses you want to print out for logging
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = []
        if hasattr(self.opt, "train_for_each_epoch") and self.opt.train_for_each_epoch:
            # loss for visual synthesis
            if opt.lambda_G1_GAN > 0.0:
                self.loss_names.extend(["G_GAN", "D_real_I", "D_fake_I", "D_I_grad_penalty"])
                if hasattr(opt, "use_vision_aided_loss") and opt.use_vision_aided_loss:
                    self.loss_names.extend(["G_D3", "D3_real_I", "D3_fake_I"])

            if opt.lambda_G1_L1 > 0.0:
                self.loss_names.append("G_L1")

            if opt.lambda_G1_lpips > 0.0:
                self.loss_names.append("G_lpips")

            # loss for tactile synthesis
            if opt.lambda_G2_GAN > 0.0:
                self.loss_names.extend(["G2_GAN", "D_real_T_concat", "D_fake_T_concat", "D_T_grad_penalty"])
                if hasattr(self.opt, "use_more_fakeT") and self.opt.use_more_fakeT:
                    self.loss_names.append("D_more_fake_T")

            if opt.lambda_G2_L1 > 0.0:
                self.loss_names.append("G2_L1")

            if opt.lambda_G2_lpips > 0.0:
                self.loss_names.append("G2_lpips")

            if opt.lambda_G2_GAN_feat > 0.0:
                self.loss_names.append("G2_GAN_feat")

        # specify the metrics you want to print out for logging
        # The training/test scripts will call <BaseModel.get_current_metrics>
        self.metric_names = []
        self.eval_metrics = [
            "I_SIFID",
            "I_LPIPS",
            "I_PSNR",
            "I_SSIM",
            "T_SIFID",
            "T_LPIPS",
            "T_AE",
            "T_MSE",
        ]
        # For train.py, compute metric for training samples and eval samples. For test.py, compute metric for testing samples.
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

        # define loss function
        if opt.smooth_GAN_label:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, target_real_label=0.8, target_fake_label=0.0).to(
                self.device
            )
        else:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL1_none = torch.nn.L1Loss(reduction="none")
        # perceptual loss. Use different features for train and validation.
        # Ref: https://github.com/richzhang/PerceptualSimilarity
        self.criterionLPIPS_vgg = lpips.LPIPS(net="vgg").to(self.device)
        self.criterionFeat = torch.nn.L1Loss()
        if self.isTrain:
            # use the same backbone for LPIPS to aviod CUDA OOM issue
            self.eval_LPIPS = self.criterionLPIPS_vgg
        else:
            self.eval_LPIPS = lpips.LPIPS(net="alex").to(self.device)

        # define networks (both generators and discriminators)
        print("Define G network")
        if hasattr(opt, "use_positional_encoding") and opt.use_positional_encoding:
            input_nc = opt.sketch_nc + 2 * opt.positional_encoding_dim
        else:
            input_nc = opt.sketch_nc
        self.netG = networks.define_G(
            input_nc,
            opt.image_nc + opt.touch_nc,
            opt.ngf,
            opt.netG,
            opt.normG,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.no_antialias,
            opt.no_antialias_up,
            self.gpu_ids,
            opt,
            num_layer_separate=opt.num_layer_separate,
        )

        self.use_cGAN_G2_S = hasattr(opt, "use_cGAN_G2_S") and opt.use_cGAN_G2_S
        self.use_cGAN_G2_I = hasattr(opt, "use_cGAN_G2_I") and opt.use_cGAN_G2_I
        if self.isTrain:
            print("Define D network")
            if hasattr(opt, "use_cGAN") and opt.use_cGAN:
                input_nc_I = opt.image_nc + opt.sketch_nc
            else:
                input_nc_I = opt.image_nc
            self.netD = networks.define_D(
                input_nc_I,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.normD,
                opt.init_type,
                opt.init_gain,
                opt.no_antialias,
                num_D=opt.num_D_D1,
                gpu_ids=self.gpu_ids,
                opt=opt,
            )
            if hasattr(opt, "use_vision_aided_loss") and opt.use_vision_aided_loss:
                # Add vision-aided discriminator to improve the quality of the generated RGB images
                self.netD3 = vision_aided_loss.Discriminator(
                    cv_type="clip", loss_type="multilevel_sigmoid_s", device=self.device
                ).to(self.device)
                self.netD3.cv_ensemble.requires_grad_(False)  # Freeze feature extractor

            if "D2" in self.model_names:
                print("Define D2 network")
                if hasattr(opt, "use_cGAN_G2") and opt.use_cGAN_G2:
                    input_nc = opt.touch_nc
                    if self.use_cGAN_G2_S:
                        input_nc += opt.sketch_nc
                    if self.use_cGAN_G2_I:
                        input_nc += opt.image_nc + 1  # for the mask
                    self.netD2 = networks.define_D(
                        input_nc,
                        opt.ndf,
                        opt.netD2,
                        opt.n_layers_D2,
                        opt.normD,
                        opt.init_type,
                        opt.init_gain,
                        opt.no_antialias,
                        num_D=opt.num_D_D2,
                        gpu_ids=self.gpu_ids,
                        opt=opt,
                    )
                else:
                    self.netD2 = networks.define_D(
                        opt.touch_nc,
                        opt.ndf,
                        opt.netD2,
                        opt.n_layers_D2,
                        opt.normD,
                        opt.init_type,
                        opt.init_gain,
                        opt.no_antialias,
                        self.gpu_ids,
                        opt,
                    )

            # define optimizer; schedulers will be automatically created by function <BaseModel.setup>.
            if "G" in self.model_names:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_G)
            if "D" in self.model_names:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D)
            if "D2" in self.model_names:
                self.optimizer_D2 = torch.optim.Adam(
                    self.netD2.parameters(), lr=opt.lr_G2, betas=(opt.beta1, opt.beta2)
                )
                self.optimizers.append(self.optimizer_D2)

    def optimize_parameters(self, epoch=0, timing=False):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        Args:
            epoch (int, optional): index of epoch. Defaults to 0.
            timing (bool, optional): option to print computation time for each step. Defaults to False.
        """
        # forward the model
        if timing:
            start_time = time.time()
        self.forward()
        if timing:
            print("forward takes time", time.time() - start_time)
            start_time = time.time()

        # compute additional output for loss computation
        (
            S_concat,
            fake_T_concat,
            real_I_concat,
            fake_I_concat,
            aug_real_I_concat,
            aug_fake_I_concat,
        ) = self.compute_additional_output(coords=self.train_T_coords)
        S_concat = S_concat.detach()
        if hasattr(self, "aug_fake_I"):
            fake_I_concat = aug_fake_I_concat.detach()
            real_I_concat = aug_real_I_concat.detach()
            fake_I_full = self.aug_fake_I.detach()
        else:
            fake_I_concat = fake_I_concat.detach()
            real_I_concat = real_I_concat.detach()
            fake_I_full = self.fake_I.detach()

        # concat with contact mask
        real_I_concat = torch.cat([real_I_concat, self.train_I_masks], dim=1)
        fake_I_concat = torch.cat([fake_I_concat, self.train_I_masks], dim=1)
        fake_I_full = torch.cat([fake_I_full, self.M.detach()], dim=1)

        if timing:
            print("compute_additional_output takes time", time.time() - start_time)
            start_time = time.time()

        # update D
        if "G" in self.model_names:
            self.set_requires_grad(self.netG, False)

        if "D" in self.model_names:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad(set_to_none=True)
            loss_D1 = self.compute_D1_loss(epoch)
            loss_D1.backward()
            self.optimizer_D.step()
        if "D2" in self.model_names:
            self.set_requires_grad(self.netD2, True)
            self.optimizer_D2.zero_grad(set_to_none=True)
            loss_D2, pred_real_T = self.compute_D2_loss(
                self.train_real_T_concat,
                fake_T_concat.detach(),
                S_concat,
                real_I_concat,
                fake_I_concat,
                fake_I_full,
                return_pred_real_T=True,
            )
            loss_D2.backward()
            self.optimizer_D2.step()
        else:
            pred_real_T = None

        if timing:
            print("update D takes time", time.time() - start_time)
            start_time = time.time()

        # update G
        if "D" in self.model_names:
            self.set_requires_grad(self.netD, False)
        if "D2" in self.model_names:
            self.set_requires_grad(self.netD2, False)
        if "G" in self.model_names:
            self.set_requires_grad(self.netG, True)
            self.optimizer_G.zero_grad(set_to_none=True)
            loss_G1 = self.compute_G1_loss(epoch)
            loss_G2 = self.compute_G2_loss(
                self.train_real_T_concat,
                fake_T_concat,
                S_concat,
                real_I_concat,
                fake_I_concat,
                pred_real_T=pred_real_T,
            )
            loss_G = loss_G1 + loss_G2
            loss_G.backward()
            self.optimizer_G.step()
        else:
            loss_G1 = 0
            loss_G2 = 0

        if timing:
            print("update G takes time", time.time() - start_time)

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

        self.data_phase = phase
        self.real_S = input["S"].to(self.device, non_blocking=True)
        self.name = input["name"]
        self.image_paths = input["S_paths"]  # for save_image in test.py
        # the dictionary stores the params for data augmentation
        self.augmentation_params = input["augmentation_params"]

        if self.opt.use_bg_mask:
            self.M = input["M"].to(self.device, non_blocking=True)
            # elementwise multiplication of the feature map and the mask
            self.real_S = self.real_S * self.M
            self.M_T = F.interpolate(self.M, size=self.M.shape[-1] * self.opt.T_resolution_multiplier).to(
                self.device, non_blocking=True
            )  # create mask for tactile images

        # If we have ground truth I and T
        if "I" in input.keys():
            self.real_I = input["I"].to(self.device, non_blocking=True)
            self.full_T_coords = input["full_T_coords"]
            if self.opt.use_bg_mask:
                self.real_I = self.real_I * self.M
                if verbose:
                    # real_S torch.Size([1, 1, 1536, 1536]) real_I torch.Size([1, 3, 1536, 1536]) M torch.Size([1, 1, 1536, 1536])  M_T torch.Size([1, 1, 3072, 3072])
                    print(
                        "check shape after multiplying with the mask, \n real_S {} real_I {} M {}  M_T {}".format(
                            self.real_S.shape,
                            self.real_I.shape,
                            self.M.shape,
                            self.M_T.shape,
                        )
                    )

        if hasattr(self.opt, "use_positional_encoding") and self.opt.use_positional_encoding:
            # create positional encoding for sketch
            if self.opt.positional_encoding_mode == "spe":
                spe = SPE(self.opt.positional_encoding_dim, 0, 1024)
                self.S_pe = spe(self.real_S)
            elif self.opt.positional_encoding_mode == "csg":
                csg = CSG()
                self.S_pe = csg(self.real_S)
            else:
                raise NotImplementedError(
                    "Positional encoding mode {} is not implemented".format(self.opt.positional_encoding_mode)
                )
            assert self.S_pe.is_cuda == self.real_S.is_cuda

        # Note: real_T_concat and T_mask will automatically convert to batch_size = 1. Manually reshape here
        # N: batch size, NT: number of tactile images for each visual sample, C: number of channels, H: height, W: width
        if len(input["T_images"]) > 0 and input["T_images"][0] is not None:
            self.train_T_coords = input["T_coords"].numpy()
            N, NT, C, H, W = input["T_images"].shape
            self.train_real_T_concat = (
                input["T_images"].reshape(-1, C, H, W).to(torch.float32).to(self.device, non_blocking=True)
            )
            self.train_I_masks = (
                input["I_masks"].reshape(-1, 1, H, W).to(torch.float32).to(self.device, non_blocking=True)
            )

            if len(input["val_T_images"]) > 0:
                self.val_T_coords = input["val_T_coords"].numpy()
                N, NT, C, H, W = input["val_T_images"].shape
                self.val_real_T_concat = (
                    input["val_T_images"].reshape(-1, C, H, W).to(torch.float32).to(self.device, non_blocking=True)
                )
                self.val_I_masks = (
                    input["val_I_masks"].reshape(-1, 1, H, W).to(torch.float32).to(self.device, non_blocking=True)
                )

            elif phase == "test":
                self.val_T_coords = self.train_T_coords
                self.val_real_T_concat = self.train_real_T_concat
                self.val_I_masks = self.train_I_masks

            # Multiply the real tactile image with the mask
            # train_real_T_concat torch.Size([62, 2, 32, 32]) train_I_masks torch.Size([62, 1, 32, 32])
            self.train_real_T_concat = self.train_real_T_concat * self.train_I_masks  # [62, 2, 32, 32]
            self.val_real_T_concat = self.val_real_T_concat * self.val_I_masks

        if timing:
            print("set_input takes time", time.time() - start_time)

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
        additional_visual_dict = self.compute_additional_visuals()
        # merge two dictionaries
        visual_ret.update(additional_visual_dict)
        return visual_ret

    def compute_additional_visuals(self, collage_patch_as_array=True, draw_bounding_box=True, colormap="gray"):
        """
        Compute additional visuals (patches of S, I, I', T(gx), T'(gx), T(gy), T'(gy) for both train & eval) that are not self attributes of the model. Return the images as an OrderedDict.
        Args:
            collage_patch_as_array (bool, optional): option to concatenate the (S, I, T) patches as one image in a numpy array. Defaults to True.
            draw_bounding_box (bool, optional): option to draw bounding box on I and T, to show the location of sampled patch. Defaults to True.
            colormap (str, optional): colormap for the tactile patches (single channel). Defaults to 'gray'.
        Returns:
            OrderedDict: a dictionary of additional visuals
        """
        additional_visual_names = OrderedDict()
        with torch.no_grad():
            if not self.test_edit_S:  # only visualize patches when the ground truth is available
                if self.data_phase == "test":
                    # For testing, we only have one set of coords, stored with prefix "val"
                    phase_list = ["test"]
                    coords_list = [self.val_T_coords]
                    real_T_concat_list = [self.val_real_T_concat]
                    I_mask_list = [self.val_I_masks]

                else:
                    # For training, we have two sets of coords, stored with prefix "train" and "val"
                    phase_list = ["train", "val"]
                    coords_list = [self.train_T_coords, self.val_T_coords]
                    real_T_concat_list = [
                        self.train_real_T_concat,
                        self.val_real_T_concat,
                    ]
                    I_mask_list = [self.train_I_masks, self.val_I_masks]

                # draw the bounding boxes on the generated visual and tactile images
                I = util.tensor2im(self.fake_I.detach()).astype(np.uint8).copy()
                # M = util.tensor2im(self.M_T.detach()).astype(np.uint8).copy()
                gx = util.tensor2im(self.fake_gx.detach()).astype(np.uint8).copy()
                if self.real_I is not None:  # real S, I, T are available
                    for phase, coords, real_T_concat, I_mask_concat in zip(
                        phase_list, coords_list, real_T_concat_list, I_mask_list
                    ):
                        # obtain the patches
                        real_gx_concat = torch.unsqueeze(real_T_concat[:, 0, :, :], 1)
                        real_gy_concat = torch.unsqueeze(real_T_concat[:, 1, :, :], 1)
                        real_normal_concat = compute_normal(real_T_concat[:, :2, :, :], scale_nz=self.opt.scale_nz)
                        S_concat = get_patch_in_input(self.real_S, coords)
                        fake_I_concat = get_patch_in_input(self.fake_I, coords)
                        (
                            real_I_concat,
                            offset_x,
                            offset_y,
                            cutout_size,
                        ) = get_patch_in_input(self.real_I, coords, return_offset=True)
                        fake_T_concat = get_patch_in_input(
                            self.fake_T,
                            coords,
                            scale_multiplier=self.opt.T_resolution_multiplier,
                        )
                        fake_gx_concat = torch.unsqueeze(fake_T_concat[:, 0, :, :], 1)
                        fake_gy_concat = torch.unsqueeze(fake_T_concat[:, 1, :, :], 1)
                        fake_normal_concat = compute_normal(fake_T_concat[:, :2, :, :], scale_nz=self.opt.scale_nz)

                        # compute eval metric
                        if phase == "val" or phase == "test":
                            metric_dict = compute_evaluation_metric(
                                model_names=self.model_names,
                                real_I=self.real_I.detach(),
                                fake_I=self.fake_I.detach(),
                                real_T_concat=self.val_real_T_concat.detach(),
                                fake_T_concat=fake_T_concat.detach(),
                                eval_metrics=self.eval_metrics,
                                eval_LPIPS=self.eval_LPIPS,
                                opt=self.opt,
                                device=self.device,
                                verbose=False,
                                timing=False,
                                prefix="",
                            )
                            # assign each key, value pair of metric_dict to self attributes
                            for key, value in metric_dict.items():
                                setattr(self, key, value)

                        elif phase == "train":
                            metric_dict = compute_evaluation_metric(
                                model_names=self.model_names,
                                real_I=self.real_I.detach(),
                                fake_I=self.fake_I.detach(),
                                real_T_concat=self.train_real_T_concat.detach(),
                                fake_T_concat=fake_T_concat.detach(),
                                eval_metrics=self.eval_metrics,
                                eval_LPIPS=self.eval_LPIPS,
                                opt=self.opt,
                                device=self.device,
                                verbose=False,
                                timing=False,
                                prefix="train_",
                            )
                            # assign each key, value pair of metric_dict to self attributes
                            for key, value in metric_dict.items():
                                setattr(self, key, value)

                        else:
                            raise AssertionError("Unknown phase", phase)

                        # draw bounding boxes on the visual and tactile images to show the location of sampled patches
                        H, W = real_gx_concat.shape[-2:]
                        if draw_bounding_box:
                            offset_x = offset_x.detach().cpu().numpy().astype(np.int32)
                            offset_y = offset_y.detach().cpu().numpy().astype(np.int32)
                            cutout_size = cutout_size.detach().cpu().numpy().astype(np.int32)
                            c = (255, 0, 0) if phase == "train" else (0, 255, 0)
                            # create copies of the images
                            I_phase = I.copy()
                            gx_phase = gx.copy()
                            # M_phase = M.copy()
                            for i in range(len(offset_x)):
                                # draw bounding box for 32x32 square patches
                                I_phase = cv2.rectangle(
                                    I_phase,
                                    (int(offset_x[i]), int(offset_y[i])),
                                    (
                                        int(offset_x[i] + cutout_size[i]),
                                        int(offset_y[i] + cutout_size[i]),
                                    ),
                                    (255, 0, 0),
                                    2,
                                )
                                gx_phase = cv2.rectangle(
                                    gx_phase,
                                    (
                                        int(offset_x[i] * self.opt.T_resolution_multiplier),
                                        int(offset_y[i] * self.opt.T_resolution_multiplier),
                                    ),
                                    (
                                        int((offset_x[i] + cutout_size[i]) * self.opt.T_resolution_multiplier),
                                        int((offset_y[i] + cutout_size[i]) * self.opt.T_resolution_multiplier),
                                    ),
                                    (255, 0, 0),
                                    2,
                                )  # current epoch
                            for i in range(len(self.full_T_coords)):
                                # draw bounding box for each full GelSight data patch, where 32x32 squares are sampled from based on contact mask
                                (
                                    new_ROI_x,
                                    new_ROI_y,
                                    new_ROI_h,
                                    new_ROI_w,
                                ) = self.full_T_coords[i]
                                new_ROI_x = new_ROI_x.cpu().numpy()[0]
                                new_ROI_y = new_ROI_y.cpu().numpy()[0]
                                new_ROI_h = new_ROI_h.cpu().numpy()[0]
                                new_ROI_w = new_ROI_w.cpu().numpy()[0]
                                gx_phase = cv2.rectangle(
                                    gx_phase,
                                    (
                                        int(new_ROI_x * self.opt.T_resolution_multiplier),
                                        int(new_ROI_y * self.opt.T_resolution_multiplier),
                                    ),
                                    (
                                        int((new_ROI_x + new_ROI_w) * self.opt.T_resolution_multiplier),
                                        int((new_ROI_y + new_ROI_h) * self.opt.T_resolution_multiplier),
                                    ),
                                    (0, 255, 0),
                                    2,
                                )
                                # M_phase = cv2.rectangle(M_phase, (int(new_ROI_x*self.opt.T_resolution_multiplier), int(new_ROI_y*self.opt.T_resolution_multiplier)), (int((new_ROI_x+new_ROI_w)*self.opt.T_resolution_multiplier), int((new_ROI_y+new_ROI_h)*self.opt.T_resolution_multiplier)), (0, 255, 0), 2)
                            additional_visual_names["%s_I_bb" % phase] = I_phase
                            additional_visual_names["%s_gx_bb" % phase] = gx_phase
                            # additional_visual_names['%s_M_bb' % phase] = M_phase

                        # save coordinates information
                        coords = coords.squeeze()  # [1, 60, 8] -> [60, 8]
                        # print("check coords", coords.shape, type(coords)) # [1, 60, 8]
                        additional_visual_names["%s_patch_coords" % phase] = coords
                        # elementwise round and convert to int for indexing
                        # .astype(int) # move dtype casting to FloatTensor
                        offset_x = np.round(
                            (coords[:, 0] + coords[:, -2] / coords[:, -3]) * self.opt.T_resolution_multiplier
                        )
                        offset_y = np.round(
                            (coords[:, 1] + coords[:, -1] / coords[:, -3]) * self.opt.T_resolution_multiplier
                        )
                        cutout_size = np.round(coords[:, -4] / coords[:, -3] * self.opt.T_resolution_multiplier)
                        additional_visual_names["%s_patch_coords" % phase] = np.vstack(
                            [offset_x + cutout_size // 2, offset_y + cutout_size // 2]
                        ).T  # find the center

                        # visualize the patches
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
                            coords_i,
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
                            coords,
                        ):
                            if i == self.num_patch_for_logging:
                                # num of patches for logging reaches num_patch_for_logging limit
                                break
                            if collage_patch_as_array:
                                # For each sample, combine S, I, T patches as one large numpy array image
                                # convert each tensor to im indivisually in order to not mess up the channel and range
                                # resize vision patches to the same size as touch patches

                                S_patch_arr = util.tensor2im(
                                    torch.squeeze(
                                        F.interpolate(torch.unsqueeze(S_patch, 0), (H, W)),
                                        dim=0,
                                    )
                                )
                                real_I_patch_tensor = torch.squeeze(
                                    F.interpolate(torch.unsqueeze(real_I_patch, dim=0), (H, W)),
                                    dim=0,
                                )
                                real_I_patch_arr = util.tensor2im(real_I_patch_tensor)
                                fake_I_patch = torch.squeeze(
                                    F.interpolate(torch.unsqueeze(fake_I_patch, dim=0), (H, W)),
                                    dim=0,
                                )
                                fake_I_patch_arr = util.tensor2im(fake_I_patch)
                                real_gx_patch_arr = util.tensor2im(
                                    real_gx_patch,
                                    convert_gray_imgs=True,
                                    colormap=colormap,
                                )
                                fake_gx_patch_arr = util.tensor2im(
                                    fake_gx_patch,
                                    convert_gray_imgs=True,
                                    colormap=colormap,
                                )
                                real_gy_patch_arr = util.tensor2im(
                                    real_gy_patch,
                                    convert_gray_imgs=True,
                                    colormap=colormap,
                                )
                                fake_gy_patch_arr = util.tensor2im(
                                    fake_gy_patch,
                                    convert_gray_imgs=True,
                                    colormap=colormap,
                                )
                                real_normal_patch_arr = util.tensor2im(real_normal_patch)
                                fake_normal_patch_arr = util.tensor2im(fake_normal_patch)

                                # for the collage, the top row are real images, the bottom row are generated images. For S, we only have real images, so we put a white placeholder
                                place_holder_S = I_mask_patch.cpu().numpy().astype(np.uint8) * 255
                                place_holder_S = np.stack(
                                    [place_holder_S, place_holder_S, place_holder_S],
                                    axis=0,
                                )
                                place_holder_S = np.transpose(np.squeeze(place_holder_S), (1, 2, 0))

                                collage_img = np.hstack(
                                    (
                                        np.vstack((S_patch_arr, place_holder_S)),
                                        np.vstack((real_I_patch_arr, fake_I_patch_arr)),
                                        np.vstack((real_gx_patch_arr, fake_gx_patch_arr)),
                                        np.vstack((real_gy_patch_arr, fake_gy_patch_arr)),
                                        np.vstack(
                                            (
                                                real_normal_patch_arr,
                                                fake_normal_patch_arr,
                                            )
                                        ),
                                    )
                                )
                                additional_visual_names["%s_patch_%d" % (phase, i)] = collage_img

                            # save individual patch for raw array output
                            additional_visual_names["%s_real_gx_patch_%d" % (phase, i)] = real_gx_patch
                            additional_visual_names["%s_fake_gx_patch_%d" % (phase, i)] = fake_gx_patch
                            additional_visual_names["%s_real_gy_patch_%d" % (phase, i)] = real_gy_patch
                            additional_visual_names["%s_fake_gy_patch_%d" % (phase, i)] = fake_gy_patch
                            additional_visual_names["%s_real_normal_patch_%d" % (phase, i)] = real_normal_patch
                            additional_visual_names["%s_fake_normal_patch_%d" % (phase, i)] = fake_normal_patch
                            if self.opt.save_S_patch:
                                additional_visual_names["%s_S_patch_%d" % (phase, i)] = S_patch
                                additional_visual_names["%s_real_I_patch_%d" % (phase, i)] = real_I_patch
                                additional_visual_names["%s_I_mask_patch_%d" % (phase, i)] = I_mask_patch
                                additional_visual_names["%s_fake_I_patch_%d" % (phase, i)] = fake_I_patch

                else:  # only S is available, draw S, generated I, T
                    for (
                        phase,
                        coords,
                    ) in zip(phase_list, coords_list):
                        S_concat = get_patch_in_input(self.real_S, coords)
                        fake_I_concat = get_patch_in_input(self.fake_I, coords)
                        fake_T_concat = get_patch_in_input(
                            self.fake_T,
                            coords,
                            scale_multiplier=self.opt.T_resolution_multiplier,
                        )
                        fake_gx_concat = torch.unsqueeze(fake_T_concat[:, 0, :, :], 1)
                        fake_gy_concat = torch.unsqueeze(fake_T_concat[:, 1, :, :], 1)
                        fake_normal_concat = compute_normal(fake_T_concat[:, :2, :, :], scale_nz=self.opt.scale_nz)
                        draw_bounding_box = False
                        H, W = fake_gx_concat.shape[-2:]
                        additional_visual_names["%s_I_bb" % phase] = I.copy()
                        additional_visual_names["%s_gx_bb" % phase] = gx.copy()

                        # save coordinates information
                        coords = coords.squeeze()  # [1, 60, 8] -> [60, 8]
                        # print("check coords", coords.shape, type(coords)) # [1, 60, 8]
                        additional_visual_names["%s_patch_coords" % phase] = coords
                        # elementwise round and convert to int for indexing
                        # .astype(int) # move dtype casting to FloatTensor
                        offset_x = np.round(
                            (coords[:, 0] + coords[:, -2] / coords[:, -3]) * self.opt.T_resolution_multiplier
                        )
                        offset_y = np.round(
                            (coords[:, 1] + coords[:, -1] / coords[:, -3]) * self.opt.T_resolution_multiplier
                        )
                        cutout_size = np.round(coords[:, -4] / coords[:, -3] * self.opt.T_resolution_multiplier)
                        additional_visual_names["%s_patch_coords" % phase] = np.vstack(
                            [offset_x + cutout_size // 2, offset_y + cutout_size // 2]
                        ).T  # find the center

                        for (
                            i,
                            S_patch,
                            fake_I_patch,
                            fake_gx_patch,
                            fake_gy_patch,
                            fake_normal_patch,
                            coords_i,
                        ) in zip(
                            range(len(S_concat)),
                            S_concat,
                            fake_I_concat,
                            fake_gx_concat,
                            fake_gy_concat,
                            fake_normal_concat,
                            coords,
                        ):
                            if i == self.num_patch_for_logging:
                                # num of patches for logging reaches num_patch_for_logging limit
                                break
                            if collage_patch_as_array:
                                # For each sample, combine S, I, T patches as one large numpy array image
                                # convert each tensor to im indivisually in order to not mess up the channel and range
                                # resize vision patches to the same size as touch patches
                                S_patch_arr = util.tensor2im(
                                    torch.squeeze(
                                        F.interpolate(torch.unsqueeze(S_patch, 0), (H, W)),
                                        dim=0,
                                    )
                                )
                                fake_I_patch_arr = util.tensor2im(
                                    torch.squeeze(
                                        F.interpolate(torch.unsqueeze(fake_I_patch, dim=0), (H, W)),
                                        dim=0,
                                    )
                                )
                                fake_gx_patch_arr = util.tensor2im(
                                    fake_gx_patch,
                                    convert_gray_imgs=True,
                                    colormap=colormap,
                                )
                                fake_gy_patch_arr = util.tensor2im(
                                    fake_gy_patch,
                                    convert_gray_imgs=True,
                                    colormap=colormap,
                                )
                                fake_normal_patch_arr = util.tensor2im(fake_normal_patch)

                                # for the collage, the top row are real images, the bottom row are generated images. For S, we only have real images, so we put a white placeholder
                                place_holder_S = I_mask_patch.cpu().numpy().astype(np.uint8) * 255
                                place_holder_S = np.stack(
                                    [place_holder_S, place_holder_S, place_holder_S],
                                    axis=0,
                                )
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

                            # else: # change else condtion to "always" case. save individual patch for raw array output
                            # pass each individual patch as a visual item
                            additional_visual_names["%s_fake_gx_patch_%d" % (phase, i)] = fake_gx_patch
                            additional_visual_names["%s_fake_gy_patch_%d" % (phase, i)] = fake_gy_patch

                # draw bounding boxes for more fake T samples during training to verify their positions
                if (
                    draw_bounding_box
                    and hasattr(self.opt, "use_more_fakeT")
                    and self.opt.use_more_fakeT
                    and hasattr(self, "fake_sample_offset_x")
                ):
                    offset_x = self.fake_sample_offset_x.detach().cpu().numpy().astype(np.int32)
                    offset_y = self.fake_sample_offset_y.detach().cpu().numpy().astype(np.int32)
                    cutout_size = self.fake_sample_cutout_size.detach().cpu().numpy().astype(np.int32)
                    # draw bounding boxes
                    I_phase = I.copy()
                    for i in range(len(offset_x)):
                        I_phase = cv2.rectangle(
                            I_phase,
                            (int(offset_x[i]), int(offset_y[i])),
                            (
                                int(offset_x[i] + cutout_size[i]),
                                int(offset_y[i] + cutout_size[i]),
                            ),
                            (255, 0, 0),
                            2,
                        )
                        fake_S_sample_arr = util.tensor2im(
                            torch.squeeze(
                                F.interpolate(torch.unsqueeze(self.fake_S_samples[i], 0), (H, W)),
                                dim=0,
                            )
                        )  # shape (32, 32, 3)
                        additional_visual_names["fake_S_sample_%d" % (i)] = fake_S_sample_arr
                    additional_visual_names["morefakeT_I_bb"] = I_phase

        return additional_visual_names

    def compute_additional_output(self, coords):
        """
        Compute additional output for the given coordinates. The full sketch S, visual image I, and tactile image T are self attributes. This function computes additional output (corresponding patches) given the full images and the coordinates.
        Args:
            coords (array): coordinate of patches.
        Returns:
            list of concated patches
        """
        # shape: self.fake_T torch.Size([1, 2, 1536, 1536]), self.fake_T_concat torch.Size([16, 2, 32, 32])
        fake_T_concat = get_patch_in_input(self.fake_T, coords, scale_multiplier=self.opt.T_resolution_multiplier)
        real_I_concat = get_patch_in_input(self.real_I, coords)
        fake_I_concat = get_patch_in_input(self.fake_I, coords)
        if hasattr(self.opt, "use_diffaug") and self.opt.use_diffaug:
            aug_real_I_concat = get_patch_in_input(self.aug_real_I, coords)
            aug_fake_I_concat = get_patch_in_input(self.aug_fake_I, coords)
        S_concat = get_patch_in_input(self.real_S, coords)
        return (
            S_concat,
            fake_T_concat,
            real_I_concat,
            fake_I_concat,
            aug_real_I_concat,
            aug_fake_I_concat,
        )

    def forward(self, timing=False):
        """
        Run forward pass; called by both functions <optimize_parameters> and <test>.

        Args:
            timing (bool, optional): option to print out timing. Defaults to False.
        """
        if timing:
            print("In forward pass")
            start_time = time.time()

        if hasattr(self.opt, "use_positional_encoding") and self.opt.use_positional_encoding:
            output = self.netG(torch.cat((self.real_S, self.S_pe), 1))
        else:
            output = self.netG(self.real_S)
        # separate the output into fake_I and fake_T
        self.fake_I = output[:, 0:3, :, :]
        self.fake_T = output[:, -2:, :, :]

        if timing:
            print("forward model G takes time ", time.time() - start_time)
            start_time = time.time()

        # apply mask to the generated data
        if self.opt.use_bg_mask:
            self.fake_I = self.fake_I * self.M
            self.fake_T = self.fake_T * self.M
        self.fake_gx = torch.unsqueeze(self.fake_T[:, 0, :, :], 1)
        self.fake_gy = torch.unsqueeze(self.fake_T[:, 1, :, :], 1)
        self.fake_N = compute_normal(
            self.fake_T[:, :2, :, :].detach(), scale_nz=self.opt.scale_nz
        )  # set scale_nz = 1 for visualization

        if not self.test_edit_S:
            # apply DiffAug and mask
            if hasattr(self.opt, "use_diffaug") and self.opt.use_diffaug:
                # apply DiffAugment to real_I and fake_I
                self.aug_real_I = (
                    DiffAugment(self.real_I, policy=self.opt.diffaugment) if hasattr(self, "real_I") else None
                )
                self.aug_fake_I = DiffAugment(self.fake_I, policy=self.opt.diffaugment)
            else:
                # no augmentation
                self.aug_real_I = self.real_I if hasattr(self, "real_I") else None
                self.aug_fake_I = self.fake_I
            if self.opt.use_bg_mask:
                self.aug_real_I = self.aug_real_I * self.M
                self.aug_fake_I = self.aug_fake_I * self.M

        if timing:
            print("process output in forward pass takes time ", time.time() - start_time)
            start_time = time.time()

    def compute_D1_loss(self, epoch=0):
        """
        Compute loss for the discriminator for visual output.
        Args:
            epoch (int, optional): current epoch. Defaults to 0.

        Returns:
            loss (float): loss for the discriminator for visual output
        """
        if self.opt.lambda_G1_GAN > 0.0:
            real_S = self.real_S.detach()
            # Fake
            fake_I = self.fake_I.detach()  # detach to avoid backprop to G
            if hasattr(self.opt, "use_cGAN") and self.opt.use_cGAN:
                fake_I = torch.cat((real_S, fake_I), 1)
            pred_fake_I = self.netD(fake_I)
            loss_D_fake_I = self.criterionGAN(pred_fake_I, target_is_real=False).mean() * self.opt.lambda_G1_GAN

            # Add pred_fake_I to model attributes for visualization
            if self.opt.netD == "multiscale":
                self.pred_fake_I = pred_fake_I[-1][-1].detach()
            else:
                self.pred_fake_I = pred_fake_I.detach()

            # Real
            real_I = self.real_I.detach()
            if hasattr(self.opt, "use_cGAN") and self.opt.use_cGAN:
                real_I = torch.cat((real_S, real_I), 1)
            pred_real_I = self.netD(real_I)
            loss_D_real_I = self.criterionGAN(pred_real_I, target_is_real=True).mean() * self.opt.lambda_G1_GAN

            if self.opt.gan_mode == "wgangp":
                gradient_penalty, gradients = networks.cal_gradient_penalty(self.netD, real_I, fake_I, self.device)
            else:
                gradient_penalty = 0.0

        else:
            loss_D_fake_I = 0.0
            loss_D_real_I = 0.0

        # for backprop
        loss_D1 = (loss_D_fake_I + loss_D_real_I + gradient_penalty) * 0.5
        # for logging
        self.loss_D_fake_I = loss_D_fake_I.item() if torch.is_tensor(loss_D_fake_I) else loss_D_fake_I
        self.loss_D_real_I = loss_D_real_I.item() if torch.is_tensor(loss_D_real_I) else loss_D_real_I
        self.loss_D_I_grad_penalty = gradient_penalty.item() if torch.is_tensor(gradient_penalty) else gradient_penalty

        if hasattr(self, "netD3") and epoch >= self.opt.vision_aided_warmup_epoch:
            # add vision-aided discriminator
            loss_D3_real_I = self.netD3(self.real_I.detach(), for_real=True) * self.opt.lambda_G1_GAN
            loss_D3_fake_I = self.netD3(self.fake_I.detach(), for_real=False) * self.opt.lambda_G1_GAN
            loss_D3 = (loss_D3_real_I + loss_D3_fake_I) * 0.5
            loss_D3 = loss_D3.mean()
        else:
            loss_D3_real_I = 0.0
            loss_D3_fake_I = 0.0
            loss_D3 = 0.0
        self.loss_D3_real_I = loss_D3_real_I.item() if torch.is_tensor(loss_D3_real_I) else loss_D3_real_I
        self.loss_D3_fake_I = loss_D3_fake_I.item() if torch.is_tensor(loss_D3_fake_I) else loss_D3_fake_I

        loss_D1 += loss_D3
        return loss_D1

    def compute_D2_loss(
        self,
        real_T_concat,
        fake_T_concat,
        S_concat,
        real_I_concat,
        fake_I_concat,
        fake_I_full,
        return_pred_real_T=False,
    ):
        """
        Compute loss for the discriminator for tactile output.

        Args:

            real_T_concat (tensor): real tactile patch
            fake_T_concat (tensor): fake tactile patch
            S_concat (tensor): sketch patch
            real_I_concat (tensor): real visual patch
            fake_I_concat (tensor): fake visual patch
            fake_I_full (tensor): fake visual image with full resolution
            return_pred_real_T (bool, optional): whether to return pred_real_T. Defaults to False

        Returns:

            loss (float): loss for the discriminator for tactile output
            pred_real_T (tensor): prediction for real tactile image by the discrinimator
        """
        if self.opt.lambda_G2_GAN > 0.0 and len(real_T_concat) > 0:
            # Fake
            nh, nw = fake_T_concat.shape[-2:]
            S_concat = F.interpolate(
                S_concat,
                size=(nh, nw),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            fake_I_concat = F.interpolate(
                fake_I_concat,
                size=(nh, nw),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).to(self.device)
            real_I_concat = F.interpolate(
                real_I_concat,
                size=(nh, nw),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).to(self.device)

            fake_T_full = self.fake_T.detach()
            fake_I_full = F.interpolate(
                fake_I_full,
                size=(fake_T_full.shape[-2], fake_T_full.shape[-1]),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            S_full = F.interpolate(
                self.real_S.detach(),
                size=(fake_T_full.shape[-2], fake_T_full.shape[-1]),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            assert (
                fake_T_concat.shape[0] == S_concat.shape[0] and fake_T_concat.shape[0] == fake_I_concat.shape[0]
            ), "fake_T_concat, S_concat, fake_I_concat should have same number of patches"

            if hasattr(self.opt, "use_cGAN_G2") and self.opt.use_cGAN_G2:
                if self.use_cGAN_G2_S:
                    fake_T_concat = torch.cat((fake_T_concat, S_concat), dim=1)
                    fake_T_full = torch.cat((fake_T_full, S_full), dim=1)
                if self.use_cGAN_G2_I:
                    fake_T_concat = torch.cat((fake_T_concat, fake_I_concat), dim=1)
                    fake_T_full = torch.cat((fake_T_full, fake_I_full), dim=1)

            # fake_T_concat.shape [128, 7, 32, 32], pred_fake_T.shape [128, 1, 2, 2]
            pred_fake_T_concat = self.netD2(fake_T_concat)
            loss_D_fake_T_concat = (
                self.criterionGAN(pred_fake_T_concat, target_is_real=False).mean() * self.opt.lambda_G2_GAN
            )

            self.pred_fake_T_full = self.netD2(fake_T_full)

            if self.opt.netD2 == "multiscale":
                # [1,1,190,190]
                self.pred_fake_T_full = self.pred_fake_T_full[-1][-1].detach()
            else:
                self.pred_fake_T_full = self.pred_fake_T_full.detach()

            if hasattr(self.opt, "use_more_fakeT") and self.opt.use_more_fakeT:
                (
                    fake_T_samples,
                    self.fake_sample_offset_x,
                    self.fake_sample_offset_y,
                    self.fake_sample_cutout_size,
                ) = get_patch_in_input(
                    self.fake_T.detach(),
                    coords=None,
                    sample_size=self.opt.add_fake_T_sample_size,
                    scale_multiplier=self.opt.T_resolution_multiplier,
                    return_offset=True,
                    center_h=self.opt.center_h,
                    center_w=self.opt.center_w,
                    augmentation_params=self.augmentation_params,
                    M=self.M,
                    device=self.device,
                )
                if hasattr(self.opt, "use_cGAN_G2") and self.opt.use_cGAN_G2:
                    if self.use_cGAN_G2_S:
                        fake_S_samples = get_patch_in_input(
                            self.real_S.detach(),
                            coords=None,
                            sample_size=self.opt.add_fake_T_sample_size,
                            offset_x=self.fake_sample_offset_x,
                            offset_y=self.fake_sample_offset_y,
                            cutout_size=self.fake_sample_cutout_size,
                        )
                        fake_S_samples = F.interpolate(
                            fake_S_samples,
                            size=(nh, nw),
                            mode="bicubic",
                            align_corners=False,
                            antialias=True,
                        )
                        self.fake_S_samples = fake_S_samples  # for visualization
                        fake_T_samples = torch.cat((fake_T_samples, fake_S_samples), dim=1)
                    if self.use_cGAN_G2_I:
                        I = self.fake_I.detach()
                        fake_I_samples = get_patch_in_input(
                            I,
                            coords=None,
                            sample_size=self.opt.add_fake_T_sample_size,
                            offset_x=self.fake_sample_offset_x,
                            offset_y=self.fake_sample_offset_y,
                            cutout_size=self.fake_sample_cutout_size,
                        )

                        fake_I_samples = F.interpolate(
                            fake_I_samples,
                            size=(nh, nw),
                            mode="bicubic",
                            align_corners=False,
                            antialias=True,
                        ).to(self.device)
                        # concat with contact mask
                        fake_I_mask_samples = torch.ones(
                            fake_I_samples.shape[0],
                            1,
                            fake_I_samples.shape[2],
                            fake_I_samples.shape[3],
                        ).to(self.device)
                        fake_I_samples = torch.cat((fake_I_samples, fake_I_mask_samples), dim=1)
                        fake_T_samples = torch.cat((fake_T_samples, fake_I_samples), dim=1)
                pred_more_fake_T = self.netD2(fake_T_samples)
                loss_D_more_fake_T = self.criterionGAN(pred_more_fake_T, False)
                loss_D_more_fake_T = loss_D_more_fake_T.mean() * self.opt.lambda_G2_GAN

            else:
                loss_D_more_fake_T = 0.0

            # Real
            assert (
                real_T_concat.shape[0] == S_concat.shape[0] and real_T_concat.shape[0] == real_I_concat.shape[0]
            ), "real_T_concat, S_concat, real_I_concat should have same number of patches"
            if hasattr(self.opt, "use_cGAN_G2") and self.opt.use_cGAN_G2:
                if self.use_cGAN_G2_S:
                    real_T_concat = torch.cat((real_T_concat, S_concat), dim=1)
                if self.use_cGAN_G2_I:
                    real_T_concat = torch.cat((real_T_concat, real_I_concat), dim=1)

            pred_real_T_concat = self.netD2(real_T_concat)
            loss_D_real_T_concat = self.criterionGAN(pred_real_T_concat, True).mean() * self.opt.lambda_G2_GAN

            if self.opt.gan_mode == "wgangp":
                gradient_penalty, gradients = networks.cal_gradient_penalty(
                    self.netD2, real_T_concat, fake_T_concat, self.device
                )
            else:
                gradient_penalty = 0.0

        else:
            loss_D_fake_T_concat = 0.0
            loss_D_more_fake_T = 0.0
            loss_D_real_T_concat = 0.0

        # for backprop
        loss_D2 = (loss_D_fake_T_concat + loss_D_more_fake_T + loss_D_real_T_concat + gradient_penalty) * 0.5
        # for logging
        self.loss_D_fake_T_concat = (
            loss_D_fake_T_concat.item() if torch.is_tensor(loss_D_fake_T_concat) else loss_D_fake_T_concat
        )
        if hasattr(self.opt, "use_more_fakeT") and self.opt.use_more_fakeT:
            self.loss_D_more_fake_T = (
                loss_D_more_fake_T.item() if torch.is_tensor(loss_D_more_fake_T) else loss_D_more_fake_T
            )
        self.loss_D_real_T_concat = (
            loss_D_real_T_concat.item() if torch.is_tensor(loss_D_real_T_concat) else loss_D_real_T_concat
        )
        self.loss_D_T_grad_penalty = gradient_penalty.item() if torch.is_tensor(gradient_penalty) else gradient_penalty

        if return_pred_real_T:
            return loss_D2, pred_real_T_concat
        else:
            return loss_D2

    def _compute_touch_lpips_loss(
        self,
        criterionLPIPS,
        fake_T_concat,
        real_T_concat,
        lambda_lpips=1,
        separate_xy=False,
    ):
        """
        Compute lpips loss for 2 channel tactile image. Since the lpips loss is designed for RGB images, we compute the loss for each channel (gx, gy) separately by tiling the channel to 3 channels, and then summing the loss together.
        Args:
            criterionLPIPS (nn.Module): LPIPS loss module
            fake_T_concat (torch.Tensor): [NxNT, 2, H, W]
            real_T_concat (torch.Tensor): [NxNT, 2, H, W]
            lambda_lpips (float, optional): weights for lpips loss. Defaults to 1.
            separate_xy (bool, optional): option to seprate lpips into two losses, lpips_gx and lpips_gy; otherwise, we sum them up as one loss lpips_T. Defaults to False.

        Returns:
            torch.Tensor: lpips loss
        """
        loss_lpips_gx = criterionLPIPS(
            torch.unsqueeze(fake_T_concat[:, 0, :, :], 1),
            torch.unsqueeze(real_T_concat[:, 0, :, :], 1),
        )
        loss_lpips_gy = criterionLPIPS(
            torch.unsqueeze(fake_T_concat[:, 1, :, :], 1),
            torch.unsqueeze(real_T_concat[:, 1, :, :], 1),
        )
        # reshape from [NxNT, C, H, W] to [N, NT, C, H, W], where C=1, H=1, W=1
        N, C, H, W = loss_lpips_gx.shape
        loss_lpips_gx = loss_lpips_gx.view(-1, self.opt.batch_size_G2, C, H, W)
        loss_lpips_gx = loss_lpips_gx.sum(dim=1).mean()
        loss_lpips_gy = loss_lpips_gy.view(-1, self.opt.batch_size_G2, C, H, W)
        loss_lpips_gy = loss_lpips_gy.sum(dim=1).mean()

        if separate_xy:
            return (loss_lpips_gx * lambda_lpips), (loss_lpips_gy * lambda_lpips)
        else:
            loss_lpips = lambda_lpips * (loss_lpips_gx + loss_lpips_gy)
            return loss_lpips

    def compute_G1_loss(self, epoch=0, pred_real_I=None):
        """
        Compute loss for the generator for visual image (GAN, L1, perceptual, etc.)
        Args:
            epoch (int, optional): current epoch. Defaults to 0.
            pred_real_I (torch.Tensor, optional): prediction of the discriminator for real visual image. Used for GAN feature matching loss. Defaults to None.
        Returns:
            torch.Tensor: loss for the generator for visual image
        """
        # GAN loss
        if self.opt.lambda_G1_GAN > 0.0:
            if hasattr(self.opt, "use_cGAN") and self.opt.use_cGAN:
                # use conditional GAN loss. feed both input sketch and the output fake visual image to the discriminator
                fake_I = torch.cat((self.real_S, self.fake_I), 1)
            else:
                # use vanilla GAN loss
                fake_I = self.fake_I
            pred_fake_I = self.netD(fake_I)
            loss_G_GAN = self.criterionGAN(pred_fake_I, True).mean() * self.opt.lambda_G1_GAN
        else:
            loss_G_GAN = 0.0
        self.loss_G_GAN = loss_G_GAN.item() if torch.is_tensor(loss_G_GAN) else loss_G_GAN

        # GAN feature matching loss for multiscale discriminator
        loss_G1_GAN_feat = 0
        if hasattr(self.opt, "lambda_G1_GAN_feat") and self.opt.lambda_G1_GAN_feat > 0.0 and self.netD == "multiscale":
            assert self.opt.lambda_G2_GAN > 0.0, "G2 is required to match GAN feature"
            assert pred_real_I is not None, "pred_real_I is required to match GAN feature"
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D_D1
            for i in range(self.opt.num_D_D1):
                for j in range(len(pred_real_I[i]) - 1):
                    loss_G1_GAN_feat += (
                        D_weights
                        * feat_weights
                        * self.criterionFeat(pred_fake_I[i][j], pred_real_I[i][j].detach())
                        * self.opt.lambda_G1_GAN_feat
                    )
        self.loss_G1_GAN_feat = loss_G1_GAN_feat.item() if torch.is_tensor(loss_G1_GAN_feat) else loss_G1_GAN_feat

        # L1 loss
        if self.opt.lambda_G1_L1 > 0.0:
            loss_G_L1 = self.criterionL1(self.fake_I, self.real_I) * self.opt.lambda_G1_L1
        else:
            loss_G_L1 = 0.0
        self.loss_G_L1 = loss_G_L1.item() if torch.is_tensor(loss_G_L1) else loss_G_L1

        # perceptual loss (lpips)
        if self.opt.lambda_G1_lpips > 0.0:
            N, C, H, W = self.fake_I.shape
            if self.isTrain:
                loss_G_lpips = self.criterionLPIPS_vgg(self.fake_I, self.real_I).mean() * self.opt.lambda_G1_lpips
            else:
                loss_G_lpips = self.eval_LPIPS(self.fake_I, self.real_I).mean() * self.opt.lambda_G1_lpips
        else:
            loss_G_lpips = 0.0
        self.loss_G_lpips = loss_G_lpips.item() if torch.is_tensor(loss_G_lpips) else loss_G_lpips

        # vision-aided loss
        if hasattr(self, "netD3") and epoch >= self.opt.vision_aided_warmup_epoch:
            loss_G_D3 = self.netD3(self.fake_I, for_G=True) * self.opt.lambda_G1_GAN
        else:
            loss_G_D3 = 0.0
        self.loss_G_D3 = loss_G_D3.item() if torch.is_tensor(loss_G_D3) else loss_G_D3

        loss_G1 = loss_G_GAN + loss_G_L1 + loss_G_lpips + loss_G_D3 + loss_G1_GAN_feat
        return loss_G1

    def compute_G2_loss(
        self,
        real_T_concat,
        fake_T_concat,
        S_concat,
        real_I_concat,
        fake_I_concat,
        pred_real_T=None,
    ):
        """
        Compute loss for the generator for tactile image (GAN, L1, perceptual, etc.)
        Args:
            real_T_concat (torch.Tensor): real tactile patch
            fake_T_concat (torch.Tensor): fake tactile patch
            S_concat (torch.Tensor): input sketch patch
            real_I_concat (torch.Tensor): real visual patch
            fake_I_concat (torch.Tensor): fake visual patch
            pred_real_T (torch.Tensor, optional): prediction of the discriminator for real tactile image. Used for GAN feature matching loss. Defaults to None.
        Returns:
            torch.Tensor: loss for the generator for tactile image
        """
        if len(real_T_concat) > 0:
            # save a copy of fake_T_concat for L1 and lpips loss
            fake_T_concat_org = fake_T_concat.clone().detach()
            if self.opt.lambda_G2_GAN > 0.0:
                nh, nw = fake_T_concat.shape[-2:]
                S_concat = F.interpolate(
                    S_concat,
                    size=(nh, nw),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                real_I_concat = F.interpolate(
                    real_I_concat,
                    size=(nh, nw),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                fake_I_concat = F.interpolate(
                    fake_I_concat,
                    size=(nh, nw),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                if hasattr(self.opt, "use_cGAN_G2") and self.opt.use_cGAN_G2:
                    if self.use_cGAN_G2_S:
                        fake_T_concat_org = torch.cat((fake_T_concat_org, S_concat), dim=1)
                    if self.use_cGAN_G2_I:
                        fake_T_concat_org = torch.cat((fake_T_concat_org, fake_I_concat), dim=1)

                pred_fake_T = self.netD2(fake_T_concat_org)
                loss_G2_GAN = self.criterionGAN(pred_fake_T, True) * self.opt.lambda_G2_GAN  # shape [NxNT]
                if len(loss_G2_GAN) > 1:  # for multiscale discriminator
                    loss_G2_GAN = loss_G2_GAN.view(-1, self.opt.batch_size_G2).mean(dim=0).sum()
            else:
                loss_G2_GAN = 0.0
            self.loss_G2_GAN = loss_G2_GAN.item() if torch.is_tensor(loss_G2_GAN) else loss_G2_GAN

            # GAN feature matching loss for multiscale discriminator
            loss_G2_GAN_feat = 0.0
            if (
                hasattr(self.opt, "lambda_G2_GAN_feat")
                and self.opt.lambda_G2_GAN_feat > 0.0
                and self.netD2 == "multiscale"
            ):
                assert self.opt.lambda_G2_GAN > 0.0, "G2 is required to match GAN feature"
                assert pred_real_T is not None, "pred_fake_T is required to match GAN feature"
                feat_weights = 4.0 / (self.opt.n_layers_D + 1)
                D_weights = 1.0 / self.opt.num_D_D2
                for i in range(self.opt.num_D_D2):
                    for j in range(len(pred_fake_T[i]) - 1):
                        loss_G2_GAN_feat += (
                            D_weights
                            * feat_weights
                            * self.criterionFeat(pred_fake_T[i][j], pred_real_T[i][j].detach())
                            * self.opt.lambda_G2_GAN_feat
                        )
            self.loss_G2_GAN_feat = loss_G2_GAN_feat.item() if torch.is_tensor(loss_G2_GAN_feat) else loss_G2_GAN_feat

            # L1 loss
            if self.opt.lambda_G2_L1 > 0.0:
                loss_G2_L1 = self.criterionL1_none(fake_T_concat, real_T_concat) * self.opt.lambda_G2_L1
                N, C, H, W = loss_G2_L1.shape
                loss_G2_L1 = loss_G2_L1.view(-1, self.opt.batch_size_G2, C, H, W).sum(dim=1).mean()
            else:
                loss_G2_L1 = 0.0
            self.loss_G2_L1 = loss_G2_L1.item() if torch.is_tensor(loss_G2_L1) else loss_G2_L1

            # perceptual loss (lpips)
            if self.opt.lambda_G2_lpips > 0.0:
                # Note: LPIPS only works for 3-channel images, so we convert (gx, gy) to 3-channel
                if self.isTrain:
                    loss_G2_lpips = self._compute_touch_lpips_loss(
                        self.criterionLPIPS_vgg,
                        fake_T_concat,
                        real_T_concat,
                        self.opt.lambda_G2_lpips,
                    )
                else:
                    loss_G2_lpips = self._compute_touch_lpips_loss(
                        self.eval_LPIPS,
                        fake_T_concat,
                        real_T_concat,
                        self.opt.lambda_G2_lpips,
                    )
            else:
                loss_G2_lpips = 0.0
            self.loss_G2_lpips = loss_G2_lpips.item() if torch.is_tensor(loss_G2_lpips) else loss_G2_lpips

        loss_G2 = loss_G2_GAN + loss_G2_L1 + loss_G2_lpips + loss_G2_GAN_feat

        return loss_G2
