import os
import sys
import time
from collections import OrderedDict

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import util.util as util
from models.model_utils import (
    compute_additional_visuals,
    compute_evaluation_metric,
    compute_normal,
)
from util.image_pool import ImagePool

from . import networks
from .base_model import BaseModel


class Pix2PixHDModel(BaseModel):
    """
    This class implements the pix2pixHD model, for learning a mapping from input images to output images given paired data.
    The script is adapted from https://github.com/NVIDIA/pix2pixHD/blob/master/models/pix2pixHD_model.py
    pix2pixHD paper: https://arxiv.org/pdf/1711.11585.pdf

    The modifiications we have made are similar to what we have done in pix2pix_model.py.
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

        #### Pix2pixHD options ####
        # input/output sizes
        parser.add_argument(
            "--label_nc",
            type=int,
            default=0,
            help="# of input label channels, if 0, use the lael map; otherwise, create one-hot vector for the label map. To compare with skit, we set it to 0",
        )
        parser.add_argument(
            "--data_type", default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit"
        )

        # for instance-wise features
        parser.add_argument(
            "--no_instance", type=util.str2bool, default=True, help="if specified, do *not* add instance map as input"
        )
        parser.add_argument(
            "--instance_feat", action="store_true", help="if specified, add encoded instance features as input"
        )
        parser.add_argument(
            "--label_feat", action="store_true", help="if specified, add encoded label features as input"
        )
        parser.add_argument("--feat_num", type=int, default=3, help="vector length for encoded features")
        parser.add_argument("--load_features", action="store_true", help="if specified, load precomputed feature maps")
        parser.add_argument("--n_downsample_E", type=int, default=4, help="# of downsampling layers in encoder")
        parser.add_argument("--nef", type=int, default=16, help="# of encoder filters in the first conv layer")
        parser.add_argument("--n_clusters", type=int, default=10, help="number of clusters for features")

        # for generator
        parser.add_argument("--n_downsample_global", type=int, default=4, help="number of downsampling layers in netG")
        parser.add_argument(
            "--n_blocks_global", type=int, default=9, help="number of residual blocks in the global generator network"
        )
        parser.add_argument(
            "--n_blocks_local", type=int, default=3, help="number of residual blocks in the local enhancer network"
        )
        parser.add_argument("--n_local_enhancers", type=int, default=1, help="number of local enhancers to use")
        parser.add_argument(
            "--niter_fix_global",
            type=int,
            default=0,
            help="number of epochs that we only train the outmost local enhancer",
        )

        # for discriminator
        parser.add_argument(
            "--getIntermFeat_D",
            type=util.str2bool,
            default=True,
            help="option to get intermediate features from multiscale discriminator",
        )
        parser.add_argument(
            "--num_D_D1", type=int, default=2, help="num of layer output to use in the multiple discriminators"
        )
        parser.add_argument(
            "--num_D_D2", type=int, default=2, help="num of layer output to use in the multiple discriminators"
        )
        parser.add_argument(
            "--no_gan_loss", type=util.str2bool, default=False, help="if specified, do *not* use GAN loss"
        )
        parser.add_argument(
            "--no_ganFeat_loss",
            type=util.str2bool,
            default=False,
            help="if specified, do *not* use discriminator feature matching loss",
        )
        parser.add_argument(
            "--no_vgg_loss",
            type=util.str2bool,
            default=False,
            help="if specified, do *not* use VGG feature matching loss",
        )
        parser.add_argument("--lambda_feat", type=float, default=10.0, help="weight for feature matching loss")
        parser.add_argument("--lambda_vgg", type=float, default=10.0, help="weight for vgg loss")

        # for training
        parser.add_argument(
            "--niter_decay", type=int, default=100, help="# of iter to linearly decay learning rate to zero"
        )

        # for dataset
        parser.add_argument(
            "--separate_val_set",
            type=util.str2bool,
            default=False,
            help="if specified, preprocess a validation patchwise dataset",
        )

        # experiment specific
        parser.add_argument(
            "--fp16", action="store_true", default=False, help="train with Automatic Mixed Precision (AMP)"
        )

        # Set the default values to match the pix2pixHD paper
        parser.set_defaults(
            norm="batch",
            netG="global",
            netD="multiscale",
            ngf=64,
            dataset_mode="aligned",
            dataset="patchskit",
            crop_size=1536,
            normG="instance",
            normD="instance",
            pool_size=0,
            n_epochs=50,
            n_epcohs_decay=150,  # pix2pixHD diverges fast, so reduce the number of epochs for training
            gan_mode="lsgan",  # the default setting for pix2pixHD is lsgan
        )
        verbose_freq = 320  # set it to be divisble by 32 and smaller than len(data_set)
        if is_train:
            parser.set_defaults(
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
                return_patch=False, 
                batch_size=1, 
                save_S_patch=True, 
                save_raw_arr_vis=False, 
                sample_bbox_per_patch=1,
                data_len=1
            )

        return parser

    def name(self):
        return "Pix2PixHDModel"

    def __init__(self, opt):
        """
        Initialize the pix2pixHD class.

        Args:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        opt.no_lsgan = False if opt.gan_mode == "lsgan" else True

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
            if self.use_features:
                self.model_names += ["E"]
        else:  # during test time, only load G
            self.model_names = ["G"]

        # specify the images you want to log for visualization.
        # The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ["real_S", "M", "fake_I", "fake_gx", "fake_gy", "fake_N"]
        if not self.test_edit_S:
            self.visual_names.insert(2, "real_I")

        # specify the losses you want to print out for logging.
        # The training/test scripts will call <BaseModel.get_current_losses>
        if self.isTrain:
            self.loss_names = []
            if not opt.no_gan_loss:
                self.loss_names += ["G_GAN_I", "G_GAN_T", "G_GAN", "D_real", "D_fake", "D2_real", "D2_fake"]
            if not opt.no_ganFeat_loss:
                self.loss_names += ["G_GAN_Feat", "G_GAN_Feat_I", "G_GAN_Feat_T"]
            if not opt.no_vgg_loss:
                self.loss_names += ["G_VGG", "G_VGG_I", "G_VGG_T"]

        # specify the metrics you want to save
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
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        self.criterionFeat = torch.nn.L1Loss()
        if not opt.no_vgg_loss:
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)
        self.criterionLPIPS_vgg = lpips.LPIPS(net="vgg").to(self.device)
        if self.isTrain:
            self.eval_LPIPS = self.criterionLPIPS_vgg  # use the same backbone for LPIPS to aviod CUDA OOM issue
        else:
            self.eval_LPIPS = lpips.LPIPS(net="alex").to(self.device)

        # define networks (both generator and discriminator)
        input_nc = opt.sketch_nc
        output_nc = opt.image_nc + opt.touch_nc

        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        print("Define G network")
        self.netG = networks.define_G(
            netG_input_nc, output_nc, opt.ngf, opt.netG, opt.norm, gpu_ids=self.gpu_ids, opt=opt
        )

        # Discriminator network
        if self.isTrain:
            print("Define D networks")
            netD_input_nc = opt.image_nc + opt.sketch_nc
            netD2_input_nc = opt.touch_nc + opt.sketch_nc
            if not opt.no_instance:
                netD_input_nc += 1
                netD2_input_nc += 1
            self.netD = networks.define_D(
                netD_input_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                num_D=opt.num_D_D1,
                gpu_ids=self.gpu_ids,
                opt=opt,
            )
            self.netD2 = networks.define_D(
                netD2_input_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                num_D=opt.num_D_D2,
                gpu_ids=self.gpu_ids,
                opt=opt,
            )

        # Encoder network
        if self.gen_features:
            self.netE = networks.define_G(
                output_nc, opt.feat_num, opt.nef, "encoder", opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids
            )
            self.model_names.append("E")
        if self.opt.verbose:
            print("---------- Networks initialized -------------")

        # define optimizers
        if self.isTrain:
            # optimizer G
            if opt.niter_fix_global > 0:
                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith("model" + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split(".")[0])
                print(
                    "------------- Only training the local enhancer network (for %d epochs) ------------"
                    % opt.niter_fix_global
                )
                print("The layers that are finetuned are ", sorted(finetune_list))
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            params = list(self.netD2.parameters())
            self.optimizer_D2 = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

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
        self.image_paths = input["S_paths"]  # for save_image in test.py
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
            self.real_gx = torch.unsqueeze(self.real_T[:, 0, :, :], 1)
            self.real_gy = torch.unsqueeze(self.real_T[:, 1, :, :], 1)
        if timing:
            print("set_input takes time", time.time() - start_time)

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        """
        Encode the input based on the label map and instance map.
        The function is adapted from the original pix2pixHD code for the patcchskit dataset.
        In our case, the label_map would be the grayscale sketch image. Since we don't use any other semantic map or feature of the input image, we set the other inputs to None by default.

        Args:
            label_map (tensor): the input label map (sketch in this case)
            inst_map (tensor, optional): the instance map. Defaults to None.
            real_image (tensor, optional): real image for input. Defaults to None.
            feat_map (tensor, optional): feature map for input. Defaults to None.
            infer (bool, optional): whether to run inference. Defaults to False.

        Returns:
            input_label (tensor): the input label map (sketch in this case)
            inst_map (tensor): the instance map
            edge_map (tensor): the edge map
            real_image (tensor): real image for input
            feat_map (tensor): feature map for input
        """
        if self.opt.label_nc == 0:
            input_label = label_map.data.to(self.device)
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_().to(self.device)
            input_label = input_label.scatter_(1, label_map.data.long().to(self.device), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.to(self.device)
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.to(self.device))

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.to(self.device))
            if self.opt.label_feat:
                inst_map = label_map.to(self.device)

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False, use_D2=False):
        """
        Run discriminators on input_label and test_image.

        Args:
            input_label (tensor): label map (sketch in this case) as the conditional input
            test_image (tensor): the output image to be discriminated
            use_pool (bool, optional): whether to use the image pool, inherited from original pix2pixHD implementation. Defaults to False.
            use_D2 (bool, optional): option to run the D2 discriminator. For visual-tactile synthesis, we have D for visual output and D2 for tactile output. Defaults to False.

        Returns:
            discriminator output (tensor): the output of the discriminator
        """
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_D2:
            return self.netD2.forward(input_concat)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, infer=False):
        """ "
        Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        # encode inputs. For visual-tactile synthesis, we only need the sketch as the input.
        inst = None
        image = None
        feat = None
        input_label, inst_map, real_image, feat_map = self.encode_input(
            self.real_S, inst, image, feat
        )  # shape [32, 1, 32, 32] for patchskit training
        self.input_label = input_label

        # forward pass
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        fake_image = self.netG.forward(input_concat)  # shape [32, 5, 32, 32]
        # separate the output into visual and tactile outputs
        self.fake_I = fake_image[:, :3, :, :]
        self.fake_T = fake_image[:, -2:, :, :]

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
        Compute loss and backpropagate for the discriminators.
        """
        # Discriminator D for visual output
        pred_fake_I_pool = self.discriminate(self.input_label, self.fake_I, use_pool=True)
        self.loss_D_fake = self.criterionGAN(pred_fake_I_pool, False)
        pred_real_I = self.discriminate(self.input_label, self.real_I)
        self.loss_D_real = self.criterionGAN(pred_real_I, True)
        self.loss_D1 = (self.loss_D_fake + self.loss_D_real) * 0.5

        # Discriminator D2 for tactile output
        pred_fake_T = self.discriminate(self.input_label, self.fake_T, use_pool=False, use_D2=True)
        self.loss_D2_fake = self.criterionGAN(pred_fake_T, False)
        pred_real_T = self.discriminate(self.input_label, self.real_T, use_pool=False, use_D2=True)
        self.loss_D2_real = self.criterionGAN(pred_real_T, True)
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5

        # Combined losses for the discriminators
        self.loss_D = self.loss_D1 + self.loss_D2
        self.loss_D.backward()

    def backward_G(self):
        """
        Compute loss and backpropagate for the generator.
        """
        # GAN loss
        pred_fake_I = self.netD.forward(torch.cat((self.input_label, self.fake_I), dim=1))
        self.loss_G_GAN_I = self.criterionGAN(pred_fake_I, True)
        pred_fake_T = self.netD2.forward(torch.cat((self.input_label, self.fake_T), dim=1))
        self.loss_G_GAN_T = self.criterionGAN(pred_fake_T, True)
        self.loss_G_GAN = self.loss_G_GAN_I + self.loss_G_GAN_T

        # GAN feature matching loss
        self.loss_G_GAN_Feat_I = 0
        self.loss_G_GAN_Feat_T = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D_D1
            for i in range(self.opt.num_D_D1):
                for j in range(len(pred_fake_I[i]) - 1):
                    self.loss_G_GAN_Feat_I += (
                        D_weights
                        * feat_weights
                        * self.criterionFeat(pred_fake_I[i][j], pred_fake_I[i][j].detach())
                        * self.opt.lambda_feat
                    )
            D_weights = 1.0 / self.opt.num_D_D2
            for i in range(self.opt.num_D_D2):
                for j in range(len(pred_fake_T[i]) - 1):
                    self.loss_G_GAN_Feat_T += (
                        D_weights
                        * feat_weights
                        * self.criterionFeat(pred_fake_T[i][j], pred_fake_T[i][j].detach())
                        * self.opt.lambda_feat
                    )
        self.loss_G_GAN_Feat = self.loss_G_GAN_Feat_I + self.loss_G_GAN_Feat_T

        # VGG feature matching loss
        self.loss_G_VGG_I = 0
        self.loss_G_VGG_T = 0
        if not self.opt.no_vgg_loss:
            self.loss_G_VGG_I = self.criterionVGG(self.fake_I, self.real_I) * self.opt.lambda_vgg
            fake_gx_3C = torch.cat((self.fake_gx, self.fake_gx, self.fake_gx), dim=1)
            fake_gy_3C = torch.cat((self.fake_gy, self.fake_gy, self.fake_gy), dim=1)
            real_gx_3C = torch.cat((self.real_gx, self.real_gx, self.real_gx), dim=1)
            real_gy_3C = torch.cat((self.real_gy, self.real_gy, self.real_gy), dim=1)
            # shape [32, 3, 32, 32]
            self.loss_G_VGG_T = (
                self.criterionVGG(fake_gx_3C, real_gx_3C) + self.criterionVGG(fake_gy_3C, real_gy_3C)
            ) * self.opt.lambda_vgg
        self.loss_G_VGG = self.loss_G_VGG_I + self.loss_G_VGG_T

        # Combined loss for the generator
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_Feat + self.loss_G_VGG
        self.loss_G.backward()

    def optimize_parameters(self, epoch=0):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration.
        Add D2 to the training process.

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
        self.optimizer_D2.step()
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD2, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad(set_to_none=True)  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    ### The following three functions (test, get_current_visuals, and save) are the same as pix2pix_model.py. All of them overwrite similar functions in base_model.py to adapt to our data and model settings.###
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

    ### The following functions are inherited from the original pix2pixHD model, not used in visual-tactile synthesis project. ###
    def inference(self, label, inst, image=None):
        """
        Forward function used in inference time.

        Args:
            label (tensor): label map
            inst (tensor): instance map
            image (tensor, optional): image tensor. Defaults to None.

        Returns:
            fake_image (tensor): the generated image
        """
        # Encode Inputs
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)

        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label

        if torch.__version__.startswith("0.4"):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding="latin1").item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.to(self.device), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.to(self.device))
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num // 2, :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_().to(self.device)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, "G", which_epoch, self.gpu_ids)
        self.save_network(self.netD, "D", which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, "E", which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print("------------ Now also finetuning global generator -----------")

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group["lr"] = lr
        for param_group in self.optimizer_D2.param_groups:
            param_group["lr"] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group["lr"] = lr
        if self.opt.verbose:
            print("update learning rate: %f -> %f" % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)
