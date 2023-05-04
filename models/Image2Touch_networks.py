import functools
import glob
import math
from email.policy import strict
from multiprocessing.sharedctypes import Value

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from click import version_option
from numpy import isin
from torch.autograd import Variable
from torch.nn import init
from torch.utils import model_zoo
from torchvision import models
from torchvision.models.resnet import model_urls

"""Customize ResNet for Tactile Syntheis"""


### Helper functions for weight initialization ###
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find("Linear") != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_normal(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm2d") != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find("BatchNorm2d") != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type="normal"):
    if init_type == "normal":
        net.apply(weights_init_normal)
    elif init_type == "xavier":
        net.apply(weights_init_xavier)
    elif init_type == "kaiming":
        net.apply(weights_init_kaiming)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError("initialization method [%s] is not implemented" % init_type)


### Setting up ResNet and its variants ###
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    # RH 2022-03-04 change the default padding_mode from "zero" to "reflect"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, padding_mode="reflect")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_stats=False):
        super(BasicBlock, self).__init__()

        self.track_stats = track_stats
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes, track_running_stats=self.track_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes, track_running_stats=self.track_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_stats=False):
        super(Bottleneck, self).__init__()

        self.track_stats = track_stats

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.InstanceNorm2d(track_running_stats=self.track_stats)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.InstanceNorm2d(planes, track_running_stats=self.track_stats)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.InstanceNorm2d(planes * 4, track_running_stats=self.track_stats)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, track_stats=False):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.track_stats = track_stats

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.InstanceNorm2d(64, track_running_stats=self.track_stats)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)

        # self.fc = nn.Linear(512 * block.expansion, 1)
        # Rewrite fc as fc_conv
        n_in = 512 * block.expansion
        n_out = 1
        self.fc_conv = nn.Conv2d(n_in, n_out, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(planes * block.expansion, track_running_stats=self.track_stats),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class ResNet_CIFAR(nn.Module):
    """
    Modified version of resnet in the resnet paper that they used for CIFAR
    Ref: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    """

    def __init__(self, block, layers, track_stats=False, ni=3, inplanes=16):
        super(ResNet_CIFAR, self).__init__()
        self.inplanes = inplanes
        self.track_stats = track_stats
        self.conv1 = nn.Conv2d(ni, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(self.inplanes, track_running_stats=self.track_stats)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, self.inplanes, layers[0], stride=1)
        self.layer2 = self._make_layer(block, self.inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.inplanes * 4, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(planes * block.expansion, track_running_stats=self.track_stats),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # rewrite the original "forward" as "forward_resnset_I32"
    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = F.avg_pool2d(out, out.size()[3])
    #     out = out.view(out.size(0), -1)
    #     out = self.linear(out)
    #     return out


def resnet_CIFAR(model_name="resnet32", pretrained=True, **kwargs):
    pretrained_model_parent_dir = "../pytorch_resnet_cifar10/pretrained_models"
    if model_name == "resnet20":
        model = ResNet_CIFAR(BasicBlock, [3, 3, 3], **kwargs)
    elif model_name == "resnet32":
        # model = ResNet_CIFAR(BasicBlock, [5,5,5], **kwargs)
        model = ResNet_CIFAR(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif model_name == "resnet44":
        model = ResNet_CIFAR(BasicBlock, [7, 7, 7], **kwargs)
    elif model_name == "resnet56":
        model = ResNet_CIFAR(BasicBlock, [9, 9, 9], **kwargs)
    elif model_name == "resnet110":
        model = ResNet_CIFAR(BasicBlock, [18, 18, 18], **kwargs)
    else:
        raise NotImplementedError("Invalid name %s for resnet_cifar" % model_name)

    if pretrained:
        print(f"Load pretrained model {model_name}")
        model_dict = model.state_dict()
        pretrained_model_path = glob.glob("%s/%s-*.th" % (pretrained_model_parent_dir, model_name))[0]
        print(f"Load model from {pretrained_model_path}")

        pretrained_dict = torch.load(pretrained_model_path)
        # 1. filter out unnecessary keys (also check the shape)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)
        }

        pretrained_keys = pretrained_dict.keys()
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        pretrained_keys = None

    return model, pretrained_keys


class ResNet_I32(nn.Module):
    """
    Modify the initial convolution and combination of the resnet block so that it fits to out dataset
    with input size (32x32) instead of the original input (256x256)
    """

    def __init__(self, ni, block, layers, track_stats=False):
        ngf = 16  # 64
        self.inplanes = ngf
        super(ResNet_I32, self).__init__()
        self.track_stats = track_stats
        layers0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ni, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf, track_running_stats=self.track_stats),
            nn.ReLU(inplace=True),
        ]
        self.layer0 = nn.Sequential(*layers0)
        self.layer1 = self._make_layer(block, ngf * 1, layers[0])
        self.layer2 = self._make_layer(block, ngf * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, ngf * 4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
        n_in = ngf * 4 * block.expansion
        n_out = 1
        self.fc_conv = nn.Conv2d(n_in, n_out, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(planes * block.expansion, track_running_stats=self.track_stats),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def resnet_I32(ni, pretrained=False, **kwargs):
    model = ResNet_I32(ni, BasicBlock, [2, 2, 2], **kwargs)
    if pretrained:
        print("Load pretrained resnet18 for resnet_I32")
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls["resnet18"])
        # 1. filter out unnecessary keys (also check the shape)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)
        }

        pretrained_keys = pretrained_dict.keys()
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        pretrained_keys = None

    return model, pretrained_keys


class ResNet_I64(nn.Module):
    """
    Modify the initial convolution and combination of the resnet block so that it fits to out dataset
    with input size (64x64) instead of the original input (256x256)
    """

    def __init__(self, block, layers, track_stats=False):
        self.inplanes = 64
        super(ResNet_I64, self).__init__()
        self.track_stats = track_stats
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.InstanceNorm2d(self.inplanes, track_running_stats=self.track_stats)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
        n_in = 512 * block.expansion
        n_out = 1
        self.fc_conv = nn.Conv2d(n_in, n_out, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(planes * block.expansion, track_running_stats=self.track_stats),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def resnet_I64(pretrained=False, **kwargs):
    model = ResNet_I64(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        print("Load pretrained resnet18 for resnet_I64")
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls["resnet18"])
        # 1. filter out unnecessary keys (also check the shape)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)
        }

        pretrained_keys = pretrained_dict.keys()
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        pretrained_keys = None

    return model, pretrained_keys


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


class Interpolate(nn.Module):
    """
    Create a class to use `interpolate` in sequence
    Ref: https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    """

    def __init__(self, size=None, mode="bilinear", scale_factor=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, scale_factor=self.scale_factor)
        return x


class _netG_resnet(nn.Module):
    def __init__(
        self,
        ni,
        no,
        ngf=64,
        kernel_size=3,
        stride=1,
        padding=1,
        padding_mode="reflect",
        src_skip=True,
        norm="Instance",
        init_type="normal",
        track_stats=False,
        pretrained_resnet=False,
        input_size=32,
        encoder="resnet32",
        T_resolution_multiplier=2,
        opt=None,
    ):
        super(_netG_resnet, self).__init__()
        print("Initializing _netG_resnet ni {} input_size {}".format(ni, input_size))

        self.input_size = input_size

        if input_size == 32:
            # Using ResNet_CIFAR as the encoder for input 32x32
            print("Using ResNet_CIFAR as the encoder for input 32x32")
            self.resnet_src, pretrained_keys = resnet_I32(ni, pretrained=pretrained_resnet)
            decoder_n_in = self.resnet_src.fc_conv.in_channels
        else:
            raise NotImplementedError("Unknown input_size", self.input_size)

        self.relu = nn.ReLU()
        self.decoder = nn.Module()
        self.src_skip = src_skip

        # set normalization
        self.norm = norm
        self.track_stats = track_stats
        # replace ConvTranspose2d by Interpolate and Conv2d
        # Note: the channel after Interpolation change in each layer because of the concatenation (# layers doubles)

        ft_scaling_factor = 1
        if self.src_skip:
            ft_scaling_factor += 1
        self.T_resolution_multiplier = T_resolution_multiplier

        ngf = 16
        ngf_mulitplier = 4  # for 64x64 input, use 8
        self.decoder.itpl_00 = Interpolate(scale_factor=2)
        self.decoder.conv_00 = nn.Conv2d(
            decoder_n_in, ngf * ngf_mulitplier, kernel_size, stride, padding, padding_mode=padding_mode
        )
        self.decoder.norm_00 = nn.InstanceNorm2d(ngf * 8, track_running_stats=self.track_stats)

        self.decoder.itpl_01 = Interpolate(scale_factor=2)
        self.decoder.conv_01 = nn.Conv2d(
            ngf * ngf_mulitplier, ngf * ngf_mulitplier, kernel_size, stride, padding, padding_mode=padding_mode
        )
        self.decoder.norm_01 = nn.InstanceNorm2d(ngf * 8, track_running_stats=self.track_stats)

        current_num_ft = ngf * ngf_mulitplier
        self.decoder.itpl_2 = Interpolate(scale_factor=2)
        self.decoder.conv_2 = nn.Conv2d(
            ft_scaling_factor * current_num_ft,
            current_num_ft // 2,
            kernel_size,
            stride,
            padding,
            padding_mode=padding_mode,
        )
        self.decoder.norm_2 = nn.InstanceNorm2d(current_num_ft // 2, track_running_stats=self.track_stats)
        self.decoder.itpl_3 = Interpolate(scale_factor=2)

        current_num_ft = current_num_ft // 2
        self.decoder.conv_3 = nn.Conv2d(
            ft_scaling_factor * current_num_ft,
            current_num_ft // 2,
            kernel_size,
            stride,
            padding,
            padding_mode=padding_mode,
        )
        self.decoder.norm_3 = nn.InstanceNorm2d(current_num_ft // 2, track_running_stats=self.track_stats)

        if self.input_size == 32:
            current_num_ft = current_num_ft // 2
            self.decoder.itpl_4 = Interpolate(scale_factor=2)
            if T_resolution_multiplier == 2:
                self.decoder.conv_4 = nn.Conv2d(
                    ft_scaling_factor * current_num_ft, no, kernel_size, stride, padding, padding_mode=padding_mode
                )
                self.decoder.norm_4 = nn.InstanceNorm2d(current_num_ft // 2, track_running_stats=self.track_stats)
            elif T_resolution_multiplier == 4:
                # Enlarge the resolution by 4 times
                self.decoder.conv_4 = nn.Conv2d(
                    current_num_ft, current_num_ft // 2, kernel_size, stride, padding, padding_mode=padding_mode
                )
                self.decoder.norm_4 = nn.InstanceNorm2d(current_num_ft // 2, track_running_stats=self.track_stats)
                current_num_ft = current_num_ft // 2
                self.decoder.itpl_5 = Interpolate(scale_factor=2)
                self.decoder.conv_5 = nn.Conv2d(
                    current_num_ft, no, kernel_size, stride, padding, padding_mode=padding_mode
                )
                self.decoder.norm_5 = nn.InstanceNorm2d(current_num_ft // 2, track_running_stats=self.track_stats)
            else:
                raise NotImplementedError("Unknown T_resolution_multiplier", T_resolution_multiplier)

        init_weights(self.decoder, init_type=init_type)

        if not pretrained_resnet:
            # init weights for the entire resnet_src model
            init_weights(self.resnet_src, init_type=init_type)
        else:
            # init weights only for the conv1 layer of the resnet_src model
            # use pretrained weights for the rest of resnet_src model
            # print("init weight for self.resnet_src.conv1")
            init_weights(self.resnet_src.conv1, init_type=init_type)
            # pass

    def forward_resnet_I64(self, net, x, verbose=False):
        """
        Since input size reduce from 256x256 to 64x64, we remove maxpooling before layer1 and avgpool after layer5 from originial Resnet18.
        """
        # input shape [N, in_channel, 64, 64]
        x = net.conv1(x)
        x = net.bn1(x)
        ft_0 = net.relu(x)  # [N, 64, 32, 32]
        ft_1 = net.layer1(ft_0)  # [N, 64, 32, 32]
        ft_2 = net.layer2(ft_1)  # [N, 128, 16, 16]
        ft_3 = net.layer3(ft_2)  # [N, 256, 8, 8]
        ft_4 = net.layer4(ft_3)  # [N, 512, 4, 4]
        ft_5 = net.avgpool(ft_4)  # [N, 512, 1, 1]
        return ft_0, ft_1, ft_2, ft_3, ft_4, ft_5

    def forward_resnet_I32(self, net, x, verbose=False):
        # input shape [1, 4, 1536, 1536]
        ft_0 = net.layer0(x)
        ft_1 = net.layer1(ft_0)  # [1, 64, 768, 768]
        ft_2 = net.layer2(ft_1)  # [1, 128, 384, 384]
        ft_3 = net.layer3(ft_2)  # [1, 256, 192, 192]
        ft_4 = net.avgpool(ft_3)  # [1, 256, 48, 48]
        return ft_0, ft_1, ft_2, ft_3, ft_4

    def forward(self, src, verbose=False):
        ### Encoder ###
        if self.input_size == 64:
            src_fts = self.forward_resnet_I64(self.resnet_src, src, verbose=verbose)
        elif self.input_size == 32:
            src_fts = self.forward_resnet_I32(self.resnet_src, src, verbose=verbose)

        ### Decoder ###
        src_fts_copy = [x.clone() for x in src_fts]
        # (512 + 512 + 512 + 512) x 1 x 1
        cat_gxy = src_fts[-1].clone()
        cat_gxy = self.decoder.itpl_00(cat_gxy)
        cat_gxy = self.decoder.conv_00(cat_gxy)
        cat_gxy = self.decoder.norm_00(cat_gxy)
        cat_gxy = self.relu(cat_gxy)
        cat_gxy = self.decoder.itpl_01(cat_gxy)
        cat_gxy = self.decoder.conv_01(cat_gxy)
        cat_gxy = self.decoder.norm_01(cat_gxy)
        cat_gxy = self.relu(cat_gxy)

        # (512 + 512) x 8 x 8
        if self.src_skip:
            cat_gxy = torch.cat((cat_gxy, src_fts[-2]), 1)
        cat_gxy = self.decoder.itpl_2(cat_gxy)
        cat_gxy = self.decoder.conv_2(cat_gxy)
        cat_gxy = self.decoder.norm_2(cat_gxy)
        cat_gxy = self.relu(cat_gxy)

        # (256 + 256) x 16 x 16
        if self.src_skip:
            cat_gxy = torch.cat((cat_gxy, src_fts[-3]), 1)
        cat_gxy = self.decoder.itpl_3(cat_gxy)
        cat_gxy = self.decoder.conv_3(cat_gxy)
        cat_gxy = self.decoder.norm_3(cat_gxy)
        cat_gxy = self.relu(cat_gxy)

        # (128 + 128) x 32 x 32
        if self.src_skip:
            cat_gxy = torch.cat((cat_gxy, src_fts[-4]), 1)

        if self.input_size == 32:
            cat_gxy = self.decoder.itpl_4(cat_gxy)
            cat_gxy = self.decoder.conv_4(cat_gxy)
            cat_gxy = self.decoder.norm_4(cat_gxy)
            cat_gxy = self.relu(cat_gxy)

            if self.T_resolution_multiplier == 2:
                cat_gxy = nn.Tanh()(cat_gxy)
            elif self.T_resolution_multiplier == 4:
                cat_gxy = self.relu(cat_gxy)  # 14, 16, 64, 64]
                cat_gxy = self.decoder.itpl_6(cat_gxy)  # 14, 16, 128, 128]
                cat_gxy = self.decoder.conv_5(cat_gxy)  # [14, 2, 128, 128]
                cat_gxy = nn.Tanh()(cat_gxy)
        return cat_gxy


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(
        self,
        use_lsgan=True,
        use_gpu=True,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
        use_lpips=False,
        use_vgg=True,
    ):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.use_gpu = use_gpu
        if use_lsgan:
            self.loss = nn.MSELoss()
        elif use_lpips:  # add by RH 2022-02-18. Ref: https://github.com/richzhang/PerceptualSimilarity
            if use_vgg:  # # closer to "traditional" perceptual loss, when used for optimization
                self.loss = lpips.LPIPS(net="vgg")
            else:  # best forward scores
                self.loss = lpips.LPIPS(net="alex")
        else:
            self.loss = nn.BCELoss()

        if use_gpu:
            self.loss.cuda()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = (self.real_label_var is None) or (self.real_label_var.numel() != input.numel())
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                if self.use_gpu:
                    real_tensor = real_tensor.cuda()
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel())
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                if self.use_gpu:
                    fake_tensor = fake_tensor.cuda()
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # print("check tensor in loss function, input {} target_tensor {}".format(input.shape, target_tensor.shape))
        return self.loss(input, target_tensor)


### 2022-03-25 Copy from non-stationary texture synthesis repo ###
# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        gpu_ids=[],
        padding_type="reflect",
    ):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
