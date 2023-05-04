""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class Down(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, outer_nc, inner_nc, input_nc=None, innermost=False, outermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()

        if input_nc is None: input_nc = outer_nc
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # print("In Down: outer_nc = {}, inner_nc = {}, input_nc = {}".format(outer_nc, inner_nc, input_nc))
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        if outermost:
            down = [downconv]
        elif innermost:
            down = [downrelu, downconv]
        else:
            down = [downrelu, downconv, downnorm]
        self.model = nn.Sequential(*down)

        self.outermost = outermost

    def forward(self, x):
        return self.model(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, outer_nc, inner_nc, style_inner_nc=0, innermost=False, outermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        if outermost:
            inner_nc_scaler = 1; bias=True; upnorm = nn.Tanh()
        elif innermost:
            inner_nc_scaler = 1; bias=use_bias
        else:
            inner_nc_scaler = 2; bias=use_bias
        inner_nc = inner_nc_scaler * inner_nc + style_inner_nc
        # print("In Up: outer_nc = {}, inner_nc = {}".format(outer_nc, inner_nc))
        upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)

        up = [uprelu, upconv, upnorm]
        if use_dropout:
            up += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*up)
        self.outermost = outermost
        self.innermost = innermost
        # print("init Up: outermost = {}, innermost = {}, innermost {}, outermost {}".format(self.outermost, self.innermost, innermost, outermost))


    def forward(self, x, x2):
        # print("In Up: innermost = {}, outermost = {}".format(self.innermost, self.outermost))
        if self.outermost or self.innermost:
            return self.model(x)
        else:
            return self.model(torch.cat([x, x2], 1))
        # return self.model(x)

