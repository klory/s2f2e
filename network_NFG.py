import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import functools

# Generator
class Unet_G(nn.Module):
    """
    input_nc: number of channels of the input(image)
    output_nc: number of channels of the output(image)
    num_downs: down sampling times, if the image is 128*128, then the num_downs should be 7
    ngf: number of filtes of the first conv layer
    norm_layer: normalize function
    gpu_ids: not used for now. I don't know what this mean. ^.^
    """
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_G, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, input_nc=None, innermost=True, norm_layer=norm_layer)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf*2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc,submodule=unet_block, outmost=True, norm_layer=norm_layer)
        self.model = unet_block
    
    def forward(self, input):
        return self.model(input)
    

"""
This generator is like a U-shape network. Starts from the innermost part, then built the outter layers.
So the code is like:
|--downsampling -- |submodule| -- upsampling--|
"""
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outmost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outmost = outmost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc == None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outmost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            # TODO: check paper about the last layer of encoder, do we have a batch_norm here?
            #down = [downrelu, downconv, downnorm]
            down = [downrelu, downconv]
            up = [upconv, uprelu, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            # TODO: check the structure of decoder, the position of relu, code and paper don't match
            up = [upconv, uprelu, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Discriminator
class NLayerDiscriminator(nn.Module):
    """
    input_nc: number of channels of input(image)
    ndf: number of filters of the first conv layer of discriminator
    n_layers: number of layers in the middle of discriminator, which is number of layers execpt the first and last layer
    norm_layer: normalize function
    """
    def __init__(self, input_nc, ndf=128, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # kernel size 4
        kw = 4
        # padding 1
        padw = 1
        sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
                ]

        # use nf_mult and fn_mult_prev to track input/output channel numbers of in each layer
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                    nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf*nf_mult),
                    nn.LeakyReLU(0.2, True)
                    ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=1, padding=padw),
                norm_layer(ndf*nf_mult),
                nn.LeakyReLU(0.2, True)
                ]

        sequence += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
