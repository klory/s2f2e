import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import functools

"""
Basically, the generator is same as network_NFG.py. Except that in the bottle-neck layer, we add up a linear layer to 
map (512+3) dimension feature into 512 dimension space.
Encoder: downsampling
Decoder: upsampling

Discriminator is the same as discriminator in network_NFG.py
"""
class Encoder(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        model = []
        model += [ConvBlock(input_nc, ngf, norm_layer=norm_layer, first_layer=True)]
        model += [ConvBlock(ngf, ngf*2, norm_layer=norm_layer)]
        model += [ConvBlock(ngf*2, ngf*4, norm_layer=norm_layer)]
        model += [ConvBlock(ngf*4, ngf*8, norm_layer=norm_layer)]
        model += [ConvBlock(ngf*8, ngf*8, norm_layer=norm_layer)]
        model += [ConvBlock(ngf*8, ngf*8, norm_layer=norm_layer, last_layer=True)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=False, first_layer=False, last_layer=False):
        super(ConvBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        model = [nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        if not last_layer and not first_layer:
            model += [norm_layer(output_nc)]
            model += [nn.LeakyReLU(0.2, True)]
        if first_layer:
            model += [nn.LeakyReLU(0.2, True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Decoder, self).__init__()
        model = []
        self.ngf = ngf
        self.linear = nn.Linear(ngf*8+3, ngf*8)
        model += [ConvTransBlock(ngf*8, ngf*8, norm_layer=norm_layer)]
        model += [ConvTransBlock(ngf*8, ngf*8, norm_layer=norm_layer)]
        model += [ConvTransBlock(ngf*8, ngf*4, norm_layer=norm_layer)]
        model += [ConvTransBlock(ngf*4, ngf*2, norm_layer=norm_layer)]
        model += [ConvTransBlock(ngf*2, ngf, norm_layer=norm_layer)]
        model += [ConvTransBlock(ngf, input_nc, norm_layer=norm_layer, last_layer=True)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        batch_size = x.size()[0]
        x = self.linear(x)
        x = x.view(batch_size, self.ngf*8, 1, 1)
        x = self.model(x)
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=False, first_layer=False, last_layer=False):
        super(ConvTransBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        model = [nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        if not last_layer:
            model += [norm_layer(output_nc)]
            model += [nn.ReLU(True)]
            if use_dropout:
                model += [nn.Dropout(0.5)]
        if last_layer:
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

"""
input_nc = 3
ngf = 64
no_dropout = True
norm = functools.partial(nn.BatchNorm2d, affine=True)

decoder = Decoder(input_nc, norm_layer=norm)
print decoder
"""
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=128, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
                ]

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