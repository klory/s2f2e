import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import functools

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=128, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
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
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf*nf_mult),
                nn.LeakyReLU(0.2, True)
                ]

        sequence += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

class Unet_G(nn.Module):
    def __init__(self, input_nc, output_nc, which_model, nfg=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_G, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.nfg = nfg
        self.gpu_ids = gpu_ids
        self.which_model = which_model
        padding_type = 'reflect' 
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv1 = nn.Sequential(nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, nfg//2, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(nfg//2),
                 nn.ReLU(True))
        self.conv2 = ConvBlock(nfg//2, nfg, 3, 2, 1, norm_layer, use_dropout)

        self.conv3 = ConvBlock(nfg, nfg*2, 3, 2, 1, norm_layer, use_dropout)

        self.res1 = ResnetBlock(nfg*2, padding_type, norm_layer, use_dropout, use_bias)
        self.res2 = ResnetBlock(nfg*2, padding_type, norm_layer, use_dropout, use_bias)
        if which_model == 'EFG':
            self.conv4 = ConvBlock(nfg*2, nfg*4, 3, 2, 1, norm_layer, use_dropout, use_bias)
            self.conv5 = ConvBlock(nfg*4, nfg*8, 3, 2, 1, norm_layer, use_dropout, use_bias)
            self.convTran1 = ConvTransBlock(nfg*8 + 3, nfg*4, norm_layer=nn.BatchNorm2d, use_dropout=False, first_layer=False, last_layer=False)
            self.convTran2 = ConvTransBlock(nfg*4*2, nfg*2, norm_layer=nn.BatchNorm2d, use_dropout=False, first_layer=False, last_layer=False)
        self.res3 = ResnetBlock(nfg*2, padding_type, norm_layer, use_dropout, use_bias)
        self.res4 = ResnetBlock(nfg*2, padding_type, norm_layer, use_dropout, use_bias)
        self.res5 = ResnetBlock(nfg*2, padding_type, norm_layer, use_dropout, use_bias)

        self.convTran3 = ConvTransBlock(nfg*2*2, nfg, norm_layer=nn.BatchNorm2d, use_dropout=False, first_layer=False, last_layer=False)
        self.convTran4 = ConvTransBlock(nfg*2, nfg//2, norm_layer=nn.BatchNorm2d, use_dropout=False, first_layer=False, last_layer=False)

        self.conv6 = nn.Sequential(nn.ReflectionPad2d(3),
                 nn.Conv2d(nfg, 3, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(3),
                 nn.tanh()

    def forward(self, x, v):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_res1 = self.res1(out_conv3)
        out_res2 = self.res2(out_res1)
        if self.which_model == 'EFG':
            out_conv4 = self.conv4(out_res2)
            out_conv5 = self.conv5(out_conv4)
            out_convtran1 = self.convTran1(torch.cat((out_conv5, v), 1))
            out_convtran2 = self.convTran2(torch.cat((out_convtran1, out_conv4), 1))
            out_res3 = self.res3(out_convtran2)
        else:
            out_res3 = self.res3(out_res2)
        out_res4 = self.res4(out_res3)
        out_tran3 = self.convTran3(torch.cat((out_res4, out_conv3), 1))
        out_tran4 = self.convTran4(torch.cat((out_tran3, out_conv2), 1))
        out_conv6 = self.conv6(torch.cat((out_tran4, out_conv1), 1))

        return out_conv6

class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, k, s, p, norm_layer=nn.BatchNorm2d, use_dropout=False, first_layer=False, last_layer=False):
        super(ConvBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        model = [nn.Conv2d(input_nc, output_nc, kernel_size=k, stride=s, padding=p, bias=use_bias)]
        if not last_layer and not first_layer:
            model += [norm_layer(output_nc)]
            model += [nn.LeakyReLU(0.2, True)]
        if first_layer:
            model += [nn.LeakyReLU(0.2, True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

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
