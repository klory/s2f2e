import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import functools

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, batch_size, ndf=128, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        self.batch_size = batch_size
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

        sequence += [nn.Conv2d(ndf*nf_mult, 4, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        out = out.view(self.batch_size, -1)
        return out

class Unet_G1(nn.Module):
    """
    Out proposed network. Unet + ResNet
    """
    def __init__(self, input_nc, output_nc, which_model, nfg=64, norm_layer=nn.BatchNorm2d, use_dropout=True, gpu_ids=[]):
        super(Unet_G1, self).__init__()
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
                 norm_layer(nfg//2), nn.Dropout(0.5))
        self.conv2 = ConvBlock(nfg//2, nfg, 3, 2, 1, norm_layer=norm_layer, use_dropout=use_dropout)
        self.conv3 = ConvBlock(nfg, nfg*2, 3, 2, 1, norm_layer=norm_layer, use_dropout=use_dropout)

        self.res1 = ResnetBlock(nfg*2, padding_type, norm_layer, use_dropout, use_bias)
        self.res2 = ResnetBlock(nfg*2, padding_type, norm_layer, use_dropout, use_bias)

        self.conv4 = ConvBlock(nfg*2, nfg*4, 3, 2, 1, norm_layer=norm_layer, use_dropout=use_dropout)
        self.conv5 = ConvBlock(nfg*4, nfg*8, 3, 2, 1, norm_layer=norm_layer, use_dropout=use_dropout)

        self.res3 = ResnetBlock(nfg*8, padding_type, norm_layer, use_dropout, use_bias)
        self.res4 = ResnetBlock(nfg*8, padding_type, norm_layer, use_dropout, use_bias)

        self.conv6 = ConvBlock(nfg*8, nfg*8, 3, 2, 1, norm_layer=norm_layer, use_dropout=use_dropout)
        self.conv7 = ConvBlock(nfg*8, nfg*8, 3, 2, 1, norm_layer=norm_layer, use_dropout=use_dropout)

        if which_model == 'EFG':
            self.convTran0 = ConvTransBlock(nfg*8 + 3, nfg*8, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, first_layer=False, last_layer=False)

        self.convTran1 = ConvTransBlock(nfg*8, nfg*8, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, first_layer=False, last_layer=False)
        self.convTran2 = ConvTransBlock(nfg*8*2, nfg*8, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, first_layer=False, last_layer=False)

        self.res5 = ResnetBlock(nfg*8, padding_type, norm_layer, use_dropout, use_bias)
        self.res6 = ResnetBlock(nfg*8, padding_type, norm_layer, use_dropout, use_bias)

        self.convTran3 = ConvTransBlock(nfg*8*2, nfg*4, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, first_layer=False, last_layer=False)
        self.convTran4 = ConvTransBlock(nfg*4*2, nfg*2, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, first_layer=False, last_layer=False)

        self.res7 = ResnetBlock(nfg*2, padding_type, norm_layer, use_dropout, use_bias)
        self.res8 = ResnetBlock(nfg*2, padding_type, norm_layer, use_dropout, use_bias)

        self.convTran5 = ConvTransBlock(nfg*2, nfg, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, first_layer=False, last_layer=False)
        self.convTran6 = ConvTransBlock(nfg, nfg//2, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, first_layer=False, last_layer=False)

        self.conv8 = nn.Sequential(nn.ReflectionPad2d(3),
                 nn.Conv2d(nfg//2, 3, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(3),
                 nn.Tanh())

    def forward(self, x, v):
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_res1 = self.res1(out_conv3)
        out_res2 = self.res2(out_res1)

        out_conv4 = self.conv4(out_res2)
        out_conv5 = self.conv5(out_conv4)
        out_res3 = self.res3(out_conv5)
        out_res4 = self.res4(out_res3)

        out_conv6 = self.conv6(out_res4)
        out_conv7 = self.conv7(out_conv6)

        if self.which_model == 'EFG':
            out_tran0 = self.convTran0(torch.cat((out_conv7, v), 1))
            out_tran1 = self.convTran1(out_tran0)
        else:
            out_tran1 = self.convTran1(out_conv7)

        out_tran2 = self.convTran2(torch.cat((out_tran1, out_conv6), 1))

        out_res5 = self.res5(out_tran2)
        out_res6 = self.res6(out_res5)
        out_tran3 = self.convTran3(torch.cat((out_res6, out_conv5), 1))
        out_tran4 = self.convTran4(torch.cat((out_tran3, out_conv4), 1))

        out_res7 = self.res7(out_tran4)
        out_res8 = self.res8(out_res7)
        out_tran5 = self.convTran5(out_res7)
        out_tran6 = self.convTran6(out_res8)

        out_conv6 = self.conv6(out_tran6)

        return out_conv6

class Unet_G2(nn.Module):
    """
    The same network structure as in the paper Pix2Pix.
    """
    def __init__(self, input_nc, output_nc, which_model, nfg=64, norm_layer=nn.BatchNorm2d, use_dropout=True, gpu_ids=[]):
        super(Unet_G2, self).__init__()
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

        k = 4
        s = 2
        p = 1

        # Encoder
        self.conv1 = ConvBlock(input_nc, nfg, k, s, p, norm_layer=norm_layer, use_dropout=not use_dropout)
        self.conv2 = ConvBlock(nfg, nfg*2, k, s, p, norm_layer=None, use_dropout=not use_dropout)

        self.conv3 = ConvBlock(nfg*2, nfg*4, k, s, p, norm_layer=norm_layer, use_dropout=not use_dropout)
        self.conv4 = ConvBlock(nfg*4, nfg*8, k, s, p, norm_layer=norm_layer, use_dropout=not use_dropout)
        self.conv5 = ConvBlock(nfg*8, nfg*8, k, s, p, norm_layer=norm_layer, use_dropout=not use_dropout)
        self.conv6 = ConvBlock(nfg*8, nfg*8, k, s, p, norm_layer=norm_layer, use_dropout=not use_dropout)
        self.conv7 = ConvBlock(nfg*8, nfg*8, k, s, p, norm_layer=norm_layer, use_dropout=not use_dropout)

        # For one-hot expression code
        if which_model == 'EFG':
            self.linear = nn.Linear(nfg*8+3, nfg*8)

        # Dncoder
        self.convTran1 = ConvTransBlock(nfg*8, nfg*8, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, first_layer=False, last_layer=False)
        self.convTran2 = ConvTransBlock(nfg*8*2, nfg*8, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, first_layer=False, last_layer=False)
        self.convTran3 = ConvTransBlock(nfg*8*2, nfg*8, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, first_layer=False, last_layer=False)
        self.convTran4 = ConvTransBlock(nfg*8*2, nfg*2*2, norm_layer=nn.BatchNorm2d, use_dropout=not use_dropout, first_layer=False, last_layer=False)
        self.convTran5 = ConvTransBlock(nfg*4*2, nfg*1*2, norm_layer=nn.BatchNorm2d, use_dropout=not use_dropout, first_layer=False, last_layer=False)
        self.convTran6 = ConvTransBlock(nfg*2*2, nfg, norm_layer=nn.BatchNorm2d, use_dropout=not use_dropout, first_layer=False, last_layer=False)
        self.convTran7 = ConvTransBlock(nfg*1*2, nfg, norm_layer=nn.BatchNorm2d, use_dropout=not use_dropout, first_layer=False, last_layer=False)
        self.conv8 = ConvBlock(nfg, 3, 3, 1, 1, norm_layer=nn.BatchNorm2d, use_dropout=not use_dropout, first_layer=False, last_layer=True)

    def forward(self, x, v):
        batch_size = x.size()[0]
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        if self.which_model == 'EFG':
            out_conv7_reshape = out_conv7.view(batch_size, -1)
            out_linear = self.linear(torch.cat((out_conv7_reshape, v), 1))
            out_linear = out_linear.view(batch_size, -1, 1, 1)
            out_tran1 = self.convTran1(out_linear)
        else:
            out_tran1 = self.convTran1(out_conv7)
        out_tran2 = self.convTran2(torch.cat((out_tran1, out_conv6), 1))
        out_tran3 = self.convTran3(torch.cat((out_tran2, out_conv5), 1))
        out_tran4 = self.convTran4(torch.cat((out_tran3, out_conv4), 1))
        out_tran5 = self.convTran5(torch.cat((out_tran4, out_conv3), 1))
        out_tran6 = self.convTran6(torch.cat((out_tran5, out_conv2), 1))
        out_tran7 = self.convTran7(torch.cat((out_tran6, out_conv1), 1))

        out_conv8 = self.conv8(out_tran7)

        return out_conv8

class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, k, s, p, norm_layer=nn.BatchNorm2d, use_dropout=True, first_layer=False, last_layer=False):
        super(ConvBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        model = [nn.Conv2d(input_nc, output_nc, kernel_size=k, stride=s, padding=p, bias=use_bias)]
        if norm_layer:
            model += [norm_layer(output_nc)]
        if not last_layer:
            model += [nn.LeakyReLU(0.2, True)]
        if use_dropout:
            model += [nn.Dropout(0.5)]
        if last_layer:
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.model = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

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
                       norm_layer(dim)]
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
        out = x + self.model(x)
        return out

class ConvTransBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=True, first_layer=False, last_layer=False):
        super(ConvTransBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        model = [nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        model += [norm_layer(output_nc)]
        if use_dropout:
            model += [nn.Dropout(0.5)]
        model += [nn.ReLU(True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
