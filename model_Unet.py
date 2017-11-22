from __future__ import print_function
import torch
import torch.nn as nn
import functools
from network import *
import os
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
from PIL import Image
from load_data import *
from base_model import *

class Unet(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.model = opt.model
        if 'EFG' in opt.model:
            self.which_model = 'EFG'
        elif 'NFG' in opt.model:
            self.which_model = 'NFG'
        else:
            raise ValueError("%s is not supported." % opt.model)

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.nfg = opt.nfg
        if self.nfg == 128:
            self.num_downs = 7
        elif self.nfg == 64:
            self.num_downs = 6
        else:
            raise ValueError("Only support nfg = 128 or 64. Got %d" % self.nfg)
        # configuration of network
        if opt.isTrain:
            self.no_dropout = opt.no_dropout
            self.use_sigmoid = False
            self.batch_size = opt.batch_size
            self.norm = functools.partial(nn.BatchNorm2d, affine=True)
            self.lr = opt.learning_rate
            self.lam = opt.lam_cyc
            self.beta1 = opt.beta1
            self.beta2 = opt.beta2
            self.criterionGAN = nn.MSELoss()
            self.criterionL1 = nn.L1Loss()

        # setup input
        self.input_A = self.Tensor(opt.batch_size, opt.input_nc, opt.img_size, opt.img_size)
        self.input_tgt = self.Tensor(opt.batch_size, opt.input_nc, opt.img_size, opt.img_size)
        if 'EFG' in self.model:
            self.expression_label = self.Tensor(opt.batch_size, 3)

        # intializer network
        self.net_D = NLayerDiscriminator(self.input_nc, norm_layer=self.norm, use_sigmoid=self.use_sigmoid)
        self.net_G = Unet_G(self.input_nc, self.output_nc, self.which_model, nfg=self.nfg, norm_layer=self.norm)
        if torch.cuda.device_count() > 1:
            self.net_G = nn.DataParallel(self.net_G)
            self.net_D = nn.DataParallel(self.net_D)
        if torch.cuda.is_available():
            print("Using %d GPUs." % torch.cuda.device_count())
            self.net_G.cuda()
            self.net_D.cuda()

        # setup optimizer
        if 'LSGAN' in self.model:
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        elif 'WGAN' in self.model:
            self.optimizer_G = torch.optim.RMSprop(self.net_G.parameters(), lr=self.lr)
            self.optimizer_D = torch.optim.RMSprop(self.net_D.parameters(), lr=self.lr)

        # save generated images
        self.out_dir = opt.out_dir + self.model + '/expression/'
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        print("initializing completed:\n model name: %s\n input_nc: %s\n use_sigmoid: %s\n" % (self.model, self.input_nc, self.use_sigmoid))

    def set_input(self, input):
        if 'NFG' in self.model:
            input_A = input['source']
            input_tgt = input['target']
        elif 'EFG' in self.model:
            input_A = input[0]['source']
            input_tgt = input[0]['target']
            label = self.label_generate(input[1][0], self.batch_size)
        else:
            raise ValueError("%s is not suppported." % self.model)
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_tgt.resize_(input_tgt.size()).copy_(input_tgt)
        if 'EFG' in self.model:
            self.expression_label.resize_(label.size()).copy_(label)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_tgt = Variable(self.input_tgt)
        if 'EFG' in self.model:
            self.real_label = Variable(self.expression_label)
            self.fake_tgt = self.net_G(self.real_A, self.real_label)
        else:
            self.fake_tgt = self.net_G(self.real_A, None)

    def backward_D(self):
        D_real = self.net_D(self.real_tgt)
        D_fake = self.net_D(self.fake_tgt)

        if 'WGAN' in self.model:
            self.loss_D_real = torch.mean(D_real)
            self.loss_D_fake = torch.mean(D_fake)
        else:
            self.loss_D_real = self.criterionGAN(D_real, Variable(self.Tensor(D_real.size()).fill_(1.0), requires_grad=False))
            self.loss_D_fake = self.criterionGAN(D_fake, Variable(self.Tensor(D_real.size()).fill_(0.0), requires_grad=False))

        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        D_fake = self.net_D(self.fake_tgt)
        D_real = self.net_D(self.real_tgt)
        self.loss_G_L1 = self.criterionL1(self.fake_tgt, self.real_tgt)
        if 'WGAN' in self.model:
            self.loss_G_GAN = -torch.mean(D_fake)
        else:
            self.loss_G_GAN = self.criterionGAN(D_fake, Variable(self.Tensor(D_real.size()).fill_(1.0), requires_grad=False))

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.lam
        self.loss_G.backward()

    def optimize(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        for p in self.net_D.parameters():
            p.data.clamp_(-0.01, 0.01)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def save(self, label):
        self.save_network(self.net_G, 'G', label)
        self.save_network(self.net_D, 'D', label)

    def print_current_loss(self):
        print('loss_D_real: %f\t, loss_D_fake: %f\t, loss_G_GAN: %f\t loss_G_L1: %f\t' % (self.loss_D_real.data[0], self.loss_D_fake.data[0], self.loss_G_GAN.data[0], self.loss_G_L1.data[0]))

    def save_img(self, epoch):
        for i, data in enumerate(self.data_loader):
            if i > 20:
                break
            # test_A: neutral face image
            if torch.cuda.is_available():
                test_A = Variable(data[0]['source'].cuda())
                expression_label = data[1][0]
                test = Variable(data[0]['target'].cuda())
                # one-hot expression code
                v = Variable(torch.Tensor([[1,0,0]]).cuda())
            else:
                test_A = Variable(data[0]['source'])
                expression_label = data[1][0]
                test = Variable(data[0]['target'])
                v = Variable(torch.Tensor([[1,0,0]]))
            # obtain the bottle-neck of encoder
            fake_inter = self.encoder_G(test_A)
            # reshape tensor for linear layer in decoder
            fake_inter = fake_inter.view(1, -1)

            # expression_label, 0: smile, 1: anger, 2: scream
            fake = self.decoder_G(torch.cat((fake_inter, v), 1))
            fake_numpy = fake[0].cpu().data.numpy()
            img_fake = ((np.transpose(fake_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            if expression_label == 0: 
                Image.fromarray(img_fake).save(self.out_dir + str(epoch) + '_' + str(i) + 'fake_smile' + '.jpg')
            elif expression_label == 1:
                Image.fromarray(img_fake).save(self.out_dir + str(epoch) + '_' + str(i) + 'fake_anger' + '.jpg')
            else:
                Image.fromarray(img_fake).save(self.out_dir + str(epoch) + '_' + str(i) + 'fake_scream' + '.jpg')

            img_A = ((np.transpose(test_A[0].cpu().data.numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            Image.fromarray(img_A).save(self.out_dir + str(epoch) + '_' + str(i) + 'A' + '.jpg')

    def __call__(self):
        self.train()
        print('training complete')
