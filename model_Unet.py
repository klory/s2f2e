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
import matplotlib.pyplot as plt

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
            self.dropout = opt.dropout
            self.use_sigmoid = False
            self.batch_size = opt.batch_size
            self.norm = functools.partial(nn.BatchNorm2d, affine=True)
            self.lr_adam = opt.learning_rate_adam
            self.lr_rmsprop = opt.learning_rate_rmsprop
            self.lam_cyc = opt.lam_cyc
            self.lam_l1 = opt.lam_l1
            self.beta1 = opt.beta1
            self.beta2 = opt.beta2
            self.criterionGAN = nn.MSELoss()
            self.criterionL1 = nn.L1Loss()
            self.criterionCrossEnt = nn.CrossEntropyLoss()

        # setup input
        self.input_A = self.Tensor(opt.batch_size, opt.input_nc, opt.img_size, opt.img_size)
        self.input_tgt = self.Tensor(opt.batch_size, opt.input_nc, opt.img_size, opt.img_size)
        if 'EFG' in self.model:
            self.expression_label = self.Tensor(opt.batch_size, 2, 2, 3)

        # intializer network
        self.net_D = NLayerDiscriminator(self.input_nc, self.batch_size, norm_layer=self.norm, use_sigmoid=self.use_sigmoid)
        self.net_G = Unet_G2(self.input_nc, self.output_nc, self.which_model, nfg=self.nfg, norm_layer=self.norm, use_dropout=self.dropout)
        devices = self.opt.gpu_ids 
        if torch.cuda.device_count() > 1:
            self.net_G = nn.DataParallel(self.net_G, device_ids=devices)
            self.net_D = nn.DataParallel(self.net_D, device_ids=devices)
        if torch.cuda.is_available():
            print("Using %d GPUs." % len(devices))
            self.net_G.cuda()
            self.net_D.cuda()

        # setup optimizer
        if 'LSGAN' in self.model:
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=self.lr_adam, betas=(self.beta1, self.beta2))
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.lr_adam, betas=(self.beta1, self.beta2))
        elif 'WGAN' in self.model:
            self.optimizer_G = torch.optim.RMSprop(self.net_G.parameters(), lr=self.lr_rmsprop)
            self.optimizer_D = torch.optim.RMSprop(self.net_D.parameters(), lr=self.lr_rmsprop)

        # initialize loss lists
        self.loss_G_GANs = []
        self.loss_G_L1s = []
        self.loss_D_reals = []
        self.loss_D_fakes = []

        # save generated images
        self.out_dir = opt.out_dir + self.model + '/images/'
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.out_loss = opt.out_dir + self.model + '/losses/'
        if not os.path.exists(self.out_loss):
            os.makedirs(self.out_loss)

        print("initializing completed:\n model name: %s\n input_nc: %s\n use_sigmoid: %s\n" % (self.model, self.input_nc, self.use_sigmoid))
        print(self.net_G)
        print(self.net_D)

    def set_input(self, input):
        if 'NFG' in self.model:
            input_A = input['source']
            input_tgt = input['target']
        elif 'EFG' in self.model:
            input_A = input[0]['source']
            input_tgt = input[0]['target']
            label = self.label_generate(input[1][0], input_A.size(0))
            self.expres_code = input[1]
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
        expres_code = Variable(self.expres_code)

        out_D_real = self.net_D(self.real_tgt)
        out_D_fake = self.net_D(self.fake_tgt)

        D_real = out_D_real[:,0]
        D_fake = out_D_fake[:,0]

        self.loss_D_real = -torch.mean(D_real)
        self.loss_D_fake = torch.mean(D_fake)

        self.loss_D_entro_real = self.criterionCrossEnt(out_D_real[:,1:], expres_code)
        self.loss_D_entro_fake = self.criterionCrossEnt(out_D_fake[:,1:], expres_code)

        self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_D_entro_real+ self.loss_D_entro_fake

        

        # using AC-GAN

        """
        if 'WGAN' in self.model:
            self.loss_D_real = -torch.mean(D_real)
            self.loss_D_fake = torch.mean(D_fake)
            self.loss_D = self.loss_D_real + self.loss_D_fake
        else:
            self.loss_D_real = self.criterionGAN(D_real, Variable(self.Tensor(D_real.size()).fill_(1.0), requires_grad=False))
            self.loss_D_fake = self.criterionGAN(D_fake, Variable(self.Tensor(D_real.size()).fill_(0.0), requires_grad=False))
            self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        """

        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        D_fake = self.net_D(self.fake_tgt)
        D_real = self.net_D(self.real_tgt)
        self.loss_G_L1 = self.criterionL1(self.fake_tgt, self.real_tgt)
        if 'WGAN' in self.model:
            self.loss_G_GAN = -torch.mean(D_fake)
        else:
            self.loss_G_GAN = self.criterionGAN(D_fake, Variable(self.Tensor(D_real.size()).fill_(1.0), requires_grad=False))

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.lam_l1
        #self.loss_G = self.loss_G_L1 * self.lam_l1
        self.loss_G.backward()

    def optimize(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        if 'WGAN' in self.model:
            for p in self.net_D.parameters():
                p.data.clamp_(-0.01, 0.01)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def save_loss(self):
        self.loss_G_GANs.append(self.loss_G_GAN.cpu().data.numpy())
        self.loss_G_L1s.append(self.loss_G_L1.cpu().data.numpy())
        self.loss_D_reals.append(self.loss_D_real.cpu().data.numpy())
        self.loss_D_fakes.append(self.loss_D_fake.cpu().data.numpy())

    def save(self, label):
        #self.save_network(self.net_G, 'G', label)
        #self.save_network(self.net_D, 'D', label)

        img_A = ((np.transpose(self.real_A.cpu().data.numpy(), (0, 2, 3, 1)) + 1) / 2.0 * 255.0).astype(np.uint8)
        img_tgt = ((np.transpose(self.real_tgt.cpu().data.numpy(), (0, 2, 3, 1)) + 1) / 2.0 * 255.0).astype(np.uint8)
        fake_numpy = self.fake_tgt.cpu().data.numpy()
        img_fake = ((np.transpose(fake_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0).astype(np.uint8)
        for i in range(self.real_A.size()[0]):
            Image.fromarray(img_A[i]).save(self.out_dir + str(label) + '_' + str(i) + '_source.jpg')
            Image.fromarray(img_tgt[i]).save(self.out_dir + str(label) + '_'+ str(i) + '_tgt.jpg')
            if 'EFG' in self.model:
                if self.expres_code[0] == 0:
                    Image.fromarray(img_fake[i]).save(self.out_dir + str(label) + '_' + str(i) + '_fake_smile.jpg')
                elif self.expres_code[0] == 1:
                    Image.fromarray(img_fake[i]).save(self.out_dir + str(label) + '_' + str(i) + '_fake_anger.jpg')
                else:
                    Image.fromarray(img_fake[i]).save(self.out_dir + str(label) + '_' + str(i) + '_fake_scream.jpg')
            else:
                Image.fromarray(img_fake[i]).save(self.out_dir + str(label) + '_' + str(i) + '_fake_tgt.jpg')
                """
                print('mean: ', np.mean(img_fake[i][0]), np.mean(img_fake[i][1]), np.mean(img_fake[i][2]))
                print('min: ', np.min(img_fake[i][0]), np.min(img_fake[i][1]), np.min(img_fake[i][2]))
                print('max', np.max(img_fake[i][0]), np.max(img_fake[i][1]), np.max(img_fake[i][2]))
                print('std', np.std(img_fake[i][0]), np.std(img_fake[i][1]), np.std(img_fake[i][2]))
                raw_input("Press Enter to continue...")
                """
        # save loss plt
        length = len(self.loss_D_reals)
        x = np.arange(length)
        losses = [self.loss_D_reals, self.loss_D_fakes, self.loss_G_GANs, self.loss_G_L1s]
        labels = ['loss_D_real', 'loss_D_fake', 'loss_G_GAN', 'loss_G_L1']
        for i in range(4):
            plt.plot(x, losses[i], label=labels[i])
        plt.legend()
        plt.savefig(self.out_loss + '_loss.jpg')
        plt.close()

    def print_current_loss(self):
        print('loss_D_real: %f, loss_D_fake: %f, loss_G_GAN: %f, loss_G_L1: %f' % (self.loss_D_real.data[0], self.loss_D_fake.data[0], self.loss_G_GAN.data[0], self.loss_G_L1.data[0]))
