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
import random
import torch.autograd as autograd

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
            self.lr_ls = opt.learning_rate_ls
            self.lr_wgan = opt.learning_rate_wgan

            self.lam_cyc = opt.lam_cyc
            self.lam_l1 = opt.lam_l1
            self.lam_gp = opt.lam_gp

            self.beta1_ls = opt.beta1_ls
            self.beta2_ls = opt.beta2_ls
            self.beta1_wgan = opt.beta1_wgan
            self.beta2_wgan = opt.beta2_wgan
            self.criterionGAN = nn.MSELoss()
            self.criterionL1 = nn.L1Loss()
            self.criterionCrossEnt = nn.CrossEntropyLoss()

        # setup input
        self.input_A = self.Tensor(opt.batch_size, opt.input_nc, opt.img_size, opt.img_size)
        self.input_tgt = self.Tensor(opt.batch_size, opt.input_nc, opt.img_size, opt.img_size)

        if torch.cuda.is_available():
            self.expression_label = torch.cuda.LongTensor(opt.batch_size, 1)
        else:
            self.expression_label = torch.LongTensor(opt.batch_size, 1)

        if 'EFG' in self.model:
            self.expression_code = self.Tensor(opt.batch_size, 2, 2, 3)

        # intializer network
        self.net_D = NLayerDiscriminator(self.input_nc, self.batch_size, norm_layer=self.norm, use_sigmoid=self.use_sigmoid)
        self.net_D.apply(self.init_weights)
        self.net_G = Unet_G2(self.input_nc, self.output_nc, self.which_model, nfg=self.nfg, norm_layer=self.norm, use_dropout=self.dropout)
        self.net_G.apply(self.init_weights)

        devices = self.opt.gpu_ids 
        if len(devices) > 1:
            self.net_G = nn.DataParallel(self.net_G, device_ids=devices)
            self.net_D = nn.DataParallel(self.net_D, device_ids=devices)
        if torch.cuda.is_available():
            print("Using %d GPUs." % len(devices))
            self.net_G.cuda()
            self.net_D.cuda()

        # setup optimizer
        if 'LSGAN' in self.model:
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=self.lr_ls, betas=(self.beta1_ls, self.beta2_ls))
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.lr_ls, betas=(self.beta1_ls, self.beta2_ls))
        elif 'WGAN' in self.model:
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=self.lr_wgan, betas=(self.beta1_wgan, self.beta2_wgan))
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.lr_wgan, betas=(self.beta1_wgan, self.beta2_wgan))

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

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
        if classname.find('ConvTranspose') != -1:
            m.weight.data.normal_(0.0, 0.02)
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0.0)
        if classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0.0)

    def set_input(self, input):
        if 'NFG' in self.model:
            input_A = input['source']
            input_tgt = input['target']
        elif 'EFG' in self.model:
            input_A = input[0]['source']
            input_tgt = input[0]['target']
            code = self.code_generate(input[1][0], input_A.size(0))
            self.exp_label = input[1]
        else:
            raise ValueError("%s is not suppported." % self.model)
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_tgt.resize_(input_tgt.size()).copy_(input_tgt)
        if 'EFG' in self.model:
            self.expression_code.resize_(code.size()).copy_(code)
            self.expression_label.resize_(self.exp_label.size()).copy_(self.exp_label)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_tgt = Variable(self.input_tgt)
        if 'EFG' in self.model:
            self.real_code = Variable(self.expression_code)
            self.fake_tgt = self.net_G(self.real_A, self.real_code)
        else:
            self.fake_tgt = self.net_G(self.real_A, None)

    def backward_D(self):
        # using AC-GAN and gp-WGAN
        eps = random.uniform(0, 1)
        x_rand = Variable(eps * self.real_tgt.data + (1-eps) * self.fake_tgt.data, requires_grad=True)
        loss_x_rand = self.net_D(x_rand)[1]
        grad_outputs = torch.ones(loss_x_rand.size())
        grads = autograd.grad(loss_x_rand, x_rand, grad_outputs=grad_outputs.cuda() if torch.cuda.is_available() else grad_outputs, create_graph=True)[0]
        self.loss_gp = torch.mean((grads.view(-1, 3*128*128).pow(2).sum(1).sqrt() - 1).pow(2))

        exp_label = Variable(self.expression_label)
        v_real, s_real = self.net_D(self.real_tgt)
        v_fake, s_fake = self.net_D(self.fake_tgt)

        self.loss_D_real = -torch.mean(s_real)
        self.loss_D_fake = torch.mean(s_fake)

        self.loss_D_entro_real = self.criterionCrossEnt(v_real, exp_label)
        #self.loss_D_entro_fake = self.criterionCrossEnt(v_fake, exp_label)

        self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_D_entro_real + self.lam_gp * self.loss_gp
        

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
        v_D_fake, _ = self.net_D(self.fake_tgt)
        v_D_real, _ = self.net_D(self.real_tgt)
        self.loss_G_L1 = self.criterionL1(self.fake_tgt, self.real_tgt)
        if 'WGAN' in self.model:
            self.loss_G_GAN = -torch.mean(v_D_fake)
        else:
            self.loss_G_GAN = self.criterionGAN(v_D_fake, Variable(self.Tensor(v_D_real.size()).fill_(1.0), requires_grad=False))

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
                if self.expression_label[0] == 0:
                    Image.fromarray(img_fake[i]).save(self.out_dir + str(label) + '_' + str(i) + '_fake_smile.jpg')
                elif self.expression_label[0] == 1:
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
