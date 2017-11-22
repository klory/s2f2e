from __future__ import print_function
from base_model import BaseModel
from network import *
import functools
import numpy as np
from PIL import Image
from load_data import *
from torch.autograd import Variable
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os



class Cycle(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        
        self.model = opt.model
        if 'EFG' in self.model:
            self.which_model = 'EFG'
        elif 'NFG' in self.model:
            self.which_model = 'NFG'
        else:
            raise ValueError("model %s is not supported." % opt.model)
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.nfg = opt.nfg
        self.batch_size = opt.batch_size
        self.lam_cyc = opt.lam_cyc
        self.lam_idt = opt.lam_idt
        if self.nfg == 128:
            self.num_downs = 7
        elif self.nfg == 64:
            self.num_downs = 6
        else:
            raise ValueError("Only support nfg = 128 or 64. Got %d" % self.nfg)
        # configuration of NFG network
        if opt.isTrain:
            self.no_dropout = opt.no_dropout
            self.use_sigmoid = False
            self.norm = functools.partial(nn.BatchNorm2d, affine=True)
            self.lr = opt.learning_rate
            self.beta1 = opt.beta1
            self.beta2 = opt.beta2
            self.criterionGAN = nn.MSELoss()
            self.criterionL1 = nn.L1Loss()

        # setup input
        self.input_A = self.Tensor(opt.batch_size, opt.input_nc, opt.img_size, opt.img_size)
        self.input_B = self.Tensor(opt.batch_size, opt.input_nc, opt.img_size, opt.img_size)
        if 'EFG' in self.model:
            self.expression_label = self.Tensor(opt.batch_size, 3, 8, 8)
        # build up network
        self.net_G_AtoB = Unet_G(self.input_nc, self.output_nc, self.which_model, self.nfg, norm_layer=self.norm, use_dropout=not self.no_dropout)
        self.net_G_BtoA = Unet_G(self.input_nc, self.output_nc, self.which_model, self.nfg, norm_layer=self.norm, use_dropout=not self.no_dropout)

        self.net_D_A = NLayerDiscriminator(self.input_nc, norm_layer=self.norm, use_sigmoid=self.use_sigmoid)
        self.net_D_B = NLayerDiscriminator(self.input_nc, norm_layer=self.norm, use_sigmoid=self.use_sigmoid)

        if torch.cuda.device_count() > 1:
            self.net_G_AtoB = nn.DataParallel(self.net_G_AtoB)
            self.net_G_BtoA = nn.DataParallel(self.net_G_BtoA)
            self.net_D_A = nn.DataParallel(self.net_D_A)
            self.net_D_B = nn.DataParallel(self.net_D_B)
        if torch.cuda.is_available():
            print("Using %d GPUS." % torch.cuda.device_count())
            self.net_G_AtoB.cuda()
            self.net_G_BtoA.cuda()
            self.net_D_A.cuda()
            self.net_D_B.cuda()
        # set up optimizer
        if 'LSGAN' in self.model:
            self.optimizer_G = torch.optim.Adam(list(self.net_G_AtoB.parameters()) + list(self.net_G_BtoA.parameters()), lr=self.lr, betas=(self.beta1, self.beta2))
            self.optimizer_D_A = torch.optim.Adam(self.net_D_A.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
            self.optimizer_D_B = torch.optim.Adam(self.net_D_B.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        elif 'WGAN' in self.model:
            self.optimizer_G = torch.optim.RMSprop(list(self.net_G_AtoB.parameters()) + list(self.net_G_BtoA.parameters()), lr=self.lr)
            self.optimizer_D_A = torch.optim.RMSprop(self.net_D_A.parameters(), lr=self.lr)
            self.optimizer_D_B = torch.optim.RMSprop(self.net_D_B.parameters(), lr=self.lr)
        else:
            raise ValueError('%s is not supported.' % self.model)
        # save generated images
        self.out_dir = opt.out_dir + self.model + '/images/'
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.out_loss = opt.out_dir + self.model + '/losses/'
        if not os.path.exists(self.out_loss):
            os.makedirs(self.out_loss)

        # initialize loss lists
        self.loss_G_AGANs = []
        self.loss_G_BGANs = []
        self.loss_cyc_As = []
        self.loss_cyc_Bs = []
        self.loss_D_As = []
        self.loss_D_Bs = []

        print("initializing completed:\n model name: %s\n input_nc: %s\n use_sigmoid: %s\n" % (self.model, self.input_nc, self.use_sigmoid))

    def set_input(self, input):
        if 'NFG' in self.model:
            input_A = input['source']
            input_B = input['target']
        elif 'EFG' in self.model:
            input_A = input[0]['source']
            input_B = input[0]['target']
            label = self.label_generate(input[1][0], self.batch_size)
            self.expres_code = input[1][0]
        else:
            raise ValueError("%s is not suppported." % self.model)
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        if 'EFG' in self.model:
            self.expression_label.resize_(label.size()).copy_(label)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        if 'EFG' in self.model:
            self.real_label = Variable(self.expression_label)

    def backward_D(self, netD, real, fake):
        D_real = netD(real)
        D_fake = netD(fake) 

        if 'LSGAN' in self.model:
            self.loss_D_real = self.criterionGAN(D_real, Variable(self.Tensor(D_real.size()).fill_(1.0), requires_grad=False))
            self.loss_D_fake = self.criterionGAN(D_fake, Variable(self.Tensor(D_real.size()).fill_(0.0), requires_grad=False))

            self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        else:
            self.loss_D_real = torch.mean(D_real)
            self.loss_D_fake = torch.mean(D_fake)
            self.loss_D = -(self.loss_D_real -self.loss_D_fake)

        self.loss_D.backward()
        return self.loss_D

    def backward_D_A(self):
        if 'NFG' in self.model:
            fake_B = self.net_G_AtoB(self.real_A, None)
        else:
            fake_B = self.net_G_AtoB(self.real_A, self.real_label)
        # Note: this part is different from the original code. We follow paper "DY encouragesGtotranslateXintooutputsindistinguishablefromdomainY,andviceversa for DX and F"
        loss_D_A = self.backward_D(self.net_D_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        if 'NFG' in self.model:
            fake_A = self.net_G_BtoA(self.real_B, None)
        else:
            fake_A = self.net_G_BtoA(self.real_B, self.real_label)
        loss_D_B = self.backward_D(self.net_D_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.data[0]

    def backward_G(self):
        if 'NFG' in self.model:
            fake_B = self.net_G_AtoB(self.real_A, None)
            fake_A = self.net_G_BtoA(self.real_B, None)
            cyc_A = self.net_G_BtoA(fake_B, None)
            cyc_B = self.net_G_AtoB(fake_A, None)
        else:
            fake_B = self.net_G_AtoB(self.real_A, self.real_label)
            fake_A = self.net_G_BtoA(self.real_B, self.real_label)
            cyc_A = self.net_G_BtoA(fake_B, self.real_label)
            cyc_B = self.net_G_AtoB(fake_A, self.real_label)


        loss_cyc_A = self.criterionL1(cyc_A, self.real_A)
        loss_cyc_B = self.criterionL1(cyc_B, self.real_B)

        D_fake_B = self.net_D_A(fake_B)
        #loss_idt_A = self.criterionL1(self.real_B, fake_B)

        D_fake_A = self.net_D_B(fake_A)
        #loss_idt_B = self.criterionL1(self.real_A, fake_A)

        if 'LSGAN' in self.model:
            loss_G_AGAN = self.criterionGAN(D_fake_B, Variable(self.Tensor(D_fake_B.size()).fill_(1.0), requires_grad=False))
            loss_G_BGAN = self.criterionGAN(D_fake_A, Variable(self.Tensor(D_fake_B.size()).fill_(1.0), requires_grad=False))
        else:
            loss_G_AGAN = - torch.mean(D_fake_B)
            loss_G_BGAN = - torch.mean(D_fake_A)
        #loss_G = loss_G_A + loss_G_B + (loss_cyc_A + loss_cyc_B) * self.lam_cyc + (loss_idt_A + loss_idt_B) * self.lam_idt
        loss_G = loss_G_AGAN + loss_G_BGAN + (loss_cyc_A + loss_cyc_B) * self.lam_cyc
        loss_G.backward()

        self.fake_B = fake_B
        self.fake_A = fake_A
        self.cyc_A = cyc_A
        self.cyc_B = cyc_B

        self.loss_G_AGAN = loss_G_AGAN.data[0]
        self.loss_G_BGAN = loss_G_BGAN.data[0]
        self.loss_cyc_A = loss_cyc_A.data[0]
        self.loss_cyc_B = loss_cyc_B.data[0]
        #self.loss_idt_A = loss_idt_A.data[0]
        #self.loss_idt_B = loss_idt_B.data[0]

    def optimize(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()

        for p in self.net_D_A.parameters():
            p.data.clamp_(-0.01, 0.01)

        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

        for p in self.net_D_B.parameters():
            p.data.clamp_(-0.01, 0.01)

    def print_current_loss(self):
        print("loss_D_A: %f\t loss_D_B: %f\t loss_G_A: %f\t loss_G_A: %f\t loss_cyc_A: %f\t loss_cyc_B: %f\t" % (self.loss_D_A, self.loss_D_B, self.loss_G_AGAN, self.loss_G_BGAN, self.loss_cyc_A, self.loss_cyc_B))
    def save_loss(self):
        self.loss_D_As.append(self.loss_D_A)
        self.loss_D_Bs.append(self.loss_D_B)
        self.loss_G_AGANs.append(self.loss_G_AGAN)
        self.loss_G_BGANs.append(self.loss_G_BGAN)
        self.loss_cyc_As.append(self.loss_cyc_A)
        self.loss_cyc_Bs.append(self.loss_cyc_B)

    def save(self, label):
        # save network
        self.save_network(self.net_G_AtoB, 'G_Ato_B', label)
        self.save_network(self.net_G_BtoA, 'G_Bto_A', label)
        self.save_network(self.net_D_A, 'D_A', label)
        self.save_network(self.net_D_B, 'D_B', label)

        # save generated images
        img_A = ((np.transpose(self.real_A.cpu().data.numpy(), (0, 2, 3, 1)) + 1) / 2.0 * 255.0).astype(np.uint8)
        img_tgt = ((np.transpose(self.real_B.cpu().data.numpy(), (0, 2, 3, 1)) + 1) / 2.0 * 255.0).astype(np.uint8)
        fake_A_numpy = self.fake_A.cpu().data.numpy()
        fake_B_numpy = self.fake_B.cpu().data.numpy()
        cyc_A_numpy = self.cyc_A.cpu().data.numpy()
        cyc_B_numpy = self.cyc_B.cpu().data.numpy()
        A_fake = ((np.transpose(fake_A_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0).astype(np.uint8)
        B_fake = ((np.transpose(fake_B_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0).astype(np.uint8)
        A_cyc = ((np.transpose(cyc_A_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0).astype(np.uint8)
        B_cyc = ((np.transpose(cyc_B_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0).astype(np.uint8)
        for i in range(self.real_A.size()[0]):
            Image.fromarray(img_A[i]).save(self.out_dir + str(label) + '_' + str(i) + 'source.jpg')
            Image.fromarray(img_tgt[i]).save(self.out_dir + str(label) + '_' + str(i) + 'source.jpg')
            if 'EFG' in self.model:
                if self.expres_code == 0:
                    Image.fromarray(A_fake[i]).save(self.out_dir + str(label) + '_' + str(i) + 'fake_smile.jpg')
                    Image.fromarray(B_fake[i]).save(self.out_dir + str(label) + '_' + str(i) + 'fake_smile.jpg')
                    Image.fromarray(A_cyc[i]).save(self.out_dir + str(label) + '_' + str(i) + 'cyc_smile.jpg')
                    Image.fromarray(B_cyc[i]).save(self.out_dir + str(label) + '_' + str(i) + 'cyc_smile.jpg')
                elif self.expres_code == 1:
                    Image.fromarray(A_fake[i]).save(self.out_dir + str(label) + '_' + str(i) + 'fake_smile.jpg')
                    Image.fromarray(B_fake[i]).save(self.out_dir + str(label) + '_' + str(i) + 'fake_smile.jpg')
                    Image.fromarray(A_cyc[i]).save(self.out_dir + str(label) + '_' + str(i) + 'cyc_smile.jpg')
                    Image.fromarray(B_cyc[i]).save(self.out_dir + str(label) + '_' + str(i) + 'cyc_smile.jpg')
                else:
                    Image.fromarray(A_fake[i]).save(self.out_dir + str(label) + '_' + str(i) + 'fake_smile.jpg')
                    Image.fromarray(B_fake[i]).save(self.out_dir + str(label) + '_' + str(i) + 'fake_smile.jpg')
                    Image.fromarray(A_cyc[i]).save(self.out_dir + str(label) + '_' + str(i) + 'cyc_smile.jpg')
                    Image.fromarray(B_cyc[i]).save(self.out_dir + str(label) + '_' + str(i) + 'cyc_smile.jpg')

        # save loss plt
        length = len(self.loss_D_As)
        x = np.arange(length)
        x = np.tile(x, 6).reshape(6, -1)
        losses = [self.loss_D_As, self.loss_D_Bs, self.loss_G_AGANs, self.loss_G_BGANs, self.loss_cyc_As, self.loss_cyc_Bs]
        z = zip(x, losses)
        labels = ['loss_D_A', 'loss_D_B', 'loss_G_AGAN', 'loss_G_BGAN', 'loss_cyc_A', 'loss_cyc_B']
        for i in range(6):
            plt.plot(z[i][0], z[i][1], label=labels[i])
        plt.legend()
        plt.savefig(self.out_loss + str(label) + 'loss.jpg')
        plt.close()
