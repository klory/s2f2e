from __future__ import print_function
import torch
import torch.nn as nn
import functools
from network_NFG import *
import os
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
from PIL import Image
from load_data import *

class ModelNFG():
    def __init__(self, opt):
        self.model = opt.model
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.nfg = opt.nfg
        if self.nfg == 128:
            self.num_downs = 7
        elif self.nfg == 64:
            self.num_downs = 6
        else:
            raise ValueError("Only support nfg = 128 or 64. Got %d" % self.nfg)
        # configuration of NFG network
        self.no_dropout = opt.no_dropout
        self.use_sigmoid = False
        self.batch_size = opt.batch_size
        self.norm = functools.partial(nn.BatchNorm2d, affine=True)
        self.lr = opt.learning_rate
        self.lam = opt.lam
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2

        # initialize NFG network
        self.net_G = Unet_G(self.input_nc, self.output_nc, self.num_downs, self.nfg, norm_layer=self.norm, use_dropout=not self.no_dropout)
        self.net_D = NLayerDiscriminator(self.input_nc, norm_layer=self.norm, use_sigmoid=self.use_sigmoid)
        if torch.cuda.device_count() > 1:
            self.net_G = nn.DataParallel(self.net_G)
            self.net_D = nn.DataParallel(self.net_D)
        if torch.cuda.is_available():
            print("Using %d GPUs." % torch.cuda.device_count())
            self.net_G.cuda()
            self.net_D.cuda()

        # set up optimizer
        if "LSGAN" in self.model:
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        elif "WGAN" in self.model:
            self.optimizer_G = torch.optim.RMSprop(self.net_G.parameters(), lr=self.lr)
            self.optimizer_D = torch.optim.RMSprop(self.net_D.parameters(), lr=self.lr)
        else:
            raise ValueError("%s not supported." % self.optimizer)
        # set up input dataset
        transformed_dataset = NFGDataset(mode='training',
                transform=transforms.Compose([
                    AugmentImage(),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ]))
        self.data_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=self.batch_size, shuffle=False)
        transformed_dataset_test = NFGDataset(mode='testing',
                transform=transforms.Compose([
                    AugmentImage(),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ]))
        self.data_loader_test = torch.utils.data.DataLoader(transformed_dataset_test, batch_size=self.batch_size, shuffle=False)
        # save generated images
        self.out_dir = opt.out_dir + '/expression/'
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        print("initializing completed:\n model name: %s\n input_nc: %s\n use_sigmoid: %s\n" % (self.model, self.input_nc, self.use_sigmoid))

    def save_img(self, epoch):
        num_test = 2
        for i, data in enumerate(self.data_loader_test):
            if i > num_test:
                break
            if torch.cuda.is_available():
                test_A = Variable(data['source'].cuda())
                test_B = data['target'].cuda()
            else:
                test_A = Variable(data['source'])
                test_B = data['target']

            fake_B = self.net_G(test_A)
            fake_B_numpy = fake_B[0].data.numpy()
            img_fake_B = ((np.transpose(fake_B_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            Image.fromarray(img_fake_B).save(self.out_dir + str(epoch) + '_' + str(i) + 'fake_B' + '.jpg')
            img_B = ((np.transpose(test_B[0].numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            img_A = ((np.transpose(test_A[0].data.numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            Image.fromarray(img_B).save(self.out_dir + str(epoch) + '_' + str(i) + 'B' + '.jpg')
            Image.fromarray(img_A).save(self.out_dir + str(epoch) + '_' + str(i) + 'A' + '.jpg')


# TODO:configure criterion and optimizer for net_G and net_D
    def train(self):
        criterion = nn.MSELoss()
        criterionL1 = nn.L1Loss()
        real_label = 1.0 # labels for real image, used in loss api
        fake_label = 0.0 # labels for fake image, used in loss api


        epoch = 30
        for e in range(epoch):
            print("training epoch: %d" % e)
            for i, data in enumerate(self.data_loader):
                print('iteration %d' % i)
                if torch.cuda.is_available():
                    input_A = Variable(data['source'].cuda())
                    input_B = Variable(data['target'].cuda())
                else:
                    input_A = Variable(data['source'])
                    input_B = Variable(data['target'])
                # forward of G
                fake_B = self.net_G(input_A)
                #forward of D
                D_real = self.net_D(input_B)
                D_fake = self.net_D(fake_B)
                if torch.cuda.is_available():
                    real_tensor = Variable(torch.Tensor(D_real.size()).fill_(real_label).cuda(), requires_grad=False)
                    fake_tensor = Variable(torch.Tensor(D_fake.size()).fill_(fake_label).cuda(), requires_grad=False)
                else:
                    real_tensor = Variable(torch.Tensor(D_real.size()).fill_(real_label), requires_grad=False)
                    fake_tensor = Variable(torch.Tensor(D_fake.size()).fill_(fake_label), requires_grad=False)
                # compute loss of D
                if "WGAN" in self.model:
                    loss_D_real = torch.mean(D_real)
                    loss_D_fake = torch.mean(D_fake)
                    loss_D = -(loss_D_real - loss_D_fake)
                else:
                    loss_D_real = criterion(D_real, real_tensor)
                    loss_D_fake = criterion(D_fake, fake_tensor)
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                # backward of D
                self.optimizer_D.zero_grad()
                loss_D.backward()
                self.optimizer_D.step()

                # compute loss of G
                loss_G_L1 = criterionL1(fake_B, input_B) * self.lam
                if "WGAN" in self.model:
                    loss_G_GAN = -torch.mean(D_fake)
                else:
                    loss_G_GAN = criterion(D_fake, real_tensor)
                loss_G = loss_G_GAN + loss_G_L1 * self.lam
                # backward of G
                if i%5 != 0:
                    if self.model == "NFG_WGAN":
                        continue
                    else:
                        self.optimizer_G.zero_grad()
                        loss_G.backward()
                        self.optimizer_G.step()

                if self.model == "NFG_WGAN":
                    for p in self.net_D.parameters():
                        p.data.clamp_(-0.01, 0.01)

                print('epoch: %d, it: %d, G_GAN: %f\t, G_L1: %f\t, D_real: %f\t, D_fake: %f\t' % (e, i, loss_G_GAN.data[0], loss_G_L1.data[0], loss_D_real.data[0], loss_D_fake.data[0]))
        if e%5 == 0:
            save_img(e)

    def __call__(self):
        self.train()
        print('testing complete')
