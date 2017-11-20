from __future__ import print_function
import torch
import torch.nn as nn
import functools
from network_EFG import *
import os
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
from PIL import Image
from load_data import *

class ModelEFG(object):
    def __init__(self, opt):
        self.model = opt.model
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.nfg = opt.nfg
        if self.nfg == 128:
            self.num_downs = 7
        elif self.nfg == 64:
            self.num_downs = 7
        else:
            raise ValueError("Only support nfg = 128 or 64. Got %d" % self.nfg)
        # configuration of network
        self.no_dropout = opt.no_dropout
        self.use_sigmoid = False
        self.batch_size = opt.batch_size
        self.norm = functools.partial(nn.BatchNorm2d, affine=True)
        self.lr = opt.learning_rate
        self.lam = opt.lam
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2

        # intializer network
        self.encoder_G = Encoder(self.input_nc, nfg=self.nfg, num_downs=self.num_downs, norm_layer=self.norm)
        self.decoder_G = Decoder(self.input_nc, nfg=self.nfg, num_downs=self.num_downs, norm_layer=self.norm, use_dropout=self.no_dropout)
        self.net_D = NLayerDiscriminator(self.input_nc, norm_layer=self.norm, use_sigmoid=self.use_sigmoid)
        if torch.cuda.device_count() >= 1:
            print("Using %d GPUs." % torch.cuda.device_count())
            self.encoder_G = nn.DataParallel(self.encoder_G)
            self.decoder_G = nn.DataParallel(self.decoder_G)
            self.net_D = nn.DataParallel(self.net_D)
        if torch.cuda.is_available():
            self.encoder_G.cuda()
            self.decoder_G.cuda()
            self.net_D.cuda()

        # setup optimizer
        if "LSGAN" in self.model:
            self.optimizer_G = torch.optim.Adam(list(self.encoder_G.parameters()) + list(self.decoder_G.parameters()), lr=self.lr, betas=(self.beta1, self.beta2))
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        elif "WGAN" in self.model:
            self.optimizer_G = torch.optim.RMSprop(list(self.encoder_G.parameters()) + list(self.decoder_G.parameters()), lr=self.lr)
            self.optimizer_D = torch.optim.RMSprop(self.net_D.parameters(), lr=self.lr)
        else:
            raise ValueError("%s model not supported." % self.model)
        
        # setup input dataset
        if "E2E" not in self.model:
            self.transformed_dataset = EFGDataset(mode='training',
                    transform=transforms.Compose([AugmentImage(),
                        ToTensor(),
                        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
            self.transformed_dataset_test = EFGDataset(mode='testing',
                    transform=transforms.Compose([AugmentImage(),
                        ToTensor(),
                        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
        else:
            self.transformed_dataset = EFGDataset(mode='training',
                    end_to_end=True,
                    transform=transforms.Compose([AugmentImage(),
                        ToTensor(),
                        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
            self.transformed_dataset_test = EFGDataset(mode='testing',
                    end_to_end=True,
                    transform=transforms.Compose([AugmentImage(),
                        ToTensor(),
                        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))

        self.data_loader = torch.utils.data.DataLoader(self.transformed_dataset, batch_size=self.batch_size, shuffle=False)
        self.data_loader_test = torch.utils.data.DataLoader(self.transformed_dataset_test, batch_size=self.batch_size, shuffle=False)
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
            fake_numpy = fake[0].data.numpy()
            img_fake = ((np.transpose(fake_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            if expression_label == 0: 
                Image.fromarray(img_fake).save(self.out_dir + str(epoch) + '_' + str(i) + 'fake_smile' + '.jpg')
            elif expression_label == 1:
                Image.fromarray(img_fake).save(self.out_dir + str(epoch) + '_' + str(i) + 'fake_anger' + '.jpg')
            else:
                Image.fromarray(img_fake).save(self.out_dir + str(epoch) + '_' + str(i) + 'fake_scream' + '.jpg')

            img_A = ((np.transpose(test_A[0].data.numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            Image.fromarray(img_A).save(self.out_dir + str(epoch) + '_' + str(i) + 'A' + '.jpg')

    def train(self):
        criterion = nn.MSELoss()
        criterionL1 = nn.L1Loss()
        real_label = 1.0
        fake_label = 0.0

        epoch = 100
        for e in range(epoch):
            print("training epoch: %d" % e)
            for i, data in enumerate(self.data_loader):
                # set input of network
                if torch.cuda.is_available():
                    input_A = Variable(data[0]['source'].cuda())
                    input_Tgt = Variable(data[0]['target'].cuda())
                    expression_label = data[1][0]
                    if expression_label == 0:
                        v = Variable(torch.Tensor([[1,0,0]]).cuda())
                    elif expression_label == 1:
                        v = Variable(torch.Tensor([[0,1,0]]).cuda())
                    else:
                        v = Variable(torch.Tensor([[0,0,1]]).cuda())
                else:
                    input_A = Variable(data[0]['source'])
                    input_Tgt = Variable(data[0]['target'])
                    expression_label = data[1][0]
                    if expression_label == 0:
                        v = Variable(torch.Tensor([[1,0,0]]))
                    elif expression_label == 1:
                        v = Variable(torch.Tensor([[0,1,0]]))
                    else:
                        v = Variable(torch.Tensor([[0,0,1]]))

                fake_inter = self.encoder_G(input_A)
                fake_inter = fake_inter.view(1, -1)
                fake_img = self.decoder_G(torch.cat((fake_inter, v), 1))
                D_real = self.net_D(input_A)
                D_fake = self.net_D(fake_img)
                if torch.cuda.is_available():
                    real_tensor = Variable(torch.Tensor(D_real.size()).fill_(real_label).cuda(), requires_grad=False)
                    fake_tensor = Variable(torch.Tensor(D_fake.size()).fill_(fake_label).cuda(), requires_grad=False)
                else:
                    real_tensor = Variable(torch.Tensor(D_real.size()).fill_(real_label), requires_grad=False)
                    fake_tensor = Variable(torch.Tensor(D_fake.size()).fill_(fake_label), requires_grad=False)
                # loss of D
                if "WGAN" in self.model:
                    loss_D_real = torch.mean(D_real)
                    loss_D_fake = torch.mean(D_fake)
                    loss_D = -(loss_D_real - loss_D_fake)
                else:
                    loss_D_real = criterion(D_real, real_tensor)
                    loss_D_fake = criterion(D_fake, real_tensor)
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                # backward of D
                self.optimizer_D.zero_grad()
                loss_D.backward()
                self.optimizer_D.step()
                # loss of G
                loss_G_L1 = criterionL1(fake_img, input_Tgt)
                if "WGAN" in self.model:
                    loss_G_GAN= -torch.mean(D_fake)
                else:
                    loss_G_GAN= criterion(D_fake, real_tensor)
                loss_G = loss_G_GAN + loss_G_L1 * self.lam

                # backward of G
                if i%5 != 0:
                    if "WGAN" in self.model:
                        continue
                    else:
                        self.optimizer_G.zero_grad()
                        loss_G.backward()
                        self.optimizer_G.step()
                if "WGAN" in self.model:
                    for p in self.net_D.parameters():
                        p.data.clamp_(-0.01, 0.01)
                print('epoch: %d, it: %d, loss_G: %f, loss_D: %f' % (e, i, loss_G.data[0], loss_D.data[0]))
            if e%5 == 0:
                self.save_img(e)

    def __call__(self):
        self.train()
        print('training complete')
