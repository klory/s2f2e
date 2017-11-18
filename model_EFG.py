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

class ModelNFG(object):
    def __init__(self, opt):
        self.model = opt.model
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.ngf = opt.ngf
        if ngf == 128:
            self.num_downs = 7
        elif ngf == 64:
            self.num_downs = 7
        else:
            raise ValueError("Only support ngf = 128 or 64. Got %d" % ngf)
        self.no_dropout = opt.no_dropout
        self.use_sigmoid = opt.use_sigmoid
        self.batch_size = opt.batch_size
        self.norm = functools.partial(nn.BatchNorm2d, affine=True)
        self.lr = self.learning_rate
        self.lam = opt.lam
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2

        self.encoder_G = Encoder(self.input_nc, ngf=self.ngf, num_downs=self.num_downs, norm_layer=self.norm)
        self.decoder_G = Decoder(self.input_nc, ngf=self.ngf, num_downs=self.num_downs, norm_layer=self.norm, use_dropout=self.no_dropout)
        self.net_D = NLayerDiscriminator(self.input_nc, norm_layer=self.norm, use_sigmoid=self.use_sigmoid)
        if self.model == "EFG":
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

        self.data_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False)
        self.data_loader_test = torch.utils.data.DataLoader(transformed_dataset_test, batch_size=batch_size, shuffle=False)
        # save generated images
        self.out_dir = opt.out_dir + '/expression/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def save_img(self, epoch):
        num_test = 2
        for i, data in enumerate(self.data_loader_test):
            if i > num_test:
                break
            # test_A: neutral face image
            test_A = Variable(data[0]['source'])
            # obtain the bottle-neck of encoder
            fake_inter = self.encoder_G(test_A)
            # reshape tensor for linear layer in decoder
            fake_inter = fake_inter.view(1, -1)

            # expression_label, 0: smile, 1: anger, 2: scream
            # v_smile, v_anger, v_scream: one-hot expression code
            # stack bottle_neck layer with one-hot key, then feed into decoder to get fake images
            expression_label = data[1][0]
            if expression_label == 0: 
                test_smile = Variable(data[0]['target'])
                v_smile = Variable(torch.Tensor([[1,0,0]]))
                fake_smile = self.decoder_G(torch.cat((fake_inter, v_smile), 1))
                fake_smile_numpy = fake_smile[0].data.numpy()
                img_fake_smile = ((np.transpose(fake_smile_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
                Image.fromarray(img_fake_smile).save(self.out_dir + str(epoch) + '_' + str(i) + 'fake_smile' + '.jpg')

            elif expression_label == 1:
                test_anger = Variable(data[0]['target'])
                v_anger = Variable(torch.Tensor([[0,1,0]]))
                fake_anger = self.decoder_G(torch.cat((fake_inter, v_anger), 1))
                fake_anger_numpy = fake_anger[0].data.numpy()
                img_fake_anger = ((np.transpose(fake_anger_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
                Image.fromarray(img_fake_anger).save(self.out_dir + str(epoch) + '_' + str(i) + 'fake_anger' + '.jpg')

            else:
                test_scream = Variable(data[0]['target'])
                v_scream = Variable(torch.Tensor([[0,0,1]]))
                fake_scream = self.decoder_G(torch.cat((fake_inter, v_scream), 1))
                fake_scream_numpy = fake_scream[0].data.numpy()
                img_fake_scream = ((np.transpose(fake_scream_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
                Image.fromarray(img_fake_scream).save(self.out_dir + str(epoch) + '_' + str(i) + 'fake_scream' + '.jpg')

            img_A = ((np.transpose(test_A[0].data.numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            Image.fromarray(img_A).save(self.out_dir + str(epoch) + '_' + str(i) + 'A' + '.jpg')

            # loss criterion
    def train(self):
        criterion = nn.MSELoss()
        criterionL1 = nn.L1Loss()
        real_label = 1.0
        fake_label = 0.0
        if self.model == "EFG":
            optimizer_G = torch.optim.Adam(list(self.encoder_G.parameters()) + list(self.decoder_G.parameters()), lr=lr, betas=(beta1, beta2))
            optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=lr, betas=(beta1, beta2))
        elif self.model == "EFG_WGAN" or self.model == "E2E":
            optimizer_G = torch.optim.RMSProp(list(self.encoder_G.parameters()) + list(self.decoder_G.parameters()), lr=self.lr)
            optimizer_D = torch.optim.RMSProp(self.net_D.parameters(), lr=self.lr)
        else:
            raise ValueError("%s model not supported." % self.model)

        epoch = 100
        for e in range(epoch):
            print("training epoch: %d" % e)
            for i, data in enumerate(self.data_loader):
                # set input of network
                input_A = Variable(data[0]['source'])
                input_Tgt = Variable(data[0]['target'])
                expression_label = data[1][0]
                if expression_label == 0:
                    v = Variable(torch.Tensor([[1,0,0]]))
                elif expression_label == 1:
                    v = Variable(torch.Tensor([[0,1,0]]))
                else:
                    v = Variable(torch.Tensor([[0,0,1]]))

                # obtain the bottle-neck layer
                fake_inter = self.encoder_G(input_A)
                # reshape into 2d tensor
                fake_inter = fake_inter.view(1, -1)
                # one-hot expression code
                fake_img = self.decoder_G(torch.cat((fake_inter, v), 1))
                # forward of discriminator
                D_real = self.net_D(input)
                D_fake = self.net_D(fake_img)
                # labels for loss function
                real_tensor = Variable(torch.Tensor(D_real.size()).fill_(real_label), requires_grad=False)
                fake_tensor = Variable(torch.Tensor(D_fake.size()).fill_(fake_label), requires_grad=False)
                # loss of D
                loss_D = criterion(D_real, real_tensor)
                loss_D_fake = criterion(D_fake, fake_tensor)
                loss_D = (loss_D + loss_D_fake) * 0.5
                # backward of D
                optimizer_D.zero_grad()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
                # loss GAN_G
                loss_G_GAN= criterion(D_fake, real_tensor)
                # L1 loss of G
                loss_G_L1 = criterionL1(fake_img, input_Tgt)
                loss_G = loss_G_GAN + loss_G_L1 * lam
                # backward of G
                if i%5 != 0:
                    if self.model == "WGAN" or self.model == "E2E":
                        continue
                    else:
                        optimizer_G.zero_grad()
                        loss_G.backward()
                        optimizer_G.step()
                if self.model == "EFG_WGAN" or self.model == "E2E":
                    p = self.net_D.parameters().clamp_(-0.01, 0.01)


            print('epoch: %d, it: %d, loss_G: %f, loss_D: %f' % (e, i, loss_G.data[0], loss_D.data[0]))
            if e%5 == 0:
                self.save_img(e)

    def __call__(self, opt):
        train()
        print('training complete')
