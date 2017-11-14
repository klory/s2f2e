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

input_nc = 3 # number of channels of input(image)
output_nc = 3 # number of channels of output(image)
ngf = 64 # number of filters of the first conv layers of generator
no_dropout = True
norm = functools.partial(nn.BatchNorm2d, affine=True) # normalize function to use
num_downs = 6 # downsampling times
net_G = Unet_G(input_nc, output_nc, num_downs, ngf, norm_layer=norm, use_dropout=not no_dropout)
use_sigmoid=True

net_D = NLayerDiscriminator(input_nc, norm_layer=norm, use_sigmoid=use_sigmoid)

batch_size = 1

# preprocess data
transformed_dataset = NFGDataset(mode='training',
        transform=transforms.Compose([
            AugmentImage(),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]))
data_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False)
                                     
print 'complete data preprocess, mean and var of first pic: ', np.mean(transformed_dataset[0]['source'].numpy()), np.var(transformed_dataset[0]['target'].numpy())


lr = 0.0002
# betas for Adam
beta1 = 0.5
beta2 = 0.999
optimizer_G = torch.optim.Adam(net_G.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr, betas=(beta1, beta2))

# dir to save test image
out_dir = '../out/images/neutral/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
num_test = 2
def save_img(epoch):
    for i, data in enumerate(data_loader):
        if i > num_test:
            break
        test_A = Variable(data['source'])
        test_B = data['target']
        fake_B = net_G(test_A)
        fake_B_numpy = fake_B[0].data.numpy()
        img_fake_B = ((np.transpose(fake_B_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
        Image.fromarray(img_fake_B).save(out_dir + str(epoch) + '_' + str(i) + 'fake_B' + '.jpg')
        img_B = ((np.transpose(test_B[0].numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
        img_A = ((np.transpose(test_A[0].data.numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
        Image.fromarray(img_B).save(out_dir + str(epoch) + '_' + str(i) + 'B' + '.jpg')
        Image.fromarray(img_A).save(out_dir + str(epoch) + '_' + str(i) + 'A' + '.jpg')


# TODO:configure criterion and optimizer for net_G and net_D
criterion = nn.MSELoss()
criterionL1 = nn.L1Loss()
lam = 15 # lambda to balance GAN loss and L1 loss of generator
real_label = 1.0 # labels for real image, used in loss api
fake_label = 0.0 # labels for fake image, used in loss api

epoch = 30
for e in range(epoch):
    print "training epoch: %d" % e
    for i, data in enumerate(data_loader):
        print 'iteration %d' % i
        input_A = Variable(data['source'])
        input_B = Variable(data['target'])
        # forward of G
        fake_B = net_G(input_A)
        #forward of D
        real_logits = net_D(input_B)
        fake_logits = net_D(fake_B)
        real_tensor = Variable(torch.Tensor(real_logits.size()).fill_(real_label), requires_grad=False)
        fake_tensor = Variable(torch.Tensor(fake_logits.size()).fill_(fake_label), requires_grad=False)
        # compute loss of D
        loss_D_real = criterion(real_logits, real_tensor)
        loss_D_fake = criterion(fake_logits, fake_tensor)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward of D
        optimizer_D.zero_grad()
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        # compute loss of G
        loss_G_GAN = criterion(fake_logits, real_tensor)
        loss_G_L1 = criterionL1(fake_B, input_B) * lam
        loss_G = loss_G_GAN + loss_G_L1
        # backward of G
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        print 'epoch: %d, it: %d, G_GAN: %f\t, G_L1: %f\t, D_real: %f\t, D_fake: %f\t' % (e, i, loss_G_GAN.data[0], loss_G_L1.data[0], loss_D_real.data[0], loss_D_fake.data[0])
    save_img(e)


print 'testing complete'
