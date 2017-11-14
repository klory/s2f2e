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

input_nc = 3
output_nc = 3
ngf = 64
no_dropout = True
norm = functools.partial(nn.BatchNorm2d, affine=True)
# WGAN do not use sigmoid
use_sigmoid = False

batch_size = 1

encoder_G = Encoder(input_nc, norm_layer=norm)
decoder_G = Decoder(input_nc, norm_layer=norm, use_dropout=no_dropout)
net_D = NLayerDiscriminator(input_nc, norm_layer=norm, use_sigmoid=use_sigmoid)


transformed_dataset = EFGDataset(mode='training',
        transform=transforms.Compose([ToTensor(),
                                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))


data_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False)

transformed_dataset_test = EFGDataset(mode='testing',
        transform=transforms.Compose([ToTensor(),
                                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))


data_loader_test = torch.utils.data.DataLoader(transformed_dataset_test, batch_size=batch_size, shuffle=False)

#print 'complete data preprocess, mean of first pic: ', np.mean(dataSet[0]['A'].numpy()), np.mean(dataSet[0]['smile'].numpy()), np.mean(dataSet[0]['anger'].numpy()), np.mean(dataSet[0]['scream'].numpy())


# save generated images
out_dir = '../out/images/expression/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def save_img(epoch):
    for i, data in enumerate(data_loader_test):
        # test_A: neutral face image
        test_A = Variable(data[0]['source'])
        # obtain the bottle-neck of encoder
        fake_inter = encoder_G(test_A)
        # reshape tensor for linear layer in decoder
        fake_inter = fake_inter.view(1, -1)

        # expression_label, 0: smile, 1: anger, 2: scream
        # v_smile, v_anger, v_scream: one-hot expression code
        # stack bottle_neck layer with one-hot key, then feed into decoder to get fake images
        expression_label = data[1][0]
        if expression_label == 0: 
            test_smile = Variable(data[0]['target'])
            v_smile = Variable(torch.Tensor([[1,0,0]]))
            fake_smile = decoder_G(torch.cat((fake_inter, v_smile), 1))
            fake_smile_numpy = fake_smile[0].data.numpy()
            img_fake_smile = ((np.transpose(fake_smile_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            Image.fromarray(img_fake_smile).save(out_dir + str(epoch) + '_' + str(i) + 'fake_smile' + '.jpg')

        elif expression_label == 1:
            test_anger = Variable(data[0]['target'])
            v_anger = Variable(torch.Tensor([[0,1,0]]))
            fake_anger = decoder_G(torch.cat((fake_inter, v_anger), 1))
            fake_anger_numpy = fake_anger[0].data.numpy()
            img_fake_anger = ((np.transpose(fake_anger_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            Image.fromarray(img_fake_anger).save(out_dir + str(epoch) + '_' + str(i) + 'fake_anger' + '.jpg')

        else:
            test_scream = Variable(data[0]['target'])
            v_scream = Variable(torch.Tensor([[0,0,1]]))
            fake_scream = decoder_G(torch.cat((fake_inter, v_scream), 1))
            fake_scream_numpy = fake_scream[0].data.numpy()
            img_fake_scream = ((np.transpose(fake_scream_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            Image.fromarray(img_fake_scream).save(out_dir + str(epoch) + '_' + str(i) + 'fake_scream' + '.jpg')

        img_A = ((np.transpose(test_A[0].data.numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
        Image.fromarray(img_A).save(out_dir + str(epoch) + '_' + str(i) + 'A' + '.jpg')

# loss criterion
criterion = nn.MSELoss()
criterionL1 = nn.L1Loss()
lr = 5e-5
lam = 10
real_label = 1.0
fake_label = 0.0
beta1 = 0.5
beta2 = 0.999
optimizer_G = torch.optim.RMSprop(list(encoder_G.parameters()) + list(decoder_G.parameters()), lr=lr)
optimizer_D = torch.optim.RMSprop(net_D.parameters(), lr=lr)

num_train = 5
epoch = 100
for e in range(epoch):
    print "training epoch: %d" % e
    for i, data in enumerate(data_loader):
        if i > num_train:
            break
        # set input of network
        input_A = Variable(data[0]['source'])
        expression_label = data[1][0]

        # obtain the bottle-neck layer
        fake_inter = encoder_G(input_A)
        # reshape into 2d tensor
        fake_inter = fake_inter.view(1, -1)
        if expression_label == 0:
            input_smile = Variable(data[0]['target'])
            # one-hot expression code
            v_smile = Variable(torch.Tensor([[1,0,0]]))
            # contat expression code and bottle-neck layer from encoder
            fake_smile = decoder_G(torch.cat((fake_inter, v_smile), 1))
            # forward of discriminator
            D_real_smile = net_D(input_smile)
            D_fake_smile = net_D(fake_smile)
            # labels for loss function
            real_tensor = Variable(torch.Tensor(D_real_smile.size()).fill_(real_label), requires_grad=False)
            fake_tensor = Variable(torch.Tensor(D_fake_smile.size()).fill_(fake_label), requires_grad=False)
            # loss of D
            loss_D_smile = criterion(D_real_smile, real_tensor)
            loss_D_fake_smile = criterion(D_fake_smile, fake_tensor)
            loss_D = (loss_D_smile + loss_D_fake_smile) * 0.5
            # backward of D
            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()
            # loss GAN_G
            loss_G_GAN_smile = criterion(D_fake_smile, real_tensor)
            # L1 loss of G
            loss_G_L1_smile = criterionL1(fake_smile, input_smile)
            loss_G = loss_G_GAN_smile + loss_G_L1_smile * lam
            # backward of G
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        elif expression_label == 1:
            input_anger = Variable(data[0]['target'])
            v_anger = Variable(torch.Tensor([[0,1,0]]))
            fake_anger = decoder_G(torch.cat((fake_inter, v_anger), 1))

            D_real_anger = net_D(input_anger)
            D_fake_anger = net_D(fake_anger)
            # labels for loss function
            real_tensor = Variable(torch.Tensor(D_real_anger.size()).fill_(real_label), requires_grad=False)
            fake_tensor = Variable(torch.Tensor(D_fake_anger.size()).fill_(fake_label), requires_grad=False)
            loss_D_anger = criterion(D_real_anger, real_tensor)
            loss_D_fake_anger = criterion(D_fake_anger, fake_tensor)
            loss_D = (loss_D_anger + loss_D_fake_anger) * 0.5

            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            loss_G_GAN_anger = criterion(D_fake_anger, real_tensor)
            loss_G_L1_anger = criterionL1(fake_anger, input_anger)

            loss_G = loss_G_GAN_anger + loss_G_L1_anger * lam
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
        else:
            input_scream = Variable(data[0]['target'])
            v_scream = Variable(torch.Tensor([[0,0,1]]))
            fake_scream = decoder_G(torch.cat((fake_inter, v_scream), 1))

            D_real_scream = net_D(input_scream)
            D_fake_scream = net_D(fake_scream)
            # labels for loss function
            real_tensor = Variable(torch.Tensor(D_real_scream.size()).fill_(real_label), requires_grad=False)
            fake_tensor = Variable(torch.Tensor(D_fake_scream.size()).fill_(fake_label), requires_grad=False)
            loss_D_scream = criterion(D_real_scream, real_tensor)
            loss_D_fake_scream = criterion(D_fake_scream, fake_tensor)
            loss_D = (loss_D_scream + loss_D_fake_scream) * 0.5
            
            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            loss_G_GAN_scream = criterion(D_fake_scream, real_tensor)
            loss_G_L1_scream = criterionL1(fake_scream, input_scream)
            
            loss_G = loss_G_GAN_scream + loss_G_L1_scream * lam
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
        for p in net_D.parameters():
            p.data.clamp_(-0.01, 0.01)

    print 'epoch: %d, it: %d, loss_G: %f, loss_D: %f' % (e, i, loss_G.data[0], loss_D.data[0])
    if e%5 == 0:
        save_img(e)

print 'testing complete'
