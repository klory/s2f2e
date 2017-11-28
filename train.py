from __future__ import print_function
from option import *
from model import *
from load_data import *
import torch

opt = Option()()
model = create_model(opt)
batch_size = opt.batch_size
is_small = opt.is_small

if 'EFG' in opt.model:
    if 'CYC' in opt.model:
        transformed_dataset = EFGDataset(mode='training', transform=transforms.Compose(
            [AugmentImage(),
            ToTensor(),
            Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]), is_unpaired=True, is_small=is_small)
    else:
        transformed_dataset = EFGDataset(mode='training', transform=transforms.Compose(
            [AugmentImage(),
            ToTensor(),
            Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]), is_unpaired=True, is_small=is_small)
elif 'NFG' in opt.model:
    if 'CYC' in opt.model:
        transformed_dataset = NFGDataset(mode='training',transform=transforms.Compose(
            [AugmentImage(),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]), is_small=is_small)
    else:
        transformed_dataset = NFGDataset(mode='training',transform=transforms.Compose(
            [AugmentImage(),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]), is_small=is_small)

data_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False)

epoch_num = opt.epoch_num
data_size = len(data_loader)
total_step = 0
print('data_size: %d, scheduled_iter_num: %d' %(data_size, epoch_num*data_size))


which_model = opt.model
for e in range(epoch_num):
    for i, data in enumerate(data_loader):
        model.set_input(data)
        if 'LSGAN' in which_model:
            model.optimize()
        elif 'WGAN' in which_model:
            model.forward()
            if 'CYC' in which_model:
                for _ in range(5):
                    model.optimizer_D_A.zero_grad()
                    model.backward_D_A()
                    model.optimizer_D_A.step()

                    if 'WGAN' in model.model:
                        for p in model.net_D_A.parameters():
                            p.data.clamp_(-0.01, 0.01)

                    model.optimizer_D_B.zero_grad()
                    model.backward_D_B()
                    model.optimizer_D_B.step()

                    if 'WGAN' in model.model:
                        for p in model.net_D_B.parameters():
                            p.data.clamp_(-0.01, 0.01)

                model.optimizer_G.zero_grad()
                model.backward_G()
                model.optimizer_G.step()

            else:
                for _ in range(5):
                    model.optimizer_D.zero_grad()
                    model.backward_D()
                    model.optimizer_D.step()

                    for p in model.net_D.parameters():
                        p.data.clamp_(-0.01, 0.01)

                model.optimizer_G.zero_grad()
                model.backward_G()
                model.optimizer_G.step()
        else:
            raise ValueError('%s is not supported.' % which_model)
        model.save_loss()

        if total_step % opt.disp_freq == 0:
            print("iter: {0:5d}  ".format(total_step), end='')
            model.print_current_loss()

        if total_step != 0 and total_step % opt.save_freq == 0: 
            print('saving model at iteration {0}...'.format(total_step))
            model.save(total_step)

        total_step += 1

print('saving model at iteration {0}...'.format(total_step))
model.save(total_step)
print('Training complete.')
