from __future__ import print_function
from option import *
from model import *
from load_data import *
import torch

opt = Option()()
model = create_model(opt)
batch_size = opt.batch_size


if 'EFG' in opt.model:
    if 'CYC' in opt.model:
        transformed_dataset = EFGDataset(mode='training', transform=transforms.Compose(
            [AugmentImage(),
            ToTensor(),
            Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]), is_unpaired=True)
    else:
        transformed_dataset = EFGDataset(mode='training', transform=transforms.Compose(
            [AugmentImage(),
            ToTensor(),
            Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]), is_unpaired=True)
elif 'NFG' in opt.model:
    if 'CYC' in opt.model:
        transformed_dataset = NFGDataset(mode='training',transform=transforms.Compose(
            [AugmentImage(),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
    else:
        transformed_dataset = NFGDataset(mode='training',transform=transforms.Compose(
            [AugmentImage(),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))

data_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False)

epoch = 100
data_size = len(data_loader)
#total_steps = epoch * data_size / batch_size
total_step = 0
print('data_size: %d, total_step: %d' %(data_size, total_step))


for e in range(epoch):
    for i, data in enumerate(data_loader):
        model.set_input(data)
        model.optimize()
        model.save_loss()

        if total_step % opt.disp_freq == 0:
            model.print_current_loss()

    if total_step % opt.save_freq == 0:
            model.save(e)

    total_step += 1

print('Training complete.')
