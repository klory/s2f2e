from network import *

net = Unet_G1(3, 3, 'EFG')
print(net)

transformed_dataset = EFGDataset(mode='training', end_to_end=False,
        transform=transforms.Compose([AugmentImage(),
				                  	ToTensor(),
                                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))

dataloader = torch.utils.data.DataLoader(transformed_dataset, batch_size=1, shuffle=False)

for i, data in enumerate(dataloader):
    if i > 2:
        break
