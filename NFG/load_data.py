import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class SketchFaceDataset(Dataset):
    def __init__(self, mode, transform=None):
        import glob
        self.s_filenames = glob.glob("data/ar_s/*.jpg")
        self.nf_filenames = glob.glob("data/ar_nf/*.jpg")
        assert mode == 'training' or mode == 'testing'
        if mode == 'training':  # training
            self.s_filenames = glob.glob("data/ar_s/*.jpg")
            self.nf_filenames = glob.glob("data/ar_nf/*.jpg")

        self.transform = transform

    def __len__(self):
        return len(self.s_filenames)

    def __getitem__(self, idx):
        s_filename = self.s_filenames[idx]
        nf_filename = self.nf_filenames[idx]
        sketch = io.imread(s_filename)
        neutral = io.imread(nf_filename)
        sample = {'sketch': sketch, 'neutral': neutral}
        if self.transform:
            sample = self.transform(sample)
        return sample


class AugmentImage(object):
    def __call__(self, sample):
        sketch, neutral = sample['sketch'], sample['neutral']
        toPIL = transforms.ToPILImage()
        sketch = toPIL(sketch)
        neutral = toPIL(neutral)
        # ramdom rotate
        angle = 20 * np.random.random_sample() - 10
        sketch = sketch.rotate(angle)
        neutral = neutral.rotate(angle)
        # random resize
        output_size = np.random.randint(44, 85)
        resize = transforms.Scale(output_size)
        crop = transforms.CenterCrop(64)
        sketch = crop(resize(sketch))
        neutral = crop(resize(neutral))
        return {'sketch': sketch, 'neutral': neutral}


class ToTensor(object):
    def __call__(self, sample):
        sketch, neutral = sample['sketch'], sample['neutral']
        # sketch = sketch.transpose((2,0,1))
        # neutral = neutral.transpose((2,0,1))
        toTensor = transforms.ToTensor()
        return {'sketch': toTensor(sketch), 'neutral': toTensor(neutral)}


transformed_dataset = SketchFaceDataset(mode='training',
                                        transform=transforms.Compose([
                                            AugmentImage(),
                                            ToTensor()
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ]))

# test code                                        
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample['sketch'].size(), sample['neutral'].size())
    toPIL = transforms.ToPILImage()
    sketch = toPIL(sample['sketch'])
    sketch.show()
    neutral = toPIL(sample['neutral'])
    neutral.show()
    if i == 3:
        break
