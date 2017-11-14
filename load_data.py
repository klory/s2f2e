import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class NFGDataset(Dataset):
    def __init__(self, mode, transform=None):
        assert mode == 'training' or mode == 'testing'
        import glob
        self.s_filenames = glob.glob("data/64/nfg_training/sketches/*.jpg")
        self.p_filenames = glob.glob("data/64/nfg_training/photos/*.jpg")
        if mode == 'testing':  # training
            self.s_filenames = glob.glob("data/64/nfg_testing/sketches/*.jpg")
            self.p_filenames = glob.glob("data/64/nfg_testing/photos/*.jpg")

        assert len(self.s_filenames) == len(self.p_filenames)
        self.transform = transform

    def __len__(self):
        return len(self.s_filenames)

    def __getitem__(self, idx):
        s_filename = self.s_filenames[idx]
        p_filename = self.p_filenames[idx]
        sketch = Image.open(s_filename).convert('RGB')
        photo = Image.open(p_filename).convert('RGB')
        sample = {'source': sketch, 'target': photo}
        if self.transform:
            sample = self.transform(sample)
        return sample


class EFGDataset(Dataset):
    def __init__(self, mode, transform=None):
        assert mode == 'training' or mode == 'testing'
        import glob
        p_filenames = glob.glob("data/64/efg_training/photos/*.jpg")
        smile_filenames = glob.glob("data/64/efg_training/expressions/smile/*.jpg")
        anger_filenames = glob.glob("data/64/efg_training/expressions/anger/*.jpg")
        scream_filenames = glob.glob("data/64/efg_training/expressions/scream/*.jpg")
        if mode == 'testing':  # training
            p_filenames = glob.glob("data/64/efg_testing/photos/*.jpg")
            smile_filenames = glob.glob("data/64/efg_testing/expressions/smile/*.jpg")
            anger_filenames = glob.glob("data/64/efg_testing/expressions/anger/*.jpg")
            scream_filenames = glob.glob("data/64/efg_testing/expressions/scream/*.jpg")

        self.p_filenames = p_filenames + p_filenames + p_filenames
        self.e_filenames = smile_filenames + anger_filenames + scream_filenames
        assert len(self.p_filenames) == len(self.e_filenames)
        self.transform = transform
        self.classes = ('smile', 'anger', 'scream')
        # generate labels
        self.labels = []
        for i in range(len(self.classes)):
            label = i
            for _ in smile_filenames:
                self.labels.append(label)


    def __len__(self):
        return len(self.p_filenames)

    def __getitem__(self, idx):
        p_filename = self.p_filenames[idx]
        e_filename = self.e_filenames[idx]
        photo = Image.open(p_filename).convert('RGB')
        expression = Image.open(e_filename).convert('RGB')
        sample = {'source': photo, 'target': expression}

        if self.transform:
            sample = self.transform(sample)

        label = self.labels[idx]

        return sample, label


class AugmentImage(object):
    def __call__(self, sample):
        sketch, photo = sample['source'], sample['target']
        # toPIL = transforms.ToPILImage()
        # sketch = toPIL(sketch)
        # photo = toPIL(photo)
        # # ramdom rotate between [-15, 15]
        # angle = 30 * np.random.random_sample() - 15
        # sketch = sketch.rotate(angle)
        # photo = photo.rotate(angle)

        # random flip
        hflip = np.random.random() < 0.5
        if hflip:
          sketch = sketch.transpose(Image.FLIP_LEFT_RIGHT)
          photo = photo.transpose(Image.FLIP_LEFT_RIGHT)


        # random resize between [44, 84]
        output_size = np.random.randint(44, 85)
        resize = transforms.Scale(output_size)  # for pytorch lower version!!!
        crop = transforms.CenterCrop(64)
        sketch = crop(resize(sketch))
        photo = crop(resize(photo))
        return {'source': sketch, 'target': photo}


class ToTensor(object):
    def __call__(self, sample):
        sketch, photo = sample['source'], sample['target']
        toTensor = transforms.ToTensor()
        return {'source': toTensor(sketch), 'target': toTensor(photo)}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        norm = transforms.Normalize(mean=self.mean, std=self.std)
        sketch, photo = sample['source'], sample['target']
        assert sketch.size(0) == 3
        return {'source': norm(sketch), 'target': norm(photo)}


if __name__ == "__main__":
    # test dataloader
    transformed_dataset = NFGDataset(mode='training',
                                    transform=transforms.Compose([ AugmentImage(),
                                                                    ToTensor(),
                                                                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Helper function to show a batch
    def show_batch(images, labels=None, classes=None):
        """Show image with landmarks for a batch of samples."""
        sources, targets = images['source'], images['target']

        grid = utils.make_grid(sources)
        plt.subplot(211)
        plt.imshow((grid.numpy()*0.5+0.5).transpose((1, 2, 0)))
        plt.axis('off')

        grid = utils.make_grid(targets)
        plt.subplot(212)
        plt.imshow((grid.numpy()*0.5+0.5).transpose((1, 2, 0)))
        plt.axis('off')

        title = ''
        if labels is not None:
            for label in labels:
                title += classes[label]
                title += ', '
        plt.title(title)

    if hasattr(transformed_dataset, 'classes'):
        classes = transformed_dataset.classes

    for epoch in range(10):
        for i_batch, sample_batched in enumerate(dataloader, 0):
            # images, labels = sample_batched
            images = sample_batched  # NO labels for NFGDataset

            # show 4th batch and stop.
            if i_batch == 3:
                plt.figure()
                # show_batch(images, labels, classes)
                show_batch(images)
                plt.show()
                break
