import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from random import shuffle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

image_size = 128
data_folder = "./data/" + str(image_size) + "/"


class NFGDataset(Dataset):
    def __init__(self, mode, transform=None, is_small=False):
        assert mode == 'training' or mode == 'testing'
        import glob
        self.s_filenames = glob.glob(
            data_folder + "nfg_training/sources/*.jpg")
        self.p_filenames = glob.glob(data_folder + "nfg_training/targets/*.jpg")
        if mode == 'testing':  # training
            self.s_filenames = glob.glob(
                data_folder + "nfg_testing/sources/*.jpg")
            self.p_filenames = glob.glob(
                data_folder + "nfg_testing/targets/*.jpg")
        
        if is_small:
            self.s_filenames = self.s_filenames[:8]
            self.p_filenames = self.p_filenames[:8]

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
    def __init__(self, mode="training", end_to_end=False, transform=None, is_unpaired=False, is_small=False):
        assert mode == 'training' or mode == 'testing'
        network = "efg"
        if end_to_end is True:
            network = "end2end"
        current_data_folder = data_folder + network + "_" + mode

        import glob
        s_filenames = glob.glob(current_data_folder + "/sources/*.jpg")
        smile_filenames = glob.glob(
            current_data_folder + "/targets/smile/*.jpg")
        anger_filenames = glob.glob(
            current_data_folder + "/targets/anger/*.jpg")
        scream_filenames = glob.glob(
            current_data_folder + "/targets/scream/*.jpg")

        if is_small:
            s_filenames = s_filenames[:8]
            smile_filenames = smile_filenames[:8]
            anger_filenames = anger_filenames[:8]
            scream_filenames = scream_filenames[:8]

        self.s_filenames = s_filenames + s_filenames + s_filenames
        if is_unpaired:
            shuffle(self.s_filenames)
        self.e_filenames = smile_filenames + anger_filenames + scream_filenames
        assert len(self.s_filenames) == len(self.e_filenames)
        self.transform = transform
        self.classes = ('smile', 'anger', 'scream')
        # generate labels
        self.labels = []
        for i in range(len(self.classes)):
            label = i
            for _ in smile_filenames:
                self.labels.append(label)

    def __len__(self):
        return len(self.s_filenames)

    def __getitem__(self, idx):
        s_filename = self.s_filenames[idx]
        e_filename = self.e_filenames[idx]
        photo = Image.open(s_filename).convert('RGB')
        expression = Image.open(e_filename).convert('RGB')
        sample = {'source': photo, 'target': expression}

        if self.transform:
            sample = self.transform(sample)

        label = self.labels[idx]

        return sample, label


class AugmentImage(object):
    def __call__(self, sample):
        sketch, photo = sample['source'], sample['target']

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
        output_size = np.random.randint(
            int(0.8 * image_size), int(1.2 * image_size))
        resize = transforms.Scale(output_size)  # for pytorch lower version!!!
        crop = transforms.CenterCrop(image_size)
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
    transformed_dataset = EFGDataset(mode='testing', end_to_end=True, is_unpaired=False,
                                     transform=transforms.Compose([AugmentImage(), ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))

    dataloader = DataLoader(transformed_dataset,
                            batch_size=4, shuffle=True, num_workers=4)

"""
    # Helper function to show a batch
    def show_batch(images, labels=None, classes=None):
        #Show image with landmarks for a batch of samples.
        sources, targets = images['source'], images['target']
        print(sources[0].size())
        grid = utils.make_grid(sources)
        plt.subplot(211)
        plt.imshow((grid.numpy() * 0.5 + 0.5).transpose((1, 2, 0)))
        plt.axis('off')

        grid = utils.make_grid(targets)
        plt.subplot(212)
        plt.imshow((grid.numpy() * 0.5 + 0.5).transpose((1, 2, 0)))
        plt.axis('off')

        title = ''
        if labels is not None:
            for label in labels:
                title += classes[label]
                title += ', '
        plt.title(title)

    if hasattr(transformed_dataset, 'classes'):
        classes = transformed_dataset.classes

    for epoch in range(2):
        for i_batch, sample_batched in enumerate(dataloader, 0):
            images, labels = sample_batched
            # images = sample_batched  # NO labels for NFGDataset

            # show 4th batch and stop.
            if i_batch == 3:
                plt.figure()
                show_batch(images, labels, classes)
                # show_batch(images)
                plt.show()
                break
"""
