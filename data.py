import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image


class Omniglot(Dataset):

    def __init__(self, dataset, way, transform=None):
        super(Omniglot, self).__init__()
        np.random.seed(0)
        self.dataset = dataset
        self.transform = transform
        self.img1 = None
        self.way = way

    def __len__(self):
        # means infinite
        # only use batch to train, without epoch
        return  21000000

    def __getitem__(self, index):
        labels = set()
        images = []
        while len(labels) < self.way:
            img = random.choice(self.dataset.imgs)
            if img[1] not in labels:
                labels.add(img[1])
                images.append(img)
        img = random.choice(self.dataset.imgs)
        while img[1] not in labels:
            img = random.choice(self.dataset.imgs)
        images.append(img)
        idx = 0
        for i, item in enumerate(images):
            if item[1] == img[1]:
                idx = i
                break
        y = np.array([idx])
        y = torch.from_numpy(y)
        images = [Image.open(it[0]) for it in images]
        images = [it.convert('L') for it in images]
        if self.transform:
            images = [self.transform(it) for it in images]
        return images, y


# test
if __name__=='__main__':
    omniglotTrain = OmniglotTrain('./images_background', 30000*8)
    print(omniglotTrain)
