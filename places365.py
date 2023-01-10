# %% [code]
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch import nn
from torchvision.transforms import ToTensor, Lambda, ToPILImage, RandomRotation
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class PlacesDataset(Dataset):
    
    train_annot = "../input/annot-custom/train_custom.csv"
    train_annot_v2 = "../input/annot-custom-v2/train_custom.csv"
    val_annot = "../input/annot-custom/val_custom.csv"
    
    def __init__(self, train=True, train_version=None, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            if train_version:
                self.img_label = pd.read_csv(self.train_annot_v2)
            else:
                self.img_label = pd.read_csv(self.train_annot)
        else:
            self.img_label = pd.read_csv(self.val_annot)

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        img_path = self.img_label.iloc[idx, 0]
        image = read_image(img_path)
        label = self.img_label.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    return DataLoader(dataset=dataset, batch_size=batch_size, shullfle=shuffle, num_workers=num_workers)