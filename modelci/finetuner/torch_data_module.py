#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Jiang Shanshan
Email: univeroner@gmail.com
Date: 2021/1/20

"""

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from PIL import Image
from torchvision.transforms import Compose, transforms
import tensorflow_datasets as tfds
from typing import List, Optional, Tuple

class TorchDataset(Dataset):
    """Dataset with support of transforms for pytorch models.
    """

    def __init__(self, data: np.array, targets: List, transform: Optional[Compose] = None,
                 target_transform: Optional[Compose] = None):
        """

        Args:
            data (np.array): image data in numpy array format
            targets (List): label data
            transform (Optional[Compose]): data transform applied to image data
            target_transform (Optional[Compose]):  data transform applied to label data
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[object, int]:
        """

        Args:
            index (int): Index
        Returns:
            tuple: (Tuple[object, int]) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class PyTorchDataModule(pl.LightningDataModule):

    def __init__(self, dataset_name: str, input_size: Tuple[int, int]=(224,224), batch_size: int=8):
        """

        Args:
            dataset_name (str): name of dataset to be load, full lists of supported datasets: https://www.tensorflow.org/datasets/catalog/overview#all_datasets
            input_size (Tuple[int, int]): input image size of target model
            batch_size (int): samples per batch to load
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.input_size = input_size
        self.batch_size = batch_size
        self.data_builder = tfds.builder(self.dataset_name)

    def prepare_data(self):
        self.data_builder.download_and_prepare()

    def setup(self, stage=None):
        # transforms
        train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        test_transforms = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        # split dataset
        if stage == 'fit':
            train_data = self.data_builder.as_dataset(split='train', batch_size=-1)
            train_image = tfds.as_numpy(train_data)['image']
            train_label = tfds.as_numpy(train_data)['label'].tolist()
            train = TorchDataset(train_image, train_label, transform=train_transforms)
            self.train, self.val = random_split(train, [int(len(train)*0.7), int(len(train)*0.3)], generator=torch.Generator().manual_seed(42))

        if stage == 'test':
            test_data = self.data_builder.as_dataset(split='test', batch_size=-1)
            test_image = tfds.as_numpy(test_data)['image']
            test_label = tfds.as_numpy(test_data)['label'].tolist()
            self.test = TorchDataset(test_image, test_label, transform=test_transforms)

    # return the dataloader for each split
    def train_dataloader(self):
        train = DataLoader(self.train, batch_size=self.batch_size, num_workers=2)
        return train

    def val_dataloader(self):
        val = DataLoader(self.val, batch_size=self.batch_size, num_workers=2)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test, batch_size=self.batch_size, num_workers=2)
        return test