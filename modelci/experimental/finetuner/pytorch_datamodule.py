#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Jiang Shanshan
Email: univeroner@gmail.com
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2021/1/20

References:
    https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
"""
import warnings

import pytorch_lightning as pl
import torchvision
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms

# transforms
from modelci.experimental.finetuner import OUTPUT_DIR

INPUT_SIZE = (224, 224)
TRAIN_VAL_SPLIT_RATIO = 0.7

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# noinspection PyAbstractClass
class PyTorchDataModule(pl.LightningDataModule):

    def __init__(self, dataset_name: str, batch_size: int = 8, num_workers=2, data_dir=OUTPUT_DIR):
        """

        Args:
            dataset_name (str): name of vision dataset to be load, full lists of supported datasets:
                https://pytorch.org/docs/stable/torchvision/datasets.html
            batch_size (int): samples per batch to load
        """
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        if hasattr(torchvision.datasets, self.dataset_name):
            self.dataset = getattr(torchvision.datasets, self.dataset_name)
        else:
            # TODO: support user local dataset folder
            warnings.warn('If you are using customized dataset, please specify a dataset type.')
            raise ValueError(f'torchvision.datasets does not have dataset name {self.dataset_name}')

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self, *args, **kwargs):
        self.dataset(root=self.data_dir, train=True, download=True)
        self.dataset(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # split dataset
        if stage == 'fit':
            full = self.dataset(root=self.data_dir, train=True, transform=train_transforms)
            train_size = int(len(full) * TRAIN_VAL_SPLIT_RATIO)
            test_size = len(full) - train_size
            self.train_dataset, self.val_dataset = random_split(full, [train_size, test_size])

        if stage == 'test':
            self.test_dataset = self.dataset(root=self.data_dir, train=False, transform=test_transforms)

    # return the dataloader for each split
    def train_dataloader(self):
        train = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return train

    def val_dataloader(self):
        val = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return test
