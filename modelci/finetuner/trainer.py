#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 1/12/2021
"""
import abc
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm
import torch.utils.data
from torch import nn
from torch.optim.optimizer import Optimizer

from modelci.finetuner import OUTPUT_DIR


class Trainer(abc.ABC):
    """
    Trainer base class. When used train_net API, a customer trainer should extend from this class.
    """

    def __init__(
            self,
            net: nn.Module,
            optimizer: Optimizer,
            train_data_loader: torch.utils.data.DataLoader,
            test_data_loader: torch.utils.data.DataLoader,
            train_data_size: int = None,
            test_data_size: int = None,
            device: torch.device = None,
    ):
        self.net = net
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.train_data_size = train_data_size or len(train_data_loader.dataset)
        self.test_data_size = test_data_size or len(test_data_loader.dataset)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def train_one_batch(self, samples) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def evaluate_one_batch(self, samples) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def get_sample_size(self, samples) -> int:
        pass


def train_net(
        net: nn.Module,
        epochs: int,
        trainer: Trainer,
        save_name: str = '',
        log_batch_num: int = None,
):
    """
    Train model general utility functions.

    Args:
        net (nn.Module):
        epochs (int):
        trainer (Callable):
        save_name (str): Optional. Saved name of model and training statistic result.
        log_batch_num (int): Optional. Update loss and accuracy log frequency.

    Return:
        A dictionary containing statistic results: training and validation loss and accuracy, and training
            parameters.
    """
    # statistics
    train_losses = np.zeros(epochs, dtype=np.float)
    train_accs = np.zeros(epochs, dtype=np.float)
    val_losses = np.zeros(epochs, dtype=np.float)
    val_accs = np.zeros(epochs, dtype=np.float)
    best_val_loss = float('inf')

    # misc
    timestamp = datetime.now().strftime('%m%d-%H%M%S')
    save_path = OUTPUT_DIR / '{name}_{timestamp:s}.pth'.format(name=save_name, timestamp=timestamp)

    val_loss, val_corrects = 0., 0
    pbar = tqdm(total=epochs * (len(trainer.train_data_loader) + len(trainer.test_data_loader)))
    for epoch in range(epochs):
        trainer.net.train()
        num_samples = 0
        train_loss, train_corrects = 0., 0
        running_loss, running_corrects = 0., 0

        for batch_num, samples in enumerate(trainer.train_data_loader):
            trainer.optimizer.zero_grad()
            num_samples += trainer.get_sample_size(samples)
            loss, corrects = trainer.train_one_batch(samples)
            running_loss += loss
            train_loss += loss
            running_corrects += corrects
            train_corrects += corrects

            pbar.update(1)

            if log_batch_num is not None and batch_num % log_batch_num == 0:
                pbar.set_description(
                    f'[epoch {epoch:d}] [Training step {batch_num:d}] '
                    f'train loss {running_loss / num_samples:2.5g} | acc {running_corrects / num_samples:2.5g} '
                    f'|| test loss {val_loss:2.5g} | acc {val_corrects:2.5g}'
                )
                num_samples = 0
                running_loss = 0.
                running_corrects = 0

        train_loss /= trainer.train_data_size
        train_corrects /= trainer.train_data_size

        pbar.set_description(
            f'[epoch {epoch:d}] [Evaluating] '
            f'train loss {train_loss:2.5g} | acc {train_corrects:2.5g} '
            f'|| test loss --- | acc ---'
        )

        trainer.net.eval()
        val_loss, val_corrects = 0., 0

        for samples in trainer.test_data_loader:
            loss, corrects = trainer.evaluate_one_batch(samples)
            val_loss += loss
            val_corrects += corrects
            pbar.update(1)

        val_loss /= trainer.test_data_size
        val_corrects /= trainer.test_data_size

        pbar.set_description(
            f'[epoch {epoch:d}] [Saving Model] '
            f'train loss {train_loss:2.5g} | acc {train_corrects:2.5g} '
            f'|| test loss {val_loss:2.5g} | acc {val_corrects:2.5g}'
        )

        # process statistics
        train_losses[epoch], train_accs[epoch] = train_loss, train_corrects
        val_losses[epoch], val_accs[epoch] = val_loss, val_corrects

        # save model
        if val_loss < best_val_loss:
            pbar.set_description(
                f'[epoch {epoch:d}] [Done] '
                f'train loss {train_loss:2.5g} | acc {train_corrects:2.5g} '
                f'|| test loss {val_loss:2.5g} | acc {val_corrects:2.5g}'
            )
            best_val_loss = val_loss
            torch.save(net.state_dict(), save_path)

    stat = {'train_loss': train_losses, 'train_acc': train_accs, 'test_loss': val_losses, 'test_acc': val_accs}

    return {'stat': stat, 'info': {'save_name': save_path}}


class CIFAR10Trainer(Trainer):
    def __init__(
            self,
            net: nn.Module,
            optimizer: Optimizer,
            train_data_loader: torch.utils.data.DataLoader,
            test_data_loader: torch.utils.data.DataLoader,
            train_data_size: int = None,
            test_data_size: int = None,
            device: torch.device = None,
    ):
        """
        Args:
            net (nn.Module): PyTorch model.
            train_data_loader (torch.utils.data.DataLoader): Data loader for training.
            test_data_loader (torch.utils.data.DataLoader): Data loader for evaluating.
            optimizer (torch.optim.optimizer.Optimizer): Optimizer.
            train_data_size (int): Size of training data. Optional. Needed when using a sampler to split
                training and validation data.
            test_data_size (int): Size of evaluation data. Optional. Needed when using a sampler to split
                training and validation data.
        """
        super().__init__(
            net, optimizer=optimizer, train_data_loader=train_data_loader, test_data_loader=test_data_loader,
            train_data_size=train_data_size, test_data_size=test_data_size, device=device,
        )

    def train_one_batch(self, samples):

        """Train one epoch.

        Args:
            samples: One batch of data

        Returns:
            Tuple of training loss and number of correctly predicted examples
        """
        inputs, labels = samples
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # forward + backward + optimize
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        predicted_labels = torch.argmax(outputs, dim=1)
        corrects = torch.sum(predicted_labels == labels)

        return loss.item(), corrects.item()

    def evaluate_one_batch(self, samples):
        inputs, labels = samples
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # forward
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        predicted_labels = torch.argmax(outputs, dim=1)
        corrects = torch.sum(predicted_labels == labels)

        return loss.item(), corrects.item()

    def get_sample_size(self, samples):
        _, labels = samples
        return len(labels)
