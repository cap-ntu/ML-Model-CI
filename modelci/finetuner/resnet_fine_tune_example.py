#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 1/25/2021
"""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from modelci.finetuner.pytorch_datamodule import PyTorchDataModule
from modelci.finetuner.trainer import PyTorchTrainer
from modelci.finetuner.transfer_learning import freeze, FineTuneModule


def main(args: argparse.Namespace) -> None:
    """Train the model.

    Args:
        args: Model hyper-parameters

    Note:
        For the sake of the example, the images dataset will be downloaded
        to a temporary directory.
    """

    with TemporaryDirectory(dir=args.root_data_path):
        # TODO: Done in editor (L247-251)
        net = torch.hub.load('pytorch/vision:v0.6.0', args.backbone, pretrained=True)
        freeze(module=net, train_bn=True)

        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, 10)

        model = FineTuneModule(**vars(args), net=net, loss=torch.nn.CrossEntropyLoss())
        data_module = PyTorchDataModule('CIFAR10', batch_size=args.batch_size, data_dir=args.root_data_path)

        trainer = PyTorchTrainer(
            model=model,
            data_loader_kwargs={'datamodule': data_module},
            trainer_kwargs={
                'weights_summary': None,
                'progress_bar_refresh_rate': 1,
                'num_sanity_val_steps': 0,
                'gpus': args.gpus,
                'min_epochs': args.nb_epochs,
                'max_epochs': args.nb_epochs,
            }
        )
        trainer.start()
        trainer.join()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--root-data-path",
        metavar="DIR",
        type=str,
        default=Path.cwd().as_posix(),
        help="Root directory where to download the data",
        dest="root_data_path",
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        metavar="BK",
        help="Name (as in ``torchvision.models``) of the feature extractor",
    )
    parser.add_argument(
        "--epochs", default=15, type=int, metavar="N", help="total number of epochs", dest="nb_epochs"
    )
    parser.add_argument("--batch-size", default=128, type=int, metavar="B", help="batch size", dest="batch_size")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument(
        "--lr", "--learning-rate", default=1e-2, type=float, metavar="LR", help="initial learning rate", dest="lr"
    )
    parser.add_argument(
        "--lr-scheduler-gamma",
        default=1e-1,
        type=float,
        metavar="LRG",
        help="Factor by which the learning rate is reduced at each milestone",
    )
    parser.add_argument(
        "--step-size",
        default=7,
        type=int,
        metavar="SS",
        help="Step size used in step scheduler",
    )
    parser.add_argument(
        "--num-workers", default=6, type=int, metavar="W", help="number of CPU workers", dest="num_workers"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
