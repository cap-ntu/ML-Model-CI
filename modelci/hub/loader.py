#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/17/2021

Load model from path to runnable object
"""
import os
from pathlib import Path

import tensorflow as tf
import torch

from modelci.hub.utils import parse_path_plain


def pytorch_loader(model_weight_path: Path):
    return torch.load(model_weight_path)


def savedmodel_loader(model_weight_path: Path):
    """Load from TensorFlow saved model or HDF5 model.

    References:
        https://www.tensorflow.org/tutorials/keras/save_and_load
    """
    return tf.keras.models.load_model(model_weight_path)


def load(model_weight_path: os.PathLike, *args, **kwargs):
    model_weight_path = Path(model_weight_path)
    try:
        model_info = parse_path_plain(model_weight_path)
    except ValueError as e:
        # TODO: handle other path format, e.g. torch hub
        raise e

    if model_info['framework'] == 'PYTORCH' and model_info['engine'] in ('NONE', 'PYTORCH'):  # PyTorch
        pytorch_loader(model_weight_path)
    elif model_info['framework'] == 'TENSORFLOW' and model_info['engine'] in ('None', 'PYTORCH'):  # TensorFlow
        savedmodel_loader(model_weight_path)
