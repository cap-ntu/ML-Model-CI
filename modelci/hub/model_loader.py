#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/17/2021

Load model from weight files stored in local disk to the memory. Such loaded model can be used for inference.

This file provides a unify API for loading models that stores in different formats (e.g. PyTorch pickle,
TensorFlow saved model).
"""
import os
from pathlib import Path

import tensorflow as tf
import torch
import joblib
from modelci.hub.utils import parse_path_plain


def sklearn_loader(model_weight_path: Path):
    """Load from sklearn api of XGBoost or LightGBM, and sklearn model.
    """
    return joblib.load(model_weight_path)


def pytorch_loader(model_weight_path: Path):
    return torch.load(model_weight_path)


def savedmodel_loader(model_weight_path: Path):
    """Load from TensorFlow saved model or HDF5 model.

    References:
        https://www.tensorflow.org/tutorials/keras/save_and_load
    """
    return tf.keras.models.load_model(model_weight_path)


def load(model_weight_path: os.PathLike, *args, **kwargs):
    """A unify API to load model weight files in various format.

    Args:
        model_weight_path: Path to the model weight file. The model is saved in ModelCI standard directory.
    """

    model_weight_path = Path(model_weight_path)
    try:
        model_info = parse_path_plain(model_weight_path)
    except ValueError as e:
        # TODO: handle other path format, e.g. torch hub
        raise e
    if model_info['framework'] == 'PYTORCH' and model_info['engine'] in ('NONE', 'PYTORCH'):  # PyTorch
        return pytorch_loader(model_weight_path)
    elif model_info['framework'] == 'TENSORFLOW' and model_info['engine'] in ('None', 'TENSORFLOW'):  # TensorFlow
        return savedmodel_loader(model_weight_path)
    elif model_info['framework'] in ('SKLEARN', 'XGBOOST', 'LIGHTGBM') and model_info['engine'] == 'NONE':  # sklearn
        return sklearn_loader(model_weight_path)
