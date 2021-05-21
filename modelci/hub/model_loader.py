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
from modelci.types.models.common import Framework, Engine
from modelci.types.models.mlmodel import MLModel


def joblib_loader(model_weight_path: Path):
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


def load(model: MLModel):
    # TODO only support torch.save, saved_model, and joblib serialization for the time being
    """A unify API to load model weight files in various format.

    Args:
        model: MLModel
    """
    if not Path(model.saved_path).exists():
        (filepath, filename) = os.path.split(model.saved_path)
        os.makedirs(filepath)
        with open(model.saved_path, 'wb') as f:
            f.write(model.weight.__bytes__())
    if model.framework == Framework.PyTorch and model.engine in (Engine.PYTORCH, Engine.NONE):  # PyTorch
        return pytorch_loader(model.saved_path)
    elif model.framework == Framework.TensorFlow and model.engine in (Engine.TFS, Engine.NONE):  # TensorFlow
        return savedmodel_loader(model.saved_path)
    elif model.framework in (Framework.Sklearn, Framework.XGBoost, Framework.LightGBM) and model.engine == 'NONE':  # sklearn
        return joblib_loader(model.saved_path)
