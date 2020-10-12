#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: USER
Email: yli056@e.ntu.edu.sg
Date: 10/11/2020
"""
import lightgbm as lgb
import numpy as np
import torch
import xgboost as xgt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from modelci.hub.converter import PyTorchConverter
from modelci.types.bo import IOShape

num_classes = 2
X = np.random.rand(100000, 28)
y = np.random.randint(num_classes, size=100000)

xgboost_model = xgt.XGBRegressor()
# xgboost_model.fit(X, y)

lgbm_model = lgb.LGBMClassifier()
# lgbm_model.fit(X, y)

inputs = [IOShape(shape=[-1, 28], dtype=int, name='input_0')]

X_bc, y_bc = load_breast_cancer(return_X_y=True)
nrows = 15000
X_bc: np.ndarray = X_bc[0:nrows]
y_bc: np.ndarray = y_bc[0:nrows]

sklearn_model = RandomForestClassifier(n_estimators=10, max_depth=10)
sklearn_model.fit(X_bc, y_bc)


def test_xgboost_to_torch():
    model = PyTorchConverter.from_xgboost(xgboost_model, inputs)
    sample_input = torch.rand((1, 28))
    classes = model(sample_input)
    assert tuple(classes.shape) == (1, 1)


def test_lightgbm_to_torch():
    model = PyTorchConverter.from_lightgbm(lgbm_model)
    sample_input = torch.rand((2, 28))
    classes, probs = model(sample_input)
    assert tuple(classes.shape) == (2,)
    assert tuple(probs.shape) == (2, num_classes)


def test_sklearn_to_torch():
    model = PyTorchConverter.from_sklearn(sklearn_model, extra_config={'tree_implementation': 'gemm'})
    sample_input = X_bc[0:1]
    out, probs = model(sample_input)
    assert tuple(out.shape) == (1,)
    assert tuple(probs.shape) == (1, 2)


if __name__ == '__main__':
    # test_xgboost_to_torch()
    # test_lightgbm_to_torch()
    test_sklearn_to_torch()
