#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 10/11/2020

TODO: Cover converter test from here: https://github.com/microsoft/hummingbird/tree/master/tests
"""
import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime
import torch
import xgboost as xgt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from modelci.hub.converter import PyTorchConverter, ONNXConverter
from modelci.types.bo import IOShape
from modelci.types.type_conversion import model_data_type_to_torch

num_classes = 2
X = np.random.rand(100000, 28).astype(np.float32)
y = np.random.randint(num_classes, size=100000)
inputs = [IOShape(shape=[-1, 28], dtype=X.dtype, name='input_0')]

xgboost_model = xgt.XGBRegressor()
xgboost_model.fit(X, y)

lgbm_model = lgb.LGBMClassifier()
lgbm_model.fit(X, y)

X_bc, y_bc = load_breast_cancer(return_X_y=True)
nrows = 15000
X_bc: np.ndarray = X_bc[0:nrows]
y_bc: np.ndarray = y_bc[0:nrows]

sklearn_model = RandomForestClassifier(n_estimators=10, max_depth=10)
sklearn_model.fit(X_bc, y_bc)
inputs_bc = [IOShape(shape=[-1, X_bc.shape[1]], dtype=X_bc.dtype, name='input_0')]


def test_xgboost_to_onnx():
    onnx_model = ONNXConverter.from_xgboost(xgboost_model, inputs)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    ort_inputs = {ort_session.get_inputs()[0].name: X[0:2, :]}
    ort_outs = ort_session.run(None, ort_inputs)
    assert len(ort_outs) == 1
    assert tuple(ort_outs[0].shape) == (2, 1)


# noinspection DuplicatedCode
def test_lgbm_to_onnx():
    onnx_model = ONNXConverter.from_lightgbm(lgbm_model, inputs, opset=9, optimize=False)  # noqa
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    ort_inputs = {ort_session.get_inputs()[0].name: X[0:2, :]}
    out, probs = ort_session.run(None, ort_inputs)
    assert tuple(out.shape) == (2,)
    assert len(probs) == 2
    assert len(probs[0]) == 2


# noinspection DuplicatedCode
def test_sklearn_to_onnx():
    onnx_model = ONNXConverter.from_sklearn(sklearn_model, inputs_bc, optimize=False)  # noqa
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    ort_inputs = {ort_session.get_inputs()[0].name: X_bc[0:2, :]}
    out, probs = ort_session.run(None, ort_inputs)
    assert tuple(out.shape) == (2,)
    assert len(probs) == 2
    assert len(probs[0]) == 2


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
    sample_input = X_bc[0:1, :]
    out, probs = model(sample_input)
    assert tuple(out.shape) == (1,)
    assert tuple(probs.shape) == (1, 2)


def test_onnx_to_pytorch():
    onnx_model = ONNXConverter.from_sklearn(sklearn_model, inputs_bc, optimize=False)  # noqa
    inputs = list()
    for input_ in onnx_model.graph.input:
        name = input_.name
        t = input_.type.tensor_type
        shape = list()
        if t.HasField('shape'):
            for d in t.shape.dim:
                if d.HasField('dim_value'):
                    shape.append(d.dim_value)
                elif d.HasField('dim_param'):
                    shape.append(d.dim_param)
                else:
                    shape.append(-1)
        dtype = t.elem_type
        inputs.append(IOShape(name=name, dtype=dtype, shape=shape))

    dtype = model_data_type_to_torch(inputs[0].dtype)
    sample_input = torch.rand([2, *inputs[0].shape[1:]], dtype=dtype)
    model = PyTorchConverter.from_onnx(onnx_model)

    model(sample_input)


if __name__ == '__main__':
    # test_xgboost_to_onnx()
    # test_lgbm_to_onnx()
    # test_sklearn_to_onnx()
    # test_xgboost_to_torch()
    # test_lightgbm_to_torch()
    # test_sklearn_to_torch()
    test_onnx_to_pytorch()
