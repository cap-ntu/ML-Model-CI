#!/usr/bin/python3
# -*- coding: utf-8 -*-
#  Copyright (c) NTU_CAP 2021. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import os
from pathlib import Path

from modelci.hub.converter.to_onnx import ONNXConverter
from modelci.hub.converter.to_pytorch import PyTorchConverter
from modelci.hub.converter.to_tfs import TFSConverter
from modelci.hub.converter.to_torchscript import TorchScriptConverter
from modelci.hub.converter.to_trt import TRTConverter

import torch
import tensorflow as tf
from functools import partial
import xgboost as xgb
import lightgbm as lgb
import sklearn as skl
from modelci.hub.model_loader import load
from modelci.hub.utils import generate_path_plain
from modelci.types.models import MLModel, Engine

framework_supported = {
    "onnx": ONNXConverter,
    "pytorch": PyTorchConverter,
    "tfs": TFSConverter,
    "torchscript": TorchScriptConverter,
    "trt": TRTConverter
}


def convert(model, src_framework: str, dst_framework: str, **kwargs):
    if dst_framework not in framework_supported.keys():
        raise NotImplementedError(f"Conversion to {dst_framework} is not supported yet")

    elif src_framework in getattr(framework_supported[dst_framework], "supported_framework"):
        converter = getattr(framework_supported[dst_framework], f"from_{src_framework}")
        return converter(model, **kwargs)

    else:
        raise NotImplementedError(f"Conversion from {src_framework} to {dst_framework} is not supported yet")


def generate_model_family(
        model: MLModel,
        max_batch_size: int = -1
):
    model_weight_path = model.saved_path
    if not Path(model.saved_path).exists():
        (filepath, filename) = os.path.split(model.saved_path)
        os.makedirs(filepath)
        with open(model.saved_path, 'wb') as f:
            f.write(model.weight.__bytes__())
    net = load(model_weight_path)
    build_saved_dir_from_engine = partial(
        generate_path_plain,
        **model.dict(include={'architecture', 'framework', 'task', 'version'}),
    )
    inputs = model.inputs
    outputs = model.outputs
    model_input = model.model_input

    generated_dir_list = list()

    torchscript_dir = build_saved_dir_from_engine(engine=Engine.TORCHSCRIPT)
    tfs_dir = build_saved_dir_from_engine(engine=Engine.TFS)
    onnx_dir = build_saved_dir_from_engine(engine=Engine.ONNX)
    trt_dir = build_saved_dir_from_engine(engine=Engine.TRT)

    if isinstance(net, torch.nn.Module):
        _torchfamily(net, False, torchscript_dir, onnx_dir, generated_dir_list, inputs, outputs, model_input)
    elif isinstance(net, tf.keras.Model):
        _tffamily(net, tfs_dir, generated_dir_list, trt_dir, inputs, outputs)
    elif isinstance(net, xgb.XGBModel):
        _xgbfamily(net, inputs, onnx_dir, generated_dir_list, torchscript_dir, outputs, model_input)
    elif isinstance(net, lgb.LGBMModel):
        _lgbfamily(net, inputs, onnx_dir, generated_dir_list,torchscript_dir, outputs, model_input)
    elif isinstance(net, skl.base.BaseEstimator):
        _sklfamily(net, inputs, onnx_dir, generated_dir_list, torchscript_dir, outputs, model_input)
    return generated_dir_list


def _torchfamily(torchmodel: torch.nn.Module, mlconvert: bool, torchscript_dir: Path, onnx_dir, generated_dir_list, inputs, outputs, model_input):
    # to TorchScript
    if convert(torchmodel, 'pytorch', 'torchscript', save_path=torchscript_dir):
        generated_dir_list.append(torchscript_dir.with_suffix('.zip'))

    # to ONNX, TODO(lym): batch cache, input shape, opset version
    if not mlconvert and convert(torchmodel, 'pytorch', 'onnx', save_path=onnx_dir, inputs=inputs,
                                     outputs=outputs, model_input=model_input, optimize=False):
        generated_dir_list.append(onnx_dir.with_suffix('.onnx'))

    # to TRT
    # TRTConverter.from_onnx(
    #     onnx_path=onnx_dir.with_suffix('.onnx'), save_path=trt_dir, inputs=inputs, outputs=outputs
    # )
    # TODO: expose custom settings to usrs


def _tffamily(tfmodel: tf.keras.Model, tfs_dir: Path, generated_dir_list, trt_dir, inputs, outputs):
    # to TFS
    convert(tfmodel, 'tensorflow', 'tfs', save_path=tfs_dir)
    generated_dir_list.append(tfs_dir.with_suffix('.zip'))

    # to TRT
    convert(tfmodel, 'tfs', 'trt', tf_path=tfs_dir, save_path=trt_dir, inputs=inputs, outputs=outputs,
            max_batch_size=32)
    generated_dir_list.append(trt_dir.with_suffix('.zip'))


def _xgbfamily(xgbmodel: xgb.XGBModel, inputs, onnx_dir: Path, generated_dir_list, torchscript_dir: Path, outputs, model_input):
    convert(xgbmodel, 'xgboost', 'onnx', inputs=inputs, save_path=onnx_dir)
    generated_dir_list.append(onnx_dir.with_suffix('.onnx'))
    torch_model = convert(xgbmodel, 'xgboost', 'pytorch', inputs=inputs)
    _torchfamily(torch_model, True, torchscript_dir, onnx_dir, generated_dir_list, inputs, outputs, model_input)


def _lgbfamily(lgbmodel: lgb.LGBMModel, inputs, onnx_dir, generated_dir_list, torchscript_dir: Path, outputs, model_input):
    convert(lgbmodel, 'lightgbm', 'onnx', inputs=inputs, save_path=onnx_dir)
    generated_dir_list.append(onnx_dir.with_suffix('.onnx'))
    torch_model = convert(lgbmodel, 'lightgbm', 'pytorch')
    _torchfamily(torch_model, True, torchscript_dir, onnx_dir, generated_dir_list, inputs, outputs, model_input)


def _sklfamily(sklmodel: skl.base.BaseEstimator, inputs, onnx_dir: Path, generated_dir_list, torchscript_dir: Path, outputs, model_input):
    convert(sklmodel, 'sklearn', 'onnx', inputs=inputs, save_path=onnx_dir)
    generated_dir_list.append(onnx_dir.with_suffix('.onnx'))
    torch_model = convert(sklmodel, 'sklearn', 'pytorch')
    _torchfamily(torch_model, True, torchscript_dir, onnx_dir, generated_dir_list, inputs, outputs, model_input)
