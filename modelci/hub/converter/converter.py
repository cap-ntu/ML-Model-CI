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

    # TODO: expose custom settings to usrs
    def torchfamily(torchmodel: torch.nn.Module, mlconvert: bool):
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
    def tffamily(tfmodel: tf.keras.Model):
        # to TFS
        convert(tfmodel, 'tensorflow', 'tfs', save_path=tfs_dir)
        generated_dir_list.append(tfs_dir.with_suffix('.zip'))

        # to TRT
        convert(tfmodel, 'tfs', 'trt', tf_path=tfs_dir, save_path=trt_dir, inputs=inputs, outputs=outputs,
                max_batch_size=32)
        generated_dir_list.append(trt_dir.with_suffix('.zip'))

    def xgbfamily(xgbmodel: xgb.XGBModel):
        convert(xgbmodel, 'xgboost', 'onnx', inputs=inputs, save_path=onnx_dir)
        generated_dir_list.append(onnx_dir.with_suffix('.onnx'))
        torch_model = convert(xgbmodel, 'xgboost', 'pytorch', inputs=inputs)
        torchfamily(torch_model, True)

    def lgbfamily(lgbmodel: lgb.LGBMModel):
        convert(lgbmodel, 'lightgbm', 'onnx', inputs=inputs, save_path=onnx_dir)
        generated_dir_list.append(onnx_dir.with_suffix('.onnx'))
        torch_model = convert(lgbmodel, 'lightgbm', 'pytorch')
        torchfamily(torch_model, True)

    def sklfamily(sklmodel: skl.base.BaseEstimator):
        convert(sklmodel, 'sklearn', 'onnx', inputs=inputs, save_path=onnx_dir)
        generated_dir_list.append(onnx_dir.with_suffix('.onnx'))
        torch_model = convert(sklmodel, 'sklearn', 'pytorch')
        torchfamily(torch_model, True)

    if isinstance(net, torch.nn.Module):
        torchfamily(net, mlconvert=False)
    elif isinstance(net, tf.keras.Model):
        tffamily(net)
    elif isinstance(net, xgb.XGBModel):
        xgbfamily(net)
    elif isinstance(net, lgb.LGBMModel):
        lgbfamily(net)
    elif isinstance(net, skl.base.BaseEstimator):
        sklfamily(net)
    return generated_dir_list
