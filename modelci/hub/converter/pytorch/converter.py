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

from typing import Sequence, Optional

import numpy as np
import onnx
import torch.onnx
from hummingbird.ml import constants as hb_constants
from hummingbird.ml.convert import _convert_xgboost, _convert_lightgbm, _convert_onnxml, _convert_sklearn  # noqa
from hummingbird.ml.operator_converters import constants as hb_op_constants
from lightgbm import LGBMModel
from xgboost import XGBModel

from modelci.types.bo import IOShape


class PyTorchConverter(object):
    supported_framework = ["xgboost", "lightgbm", "sklearn", "onnx"]

    hb_common_extra_config = {hb_constants.CONTAINER: False}

    @staticmethod
    def from_xgboost(
            model: XGBModel,
            inputs: Sequence[IOShape],
            device: str = 'cpu',
            extra_config: Optional[dict] = None,
            **kwargs
    ) -> torch.nn.Module:
        """Convert PyTorch module from XGBoost"""
        # inputs for XGBoost should contains only 1 argument with 2 dim
        if not (len(inputs) == 1 and len(inputs[0].shape) == 2):
            raise RuntimeError(
                'XGboost does not support such input data for inference. The input data should contains only 1\n'
                'argument with exactly 2 dimensions.'
            )

        if extra_config is None:
            extra_config = dict()

        # assert batch size
        batch_size = inputs[0].shape[0]
        if batch_size == -1:
            batch_size = 1
        test_input = np.random.rand(batch_size, inputs[0].shape[1])

        extra_config_ = PyTorchConverter.hb_common_extra_config.copy()
        extra_config_.update(extra_config)

        return _convert_xgboost(
            model, 'torch', test_input=test_input, device=device, extra_config=extra_config_
        )

    @staticmethod
    def from_lightgbm(
            model: LGBMModel,
            device: str = 'cpu',
            extra_config: Optional[dict] = None
    ):
        if extra_config is None:
            extra_config = dict()

        extra_config_ = PyTorchConverter.hb_common_extra_config.copy()
        extra_config_.update(extra_config)

        return _convert_lightgbm(
            model, 'torch', test_input=None, device=device, extra_config=extra_config_
        )

    @staticmethod
    def from_sklearn(
            model,
            device: str = 'cpu',
            extra_config: Optional[dict] = None,
    ):
        if extra_config is None:
            extra_config = dict()

        extra_config_ = PyTorchConverter.hb_common_extra_config.copy()
        extra_config_.update(extra_config)

        return _convert_sklearn(
            model, 'torch', test_input=None, device=device, extra_config=extra_config_
        )

    @staticmethod
    def from_onnx(
            model: onnx.ModelProto,
            opset: int = 10,
            device: str = 'cpu',
            extra_config: dict = None,
    ):
        if extra_config is None:
            extra_config = dict()
        inputs = {input_.name: input_ for input_ in model.graph.input}

        extra_config_ = PyTorchConverter.hb_common_extra_config.copy()
        extra_config_.update({
            hb_constants.ONNX_TARGET_OPSET: opset,
            hb_op_constants.N_FEATURES: None
        })
        extra_config_.update(extra_config)

        return _convert_onnxml(model, 'torch', test_input=None, device=device, extra_config=extra_config_)

