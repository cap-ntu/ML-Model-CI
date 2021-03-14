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

from modelci.hub.converter.to_onnx import ONNXConverter
from modelci.hub.converter.to_pytorch import PyTorchConverter
from modelci.hub.converter.to_tfs import TFSConverter
from modelci.hub.converter.to_torchscript import TorchScriptConverter
from modelci.hub.converter.to_trt import TRTConverter
from modelci.hub.converter.to_tvm import TVMConverter

framework_supported = {
    "onnx": ONNXConverter,
    "pytorch": PyTorchConverter,
    "tfs": TFSConverter,
    "torchscript": TorchScriptConverter,
    "trt": TRTConverter,
    "tvm": TVMConverter
}


def convert(model, src_framework: str, dst_framework: str, **kwargs):
    if dst_framework not in framework_supported.keys():
        raise NotImplementedError(f"Conversion to {dst_framework} is not supported yet")

    elif src_framework in getattr(framework_supported[dst_framework], "supported_framework"):
        converter = getattr(framework_supported[dst_framework], f"from_{src_framework}")
        return converter(model, **kwargs)

    else:
        raise NotImplementedError(f"Conversion from {src_framework} to {dst_framework} is not supported yet")
