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
import unittest
from pathlib import Path

import torch
import onnx
from modelci.types.bo import IOShape

from modelci.types.trtis_objects import ModelInputFormat
import numpy as np
import tempfile
from modelci.hub.converter import convert
import onnxruntime as rt


class TestPytorchConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.torch_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        cls.inputs = [IOShape([-1, 3, 224, 224], dtype=float, name='INPUT__0', format=ModelInputFormat.FORMAT_NCHW)]
        cls.outputs = [IOShape([-1, 1000], dtype=float, name='probs')]
        cls.X = torch.rand(1, 3, 224, 224, dtype=torch.float)
        cls.onnx_path = Path(tempfile.gettempdir() + '/test.onnx')
        cls.torchscript_path = Path(tempfile.gettempdir() + '/test_torchscript.zip')

    def test_torch_to_onnx(self):
        convert(self.torch_model, 'pytorch', 'onnx', save_path=self.onnx_path, inputs=self.inputs, outputs=self.outputs)
        onnx_model = onnx.load(self.onnx_path)
        # TODO add checker after upgrade ONNX version to 1.7
        sess = rt.InferenceSession(onnx_model.SerializeToString())
        onnx_model_predict = sess.run(['probs'], {'INPUT__0': self.X.numpy()})[0].flatten()
        self.torch_model.eval()
        torch_model_predict = self.torch_model(self.X)[0].data.numpy()
        np.testing.assert_allclose(onnx_model_predict, torch_model_predict, rtol=1e-05, atol=1e-05)

    def test_torch_to_torchscript(self):
        convert(self.torch_model, 'pytorch', 'torchscript', save_path=self.torchscript_path)
        torchscript_model = torch.jit.load(str(self.torchscript_path))
        torchscript_model_predict = torchscript_model(self.X)[0].data.numpy()
        self.torch_model.eval()
        torch_model_predict = self.torch_model(self.X)[0].data.numpy()
        np.testing.assert_allclose(torchscript_model_predict, torch_model_predict, rtol=1e-05, atol=1e-05)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.onnx_path):
            os.remove(cls.onnx_path)
        if os.path.exists(cls.torchscript_path):
            os.remove(cls.torchscript_path)

if __name__ == '__main__':
    unittest.main()