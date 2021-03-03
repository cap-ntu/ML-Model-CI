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

import unittest

import numpy as np
import onnxruntime as rt
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from modelci.hub.converter import convert
from modelci.types.bo import IOShape


class TestONNXConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        X_bc, y_bc = load_breast_cancer(return_X_y=True)
        nrows = 15000
        X_bc: np.ndarray = X_bc[0:nrows]
        y_bc: np.ndarray = y_bc[0:nrows]
        sklearn_model = RandomForestClassifier(n_estimators=10, max_depth=10)
        sklearn_model.fit(X_bc, y_bc)
        inputs_bc = [IOShape(shape=[-1, X_bc.shape[1]], dtype=float, name='input_0')]
        cls.onnx_model = convert(sklearn_model, 'sklearn', 'onnx', inputs=inputs_bc, optimize=False)
        sess = rt.InferenceSession(cls.onnx_model.SerializeToString())
        cls.sample_input = torch.rand(2, X_bc.shape[1], dtype=torch.float32)
        cls.onnx_model_predict = sess.run(None, {'input_0': cls.sample_input.numpy()})[0].flatten()

    # noinspection DuplicatedCode
    def test_onnx_to_pytorch(self):
        torch_model = convert(self.onnx_model,'onnx','pytorch')
        torch_model.eval()
        torch_model_predict = torch_model(self.sample_input)[0].data.numpy()
        np.testing.assert_allclose(self.onnx_model_predict, torch_model_predict, rtol=1e-05, atol=1e-05)

if __name__ == '__main__':
    unittest.main()