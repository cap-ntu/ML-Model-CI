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
import onnx
import onnxruntime
import xgboost as xgt

from modelci.hub.converter import convert
from modelci.types.bo import IOShape


class TestXgboostConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        num_classes = 2
        X = np.random.rand(100000, 28).astype(np.float32)
        y = np.random.randint(num_classes, size=100000)
        cls.xgboost_model = xgt.XGBClassifier()
        cls.xgboost_model.fit(X, y)
        cls.inputs = [IOShape(shape=[-1, 28], dtype=X.dtype, name='input_0')]
        cls.sample_input = X[0:2, :]
        cls.xgboost_model_out = cls.xgboost_model.predict(cls.sample_input)
        cls.xgboost_model_probs = cls.xgboost_model.predict_proba(cls.sample_input)

    def test_xgboost_to_onnx(self):
        onnx_model = convert(self.xgboost_model, 'xgboost', 'onnx', inputs=self.inputs)
        onnx.checker.check_model(onnx_model)
        ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        ort_inputs = {ort_session.get_inputs()[0].name: self.sample_input}
        onnx_model_out, onnx_model_probs = ort_session.run(None, ort_inputs)
        np.testing.assert_array_equal(onnx_model_out, self.xgboost_model_out)
        np.testing.assert_allclose(np.array(onnx_model_probs), self.xgboost_model_probs, rtol=1e-05, atol=1e-05)

    def test_xgboost_to_torch(self):
        model = convert(self.xgboost_model, 'xgboost', 'pytorch', inputs=self.inputs)
        torch_model_out, torch_model_probs = model(self.sample_input)
        np.testing.assert_array_equal(torch_model_out.numpy(), self.xgboost_model_out)
        np.testing.assert_allclose(torch_model_probs.numpy(), self.xgboost_model_probs, rtol=1e-05, atol=1e-05)


if __name__ == '__main__':
    unittest.main()
