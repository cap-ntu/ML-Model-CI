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

import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime
import torch

from modelci.types.bo import IOShape
from modelci.hub.converter import convert
import unittest


class TestLightgbmConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        num_classes = 2
        X = np.random.rand(100000, 28).astype(np.float32)
        y = np.random.randint(num_classes, size=100000)
        cls.lgbm_model = lgb.LGBMClassifier()
        cls.lgbm_model.fit(X, y)
        cls.inputs = [IOShape(shape=[-1, 28], dtype=X.dtype, name='input_0')]
        cls.sample_input = X[0:2, :]
        cls.lgbm_model_out = cls.lgbm_model.predict(cls.sample_input)
        cls.lgbm_model_probs = cls.lgbm_model.predict_proba(cls.sample_input)

    # noinspection DuplicatedCode
    def test_lightgbm_to_onnx(self):
        onnx_model = convert(self.lgbm_model, 'lightgbm', 'onnx', inputs=self.inputs, opset=9, optimize=False)
        onnx.checker.check_model(onnx_model)
        ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        ort_inputs = {ort_session.get_inputs()[0].name: self.sample_input}
        onnx_model_out, onnx_model_probs = ort_session.run(None, ort_inputs)
        np.testing.assert_array_equal(onnx_model_out, self.lgbm_model_out)
        np.testing.assert_allclose(np.array([list(item.values()) for item in onnx_model_probs]), self.lgbm_model_probs, rtol=1e-05, atol=1e-05)

    def test_lightgbm_to_torch(self):
        model = convert(self.lgbm_model,'lightgbm', 'pytorch')
        classes, probs = model(torch.from_numpy(self.sample_input))
        np.testing.assert_array_equal(classes.numpy(), self.lgbm_model_out)
        np.testing.assert_allclose(probs.numpy(), self.lgbm_model_probs, rtol=1e-05, atol=1e-05)

if __name__ == '__main__':
    unittest.main()