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
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from modelci.hub.converter import convert
from modelci.types.bo import IOShape


class TestSklearnConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        X_bc, y_bc = load_breast_cancer(return_X_y=True)
        nrows = 15000
        X_bc: np.ndarray = X_bc[0:nrows]
        y_bc: np.ndarray = y_bc[0:nrows]

        cls.sklearn_model = RandomForestClassifier(n_estimators=10, max_depth=10)
        cls.sklearn_model.fit(X_bc, y_bc)
        cls.inputs_bc = [IOShape(shape=[-1, X_bc.shape[1]], dtype=float, name='input_0')]
        cls.sample_input = X_bc[0:2, :].astype(np.float32)
        cls.sklearn_model_out = cls.sklearn_model.predict(cls.sample_input)
        cls.sklearn_model_probs = cls.sklearn_model.predict_proba(cls.sample_input)

    # noinspection DuplicatedCode
    def test_sklearn_to_onnx(self):
        onnx_model = convert(self.sklearn_model, 'sklearn', 'onnx', inputs=self.inputs_bc, optimize=False)
        onnx.checker.check_model(onnx_model)
        ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        ort_inputs = {ort_session.get_inputs()[0].name: self.sample_input}
        onnx_model_out, onnx_model_probs = ort_session.run(None, ort_inputs)
        np.testing.assert_array_equal(onnx_model_out, self.sklearn_model_out)
        np.testing.assert_allclose(np.array([list(item.values()) for item in onnx_model_probs]), self.sklearn_model_probs, rtol=1e-05, atol=1e-05)

    def test_sklearn_to_torch(self):
        model = convert(self.sklearn_model, 'sklearn', 'pytorch', extra_config={'tree_implementation': 'gemm'})
        torch_model_out, torch_model_probs = model(self.sample_input)
        np.testing.assert_array_equal(torch_model_out.numpy(), self.sklearn_model_out)
        np.testing.assert_allclose(torch_model_probs.numpy(), self.sklearn_model_probs, rtol=1e-05, atol=1e-05)

    if __name__ == '__main__':
        unittest.main()