#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Jiang Shanshan
Email: univeroner@gmail.com
Date: 2021/2/28

"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import onnx
import onnxruntime
import tensorflow as tf
import numpy as np
from modelci.types.trtis_objects import ModelInputFormat

from modelci.types.bo import IOShape

from modelci.hub.converter import convert


class TestKerasConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.keras_model = tf.keras.applications.ResNet50()
        cls.sample_input = np.random.random((1, 224, 224, 3)).astype('float32')
        cls.keras_model_predict = cls.keras_model.predict(cls.sample_input)
        cls.tfs_model_path = Path(tempfile.gettempdir() + '/ResNet50/tensorflow-tfs/image_classification/1')
        cls.trt_model_path = Path(tempfile.gettempdir() + '/ResNet50/tensorflow-trt/image_classification/1')
        cls.inputs = [IOShape([-1, 224, 224, 3], dtype=float, name='input_1', format=ModelInputFormat.FORMAT_NHWC)]
        cls.outputs = [IOShape([-1, 1000], dtype=float, name='probs')]

    def test_keras_to_onnx(self):
        onnx_model = convert(self.keras_model, 'keras', 'onnx')
        onnx.checker.check_model(onnx_model)
        ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        ort_inputs = {ort_session.get_inputs()[0].name: self.sample_input}
        onnx_model_predict = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(onnx_model_predict[0], self.keras_model_predict, rtol=1e-05, atol=1e-05)

    def test_keras_to_tfs(self):
        convert(self.keras_model, 'tensorflow', 'tfs', save_path=self.tfs_model_path)
        tfs_model = tf.keras.models.load_model(self.tfs_model_path)
        tfs_model_predict = tfs_model.predict(self.sample_input)
        np.testing.assert_allclose(tfs_model_predict, self.keras_model_predict)


    @classmethod
    def tearDownClass(cls):
        if os.path.exists(str(cls.tfs_model_path)):
            shutil.rmtree(cls.tfs_model_path)
        if os.path.exists(str(cls.tfs_model_path) + '.zip'):
            os.remove(str(cls.tfs_model_path) + '.zip')

    if __name__ == '__main__':
        unittest.main()