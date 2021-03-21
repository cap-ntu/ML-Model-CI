#!/usr/bin/python3
# -*- coding: utf-8 -*-
import unittest

import onnxruntime

from modelci.utils import trt_builder
import tensorflow as tf
import numpy as np
from modelci.hub.converter import convert
import onnx
import os
import tempfile
class TestTensorflowConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tmpdir = tempfile.mkdtemp()

        file = tf.keras.utils.get_file(
            "grace_hopper.jpg",
            "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
        img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
        cls.x = tf.keras.preprocessing.image.img_to_array(img)
        cls.x = tf.keras.applications.mobilenet.preprocess_input(
            cls.x[tf.newaxis, ...])
        cls.shape = [1,224,224,3]
        labels_path = tf.keras.utils.get_file(
            'ImageNetLabels.txt',
            'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        cls.imagenet_labels = np.array(open(labels_path).read().splitlines())
        cls.tf_model = tf.keras.applications.MobileNet()
        result_before_save = cls.tf_model(cls.x)
        cls.mobilenet_save_path = os.path.join(tmpdir, "mobilenet/1/")
        tf.saved_model.save(cls.tf_model, cls.mobilenet_save_path)

        cls.decoded = cls.imagenet_labels[np.argsort(result_before_save)[0, ::-1][:5] + 1]

    def test_tensorflow_to_onnx(self):
        onnx_model = convert(self.mobilenet_save_path, 'tensorflow', 'onnx')
        onnx.checker.check_model(onnx_model)
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        res = np.array(sess.run([output_name], {input_name: self.x}))
        onnxres = self.imagenet_labels[np.argsort(res)[0, ::-1][:5] + 1]
        pred = onnxres[-1][-5:]
        np.testing.assert_string_equal(str(pred[::-1]), str(self.decoded))



    def test_tensorflow_to_trt(self):
        engine = convert(self.mobilenet_save_path, 'tensorflow', 'trt', shape=self.shape)
        inputs, outputs, bindings, stream = trt_builder.allocate_buffers(engine)

        with engine.create_execution_context() as context:
            imput_img = self.x.reshape(150528)
            np.copyto(inputs[0].host, imput_img)
            output = trt_builder.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            pred = self.imagenet_labels[np.argsort(output)[0,::-1][:5]+1]
   
        np.testing.assert_string_equal(str(pred), str(self.decoded))


    if __name__ == '__main__':
        unittest.main()
