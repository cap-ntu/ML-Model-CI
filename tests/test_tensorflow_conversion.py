#!/usr/bin/python3
# -*- coding: utf-8 -*-
import unittest
import onnxruntime
import tensorflow as tf
import numpy as np
from modelci.hub.converter import convert
import onnx
from modelci.types.bo import IOShape


class TestTensorflowConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        file = tf.keras.utils.get_file(
            "grace_hopper.jpg",
            "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
        img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
        cls.x = tf.keras.preprocessing.image.img_to_array(img)
        cls.x = tf.keras.applications.mobilenet.preprocess_input(
            cls.x[tf.newaxis, ...])
        shape = [1, 224, 224, 3]
        cls.inputs = IOShape(shape, tf.float32, 'input_0')
        labels_path = tf.keras.utils.get_file(
            'ImageNetLabels.txt',
            'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        cls.imagenet_labels = np.array(open(labels_path).read().splitlines())
        cls.tf_model = tf.keras.applications.MobileNet()
        result_before_save = cls.tf_model(cls.x)

        cls.decoded = cls.imagenet_labels[np.argsort(result_before_save)[0, ::-1][:5] + 1]

    def test_tensorflow_to_onnx(self):
        onnx_model = convert(self.tf_model, 'tensorflow', 'onnx', inputs=self.inputs)
        onnx.checker.check_model(onnx_model)
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        res = np.array(sess.run([output_name], {input_name: self.x}))
        onnxres = self.imagenet_labels[np.argsort(res)[0, ::-1][:5] + 1]
        pred = onnxres[-1][-5:]
        np.testing.assert_string_equal(str(pred[::-1]), str(self.decoded))
    if __name__ == '__main__':
        unittest.main()
