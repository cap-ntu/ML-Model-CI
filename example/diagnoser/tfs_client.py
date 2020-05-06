"""
Author: huangyz0918
Desc: template client for TF-Serving of ResNet-50
Date: 26/04/2020
"""

import grpc
import numpy as np
import requests

from modelci.metrics.benchmark.metric import BaseModelInspector

import tensorflow.compat.v1 as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('image', './cat.jpg', 'path to image in JPEG format')


class ExampleTFSClient(BaseModelInspector):
    def __init__(self, repeat_data, batch_num=1, batch_size=1, asynchronous=None):
        super().__init__(repeat_data=repeat_data, batch_num=batch_num, batch_size=batch_size, asynchronous=asynchronous)
        self.request = None
        self.stub = None

    def data_preprocess(self):
        pass

    def make_request(self, input_batch):
        FLAGS = tf.app.flags.FLAGS
        channel = grpc.insecure_channel(FLAGS.server)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = 'resnet'
        self.request.model_spec.signature_name = 'serving_default'

    def infer(self, input_batch):
        self.request.inputs['image_bytes'].CopyFrom(tf.make_tensor_proto(input_batch, shape=[len(input_batch)]))
        result = self.stub.Predict(self.request, 10.0)




