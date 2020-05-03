"""
Author: huangyz0918
Desc: sample client for TF-Serving of ResNet-50
Date: 26/04/2020
"""

import grpc
import numpy as np
import requests

from modelci.metrics.benchmark.metric import BaseModelInspector, BaseDataWrapper

import tensorflow.compat.v1 as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('image', './cat.jpg', 'path to image in JPEG format')


class TestDataWrapper(BaseDataWrapper):
    '''
    Tested sub-class for DataWrapper to implement a custom data preprocessing.
    '''

    def __init__(self, meta_data_url, raw_data, batch_size=None):
        super().__init__(meta_data_url, raw_data, batch_size)

    def data_preprocess(self):
        return self.raw_data # don't do any data preprocess


class TestModelClient(BaseModelInspector):
    '''
    Tested sub-class for BaseModelInspector to implement a custom model runner.
    '''
    def __init__(self, data_wrapper:BaseDataWrapper, asynchronous=None):
        super().__init__(data_wrapper=data_wrapper, asynchronous=asynchronous)
        self.request = None
        self.stub = None

    def server_batch_request(self):
        '''
        should batch the data from server side, leave blank if don't.
        '''
        pass

    def setup_inference(self):
        '''
        setup inference method, you can setup some requests here, implemented from parent class.
        '''
        FLAGS = tf.app.flags.FLAGS
        channel = grpc.insecure_channel(FLAGS.server)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = 'resnet'
        self.request.model_spec.signature_name = 'serving_default'

    def infer(self, input_batch):
        '''
        inference method, implemented from parent class.
        '''
        self.request.inputs['image_bytes'].CopyFrom(tf.make_tensor_proto(input_batch, shape=[len(input_batch)]))
        result = self.stub.Predict(self.request, 10.0)




