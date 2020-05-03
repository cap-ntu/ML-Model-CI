"""
Author: huangyz0918
Desc: sample testing script for metric.py
Date: 26/04/2020
"""

import grpc
import numpy as np
import requests

from metric import BaseDataWrapper, BaseModelInspector

import tensorflow.compat.v1 as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('image', './sample_data/cat.jpg', 'path to image in JPEG format')


class TestModelInspector(BaseModelInspector):
    '''
    Tested sub-class for BaseModelInspector to implement a custom model runner.
    '''
    def __init__(self, raw_data:list, batch_size=1, asynchronous=None):
        super().__init__(raw_data=raw_data, batch_size=batch_size, asynchronous=asynchronous)
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

# Tests
if __name__ == "__main__":
    fake_image_data = []
    for i in range(6400): # number of fake data
        with open('./sample_data/cat.jpg', 'rb') as f:
            fake_image_data.append(f.read())

    # test functions, set batch size and other parameter here.
    testModelInspector = TestModelInspector(raw_data=fake_image_data, batch_size=32, asynchronous=True)
    testModelInspector.run_model('tfserving_resnet') 

