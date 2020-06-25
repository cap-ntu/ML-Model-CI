"""
Author: huangyz0918
Author: Li Yuanming
Desc: template client for TF-Serving of ResNet-50
Date: 26/04/2020
"""
from typing import List

import grpc
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from modelci.hub.deployer.config import TFS_GRPC_PORT
from modelci.metrics.benchmark.metric import BaseModelInspector
from modelci.types.bo import IOShape


class CVTFSClient(BaseModelInspector):
    def __init__(
            self,
            repeat_data,
            model_name: str,
            inputs: List[IOShape],
            batch_num=1,
            batch_size=1,
            asynchronous=None,
            signature_name: str = 'serving_default'
    ):
        super().__init__(repeat_data=repeat_data, batch_num=batch_num, batch_size=batch_size, asynchronous=asynchronous)
        tf.app.flags.DEFINE_string('server', f'0.0.0.0:{TFS_GRPC_PORT}', 'PredictionService host:port')
        tf.app.flags.DEFINE_string('image', './data/cat.jpg', 'path to image in JPEG format')
        self.request = None
        self.stub = None
        self.model_name = model_name
        self.inputs = inputs
        self.signature_name = signature_name

    def data_preprocess(self):
        pass

    def make_request(self, input_batch):
        flags = tf.app.flags.FLAGS
        channel = grpc.insecure_channel(flags.server)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = self.model_name
        self.request.model_spec.signature_name = self.signature_name

    def infer(self, input_batch):
        input_batch = np.stack(input_batch)
        for input_ in self.inputs:
            tensor_proto = tf.make_tensor_proto(input_batch, shape=input_batch.shape)
            self.request.inputs[input_.name].CopyFrom(tensor_proto)
        self.stub.Predict(self.request, 10.0)
