"""
Author: huangyz0918
Author: Li Yuanming
Desc: template client for TF-Serving of ResNet-50
Date: 26/04/2020
"""
import sys

import cv2
import grpc
import numpy as np
import requests
import tensorflow.compat.v1 as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from modelci.hub.deployer.config import TFS_GRPC_PORT, TFS_HTTP_PORT
from modelci.metrics.benchmark.metric import BaseModelInspector
from modelci.types.bo import ModelBO
from modelci.types.type_conversion import model_data_type_to_np


class CVTFSClient(BaseModelInspector):
    SERVER_HOST = 'localhost'

    def __init__(
            self,
            repeat_data,
            model_info: ModelBO,
            batch_num=1,
            batch_size=1,
            asynchronous=None,
            signature_name: str = 'serving_default',
    ):
        super().__init__(
            repeat_data=repeat_data,
            model_info=model_info,
            batch_num=batch_num,
            batch_size=batch_size,
            asynchronous=asynchronous
        )

        self.stub = None
        self.version = model_info.version
        self.signature_name = signature_name

    def data_preprocess(self, x):
        input_ = self.model_info.inputs[0]
        dtype = model_data_type_to_np(input_.dtype)
        return cv2.resize(x, (input_.height, input_.width)).astype(dtype)

    def make_request(self, input_batch):
        channel = grpc.insecure_channel(f'{self.SERVER_HOST}:{TFS_GRPC_PORT}')
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        input_batch = np.stack(input_batch)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_info.name
        request.model_spec.signature_name = self.signature_name
        for input_ in self.model_info.inputs:
            tensor_proto = tf.make_tensor_proto(input_batch, shape=input_batch.shape)
            request.inputs[input_.name].CopyFrom(tensor_proto)

        return request

    def check_model_status(self) -> bool:
        api_url = f'http://{self.SERVER_HOST}:{TFS_HTTP_PORT}/v1/models/{self.model_info.name}/versions/{self.version}'
        try:
            response = requests.get(api_url)
            state = response.json()['model_version_status'][0]['state']
            return state == 'AVAILABLE'
        except (requests.exceptions.BaseHTTPError, requests.exceptions.ConnectionError) as e:
            print(e, file=sys.stderr)
            return False
        except AttributeError as e:
            print(e, file=sys.stderr)
            return False

    def infer(self, request):
        self.stub.Predict(request, 10.0)
