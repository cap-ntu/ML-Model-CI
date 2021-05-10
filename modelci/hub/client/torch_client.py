"""
Author: huangyz0918
Author: Li Yuanming
Desc: template client for TorchScript of ResNet-50
Date: 26/04/2020
"""

import json
import time

import grpc
import torch
from torchvision import transforms

from modelci.hub.deployer.config import TORCHSCRIPT_GRPC_PORT
from modelci.metrics.benchmark.metric import BaseModelInspector
from modelci.types.bo import ModelBO
from modelci.types.models import MLModel
from modelci.types.proto.service_pb2 import InferRequest
from modelci.types.proto.service_pb2_grpc import PredictStub
from modelci.types.type_conversion import model_data_type_to_torch


class CVTorchClient(BaseModelInspector):
    SERVER_HOST = 'localhost'

    def __init__(self, repeat_data, model_info: MLModel, batch_num=1, batch_size=1, asynchronous=None):
        super().__init__(
            repeat_data=repeat_data,
            model_info=model_info,
            batch_num=batch_num,
            batch_size=batch_size,
            asynchronous=asynchronous
        )
        self.channel = grpc.insecure_channel(f'{self.SERVER_HOST}:{TORCHSCRIPT_GRPC_PORT}')
        self.stub = PredictStub(self.channel)

    def data_preprocess(self, x):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(255),
                transforms.CenterCrop(self.model_info.inputs[0].shape[2:]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                torch.Tensor.numpy
            ]
        )
        return transform(x)

    def make_request(self, input_batch):
        dtype = str(self.model_info.inputs[0].dtype).split('.')[1]
        meta = json.dumps(
            {'shape': self.model_info.inputs[0].shape[1:], 'dtype': dtype, 'torch_flag': True}
        )
        request = InferRequest()
        request.model_name = self.model_info.architecture
        request.meta = meta
        request.raw_input.extend(list(map(bytes, input_batch)))
        return request

    def check_model_status(self) -> bool:
        """TODO: wait for status API for TorchServing."""
        time.sleep(5)
        return True

    def infer(self, request):
        self.stub.Infer(request)
