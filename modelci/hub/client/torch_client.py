"""
Author: huangyz0918
Desc: template client for TorchScript of ResNet-50
Date: 26/04/2020
"""

import torch
import grpc, json
import numpy as np
from toolz import compose
from torchvision import transforms

from proto.service_pb2 import InferRequest
from proto.service_pb2_grpc import PredictStub

from modelci.utils.misc import json_update
from modelci.persistence.bo.type_conversion import type_to_data_type
from modelci.metrics.benchmark.metric import BaseModelInspector


class CVTorchClient(BaseModelInspector):
    def __init__(self, repeat_data, batch_num=1, batch_size=1, asynchronous=None):
        super().__init__(repeat_data=repeat_data, batch_num=batch_num, batch_size=batch_size, asynchronous=asynchronous)
        self.batches = self.__client_batch_request() # FIXME: creating batches twice will increase the data preprocessing time
        self.stub = PredictStub(grpc.insecure_channel("localhost:8001"))

    def data_preprocess(self):
        pass

    def __client_batch_request(self):
        print('TorchScript: start data preprocessing...')
        batches = []
        batch = np.repeat(self.raw_data[np.newaxis, :, :, :], self.batch_size, axis=0)
        for i in range(self.batch_num):
            batches.append(self.transform_image(images=batch))
        return batches

    def transform_image(self, images):
        t = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return list(map(t, images))

    def infer(self, input_batch):
        example = input_batch[0]
        meta = dict()
        to_byte = compose(bytes, torch.Tensor.numpy)
        raw_input = list(map(to_byte, input_batch))
        shape = example.shape
        dtype = type_to_data_type(example.dtype).value
        meta = json_update({'shape': shape, 'dtype': dtype, 'torch_flag': True}, meta)
        self.stub.Infer(InferRequest(model_name='resnet50', raw_input=raw_input, meta=json.dumps(meta)))