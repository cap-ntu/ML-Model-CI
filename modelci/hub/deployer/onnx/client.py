import json
from typing import Generator, Iterable, Union

import cv2
import grpc
import numpy as np
import torch
from toolz import compose
from torchvision import transforms

from modelci.persistence.bo.type_conversion import type_to_data_type
from modelci.utils.misc import json_update
from proto.service_pb2 import InferRequest
from proto.service_pb2_grpc import PredictStub


class RpcClient(object):
    def __init__(self, port):
        self.channel = grpc.insecure_channel("localhost:" + str(port))
        self.stub = PredictStub(self.channel)

    def service_request(
            self,
            request: Union[InferRequest, Generator[None, InferRequest, None]],
            stream=False
    ):
        if stream:
            return self._service_request_stream(request)
        else:
            return self._service_request(request)

    def _service_request(self, request: InferRequest):
        response = self.stub.Infer(request)
        return response

    def _service_request_stream(self, request_generator: Generator[None, InferRequest, None]):
        responses = self.stub.StreamInfer(request_generator)
        return responses

    @staticmethod
    def make_request(model_name, inputs: Iterable[Union[np.ndarray, torch.Tensor]], meta=None):
        inputs = list(inputs)
        example = inputs[0]
        if meta is None:
            meta = dict()

        if isinstance(example, np.ndarray):
            to_byte = bytes
            torch_flag = False
        elif isinstance(example, torch.Tensor):
            to_byte = compose(bytes, torch.Tensor.numpy)
            torch_flag = True
        else:
            raise ValueError(
                'Argument `image` is expected to be an iterative numpy array, or an iterative torch Tensor')

        raw_input = list(map(to_byte, inputs))
        shape = example.shape
        dtype = type_to_data_type(example.dtype).value
        meta = json_update({'shape': shape, 'dtype': dtype, 'torch_flag': torch_flag}, meta)

        return InferRequest(model_name=model_name, raw_input=raw_input, meta=json.dumps(meta))

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def transform_image(images):
    my_transforms = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
         transforms.Normalize(
             [0.485, 0.456, 0.406],
             [0.229, 0.224, 0.225]),
         torch.Tensor.numpy]
    )
    images = map(my_transforms, images)

    return RpcClient.make_request(model_name='resnet50', inputs=images)


def mock_frame_fetch(data_path, batch_size):
    image: np.ndarray = cv2.imread(data_path)
    # expend HWC to NHWC and repeat to form a batch
    images = np.repeat(image[np.newaxis, :, :, :], batch_size, axis=0)

    return images


if __name__ == '__main__':
    batch_size = 8

    raw_images = mock_frame_fetch('img_bigbang_scene.jpg', batch_size=batch_size)
    with RpcClient(port='8001') as rpc_client:
        request = transform_image(images=raw_images)
        response = rpc_client.service_request(request)

        print(response)
