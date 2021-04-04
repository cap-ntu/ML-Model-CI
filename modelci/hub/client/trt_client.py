"""
Author: huangyz0918
Author: Li Yuanming
Desc: template client for TensorRT Serving of ResNet-50
Date: 26/04/2020
"""
import sys
import warnings

import cv2

from modelci.metrics.benchmark.metric import BaseModelInspector
from modelci.types.bo import ModelBO
from modelci.types.type_conversion import model_data_type_to_np
from ..deployer.config import TRT_GRPC_PORT

try:
    from tensorrtserver.api import (
        InferContext, ProtocolType, ServerStatus, ServerStatusContext, InferenceServerException
    )
except ModuleNotFoundError:
    warnings.warn('Module `tensorrtserver` not installed. You are not able to use TRT Client.')


class CVTRTClient(BaseModelInspector):
    """Tested sub-class for BaseModelInspector to implement a custom model runner."""

    SERVER_URI = f'localhost:{TRT_GRPC_PORT}'

    def __init__(
            self,
            repeat_data,
            model_info: ModelBO,
            batch_num=1,
            batch_size=1,
            asynchronous=None
    ):
        super().__init__(
            repeat_data=repeat_data,
            model_info=model_info,
            batch_num=batch_num,
            batch_size=batch_size,
            asynchronous=asynchronous
        )

    def data_preprocess(self, x):
        """Resize the inputs into given size and data type."""
        input_ = self.model_info.inputs[0]
        dtype = model_data_type_to_np(input_.dtype)
        return cv2.resize(x, (input_.height, input_.width)).astype(dtype)

    def make_request(self, input_batch):
        input_ = {self.model_info.inputs[0].name: input_batch}
        output = {self.model_info.outputs[0].name: (InferContext.ResultFormat.CLASS, 1)}

        return input_, output

    def infer(self, request):
        name = self.model_info.architecture
        version = self.model_info.version.ver
        ctx = InferContext(self.SERVER_URI, ProtocolType.GRPC, name, version)
        ctx.run(request[0], request[1], self.batch_size)

    def check_model_status(self) -> bool:
        name = self.model_info.architecture
        version = self.model_info.version.ver
        ctx = ServerStatusContext(self.SERVER_URI, ProtocolType.GRPC, self.model_info.architecture)
        try:
            server_status: ServerStatus = ctx.get_server_status()
            if server_status.model_status[name].version_status[version].ready_state == 1:
                return True
            else:
                return False
        except InferenceServerException as e:
            print(e, file=sys.stderr)
            return False
