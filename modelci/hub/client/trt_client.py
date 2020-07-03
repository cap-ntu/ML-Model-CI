"""
Author: huangyz0918
Author: Li Yuanming
Desc: template client for TensorRT Serving of ResNet-50
Date: 26/04/2020
"""

from tensorrtserver.api import InferContext, ProtocolType, ServerStatus, ServerStatusContext

from modelci.data_engine.preprocessor import image_classification_preprocessor
from modelci.hub.deployer.config import TRT_GRPC_PORT
from modelci.metrics.benchmark.metric import BaseModelInspector
from modelci.types.bo import ModelBO


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
        input_ = self.model_info.inputs[0]
        return image_classification_preprocessor(
            self.raw_data, input_.format, input_.dtype, input_.channel, input_.height, input_.width, 'NONE'
        )

    def infer(self, input_batch):
        input_ = {self.model_info.inputs[0].name: input_batch}
        output = {self.model_info.outputs[0].name: (InferContext.ResultFormat.CLASS, 1)}
        name = self.model_info.name
        version = self.model_info.version.ver
        ctx = InferContext(self.SERVER_URI, ProtocolType.GRPC, name, version)
        ctx.run(input_, output, self.batch_size)

    def check_model_status(self) -> bool:
        ctx = ServerStatusContext(self.SERVER_URI, ProtocolType.GRPC, self.model_info.name)
        server_status: ServerStatus = ctx.get_server_status()

        if self.model_info.name in server_status.model_status:
            return True
        else:
            return False
