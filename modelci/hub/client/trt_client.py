"""
Author: huangyz0918
Desc: template client for TensorRT Serving of ResNet-50
Date: 26/04/2020
"""

from tensorrtserver.api import InferContext, ProtocolType

from modelci.data_engine.preprocessor import image_classification_preprocessor
from modelci.hub.deployer.config import TRT_GRPC_PORT
from modelci.hub.utils import parse_trt_model
from modelci.metrics.benchmark.metric import BaseModelInspector


class CVTRTClient(BaseModelInspector):
    '''
    Tested sub-class for BaseModelInspector to implement a custom model runner.
    '''

    def __init__(self, repeat_data, batch_num=1, batch_size=1, asynchronous=None):
        self.input_name = None
        self.output_name = None
        super().__init__(repeat_data=repeat_data, batch_num=batch_num, batch_size=batch_size, asynchronous=asynchronous)

    def data_preprocess(self):
        self.input_name, self.output_name, c, h, w, format, dtype = parse_trt_model(f"localhost:{TRT_GRPC_PORT}",
                                                                                    ProtocolType.from_str('gRPC'),
                                                                                    'ResNet50', self.batch_size, False)
        self.processed_data = image_classification_preprocessor(self.raw_data, format, dtype, c, h, w, 'NONE')

    def infer(self, input_batch):
        ctx = InferContext(f"localhost:{TRT_GRPC_PORT}", ProtocolType.from_str('gRPC'), 'ResNet50', -1, False, 0, False)
        ctx.run({self.input_name: input_batch}, {self.output_name: (InferContext.ResultFormat.CLASS, 1)},
                self.batch_size)
