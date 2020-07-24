import json
import os
import sys
from concurrent import futures
from functools import partial
from pathlib import Path
from typing import Iterable

import grpc
import numpy as np
import torch.jit
from grpc._cython import cygrpc
from proto import service_pb2_grpc
from proto.service_pb2 import InferResponse
from proto.service_pb2_grpc import add_PredictServicer_to_server
from toolz import compose
from utils import model_data_type_to_np, DataType


class ServingEngine(object):
    def __init__(self):
        model_base_dir = Path('/models') / sys.argv[1]
        # get valid version sub dir
        model_dir = list(filter(lambda x: os.path.isfile(x) and str(x.stem).isdigit(), model_base_dir.glob('**/*')))
        # set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load the latest version of a TorchScript model
        self.model = torch.jit.load(str(max(model_dir)), map_location=self.device)
        self.model.eval()

    def batch_predict(self, inputs: torch.Tensor):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)

        return outputs.cpu().detach().tolist()


class PredictServicer(service_pb2_grpc.PredictServicer):

    def __init__(self):
        self.engine = ServingEngine()

        print('Finish loading')

    @classmethod
    def grpc_decode(cls, buffer: Iterable, meta):
        meta = json.loads(meta)
        shape = meta['shape']
        dtype = model_data_type_to_np(DataType(meta['dtype']))
        torch_flag = meta['torch_flag']

        decode_pipeline = compose(
            partial(np.reshape, newshape=shape),
            partial(np.fromstring, dtype=dtype),
        )

        buffer = list(map(decode_pipeline, buffer))

        buffer = np.stack(buffer)

        if torch_flag:
            buffer = torch.from_numpy(buffer)
        return buffer

    def Infer(self, request, context):
        raw_input = request.raw_input
        meta = request.meta
        inputs = self.grpc_decode(raw_input, meta=meta)
        result = self.engine.batch_predict(inputs)

        return InferResponse(json=json.dumps(result))

    def StreamInfer(self, request_iterator, context):
        raise NotImplementedError()


def grpc_serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8), options=[
            (cygrpc.ChannelArgKey.max_send_message_length, -1),
            (cygrpc.ChannelArgKey.max_receive_message_length, -1)
        ]
    )
    servicer = PredictServicer()
    add_PredictServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:8001')
    server.start()
    print('Listening on port 8001')
    server.wait_for_termination()


# app = FastAPI(title=sys.argv[1], openapi_url="/openapi.json")


# @app.get("/")
# def index():
#     return "Hello World!"
#
#
# @app.post('/predict')
# async def predict(img_file: bytes = File(...)):
#     class_id, class_name = get_prediction(img_file)
#     response = {'class_id': class_id, 'class_name': class_name}
#     return response


if __name__ == '__main__':
    grpc_serve()
    # uvicorn.run(app, host='0.0.0.0', port=8000)
