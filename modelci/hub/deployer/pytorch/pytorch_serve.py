import io
import json
import os
import sys
from concurrent import futures
from pathlib import Path

import grpc
import torch.jit
import torchvision.transforms as transforms
import uvicorn
from PIL import Image
from fastapi import FastAPI
from fastapi import File
from proto import service_pb2_grpc, service_pb2

# get path
model_base_dir = Path('/model') / sys.argv[1]
# get valid version sub dir
model_dir = list(filter(lambda x: os.path.isfile(x) and str(x.stem).isdigit(), model_base_dir.glob('**/*')))

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the latest version of a TorchScript model
model = torch.jit.load(str(max(model_dir))).to(device)
model.eval()

imagenet_class_index = json.load(open('imagenet_class_index.json'))


def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes).to(device)
    outputs = model(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


class PredictServicer(service_pb2_grpc.PredictServicer):
    def predict(self, request, context):
        response = service_pb2.PredictionReceive()
        class_id, class_name = get_prediction(request.imgBuf)
        response.classId = class_id
        response.className = class_name

        return response


def grpc_serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_PredictServicer_to_server(
        PredictServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()


app = FastAPI(title=sys.argv[1], openapi_url="/openapi.json")


@app.get("/")
def index():
    return "Hello World!"


@app.post('/predict')
async def predict(img_file: bytes = File(...)):
    class_id, class_name = get_prediction(img_file)
    response = {'class_id': class_id, 'class_name': class_name}
    return response


if __name__ == '__main__':
    grpc_serve()
    uvicorn.run(app, host='0.0.0.0', port=8000)
