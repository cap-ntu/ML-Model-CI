import io
import json
import os
import sys
from pathlib import Path

import onnx
import onnxruntime
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, File
from torchvision import transforms

# get path
model_base_dir = Path('/model') / sys.argv[1]
# get valid version sub dir
model_dir = list(filter(lambda x: os.path.isfile(x) and str(x.stem).isdigit(), model_base_dir.glob('**/*')))

onnx_model = onnx.load(str(max(model_dir)))
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(str(max(model_dir)))

imagenet_class_index = json.load(open('imagenet_class_index.json'))


def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_prediction(image_bytes):
    x = to_numpy(transform_image(image_bytes))
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    y_hat = ort_outs[0].flatten().argmax()
    predicted_idx = str(y_hat)
    return imagenet_class_index[predicted_idx]


app = FastAPI(title=sys.argv[1], openapi_url="/openapi.json")


@app.post('/predict')
async def predict(img_file: bytes = File(...)):
    class_id, class_name = get_prediction(img_file)
    response = {'class_id': class_id, 'class_name': class_name}
    return response


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
