"""
Mask R-CNN registering, managing, serving and profiling code demo using ModelCI.
"""
import requests
from PIL import Image

from modelci.config import app_settings
from modelci.hub.client.tfs_client import CVTFSClient
from modelci.hub.profiler import Profiler
from modelci.hub.registrar import register_model_from_yaml
from modelci.persistence import service_
from modelci.types.bo import Engine, Framework

if __name__ == "__main__":
    test_img = Image.open("path to the test data")

    register_model_from_yaml("resnet50_explicit_path.yml")  # register your model in the database
    with requests.get(f'{app_settings.api_v1_prefix}/model/') as r:
        model_list = r.json()
    model_id = None
    for model in model_list:
        if model["architecture"] == "ResNet18":
            model_id = model["id"]
            break
    if model_id is None:
        raise ValueError("Test model is not registered!")
    model_info = service_.get_by_id(model_id)


    tfs_client = CVTFSClient(
        test_img, batch_num=100, batch_size=32, asynchronous=True, model_info=model_info
    )

    profiler = Profiler(model_info=model_info, server_name='tfs', inspector=tfs_client)
    profiler.diagnose(device='cuda:0')  # profile batch size 32
