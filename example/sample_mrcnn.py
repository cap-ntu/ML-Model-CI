"""
Mask R-CNN registering, managing, serving and profiling code demo using ModelCI.
"""

from PIL import Image

from modelci.hub.client.tfs_client import CVTFSClient
from modelci.hub.manager import retrieve_model, register_model_from_yaml
from modelci.hub.profiler import Profiler
from modelci.types.bo import Engine, Framework

if __name__ == "__main__":
    test_img = Image.open("path to the test data")

    register_model_from_yaml("path to your yaml file")  # register your model in the database
    model_info = retrieve_model(  # retrieve model information
        architecture_name='MRCNN',
        framework=Framework.TENSORFLOW,
        engine=Engine.TFS
    )[0]

    tfs_client = CVTFSClient(
        test_img, batch_num=100, batch_size=32, asynchronous=True, model_info=model_info
    )

    profiler = Profiler(model_info=model_info, server_name='tfs', inspector=tfs_client)
    profiler.diagnose(device='cuda:0')  # profile batch size 32
