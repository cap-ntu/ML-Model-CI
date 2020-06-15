"""
Mask R-CNN registering, managing, serving and profiling code demo using ModelCI.
"""

from PIL import Image

from modelci.hub.client.tfs_client import CVTFSClient
from modelci.hub.manager import retrieve_model_by_name, register_model_from_yaml
from modelci.hub.profiler import Profiler
from modelci.types.bo import Engine, Framework

if __name__ == "__main__":
    test_img = Image.open("path to the test data")
    tfs_client = CVTFSClient(test_img, batch_num=100, batch_size=32, asynchronous=True)

    register_model_from_yaml("path to your yaml file")  # register your model in the database
    mode_info = retrieve_model_by_name(  # retrieve model information
        architecture_name='MRCNN',
        framework=Framework.TENSORFLOW,
        engine=Engine.TFS
    )

    profiler = Profiler(model_info=mode_info, server_name='tfs', inspector=tfs_client)
    profiler.diagnose()  # profile batch size 32 in all the available devices
