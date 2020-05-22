'''
An example usage of diagnoser.py
'''
import cv2
import numpy as np
from PIL import Image
from tfs_client import CVTFSClient

from modelci.hub.diagnoser import Diagnoser
from modelci.hub.manager import retrieve_model_by_name
from modelci.persistence.bo.model_objects import Engine, Framework

if __name__ == "__main__":
    # Fake data for testing
    data_path = './data/cat.jpg'

    # for TensorFlow Serving
    test_img_bytes = None
    with open(data_path, 'rb') as f:
        test_img_bytes = f.read()

    # for TensorRT Serving
    test_img = Image.open(data_path)

    # for TorchScript and ONNX
    test_img_ndarray: np.ndarray = cv2.imread(data_path)
    # input = torch.randn(1, 3, 224, 224)

    # init clients for different serving platforms, you can custom a client by implementing the BaseModelInspector class.
    tfs_client = CVTFSClient(test_img_bytes, batch_num=100, batch_size=32, asynchronous=True)
    # trt_client = CVTRTClient(test_img, batch_num=100, batch_size=32, asynchronous=False)
    # torch_client = CVTorchClient(test_img_ndarray, batch_num=100, batch_size=32, asynchronous=False)
    # onnx_client = CVONNXClient(test_img_ndarray, batch_num=100, batch_size=32, asynchronous=False)

    # model_path = '../resnet50_explicit_path.yml'
    # register_model_from_yaml(model_path)

    mode_info = retrieve_model_by_name(architecture_name='ResNet50', framework=Framework.PYTORCH,
                                       engine=Engine.TORCHSCRIPT)
    diagnoser = Diagnoser(model_info=mode_info, server_name='tfs', inspector=tfs_client)
    # diagnoser.diagnose()
    # diagnoser.diagnose(batch_size=1) # you can use a new batch_size to overwrite the client's.
    # diagnoser.diagnose_all_batches([1, 2, 4, 8, 16, 32]) # run all 1, 2, 4, 8, 16, 32 batch size 

    diagnoser.auto_diagnose([2, 4, 16])
