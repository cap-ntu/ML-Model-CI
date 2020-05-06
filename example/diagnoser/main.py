'''
An example usage of diagnoser.py
'''
import cv2 
import numpy as np
from modelci.hub.diagnoser import Diagnoser

from tfs_client import ExampleTFSClient
from torch_client import ExampleTorchClient
from onnx_client import ExampleONNXClient

if __name__ == "__main__":

    # Fake data for testing
    data_path = './data/cat.jpg'

    # for TensorFlow Serving
    test_img_bytes = None
    with open(data_path, 'rb') as f:
        test_img_bytes = f.read()

    # for TorchScript and ONNX
    test_img_ndarray: np.ndarray = cv2.imread(data_path)

    # init clients for different serving platforms, you can custom a client by implementing the BaseModelInspector class.
    tfs_client = ExampleTFSClient(test_img_bytes, batch_num=100, batch_size=32, asynchronous=False)
    torch_client = ExampleTorchClient(test_img_ndarray, batch_num=100, batch_size=32, asynchronous=False)
    onnx_client = ExampleONNXClient(test_img_ndarray, batch_num=100, batch_size=32, asynchronous=False)

    diagnoser = Diagnoser(inspector=onnx_client, server_name='onnxs')
    diagnoser.diagnose()
    # diagnoser.diagnose(batch_size=1) # you can use a new batch_size to overwrite the client's.
    # diagnoser.diagnose_all_batches() # run all 1, 2, 4, 8, 16, 32, 64, 128 batch size 