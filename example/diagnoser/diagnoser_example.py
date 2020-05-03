'''
An example usage of diagnoser.py
'''

from client_example import TestDataWrapper, TestModelClient
from modelci.hub.diagnoser import Diagnoser

if __name__ == "__main__":
    # meta data and url for testing
    meta_data_url = 'http://localhost:8501/v1/models/resnet'
    fake_image_data = []
    for i in range(6400): # number of fake data
        with open('./cat.jpg', 'rb') as f:
            fake_image_data.append(f.read())

    # test functions, set batch size and other parameter here.
    testDataWrapper = TestDataWrapper(meta_data_url=meta_data_url, raw_data=fake_image_data, batch_size=32) 
    testClient = TestModelClient(testDataWrapper, asynchronous=True)
    diagnoser = Diagnoser(testClient, 'tfs')
    diagnoser.diagnose()