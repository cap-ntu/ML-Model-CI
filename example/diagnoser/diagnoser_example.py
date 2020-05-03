'''
An example usage of diagnoser.py
'''

from client_example import TestModelClient
from modelci.hub.diagnoser import Diagnoser

if __name__ == "__main__":

    # Fack data for testing
    fake_image_data = []
    for i in range(6400): # number of fake data
        with open('./cat.jpg', 'rb') as f:
            fake_image_data.append(f.read())

    # custom client here
    testClient = TestModelClient(raw_data=fake_image_data, batch_size=32, asynchronous=False)

    # Diagnoser usage
    diagnoser = Diagnoser(inspector=testClient, server_name='tfs')
    diagnoser.diagnose(batch_size=1)
    # diagnoser.diagnose_all_batches() # run all 1, 2, 4, 8, 16, 32, 64, 128 batch size 