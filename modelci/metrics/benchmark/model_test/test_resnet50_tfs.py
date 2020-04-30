"""
Author: huangyz0918
Desc: A demo script for showing the testing methods of latency and 
throughput using ResNet and TF-Serving.
Date: 28/04/2020
"""

import grpc
import numpy as np
import requests
import yaml
from time import time, sleep
from multiprocessing import Process, cpu_count, Pool

import tensorflow.compat.v1 as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')

# overall paramters here
BATCH_SIZE = 32
DATA_LENGTH = 6400
ASYNCHRONOUS = False

def client_batch_request(raw_data, batch_size):
    '''
    Batching input data according to the specific batch size
    '''
    batches = []
    data_item_id = 0
    last_request = False
    while not last_request:
        input_batch = []
        for idx in range(batch_size):
            input_batch.append(raw_data[data_item_id])
            data_item_id = (data_item_id + 1) % len(raw_data)
            if data_item_id == 0:
                last_request = True
        batches.append(input_batch)
    return batches

def infer(batch):
    # ======= Init gRPC request (a batch per request)=======
    FLAGS = tf.app.flags.FLAGS
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'
    request.model_spec.signature_name = 'serving_default'

    # ======= Start inference and counting time =======
    infer_start_time = time()
    request.inputs['image_bytes'].CopyFrom(tf.make_tensor_proto(batch, shape=[len(batch)])) # copying data
    result = stub.Predict(request, 10.0)
    infer_end_time = time()
    return infer_end_time - infer_start_time # return a request (a batch imgs) latency

def inference_callback(a_batch_latency):
    """
    A callback function which handles the results of a asynchronous inference request
    @param elapsed_time: The amount of required for the inference request to complete
    """
    latencies.append(a_batch_latency) 
    a_batch_throughput =  BATCH_SIZE / a_batch_latency
    throughputs.append(a_batch_throughput)
    print(" a_batch_latency: {:.4f}".format(a_batch_latency), 'sec', " a_batch_throughput: {:.4f}".format(a_batch_throughput), ' req/sec')

# ======= Init dicts for restoring results =======
throughputs = []
latencies = []

if __name__ == "__main__":
    pool = Pool(processes=(cpu_count() * 24))

    # ======= Fake image inference data for testing =======
    fake_image_data = []
    for i in range(DATA_LENGTH):
        with open('../sample_data/cat.jpg', 'rb') as f:
            fake_image_data.append(f.read())

    # ======= Batching data before sending requests =======
    data_batches = client_batch_request(raw_data=fake_image_data, batch_size=BATCH_SIZE)

    # ======= warm-up =======
    print("start warm-up...")
    if len(data_batches) > 10:
        warm_up_batches = data_batches[:10]
        for batch in warm_up_batches:
            infer(batch)
    else:
        raise ValueError("Not enough test values, try to make more testing data.")
    print("warm-up finished, start inference...")

    all_batch_start_time = time()
    for batch in data_batches:
        if ASYNCHRONOUS:
            pool.apply_async(infer, args=(batch, ), callback=inference_callback)
        else:
            a_batch_latency = infer(batch)
            a_batch_throughput =  len(batch) / a_batch_latency
            throughputs.append(a_batch_throughput)
            latencies.append(a_batch_latency)
            print(" a_batch_latency: {:.4f}".format(a_batch_latency), 'sec', " a_batch_throughput: {:.4f}".format(a_batch_throughput), ' req/sec')

    # ======= Waitting until all the requests have responses =======
    while len(latencies) != len(data_batches):
        pass

    all_batch_end_time = time()

    all_batch_latency = all_batch_end_time - all_batch_start_time
    all_batch_throughput = len(fake_image_data) / all_batch_latency

    twenty_fifth_percentile = np.percentile(latencies, 25)
    fiftieth_percentile = np.percentile(latencies, 50)
    seventy_fifth_percentile = np.percentile(latencies, 75)

    print("all_batch_latency: ", all_batch_latency, 'sec')
    print("all_batch_throughput: ", all_batch_throughput, ' req/sec')
    print(f'overall 25th-percentile latiency: {twenty_fifth_percentile} s')
    print(f'overall 50th-percentile latiency: {fiftieth_percentile} s')
    print(f'overall 75th-percentile latiency: {seventy_fifth_percentile} s')


