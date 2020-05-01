import argparse
import sys
import threading
import time

import grpc
import numpy as np
# tensorflow 2.0.0
import tensorflow as tf
# tensorflow-serving-api 2.0.0
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


class _ResultCounter(object):

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()
        self._start_time = -1
        self._end_time = 0

    def inc_done(self):
        with self._condition:
            self._done += 1
            if self._done == self._num_tests:
                self.set_end_time(time.time())
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def throttle(self):
        with self._condition:
            if self._start_time == -1:
                self._start_time = time.time()
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1

    def set_start_time(self, start_time):
        self._start_time = start_time

    def set_end_time(self, end_time):
        self._end_time = end_time

    def get_throughput(self):
        if self._end_time == 0:
            self.set_end_time(time.time())
        print("Latency: {}".format((self._end_time - self._start_time) / self._num_tests))
        return self._num_tests / (self._end_time - self._start_time)


def _create_rpc_callback(label, result_counter):
    def _callback(result_future):

        exception = result_future.exception()
        if exception:
            # result_counter.inc_error()
            print(exception)
        else:
            print('normal')
            sys.stdout.write('.')
            sys.stdout.flush()
            response = np.array(result_future.result().outputs['output'].float_val)
        result_counter.inc_done()
        result_counter.dec_active()

    return _callback


def inference(model: str,
              input_name: str,
              hostport: str = "0.0.0.0:8500",
              concurrency: int = 1,
              repeat: int = 100,
              signature: str = 'serving_default',
              test: bool = False):
    file = tf.keras.utils.get_file(
        "grace_hopper.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
    img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
    x = tf.keras.preprocessing.image.img_to_array(img)

    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    if test:
        result_counter = _ResultCounter(repeat, concurrency)
        for _ in range(repeat):
            request = predict_pb2.PredictRequest()
            request.model_spec.name = model
            request.model_spec.signature_name = signature
            request.inputs[input_name].CopyFrom(tf.make_tensor_proto(x, shape=[1, 224, 224, 3]))
            result_counter.throttle()
            result_future = stub.Predict.future(request, 5.0)
            result_future.add_done_callback(_create_rpc_callback(None, result_counter))
        throughput = result_counter.get_throughput()
        print("Throughput: {}".format(throughput))
    else:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model
        request.model_spec.signature_name = signature
        request.inputs[input_name].CopyFrom(tf.make_tensor_proto(x, shape=[1, 224, 224, 3]))
        result_future = stub.Predict.future(request, 5.0)
        labels_path = tf.keras.utils.get_file(
            'ImageNetLabels.txt',
            'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        imagenet_labels = np.array(open(labels_path).read().splitlines())
        response = np.array(result_future.result().outputs['probs'].float_val)
        decoded = imagenet_labels[np.argmax(response)]
        print(decoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GRPC client test.")
    parser.add_argument("--model", type=str, help="Model name.")
    parser.add_argument("--input_name", type=str, help="Input name.")
    parser.add_argument("--signature", default="serving_default", type=str, help="Signature.")
    parser.add_argument("--repeat", default=100, type=int, help="Repeat time.")
    parser.add_argument("--host", default="0.0.0.0:8500", type=str, help="Host IP.")
    parser.add_argument("--concurrency", default=1, type=int, help="Concurrent inference requests limit")
    parser.add_argument('-t', default=False, action="store_true", help="Test throughput")
    args = parser.parse_args()

    inference(args.model, args.input_name, args.host, args.concurrency, args.repeat, args.signature, args.t)
