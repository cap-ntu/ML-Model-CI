import argparse
import json
import time

import numpy as np
import requests
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make image request.")
    parser.add_argument('-t', default=False, action="store_true", help="Test throughput and latency mode.")
    parser.add_argument('--model', type=str, help="Model name.")
    parser.add_argument('--port', type=str, help="Port number.")
    parser.add_argument('--repeat', type=int, default=100, help="Repeat time.")

    args = parser.parse_args()

    file = tf.keras.utils.get_file(
        "grace_hopper.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
    img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])

    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.mobilenet.preprocess_input(
        x[tf.newaxis, ...])
    data = json.dumps({"signature_name": "serving_default",
                       "instances": x.tolist()})
    headers = {"content-type": "application/json"}

    if not args.t:
        json_response = requests.post('http://localhost:{}/v1/models/{}:predict'.format(args.port, args.model),
                                      data=data, headers=headers)
        predictions = np.array(json.loads(json_response.text)["predictions"])
        decoded = imagenet_labels[np.argmax(predictions)]
        print(decoded)

    else:
        start = time.time()
        for i in range(args.repeat):
            json_response = requests.post('http://localhost:{}/v1/models/{}:predict'.format(args.port, args.model),
                                          data=data, headers=headers)
        end = time.time()
        duration = end - start
        avg_latency = duration / float(args.repeat)
        avg_throughput = args.repeat / duration

        print("Avg Latency: {}\nAvg Throughput: {}".format(avg_latency, avg_throughput))
