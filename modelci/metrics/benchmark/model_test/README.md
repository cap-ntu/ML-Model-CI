## A quick test using ResNet-50 and TF-Serving

### Quick Start

1. start a tensorflow-serving resnet service according to the [official demo](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md).
2. run python script `test_resnet50_tfs.py`.

### Parameter

in the source you can find some global variables, you can set by your preference.

```python
BATCH_SIZE = 32 # batch size (client side batching)
DATA_LENGTH = 6400 # the testing data number you want
ASYNCHRONOUS = False # to enable the asynchronous method sending requests
```