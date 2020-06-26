# Profiler Example

Use the following tools to profile your Deep Learning model.

## Quick Start 

### Load Data

This step depends on your model input format. You can pass one processed data item (we can duplicate that for testing) or give a set of testing data to test.

Here are several examples,

```python
import cv2
import numpy as np
from PIL import Image

data_path = './data/cat.jpg'

# for TensorFlow Serving
test_img_bytes = None
with open(data_path, 'rb') as f:
    test_img_bytes = f.read()

# for TensorRT Serving
test_img = Image.open(data_path)

# for TorchScript and ONNX
test_img_ndarray: np.ndarray = cv2.imread(data_path)
```

### Implement a Client

We have a basic client class (`BaseModelInspector`) for you to implement. By implementing the client, you can get the profiling results using your own client. 

```python
from modelci.metrics.benchmark.metric import BaseModelInspector

class CustomClient(BaseModelInspector):
    def __init__(self, repeat_data, batch_num=1, batch_size=1, asynchronous=None):
        """implement the __init_ from parent class."""
        super().__init__(repeat_data=repeat_data, batch_num=batch_num, batch_size=batch_size, asynchronous=asynchronous)

    def data_preprocess(self):
        """Handle raw data, after preprocessing we can get the processed_data, which is using for benchmarking."""
        pass

    def make_request(self, input_batch):
        """Create the request here."""
        pass

    def infer(self, input_batch):
        """Put inference code here, the latency will be recorded in this block."""
```

### Using ModelCI Profiler

Once the testing data and the client are ready, you can instance a ModelCI Profiler and start model profiling.

```python
from modelci.hub.client.tfs_client import CVTFSClient
from modelci.hub.deployer.serving import retrieve_model
from modelci.hub.profiler import Profiler
from modelci.types.bo import Engine, Framework

test_img_bytes = ...

# init clients for different serving platforms, you can custom a client by implementing the BaseModelInspector class.
tfs_client = CVTFSClient(test_img_bytes, batch_num=100, batch_size=32, asynchronous=False)
# get the model info from model manager.
mode_info = retrieve_model(architecture_name='ResNet50', framework=Framework.TENSORFLOW, engine=Engine.TFS)
# init a profiler.
profiler = Profiler(model_bo=mode_info, server_name='tfs', inspector=tfs_client)
# start profiling.
profiler.diagnose('cuda:0')
```

The profiling result looks like, 

```
total batches: 100, batch_size: 32
total latency: 6.098069667816162 s
total throughput: 524.7562219383404 req/sec
50th-percentile latency: 0.060769081115722656 s
95th-percentile latency: 0.06291393041610718 s
99th-percentile latency: 0.06391234397888185 s
total GPU memory: 11554717696.0 bytes
average GPU memory usage percentile: 0.9819
average GPU memory used: 11345920000.0 bytes
average GPU utilization: 25.9167%
completed at 2020-05-20 09:52:23.629047
```

Will be saved into the local database automatically.

A runnable demo can be found in [sample.py](./sample.py).

### Profiling Automatically

If you don't know how to implement a custom client for your model, or even didn't prepare testing data. You can try `auto_diagnose`. 

```python
from modelci.hub.profiler import Profiler

model_info = ...

# init a profiler, the server name must be the same as your serving container's.
profiler = Profiler(model_bo=model_info, server_name='tfs')
# start profiling automatically.
profiler.auto_diagnose('device name')
# start profiling automatically with specific batch sizes, here is [2, 4, 16].
profiler.auto_diagnose('device name', [2, 4, 16])
```

the profiler will search for a suitable client according to the model information, and start the model profiling automatically.
