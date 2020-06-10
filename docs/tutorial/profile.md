# Profile a Model for Cost-Effective MLaaS

After conversion, MLModelCI runs models on the underlying devices in the clusters, and collects, aggregates, and processes running model performance. Specifically, there are six indicators that will be acquired, peak throughput, P50-latency, P95-latency, P99-latency, GPU memory usage, and GPU computation utilization.


To get real model performance in practice, the profiler simulates the real service behavior by invoking a gRPC client and a model service. In particular, the profiler contains many build-in clients and upon it receives a profiling signal, it starts the corresponding client and invokes the dispatcher to deploy a MLaaS. It then sends test data from the client to the service with a variety of batch sizes and serving systems on different devices. Users can have hundreds of combinations available, which is very useful for setting parameters for online services.

We provide many build-in gRPC clients:

- TorchScript
- ONNX
- Triton Inference Serving
- TensorFlow-Serving

## Start Profiling 

You can use the `Profiler` class to instance a profiler to start your model profiling. Before starting the profiler, you need a client. We have some existed client in the `modelci.hub` and according to the task information of a model, the profiler will invoke appropriate clients to finish the profiling.

```python 
from modelci.hub.client.torch_client import CVTorchClient
from modelci.hub.profiler import Profiler

# create a client
torch_client = CVTorchClient(test_data_item, batch_num, batch_size, asynchronous=False)

# init the profiler
profiler = Profiler(model_info=mode_info, server_name='name of your server', inspector=torch_client)

# start profiling model
profiler.diagnose()
```



## Build a Customized Client

For flexible usage, you can build a customized client if necessary. We provide a parent class `modelci.metrics.benchmark.metric.BaseModelInspector` for you to inherit. You have to implement some necessary functions in the child class. Here is an example, a customized client for ResNet50 image classification.

Once you have implemented a customized client, you can pass its instance to `Profiler`, and start profiling like the above shows.

```python 
import grpc
import tensorflow.compat.v1 as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


from modelci.metrics.benchmark.metric import BaseModelInspector

class YourClient(BaseModelInspector):
    def __init__(self, repeat_data, batch_num=1, batch_size=1, asynchronous=None):
        """
        Parameters of parent's __init__, you can choose some to implement.
        ----------
        @param batch_num: the number of batches you want to run
        @param batch_size: batch size you want
        @param repeat_data: data unit to repeat.
        @param asynchronous: running asynchronously, default is False.
        @param sla: SLA, default is 1 sec.
        @param percentile: The SLA percentile. Default is 95.
        """
        super().__init__(repeat_data=repeat_data, batch_num=batch_num, batch_size=batch_size, asynchronous=asynchronous)

    def data_preprocess(self):
        """
        Handle raw data, after preprocessing we can get the 
        processed_data, which is using for benchmarking.
        """
        pass

    def make_request(self, input_batch):
        """
        function for sub-class to implement before inferring, 
        to create the self.request
        can be override if needed.
        """
        pass

    def infer(self, input_batch):
        """
        Abstract function for sub-class to implement the 
        detailed infer function.

        Parameters
        ----------        
        @param input_batch: The batch data in the request.
        """
        pass
```

You should use the MLModelCI's rules specify a ports for your to customize clients.

You can import the ports from ModelCI like this

```python
from modelci.hub.deployer.config import TFS_GRPC_PORT
```

The full table of the default ports are list below:


| Engine Name | HTTP Port | gRPC Port | HTTP Port (GPU) | gRPC Port (GPU) |
|-------------|:---------:|:---------:|:---------------:|:---------------:|
| ONNX        | 8001      | 8002      | 8010            | 8011            |
| TorchScript | 8100      | 8101      | 8110            | 8111            |
| TRT         | 8200      | 8201      | 8202 (Prometeus)| -               |
| TFS         | 8501      | 8500      | 8510            | 8511            |


## Profiling Results 

The profiling results will be sent to the modelhub once profiling is done. And some results will be presented in your terminal as well. They look like:

```
batch size: 8
tested device: Nvidia P4
model: ResNet50
serving engine: TensorFlow Serving

all_batch_latency:  37.82002019882202 sec
all_batch_throughput:  169.2225431492324  req/sec
overall 50th-percentile latency: 0.04665029048919678 s
overall 95th-percentile latency: 0.0504256248474121 s
overall 99th-percentile latency: 0.052218921184539795 s
total GPU memory: 7981694976.0 bytes
average GPU memory usage percentile: 0.9726
average GPU memory used: 7763132416.0 bytes
average GPU utilization: 66.6216%
```

You can display the results by querying the modelhub and export them to a web application or anything you like. 
