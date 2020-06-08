# Profiling Model Automatically

After conversion, the system will deploy the model to avaliable devices and run diagnose to test their performance, we call this model profiling. 

Model profiling will start automatically once you have added a new model into the model database, but you can still build a script to control the profiling by yourself. 

## Start Profiling 

You can use the `Profiler` class to instance a profiler to start your model profiling. 

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

Before starting the profiler, you need a client. We have some existed client in the `modelci.hub`, if you don't have ideas how to pick the right client, the profiler will search for the hub automatically. 

## Build a Customed Client

For flexible usage, you can build a customed client if necessary. We have a parent class `modelci.metrics.benchmark.metric.BaseModelInspector` for you to implement. 

You have to implement some necessary functions in the child class, here is an example, a customed client for ResNet50 image classification. 

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
        @param asynchronous: runnning asynchronously, default is False.
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
        function for sub-class to implement before infering, 
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

You should use the ModelCI's rules for ports if you are trying to specify a fixed port number of gRPC or HTTP APIs.

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

Once you have implemented a customed client, you can pass its instance to the `Profiler`, and start profiling like default.

## Profiling Results 

Once you have added a new model into the database, the profiling process will start automatically while you have some free devices. 

The profiling results will be saved into the database once profiling has finished. And some results will be printed in the logger as well. They look like:

```
batch size: 8
tested device: Nvidia P4
model: ResNet50
serving engine: TensorFlow Serving

all_batch_latency:  37.82002019882202 sec
all_batch_throughput:  169.2225431492324  req/sec
overall 50th-percentile latiency: 0.04665029048919678 s
overall 95th-percentile latiency: 0.0504256248474121 s
overall 99th-percentile latiency: 0.052218921184539795 s
total GPU memory: 7981694976.0 bytes
average GPU memory usage percentile: 0.9726
average GPU memory used: 7763132416.0 bytes
average GPU utilization: 66.6216%
```

You can display the results by querying the database. And display the results in the web application or anything you like. 

If you have many Deep Learning models, you can use this feature to get an overview of the performance of models in difference devices.