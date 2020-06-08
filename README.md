<p align="center"> <img src="docs/img/iconv1.svg" width="230" alt="..."> </p>

<h1 align="center">
    Machine Learning Model CI
</h1>

<p align="center">
    <a href="https://www.python.org/downloads/release/python-370/" title="python version"><img src="https://img.shields.io/badge/Python-3.7%2B-blue.svg"></a>
    <a href="https://travis-ci.com/cap-ntu/ML-Model-CI" title="Build Status"><img src="https://travis-ci.com/cap-ntu/ML-Model-CI.svg?token=SvqJmaGbqAbwcc7DNkD2&branch=master"></a>
    <a href="https://app.fossa.com/projects/custom%2B8170%2Fgithub.com%2Fcap-ntu%2FML-Model-CI?ref=badge_shield" title="FOSSA Status"><img src="https://app.fossa.com/api/projects/custom%2B8170%2Fgithub.com%2Fcap-ntu%2FML-Model-CI.svg?type=shield"></a>
    <a href="https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cap-ntu/ML-Model-CI&amp;utm_campaign=Badge_Grade" title="Codacy Badge"><img src="https://app.codacy.com/project/badge/Grade/bfb9f8b11d634602acd8b67484a43318"></a>
    <a href="https://codebeat.co/a/yizheng-huang/projects/github-com-cap-ntu-ml-model-ci-master"><img alt="codebeat badge" src="https://codebeat.co/badges/343cc340-21c6-4d34-ae2c-48a48e2862ba" /></a>
    <a href="https://github.com/cap-ntu/ML-Model-CI/graphs/commit-activity" title="Maintenance"><img src="https://img.shields.io/badge/Maintained%3F-YES-yellow.svg"></a>
    <a href="https://gitter.im/ML-Model-CI/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge" title="Gitter"><img src="https://badges.gitter.im/ML-Model-CI/community.svg"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#tutorial">Tutorial</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</p>

## Features

- **Housekeeper** provides a refined management for model (service) registration, deletion, update and selection.
- **Converter** is to convert models to serilized and optimized formats so that the models can be deployed to cloud.
- **Profiler** simulates the real service behavior by invoking a gRPC client and a model service, and provides a detailed report about model runtime performance (e.g. P99-latency and throughput) in production environment.
- **Dispatcher** launches a serving system to load a model in a containerized manner and dispatches the MLaaS to a device.
- **Controller** receives data from the monitor and node exporter, and controls the whole workflow of our system.

## Installation

### Docker

![comming soon](https://img.shields.io/badge/comming-soon-green.svg)

<!-- Please refer to [here](/intergration/README.md) for more information. -->

### Command Line  

```shell script
bash scripts/install.sh
```
*Conda is required to run this installation script

## Quick Start

MLModelCI provides a complete platform for managing, converting, profiling, and deploying models as cloud services (MLaaS). You just need to register your models to our platform and it will take over the rest tasks. To give a more clear start, we present the whole pipeline step by step as follows.

### Register a Model 

Assume you have a ResNet50 model trained by PyTorch. To deploy it as a cloud service, the first step is to publish the model to our system.

```python
from modelci.hub.manager import register_model

# Register a Trained ResNet50 Model to ModelHub.
register_model(
    'home/ResNet50/pytorch/1.zip',
    dataset='ImageNet',
    acc=0.76,
    task='image classification',
    inputs=[IOShape([-1, 3, 224, 224], float)],
    outputs=[IOShape([-1, 1000], float)],
    convert=True,
    profile=True
)
```

### Convert a Model

As the a newly trained model can not be deployed to cloud, MLModelCI converts it to some optimized formats (e.g., TorchScript and ONNX) automatically.

You can finish this on your own:

```python 
from modelci.hub.converter import ONNXConverter
from modelci.persistence.bo import IOShape

ONNXConverter.from_torch_module('<path to torch model>', 
                                '<path to export onnx model>', 
                                input_shape=[IOShape([-1, 3, 224, 224], float)], 
                                batch_size=16)
```

### Profile a Model

Before deploying an optimized model as a cloud service, developers need to understand its runtime performance (e.g., latency and throughput) so to set up a more cost-effectie solution (batch size? device? serving system? etc.). MLModelCI provides a profile to automate the processing.

You can manually profile your models as follows:

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


### Dispatch a model

MLModelCI provides a dispatcher to deploy a model as a cloud service. The dispatcher launches a serving system (e.g. Tensorflow-Serving) to load a model in a containerized manner and dispatches the MLaaS to a device.

We search for a converted model and then dispatch it to a device with a specific batch size.

```python 
from modelci.hub.deployer import serve
from modelci.hub.manager import retrieve_model_by_name

# get saved model information
mode_info = retrieve_model_by_name(architecture_name='ResNet50', framework=Framework.PYTORCH, engine=Engine.TORCHSCRIPT)

# deploy the model to cuda device 0.
serve(save_path=model_info.saved_path, device='cuda:0', name='torchscript-serving', batch_size=16) 
```

Now your model is an effient cloud service!


For more information please take a look at our tutorials.


## Tutorial

After the Quick Start, we provide detailed tutorials for users to understand our system.

- [Register a Model in ModelHub](./docs/tutorial/register.md)
- [Convert a Model to Optimized Formats](./docs/tutorial/convert.md)
- [Profile a Model for Cost-Effective MLaaS](./docs/tutorial/profile.md)
- [Dispatch a Model as a Cloud Service](./docs/tutorial/retrieve-and-deploy.md)
- [Manage Models with Housekeeper](./docs/tutorial/housekeeper.md)

## Contributing

MLModelCI welcomes your contributions! Please refer to [here](.github/CONTRIBUTING.md) to get start.

## License
```
   Copyright 2020 Nanyang Technological University, Singapore

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
