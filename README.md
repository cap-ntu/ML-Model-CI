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

- **Housekeeper** provides a refined management for model registration, deletion, update and selection.
- **Converter** is to convert models between frameworks.
- **Profiler** can diagnose a model and provide a detailed report about model performance in production environment.
- **Dispatcher** is to place models to devices and serve them automatically.
- **Controller** supports elastic test and deployment.

## Installation

### Docker 

```bash
docker pull xxx #TODO
```

Please refer to [here](/intergration/README.md) for more information.

### Command Line  

```bash
curl xxx.get_cli.sh #TODO
# or pip install modelci
```

## Quick Start

ModelCI offers a user-friendly interface for you to manage your model related workflows. 

### Register a Saved Model 

Assume you have a ResNet50 model trained by PyTorch, you can easily add it to the database like this

```python
from modelci.hub.manager import register_model

# Register a Trained ResNet50 Model in Database.
register_model(
    'home/ResNet50/pytorch/1.zip',
    dataset='ImageNet',
    acc=0.76,
    task='image classification',
    inputs=[IOShape([-1, 3, 224, 224], float)],
    outputs=[IOShape([-1, 1000], float)]
)
```

### Convert Model

You can use ModelCI to convert your registered model to another platform, such as ONNX runtime.

```python 
from modelci.hub.converter import ONNXConverter
from modelci.persistence.bo import IOShape

ONNXConverter.from_torch_module('<path to torch model>', 
                                '<path to export onnx model>', 
                                input_shape=[IOShape([-1, 3, 224, 224], float)], 
                                batch_size=16)
```

We also support other types of conversion.

### Retrieve Model and Deploy

We can get the model information and deploy it to any specific device using ModelCI.

```python 
from modelci.hub.deployer import serve
from modelci.hub.manager import retrieve_model_by_name

# get saved model information
mode_info = retrieve_model_by_name(architecture_name='ResNet50', framework=Framework.PYTORCH, engine=Engine.TORCHSCRIPT)

# deploy the model to cuda device 0.
serve(save_path=model_info.saved_path, device='cuda:0', name='torchscript-serving') 
```

Now your model is running for inference!

### Profile Your Model

In order to measure the performance of the model running on different machines, ModelCI will automatically select the appropriate machine for performance testing and export the results to database.

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

We can get several metrics after profiling, including serving throughputs, latency, GPU utilization and memory usage.

For more information please take a look at our tutorials.


## Tutorial

Atfer the Quick Start, we have some tutorials here for detailed usages

- [Register Model in the Model Database](./docs/tutorial/register.md)
- [Converting Model to Different Frameworks](./docs/tutorial/convert.md)
- [Retrieve and Deploy Model to Specific Device](./docs/tutorial/retrieve-and-deploy.md)
- [Profiling Model Automatically](./docs/tutorial/profile.md)

## Contributing

ModelCI welcomes your contributions! Please refer to [here](.github/CONTRIBUTING.md) to get start.

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