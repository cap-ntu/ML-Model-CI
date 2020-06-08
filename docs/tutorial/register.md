# 1. Register a Model to MLModelCI

In the first tutorial, we demonstrate how to start to use MLModelCI. The first step is to publish a model in the system.

Firstly, make sure that you have started a MongoDB service and configured the MongoDB environment. See
[installation](../../README.md#installation).

We have a quick script including automatic model registration, conversion, and profiling, for you. Just run this command in the package root.

```shell script
python modelci/hub/init_data.py export --model {MODEL_NAME} --framework {FRAMEWORK}
```

Now let us finish this step by step.


## Register a Pre-trained Model

MLModelCI provides two methods to publish models, using .yaml file and Python code.

### With a simple configuration file [[template]](/example/resnet50_explicit_path.yml)

```python
from modelci.hub.manager import register_model_from_yaml

model_yaml = ...

register_model_from_yaml(model_yaml)
```

### With Python API

We can register a pre-trained model using `modelci.hub.manager.register_model(...)`. This API can trigger two functions with two parameters, respectively:

-   `convert`:

    Enabled by setting `convert=True` (default), which converts your model (a PyTorch `nn.Module` or a
    TensorFlow `keras.Model` object) into all possible optimized model formats.

-   `profile`:

    Enabled by setting `profile=True`. This will profile converted models automatically.

If you use `convert=True`, ModelCI will help you to convert your model automatically after adding to the database. Similarly, if you use `profile=True`, ModelCI will find a free device to start model benchmarking and profiling automatically.

#### Understanding `convert=True` mode

Here is an example of the registration using `convert` mode. 

```python
import torch.hub

from modelci.hub.manager import register_model
from modelci.persistence.bo.model_objects import IOShape, Framework, ModelVersion
from modelci.utils.trtis_objects import ModelInputFormat

model = torch.hub.load('pytorch/torchvision:v0.5.0', model='resnet50', pretrained=True)
inputs = [IOShape(shape=[-1, 3, 224, 224], dtype=float, format=ModelInputFormat.FORMAT_NCHW)]
outputs = [IOShape(shape=[-1, 1000], dtype=float)]

register_model(
    model,
    dataset='ImageNet',
    acc=0.76,
    task='image classification',
    inputs=inputs,
    outputs=outputs,
    architecture='ResNet50',
    framework=Framework.PYTORCH,
    version=ModelVersion(1),
    convert=True,
    profile=True
)
```

####  Understanding `convert=False` mode

Assume we have a saved pre-trained ResNet50 model at the current working directory named `1.zip`. It was trained on ImageNet and converted by TorchScript. You should register it using using `convert=False` mode and set the `engine=Engine.TorchScript`

```python
from modelci.hub.manager import register_model
from modelci.persistence.bo import Framework, IOShape, Engine, ModelVersion

register_model(
    'path/to/model/1.zip',
    dataset='ImageNet',
    acc=0.76,
    task='image classification',
    inputs=[IOShape([-1, 3, 224, 224], float)],
    outputs=[IOShape([1], int)],
    architecture='ResNet50',
    framework=Framework.PYTORCH,
    engine=Engine.TORCHSCRIPT,
    version=ModelVersion(1),
    convert=False,
    profile=True
)
```

*Trick: to save your time, you can follow the following rule to name your model:
(See [Tricks with Model Saved Path](#tricks-with-model-saved-path))  
`~/.modelci/<model name>/<framework>-<engine>/<version>.<extension>`
In this case, we are only able to specify the path, without architecture, framework, engine and version.*

```python
from modelci.hub.manager import register_model
from modelci.persistence.bo.model_objects import IOShape

register_model(
    '~/.modelci/ResNet50/pytorch-torchscript/1.zip',
    dataset='ImageNet',
    acc=0.76,
    task='image classification',
    inputs=[IOShape([-1, 3, 224, 224], float)],
    outputs=[IOShape([-1, 1000], float)],
    convert=False,
    profile=True,
)
```

## Tricks with Model Saved Path

If you use `modelci/hub/init_data.py` to download model, the default model local cache path is as the following form:  
`~/.modelci/<model name>/<framework>-<engine>/<version>.<extension>`

We can use `modelci.hub.utils.parse_path(...)` to extract model identification.

```python
from modelci.hub.utils import parse_path

# from return value of model retrieval
model_bo = ...

info = parse_path(model_bo.saved_path)
```

The extracted information is a dictionary containing:

```yaml
{
    "architecture": architecture,
    "framework": framework,
    "engine": engine,
    "version": version,
    "filename": filename,
}
```

Vice versa, we can generate a default path by `modelci.hub.utils.generate_path(...)`:

```python
from modelci.hub.utils import generate_path
from modelci.persistence.bo import Framework, Engine

saved_path = generate_path(
    model_name='ResNet50', framework=Framework.PYTORCH, engine=Engine.TORCHSCRIPT, version=1
)
```
