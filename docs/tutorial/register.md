# Register Model in the Model Database

In this tutorial, we demonstrate modelci, a toolbox providing APIs that fetch, convert diagnose and deploy pre-trained 
model in different form of variant.

Firstly, make sure that you have started a MongoDB service and configured the MongoDB environment. See 
[installation](../../README.md#installation).

## Register a Pre-trained Model

### With a simple configuration file [[template]](/example/resnet50_explicit_path.yml)

```python
from modelci.hub.manager import register_model_from_yaml

model_yaml = ...

register_model_from_yaml(model_yaml)
```

We can register a pre-trained model using `modelci.hub.manager.register_model(...)`. This API has two modes:

- `auto_generate`:

    Enabled by setting `no_generate=False` (default), which converts your model (a PyTorch `nn.Module` or a
    TensorFlow `keras.Model` object) into all possible model family.
- `no_generate`:

    Enabled by setting `no_generate=True`. This will let user register the given model only.

There is a short cut of generation, we can save the model in a standard form:
(See [Tricks with Model Saved Path](#tricks-with-model-saved-path))  
`~/.modelci/<model name>/<framework>-<engine>/<version>.<extension>`
In this case, we are only able to specify the path, without architecture, framework, engine and version.

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
    no_generate=True
)
```

For quick start (conversion + registration), run
```shell script
python init_data.py export --model {MODEL_NAME} --framework {FRAMEWORK}
```

Currently supported (tested) model name:
- ResNet50

### Registration using `auto_generate` mode

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
    version=ModelVersion(1)
)
```

### Registration using `no_generate` Mode

Assume we have a saved pre-trained ResNet50 model at current working directory named `1.zip`. It was trained on 
ImageNet and exported by TorchScript.

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
    no_generate=True
)
```

## Tricks with Model Saved Path

The default model local cache path is in the following form:  
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
    'architecture': architecture,
    'framework': framework,
    'engine': engine,
    'version': version,
    'filename': filename
}
```

Vice versa, we can generate the default path by `modelci.hub.utils.generate_path(...)`:

```python
from modelci.hub.utils import generate_path
from modelci.persistence.bo import Framework, Engine

saved_path = generate_path(
    model_name='ResNet50', framework=Framework.PYTORCH, engine=Engine.TORCHSCRIPT, version=1
)
```