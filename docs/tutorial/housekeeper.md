# Manage Models with Housekeeper

The housekeeper of MLModelCI is the essence of the model management - a team may produce hundreds of models a day. The housekeeper has four key responsibilities for the management and they are encapsulated into four APIs:

- register
- retrieve
- update
- delete

### 0. Register Model

Please refer to [[Register a Model in ModelHub]](register.md)

### 1. Retrieve Model

Users can search for a model and obtain its detailed information.


```python
from modelci.hub.manager import retrieve_model
from modelci.types.models import Framework, Engine, Task

# By model name and optionally filtered by model framework and(or) model engine
ml_model = retrieve_model(
    architecture='ResNet50', framework=Framework.PyTorch, engine=Engine.TORCHSCRIPT
)
# By task
ml_model_by_task = retrieve_model(task=Task.Image_Classification)
```

The returned tuple contains the path where the model is cached and model meta information (e.g. model name, model framework).

Additionally, we can extract model information from the default saved path using the utility function
`modelci.hub.utils.parse_path`. See [Tricks with Model Saved Path](./register.md#tricks-with-model-saved-path)


### 2. Update Model

You can update the model information manually using the update API.

Here is an example for updating the information of a ResNet50 model. The return value of the function `update_model(model)` is a boolean that indicates the status.

Since many models share a name, the function `get_models` will return a model list. You should specify the model version to get a model object.

```python
from modelci.persistence.service import get_models, update_model
from modelci.types.models import ModelUpdateSchema, Metric
from pathlib import Path

# get_models will return a list of all matched results.

model = get_models(architecture='ResNet50')[0]

# check if update success
assert update_model(str(model.id), ModelUpdateSchema(metric={Metric.acc: 0.9}, weight=Path('path-to-new-model-file')))
```


MLModelCI allows you to get the model object by id, task, architecture, framework, engine and version.

```python
from modelci.persistence.service import get_models, get_by_id
from modelci.types.models import Task

model_by_architecture = get_models(architecture='ResNet50')[0]  # get model by name
model_by_task = get_models(task=Task.Image_Classification)  # get model by task
model_by_id = get_by_id(str(model_by_architecture.id))  # get model by id
```

Getting by name or task may return more than one model objects.

### 3. Delete Model

You can delete a model record easily using MLModelCI.

```python
from modelci.persistence.service import delete_model, get_models

model_list = get_models(architecture='ResNet50')
model = model_list[0]
assert delete_model(str(model.id))  # delete the model record
```

Currently, we only support deleting model by `model.id`.
