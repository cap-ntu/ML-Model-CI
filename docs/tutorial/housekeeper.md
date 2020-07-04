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
from modelci.hub.manager import retrieve_model, retrieve_model_by_task
from modelci.types.bo import Framework, Engine

# By model name and optionally filtered by model framework and(or) model engine
model_bo = retrieve_model(
    architecture_name='ResNet50', framework=Framework.PYTORCH, engine=Engine.TORCHSCRIPT
)
# By task
model_bo2 = retrieve_model_by_task(task='image classification')
```

The returned tuple contains the path where the model is cached and model meta information (e.g. model name, model framework).

Additionally, we can extract model information from the default saved path using the utility function
`modelci.hub.utils.parse_path`. See [Tricks with Model Saved Path](./register.md#tricks-with-model-saved-path)


### 2. Update Model

You can update the model information manually using the update API.

Here is an example for updating the information of a ResNet50 model. The return value of the function `update_model(model)` is a boolean that indicates the status.

Since many models share a name, the function `get_models_by_name` will return a model list. You should specify the model version to get a model object. 

```python
from modelci.persistence.service import ModelService

# get_models_by_name will return a list of all matched results.

model = ModelService.get_models('ResNet50')[0]
model.acc = 0.9
model.weight.weight = bytes([123, 255])

# check if update success
assert ModelService.update_model(model)
```


MLModelCI allows you to get the model object by id, task and name.

```python
from modelci.persistence.service import ModelService

model_bo = ModelService.get_models('ResNet50')[0] # get model by name
models = ModelService.get_models_by_task('image classification') # get model by task
model_bo2 = ModelService.get_models('ResNet50')[0] # get model by id
```

Getting by name or task may return more than one model objects.

### 3. Delete Model

You can delete a model record easily using MLModelCI.

```python
from modelci.persistence.service import ModelService

model = ModelService.get_models('ResNet50')[0]

assert ModelService.delete_model_by_id(model.id) # delete the model record
```

Currently, we only support deleting model by `model.id`.
