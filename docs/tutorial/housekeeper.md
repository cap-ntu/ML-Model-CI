# Manage Models with Housekeeper

The housekeeper of MLModelCI is the essence of the model management - a team may produce hundreds of models a day. The housekeeper has four key responsibilities for the management and
they are encapsulated into four APIs

### 1. Retrieve Model

We can query the Model Hub for uploaded models, the retriving operation will return a model information tuple which contains all the model related information.

```python
from modelci.hub.manager import retrieve_model_by_name, retrieve_model_by_task
from modelci.persistence.bo import Framework, Engine

# By model name and optionally filtered by model framework and(or) model engine
model_bo = retrieve_model_by_name(
    architecture_name='ResNet50', framework=Framework.PYTORCH, engine=Engine.TORCHSCRIPT
)
# By task
model_bo2 = retrieve_model_by_task(task='image classification')
```

The returned tuple contains the local model cached path and model meta information (e.g. model name, model framework).

Additionally, we can extract model information from the default saved path using the utility function
`modelci.hub.utils.parse_path`. See [Tricks with Model Saved Path](./register.md#tricks-with-model-saved-path)

**If we retrieve models by task, we will get a lot of models. These models will be cached in a Redis database for
further model selection scheduling.**

### 2. Update Model

You can update the model informations manually using the model updating APIs.

Here is an example for updating the information of the ResNet50 model, the return of function `update_model(model)` returns a boolean to indicate the status.

To get the target model, you may have many versions of a single model structure, `get_models_by_name` will return a list of all matched results.

```python
from modelci.persistence.service import ModelService

model = ModelService.get_models_by_name('ResNet50')[0]
model.acc = 0.9
model.weight.weight = bytes([123, 255])

# check if update success
assert ModelService.update_model(model)
```

If you want to get the model object by other factors, we allow you to get the object by id, task and name.

```python
model_bo = ModelService.get_models_by_name('ResNet50')[0] # get model by name
models = ModelService.get_models_by_task('image classification') # get model by task
model_bo = ModelService.get_models_by_name('ResNet50')[0] # get model by id
```

Getting by name or task may return more than one model objects.

### 3. Delete Model

You can delete a model record easily using ModelCI.

```python
model = ModelService.get_models_by_name('ResNet50')[0]
assert ModelService.delete_model_by_id(model.id) # delete the model record
```

Currently, we only support deleting model by `model.id`.
