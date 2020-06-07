# Retrieve and Deploy Model to Specific Device

You can manage your models using modelci easily. Once you have added a model in the database, you can use the managing APIs to do these actions within few lines of code.

## 1. Retrieve Model

We can query the Model Hub for uploaded models, the retriving operation will return a model information object which contains all the model related information.

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
`modelci.hub.utils.parse_path`. See [Tricks with Model Saved Path](#6.-Tricks-with-Model-Saved-Path)

**TODO: If we retrieve models by task, we will get a lot of models. These models will be cached in a Redis database for 
further model selection scheduling.**

## 2. Update Model

## 3. Delete Model

## Deploy the Cached Model

Deploy a model as a service listening on specific ports. This function will use a local cached model (obtained by 
`modelci.hub.manager.retrieve_model_by_xxx`, see [Query a model](#4.1-Retrieval)).

We use the auto-generated `saved_path` (See [Tricks with Model Saved Path](#-Tricks-with-Model-Saved-Path)) to specify 
model local cache. A serving device can also be assigned using device name (i.e. `'cpu'`, `'cuda:0'`, `'cuda:0,1'`)  

```python
from modelci.hub.deployer import serve

saved_path = ...
device = 'cpu'

serve(saved_path, device)
```