## Model service
<img src="../../docs/img/model-service-block-diagram.png" alt="model service block diagram" height=500 />

### API

#### 1. Connect to mongodb
Configure your MongoDB connection setting at `modelci/env-mongodb.env`. (You may have done this by 
[installation](/README.md#installation). Source the env file when you want to connect to the database by:
```shell script
set -o allexport
source modelci/env-mongodb.env
set +o allexport
```
If you are using an IDE to run the following test, remember to add `modelci/env-mongodb.env` as an EnvFile.

#### 2. Register a model
```python
from modelci.persistence.service import ModelService
from modelci.types.bo import Framework, Engine, ModelVersion, IOShape, Weight, ModelBO
from modelci.types.trtis_objects import ModelInputFormat

# create a model business object
model = ModelBO(
    name='ResNet50', 
    framework=Framework.PYTORCH, 
    engine=Engine.TRT, 
    version=ModelVersion(1),
    dataset='Image Net', 
    acc=0.8,
    task='image classification',
    inputs=[IOShape([-1, 3, 224, 224], dtype=int, format=ModelInputFormat.FORMAT_NCHW)],
    outputs=[IOShape([-1, 1000], dtype=int)], 
    weight=Weight(bytes([123]))
)
# register
ModelService.post_model(model)
```
See test `test/test_model_service.test_register_model`.

#### 3. Get a list of models by architecture name
```python
from modelci.persistence.service import ModelService

model_service = ModelService()
models = model_service.get_models_by_name('ResNet50')
```
See test `test/test_model_service.test_get_model_by_name`.

#### 4. Get a list of models by task
```python
from modelci.persistence.service import ModelService

model_service = ModelService()
models = model_service.get_models_by_task('image classification')
```
See test `test/test_model_service.test_get_model_by_task`.

#### 5. Get model by model ID
```python
from modelci.persistence.service import ModelService

model_service = ModelService()
model = model_service.get_model_by_id('123456789012')
```
The ID must be a valid `ObjectID`.  
See test `test/test_model_service.get_model_by_id`.

#### 6. Update model
```python
from modelci.persistence.service import ModelService

model_service = ModelService()
# Query from Mongo DB, obtain model business object and update
model_bo = ...

model_service.update_model(model_bo)
```
This API will check if the model exists in Model DB. It will reject the update by raising a `ValueError`. 
If you would like to force update:
```python
from modelci.persistence.service import ModelService

model_service = ModelService()
model_bo = ...
model_service.update_model(model_bo, force_insert=True)
```
However, if there is a change of profiling results, please use profiling result related API for CRUD 
([add static profiling](#7-add-static-profiling-result-to-a-registered-model), 
[add dynamic profiling result to a registered](#8-add-dynamic-profiling-result-to-a-registered-model),
[update dynamic profiling result](#9-update-dynamic-profiling-result), 
[delete dynamic profiling result](#10-delete-dynamic-profiling-result))). 
See test `test/test_model_service.test_update_model`.

#### 7. Add static profiling result to a registered model
```python
from modelci.persistence.service import ModelService
from modelci.types.bo import StaticProfileResultBO

model_service = ModelService()

static_result = StaticProfileResultBO(
    parameters=5000, 
    flops=200000, 
    memory=200000, 
    mread=10000, 
    mwrite=10000, 
    mrw=10000
)
model_service.register_static_profiling_result('123456789012', static_result)
```
The ID must be a valid `ObjectID`.  
See test `test/test_model_service.test_register_static_profiling_result`  
Update static profiling result may use the same API.

#### 8. Add dynamic profiling result to a registered model
```python
from modelci.persistence.service import ModelService
from modelci.types.bo import DynamicProfileResultBO, ProfileLatency, ProfileMemory, ProfileThroughput

model_service = ModelService()

dynamic_result = DynamicProfileResultBO(
    device_id='gpu:01', 
    device_name='Tesla K40c', 
    batch=1, 
    memory=ProfileMemory(1000, 2000, 1000),
    latency=ProfileLatency((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    throughput=ProfileThroughput((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1))
)
model_service.append_dynamic_profiling_result('123456789012', dynamic_result)
```
The ID must be a valid `ObjectID`. This API will raise a `ValueError` if the `id` does not exist.    
See test `test/test_model_service.test_register_dynamic_profiling_result`.

#### 9. Update dynamic profiling result
```python
from modelci.persistence.service import ModelService

# the updated dynamic profiling result
dynamic_result = ...
ModelService.update_dynamic_profiling_result('123456789012', dynamic_result)
```
The ID must be a valid `ObjectID`. If a non-existent ID or a non-existent profiling result `ip`, `device_id` pair is supplied, this API will reject the update by raising a `ValueError`.  
You may set `force_insert` to register a profiling result if the `ip` and `device_id` does not exist.  
See test `test/test_model_service.test_update_dynamic_profiling_result`.  

#### 10. Delete dynamic profiling result
```python
import ipaddress

from modelci.persistence.service import ModelService

model_bo = ...

ModelService.delete_dynamic_profiling_result(
    id_=model_bo.id, 
    dynamic_result_ip=ipaddress.ip_address('localhost'), 
    dynamic_result_device_id='gpu:01'
)
```
The ID must be a valid `ObjectID`. This API will raise a `ValueError` if the `id` does not exist, or the `ip` and 
`device_id` for the given model does not exist.  
See test `test/test_model_service.test_delete_dynamic_profiling_result`.

#### 11. Delete model
```python
from modelci.persistence.service import ModelService

model_service = ModelService()
model_service.delete_model_by_id('123456789012')
```
The ID must be a valid `ObjectID`.  
See test `test/test_model_service.test_delete_model`  

### Test
Run pytest by:
```shell script
python -m pytest tests/test_model_service.py
```
