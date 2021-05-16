## Model service
<img src="../../docs/img/model-service-block-diagram.png" alt="model service block diagram" height=500 />

### API

#### 1. Setup Environment
Generate all the environment variables by one command (You may have done this by 
[installation](/README.md#installation). A single `.env` file will be generated:
```shell script
python scripts/generate_env.py
```

#### 2. Register a model
```python
from pathlib import Path
from modelci.types.models.common import Engine, Task, Framework, Metric, ModelStatus, IOShape, DataType
from modelci.types.models import MLModel

# create a MLModel object
mlmodel = MLModel(
    weight=Path('path-to-model-file'),
    architecture='ResNet50',
    dataset='ImageNet',
    framework=Framework.PyTorch,
    engine=Engine.PYTORCH,
    version=1,
    metric={Metric.acc: 0.80},
    task=Task.Image_Classification,
    inputs=[IOShape(name="input", shape=[-1, 3, 224, 224], dtype=DataType.TYPE_FP32)],
    outputs=[IOShape(name="output", shape=[-1, 1000], dtype=DataType.TYPE_FP32)],
    model_status=[ModelStatus.PUBLISHED]
)
# register

from modelci.hub.registrar import register_model
register_model(mlmodel, convert=True, profile=False)
```
See test `test/test_model_service.test_register_model`.

#### 3. Get a list of models by architecture name

```python
from modelci.persistence.service import get_models

models = get_models(architecture='ResNet50')
```
See test `test/test_model_service.test_get_model_by_name`.

#### 4. Get a list of models by task

```python
from modelci.persistence.service import get_models
from modelci.types.models import Task

models = get_models(task=Task.Image_Classification)
```
See test `test/test_model_service.test_get_model_by_task`.

#### 5. Get model by model ID

```python
from modelci.persistence.service import get_by_id

ml_model = get_by_id('123456789012')
```
The ID must be a valid `ObjectID`.  
See test `test/test_model_service.get_model_by_id`.

#### 6. Update model

```python
from modelci.persistence.service import update_model
from modelci.types.models import ModelUpdateSchema

ml_model = ...
update_model(str(ml_model.id), ModelUpdateSchema(...))
```
This API will check if the model exists in Model DB. It will reject the update by raising a `ValueError`. 

However, if there is a change of profiling results, please use profiling result related API for CRUD 
([add static profiling](#7-add-static-profiling-result-to-a-registered-model), 
[add dynamic profiling result to a registered](#8-add-dynamic-profiling-result-to-a-registered-model),
[update dynamic profiling result](#9-update-dynamic-profiling-result), 
[delete dynamic profiling result](#10-delete-dynamic-profiling-result))). 
See test `test/test_model_service.test_update_model`.

#### 7. Add static profiling result to a registered model

```python
from modelci.persistence.service import register_static_profiling_result
from modelci.types.models.profile import StaticProfileResult

static_result = StaticProfileResult(
    parameters=5000,
    flops=200000,
    memory=200000,
    mread=10000,
    mwrite=10000,
    mrw=10000
)
register_static_profiling_result('123456789012', static_result)
```
The ID must be a valid `ObjectID`.  
See test `test/test_model_service.test_register_static_profiling_result`  
Update static profiling result may use the same API.

#### 8. Add dynamic profiling result to a registered model

```python
from modelci.persistence.service import register_dynamic_profiling_result
from modelci.types.models.profile import InfoTuple, DynamicProfileResult, ProfileMemory, ProfileLatency,ProfileThroughput

info_tuple = InfoTuple(avg=1, p50=1, p95=1, p99=1)
dynamic_result = DynamicProfileResult(
    device_id='gpu:01',
    device_name='Tesla K40c',
    batch=1,
    memory=ProfileMemory(total_memory=1000, memory_usage=2000, utilization=0.5),
    latency=ProfileLatency(
        initialization_latency=info_tuple,
        preprocess_latency=info_tuple,
        inference_latency=info_tuple,
        postprocess_latency=info_tuple
    ),
    throughput=ProfileThroughput(
        batch_formation_throughput=1,
        preprocess_throughput=1,
        inference_throughput=1,
        postprocess_throughput=1,
    )
)
register_dynamic_profiling_result('123456789012', dynamic_result)
```
The ID must be a valid `ObjectID`. This API will raise a `ValueError` if the `id` does not exist.    
See test `test/test_model_service.test_register_dynamic_profiling_result`.

#### 9. Update dynamic profiling result

```python
from modelci.persistence.service import update_dynamic_profiling_result

# the updated dynamic profiling result
dynamic_result = ...
update_dynamic_profiling_result('123456789012', dynamic_result)
```
The ID must be a valid `ObjectID`. If a non-existent ID or a non-existent profiling result `ip`, `device_id` pair is supplied, this API will reject the update by raising a `ValueError`.  
You may set `force_insert` to register a profiling result if the `ip` and `device_id` does not exist.  
See test `test/test_model_service.test_update_dynamic_profiling_result`.  

#### 10. Delete dynamic profiling result

```python
import ipaddress

from modelci.persistence.service import delete_dynamic_profiling_result

ml_model = ...

delete_dynamic_profiling_result(
    id_=str(ml_model.id),
    dynamic_result_ip=ipaddress.ip_address('localhost'),
    dynamic_result_device_id='gpu:01'
)
```
The ID must be a valid `ObjectID`. This API will raise a `ValueError` if the `id` does not exist, or the `ip` and 
`device_id` for the given model does not exist.  
See test `test/test_model_service.test_delete_dynamic_profiling_result`.

#### 11. Delete model

```python
from modelci.persistence.service import delete_model

delete_model('123456789012')
```
The ID must be a valid `ObjectID`.  
See test `test/test_model_service.test_delete_model`  

### Test
Run pytest by:
```shell script
python -m pytest tests/test_model_service.py
```
