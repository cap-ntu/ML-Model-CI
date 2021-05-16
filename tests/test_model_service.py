from pathlib import Path

import torch
import torchvision

from modelci.hub.registrar import register_model_from_yaml
from modelci.persistence import mongo
from modelci.persistence.service import delete_model, register_static_profiling_result, \
    register_dynamic_profiling_result, update_dynamic_profiling_result, delete_dynamic_profiling_result
from modelci.persistence.service import get_models, get_by_id, update_model
from modelci.types.models import Task, Metric, ModelUpdateSchema
from modelci.types.models.profile import (
    StaticProfileResult,
    DynamicProfileResult,
    ProfileMemory,
    ProfileLatency,
    ProfileThroughput,
    InfoTuple
)

file_dir = str(Path.home() / '.modelci/ResNet50/pytorh-pytorch/image_classification')
Path(file_dir).mkdir(parents=True, exist_ok=True)
model_path = f'{file_dir}/1.pth'

torch_model = torchvision.models.resnet50(pretrained=True)
torch.save(torch_model, model_path)


def test_init():
    mongo.db.drop_database('test')


def test_register_model():
    register_model_from_yaml("example/resnet50_explicit_path.yml")


def test_get_model_by_name():
    models = get_models(architecture='ResNet50')

    # check length
    assert len(models) == 1
    # check name
    for model in models:
        assert model.architecture == 'ResNet50'


def test_get_model_by_task():
    models = get_models(task=Task.Image_Classification)

    # check length
    assert len(models) == 1
    # check name
    for model in models:
        assert model.task == Task.Image_Classification


def test_get_model_by_id():
    model = get_models()[0]
    model_by_id = get_by_id(str(model.id))

    # check model id
    assert model.id == model_by_id.id


def test_update_model():
    model = get_models()[0]

    # check if update success
    model_ = update_model(str(model.id), ModelUpdateSchema(metric={Metric.acc: 0.9}))

    # check updated model
    assert abs(model_.metric[Metric.acc] - 0.9) < 1e-6


def test_register_static_profiling_result():
    model = get_models()[0]
    spr = StaticProfileResult(parameters=5000, flops=200000, memory=200000, mread=10000, mwrite=10000, mrw=10000)
    assert register_static_profiling_result(str(model.id), spr)


def test_register_dynamic_profiling_result():
    model = get_models()[0]
    dummy_info_tuple = InfoTuple(avg=1, p50=1, p95=1, p99=1)
    dpr = DynamicProfileResult(
        device_id='gpu:01',
        device_name='Tesla K40c',
        batch=1,
        memory=ProfileMemory(total_memory=1000, memory_usage=1000, utilization=0.5),
        latency=ProfileLatency(
            initialization_latency=dummy_info_tuple,
            preprocess_latency=dummy_info_tuple,
            inference_latency=dummy_info_tuple,
            postprocess_latency=dummy_info_tuple,
        ),
        throughput=ProfileThroughput(
            batch_formation_throughput=1,
            preprocess_throughput=1,
            inference_throughput=1,
            postprocess_throughput=1,
        )
    )
    assert register_dynamic_profiling_result(str(model.id), dpr)


def test_update_dynamic_profiling_result():
    model = get_models(architecture='ResNet50')[0]
    dummy_info_tuple = InfoTuple(avg=1, p50=1, p95=1, p99=1)
    updated_info_tuple = InfoTuple(avg=1, p50=2, p95=1, p99=1)
    dpr = DynamicProfileResult(
        device_id='gpu:01',
        device_name='Tesla K40c',
        batch=1,
        memory=ProfileMemory(total_memory=1000, memory_usage=2000, utilization=0.5),
        latency=ProfileLatency(
            initialization_latency=dummy_info_tuple,
            preprocess_latency=dummy_info_tuple,
            inference_latency=updated_info_tuple,
            postprocess_latency=dummy_info_tuple,
        ),
        throughput=ProfileThroughput(
            batch_formation_throughput=1,
            preprocess_throughput=1,
            inference_throughput=1,
            postprocess_throughput=1,
        )
    )
    # check update
    assert update_dynamic_profiling_result(str(model.id), dpr)
    # check result
    model = get_models()[0]
    assert model.profile_result.dynamic_profile_results[0].memory.memory_usage == 2000
    assert model.profile_result.dynamic_profile_results[0].latency.inference_latency.p50 == 2


def test_delete_dynamic_profiling_result():
    model = get_models()[0]
    dummy_info_tuple1 = InfoTuple(avg=1, p50=1, p95=1, p99=2)
    dummy_info_tuple2 = InfoTuple(avg=1, p50=1, p95=1, p99=1)

    dpr = DynamicProfileResult(
        device_id='gpu:02',
        device_name='Tesla K40c',
        batch=1,
        memory=ProfileMemory(total_memory=1000, memory_usage=1000, utilization=0.5),
        latency=ProfileLatency(
            initialization_latency=dummy_info_tuple1,
            preprocess_latency=dummy_info_tuple2,
            inference_latency=dummy_info_tuple2,
            postprocess_latency=dummy_info_tuple2,
        ),
        throughput=ProfileThroughput(
            batch_formation_throughput=1,
            preprocess_throughput=1,
            inference_throughput=1,
            postprocess_throughput=1,
        )
    )
    register_dynamic_profiling_result(str(model.id), dpr)

    # reload
    model = get_models()[0]
    dpr_bo = model.profile_result.dynamic_profile_results[0]
    dpr_bo2 = model.profile_result.dynamic_profile_results[1]

    # check delete
    assert delete_dynamic_profiling_result(str(model.id), dpr_bo.ip, dpr_bo.device_id)

    # check result
    model = get_models()[0]
    assert len(model.profile_result.dynamic_profile_results) == 1

    dpr_left = model.profile_result.dynamic_profile_results[0]
    assert dpr_bo2.latency.initialization_latency.avg == dpr_left.latency.initialization_latency.avg


def test_delete_model():
    model_list = get_models(architecture='ResNet50')
    for model in model_list:
        assert delete_model(str(model.id))


def test_drop_test_database():
    mongo.db.drop_database('test')
