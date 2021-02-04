from modelci.persistence import mongo
from modelci.persistence.service import ModelService
from modelci.types.bo import (
    DynamicProfileResultBO,
    ProfileMemory,
    ProfileLatency,
    ProfileThroughput,
    ModelBO,
    Framework,
    Engine,
    ModelVersion,
    IOShape,
    Weight,
    StaticProfileResultBO,
    InfoTuple,
    Task,
    Metric,
    ModelStatus
)
from modelci.types.trtis_objects import ModelInputFormat


def test_init():
    mongo.db.drop_database('test')


def test_register_model():
    model = ModelBO(
        'ResNet50',
        framework=Framework.PYTORCH,
        engine=Engine.TRT,
        version=ModelVersion(1),
        dataset='ImageNet',
        metric={Metric.ACC: 0.80},
        task=Task.IMAGE_CLASSIFICATION,
        inputs=[IOShape([-1, 3, 224, 224], dtype=float, format=ModelInputFormat.FORMAT_NCHW)],
        outputs=[IOShape([-1, 1000], dtype=int)],
        model_status=ModelStatus.READY,
        weight=Weight(bytes([123]))
    )

    assert ModelService.post_model(model)


def test_get_model_by_name():
    models = ModelService.get_models('ResNet50')

    # check length
    assert len(models) == 1
    # check name
    for model in models:
        assert model.name == 'ResNet50'


def test_get_model_by_task():
    models = ModelService.get_models_by_task(Task.IMAGE_CLASSIFICATION)

    # check length
    assert len(models) == 1
    # check name
    for model in models:
        assert model.task == Task.IMAGE_CLASSIFICATION


def test_get_model_by_id():
    model_bo = ModelService.get_models('ResNet50')[0]
    model = ModelService.get_model_by_id(model_bo.id)

    # check model id
    assert model.id == model_bo.id


def test_update_model():
    model = ModelService.get_models('ResNet50')[0]
    model.metric[Metric.ACC] = 0.9
    model.weight.weight = bytes([123, 255])

    # check if update success
    assert ModelService.update_model(model)

    model_ = ModelService.get_models('ResNet50')[0]

    # check updated model
    assert abs(model_.metric[Metric.ACC] - 0.9) < 1e-6
    assert model_.weight.weight == model.weight.weight


def test_register_static_profiling_result():
    model = ModelService.get_models('ResNet50')[0]
    spr = StaticProfileResultBO(5000, 200000, 200000, 10000, 10000, 10000)
    assert ModelService.register_static_profiling_result(model.id, spr)


def test_register_dynamic_profiling_result():
    model = ModelService.get_models('ResNet50')[0]
    dummy_info_tuple = InfoTuple(avg=1, p50=1, p95=1, p99=1)
    dpr = DynamicProfileResultBO(
        device_id='gpu:01',
        device_name='Tesla K40c',
        batch=1,
        memory=ProfileMemory(1000, 1000, 0.5),
        latency=ProfileLatency(
            init_latency=dummy_info_tuple,
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
    assert ModelService.append_dynamic_profiling_result(model.id, dpr)


def test_update_dynamic_profiling_result():
    model = ModelService.get_models('ResNet50')[0]
    dummy_info_tuple = InfoTuple(avg=1, p50=1, p95=1, p99=1)
    updated_info_tuple = InfoTuple(avg=1, p50=2, p95=1, p99=1)
    dpr = DynamicProfileResultBO(
        device_id='gpu:01',
        device_name='Tesla K40c',
        batch=1,
        memory=ProfileMemory(1000, 2000, 0.5),
        latency=ProfileLatency(
            init_latency=dummy_info_tuple,
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
    assert ModelService.update_dynamic_profiling_result(model.id, dpr)
    # check result
    model = ModelService.get_models('ResNet50')[0]
    assert model.profile_result.dynamic_results[0].memory.memory_usage == 2000
    assert model.profile_result.dynamic_results[0].latency.inference_latency.p50 == 2


def test_delete_dynamic_profiling_result():
    model = ModelService.get_models('ResNet50')[0]
    dummy_info_tuple1 = InfoTuple(avg=1, p50=1, p95=1, p99=2)
    dummy_info_tuple2 = InfoTuple(avg=1, p50=1, p95=1, p99=1)

    dpr = DynamicProfileResultBO(
        device_id='gpu:02',
        device_name='Tesla K40c',
        batch=1,
        memory=ProfileMemory(1000, 1000, 0.5),
        latency=ProfileLatency(
            init_latency=dummy_info_tuple1,
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
    ModelService.append_dynamic_profiling_result(model.id, dpr)

    # reload
    model = ModelService.get_models('ResNet50')[0]
    dpr_bo = model.profile_result.dynamic_results[0]
    dpr_bo2 = model.profile_result.dynamic_results[1]

    # check delete
    assert ModelService.delete_dynamic_profiling_result(model.id, dpr_bo.ip, dpr_bo.device_id)

    # check result
    model = ModelService.get_models('ResNet50')[0]
    assert len(model.profile_result.dynamic_results) == 1

    dpr_left = model.profile_result.dynamic_results[0]
    assert dpr_bo2.latency.init_latency.avg == dpr_left.latency.init_latency.avg


def test_delete_model():
    model = ModelService.get_models('ResNet50')[0]
    assert ModelService.delete_model_by_id(model.id)


def test_drop_test_database():
    mongo.db.drop_database('test')
