from modelci.persistence import mongo
from modelci.persistence.bo import (
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
)
from modelci.persistence.service import ModelService
from modelci.utils.trtis_objects import ModelInputFormat


def test_register_model():
    model = ModelBO(
        'ResNet50', framework=Framework.PYTORCH, engine=Engine.TRT, version=ModelVersion(1),
        dataset='ImageNet', acc=0.8, task='image classification',
        inputs=[IOShape([-1, 3, 224, 224], dtype=float, format=ModelInputFormat.FORMAT_NCHW)],
        outputs=[IOShape([-1, 1000], dtype=int)],
        weight=Weight(bytes([123]))
    )

    assert ModelService.post_model(model)


def test_get_model_by_name():
    models = ModelService.get_models_by_name('ResNet50')

    # check length
    assert len(models) == 1
    # check name
    for model in models:
        assert model.name == 'ResNet50'


def test_get_model_by_task():
    models = ModelService.get_models_by_task('image classification')

    # check length
    assert len(models) == 1
    # check name
    for model in models:
        assert model.task == 'image classification'


def test_get_model_by_id():
    model_bo = ModelService.get_models_by_name('ResNet50')[0]
    model = ModelService.get_model_by_id(model_bo.id)

    # check model id
    assert model.id == model_bo.id


def test_update_model():
    model = ModelService.get_models_by_name('ResNet50')[0]
    model.acc = 0.9
    model.weight.weight = bytes([123, 255])

    # check if update success
    assert ModelService.update_model(model)

    model_ = ModelService.get_models_by_name('ResNet50')[0]

    # check updated model
    assert abs(model_.acc - 0.9) < 1e-6
    assert model_.weight.weight == model.weight.weight


def test_register_static_profiling_result():
    model = ModelService.get_models_by_name('ResNet50')[0]
    spr = StaticProfileResultBO(5000, 200000, 200000, 10000, 10000, 10000)
    assert ModelService.register_static_profiling_result(model.id, spr)


def test_register_dynamic_profiling_result():
    model = ModelService.get_models_by_name('ResNet50')[0]
    dpr = DynamicProfileResultBO('gpu:01', 'Tesla K40c', 1, ProfileMemory(1000, 1000, 1000),
                                 ProfileLatency((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
                                 ProfileThroughput((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)))
    assert ModelService.append_dynamic_profiling_result(model.id, dpr)


def test_update_dynamic_profiling_result():
    model = ModelService.get_models_by_name('ResNet50')[0]
    dpr = DynamicProfileResultBO('gpu:01', 'Tesla K40c', 1, ProfileMemory(1000, 2000, 1000),
                                 ProfileLatency((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
                                 ProfileThroughput((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)))
    # check update
    assert ModelService.update_dynamic_profiling_result(model.id, dpr)
    # check result
    model = ModelService.get_models_by_name('ResNet50')[0]
    assert model.profile_result.dynamic_results[0].memory.cpu_memory == 2000


def test_delete_model():
    model = ModelService.get_models_by_name('ResNet50')[0]
    assert ModelService.delete_model_by_id(model.id)


def test_drop_test_database():
    mongo.db.drop_database('test')
