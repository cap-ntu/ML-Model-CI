import os
import subprocess
from functools import partial
from pathlib import Path
from typing import Iterable, Union, List

import cv2
import tensorflow as tf
import yaml

from modelci.hub.client.onnx_client import CVONNXClient
from modelci.hub.client.tfs_client import CVTFSClient
from modelci.hub.client.torch_client import CVTorchClient
from modelci.hub.client.trt_client import CVTRTClient
from modelci.hub.converter import TorchScriptConverter, TFSConverter, TRTConverter, ONNXConverter
from modelci.hub.utils import parse_path, generate_path, TensorRTPlatform
from modelci.persistence.service import ModelService
from modelci.types.bo import IOShape, ModelVersion, Engine, Framework, Weight, DataType, ModelBO


def register_model(
        origin_model,
        dataset: str,
        acc: float,
        task: str,
        inputs: List[IOShape],
        outputs: List[IOShape],
        architecture: str = None,
        framework: Framework = None,
        engine: Engine = None,
        version: ModelVersion = None,
        convert=True,
        profile=True,
):
    """Upload a model to ModelDB.
    This function will upload the given model into the database with some variation. It may optionally generate a
        branch of models (i.e. model family) with different optimization techniques. Besides, a benchmark will be
        scheduled for each generated model, in order to gain profiling results for model selection strategies.
        In the `no_generate` model(i.e. `no_generate` flag is set to be `True`), `architecture`, `framework`, `engine`
        and `version` could be None. If any of the above arguments is `None`, all of them will be auto induced
        from the origin_model path. An `ValueError` will be raised if the mata info cannot be induced.

    Arguments:
        origin_model: The uploaded model without optimization. When `no_generate` flag is set, this parameter should
            be a str indicating model file path.
        architecture (str): Model architecture name. Default to None.
        framework (Framework): Framework name. Default to None.
        version (ModelVersion): Model version. Default to None.
        dataset (str): Model testing dataset.
        acc (float): Model accuracy on the testing dataset.
        task (str): Model task type.
        inputs (Iterable[IOShape]): Model input tensors.
        outputs (Iterable[IOShape]): Model output tensors.
        engine (Engine): Model optimization engine. Default to `Engine.NONE`.
        convert (bool): Flag for generation of model family. When set, `origin_model` should be a path to model saving
            file. Default to `True`.
        profile (bool): Flag for profiling uploaded (including converted) models. Default to `False`.
    """
    from modelci.controller import job_executor
    from modelci.controller.executor import Job

    model_dir_list = list()
    if not convert:
        # type and existence check
        assert isinstance(origin_model, str)
        model_dir = Path(origin_model).absolute()
        assert model_dir.exists(), f'model weight does not exist at {origin_model}'

        if all([architecture, framework, engine, version]):  # from explicit architecture, framework, engine and version
            ext = model_dir.suffix
            path = generate_path(architecture, framework, engine, version).with_suffix(ext)
            # if already in the destination folder
            if path == model_dir:
                pass
            # create destination folder
            else:
                if ext:
                    path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    path.mkdir(parents=True, exist_ok=True)

                # copy to cached folder
                subprocess.call(['cp', model_dir, path])
        else:  # from implicit extracted from path, check validity of the path later at registration
            path = model_dir
        model_dir_list.append(path)
    else:
        # TODO: generate from path name

        # generate model variant
        model_dir_list.extend(_generate_model_family(
            origin_model,
            architecture,
            framework,
            filename=str(version),
            inputs=inputs,
            outputs=outputs
        ))

    # register
    for model_dir in model_dir_list:
        parse_result = parse_path(model_dir)
        architecture = parse_result['architecture']
        framework = parse_result['framework']
        engine = parse_result['engine']
        version = parse_result['version']
        filename = parse_result['filename']

        with open(str(model_dir), 'rb') as f:
            model = ModelBO(
                name=architecture, framework=framework, engine=engine, version=version,
                dataset=dataset, acc=acc, task=task, inputs=inputs,
                outputs=outputs, weight=Weight(f, filename=filename)
            )

            ModelService.post_model(model)
        # TODO refresh
        model = ModelService.get_models(name=architecture, framework=framework, engine=engine, version=version)[0]

        # profile registered model
        if profile:
            file = tf.keras.utils.get_file(
                "grace_hopper.jpg",
                "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
            test_img_bytes = cv2.imread(file)

            kwargs = {
                'repeat_data': test_img_bytes,
                'batch_size': 32,
                'batch_num': 100,
                'asynchronous': False,
                'model_info': model,
            }
            if engine == Engine.TORCHSCRIPT:
                client = CVTorchClient(**kwargs)
            elif engine == Engine.TFS:
                client = CVTFSClient(**kwargs)
            elif engine == Engine.ONNX:
                client = CVONNXClient(**kwargs)
            elif engine == Engine.TRT:
                client = CVTRTClient(**kwargs)
            else:
                raise ValueError(f'No such serving engine: {engine}')

            job_cuda = Job(client=client, device='cuda:0', model_info=model)
            # job_cpu = Job(client=client, device='cpu', model_info=model)
            job_executor.submit(job_cuda)
            # job_executor.submit(job_cpu)


def register_model_from_yaml(file_path: Union[Path, str]):
    def convert_ioshape_plain_to_ioshape(ioshape_plain):
        """Convert IOShape-like dictionary to IOShape.
        """
        # unpack
        i, ioshape_plain = ioshape_plain

        assert isinstance(ioshape_plain['shape'], Iterable), \
            f'inputs[{i}].shape expected to be iterable, but got {ioshape_plain["shape"]}'
        assert isinstance(ioshape_plain['dtype'], str), \
            f'inputs[{i}].dtype expected to be a `DataType`, but got {ioshape_plain["dtype"]}.'

        ioshape_plain['dtype'] = DataType[ioshape_plain['dtype']]

        return IOShape(**ioshape_plain)

    # check if file exist
    file_path = Path(file_path)
    assert file_path.exists(), f'Model definition file at {str(file_path)} does not exist'

    # read yaml
    with open(file_path) as f:
        model_config = yaml.safe_load(f)

    origin_model = model_config['weight']
    dataset = model_config['dataset']
    acc = model_config['acc']
    task = model_config['task']
    inputs_plain = model_config['inputs']
    outputs_plain = model_config['outputs']
    architecture = model_config.get('architecture', None)
    framework = model_config.get('framework', None)
    engine = model_config.get('engine', None)
    version = model_config.get('version', None)
    convert = model_config.get('convert', True)

    # convert inputs and outputs
    inputs = list(map(convert_ioshape_plain_to_ioshape, enumerate(inputs_plain)))
    outputs = list(map(convert_ioshape_plain_to_ioshape, enumerate(outputs_plain)))

    # wrap POJO
    if framework is not None:
        framework = Framework[framework.upper()]
    if engine is not None:
        engine = Engine[engine.upper()]
    if version is not None:
        version = ModelVersion(version)

    register_model(
        origin_model=origin_model,
        dataset=dataset,
        acc=acc,
        task=task,
        inputs=inputs,
        outputs=outputs,
        architecture=architecture,
        framework=framework,
        engine=engine,
        version=version,
        convert=convert,
    )


def _generate_model_family(
        model,
        model_name: str,
        framework: Framework,
        filename: str,
        inputs: List[IOShape],
        outputs: List[IOShape] = None,
        max_batch_size: int = -1
):
    generated_dir_list = list()
    generate_this_path = partial(generate_path, model_name=model_name, framework=framework, version=filename)
    torchscript_dir = generate_this_path(engine=Engine.TORCHSCRIPT)
    tfs_dir = generate_this_path(engine=Engine.TFS)
    onnx_dir = generate_this_path(engine=Engine.ONNX)
    trt_dir = generate_this_path(engine=Engine.TRT)

    if framework == Framework.PYTORCH:
        # to TorchScript
        TorchScriptConverter.from_torch_module(model, torchscript_dir)
        generated_dir_list.append(torchscript_dir.with_suffix('.zip'))

        # to ONNX, TODO(lym): batch cache, input shape
        ONNXConverter.from_torch_module(model, onnx_dir, inputs, optimize=False)
        generated_dir_list.append(onnx_dir.with_suffix('.onnx'))

        # to TRT
        # TRTConverter.from_onnx(
        #     onnx_path=onnx_dir.with_suffix('.onnx'), save_path=trt_dir, inputs=inputs, outputs=outputs
        # )
        return generated_dir_list
    elif framework == Framework.TENSORFLOW:
        # to TFS
        TFSConverter.from_tf_model(model, tfs_dir)
        generated_dir_list.append(tfs_dir.with_suffix('.zip'))

        # to TRT
        TRTConverter.from_saved_model(tfs_dir, trt_dir, inputs, outputs, max_batch_size=32)
        generated_dir_list.append(trt_dir.with_suffix('.zip'))

    return generated_dir_list


def get_remote_model_weight(model: ModelBO):
    """Download a local cache of model from remote ModelDB in a structured path. And generate a configuration file.
    TODO(lym):
        1. set force insert config.pbtxt
        2. set other options in generation of config.pbtxt (e.g. max batch size, instance group...)
    This function will keep a local cache of the used model in the path:
        `~/.modelci/<architecture_name>/<framework>-<engine>/version`
    Arguments:
        model (ModelBO): Model business object.
    Return:
        Path: Model saved path.
    """
    save_path = model.saved_path

    save_path.parent.mkdir(exist_ok=True, parents=True)

    if not save_path.exists():
        with open(str(save_path), 'wb') as f:
            f.write(model.weight.weight)
        if model.engine == Engine.TFS:
            subprocess.call(['unzip', save_path, '-d', '/'])
            os.remove(save_path)
        elif Engine.TRT:
            subprocess.call(['unzip', save_path, '-d', '/'])
            os.remove(save_path)

            TRTConverter.generate_trt_config(
                save_path.parent,  # ~/.modelci/<model-arch-name>/<framework>-<engine>/
                inputs=model.inputs,
                outputs=model.outputs,
                arch_name=model.name,
                platform=TensorRTPlatform.TENSORFLOW_SAVEDMODEL
            )

    return save_path


def _get_remote_model_weights(models: List[ModelBO]):
    """Get remote model weights from a list of models.
    Only models with highest version of each unique architecture, framework, and engine pair are download.
    """

    # group by (architecture, framework, engine) pair
    pairs = set(map(lambda x: (x.name, x.framework, x.engine), models))
    model_groups = [
        [model for model in models if (model.name, model.framework, model.engine) == pair] for pair in pairs
    ]

    # get weights of newest version of each pair
    for model_group in model_groups:
        get_remote_model_weight(model_group[0])


def delete_remote_weight(model: ModelBO):
    save_path = model.saved_path

    if model.engine in [Engine.TORCHSCRIPT, Engine.ONNX]:
        os.remove(save_path)
    else:
        os.removedirs(save_path)


def retrieve_model(
        architecture_name: str = 'ResNet50',
        framework: Framework = None,
        engine: Engine = None,
        version: ModelVersion = None,
) -> List[ModelBO]:
    """Query a model by name, framework, engine or version.

    Arguments:
        architecture_name (str): Model architecture name.
        framework (Framework): Framework name, optional query key. Default to None.
        engine (Engine): Model optimization engine name.
        version (ModelVersion): Model version. Default to None.

    Returns:
        List[ModelBO]: A list of model business object.
    """
    # retrieve
    models = ModelService.get_models(architecture_name, framework=framework, engine=engine, version=version)
    # check if found
    if len(models) == 0:
        raise FileNotFoundError('Model not found!')

    _get_remote_model_weights(models)

    return models


def retrieve_model_by_task(task: str = 'image classification') -> List[ModelBO]:
    """Query a model by task.
    This function will download a cache model from the model DB.

    Arguments:
        task (str): Task name. Default to "image classification"

    Returns:
        List[ModelBO]: A list of model business object.
    """
    # retrieve
    models = ModelService.get_models_by_task(task)
    # check if found
    if len(models) == 0:
        raise FileNotFoundError('Model not found!')

    _get_remote_model_weights(models)

    return models
