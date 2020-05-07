import subprocess
from functools import partial
from pathlib import Path
from typing import Iterable, Union, List

import yaml

from modelci.hub.converter import TorchScriptConverter, ONNXConverter, TFSConverter, TRTConverter
from modelci.hub.utils import parse_path, generate_path, TensorRTPlatform
from modelci.persistence.bo import IOShape, ModelVersion, Engine, Framework, Weight, DataType, ModelBO
from modelci.persistence.service import ModelService


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
        no_generate=False
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
        no_generate: Flag for not generation of model family. When set, `origin_model` should be a path to model saving
            file.
    """
    model_dir_list = list()
    if no_generate:
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

        # TODO(lym): profile


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
    no_generate = model_config.get('no_generate', False)

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
        no_generate=no_generate,
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
        elif Engine.TRT:
            subprocess.call(['unzip', save_path, '-d', '/'])

            TRTConverter.generate_trt_config(
                save_path.parent,  # ~/.modelci/<model-arch-name>/<framework>-<engine>/
                inputs=model.inputs,
                outputs=model.outputs,
                arch_name=model.name,
                platform=TensorRTPlatform.TENSORFLOW_SAVEDMODEL
            )

    return save_path


def retrieve_model_by_name(architecture_name: str = 'ResNet50', framework: Framework = None, engine: Engine = None):
    """Query a model by name, framework or engine.
    Arguments:
        architecture_name (str): Model architecture name.
        framework (Framework): Framework name, optional query key. Default to None.
        engine (Engine): Model optimization engine name.
    Returns:
        ModelBO: Model business object.
    """

    # retrieve
    models = ModelService.get_models_by_name(architecture_name, framework=framework, engine=engine)
    # check if found
    if len(models) == 0:
        raise FileNotFoundError('Model not found!')
    # TODO: filter version
    model = models[0]

    get_remote_model_weight(model)

    return model


def retrieve_model_by_task(task='image classification'):
    """Query a model by task.
    This function will download a cache model from the model DB.
    Arguments:
        task (str): Task name. Default to "image classification"
    Returns:
        ModelBo: Model business object.
    """
    # retrieve
    models = ModelService.get_models_by_task(task)
    # check if found
    if len(models) == 0:
        raise FileNotFoundError('Model not found!')
    model = models[0]

    get_remote_model_weight(model)

    return model


def update_model():
    # TODO: update model
    raise NotImplementedError()


def delete_model():
    # TODO: delete model
    raise NotImplementedError()
