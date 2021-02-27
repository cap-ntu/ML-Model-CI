#  Copyright (c) NTU_CAP 2021. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import os
import subprocess
from functools import partial
from pathlib import Path
from shutil import copy2, make_archive
from typing import Union, List

import cv2
import tensorflow as tf
import torch
import yaml

from modelci.hub.client.onnx_client import CVONNXClient
from modelci.hub.client.tfs_client import CVTFSClient
from modelci.hub.client.torch_client import CVTorchClient
from modelci.hub.client.trt_client import CVTRTClient
from modelci.hub.converter import TorchScriptConverter, TFSConverter, TRTConverter, ONNXConverter
from modelci.hub.model_loader import load
from modelci.hub.utils import TensorRTPlatform, parse_path_plain, generate_path_plain
from modelci.persistence.service import ModelService
from modelci.persistence.service_ import save
from modelci.types.bo import Task, ModelVersion, Framework, ModelBO

__all__ = ['get_remote_model_weight', 'register_model', 'register_model_from_yaml', 'retrieve_model',
           'retrieve_model_by_task', 'retrieve_model_by_parent_id']

from modelci.types.models.common import Engine, ModelStatus

from modelci.types.models.mlmodel import MLModelIn, MLModelInYaml, MLModel


def register_model(
        model_in: MLModelIn,
        convert: bool = True,
        profile: bool = True,
) -> List[MLModel]:
    """Upload a model to ModelDB.
    This function will upload the given model into the database with some variation. It may optionally generate a
        branch of models (i.e. model family) with different optimization techniques. Besides, a benchmark will be
        scheduled for each generated model, in order to gain profiling results for model selection strategies.
        In the `no_generate` model(i.e. `no_generate` flag is set to be `True`), `architecture`, `framework`, `engine`
        and `version` could be None. If any of the above arguments is `None`, all of them will be auto induced
        from the origin_model path. An `ValueError` will be raised if the mata info cannot be induced.

    TODO:
        This function has a super comprehensive logic, need to be simplified.

    Arguments:
        model_in: Required inputs for register a model. All information is wrapped in such model.
        convert (bool): Flag for generation of model family. Default to True.
        profile (bool): Flag for profiling uploaded (including converted) models. Default to True.
    """
    models = list()

    model_dir_list = list()
    model_in.model_status = [ModelStatus.PUBLISHED]
    models.append(save(model_in))

    # generate model family
    if convert:
        model_dir_list.extend(_generate_model_family(model_in))

    # register
    model_in_data = model_in.dict(exclude={'weight', 'id', 'model_status', 'engine'})
    for model_dir in model_dir_list:
        parse_result = parse_path_plain(model_dir)
        engine = parse_result['engine']

        model_cvt = MLModelIn(**model_in_data, weight=model_dir, engine=engine, model_status=[ModelStatus.CONVERTED])
        models.append(save(model_cvt))

    # profile registered model
    if profile:
        from modelci.controller import job_executor
        from modelci.controller.executor import Job

        file = tf.keras.utils.get_file(
            "grace_hopper.jpg",
            "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
        test_img_bytes = cv2.imread(file)

        kwargs = {
            'repeat_data': test_img_bytes,
            'batch_size': 32,
            'batch_num': 100,
            'asynchronous': False,
        }

        for model in models:
            model.model_status = [ModelStatus.PROFILING]
            ModelService.update_model(model)
            kwargs['model_info'] = model
            engine = model.engine

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

    return models


def register_model_from_yaml(file_path: Union[Path, str]):
    # check if file exist
    file_path = Path(file_path)
    assert file_path.exists(), f'Model definition file at {str(file_path)} does not exist'

    # read yaml
    with open(file_path) as f:
        model_config = yaml.safe_load(f)
    model_in_yaml = MLModelInYaml.parse_obj(model_config)
    # copy model weight to cache directory
    model_in_saved_path = model_in_yaml.saved_path
    if model_in_saved_path != model_in_yaml.weight:
        copy2(model_in_yaml.weight, model_in_saved_path)

    # zip weight folder
    if model_in_yaml.engine == Engine.TFS:
        weight_dir = model_in_yaml.weight
        make_archive(weight_dir.with_suffix('.zip'), 'zip', weight_dir)

    model_in_data = model_in_yaml.dict(exclude_none=True, exclude={'convert', 'profile'})
    model_in = MLModelIn.parse_obj(model_in_data)
    register_model(model_in, convert=model_in_yaml.convert, profile=model_in_yaml.profile)


def _generate_model_family(
        model_in: MLModelIn,
        max_batch_size: int = -1
):
    model = load(model_in.saved_path)
    build_saved_dir_from_engine = partial(
        generate_path_plain,
        **model_in.dict(include={'architecture', 'framework', 'task', 'version'}),
    )
    inputs = model_in.inputs
    outputs = model_in.outputs
    model_input = model_in.model_input

    generated_dir_list = list()

    torchscript_dir = build_saved_dir_from_engine(engine=Engine.TORCHSCRIPT)
    tfs_dir = build_saved_dir_from_engine(engine=Engine.TFS)
    onnx_dir = build_saved_dir_from_engine(engine=Engine.ONNX)
    trt_dir = build_saved_dir_from_engine(engine=Engine.TRT)

    if isinstance(model, torch.nn.Module):
        # to TorchScript
        if TorchScriptConverter.from_torch_module(model, torchscript_dir):
            generated_dir_list.append(torchscript_dir.with_suffix('.zip'))

        # to ONNX, TODO(lym): batch cache, input shape, opset version
        if ONNXConverter.from_torch_module(model, onnx_dir, inputs, outputs, model_input, optimize=False):
            generated_dir_list.append(onnx_dir.with_suffix('.onnx'))

        # to TRT
        # TRTConverter.from_onnx(
        #     onnx_path=onnx_dir.with_suffix('.onnx'), save_path=trt_dir, inputs=inputs, outputs=outputs
        # )

    elif isinstance(model, tf.keras.Model):
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
        `~/.modelci/<architecture_name>/<framework>-<engine>/<task>/<version>`
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
        elif model.engine == Engine.TRT:
            subprocess.call(['unzip', save_path, '-d', '/'])
            os.remove(save_path)

            TRTConverter.generate_trt_config(
                save_path.parent,  # ~/.modelci/<model-arch-name>/<framework>-<engine>/<task>/
                inputs=model.inputs,
                outputs=model.outputs,
                arch_name=model.name,
                platform=TensorRTPlatform.TENSORFLOW_SAVEDMODEL
            )

    return save_path


def _get_remote_model_weights(models: List[ModelBO]):
    """Get remote model weights from a list of models.
    Only models with highest version of each unique task, architecture, framework, and engine pair are download.
    """

    # group by (task, architecture, framework, engine) pair
    pairs = set(map(lambda x: (x.task, x.name, x.framework, x.engine), models))
    model_groups = [
        [model for model in models if (model.task, model.name, model.framework, model.engine) == pair] for pair in pairs
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
        task: Task = None,
        framework: Framework = None,
        engine: Engine = None,
        version: ModelVersion = None,
        download: bool = True,
) -> List[ModelBO]:
    """Query a model by name, task, framework, engine or version.

    Arguments:
        architecture_name (str): Model architecture name.
        task (Task): which machine learn task is model used for,Default to None
        framework (Framework): Framework name, optional query key. Default to None.
        engine (Engine): Model optimization engine name.
        version (ModelVersion): Model version. Default to None.
        download (bool): Flag for whether the model needs to be cached locally.

    Returns:
        List[ModelBO]: A list of model business object.
    """
    # retrieve
    models = ModelService.get_models(architecture_name, task=task, framework=framework, engine=engine, version=version)
    # check if found
    if len(models) != 0 and download:
        _get_remote_model_weights(models)

    return models


def retrieve_model_by_task(task: Task) -> List[ModelBO]:
    """Query a model by task.
    This function will download a cache model from the model DB.

    Arguments:
        task (Task): Task name the model is used for.

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


def retrieve_model_by_parent_id(parent_id: str) -> List[ModelBO]:
    """
    Query models by specifying the parent model id

    Args:
        parent_id (str): : the parent model id of current model if this model is derived from a pre-existing one

    Returns:
        List[ModelBO]: A list of model business object.
    """
    models = ModelService.get_models_by_parent_id(parent_id)
    # check if found
    if len(models) == 0:
        raise FileNotFoundError('Model not found!')

    return models
