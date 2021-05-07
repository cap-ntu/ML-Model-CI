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
from typing import List
from modelci.hub import converter
from modelci.hub.utils import TensorRTPlatform
from modelci.persistence.service import ModelService
from modelci.types.bo import Task, ModelVersion, Framework, ModelBO

__all__ = ['get_remote_model_weight', 'retrieve_model',
           'retrieve_model_by_task', 'retrieve_model_by_parent_id']

from modelci.types.models.common import Engine


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

            converter.TRTConverter.generate_trt_config(
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
    pairs = set(map(lambda x: (x.task, x.architecture, x.framework, x.engine), models))
    model_groups = [
        [model for model in models if (model.task, model.architecture, model.framework, model.engine) == pair] for pair in pairs
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
