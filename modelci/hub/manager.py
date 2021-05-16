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

from typing import List

from modelci.hub.cache_manager import get_remote_model_weights
from modelci.persistence.service import get_by_parent_id, get_models
from modelci.types.models import MLModel, Task, Framework, Engine

__all__ = ['retrieve_model', 'retrieve_model_by_parent_id']


def retrieve_model(
        architecture: str = 'ResNet50',
        task: Task = None,
        framework: Framework = None,
        engine: Engine = None,
        version: int = None,
        download: bool = True,
) -> List[MLModel]:
    """Query a model by name, task, framework, engine or version.

    Arguments:
        architecture (str): Model architecture name.
        task (Task): which machine learn task is model used for,Default to None
        framework (Framework): Framework name, optional query key. Default to None.
        engine (Engine): Model optimization engine name.
        version (Int): Model version. Default to None.
        download (bool): Flag for whether the model needs to be cached locally.

    Returns:
        List[MLModel]: A list of model business object.
    """
    # retrieve
    models = get_models(architecture=architecture, task=task, framework=framework, engine=engine, version=version)
    # check if found
    if len(models) != 0 and download:
        get_remote_model_weights(models)

    return models


def retrieve_model_by_parent_id(parent_id: str, download: bool = True) -> List[MLModel]:
    """
    Query models by specifying the parent model id

    Args:
        parent_id (str): : the parent model id of current model if this model is derived from a pre-existing one
        download: Flag for whether the model needs to be cached locally.

    Returns:
        List[MLModel]: A list of MLModel object.
    """
    models = get_by_parent_id(parent_id)
    # check if found
    if len(models) == 0:
        raise FileNotFoundError('Model not found!')
    elif download:
        get_remote_model_weights(models)

    return models
