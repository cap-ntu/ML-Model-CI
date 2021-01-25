#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/20/2020
"""
from typing import List

from fastapi import APIRouter

from modelci.persistence.service import ModelService
from modelci.types.bo import Framework, Engine, Task
from modelci.types.vo.model_vo import ModelDetailOut, ModelListOut, Framework as Framework_, Engine as Engine_, \
    Task as Task_

router = APIRouter()


@router.get('/', response_model=List[ModelListOut])
def get_all_model(name: str = None, framework: Framework_ = None, engine: Engine_ = None, task: Task_ = None, version: int = None):
    if framework is not None:
        framework = Framework[framework.value.upper()]
    if engine is not None:
        engine = Engine[engine.value.upper()]
    if task is not None:
        engine = Task[task.value.upper()]

    models = ModelService.get_models(name=name, framework=framework, engine=engine, task=task, version=version)
    return list(map(ModelListOut.from_bo, models))


@router.get('/{id}', response_model=ModelDetailOut)
def get_model(*, id: str):  # noqa
    model = ModelService.get_model_by_id(id)
    return ModelDetailOut.from_bo(model)


@router.get('/structure/{id}')  # TODO: add response_model
async def get_model_structure(id: str):  # noqa
    """
    Get model structure as a model structure graph (connection between layer as edge, layers as nodes)

    Arguments:
        id (str): Model object ID.
    """
    # return model DAG
    raise NotImplementedError('Method `get_model_structure` not implemented.')


@router.patch('/structure/{id}')  # TODO: add response_model
def update_model_structure_as_new(id: str, structure, dry_run: bool = False):  # noqa
    """
    Update model layer and save as a new version.

    Currently, this function can only support change of last layer (i.e. fine-tune). You can indicate the
    modified layer in parameter `structure`.

    Examples:
        Fine-tune the model by modify the layer with name 'fc' (last layer) of model id=123. The layer
        has a changed argument out_features = 10. _op='M' indicates the operation to this layer ('fc')
        is 'Modify'. There is no changes in layer connections.

        >>> from modelci.types.bo.model_objects import IOShape
        ... import numpy as np
        ...
        ... structure = {'node': {'fc': {'out_features': 10, '_op': 'M'}}}
        ... update_model_structure_as_new(id=..., structure=structure)

    TODO:
        Add new layers and rewire the connections.

    Args:
        id (str): Model object ID of the original structure.
        structure: A model structure graph indicating changed layer (node) and layer connection (edge).
        dry_run (bool): Dry run update for validation.

    Returns:

    """
    raise NotImplementedError('Method `update_model_structure_as_new` not implemented.')

