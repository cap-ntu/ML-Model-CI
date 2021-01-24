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
from modelci.types.vo.model_vo import ModelDetailOut, ModelListOut, Framework as Framework_, Engine as Engine_, Task as Task_, Metric as Metric_

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
    Get model structure as a DAG.

    Arguments:
        id (str): Model object ID.
    """
    # return model DAG
    raise NotImplementedError('Method `get_model_structure` not implemented.')


@router.patch('/structure/{id}')  # TODO: add response_model
def update_model_structure_as_new(id: str, dry_run: bool = False):  # noqa
    """
    Update model structure and save as a new version.

    Args:
        id (str): Model object ID of the original structure.
        dry_run (bool): Dry run update for validation.

    Returns:

    """
    raise NotImplementedError('Method `update_model_structure_as_new` not implemented.')

