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
from modelci.types.bo import Framework, Engine
from modelci.types.vo.model_vo import ModelDetailOut, ModelListOut

router = APIRouter()


@router.get('/', response_model=List[ModelListOut])
def get_all_model(name: str = None, framework: Framework = None, engine: Engine = None, version: int = None):
    models = ModelService.get_models(name=name, framework=framework, engine=engine, version=version)
    return list(map(ModelListOut.from_bo, models))


@router.get('/{id}', response_model=ModelDetailOut)
def get_model(*, id: str):
    model = ModelService.get_model_by_id(id)
    return ModelDetailOut.from_bo(model)
