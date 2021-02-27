#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/20/2020
"""
import asyncio
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, Depends

from modelci.hub.manager import register_model
from modelci.persistence.service import ModelService
from modelci.types.bo import Framework, Engine, Task
from modelci.types.models import MLModelIn
from modelci.types.models.mlmodel import MLModelInForm
from modelci.types.vo.model_vo import ModelDetailOut, ModelListOut, Framework as Framework_, Engine as Engine_, \
    Task as Task_

router = APIRouter()


@router.get('/', response_model=List[ModelListOut])
def get_all_model(name: str = None, framework: Framework_ = None, engine: Engine_ = None, task: Task_ = None,
                  version: int = None):
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


@router.post('/', status_code=201)
async def publish_model(
        ml_model_in_form: MLModelInForm = Depends(MLModelInForm.as_form),
        files: List[UploadFile] = File(
            [],
            description='This field can be set with empty value. In such settings, the publish is a dry run to'
                        'validate the `ml_model_in_form` field. You are recommend to try a dry run before the'
                        'model wight file(s) are posted. This is because the weight file(s) are usually large and '
                        'take a while to receive the whole request and validate.'
        ),
        convert: bool = True,
        profile: bool = False
):
    # save the posted files as local cache
    loop = asyncio.get_event_loop()
    saved_path = ml_model_in_form.saved_path
    if len(files) == 0:
        # dry run
        return {'status': True}
    if len(files) == 1:
        file = files[0]
        suffix = Path(file.filename).suffix
        try:
            # create directory
            assert len(suffix) != 0, f'Expect a suffix for file {file.filename}, got None.'
            saved_path = saved_path.with_suffix(suffix)
            saved_path.parent.mkdir(exist_ok=True, parents=True)

            # save file
            await file.seek(0)
            with open(saved_path, 'wb') as buffer:
                await loop.run_in_executor(None, shutil.copyfileobj, file.file, buffer)
        finally:
            await file.close()
    else:
        raise NotImplementedError('`publish_model` not implemented for multiple files upload.')
        # zip the files

    ml_model_in = MLModelIn(**ml_model_in_form.dict(), weight=saved_path)
    models = register_model(model_in=ml_model_in, convert=convert, profile=profile)
    return {
        'data': {'id': [str(model.id) for model in models], },
        'status': True
    }
