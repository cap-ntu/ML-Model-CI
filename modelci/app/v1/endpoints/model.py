#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/20/2020
"""
import asyncio
import json
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, Depends
from fastapi.exceptions import RequestValidationError, HTTPException
from pydantic.error_wrappers import ErrorWrapper
from starlette.responses import JSONResponse, Response

from modelci.hub.registrar import register_model
from modelci.persistence.service_ import get_by_id, get_models, update_model, delete_model, exists_by_id
from modelci.types.models import MLModel, BaseMLModel, ModelUpdateSchema, Framework, Engine, Task

router = APIRouter()


@router.get('/')
def get_all_models(architecture: str = None, framework: Framework = None, engine: Engine = None, task: Task = None,
                  version: int = None):
    models = get_models(architecture=architecture, framework=framework, engine=engine, task=task, version=version)
    content = list(map( lambda item: json.loads(item.json(by_alias=False)), models))
    return JSONResponse(content=content)


@router.get('/{id}')
def get_model(*, id: str):  # noqa
    # Due to FastAPI use default json encoder before customer encoder, we have to rely on
    # Pydantic BaseModel.json and convert it back
    # Check https://github.com/tiangolo/fastapi/blob/master/fastapi/encoders.py#L118 to see if this
    # issue is fixed.
    content = json.loads(get_by_id(id).json(by_alias=False))
    return JSONResponse(content=content)


@router.patch('/{id}', response_model=MLModel)
def update(id: str, schema: ModelUpdateSchema):
    if not exists_by_id(id):
        raise HTTPException(
            status_code=404,
            detail=f'Model ID {id} does not exist. You may change the ID',
        )
    return update_model(id, schema)


@router.delete('/{id}', status_code=204, response_class=Response)
def delete(id: str):
    if not exists_by_id(id):
        raise HTTPException(
            status_code=404,
            detail=f'Model ID {id} does not exist. You may change the ID',
        )
    delete_model(id)


@router.post('/', status_code=201)
async def publish_model(
        model: BaseMLModel = Depends(BaseMLModel.as_form),
        files: List[UploadFile] = File(
            [],
            description='This field can be set with empty value. In such settings, the publish is a dry run to'
                        'validate the `ml_model_in_form` field. You are recommend to try a dry run to find input'
                        'errors before you send the wight file(s) to the server.'
        ),
        convert: bool = True,
        profile: bool = False
):
    """Publish model to the model hub. The model weight file(s) as well as its meta information (e.g.
    architecture, framework, and serving engine) will be stored into the model hub.

    The publish API will also automatically convert the published model into other deployable format such as
    TorchScript and ONNX. After successfully converted, original model and its generated models will be profiled
    on the underlying devices in the clusters, and collects, aggregates, and processes running model performance.

    Args:
        model (MLModel): Model meta information.
        files (List[UploadFile]): A list of model weight files. The files are organized accordingly. Their file name
            contains relative path to their common parent directory.
            If the files is empty value, a dry-run to this API is conducted for parameter checks. No information
            will be saved into model hub in this case.
        convert (bool): Flag for auto configuration.
        profile (bool): Flag for auto profiling.

    Returns:
        A message response, with IDs of all published model. The format of the return is:
        ```
        {
          "data": {"id": ["603e6a1f5a62b08bc0a2a7f2", "603e6a383b00cbe9bfee7277"]},
          "status": true
        }
        ```
        Specially, if the dry-run test passed, it will return a status True:
        ```
        {
          "status": true
        }
        ```
    """
    # save the posted files as local cache
    loop = asyncio.get_event_loop()
    saved_path = model.saved_path
    if len(files) == 0:
        # conduct dry run for parameter check only.
        return {'status': True}
    if len(files) == 1:
        file = files[0]
        suffix = Path(file.filename).suffix
        try:
            # create directory
            if len(suffix) == 0:
                error = ErrorWrapper(
                    ValueError(f'Expect a suffix for file {file.filename}, got None.'), loc='files[0]'
                )
                raise RequestValidationError([error])
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

    model = MLModel(**model.dict(), weight=saved_path)
    models = register_model(model=model, convert=convert, profile=profile)
    return {
        'data': {'id': [str(model.id) for model in models], },
        'status': True
    }
