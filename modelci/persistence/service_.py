#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/17/2021

Persistence service using PyMongo.
"""
import json
from typing import List

import gridfs
from bson import ObjectId
from fastapi.encoders import jsonable_encoder

from modelci.config import MONGO_DB
from modelci.experimental.mongo_client import MongoClient
from modelci.persistence.exceptions import ServiceException
from modelci.types.models import MLModel, Framework, Engine, Task, BaseMLModel

_db = MongoClient()[MONGO_DB]
_collection = _db['model_d_o']
_fs = gridfs.GridFS(_db)


def save(model_in: MLModel):
    """Register a model into ModelDB and GridFS. `model.id` should be set as `None`, otherwise, the function will
    raise a `ValueError`.

    Args:
        model_in (MLModelIn): model object to be registered

    Return:
        MLModel: Saved ML model object.

    Raises:
        BadRequestValueException: If `model.id` is not None.
        ServiceException: If model has exists with the same primary keys (name, framework, engine and version).
    """

    if _collection.count_documents(
            filter=model_in.dict(
                use_enum_values=True,
                include={'architecture', 'framework', 'engine', 'version', 'task', 'dataset'}
            ),
            limit=1
    ):
        raise ServiceException(
            f'Model with primary keys architecture={model_in.architecture}, '
            f'framework={model_in.framework}, engine={model_in.engine}, version={model_in.version},'
            f'task={model_in.task}, and dataset={model_in.dataset} has exists.'
        )

    # TODO: update weight ID in the MLModelIn
    weight_id = _fs.put(bytes(model_in.weight), filename=model_in.weight.filename)
    model = MLModel(**model_in.dict(exclude={'weight'}), weight=weight_id)
    model.id = _collection.insert_one(model.dict(exclude_none=True, by_alias=True, use_enum_values=True)).inserted_id
    return model


def get_by_id(id: str) -> MLModel:
    """Get a MLModel object by its ID.
    """
    model_data = _collection.find_one(filter={'_id': ObjectId(id)})
    if model_data is not None:
        return MLModel.parse_obj(model_data)
    else:
        raise ServiceException(f'Model with id={id} does not exist.')


def get_models(**kwargs) -> List[MLModel]:
    """

    Args:
        **kwargs:  architecture, framework, engine, task and version

    Returns: list of models

    """
    valid_keys = {'architecture', 'framework', 'engine', 'task', 'version'}
    valid_kwargs = {key: value for key, value in kwargs.items() if value is not None and key in valid_keys}
    models = _collection.find(valid_kwargs)
    return list(map(MLModel.parse_obj, models))


def update_model(id_: str, model: BaseMLModel, prev_model) -> MLModel:
    updated_data = {
        key: value for key, value in jsonable_encoder(model, exclude_unset=True).items()
        if getattr(model, key) != getattr(prev_model, key)
    }
    _collection.update_one({'_id': ObjectId(id_)}, {"$set": updated_data})
    return get_by_id(id_)
