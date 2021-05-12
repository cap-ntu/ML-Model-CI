#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/17/2021

Persistence service using PyMongo.
"""
from enum import Enum
from typing import List

import gridfs
from bson import ObjectId
from fastapi.encoders import jsonable_encoder

from modelci.config import db_settings
from modelci.experimental.mongo_client import MongoClient
from modelci.hub.cache_manager import delete_cache_weight
from modelci.persistence.exceptions import ServiceException
from modelci.types.models import MLModel, ModelUpdateSchema, Metric
from modelci.utils.misc import remove_dict_null

_db = MongoClient()[db_settings.mongo_db]
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
    model = MLModel(**model_in.dict(exclude={'weight', 'model_input'}), weight=weight_id)
    model.id = _collection.insert_one(model.dict(exclude_none=True, by_alias=True, use_enum_values=True)).inserted_id
    return model


def get_by_id(id_: str) -> MLModel:
    """Get a MLModel object by its ID.

    Args:
        id_:  Model ID

    Returns: the model object

    """
    model_data = _collection.find_one(filter={'_id': ObjectId(id_)})
    if model_data is not None:
        return MLModel.parse_obj(model_data)
    else:
        raise ServiceException(f'Model with id={id_} does not exist.')


def get_by_parent_id(id_: str) -> List[MLModel]:
    """ Get MLModel objects by its parent model ID.

    Args:
        id_:  The ID of parent model

    Returns: List of model objects

    """
    models = _collection.find(filter={'parent_model_id': ObjectId(id_)})
    if len(models):
        return list(map(MLModel.parse_obj, models))
    else:
        raise ServiceException(f'Model with parent model ID={id_} does not exist.')


def exists_by_id(id_: str) -> MLModel:
    """Check if a MLModel object with specific id exists

    Args:
        id_: model ID

    Returns: True or False

    """
    model = _collection.find_one(filter={'_id': ObjectId(id_)})
    return model is not None


def get_models(**kwargs) -> List[MLModel]:
    """

    Args:
        **kwargs:  architecture, framework, engine, task and version

    Returns: list of models

    """
    valid_keys = {'architecture', 'framework', 'engine', 'task', 'version'}

    valid_kwargs = {
        key: (value.value if isinstance(value, Enum) else value)
        for key, value in remove_dict_null(kwargs).items()
        if key in valid_keys
    }
    models = _collection.find(valid_kwargs)
    return list(map(MLModel.parse_obj, models))


def update_model(id_: str, schema: ModelUpdateSchema) -> MLModel:
    """ Update existed model info

    Args:
        id_:  the ID of targeted model
        schema:

    Returns: the updated model object

    """
    prev_model = get_by_id(id_)
    if schema.metric:
        schema.metric = {Metric(k).name: v for k, v in schema.metric.items()}
    updated_data = {
        key: value for key, value in jsonable_encoder(schema, exclude_unset=True).items()
        if getattr(schema, key) != getattr(prev_model, key)
    }
    _collection.update_one({'_id': ObjectId(id_)}, {"$set": updated_data})
    return get_by_id(id_)


def delete_model(id_: str):
    """

    Args:
        id_:  the ID of target model

    Returns: An instance of :class:`~pymongo.results.DeleteResult`

    """
    model = _collection.find_one(filter={'_id': ObjectId(id_)})
    if _fs.exists(ObjectId(model['weight'])):
        _fs.delete(ObjectId(model['weight']))
    ml_model = MLModel.parse_obj(model)
    delete_cache_weight(ml_model)
    return _collection.delete_one({'_id': ObjectId(id_)})
