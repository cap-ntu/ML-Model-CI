#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/17/2021

Persistence service using PyMongo.
"""
import gridfs

from modelci.config import MONGO_DB
from modelci.experimental.mongo_client import MongoClient
from modelci.persistence.exceptions import ServiceException
from modelci.types.models.mlmodel import MLModel, MLModelIn

_db = MongoClient()[MONGO_DB]
_collection = _db['model_d_o']
_fs = gridfs.GridFS(_db)


def save(model_in: MLModelIn):
    """Register a model into ModelDB and GridFS. `model.id` should be set as `None`, otherwise, the function will
    raise a `ValueError`.

    Args:
        model_in (MLModelIn): model object to be registered

    Return:
        str: Saved model ID.

    Raises:
        BadRequestValueException: If `model.id` is not None.
        ServiceException: If model has exists with the same primary keys (name, framework, engine and version).
    """

    if _collection.count_documents(
            filter=model_in.dict(include={'architecture', 'framework', 'engine', 'version', 'task'}), limit=1
    ):
        raise ServiceException(
            f'Model business object with primary keys architecture={model_in.architecture}, task={model_in.task}, '
            f'framework={model_in.framework}, engine={model_in.engine}, version={model_in.version} has exists.'
        )

    # TODO: update weight ID in the MLModelIn
    weight_id = _fs.put(bytes(model_in.weight), filename=model_in.weight.filename)
    model = MLModel(**model_in.dict(exclude={'weight'}), weight=weight_id)
    model.id = _collection.insert_one(model.dict()).inserted_id
    return model
