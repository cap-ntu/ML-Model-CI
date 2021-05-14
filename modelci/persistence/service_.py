#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/17/2021

Persistence service using PyMongo.
"""
from ipaddress import IPv4Address, IPv6Address
from typing import List, Union, Optional

import gridfs
from bson import ObjectId
from fastapi.encoders import jsonable_encoder

from modelci.config import db_settings
from modelci.experimental.mongo_client import MongoClient
from modelci.persistence.exceptions import ServiceException
from modelci.types.models import MLModel, ModelUpdateSchema
from modelci.types.models.profile import StaticProfileResult, DynamicProfileResult

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


def exists_by_id(id: str) -> MLModel:
    model = _collection.find_one(filter={'_id': ObjectId(id)})
    return model is not None


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


def update_model(id: str, schema: ModelUpdateSchema) -> MLModel:
    prev_model = get_by_id(id)
    updated_data = {
        key: value for key, value in jsonable_encoder(schema, exclude_unset=True).items()
        if getattr(schema, key) != getattr(prev_model, key)
    }
    _collection.update_one({'_id': ObjectId(id)}, {"$set": updated_data})
    return get_by_id(id)


def delete_model(id_: str):
    model = _collection.find_one(filter={'_id': ObjectId(id_)})
    if _fs.exists(ObjectId(model['weight'])):
        _fs.delete(ObjectId(model['weight']))
    return _collection.delete_one({'_id': ObjectId(id_)})


def register_static_profiling_result(id_: str, static_profiling_result: StaticProfileResult):
    """ Register or update static profiling result to a model.

    Args:
        id_: ID of the model
        static_profiling_result: static profiling result

    Returns:

    """
    return _collection.update_one({'_id': ObjectId(id_)},
                                  {"$set": {"profile_result.static_profile_result": static_profiling_result.dict()}},
                                  upsert=True)


def register_dynamic_profiling_result(id_: str, dynamic_result: DynamicProfileResult):
    """ Add one dynamic profiling result to a model.

    Args:
        id_: ID of the model
        dynamic_result: Dynamic profiling result

    Returns:

    """
    return _collection.update_one({'_id': ObjectId(id_)},
                                  {"$push": {
                                      "profile_result.dynamic_profile_results": jsonable_encoder(dynamic_result)}})


def exists_dynamic_profiling_result_by_pks(id_: str, ip: str, device_id: str) -> bool:
    """Check if the dynamic profiling result exists.

    Args:
        id_: ID of the model.
        ip: IP address of dynamic profiling result to be deleted.
        device_id: Device ID of dynamic profiling result to be deleted.

    Returns: True` for existence, `False` otherwise.

    """
    model = _collection.find_one(filter={
        '_id': ObjectId(id_),
        'profile_result.dynamic_profile_results.ip': ip,
        'profile_result.dynamic_profile_results.device_id': device_id
    })
    return model is not None


def update_dynamic_profiling_result(id_: str, dynamic_result: DynamicProfileResult,
                                    force_insert: Optional[bool] = False):
    """ Update one dynamic profiling result to a model.

    Args:
        id_: ID of the object
        dynamic_result: Dynamic profiling result
        force_insert: force to insert the dynamic result if it is not found

    Returns:

    """
    if exists_dynamic_profiling_result_by_pks(id_, ip=dynamic_result.ip, device_id=dynamic_result.device_id):
        return _collection.update_one({
            '_id': ObjectId(id_),
            'profile_result.dynamic_profile_results.ip': dynamic_result.ip,
            'profile_result.dynamic_profile_results.device_id': dynamic_result.device_id
        }, {"$set": {"profile_result.dynamic_profile_results.$": jsonable_encoder(dynamic_result)}})
    elif force_insert:
        return register_dynamic_profiling_result(id_, dynamic_result)
    else:
        raise ServiceException(
            f'Dynamic profiling result to be updated with ip={dynamic_result.ip}, '
            f'device_id={dynamic_result.device_id} does not exist. Either set `force_insert` or change the ip '
            f'and device_id.'
        )


def delete_dynamic_profiling_result(id_: str, dynamic_result_ip: Union[str, IPv4Address, IPv6Address],
                                    dynamic_result_device_id: str):
    """Delete one dynamic profiling result to a model.

    Args:
        id_: ID of the object.
        dynamic_result_ip: Host IP address of dynamic profiling result.
        dynamic_result_device_id: Device ID of dynamic profiling result.

    Returns:

    """
    if exists_dynamic_profiling_result_by_pks(id_, ip=dynamic_result_ip, device_id=dynamic_result_device_id):
        return _collection.update(
            {'_id': ObjectId(id_)},
            {'$pull': {
                'profile_result.dynamic_profile_results': {
                    'ip': dynamic_result_ip,
                    'device_id': dynamic_result_device_id
                }
            }
            },
            multi=True,
            upsert=False
        )
    else:
        raise ServiceException(
            f'Dynamic profiling result to be updated with ip={dynamic_result_ip}, '
            f'device_id={dynamic_result_device_id} does not exist. Either set `force_insert` or change the ip '
            f'and device_id.'
        )
