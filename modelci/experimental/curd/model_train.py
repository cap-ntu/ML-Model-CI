#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/1/2021
"""
from typing import List

from bson import ObjectId

from modelci.config import db_settings
from modelci.experimental.model.model_train import TrainingJob, TrainingJobIn, TrainingJobUpdate
from modelci.experimental.mongo_client import MongoClient
from modelci.persistence import service
from modelci.persistence.exceptions import ServiceException

_db = MongoClient()[db_settings.mongo_db]
_collection = _db['training_job']


def exists_by_id(id: str) -> bool:
    count = _collection.count_documents(filter={'_id': ObjectId(id)}, limit=1)
    return bool(count)


def get_by_id(id: str) -> TrainingJob:
    # exists by ID
    if not bool(_collection.count_documents(filter={'_id': ObjectId(id)}, limit=1)):
        raise ValueError(f'id {id} not found.')

    document = _collection.find_one(filter={'_id': ObjectId(id)})
    training_job = TrainingJob(**document)
    return training_job


def get_all() -> List[TrainingJob]:
    cursor = _collection.find({})
    training_jobs = list(map(lambda d: TrainingJob(**d), cursor))
    return training_jobs


def save(training_job_in: TrainingJobIn) -> str:
    model_id = training_job_in.model
    if not service.exists_by_id(str(model_id)):
        raise ServiceException(f'Model with ID {model_id} not exist.')

    training_job = TrainingJob(**training_job_in.dict(exclude_none=True))
    return _collection.insert_one(training_job.dict(exclude_none=True)).inserted_id


def update(training_job: TrainingJobUpdate) -> int:
    # exists by ID
    if not bool(_collection.count_documents(filter={'_id': ObjectId(training_job.id)}, limit=1)):
        raise ValueError(f'id {training_job.id} not found.')

    # check model ID
    if training_job.model and not service.exists_by_id(str(training_job.model)):
        raise ServiceException(f'Model with ID {training_job.model} not exist.')

    # save update
    update_data = training_job.dict(exclude_unset=True)
    result = _collection.update_one(filter={'_id': ObjectId(training_job.id)}, update={'$set': update_data})
    return result.modified_count


def delete_by_id(id: str) -> TrainingJob:
    document: dict = _collection.find_one_and_delete(filter={'_id': ObjectId(id)})
    return TrainingJob(**document)


def delete_all() -> int:
    return _collection.delete_many(filter={}).deleted_count
