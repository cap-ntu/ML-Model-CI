# -*- coding: utf-8 -*-
"""
Model Data Access Object.

This module contains model data access object (ModelDAO) providing methods to communicate with Mongo DB. `ModelDAO` uses
`mongoengine` as ORM for MongoDB access.
"""
from typing import List

from bson import ObjectId

from modelci.types.do import ModelDO, DynamicProfileResultDO, StaticProfileResultDO


class ModelDAO(object):
    @staticmethod
    def exists_by_id(id_: ObjectId):
        """Check if the given Object ID exists in MongoDB.

        Args:
            id_ (ObjectId): Model ID.
        """
        return bool(ModelDO.objects(id=id_))

    @staticmethod
    def exists_by_primary_keys(**kwargs):
        """

        Args:
            **kwargs: Keyword arguments of primary keys. Supported values:
                name (str): Model name.
                engine (int): Driving engine enum value.
                framework (int): Model framework enum value.
                version (int): Model version number.

        Returns:
            bool: Existence of the model.
        """
        return bool(ModelDO.objects(**kwargs))

    @staticmethod
    def get_model_by_id(id_: ObjectId) -> ModelDO:
        """Get model plain object given model ID.

        Args:
            id_ (ObjectId): Model ID.

        Return:
            ModelDO: Model plain object. None for model PO not found.
        """
        return ModelDO.objects(id=id_).first()

    @staticmethod
    def get_models(**kwargs) -> List[ModelDO]:
        """Get a list of model plain object given model name, framework and engine.

        Args:
            kwargs (dict): A dictionary of arguments:
                name (str): model name.
                framework: Model framework.
                engine: Model engine.
                version: Model version.
        Return:
            List[ModelDO]: A list of model plain objects.
        """
        return ModelDO.objects(**kwargs).order_by('name', 'framework', 'engine', '-version')

    @staticmethod
    def get_models_by_task(task: str) -> List[ModelDO]:
        """Get a list of model plain objects given task.

        Args:
            task (str): Model predictive or descriptive task name

        Return:
            List[ModelDO]: A list of model plain objects. An empty list will be returned if no such model.
        """
        return ModelDO.objects(task=task)

    @staticmethod
    def save_model(model: ModelDO, force_insert=False) -> ModelDO:
        """Save a model PO.

        Args:
            model (ModelDO): Model plain object to be saved.
            force_insert (bool): Only try to create a new document. Default to `False`.

        Returns:
            ModelPo: Updated model plain object.
        """
        return model.save(force_insert=force_insert)

    @staticmethod
    def update_model(id_: ObjectId, **kwargs) -> ModelDO:  # TODO: try ModelPO.objects(...).update()?
        """
        Update or register model PO.

        Args:
            id_ (ObjectId): Model ID.
            **kwargs: Keyword arguments to be updated.

        Returns:
            ModelDO:
        """
        return ModelDO.objects(id=id_).update(**kwargs)

    @staticmethod
    def delete_model_by_id(id_: ObjectId) -> int:
        """Delete model given model ID.

        Args:
            id_ (ObjectId): Model ID.

        Return:
            int: number of affected rows.
        """
        return ModelDO.objects(id=id_).delete()

    @staticmethod
    def register_static_profiling_result(
            id_: ObjectId,
            static_profiling_result: StaticProfileResultDO
    ) -> int:
        """Register static profiling result.

        Args:
            id_ (objectId): ID of the model, where the static profiling result is added.
            static_profiling_result (StaticProfileResultPO): Static profiling result.

        Return:
            int: number of affected rows.
        """
        return ModelDO.objects(id=id_).update_one(set__profile_result__static_profile_result=static_profiling_result)

    @staticmethod
    def register_dynamic_profiling_result(
            id_: ObjectId,
            dynamic_profiling_result: DynamicProfileResultDO
    ) -> int:
        """Register dynamic profiling result.

        Args:
            id_ (ObjectId): ID of the model, where the static profiling result is appended.
            dynamic_profiling_result (DynamicProfileResultPO): Dynamic profiling result.

        Return:
            int: number of affected rows.
        """
        return ModelDO.objects(id=id_).update_one(
            push__profile_result__dynamic_profile_results=dynamic_profiling_result)

    @staticmethod
    def exists_dynamic_profiling_result_by_pks(
            id_: ObjectId,
            ip: str,
            device_id: str,
    ) -> bool:
        """Check if the dynamic profiling result exists.

        Args:
            id_ (ObjectId): ID of the model.
            ip (str): IP address of dynamic profiling result to be deleted.
            device_id (str): Device ID of dynamic profiling result to be deleted.

        Return:
             bool: `True` for existence, `False` otherwise.
        """
        return bool(ModelDO.objects(
            id=id_,
            profile_result__dynamic_profile_results__ip=ip,
            profile_result__dynamic_profile_results__device_id=device_id)
        )

    @staticmethod
    def update_dynamic_profiling_result(
            id_: ObjectId,
            dynamic_profiling_result: DynamicProfileResultDO
    ) -> int:
        """Update dynamic profiling result.

        Args:
            id_ (ObjectId): ID of the model.
            dynamic_profiling_result (DynamicProfileResultPO): Dynamic profiling result to be updated.

        Return:
            int: number of affected rows.
        """
        return ModelDO.objects(
            id=id_,
            profile_result__dynamic_profile_results__ip=dynamic_profiling_result.ip,
            profile_result__dynamic_profile_results__device_id=dynamic_profiling_result.device_id
        ).update(
            set__profile_result__dynamic_profile_results__S=dynamic_profiling_result
        )

    @staticmethod
    def delete_dynamic_profiling_result(
            id_: ObjectId,
            ip: str,
            device_id: str,
    ) -> None:
        """Delete dynamic profiling result.

        Args:
            id_ (ObjectId): ID of the model.
            ip (str): IP address of dynamic profiling result to be deleted.
            device_id (str): Device ID of dynamic profiling result to be deleted.
        """
        return ModelDO.objects(
            id=id_,
        ).update_one(
            pull__profile_result__dynamic_profile_results__ip=ip,
            pull__profile_result__dynamic_profile_results__device_id=device_id
        )
