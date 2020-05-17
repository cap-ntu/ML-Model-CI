# -*- coding: utf-8 -*-
"""
Model Data Access Object.

This module contains model data access object (ModelDAO) providing methods to communicate with Mongo DB. `ModelDAO` uses
`mongoengine` as ORM for MongoDB access.
"""
from typing import List

from bson import ObjectId

from ..po.dynamic_profile_result_po import DynamicProfileResultPO
from ..po.model_po import ModelPO
from ..po.static_profile_result_po import StaticProfileResultPO


class ModelDAO(object):
    @staticmethod
    def exists_by_id(id_: ObjectId):
        """Check if the given Object ID exists in MongoDB.

        Args:
            id_ (ObjectId): Model ID.
        """
        return bool(ModelPO.objects(id=id_))

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
        return bool(ModelPO.objects(**kwargs))

    @staticmethod
    def get_model_by_id(id_: ObjectId) -> ModelPO:
        """Get model plain object given model ID.

        Args:
            id_ (ObjectId): Model ID.

        Return:
            ModelPO: Model plain object. None for model PO not found.
        """
        return ModelPO.objects(id=id_).first()

    @staticmethod
    def get_models_by_name(name: str, **kwargs) -> List[ModelPO]:
        """Get a list of model plain object given model name, framework and engine.

        Args:
            name (str): model name.
            kwargs (dict): A dictionary of arguments:
                framework: model framework.
                engine: model engine.
        Return:
            List[ModelPO]: A list of model plain objects.
        """
        return ModelPO.objects(name=name, **kwargs)

    @staticmethod
    def get_models_by_task(task: str) -> List[ModelPO]:
        """Get a list of model plain objects given task.

        Args:
            task (str): Model predictive or descriptive task name

        Return:
            List[ModelPO]: A list of model plain objects. An empty list will be returned if no such model.
        """
        return ModelPO.objects(task=task)

    @staticmethod
    def save_model(model: ModelPO, force_insert=False) -> ModelPO:
        """Save a model PO.

        Args:
            model (ModelPO): Model plain object to be saved.
            force_insert (bool): Only try to create a new document. Default to `False`.

        Returns:
            ModelPo: Updated model plain object.
        """
        return model.save(force_insert=force_insert)

    @staticmethod
    def update_model(id_: ObjectId, **kwargs) -> ModelPO:  # TODO: try ModelPO.objects(...).update()?
        """
        Update or register model PO.

        Args:
            id_ (ObjectId): Model ID.
            **kwargs: Keyword arguments to be updated.

        Returns:
            ModelPO:
        """
        return ModelPO.objects(id=id_).update(**kwargs)

    @staticmethod
    def delete_model_by_id(id_: ObjectId) -> int:
        """Delete model given model ID.

        Args:
            id_ (ObjectId): Model ID.

        Return:
            int: number of affected rows.
        """
        return ModelPO.objects(id=id_).delete()

    @staticmethod
    def register_static_profiling_result(
            id_: ObjectId,
            static_profiling_result: StaticProfileResultPO
    ) -> int:
        """Register static profiling result.

        Args:
            id_ (objectId): ID of the model, where the static profiling result is added.
            static_profiling_result (StaticProfileResultPO): Static profiling result.

        Return:
            int: number of affected rows.
        """
        return ModelPO.objects(id=id_).update_one(set__profile_result__static_profile_result=static_profiling_result)

    @staticmethod
    def register_dynamic_profiling_result(
            id_: ObjectId,
            dynamic_profiling_result: DynamicProfileResultPO
    ) -> int:
        """Register dynamic profiling result.

        Args:
            id_ (ObjectId): ID of the model, where the static profiling result is appended.
            dynamic_profiling_result (DynamicProfileResultPO): Dynamic profiling result.

        Return:
            int: number of affected rows.
        """
        return ModelPO.objects(id=id_).update_one(
            push__profile_result__dynamic_profile_results=dynamic_profiling_result)

    @staticmethod
    def is_dynamic_profiling_result_exist(
            id_: ObjectId,
            dynamic_profiling_result: DynamicProfileResultPO
    ) -> bool:
        """Check if the dynamic profiling result exist.

        Args:
            id_ (ObjectId): ID of the model.
            dynamic_profiling_result (DynamicProfileResultPO): Dynamic profiling result to be checked.

        Return:
             bool: True for exist, False otherwise.
        """
        return bool(ModelPO.objects(
            id=id_,
            profile_result__dynamic_profile_results__ip=dynamic_profiling_result.ip,
            profile_result__dynamic_profile_results__device_id=dynamic_profiling_result.device_id))

    @staticmethod
    def update_dynamic_profiling_result(
            id_: ObjectId,
            dynamic_profiling_result: DynamicProfileResultPO
    ) -> int:
        """Update dynamic profiling result.

        Args:
            id_ (ObjectId): ID of the model.
            dynamic_profiling_result (DynamicProfileResultPO): Dynamic profiling result to be updated.

        Return:
            int: number of affected rows.
        """
        return ModelPO.objects(
            id=id_,
            profile_result__dynamic_profile_results__ip=dynamic_profiling_result.ip,
            profile_result__dynamic_profile_results__device_id=dynamic_profiling_result.device_id) \
            .update(
            set__profile_result__dynamic_profile_results__S=dynamic_profiling_result
        )
