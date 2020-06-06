from bson import ObjectId

from . import mongo
from .bo.dynamic_profile_result_bo import DynamicProfileResultBO
from .bo.model_bo import ModelBO
from .bo.model_objects import Framework, Engine
from .bo.static_profile_result_bo import StaticProfileResultBO
from .dao.model_dao import ModelDAO
from .exceptions import ServiceException


class ModelService(object):
    __model_DAO = ModelDAO

    @classmethod
    def get_models_by_name(cls, name: str, framework: Framework = None, engine: Engine = None):
        """Get a list of model BO given name

        Args:
            name (str): model name for searching
            framework (Framework): model framework. Default to None, having no effect on the searching result.
            engine (Engine): model engine. Default to None, having no effect on the searching result.

        Return:
            A list of `ModelBO`
        """
        # build kwargs
        kwargs = dict()
        if framework is not None:
            kwargs['framework'] = framework.value
        if engine is not None:
            kwargs['engine'] = engine.value

        model_pos = cls.__model_DAO.get_models_by_name(name=name, **kwargs)
        models = list(map(ModelBO.from_model_po, model_pos))
        return models

    @classmethod
    def get_models_by_task(cls, task: str):
        """Get a list of model BO given task.

        Args:
            task (str): model task for searching

        Return:
            A list of `ModelBO`
        """
        model_pos = cls.__model_DAO.get_models_by_task(task=task)

        return list(map(ModelBO.from_model_po, model_pos))

    @classmethod
    def get_model_by_id(cls, id_: str):
        """Get model given model ID.

        Args:
            id_ (str): model ID, should be a valid BSON Object ID

        Return:
             A model BO if found
        """
        return ModelBO.from_model_po(cls.__model_DAO.get_model_by_id(ObjectId(id_)))

    @classmethod
    def post_model(cls, model: ModelBO):
        """Register a model into ModelDB and GridFS. `model.id` should be set as `None`, otherwise, the function will
        raise a `ValueError`.

        Args:
            model (ModelBO): model business object to be registered

        Return:
            bool: True if successful, False otherwise

        Raises:
            ValueError: If `model.id` is not None.
            ServiceException: If model has exists with the same primary keys (name, framework, engine and version).
        """
        # check model id
        if model.id is not None:
            raise ValueError('field `id` is expected `None`, but got {}. You should use `update_model` for a existing '
                             'model BO'.format(model.id))

        model_po = model.to_model_po()
        if cls.__model_DAO.exists_by_primary_keys(
                name=model_po.name,
                framework=model_po.framework,
                engine=model_po.engine,
                version=model_po.version
        ):
            raise ServiceException(
                f'Model business object with primary keys name={model_po.name}, framework={model_po.framework}, '
                f'engine={model_po.engine}, version={model_po.version} has exists.'
            )

        return bool(cls.__model_DAO.save_model(model_po))

    @classmethod
    def update_model(cls, model: ModelBO, force_insert=False):
        """Update a model to ModelDB and GridFS. The function will check the existence of the provided model. It will
            invoke the update.

        Args:
            model (ModelBO): model business object to be updated. The model must exist in the ModelDB based on its
                `id`. Otherwise, you should set `force_insert` to be `True`.
            force_insert (bool: `True`, optional): Force insert flag for ModelDB. The model will force to insert
                regardless its existence.

        Return:
            True for successfully update, False otherwise.

        Raises:
            ValueError: If `model.id` does not exist in ModelDB, and `force_insert` is not set.
        """
        # check for the existence of Model
        if cls.__model_DAO.exists_by_id(model.id):
            model_po_new = model.to_model_po()
            model_po = cls.__model_DAO.get_model_by_id(model.id)

            # if weight changes, save all without patch
            if model_po_new.weight is not None and model_po_new.weight != model_po.weight:
                return bool(cls.__model_DAO.save_model(model_po_new))

            # if list field changes
            # TODO update dynamic list

            # build arguments
            valid_keys = [
                'name', 'framework', 'engine', 'version', 'dataset',
                'accuracy', 'weight', 'task', 'inputs', 'outputs', 'status'
            ]
            kwargs = dict()

            for valid_key in valid_keys:
                new_value = getattr(model_po_new, valid_key)
                if new_value is not None and new_value != getattr(model_po, valid_key):
                    kwargs[valid_key] = new_value

            # if kwargs is empty, not update
            if len(kwargs) == 0:
                return False

            return bool(cls.__model_DAO.update_model(model.id, **kwargs))
            # return bool(cls.__model_DAO.update_model(model_po_new))
        else:
            # if `force_insert` is set
            if force_insert:
                model_po = model.to_model_po()
                return bool(cls.__model_DAO.save_model(model_po))
            else:
                raise ValueError('Model ID {} does not exist. You may change the ID or set `force_insert=True` '
                                 'when call.'.format(model.id))

    @classmethod
    def delete_model_by_id(cls, id_: str):
        """Delete a model from ModelDB given ID.

        Args:
            id_ (str): ID of the object

        Return:
            bool: True for successful, False otherwise
        """
        id_ = ObjectId(id_)
        model_po = cls.__model_DAO.get_model_by_id(id_)
        model_po.weight.delete()
        return cls.__model_DAO.delete_model_by_id(id_)

    @classmethod
    def register_static_profiling_result(cls, id_, static_result: StaticProfileResultBO):
        """Register or update static profiling result to a model.

        Args:
             id_ (str): ID of the object
             static_result (StaticProfileResultBO): static profiling result

         Return:
             int: number of affected rows

         Raises:
            ValueError: `model.id` does not exist in ModelDB
        """
        id_ = ObjectId(id_)
        if ModelService.__model_DAO.exists_by_id(id_):
            return ModelService.__model_DAO.register_static_profiling_result(
                id_,
                static_result.to_static_profile_result_po()
            )
        else:
            raise ValueError('Model ID {} does not exist.'.format(id_))

    @classmethod
    def append_dynamic_profiling_result(cls, id_: str, dynamic_result: DynamicProfileResultBO):
        """Add one dynamic profiling result to a model.

        Args:
             id_ (str): ID of the object
             dynamic_result (DynamicProfileResultBO): Dynamic profiling result

         Return:
             int: number of affected rows

         Raises:
            ValueError: `model.id` does not exist in ModelDB
        """
        id_ = ObjectId(id_)
        if cls.__model_DAO.exists_by_id(id_):
            return cls.__model_DAO.register_dynamic_profiling_result(
                id_,
                dynamic_result.to_dynamic_profile_result_po()
            )
        else:
            raise ValueError('Model ID {} does not exist.'.format(id_))

    @classmethod
    def update_dynamic_profiling_result(cls, id_: str, dynamic_result: DynamicProfileResultBO, force_insert=False):
        """Update one dynamic profiling result to a model.

        Args:
            id_ (str): ID of the object
            dynamic_result (DynamicProfileResultBO): Dynamic profiling result
            force_insert: force to insert the dynamic result if it is not found

        Return:
            int: number of affected rows

        Raise:
            ValueError: Model ID does not exist in ModelDB; or
                dynamic profiling result to be updated does not exist and `force_insert` is not set
        """
        id_ = ObjectId(id_)
        dpr_po = dynamic_result.to_dynamic_profile_result_po()
        # if model ID exists
        if cls.__model_DAO.exists_by_id(id_):
            # if the dynamic profiling result to be updated exists
            if cls.__model_DAO.is_dynamic_profiling_result_exist(id_, dpr_po):
                return cls.__model_DAO.update_dynamic_profiling_result(id_, dpr_po)
            # force insert is set
            elif force_insert:
                return cls.__model_DAO.register_dynamic_profiling_result(id_, dpr_po)
            else:
                raise ValueError(
                    f'Dynamic profiling result to be updated with ip={dpr_po.ip}, device_id={dpr_po.device_id} does \n'
                    f'not exist. Either set `force_insert` or change the ip and device_id'
                )
        # if model ID does not exist
        else:
            raise ValueError('Model ID {} does not exist.'.format(id_))


__all__ = ['mongo', 'ModelService']
