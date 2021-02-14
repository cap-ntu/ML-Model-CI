import getpass
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from bson import ObjectId

from modelci.types.do import ModelDO
from .model_objects import Framework, Metric, Task, Engine, ModelVersion, IOShape, Status, Weight, ModelStatus
from .profile_result_bo import ProfileResultBO


class ModelBO(object):
    """Model business object.

    This is an model entity class for the use of communication protocol in business logic layer
    (i.e. `modelci.persistence.service.ModelService`).

    Args:
        name (str): Name of the architecture.
        framework (Framework): Model framework. E.g. TensorFlow, PyTorch.
        engine (Engine): Model engine. E.g. ONNX, TensorRT.
        dataset (str): Model training dataset.
        metric (Dict[Metric, float]): Scoring metrics used for model evaluation and corresponding evaluation scores
        task (Task): Type of model detective or predictive task.
        inputs (List[IOShape]): Input shape and data type.
        outputs (List[IOShape]): Output shape and data type.
        weight (Weight): Model weight binary. Default to empty bytes.
        model_status (List[ModelStatus]): Indicate the status of current model in its lifecycle
        profile_result (ProfileResultBO): Profiling result. Default to None.
        status (Status): Model running status. Default to `UNKNOWN`.
        create_time (datetime): Model registration time. Default to current datetime.

    It wraps information about a model:
        - model architecture name
        - framework
        - serving engine
        - model version
        - training dataset name
        - task
        - inputs standard (shape, data type, format, etc.)
        - outputs standard
        - model weights
        - profiling result (i.e. static profiling result and dynamic profiling results)
        - model checking status
        - provider (uploader) of the model
        - create datetime
    """

    def __init__(
            self,
            name: str,
            framework: Framework,
            engine: Engine,
            version: ModelVersion,
            dataset: str,
            metric: Dict[Metric, float],
            task: Task,
            inputs: List[IOShape],
            outputs: List[IOShape],
            parent_model_id: Optional[str] = None,
            model_status: List[ModelStatus] = None,
            weight: Weight = Weight(),
            profile_result: ProfileResultBO = None,
            status: Status = Status.UNKNOWN,
            creator: str = getpass.getuser(),
            create_time: datetime = datetime.now()
    ):
        """Initializer."""
        self._id: Optional[str] = None
        self.name = name
        self.framework = framework
        self.engine = engine
        self.version = version
        self.dataset = dataset
        self.metric = metric
        self.task = task
        self.parent_model_id = parent_model_id
        self.inputs = inputs
        self.outputs = outputs
        self.weight = weight
        self.model_status = model_status
        self.profile_result = profile_result
        self.status = status
        self.creator = creator
        self.create_time = create_time

    @property
    def id(self):
        return self._id

    @property
    def saved_path(self):
        from ...hub.utils import generate_path

        filename = Path(self.weight.filename)
        save_path = generate_path(self.name, self.task, self.framework, self.engine, filename.stem)
        # add extension back
        save_path = save_path.with_suffix(filename.suffix)

        return save_path

    def to_model_do(self):
        """Convert business object to plain object."""
        input_dos = list(map(IOShape.to_io_shape_po, self.inputs))
        output_dos = list(map(IOShape.to_io_shape_po, self.outputs))

        model_do = ModelDO(
            name=self.name,
            framework=self.framework.value,
            engine=self.engine.value,
            version=self.version.ver,
            dataset=self.dataset,
            metric={key.name.lower(): val for key, val in self.metric.items()},
            inputs=input_dos,
            outputs=output_dos,
            task=self.task.value,
            parent_model_id = self.parent_model_id,
            status=self.status.value,
            model_status=[item.value for item in self.model_status],
            creator=self.creator,
            create_time=self.create_time,
        )
        if self._id is not None:
            model_do.id = ObjectId(self._id)
            # patch save weight to Grid FS
            model_do.weight = self.weight.gridfs_file
            # compare with the newly loaded weight and the stored weight in MongoDB
            if self.weight.is_dirty():
                new_md5 = hashlib.md5(self.weight.weight).hexdigest()
                if new_md5 != self.weight.md5:
                    model_do.weight.replace(
                        self.weight.weight, filename=self.weight.filename, content_type=self.weight.content_type
                    )
                self.weight.clear_dirty_flag()
        else:
            model_do.weight.put(
                self.weight.weight, filename=self.weight.filename, content_type=self.weight.content_type
            )

        # convert profile result
        if self.profile_result is not None:
            model_do.profile_result = self.profile_result.to_profile_result_po()

        return model_do

    @staticmethod
    def from_model_do(model_do: Optional[ModelDO], lazy_fetch=True):
        """Create a business object from plain object.

        Args:
            model_do (Optional[ModelPO]): model plain object.
            lazy_fetch (bool): Flag for lazy fetching model weight. Default to `True`. To be implemented.
        """
        # test if model_po is none, this function is a null-safe function
        if model_do is None:
            return None

        inputs = list(map(IOShape.from_io_shape_po, model_do.inputs))
        outputs = list(map(IOShape.from_io_shape_po, model_do.outputs))

        model = ModelBO(
            name=model_do.name,
            framework=Framework(model_do.framework),
            engine=Engine(model_do.engine),
            version=ModelVersion(model_do.version),
            dataset=model_do.dataset,
            metric={Metric[key.upper()]: val for key, val in model_do.metric.items()},
            inputs=inputs,
            outputs=outputs,
            task=Task(model_do.task),
            parent_model_id=model_do.parent_model_id,
            status=Status(model_do.status),
            model_status=[ModelStatus(item) for item in model_do.model_status],
            creator=model_do.creator,
            create_time=model_do.create_time,
        )
        model._id = str(model_do.id)

        model.weight = Weight(gridfs_file=model_do.weight, lazy_fetch=lazy_fetch)

        if model_do.profile_result is not None:
            model.profile_result = ProfileResultBO.from_profile_result_po(model_do.profile_result)

        return model

    def reload(self):
        """Reload model business object."""
        # TODO: reload
        raise NotImplementedError('Method `reload` not implemented.')
