import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from bson import ObjectId

from modelci.types.do import ModelDO
from .model_objects import Framework, Engine, ModelVersion, IOShape, Status, Weight
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
        acc (float): Model accuracy.
            TODO(lym): can be improved later Since different tasks use different metrics,
                this arg is better to be a dict {metric(str, input by users): value(float)}.
        task (str): Type of model detective or predictive task.
        inputs (List[IOShape]): Input shape and data type.
        outputs (List[IOShape]): Output shape and data type.
        weight (Weight): Model weight binary. Default to empty bytes.
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
        - create datetime
    """

    def __init__(
            self,
            name: str,
            framework: Framework,
            engine: Engine,
            version: ModelVersion,
            dataset: str,
            acc: float,
            task: str, inputs: List[IOShape],
            outputs: List[IOShape],
            weight: Weight = Weight(),
            profile_result: ProfileResultBO = None,
            status: Status = Status.UNKNOWN,
            create_time: datetime = datetime.now()
    ):
        """Initializer.
        """
        self._id: Optional[str] = None
        self.name = name
        self.framework = framework
        self.engine = engine
        self.version = version
        self.dataset = dataset
        self.acc = acc
        self.task = task
        self.inputs = inputs
        self.outputs = outputs
        self.weight = weight
        self.profile_result = profile_result
        self.status = status
        self.create_time = create_time

    @property
    def id(self):
        return self._id

    @property
    def saved_path(self):
        from ...hub.utils import generate_path

        filename = Path(self.weight.filename)
        save_path = generate_path(self.name, self.framework, self.engine, filename.stem)
        # add extension back
        save_path = save_path.with_suffix(filename.suffix)

        return save_path

    def to_model_po(self):
        """Convert business object to plain object.
        """
        input_dos = list(map(IOShape.to_io_shape_po, self.inputs))
        output_dos = list(map(IOShape.to_io_shape_po, self.outputs))

        model_do = ModelDO(
            name=self.name, framework=self.framework.value, engine=self.engine.value,
            version=self.version.ver, dataset=self.dataset, accuracy=self.acc,
            inputs=input_dos,
            outputs=output_dos,
            task=self.task, status=self.status.value,
            create_time=self.create_time
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
    def from_model_po(model_do: Optional[ModelDO], lazy_fetch=True):
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
            name=model_do.name, framework=Framework(model_do.framework), engine=Engine(model_do.engine),
            version=ModelVersion(model_do.version), dataset=model_do.dataset, acc=model_do.accuracy,
            inputs=inputs, outputs=outputs, task=model_do.task,
            status=Status(model_do.status), create_time=model_do.create_time
        )
        model._id = model_do.id

        model.weight = Weight(gridfs_file=model_do.weight, lazy_fetch=lazy_fetch)

        if model_do.profile_result is not None:
            model.profile_result = ProfileResultBO.from_profile_result_po(model_do.profile_result)

        return model

    def reload(self):
        """Reload model business object.
        """
        # TODO: reload
        raise NotImplementedError()
