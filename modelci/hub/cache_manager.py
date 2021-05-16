import os
import subprocess
from typing import List

from modelci.hub import converter
from modelci.hub.utils import TensorRTPlatform
from modelci.types.models import MLModel, Engine

__all__ = ['get_remote_model_weight', 'get_remote_model_weights', 'delete_remote_weight']


def get_remote_model_weight(model: MLModel):
    """Download a local cache of model from remote ModelDB in a structured path. And generate a configuration file.
    TODO(lym):
        1. set force insert config.pbtxt
        2. set other options in generation of config.pbtxt (e.g. max batch size, instance group...)
    This function will keep a local cache of the used model in the path:
        `~/.modelci/<architecture_name>/<framework>-<engine>/<task>/<version>`
    Arguments:
        model (MLModel): MLModelobject.
    Return:
        Path: Model saved path.
    """
    save_path = model.saved_path

    save_path.parent.mkdir(exist_ok=True, parents=True)

    if not save_path.exists():
        # TODO save TFS or TRT model files from gridfs
        with open(str(save_path), 'wb') as f:
            f.write(bytes(model.weight))
        if model.engine == Engine.TFS:
            subprocess.call(['unzip', save_path, '-d', '/'])
            os.remove(save_path)
        elif model.engine == Engine.TRT:
            subprocess.call(['unzip', save_path, '-d', '/'])
            os.remove(save_path)

            converter.TRTConverter.generate_trt_config(
                save_path.parent,  # ~/.modelci/<model-arch-name>/<framework>-<engine>/<task>/
                inputs=model.inputs,
                outputs=model.outputs,
                arch_name=model.name,
                platform=TensorRTPlatform.TENSORFLOW_SAVEDMODEL
            )

    return save_path


def get_remote_model_weights(models: List[MLModel]):
    """Get remote model weights from a list of models.
    Only models with highest version of each unique task, architecture, framework, and engine pair are download.
    """

    # group by (task, architecture, framework, engine) pair
    pairs = set(map(lambda x: (x.task, x.architecture, x.framework, x.engine), models))
    model_groups = [
        sorted(
            [model for model in models if (model.task, model.architecture, model.framework, model.engine) == pair],
            key=lambda model: model.version, reverse=True
        ) for pair
        in pairs
    ]

    # get weights of newest version of each pair
    for model_group in model_groups:
        get_remote_model_weight(model_group[0])


def delete_remote_weight(model: MLModel):
    save_path = model.saved_path

    if os.path.isfile(save_path):
        os.remove(save_path)
    elif os.path.isdir(save_path):
        os.removedirs(save_path)
