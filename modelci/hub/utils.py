import re
from enum import unique, Enum
from pathlib import Path
from typing import Union

from modelci.persistence.bo import Framework, Engine, ModelVersion
from modelci.utils.trtis_objects import ModelInputFormat


def parse_path(path: Path):
    """Obtain filename, framework and engine from saved path.
    """

    if re.match(r'^.*?[!/]*/[a-z]+-[a-z]+/\d+$', str(path.with_suffix(''))):
        filename = path.name
        architecture = path.parent.parent.stem
        info = path.parent.name.split('-')
        framework = Framework[info[0].upper()]
        engine = Engine[info[1].upper()]
        version = ModelVersion(Path(filename).stem)
        return {
            'architecture': architecture,
            'framework': framework,
            'engine': engine,
            'version': version,
            'filename': filename
        }
    else:
        raise ValueError('Incorrect model path pattern')


def generate_path(model_name: str, framework: Framework, engine: Engine, version: Union[ModelVersion, str, int]):
    """Generate saved path from model
    """
    model_name = str(model_name)
    if not isinstance(framework, Framework):
        raise ValueError(f'Expecting framework type to be `Framework`, but got {type(framework)}')
    if not isinstance(engine, Engine):
        raise ValueError(f'Expecting engine type to be `Engine`, but got {type(engine)}')
    if not isinstance(version, ModelVersion):
        version = ModelVersion(str(version))

    return Path.home() / '.modelci' / model_name / f'{framework.name.lower()}-{engine.name.lower()}' / str(version)


def GiB(val):
    return val * 1 << 30


@unique
class TensorRTPlatform(Enum):
    """TensorRT platform type for model configuration
    """
    TENSORRT_PLAN = 0
    TENSORFLOW_GRAPHDEF = 1
    TENSORFLOW_SAVEDMODEL = 2
    CAFFE2_NETDEF = 3
    ONNXRUNTIME_ONNX = 4
    PYTORCH_LIBTORCH = 5
    CUSTOM = 6


TensorRTModelInputFormat = ModelInputFormat
