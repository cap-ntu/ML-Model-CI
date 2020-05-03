# -*- coding: utf-8 -*-
"""Model service utility business object.

This class contains utility classes used in model service for building model business object (`ModelBO`).
"""

from enum import Enum, unique
from typing import Tuple, List, Union, Optional, BinaryIO

from mongoengine import GridFSProxy

from modelci.utils.trtis_objects import ModelInputFormat
from ..po.model_po import IOShapePO
from ...utils import trtis_objects


@unique
class Framework(Enum):
    """Enumerator of framework saved model is used.
    """
    TENSORFLOW = 0
    PYTORCH = 1


@unique
class Engine(Enum):
    """Enumerator of model engine
    """
    NONE = 0
    TFS = 1
    TORCHSCRIPT = 2
    ONNX = 3
    TRT = 4
    TVM = 5
    CUSTOMIZED = 6


@unique
class Status(Enum):
    """Enumerator of model status
    """
    UNKNOWN = 0
    PASS = 1
    RUNNING = 2
    FAIL = 3


"""Generic data type (same as TRTIS::DataType"""
DataType = trtis_objects.DataType


class ModelVersion(object):
    """Model version class.

    This class is an integer representing model version.
    """

    def __init__(self, ver_string: Union[str, int]):
        """Initializer of version string.

        Args:
            ver_string (Union[str, int]): version string. It should be an integer string.

        Raise:
            ValueError: Version string is not integer.
        """
        # try convert to int
        try:
            ver = int(ver_string)
        except ValueError:
            raise ValueError('invalid value for version string, expected a number, got {}'.format(ver_string))

        self.ver = ver

    def __str__(self):
        return str(self.ver)


class IOShape(object):
    """Class for recording input and output shape with their data type.
    """

    def __init__(
            self,
            shape: List[int],
            dtype: Union[type, str, DataType],
            name: str = None,
            format: ModelInputFormat = ModelInputFormat.FORMAT_NONE
    ):
        """Initializer of input/output shape.

        Args:
            shape (List[int]): the shape of the input or output tensor.
            dtype (Union[type, str]): The data type of the input or output tensor.
            name (str): Tensor name. Default to None.
            format (ModelInputFormat): Input format, used for TensorRT currently.
                Default to `ModelInputFormat.FORMAT_NONE`.
        """
        from .type_conversion import type_to_data_type

        # input / output name
        self.name = name
        # input / output tensor shape
        self.shape = shape
        # input format
        self.format = format
        if isinstance(dtype, str):
            dtype = type_to_data_type(eval(dtype))
        elif isinstance(dtype, type):
            dtype = type_to_data_type(dtype)
        elif isinstance(dtype, DataType):
            pass
        else:
            raise ValueError(
                f'data type should be an instance of `type`, type name or `DataType`, but got {type(dtype)}'
            )

        # warning if the dtype is DataType.TYPE_INVALID
        if dtype == DataType.TYPE_INVALID:
            print('W: `dtype` is converted to invalid.')

        # input / output datatype
        self.dtype = dtype

    def to_io_shape_po(self):
        """Convert IO shape business object to IO shape plain object.
        """

        return IOShapePO(name=self.name, shape=self.shape, dtype=self.dtype.name, format=self.format)

    @staticmethod
    def from_io_shape_po(io_shape_po: IOShapePO):
        """Create IO shape business object from IO shape plain object
        """

        io_shape_bo = IOShape(
            name=io_shape_po.name,
            shape=io_shape_po.shape,
            dtype=io_shape_po.dtype,
            format=io_shape_po.format
        )

        return io_shape_bo

    def __str__(self):
        return '{}, dtype={}'.format(self.shape, self.dtype)


class InfoTuple(object):
    """A triplet tuple containing min, max and average values of a data over a period of time.
    """

    def __init__(self, info: Tuple[float, float, float]):
        """Initializer.

        Args:
            info (Tuple[float, float, float]): a truplet containing min, max and average values of a data.

        Raise:
            AssertionError: `info` is not a tuplet.
        """
        assert len(info) == 3
        self.min = info[0]
        self.max = info[1]
        self.avg = info[2]

    def tolist(self):
        """Converting InfoTuple to a lsit of values.
        """
        return [self.min, self.max, self.avg]

    def __str__(self):
        return '({}, {}, {})'.format(self.min, self.max, self.avg)


class Weight(object):
    """Model weight fetched from Grid FS.
    """

    def __init__(
            self,
            weight: Union[bytes, BinaryIO] = bytes(),
            *,
            filename: str = 'dummy',  # TODO: file name auto-gen
            content_type: str = 'application/octet-stream',
            gridfs_file: Optional[GridFSProxy] = None
    ):
        """Initializer.

        Args:
            weight (Optional[bytes]): model weight read from Grid FS.
            gridfs_file (Optional[GridFSProxy]): Grid FS object storing the weight. Default to None.
                This attribute is to keep track of Grid FS file, in order to perform a file udpate.
        """
        self.weight = weight
        self.gridfs_file = gridfs_file
        if gridfs_file is not None:
            self.weight = gridfs_file.read()
            gridfs_file.seek(0)
            self.filename = gridfs_file.filename
            self.content_type = gridfs_file.content_type
        else:
            self.filename = filename
            self.content_type = content_type
