# -*- coding: utf-8 -*-
"""Model service utility business object.

This class contains utility classes used in model service for building model business object (`ModelBO`).
"""

from enum import Enum, unique
from typing import List, Union, Optional, BinaryIO

from mongoengine import GridFSProxy

from modelci.types.do import IOShapeDO
from modelci.types.trtis_objects import ModelInputFormat, DataType as TRTIS_DataType


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


# Generic data type (same as Triton::DataType)
DataType = TRTIS_DataType


class ModelVersion(object):
    """Model version class.

    This class is an integer representing model version.


    Args:
        ver_string (Union[str, int]): version string. It should be an integer string.
    """

    def __init__(self, ver_string: Union[str, int]):
        """Initializer of version string.

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

    Args:
        shape (List[int]): the shape of the input or output tensor.
        dtype (DataType, type, str): The data type of the input or output tensor.
        name (str): Tensor name. Default to None.
        format (ModelInputFormat): Input format, used for TensorRT currently.
            Default to `ModelInputFormat.FORMAT_NONE`.
    """

    def __init__(
            self,
            shape: List[int],
            dtype: Union[type, str, DataType],
            name: str = None,
            format: ModelInputFormat = ModelInputFormat.FORMAT_NONE
    ):
        """Initializer of input/output shape."""
        from modelci.types.type_conversion import type_to_data_type

        # input / output name
        self.name = name
        # input / output tensor shape
        self.shape = shape
        # input format
        self.format = format
        if isinstance(dtype, str):
            try:
                # if the type name is unified python type
                dtype = type_to_data_type(eval(dtype))
            except NameError:
                # try if the dtype is `DataType`
                dtype = DataType[dtype.upper()]

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

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def example_shape(self):
        return self.shape[1:]

    @property
    def height(self):
        if self.format == ModelInputFormat.FORMAT_NONE:
            raise ValueError('No height for shape format of `ModelInputFormat.FORMAT_NONE`.')
        if self.format == ModelInputFormat.FORMAT_NCHW:
            return self.shape[2]
        if self.format == ModelInputFormat.FORMAT_NHWC:
            return self.shape[1]

    @property
    def width(self):
        if self.format == ModelInputFormat.FORMAT_NONE:
            raise ValueError('No width for shape format of `ModelInputFormat.FORMAT_NONE`.')
        if self.format == ModelInputFormat.FORMAT_NCHW:
            return self.shape[3]
        if self.format == ModelInputFormat.FORMAT_NHWC:
            return self.shape[2]

    @property
    def channel(self):
        if self.format == ModelInputFormat.FORMAT_NONE:
            raise ValueError('No channel for shape format of `ModelInputFormat.FORMAT_NONE`.')
        if self.format == ModelInputFormat.FORMAT_NCHW:
            return self.shape[1]
        if self.format == ModelInputFormat.FORMAT_NHWC:
            return self.shape[3]

    def to_io_shape_po(self):
        """Convert IO shape business object to IO shape plain object."""
        return IOShapeDO(name=self.name, shape=self.shape, dtype=self.dtype.name, format=self.format)

    @staticmethod
    def from_io_shape_po(io_shape_po: IOShapeDO):
        """Create IO shape business object from IO shape plain object."""
        io_shape_bo = IOShape(
            name=io_shape_po.name,
            shape=io_shape_po.shape,
            dtype=io_shape_po.dtype,
            format=ModelInputFormat(io_shape_po.format)
        )

        return io_shape_bo

    def __str__(self):
        return '{}, dtype={}, format={}'.format(self.shape, self.dtype, self.format.name)


class InfoTuple(object):
    """A triplet tuple containing overall, average, 50th percentile, 95th percentile, and 99th percentile values of a
    data over a period of time.

    Args:
        avg (float): the average value
        p5050th percentile, 95th percentile,
        and 99th percentile values,  of a data.
    """

    def __init__(self, avg: float, p50: float, p95: float, p99: float):
        """Initializer."""
        self.avg = avg
        self.p50 = p50
        self.p95 = p95
        self.p99 = p99

    def tolist(self):
        """Convert to a list of values."""
        return [self.avg, self.p50, self.p95, self.p99]

    def __str__(self):
        return str(self.tolist())


class Weight(object):
    """Model weight fetched from Grid FS.
    """

    def __init__(
            self,
            weight: Union[bytes, BinaryIO] = None,
            *,
            filename: str = 'dummy',  # TODO: file name auto-gen
            content_type: str = 'application/octet-stream',
            gridfs_file: Optional[GridFSProxy] = None,
            lazy_fetch: bool = True,
    ):
        """Initializer.

        Args:
            weight (Optional[Union[bytes, BinaryIO]]): model weight read from Grid FS.
            gridfs_file (Optional[GridFSProxy]): Grid FS object storing the weight. Default to None.
                This attribute is to keep track of Grid FS file, in order to perform a file update.
            lazy_fetch (bool): The weight is not loaded from MongoDB initially when this flag is set `True`. Default to
                `True`.
        """
        self._weight = weight
        self.gridfs_file = gridfs_file
        # Flag for the set of model weight
        self._dirty = False
        if self.gridfs_file is not None:
            self.filename = gridfs_file.filename
            self.content_type = gridfs_file.content_type
            if not lazy_fetch:
                # lazy fetch
                _ = self.weight
        else:
            self.filename = filename
            self.content_type = content_type

    @property
    def weight(self):
        """Weight binary For lazy fetch.
        """
        if self._weight is None and self.gridfs_file is not None:
            self._weight = self.gridfs_file.read()
            self.gridfs_file.seek(0)
        return self._weight

    @weight.setter
    def weight(self, new_weight: Union[bytes, BinaryIO]):
        if isinstance(new_weight, bytes) or isinstance(new_weight, BinaryIO):
            self._weight = new_weight
            # the weight is dirty
            self._dirty = True
        else:
            raise TypeError(f'weight expected to be one of `bytes`, `BinaryIO`, but got {type(new_weight)}')

    @property
    def md5(self):
        """File md5 got from GridFS for identification. Return None if the file is not uploaded to GridFS. Note that
        if you set the weight manually, the md5 value will not be updated, unless the new value is updated to GridFS.
        """
        if self.gridfs_file is not None:
            return self.gridfs_file.md5
        else:
            return None

    def is_dirty(self):
        return self._dirty

    def clear_dirty_flag(self):
        self._dirty = False
