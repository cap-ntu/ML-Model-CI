from collections import defaultdict

import numpy as np
import onnxconverter_common
from onnx import TensorProto


def type_to_data_type(tensor_type):
    import tensorflow as tf
    import torch

    from modelci.types.models.common import DataType

    mapper = defaultdict(
        lambda: DataType.TYPE_INVALID, {
            bool: DataType.TYPE_BOOL,
            int: DataType.TYPE_INT32,
            float: DataType.TYPE_FP32,
            str: DataType.TYPE_STRING,
            torch.bool: DataType.TYPE_BOOL,
            torch.uint8: DataType.TYPE_UINT8,
            torch.int: DataType.TYPE_INT32,
            torch.int8: DataType.TYPE_INT8,
            torch.int16: DataType.TYPE_INT16,
            torch.int32: DataType.TYPE_INT32,
            torch.int64: DataType.TYPE_INT64,
            torch.float: DataType.TYPE_FP32,
            torch.float16: DataType.TYPE_FP16,
            torch.float32: DataType.TYPE_FP32,
            torch.float64: DataType.TYPE_FP64,
            tf.bool: DataType.TYPE_BOOL,
            tf.uint8: DataType.TYPE_UINT8,
            tf.uint16: DataType.TYPE_UINT16,
            tf.uint32: DataType.TYPE_UINT32,
            tf.uint64: DataType.TYPE_UINT64,
            tf.int8: DataType.TYPE_INT8,
            tf.int16: DataType.TYPE_INT16,
            tf.int32: DataType.TYPE_INT32,
            tf.int64: DataType.TYPE_INT64,
            tf.float16: DataType.TYPE_FP16,
            tf.float32: DataType.TYPE_FP32,
            tf.float64: DataType.TYPE_FP64,
            tf.string: DataType.TYPE_STRING,
            np.dtype(np.bool): DataType.TYPE_BOOL,
            np.dtype(np.uint8): DataType.TYPE_UINT8,
            np.dtype(np.uint16): DataType.TYPE_UINT16,
            np.dtype(np.uint32): DataType.TYPE_UINT32,
            np.dtype(np.uint64): DataType.TYPE_UINT64,
            np.dtype(np.float16): DataType.TYPE_FP16,
            np.dtype(np.float32): DataType.TYPE_FP32,
            np.dtype(np.float64): DataType.TYPE_FP64,
            np.dtype(np.str): DataType.TYPE_STRING,
            TensorProto.UNDEFINED: DataType.TYPE_INVALID,
            TensorProto.FLOAT: DataType.TYPE_FP32,
            TensorProto.UINT8: DataType.TYPE_UINT8,
            TensorProto.INT8: DataType.TYPE_INT8,
            TensorProto.UINT16: DataType.TYPE_UINT16,
            TensorProto.INT16: DataType.TYPE_INT16,
            TensorProto.INT32: DataType.TYPE_INT32,
            TensorProto.INT64: DataType.TYPE_INT64,
            TensorProto.STRING: DataType.TYPE_STRING,
            TensorProto.BOOL: DataType.TYPE_BOOL,
            TensorProto.FLOAT16: DataType.TYPE_FP16,
            TensorProto.DOUBLE: DataType.TYPE_FP64,
            TensorProto.UINT32: DataType.TYPE_UINT32,
            TensorProto.UINT64: DataType.TYPE_UINT64,
        }
    )

    return mapper[tensor_type]


def model_data_type_to_np(model_dtype):
    from modelci.types.bo import DataType

    mapper = {
        DataType.TYPE_INVALID: None,
        DataType.TYPE_BOOL: np.bool,
        DataType.TYPE_UINT8: np.uint8,
        DataType.TYPE_UINT16: np.uint16,
        DataType.TYPE_UINT32: np.uint32,
        DataType.TYPE_UINT64: np.uint64,
        DataType.TYPE_INT8: np.int8,
        DataType.TYPE_INT16: np.int16,
        DataType.TYPE_INT32: np.int32,
        DataType.TYPE_INT64: np.int64,
        DataType.TYPE_FP16: np.float16,
        DataType.TYPE_FP32: np.float32,
        DataType.TYPE_FP64: np.float64,
        DataType.TYPE_STRING: np.dtype(object)
    }

    if isinstance(model_dtype, int):
        model_dtype = DataType(model_dtype)
    elif isinstance(model_dtype, str):
        model_dtype = DataType[model_dtype]
    elif not isinstance(model_dtype, DataType):
        raise TypeError(
            f'model_dtype is expecting one of the type: `int`, `str`, or `DataType` but got {type(model_dtype)}'
        )
    return mapper[model_dtype]


def model_data_type_to_torch(model_dtype):
    from modelci.types.models.common import DataType
    import torch

    mapper = {
        DataType.TYPE_INVALID: None,
        DataType.TYPE_BOOL: torch.bool,
        DataType.TYPE_UINT8: torch.uint8,
        DataType.TYPE_INT8: torch.int8,
        DataType.TYPE_INT16: torch.int16,
        DataType.TYPE_INT32: torch.int32,
        DataType.TYPE_INT64: torch.int64,
        DataType.TYPE_FP16: torch.float16,
        DataType.TYPE_FP32: torch.float32,
        DataType.TYPE_FP64: torch.float64,
    }

    if isinstance(model_dtype, int):
        model_dtype = DataType(model_dtype)
    elif isinstance(model_dtype, str):
        model_dtype = DataType[model_dtype]
    elif not isinstance(model_dtype, DataType):
        raise TypeError(
            f'model_dtype is expecting one of the type: `int`, `str`, or `DataType` but got {type(model_dtype)}'
        )
    return mapper[model_dtype]


def model_data_type_to_onnx(model_dtype):
    from modelci.types.bo import DataType

    mapper = {
        DataType.TYPE_INVALID: onnxconverter_common,
        DataType.TYPE_BOOL: onnxconverter_common.BooleanTensorType,
        DataType.TYPE_INT32: onnxconverter_common.Int32TensorType,
        DataType.TYPE_INT64: onnxconverter_common.Int64TensorType,
        DataType.TYPE_FP32: onnxconverter_common.FloatTensorType,
        DataType.TYPE_FP64: onnxconverter_common.DoubleTensorType,
        DataType.TYPE_STRING: onnxconverter_common.StringType,
    }

    if isinstance(model_dtype, int):
        model_dtype = DataType(model_dtype)
    elif isinstance(model_dtype, str):
        model_dtype = DataType[model_dtype]
    elif not isinstance(model_dtype, DataType):
        raise TypeError(
            f'model_dtype is expecting one of the type: `int`, `str`, or `DataType` but got {type(model_dtype)}'
        )
    return mapper[model_dtype]
