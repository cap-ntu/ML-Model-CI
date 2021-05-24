from enum import Enum

import numpy as np


class NamedEnum(Enum):
    """
    A enumerator that can be initialized by name.

    Examples:
        Create a Color Enum:
        >>> class Color(NamedEnum):
        ...     RED = 0
        ...     GREEN = 1
        ...     BLUE = 2

        You can create a Color enum object by its name:
        >>> color = Color('RED')
        ... print(color)
        <Color.RED: 0>
    """

    @classmethod
    def get_case_insensitive(cls):
        return True

    @classmethod
    def _missing_(cls, value):
        if cls.get_case_insensitive():
            condition = lambda m: str(m.name).lower() == str(value).lower()
        else:
            condition = lambda m: m.name == value
        for member in cls:
            if condition(member):
                # save to value -> member mapper
                cls._value2member_map_[value] = member
                return member

    @classmethod
    def __modify_schema__(cls, field_schema: dict):
        """
        Use enum name rather than enum value for schema used by OpenAPI.
        """
        field_schema.update(enum=[item.name for item in cls])


class DataType(NamedEnum):
    TYPE_INVALID = 0
    TYPE_BOOL = 1
    TYPE_UINT8 = 2
    TYPE_UINT16 = 3
    TYPE_UINT32 = 4
    TYPE_UINT64 = 5
    TYPE_INT8 = 6
    TYPE_INT16 = 7
    TYPE_INT32 = 8
    TYPE_INT64 = 9
    TYPE_FP16 = 10
    TYPE_FP32 = 11
    TYPE_FP64 = 12
    TYPE_STRING = 13


def model_data_type_to_np(model_dtype):

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