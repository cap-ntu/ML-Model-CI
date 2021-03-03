#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: USER
Email: yli056@e.ntu.edu.sg
Date: 2/24/2021

Basic patterns for creation of required components.
"""
import ast
import inspect
from enum import Enum
from typing import Type, Any, Union

from fastapi import Form
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from pydantic.fields import ModelField, FieldInfo

from modelci.utils.misc import isgeneric


def _make_form_parameter(field_info: FieldInfo) -> Any:
    """
    Converts a field from a `Pydantic` model to the appropriate `FastAPI`
    parameter type.

    Args:
        field_info (FieldInfo): The field information to convert.

    Returns:
        A form.
    """
    return Form(
        default=field_info.default,
        alias=field_info.alias,
        title=field_info.title,
        description=field_info.description,
        gt=field_info.gt,
        lt=field_info.lt,
        le=field_info.le,
        min_length=field_info.min_length,
        max_length=field_info.max_length,
        regex=field_info.regex,
        **field_info.extra,
    )


def _make_form_enum(enum_cls: Type[Enum]):
    """
    Modify an :class:`Enum` class that uses int value to accept string value member.

    Args:
        enum_cls (Type[Enum]): An enum class.

    Returns:
        Type[Enum]: the modified enum class.

    """

    def _missing_(cls, value):
        for member in cls:
            if str(member.value) == value:
                # save to value -> member mapper
                cls._value2member_map_[value] = member
                return member
        return missing_old(value)

    if hasattr(enum_cls, '__form__') or all(isinstance(e.value, str) for e in enum_cls):
        return

    missing_old = getattr(enum_cls, '_missing_')
    setattr(enum_cls, '_missing_', classmethod(_missing_))
    setattr(enum_cls, '__form__', True)

    return enum_cls


def make_annotation(field: ModelField):
    """
    Convert a field annotation type to form data accepted type.

    The method convert structural field such as `BaseModel` and `Dict` to a str. Such as the model's value is
    supplied as a serialized JSON string format. Such string will be converted back to a dictionary, and used
    for initialize previous field.
    """

    field_outer_type = field.outer_type_
    is_literal = False

    # check outer type
    if isgeneric(field_outer_type):
        # outer type is a generic class
        if field_outer_type.__origin__ is Union:
            # only Union is valid generic class
            inner_types = field_outer_type.__args__
        else:
            return str, True
    else:
        inner_types = (field_outer_type,)
        field_outer_type = None

    # check inner types
    inner_types_new = list()
    for inner_type in inner_types:
        if inner_type in (str, int, float, ..., Any):
            # inner type of `str`, `int` and `float` will be natively used as form data value
            inner_types_new.append(inner_type)
        elif issubclass(inner_type, Enum):
            inner_types_new.append(_make_form_enum(inner_type))
        else:
            # other types will be converted to string literal
            is_literal = True
            inner_types_new.append(str)

    if field_outer_type is None:
        field_outer_type = inner_types_new[0]
    else:
        # set new generic type args
        field_outer_type = field_outer_type.__origin__[tuple(inner_types_new)]

    return field_outer_type, is_literal


def as_form(cls: Type[BaseModel]) -> Type[BaseModel]:
    """
    Adds an `as_form` class method to decorated models. The `as_form` class
    method can be used with `FastAPI` endpoints.

    TODO: auto generate OpenAPI example

    Args:
        cls: The model class to decorate.

    Returns:
        The decorated class.

    References:
        * https://github.com/tiangolo/fastapi/issues/2387#issuecomment-731662551
    """

    literal_fields = set()
    new_params = list()
    for field in cls.__fields__.values():
        annotation, is_literal = make_annotation(field)
        if is_literal:
            literal_fields.add(field.alias)
        new_params.append(
            inspect.Parameter(
                field.alias,
                inspect.Parameter.POSITIONAL_ONLY,
                default=_make_form_parameter(field.field_info),
                annotation=annotation,
            )
        )

    async def _as_form(**data):
        """
        Create the model as a form data.
        """

        # parse literal back to dictionary
        for field_alias in literal_fields:
            value = data.pop(field_alias, None)
            data[field_alias] = ast.literal_eval(str(value))
        try:
            cls.parse_obj(data)
            return cls(**data)
        except ValidationError as exc:
            raise RequestValidationError(exc.raw_errors)

    sig = inspect.signature(_as_form)
    sig = sig.replace(parameters=new_params)
    _as_form.__signature__ = sig
    setattr(cls, "as_form", _as_form)
    return cls
