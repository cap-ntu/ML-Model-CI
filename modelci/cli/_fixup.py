#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 3/3/2021

Fixup for typer to support more data type as annotations.
"""
import ast
import inspect
from typing import Any

import click
import typer
from pydantic import ValidationError, BaseModel
from pydantic.utils import lenient_issubclass

from modelci.types.models.common import NamedEnum
from modelci.utils.misc import isgeneric


class PydanticModelParamType(click.ParamType):
    """Customized type `pydantic.BaseModel` for typer parameter annotation.

    The input format of this type is the JSON string, whose format can be parsed by
    :meth:`BaseModel.parse_dict`.

    Examples:
        This fixup enables the parameter type annotation like:
        >>> from pydantic import BaseModel
        ...
        ... class ParamType(BaseModel):
        ...     some_param: int
        ...     other_param: str
        ...
        ...
        ... def main(param: ParamType = typer.Argument(...))
        ...     typer.echo(param.dict())
        ...
        ...
        ... if __name__ == '__main__':
        ...     typer.run(main)

        Then you can run the program:
        ```
        $ python main.py `'"some_param": 1, "other_param": "string"'`
        ```
        The expected result is
        ```
        {"some_param": 1, "other_param": "string"}
        ```
    """

    name = 'pydantic model Json'

    def __init__(self, annotation):
        self._annotation = annotation

    def convert(self, value, param, ctx):
        try:
            return self._annotation.parse_raw(value)
        except ValidationError as exc:
            typer.echo(exc, err=True, color=True)
            raise typer.Exit(code=1)


class DictParamType(click.ParamType):
    """Customized type `Dict[Any, Any]` for typer parameter annotation.

    The input format of this type is the JSON string that can be converted to a Python dictionary.

    Examples:
        This fixup enables the parameter type annotation:
        >>> from typing import Dict
        ...
        ... def main(str_int_dict: Dict[str, int] = typer.Argument(...))
        ...     typer.echo(str_int_dict)
        ...
        ...
        ... if __name__ == '__main__':
        ...     typer.run(main)
        You can run the program:
        ```shell script
        $ python main.py '{"item1": 1, "item2": 2}'
        ```
        The expected result is
        ```
        {"item1": 1, "item2": 2}
        ```
    """
    name = 'dict'

    def __init__(self, annotation):
        self._annotation = annotation

    @staticmethod
    def _get_parser(type_):
        if inspect.isclass(type_) and issubclass(type_, BaseModel):
            return type_.parse_obj
        else:
            return type_

    def convert(self, value, param, ctx):
        key_type, value_type = self._annotation.__args__

        try:
            return {
                self._get_parser(key_type)(k): self._get_parser(value_type)(v)
                for k, v in ast.literal_eval(str(value)).items()
            }
        except ValidationError as exc:
            typer.echo(exc, err=True, color=True)
            raise typer.Exit(422)


def _get_click_type_wrapper(get_click_type):
    """Wrapper for fixup `typer.get_click_type` function.

    It fixes up typer support for more data type as argument and option annotations.

    Args:
        get_click_type: The function :meth:`typer.get_click_type`.
    """

    def supersede_get_click_type(
            *, annotation: Any, parameter_info: typer.main.ParameterInfo
    ) -> click.ParamType:
        """Fixup for typer to support more customized class for type hint. Such hints can be argument and option
        annotations.

        Originated from https://github.com/tiangolo/typer/issues/111#issuecomment-635512182.
        """

        if isgeneric(annotation) and annotation.__origin__ is dict:
            return DictParamType(annotation)
        elif lenient_issubclass(annotation, BaseModel):
            return PydanticModelParamType(annotation)
        elif lenient_issubclass(annotation, NamedEnum):
            return click.Choice(
                [item.name for item in annotation],
                case_sensitive=parameter_info.case_sensitive,
            )
        else:
            return get_click_type(annotation=annotation, parameter_info=parameter_info)

    return supersede_get_click_type


def _generate_enum_convertor_wrapper(enum_converter_factory):
    """Wrapper for fixup `typer.generate_enum_convertor` function.

    The original version of `generate_enum_convertor` will break the `IntEnum`, which utilized by ModelCI.
    By wrapping such function over the origin function, the typer can support `IntEnum` as argument and option
    annotations.

    Args:
        enum_converter_factory: The function :meth:`typer.generate_enum_convertor`.
    """

    def generate_named_enum_converter(enum):
        def convertor(value: Any) -> Any:
            if value is not None:
                low = str(value).lower()
                if low in lower_val_map:
                    key = lower_val_map[low]
                    return enum(key)

        if issubclass(enum, NamedEnum):
            lower_val_map = {str(val.name).lower(): val for val in enum}
            return convertor
        else:
            return enum_converter_factory

    return generate_named_enum_converter
