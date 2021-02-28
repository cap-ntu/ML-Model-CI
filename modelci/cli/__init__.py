#  Copyright (c) NTU_CAP 2021. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import ast
import inspect
from typing import Any

import click
import typer
from pydantic import BaseModel
from typer.main import lenient_issubclass

from modelci.cli import model_cli
from modelci.cli.model_manager import modelhub, app as modelhub_app
from modelci.cli.modelci_service import service
from modelci.types.models.common import NamedEnum
from modelci.utils.misc import isgeneric


class PydanticModelParamType(click.ParamType):
    name = 'pydantic model Json'

    def __init__(self, annotation):
        self._annotation = annotation

    def convert(self, value, param, ctx):
        return self._annotation.parse_raw(value)


class DictParamType(click.ParamType):
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

        return {
            self._get_parser(key_type)(k): self._get_parser(value_type)(v)
            for k, v in ast.literal_eval(str(value)).items()
        }


def _get_click_type_wrapper(get_click_type):
    def supersede_get_click_type(
            *, annotation: Any, parameter_info: typer.main.ParameterInfo
    ) -> click.ParamType:
        """
        Monkey-patched typer for the support of customer type such as `pydantic` models and `Dict`.

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
    """Wrapper for generate enum converter."""

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


typer.main.get_click_type = _get_click_type_wrapper(typer.main.get_click_type)
typer.main.generate_enum_convertor = _generate_enum_convertor_wrapper(typer.main.generate_enum_convertor)
app = typer.Typer()


@app.callback()
def callback():
    """
    A complete MLOps platform for managing, converting and profiling models and
    then deploying models as cloud services (MLaaS)
    """


app.add_typer(modelhub_app, name='modelhub_typer')
typer_click_object: click.Group = typer.main.get_command(app)  # noqa
typer_click_object.add_command(service)
typer_click_object.add_command(modelhub)
typer_click_object.add_command(model_cli.commands)
typer_click_object.add_command(model_cli.models)

if __name__ == '__main__':
    typer_click_object()
