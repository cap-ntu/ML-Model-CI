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

import click
import typer

from modelci.cli import modelhub
from modelci.cli._fixup import _get_click_type_wrapper, _generate_enum_convertor_wrapper
from modelci.cli.service import service

# Fixup for typer argument and options annotations
typer.main.get_click_type = _get_click_type_wrapper(typer.main.get_click_type)
typer.main.generate_enum_convertor = _generate_enum_convertor_wrapper(typer.main.generate_enum_convertor)

app = typer.Typer()


@app.callback()
def callback():
    """
    A complete MLOps platform for managing, converting and profiling models and
    then deploying models as cloud services (MLaaS)
    """


app.add_typer(modelhub.app, name='modelhub')
typer_click_object: click.Group = typer.main.get_command(app)  # noqa
typer_click_object.add_command(service)

if __name__ == '__main__':
    typer_click_object()
