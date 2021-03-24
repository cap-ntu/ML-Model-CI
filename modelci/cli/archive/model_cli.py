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


# TODO(ZHZ): remove the file after moving all functions to model_manager.py
import click
import requests

from modelci.config import app_settings
from modelci.persistence.service_ import get_models
from modelci.ui import model_view
from modelci.utils.misc import remove_dict_null


@click.command()
@click.argument('name', type=click.STRING, required=False)
@click.option(
    '-f', '--framework',
    type=click.Choice(['TensorFlow', 'PyTorch'], case_sensitive=False),
    help='Model framework.'
)
@click.option(
    '-e', '--engine',
    type=click.Choice(['NONE', 'TFS', 'TORCHSCRIPT', 'ONNX', 'TRT', 'TVM', 'CUSTOMIZED'], case_sensitive=False),
    help='Model serving engine.'
)
@click.option('-v', '--version', type=click.INT, help='Model version.')
@click.option('-a', '--all', 'list_all', type=click.BOOL, is_flag=True, help='Show all models.')
@click.option('-q', '--quiet', type=click.BOOL, is_flag=True, help='Only show numeric IDs.')
def models(name, framework, engine, version, list_all, quiet):
    payload = remove_dict_null({'name': name, 'framework': framework, 'engine': engine, 'version': version})
    model_list = get_models(**payload)
    model_view(model_list, list_all=list_all, quiet=quiet)





