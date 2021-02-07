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

import os
import click
import requests

from modelci.utils import Logger

from modelci.app import SERVER_HOST, SERVER_PORT
from modelci.hub.init_data import export_model
from modelci.ui import model_view, model_detailed_view
from modelci.utils.misc import remove_dict_null
from modelci.hub.publish import _download_model_from_url
from modelci.hub.manager import register_model_from_yaml

logger = Logger(__name__)

@click.group()
def modelhub():
    pass

@modelhub.command("publish")
@click.option('-p', '--ymal_path', required=True, type=str, help='the yaml file path')
def register_model(ymal_path):
    """publish a model to our system

    Args:
        ymal_path ([type]): a ymal file that contains model registeration info
    """
    
    register_model_from_yaml(ymal_path)
    logger.info("model published")


@modelhub.command("list")
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
def show_models(name, framework, engine, version, list_all):
    """show a table that lists all models published in mlmodelci

    Args:
        name ([type]): [description]
        framework ([type]): [description]
        engine ([type]): [description]
        version ([type]): [description]
        list_all ([type]): [description]
    """
    payload = remove_dict_null({'name': name, 'framework': framework, 'engine': engine, 'version': version})
    with requests.get(f'http://{SERVER_HOST}:{SERVER_PORT}/api/v1/model/', params=payload) as r:
        model_list = r.json()
        model_view([model_list], list_all=list_all)


@modelhub.command()
def download_model():
    raise NotImplementedError

@modelhub.command("get")
@click.option('-u', '--url', required=True, type=str, help='the link to a model')
@click.option('-p', '--path', required=True, type=str, help='the saved path and file name.')
def download_model_from_url(url, path):
    """download a model weight file from an url

    Args:
        url ([type]): a model file url
        path ([type]): the saved path and file name
    """
    _download_model_from_url(url, path)
    logger.info("{} model downloaded succussfuly.".format(path))

