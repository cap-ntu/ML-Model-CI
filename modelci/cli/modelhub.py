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
from http import HTTPStatus
from pathlib import Path
from shutil import copy2, make_archive
from typing import Dict, List, Optional
import json
import requests
import typer
import yaml
from pydantic import ValidationError
import modelci.persistence.service_ as ModelDB
from modelci.hub.converter import generate_model_family

from modelci.config import app_settings
from modelci.hub.utils import parse_path_plain
from modelci.types.models.common import Framework, Engine, Task, Metric, IOShape, ModelStatus
from modelci.types.models.mlmodel import MLModelFromYaml, MLModel, ModelUpdateSchema
from modelci.ui import model_view, model_detailed_view
from modelci.utils import Logger
from modelci.utils.misc import remove_dict_null

logger = Logger(__name__, welcome=False)

app = typer.Typer()


@app.callback()
def modelhub():
    pass


@app.command()
def publish(
        file_or_dir: Optional[Path] = typer.Argument(None, help='Model weight files', exists=True),
        architecture: Optional[str] = typer.Option(None, '-n', '--name', help='Architecture'),
        framework: Optional[Framework] = typer.Option(None, '-fw', '--framework', help='Framework'),
        engine: Optional[Engine] = typer.Option(None, '-e', '--engine', help='Engine'),
        version: Optional[int] = typer.Option(None, '-v', '--version', min=1, help='Version number'),
        task: Optional[Task] = typer.Option(None, '-t', '--task', help='Task'),
        dataset: Optional[str] = typer.Option(None, '-d', '--dataset', help='Dataset name'),
        metric: Dict[Metric, float] = typer.Option(
            '{}',
            help='Metrics in the form of mapping JSON string. The map type is '
                 '`Dict[types.models.mlmodel.Metric, float]`. An example is \'{"acc": 0.76}.\'',
        ),
        inputs: List[IOShape] = typer.Option(
            [],
            '-i', '--input',
            help='List of shape definitions for input tensors. An example of one shape definition is '
                 '\'{"name": "input", "shape": [-1, 3, 224, 224], "dtype": "TYPE_FP32", "format": "FORMAT_NCHW"}\'',
        ),
        outputs: List[IOShape] = typer.Option(
            [],
            '-o', '--output',
            help='List of shape definitions for output tensors. An example of one shape definition is '
                 '\'{"name": "output", "shape": [-1, 1000], "dtype": "TYPE_FP32"}\'',
        ),
        convert: Optional[bool] = typer.Option(
            True,
            '-c', '--convert',
            is_flag=True,
            help='Convert the model to other possible format.',
        ),
        profile: Optional[bool] = typer.Option(
            False,
            '-p', '--profile',
            is_flag=True,
            help='Profile the published model(s).',
        ),
        yaml_file: Optional[Path] = typer.Option(
            None, '-f', '--yaml-file', exists=True, file_okay=True,
            help='Path to configuration YAML file. You should either set the `yaml_file` field or fields '
                 '(`FILE_OR_DIR`, `--name`, `--framework`, `--engine`, `--version`, `--task`, `--dataset`,'
                 '`--metric`, `--input`, `--output`).'
        ),
):
    meta_info = (file_or_dir, architecture, framework, engine, version, task, dataset, metric, inputs, outputs)
    # check either using parameters, or using YAML
    if yaml_file and not any(meta_info):
        with open(yaml_file) as f:
            model_config = yaml.safe_load(f)
        try:
            model_yaml = MLModelFromYaml.parse_obj(model_config)
        except ValidationError as exc:
            typer.echo(exc, err=True, color=True)
            raise typer.Exit(422)
    elif not yaml_file and all(meta_info):
        model_yaml = MLModelFromYaml(
            weight=file_or_dir, architecture=architecture, framework=framework, engine=engine, version=version,  # noqa
            dataset=dataset, metric=metric, task=task, inputs=inputs, outputs=outputs, convert=convert, profile=profile
        )
    else:
        typer.echo('Incorrect parameter, you should set either YAML_FILE, or all of the (FILE_OR_DIR, --name,'
                   '--framework, --engine, --version, --task, --dataset, --metric, --input, --output)')
        raise typer.Exit(422)

    # build request parameters
    payload = {'convert': model_yaml.convert, 'profile': model_yaml.profile}
    data = model_yaml.dict(use_enum_values=True, exclude_none=True, exclude={'convert', 'profile', 'weight'})
    form_data = {k: str(v) for k, v in data.items()}
    file_or_dir = model_yaml.weight

    # read weight files
    files = list()
    key = 'files'
    try:
        # read weight file
        if file_or_dir.is_dir():
            for file in filter(Path.is_file, file_or_dir.rglob('*')):
                name = Path(file).relative_to(file_or_dir.parent)
                files.append((key, (str(name), open(file, 'rb'), 'application/example')))
        else:
            files.append((key, (file_or_dir.name, open(file_or_dir, 'rb'), 'application/example')))
        with requests.post(
                f'{app_settings.api_v1_prefix}/model/',
                params=payload, data=form_data, files=files,
        ) as r:
            typer.echo(r.json(), color=True)
    finally:
        for file in files:
            file[1][1].close()


@app.command('ls')
def list_models(
        architecture: Optional[str] = typer.Option(None, '-n', '--name', help='Model architecture name'),
        framework: Optional[Framework] = typer.Option(None, '-fw', '--framework', case_sensitive=False,
                                                      help='Framework'),
        engine: Optional[Engine] = typer.Option(None, '-e', '--engine', case_sensitive=False, help='Serving engine'),
        version: Optional[int] = typer.Option(None, '-v', '--version', help='Version'),
        list_all: Optional[bool] = typer.Option(
            False,
            '-a', '--all', is_flag=True,
            help='Display queried models. otherwise, only partial result will be shown.'
        ),
):
    """Show a table that lists all models published in MLModelCI"""

    payload = remove_dict_null(
        {'architecture': architecture, 'framework': framework, 'engine': engine, 'version': version}
    )
    with requests.get(f'{app_settings.api_v1_prefix}/model', params=payload) as r:
        model_list = r.json()
        model_view([MLModel.parse_obj(model) for model in model_list], list_all=list_all)


@app.command()
def download():
    """Download model from model hub. (Not implemented)."""
    raise NotImplementedError


@app.command('get')
def download_model_from_url(
        url: str = typer.Argument(..., help='The link to a model'),
        path: Path = typer.Argument(..., file_okay=True, help='The saved path and file name.')
):
    """Download a model weight file from an online URL."""

    from modelci.hub.registrar import download_model_from_url

    download_model_from_url(url, path)
    typer.echo(f'{path} model downloaded successfully.')


@app.command('export')
def export(
        name: str = typer.Option(..., '-n', '--name', help='Architecture'),
        framework: Optional[Framework] = typer.Option(None, '-fw', '--framework', case_sensitive=False,
                                                      help='Framework'),
        trt: Optional[bool] = typer.Option(
            False,
            is_flag=True,
            help='Flag for exporting models served by TensorRT. Please make sure you have TensorRT installed in your '
                 'machine before set this flag.')
):
    """
    Export model from PyTorch hub / TensorFlow hub and try convert the model into various format for different serving
    engines.
    """
    from modelci.hub.init_data import export_model

    export_model(model_name=name, framework=framework, enable_trt=trt)


@app.command('detail')
def detail(model_id: str = typer.Argument(..., help='Model ID')):
    """Show a single model."""
    with requests.get(f'{app_settings.api_v1_prefix}/model/{model_id}') as r:
        data = r.json()
        model_detailed_view(MLModel.parse_obj(data))


@app.command('update')
def update(
        model_id: str = typer.Argument(..., help='Model ID'),
        architecture: Optional[str] = typer.Option(None, '-n', '--name', help='Architecture'),
        framework: Optional[Framework] = typer.Option(None, '-fw', '--framework', help='Framework'),
        engine: Optional[Engine] = typer.Option(None, '-e', '--engine', help='Engine'),
        version: Optional[int] = typer.Option(None, '-v', '--version', min=1, help='Version number'),
        task: Optional[Task] = typer.Option(None, '-t', '--task', help='Task'),
        dataset: Optional[str] = typer.Option(None, '-d', '--dataset', help='Dataset name'),
        metric: Optional[Dict[Metric, float]] = typer.Option(
            None,
            help='Metrics in the form of mapping JSON string. The map type is '
                 '`Dict[types.models.mlmodel.Metric, float]`. An example is \'{"acc": 0.76}.\'',
        ),
        inputs: Optional[List[IOShape]] = typer.Option(
            [],
            '-i', '--input',
            help='List of shape definitions for input tensors. An example of one shape definition is '
                 '\'{"name": "input", "shape": [-1, 3, 224, 224], "dtype": "TYPE_FP32", "format": "FORMAT_NCHW"}\'',
        ),
        outputs: Optional[List[IOShape]] = typer.Option(
            [],
            '-o', '--output',
            help='List of shape definitions for output tensors. An example of one shape definition is '
                 '\'{"name": "output", "shape": [-1, 1000], "dtype": "TYPE_FP32"}\'',
        )
):
    model = ModelUpdateSchema(
        architecture=architecture, framework=framework, engine=engine, version=version,  # noqa
        dataset=dataset, metric=metric, task=task, inputs=inputs, outputs=outputs
    )

    with requests.patch(f'{app_settings.api_v1_prefix}/model/{model_id}',
                        data=model.json(exclude_defaults=True)) as r:
        data = r.json()
        model_detailed_view(MLModel.parse_obj(data))


@app.command('delete')
def delete(model_id: str = typer.Argument(..., help='Model ID')):
    with requests.delete(f'{app_settings.api_v1_prefix}/model/{model_id}') as r:
        if r.status_code == HTTPStatus.NO_CONTENT:
            typer.echo(f"Model {model_id} deleted")


@app.command('convert')
def convert(
        id: str = typer.Option(None, '-i', '--id', help='ID of model.'),
        yaml_file: Optional[Path] = typer.Option(
            None, '-f', '--yaml-file', exists=True, file_okay=True,
            help='Path to configuration YAML file. You should either set the `yaml_file` field or fields '
                 '(`FILE_OR_DIR`, `--name`, `--framework`, `--engine`, `--version`, `--task`, `--dataset`,'
                 '`--metric`, `--input`, `--output`).'
        ),
        register: bool = typer.Option(False, '-r', '--register', is_flag=True, help='register the converted models to modelhub, default false')
):
    model = None
    if id is None and yaml_file is None:
        raise ValueError('WARNING: Please assign a way to find the target model! details refer to --help')
    if id is not None and yaml_file is not None:
        raise ValueError('WARNING: Do not use -id and -path at the same time!')
    elif id is not None and yaml_file is None:
        if ModelDB.exists_by_id(id):
            model = ModelDB.get_by_id(id)
        else:
            typer.echo(f"model id: {id} does not exist in modelhub")
    elif id is None and yaml_file is not None:
        # get MLModel from yaml file
        with open(yaml_file) as f:
            model_config = yaml.safe_load(f)
        model_yaml = MLModelFromYaml.parse_obj(model_config)
        model_in_saved_path = model_yaml.saved_path
        if model_in_saved_path != model_yaml.weight:
            copy2(model_yaml.weight, model_in_saved_path)
        if model_yaml.engine == Engine.TFS:
            weight_dir = model_yaml.weight
            make_archive(weight_dir.with_suffix('.zip'), 'zip', weight_dir)

        model_data = model_yaml.dict(exclude_none=True, exclude={'convert', 'profile'})
        model = MLModel.parse_obj(model_data)

    # auto execute all possible convert and return a list of save paths of every converted model
    generated_dir_list = generate_model_family(model)
    typer.echo(f"Converted models are save in: {generated_dir_list}")
    if register:
        model_data = model.dict(exclude={'weight', 'id', 'model_status', 'engine'})
        for model_dir in generated_dir_list:
            parse_result = parse_path_plain(model_dir)
            engine = parse_result['engine']
            model_cvt = MLModel(**model_data, weight=model_dir, engine=engine, model_status=[ModelStatus.CONVERTED])
            ModelDB.save(model_cvt)
            typer.echo(f"converted {engine} are successfully registered in Modelhub")


@app.command('profile')
def profile(
        model_id: str = typer.Argument(..., help='Model ID'),
        device: str = typer.Option("cuda", '-d', '--device', help='device to pre-deploy the model.'),
        batch_size: int = typer.Option(None, '-b', '--batchsize', help='batchsize of the test input')
):
    args = {'id': model_id, 'device': device, 'batch_size': batch_size}
    with requests.get(f'{app_settings.api_v1_prefix}/profiler/{model_id}', params=args) as r:
        if r.status_code == 201:
            typer.echo("Profile successfully! Results are showed below:")
            json_response = json.dumps(r.json(), sort_keys=True, indent=4, separators=(',', ':'))
            typer.echo(json_response)
        else:
            raise ConnectionError("Can not connect to profile api!")
