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
from pathlib import Path
from typing import Dict, List, Optional

import requests
import typer
import yaml
from pydantic import ValidationError

from modelci.config import app_settings
from modelci.types.models import Framework, Engine, IOShape, Task, Metric
from modelci.types.models import MLModelFromYaml, MLModel
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
    with requests.get(f'{app_settings.api_v1_prefix}/model/', params=payload) as r:
        model_list = r.json()
        model_view([model_list], list_all=list_all)


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

    from modelci.hub.publish import _download_model_from_url

    _download_model_from_url(url, path)
    logger.info(f'{path} model downloaded successfully.')


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


@app.command('convert')
def convert(
        way: str = typer.Argument(..., help= 'the type of converting, for example \'keras2onnx\''
                                             'means convert keras to onnx, available ways: keras2onnx, '
                                             'tf2onnx, keras2tfs'),
        model: str = typer.Argument(..., help='path of the model to be converted'
                                    'such as \'/home/model\''),
        save: str = typer.Argument(..., help='save path of the converted model'),
):
    import modelci.hub.converter.converter as cvt
    import time
    if way == 'keras2onnx':
        import onnx
        import tensorflow as tf
        loaded = tf.saved_model.load(model)
        localtime = time.strftime("%d%H%M%S", time.localtime())
        save = Path(save+f'/model_{str(localtime)}.onnx')
        onnx_model = cvt.convert(model=loaded, src_framework='keras', dst_framework='onnx')
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, save)
    if way == 'tf2onnx':
        import onnx
        localtime = time.strftime("%d%H%M%S", time.localtime())
        save = Path(save + f'/model_{str(localtime)}.onnx')
        onnx_model = cvt.convert(model=model, src_framework='tensorflow', dst_framework='onnx')
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, save)
    if way == 'keras2tfs':
        import tensorflow as tf
        localtime = time.strftime("%d%H%M%S", time.localtime())
        save = Path(save + f'/model_{str(localtime)}')
        loaded = tf.saved_model.load(model)
        cvt.convert(model=loaded, src_framework='tensorflow', dst_framework='tfs', save_path=save)

