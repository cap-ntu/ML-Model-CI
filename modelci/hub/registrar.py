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
import tempfile
import hashlib
from pathlib import Path
from typing import List, Union

import cv2
import tensorflow as tf
import yaml
from shutil import copy2, make_archive, move
from modelci.hub.client.onnx_client import CVONNXClient
from modelci.hub.client.tfs_client import CVTFSClient
from modelci.hub.client.torch_client import CVTorchClient
from modelci.hub.client.trt_client import CVTRTClient
from modelci.hub.converter import converter
from modelci.hub.utils import parse_path_plain
from modelci.persistence.service import ModelService
from modelci.persistence.service_ import save
from modelci.types.models.mlmodel import MLModelFromYaml, MLModel
from urllib.request import urlopen, Request
from tqdm.auto import tqdm
from modelci.types.models.common import Engine, ModelStatus
from modelci.utils.misc import make_dir


def download_model_from_url(url, file_path, hash_prefix=None, progress=True):
    file_size = None
    req = Request(url)
    u = urlopen(req)  #nosec
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    print(file_size)

    dst = os.path.expanduser(file_path)
    print(dst)
    dst_dir = os.path.dirname(dst)
    make_dir(dst_dir)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def register_model(
        model: MLModel,
        convert: bool = True,
        profile: bool = True,
) -> List[MLModel]:
    """Upload a model to ModelDB.
    This function will upload the given model into the database with some variation. It may optionally generate a
        branch of models (i.e. model family) with different optimization techniques. Besides, a benchmark will be
        scheduled for each generated model, in order to gain profiling results for model selection strategies.
        In the `no_generate` model(i.e. `no_generate` flag is set to be `True`), `architecture`, `framework`, `engine`
        and `version` could be None. If any of the above arguments is `None`, all of them will be auto induced
        from the origin_model path. An `ValueError` will be raised if the mata info cannot be induced.

    TODO:
        This function has a super comprehensive logic, need to be simplified.

    Arguments:
        model: Required inputs for register a model. All information is wrapped in such model.
        convert (bool): Flag for generation of model family. Default to True.
        profile (bool): Flag for profiling uploaded (including converted) models. Default to True.
    """
    models = list()

    model_dir_list = list()
    model.model_status = [ModelStatus.PUBLISHED]
    models.append(save(model))

    # generate model family
    if convert:
        model_dir_list.extend(converter.generate_model_family(model))

    # register
    model_data = model.dict(exclude={'weight', 'id', 'model_status', 'engine'})
    for model_dir in model_dir_list:
        parse_result = parse_path_plain(model_dir)
        engine = parse_result['engine']

        model_cvt = MLModel(**model_data, weight=model_dir, engine=engine, model_status=[ModelStatus.CONVERTED])
        models.append(save(model_cvt))

    # profile registered model
    if profile:
        from modelci.controller import job_executor
        from modelci.controller.executor import Job

        file = tf.keras.utils.get_file(
            "grace_hopper.jpg",
            "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
        test_img_bytes = cv2.imread(file)

        kwargs = {
            'repeat_data': test_img_bytes,
            'batch_size': 32,
            'batch_num': 100,
            'asynchronous': False,
        }

        for model in models:
            model.model_status = [ModelStatus.PROFILING]
            ModelService.update_model(model)
            kwargs['model_info'] = model
            engine = model.engine

            if engine == Engine.TORCHSCRIPT:
                client = CVTorchClient(**kwargs)
            elif engine == Engine.TFS:
                client = CVTFSClient(**kwargs)
            elif engine == Engine.ONNX:
                client = CVONNXClient(**kwargs)
            elif engine == Engine.TRT:
                client = CVTRTClient(**kwargs)
            else:
                raise ValueError(f'No such serving engine: {engine}')

            job_cuda = Job(client=client, device='cuda:0', model_info=model)
            # job_cpu = Job(client=client, device='cpu', model_info=model)
            job_executor.submit(job_cuda)
            # job_executor.submit(job_cpu)

    return models


def register_model_from_yaml(file_path: Union[Path, str]):
    # check if file exist
    file_path = Path(file_path)
    assert file_path.exists(), f'Model definition file at {str(file_path)} does not exist'

    # read yaml
    with open(file_path) as f:
        model_config = yaml.safe_load(f)
    model_yaml = MLModelFromYaml.parse_obj(model_config)
    # copy model weight to cache directory
    model_in_saved_path = model_yaml.saved_path
    if model_in_saved_path != model_yaml.weight:
        copy2(model_yaml.weight, model_in_saved_path)

    # zip weight folder
    if model_yaml.engine == Engine.TFS:
        weight_dir = model_yaml.weight
        make_archive(weight_dir.with_suffix('.zip'), 'zip', weight_dir)

    model_data = model_yaml.dict(exclude_none=True, exclude={'convert', 'profile'})
    model = MLModel.parse_obj(model_data)
    register_model(model, convert=model_yaml.convert, profile=model_yaml.profile)

