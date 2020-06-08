import argparse
import logging
import re
from pathlib import Path
from typing import Union

import docker
from docker.models.containers import Container
from docker.types import Mount, Ulimit

from modelci.hub.deployer import config
from modelci.hub.manager import retrieve_model_by_name, retrieve_model_by_task
from modelci.hub.utils import parse_path
from modelci.persistence.bo import Framework, Engine
from modelci.utils.misc import remove_dict_null


def serve(
        save_path: Union[Path, str],
        device: str = 'cpu',
        name: str = None,
        batch_size: int = 16,
) -> Container:
    """Serve the given model save path in a Docker container.

    Args:
        save_path (Union[Path, str]): Saved path to the model.
        device (str): Device name. E.g.: cpu, cuda, cuda:1.
        name (str): Container name. Default to None.
        batch_size (int): Batch size for passing to serving containers.

    Returns:
        Container: Docker container object created.

    """

    info = parse_path(Path(save_path))
    architecture: str = info['architecture']
    engine: Engine = info['engine']

    # obtain device
    device_num = None
    if device == 'cpu':
        cuda = False
    else:
        # match something like cuda, cuda:0, cuda:1
        matched = re.match(r'^cuda(?::([0-9]+))?$', device)
        if matched is None:  # load with CPU
            logging.warning('Wrong device specification, using `cpu`.')
            cuda = False
        else:  # load with CUDA
            cuda = True
            device_num = matched.groups()[0]
            if device_num is None:
                device_num = 0

    docker_tag = 'latest-gpu' if cuda else 'latest'

    docker_client = docker.from_env()

    # set mount
    mounts = [Mount(target=f'/models/{architecture}', source=str(info['base_dir']), type='bind', read_only=True)]

    common_kwargs = remove_dict_null({'detach': True, 'auto_remove': True, 'mounts': mounts, 'name': name})
    environment = dict()

    if cuda:
        common_kwargs['runtime'] = 'nvidia'
        environment['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        environment['CUDA_VISIBLE_DEVICES'] = device_num

    if engine == Engine.TFS:
        ports = {'8501': config.TFS_HTTP_PORT, '8500': config.TFS_GRPC_PORT}
        environment['MODEL_NAME'] = architecture
        container = docker_client.containers.run(
            f'tensorflow/serving:{docker_tag}', environment=environment, ports=ports, **common_kwargs
        )
    elif engine == Engine.TORCHSCRIPT:
        ports = {'8000': config.TORCHSCRIPT_HTTP_PORT, '8001': config.TORCHSCRIPT_GRPC_PORT}
        environment['MODEL_NAME'] = architecture
        container = docker_client.containers.run(
            f'pytorch-serving:{docker_tag}', environment=environment, ports=ports, **common_kwargs
        )
    elif engine == Engine.ONNX:
        ports = {'8000': config.ONNX_HTTP_PORT, '8001': config.ONNX_GRPC_PORT}
        environment['MODEL_NAME'] = architecture
        container = docker_client.containers.run(
            f'onnx-serving:{docker_tag}', environment=environment, ports=ports, **common_kwargs
        )
    elif engine == Engine.TRT:
        if not cuda:
            raise RuntimeError('TensorRT cannot be run without CUDA. Please specify a CUDA device.')

        ports = {'8000': config.TRT_HTTP_PORT, '8001': config.TRT_GRPC_PORT, '8002': config.TRT_PROMETHEUS_PORT}
        ulimits = [Ulimit(name='memlock', soft=-1, hard=-1), Ulimit(name='stack', soft=67100864, hard=67100864)]
        trt_kwargs = {'ulimits': ulimits, 'shm_size': '1G'}
        container = docker_client.containers.run(
            f'nvcr.io/nvidia/tensorrtserver:19.10-py3', 'trtserver --model-repository=/models',
            environment=environment, ports=ports, **common_kwargs, **trt_kwargs,
        )
    else:
        raise RuntimeError(f'Not able to serve model with path `{str(save_path)}`.')

    return container


def serve_by_name(args):
    model = args.model
    framework = Framework[args.framework.upper()]
    engine = Engine[args.engine.upper()]

    model_bo = retrieve_model_by_name(architecture_name=model, framework=framework, engine=engine)
    serve(model_bo.saved_path, device=args.device, batch_size=args.bs)


def serve_by_task(args):
    model_bo = retrieve_model_by_task(task=args.task)
    serve(model_bo.saved_path, device=args.device, batch_size=args.bs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serving')
    subparsers = parser.add_subparsers()

    by_name_parser = subparsers.add_parser('name', help='Serving by name')
    by_name_parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
    by_name_parser.add_argument('-f', '--framework', type=str, required=True, help='Framework name')
    by_name_parser.add_argument('-e', '--engine', type=str, required=True, help='Engine name')
    by_name_parser.add_argument('--device', type=str, default='cpu', help='Serving device name. E.g.: `cpu`, `cuda:0`.')
    by_name_parser.add_argument('-b', '--bs', type=str, default=16, help='Batch size for serving.')
    by_name_parser.set_defaults(func=serve_by_name)

    by_task_parser = subparsers.add_parser('task', help='Serving by task')
    by_task_parser.add_argument('--task', type=str, required=True, help='task name')
    by_task_parser.add_argument('--device', type=str, default='cpu', help='Serving device name. E.g.: `cpu`, `cuda:0`.')
    by_name_parser.add_argument('-b', '--bs', type=str, default=16, help='Batch size for serving.')
    by_task_parser.set_defaults(func=serve_by_task)

    # parse argument
    args_ = parser.parse_args()
    args_.func(args_)
