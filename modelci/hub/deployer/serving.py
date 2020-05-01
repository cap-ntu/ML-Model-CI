import argparse
import subprocess
import sys
from pathlib import Path
from typing import Union

from modelci.persistence.bo import Framework, Engine
from ..deployer import config
from ..manager import retrieve_model_by_name, retrieve_model_by_task
from ..utils import parse_path


def serve(save_path: Union[Path, str], device: str = 'cpu'):
    """Serve the given model save path.

    Arguments:
        save_path (Union[Path, str]): Saved path to the model.
        device (str):
    """
    # TODO: CUDA device specification
    info = parse_path(Path(save_path))
    to = 'cpu' if device.lower() == 'cpu' else 'gpu'

    architecture: str = info['architecture']
    engine: Engine = info['engine']
    # TODO: change to subprocess.run, see https://stackoverflow.com/a/34873354; Return code
    if engine == Engine.TFS:
        subprocess.call(['sh', f'tfs/deploy_model_{to}.sh', architecture, config.TFS_HTTP_PORT, config.TFS_GRPC_PORT])
    elif engine == Engine.TORCHSCRIPT:
        subprocess.call(['sh', f'pytorch/deploy_model_{to}.sh', architecture, config.TORCHSCRIPT_HTTP_PORT])
    elif engine == Engine.ONNX:
        subprocess.call(['sh', f'onnx/deploy_model_{to}.sh', architecture, config.ONNX_HTTP_PORT])
    elif engine == Engine.TRT:
        subprocess.call(['sh', 'trt/deploy_model.sh', architecture, config.TRT_HTTP_PORT, config.TRT_GRPC_PORT,
                         config.TRT_PROMETHEUS_PORT])
    else:
        exit('Not supported.')


if __name__ == '__main__':
    # FIXME: bug exist, caused by serve by task
    parser = argparse.ArgumentParser(description='Serving')
    parser.add_argument('-m', '--model', type=str, help='Model name')
    parser.add_argument('-f', '--framework', type=str, help='Framework name')
    parser.add_argument('-e', '--engine', type=str, help='Engine name')
    parser.add_argument('--task', type=str, help='task name')

    # parse argument
    args = parser.parse_args()
    model = args.model
    framework = Framework[args.framework.upper()] if args.framework else None
    engine = Engine[args.engine.upper()] if args.engine else None

    # serve by name
    if model:
        if not framework:
            print('--framework is missing')
            parser.print_help(sys.stderr)
            sys.exit(1)
        elif not engine:
            print('--engine is missing')
            parser.print_help(sys.stderr)
            sys.exit(1)
        else:
            save_path = retrieve_model_by_name(architecture_name=model, framework=framework, engine=engine)
            serve(save_path)
    # serve by task
    elif bool(args.task):
        save_path = retrieve_model_by_task(task=args.task)
        serve(save_path)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)
