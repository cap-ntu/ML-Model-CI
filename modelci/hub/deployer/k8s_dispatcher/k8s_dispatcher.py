import argparse
import configparser
from enum import Enum, unique

from jinja2 import Environment, FileSystemLoader

from modelci.hub.deployer import config
from modelci.types.bo import Framework, Engine
from modelci.utils.misc import remove_dict_null, get_device

jinja = Environment(loader = FileSystemLoader('./k8s/templates'), trim_blocks=True, lstrip_blocks=True)

@unique
class Storage_Type(Enum):
    """Enumerator of remote storage type.
    """
    NONE = 0
    S3 = 1

def generate_object_list(objects, uppercase_name:bool = False):
        env_object_list = []
        for name, value in objects.items():
            env_object = {}
            env_object['name'] = name.upper() if uppercase_name else name
            env_object['value'] = value
            env_object_list.append(env_object)
        return env_object_list

def serve(
        configuration: str = 'config.ini',
        output_file_path: str = './k8s/deployment.yaml',
        **additional_environment
):
    """Generate the kubernetes deployment yaml file with template.

    Args:
        configuration (str): Path of deployment configuration file. E.g.: ./sample.config.
        output_file_path (str): Path of output k8s deployment yaml file. E.g.: ./deployment.yaml.
        **additional_environment: addtional environment variables to be added to deployment.

    Returns:
        Write the deployment yaml file into expected path.

    """
    config = configparser.ConfigParser()
    config.read(configuration)

    init_container = dict()
    deployment = dict()

    template = jinja.get_template('dispatch_api_template.yml')

    # mount volume specific setting.
    model_conf = dict(config['model'])
    model_path = f"{model_conf['local_model_dir']}/{model_conf['local_model_name']}"
    init_container['model_path'] = model_path
    deployment['model_path'] = model_path

    # for init container
    storage_config = dict(config['remote_storage'])
    remote_storage = storage_config.pop('storage_type').upper()
    storage_type = Storage_Type[remote_storage]

    init_env_object_list = generate_object_list(
        dict(**storage_config, **model_conf),
        uppercase_name=True
    )

    if storage_type == Storage_Type.S3:
        # S3 model pulling with a sample image
        init_container['image'] = 'ferdinandzhong/s3-bucket-rw-docker:latest'
        init_container['args'] = ['read_file.py']
    else:
        raise RuntimeError(f'`{remote_storage}` currently is not supported.')

    # for continer
    deployment_conf = dict(config['deployment'])

    deployment['name'] = deployment_conf['name']
    deployment['namespace'] = deployment_conf['namespace']
    deployment['replicas'] = deployment_conf['replicas']

    engine: Engine = Engine[deployment_conf['engine'].upper()]

    cuda, device_num = get_device(deployment_conf['device'])

    env_objects = dict()
    port_objects = dict()

    env_objects['BATCH_SIZE'] = deployment_conf['batch_size']
    if cuda:
        env_objects['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        env_objects['CUDA_VISIBLE_DEVICES'] = device_num

    if engine == Engine.TFS:
        docker_tag = '2.1.0-gpu' if cuda else '2.1.0'
        deployment['image'] = f'tensorflow/serving:{docker_tag}'
        port_objects['tfs_http_port'] = 8501
        port_objects['tfs_grpc_port'] = 8500
        env_objects['MODEL_NAME'] = model_conf['local_model_name']
    elif engine == Engine.TORCHSCRIPT:
        docker_tag = 'latest-gpu' if cuda else 'latest'
        deployment['image'] = f'mlmodelci/pytorch-serving:{docker_tag}'
        port_objects['torchscript_http_port'] = 8000
        port_objects['torchscript_grpc_port'] = 8001
        env_objects['MODEL_NAME'] = model_conf['local_model_name']
    elif engine == Engine.ONNX:
        docker_tag = 'latest-gpu' if cuda else 'latest'
        deployment['image'] = f'mlmodelci/onnx-serving:{docker_tag}'
        port_objects['onnx_http_port'] = 8000
        port_objects['onnx_grpc_port'] = 8001
        env_objects['MODEL_NAME'] = model_conf['local_model_name']
    elif engine == Engine.TRT:
        if not cuda:
            raise RuntimeError('TensorRT cannot be run without CUDA. Please specify a CUDA device.')

        deployment['image'] = f'nvcr.io/nvidia/tensorrtserver:19.10-py3'
        deployment['trt_model_repo'] = model_conf['local_model_dir']
        port_objects['trt_http_port'] = 8000
        port_objects['trt_grpc_port'] = 8001
        port_objects['trt_prometheus_port'] = 8002

        '''ulimits currently can't be set on kubernetes

        Check open issue: https://github.com/kubernetes/kubernetes/issues/3595
        '''
    else:
        raise RuntimeError(f"Not able to serve model with engine `{deployment['engine']}`.")

    env_object_list = generate_object_list(env_objects)
    port_object_list = generate_object_list(port_objects)
    if additional_environment:
        env_object_list= env_object_list + generate_object_list(additional_environment)

    deployment_content = template.render(
            init_container = init_container,
            init_env_object_list = init_env_object_list,
            deployment = deployment,
            env_object_list = env_object_list,
            port_object_list = port_object_list
        )
    with open(output_file_path, 'w+') as fp:
        fp.write(deployment_content)


if __name__ == "__main__":
    serve(
        configuration='./modelci/hub/deployer/k8s_dispatcher/sample.config',
        output_file_path='./modelci/hub/deployer/k8s_dispatcher/sample_output.yaml'
    )