import configparser
from enum import Enum, unique

from jinja2 import Environment, FileSystemLoader

from modelci.types.bo import Engine
from modelci.utils.misc import get_device

jinja = Environment(
    loader=FileSystemLoader('./k8s/templates'),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True
)

@unique
class StorageType(Enum):
    """Enumerator of remote storage type.
    """
    NONE = 0
    S3 = 1

def generate_object_list(objects, uppercase_name: bool = False, is_env = False):
    env_object_list = []
    for name, value in objects.items():
        env_object = {}
        env_object['name'] = name.upper() if uppercase_name else name
        if is_env and str(value).isdigit():
            value = f'"{str(value)}"' # Digit in env must be with quotes. Due to a known issue of kubernetes: https://github.com/kubernetes/kubernetes/issues/82296
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
    init_container['model_dir'] = model_conf['local_model_dir']
    deployment['model_dir'] = model_conf['local_model_dir']

    # for init container
    storage_config = dict(config['remote_storage'])
    remote_storage = storage_config.pop('storage_type').upper()
    storage_type = StorageType[remote_storage]

    init_env_object_list = generate_object_list(
        dict(**storage_config, **model_conf),
        uppercase_name=True,
        is_env=True
    )

    if storage_type == StorageType.S3:
        # S3 model pulling with a sample image
        init_container['image'] = 'ferdinandzhong/s3-bucket-rw-docker:latest'
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
        port_objects['http-port'] = 8501
        port_objects['grpc-port'] = 8500
        env_objects['MODEL_NAME'] = model_conf['local_model_name']
    elif engine == Engine.TORCHSCRIPT:
        docker_tag = 'latest-gpu' if cuda else 'latest'
        deployment['image'] = f'mlmodelci/pytorch-serving:{docker_tag}'
        deployment['command'] = '["python"]'
        deployment['args'] = f'["pytorch_serve.py", "{model_conf["local_model_name"]}"]'
        port_objects['http-port'] = 8000
        port_objects['grpc-port'] = 8001
        env_objects['MODEL_NAME'] = model_conf['local_model_name']
    elif engine == Engine.ONNX:
        docker_tag = 'latest-gpu' if cuda else 'latest'
        deployment['command'] = '["python"]'
        deployment['args'] = f'["onnx_serve.py", "{model_conf["local_model_name"]}"]'
        deployment['image'] = f'mlmodelci/onnx-serving:{docker_tag}'
        port_objects['http-port'] = 8000
        port_objects['grpc-port'] = 8001
        env_objects['MODEL_NAME'] = model_conf['local_model_name']
    elif engine == Engine.TRT:
        if not cuda:
            raise RuntimeError('TensorRT cannot be run without CUDA. Please specify a CUDA device.')

        deployment['image'] = 'nvcr.io/nvidia/tensorrtserver:19.10-py3'
        # ulimits currently can't be set on kubernetes
        #Check open issue: https://github.com/kubernetes/kubernetes/issues/3595
        deployment['args'] = f'["trtserver", "--model-store={model_conf["local_model_dir"]}"]'
        port_objects['http-port'] = 8000
        port_objects['grpc-port'] = 8001
        port_objects['prometheus-port'] = 8002
    else:
        raise RuntimeError(f"Not able to serve model with engine `{deployment['engine']}`.")

    env_object_list = generate_object_list(env_objects, is_env=True)
    port_object_list = generate_object_list(port_objects)
    if additional_environment:
        env_object_list = env_object_list + generate_object_list(additional_environment, is_env=True)

    deployment_content = template.render(
        init_container=init_container,
        init_env_object_list=init_env_object_list,
        deployment=deployment,
        env_object_list=env_object_list,
        port_object_list=port_object_list
    )

    with open(output_file_path, 'w+') as output_target:
        output_target.write(deployment_content)
