import os
import random
import time

import docker
from modelci.hub.client import CVTFSClient, CVTorchClient, CVONNXClient, CVTRTClient
from modelci.metrics.benchmark.metric import BaseModelInspector
from modelci.persistence.exceptions import ServiceException
from modelci.types.models.mlmodel import MLModel, Engine
from modelci.types.models.profile_results import DynamicProfileResult, ProfileMemory, ProfileThroughput, ProfileLatency
from modelci.utils import Logger
from modelci.utils.misc import get_ip
from modelci.hub.deployer.dispatcher import serve
DEFAULT_BATCH_NUM = 100

random.seed(ord(os.urandom(1)))
logger = Logger(__name__, welcome=False)


class Profiler(object):
    """Profiler class, call this to test model performance.

    Args:
        model_info (MLModel): Information about the model, can get from `retrieve_model` method.
        server_name (str): to assign a name for the container you are creating for model profile
        inspector (BaseModelInspector): The client instance implemented from :class:`BaseModelInspector`.
    """

    def __init__(self, model_info: MLModel, server_name: str, inspector: BaseModelInspector = None):
        """Init a profiler object."""
        self.server_name = server_name
        self.model = model_info
        self.docker_client = docker.from_env()
        if inspector is None:
            pass
            #self.inspector = self.__auto_select_client()  # TODO: To Improve
        else:
            if isinstance(inspector, BaseModelInspector):
                self.inspector = inspector
            else:
                raise TypeError("The inspector should be an instance of class BaseModelInspector!")

    def pre_deploy(self, device='cuda'):
         serve(self.model.saved_path, device=device)

    def diagnose(self, batch_size: int = None, device='cuda', timeout=30) -> DynamicProfileResult:
        """Start diagnosing and profiling model.

        Args:
            batch_size (int): Batch size.
            device (str): Device name.
            timeout (float): Waiting for docker container timeout in second. Default timeout period is 30s.
        """
        # Check server status

        model_status = False
        retry_time = 0  # use binary exponential backoff algorithm
        tick = time.time()
        while time.time() - tick < timeout:
            if self.inspector.check_model_status():
                model_status = True
                break
            retry_time += 1
            # get backoff time in s
            backoff_time = random.randint(0, 2 ** retry_time - 1) * 1e-3
            time.sleep(backoff_time)

        if not model_status:  # raise an error as model is not served.
            raise ServiceException('Model not served!')

        if batch_size is not None:
            self.inspector.set_batch_size(batch_size)

        result = self.inspector.run_model(server_name=self.server_name, device=device)

        dpr = DynamicProfileResult(
            ip=get_ip(),
            device_id=result['device_id'],
            device_name=result['device_name'],
            batch=result['batch_size'],
            memory=ProfileMemory(
                total_memory=result['total_gpu_memory'],
                memory_usage=result['gpu_memory_used'],
                utilization=result['gpu_utilization'],
            ),
            latency=ProfileLatency(
                inference_latency=result['latency'],
            ),
            throughput=ProfileThroughput(inference_throughput=result['total_throughput']),
            create_time=result['completed_time'],
        )
        print(dpr)
        return dpr

    def __auto_select_client(self):
        # according to the serving engine, select the right testing client.
        # TODO: replace the input None data in each client with self-generated data.
        serving_engine = self.model.engine
        if serving_engine == Engine.NONE:
            raise Exception(
                'please choose a serving engine for the model')
            # TODO How can we deploy to all available platforms if we don't know the engine?

        kwargs = {'repeat_data': None, 'model_info': self.model, 'batch_num': DEFAULT_BATCH_NUM}
        if serving_engine == Engine.TFS:
            return CVTFSClient(**kwargs)
        elif serving_engine == Engine.TORCHSCRIPT:
            return CVTorchClient(**kwargs)
        elif serving_engine == Engine.ONNX:
            return CVONNXClient(**kwargs)
        elif serving_engine == Engine.TRT:
            return CVTRTClient(**kwargs)
        elif serving_engine == Engine.TVM:
            raise NotImplementedError
        elif serving_engine == Engine.CUSTOMIZED:
            raise Exception('please pass a custom client to the Profiler.__init__.')
        else:
            return None


