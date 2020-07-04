"""
Author: huangyz0918
Author: Li Yuanming
Dec: profiling models.
Date: 03/05/2020
"""
import os
import random
import time

import docker

from modelci.hub.client.onnx_client import CVONNXClient
from modelci.hub.client.tfs_client import CVTFSClient
from modelci.hub.client.torch_client import CVTorchClient
from modelci.hub.client.trt_client import CVTRTClient
from modelci.metrics.benchmark.metric import BaseModelInspector
from modelci.persistence.exceptions import ServiceException
from modelci.types.bo import (
    Framework,
    DynamicProfileResultBO,
    ProfileMemory,
    ProfileLatency,
    ProfileThroughput,
    ModelBO
)
from modelci.utils.misc import get_ip

DEFAULT_BATCH_NUM = 100

random.seed(ord(os.urandom(1)))


class Profiler(object):
    """Profiler class, call this to test model performance.

    Args:
        model_info (ModelBO): Information about the model, can get from `retrieve_model` method.
        server_name (str): Serving platform's docker container's name.
        inspector (BaseModelInspector): The client instance implemented from :class:`BaseModelInspector`.
    """

    def __init__(self, model_info: ModelBO, server_name: str, inspector: BaseModelInspector = None):
        """Init a profiler object."""
        if inspector is None:
            self.inspector = self.__auto_select_client()  # TODO: To Improve
        else:
            if isinstance(inspector, BaseModelInspector):
                self.inspector = inspector
            else:
                raise TypeError("The inspector should be an instance of class BaseModelInspector!")

        self.server_name = server_name
        self.model_bo = model_info
        self.docker_client = docker.from_env()

    def diagnose(self, batch_size: int = None, device='cuda', timeout=30) -> DynamicProfileResultBO:
        """Start diagnosing and profiling model.

        Args:
            batch_size (int): Batch size.
            device (str): Device name.
            timeout (float): Waiting for docker container timeout in second. Default timeout period is 30s.
        """
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

        dpr_bo = DynamicProfileResultBO(
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

        return dpr_bo

    def diagnose_all_batches(self, device_name, arr: list):
        """Run model tests in batch size from list input."""
        result = dict()
        for size in arr:
            result[size] = self.diagnose(device_name, size)
        return result

    def auto_diagnose(self, available_devices, batch_list: list = None):
        """Select the free machine and deploy automatically to test the model using available platforms."""
        from modelci.hub.deployer.dispatcher import serve

        saved_path = self.model_bo.saved_path
        model_id = self.model_bo.id
        model_name = self.model_bo.name
        model_framework = self.model_bo.framework
        serving_engine = self.model_bo.engine

        # for testing
        print('\n available GPU devices: ', [device.name for device in available_devices])
        print('model saved path: ', saved_path)
        print('model id: ', model_id)
        print('mode name: ', model_name)
        print('model framework: ', model_framework)
        print('model serving engine: ', serving_engine)

        for device in available_devices:  # deploy the model automatically in all available devices.
            print(f'deploying model in device: {device.id} ...')

            serve(save_path=saved_path, device=f'cuda:{device.id}', name=self.server_name)

            try:  # to check the container has started successfully or not.
                self.docker_client.containers.get(self.server_name)
            except Exception:
                print(
                    '\n'
                    'ModelCI Error: starting the serving engine failed, please start the Docker container manually. \n'
                )

            # start testing.
            if batch_list is not None:
                self.diagnose_all_batches(device.name, batch_list)
            else:
                self.diagnose(device.name)

        self.__stop_testing_container()
        print('finished.')

    def __auto_select_client(self):
        # according to the serving engine, select the right testing client.
        # TODO: replace the input None data in each client with self-generated data.
        serving_engine = self.model_bo.engine
        if serving_engine == Framework.NONE:
            raise Exception(
                'please choose a serving engine for the model')
            # TODO How can we deploy to all available platforms if we don't know the engine?

        kwargs = {'repeat_data': None, 'model_info': self.model_bo, 'batch_num': DEFAULT_BATCH_NUM}
        if serving_engine == Framework.TFS:
            return CVTFSClient(**kwargs)
        elif serving_engine == Framework.TORCHSCRIPT:
            return CVTorchClient(**kwargs)
        elif serving_engine == Framework.ONNX:
            return CVONNXClient(**kwargs)
        elif serving_engine == Framework.TRT:
            return CVTRTClient(**kwargs)
        elif serving_engine == Framework.TVM:
            raise NotImplementedError
        elif serving_engine == Framework.CUSTOMIZED:
            raise Exception('please pass a custom client to the Profiler.__init__.')
        else:
            return None

    def __stop_testing_container(self):
        """After testing, we should release the resources."""
        running_container = self.docker_client.containers.get(self.server_name)
        running_container.stop()
        print("successfully stop serving container: ", self.server_name)
