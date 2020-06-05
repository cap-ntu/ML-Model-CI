"""
Author: huangyz0918
Dec: prolifing models.
Date: 03/05/2020
"""
import docker

from modelci.hub.client.tfs_client import CVTFSClient
from modelci.hub.client.trt_client import CVTRTClient
from modelci.hub.client.torch_client import CVTorchClient
from modelci.hub.client.onnx_client import CVONNXClient

from modelci.persistence.bo import Framework
from modelci.metrics.benchmark.metric import BaseModelInspector
from modelci.monitor.gpu_node_exporter import GPUNodeExporter
from modelci.hub.deployer import serve

DEFAULT_BATCH_NUM = 100


class Profiler(object):
    """
    Profiler class, call this to test model performance.
    """

    def __init__(self, model_info, server_name, inspector: BaseModelInspector = None):
        """
        Init a profiler object

        @param inspector: client implemented from BaseModelInspector
        @param server_name: serving platform's docker conatiner's name.
        @param model_info: information about the model, can get from init_model_info method.
        """
        if inspector is None:
            self.inspector = self.__auto_select_client()  # TODO: To Improve
        else:
            if isinstance(inspector, BaseModelInspector):
                self.inspector = inspector
            else:
                raise TypeError("The inspector should be an instance of class BaseModelInspector!")

        self.server_name = server_name
        self.model_info = model_info
        self.docker_client = docker.from_env()
        self.available_devices = []

    def diagnose(self, batch_size=None):
        """
        start diagnosing and profiling model.
        """
        if batch_size is not None:
            self.inspector.set_batch_size(batch_size)

        self.inspector.run_model(self.server_name)

    def diagnose_all_batches(self, arr: list):
        """
        run model tests in batch size from list input.
        """
        for size in arr:
            self.diagnose(size)

    def auto_diagnose(self, batch_list: list = None):
        """
        select the free machine and deploy automatically to test the model using availibe platforms.
        """
        saved_path = self.model_info.saved_path
        model_id = self.model_info.id
        model_name = self.model_info.name
        model_framework = self.model_info.framework
        serving_engine = self.model_info.engine
        exporter = GPUNodeExporter()
        self.available_devices = exporter.get_idle_gpu(util_level=0.01, memeory_level=0.01)

        # for testing
        print('\n available GPU devices: ', self.available_devices)
        print('model saved path: ', saved_path)
        print('model id: ', model_id)
        print('mode name: ', model_name)
        print('model framework: ', model_framework)
        print('model serving engine: ', serving_engine)

        for device in self.available_devices:  # deploy the model automatically in all available devices.
            print(f'deploying model in device: {device} ...')

            serve(save_path=saved_path, device=f'cuda:{device}', name=self.server_name)

            try:  # to check the container has started successfully or not.
                self.docker_client.containers.get(self.server_name)
            except Exception:
                print(
                    '\n ModelCI Error: starting the serving engine failed, please start the Docker container manually. \n')

            # start testing.
            if batch_list is not None:
                self.diagnose_all_batches(batch_list)
            else:
                self.diagnose()

        self.__stop_testing_container()
        print('finished.')

    def __auto_select_client(self):
        # according to the serving engine, select the right testing client.
        # TODO: replace the input None data in each client with self-generated data.
        serving_engine = self.model_info.engine
        if serving_engine == Framework.NONE:
            raise Exception(
                'please choose a serving engine for the model')  # TODO How can we deploy to all available platforms if we don't know the engine?
        elif serving_engine == Framework.TFS:
            return CVTFSClient(None, batch_num=DEFAULT_BATCH_NUM, asynchronous=False)
        elif serving_engine == Framework.TORCHSCRIPT:
            return CVTorchClient(None, batch_num=DEFAULT_BATCH_NUM, asynchronous=False)
        elif serving_engine == Framework.ONNX:
            return CVONNXClient(None, batch_num=DEFAULT_BATCH_NUM, asynchronous=False)
        elif serving_engine == Framework.TRT:
            return CVTRTClient(None, batch_num=DEFAULT_BATCH_NUM, asynchronous=False)
        elif serving_engine == Framework.TVM:
            raise NotImplementedError
        elif serving_engine == Framework.CUSTOMIZED:
            raise Exception('please pass a custom client to the Profiler.__init__.')
        else:
            return None

    def __stop_testing_container(self):
        """
        After testing, we should release the resources.
        """
        running_container = self.docker_client.containers.get(self.server_name)
        running_container.stop()
        print("successfully stop serving container: ", self.server_name)
