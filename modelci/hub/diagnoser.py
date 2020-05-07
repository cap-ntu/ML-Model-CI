"""
Author: huangyz0918
Dec: prolifing models.
Date: 03/05/2020
"""
import docker 

from modelci.persistence.bo import IOShape, Engine, Framework, ModelVersion
from modelci.metrics.benchmark.metric import BaseModelInspector
from modelci.monitor.gpu_node_exporter import GPUNodeExporter


class Diagnoser(object):
    """
    Diagnoser class, call this to test model performance.
    """
    def __init__(self, model_info, server_name, inspector:BaseModelInspector=None):
        """
        Init a diagnoser object

        @param inspector: client implemented from BaseModelInspector
        @param server_name: serving platform's docker conatiner's name.
        @param model_info: information about the model, can get from init_model_info method.
        """
        if inspector is None:
            self.inspector = self.__auto_select_client() #TODO
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


    def diagnose_all_batches(self, arr:list):
        """
        run model tests in batch size from list input.
        """
        for size in arr:
            self.inspector.set_batch_size(size)
            self.inspector.run_model(self.server_name)


    def auto_diagnose(self, batch_list:list=None):
        """
        select the free machine and deploy automatically to test the model using availibe platforms.
        """
        self.__deploy_model() # deploy the model automatically.
        for device in self.available_devices:
            print(f'deploying model in device: {device} ...')

            # deploy and start serving
            self.__deploy_model()
            try: # to check the container has started successfully or not.
                self.docker_client.containers.get(self.server_name)
            except Exception:
                print('\n ModelCI Error: starting the serving engine failed, please start the Docker container manually. \n')

            # start testing.
            if batch_list is not None:
                self.diagnose_all_batches(batch_list)
            else:
                self.diagnose()

        self.__stop_testing_container()
        print('finished.')

    def __deploy_model(self):
        """
        deploy model here. # TODO
        """
        saved_path = self.model_info.saved_path
        model_id = self.model_info.id
        model_name = self.model_info.name
        model_framework = self.model_info.framework
        serving_engine = self.model_info.engine
        exporter = GPUNodeExporter()
        self.available_devices = exporter.get_idle_gpu(util_level=0.01, memeory_level=0.01)

        print('available GPU devices: ', self.available_devices)
        print('model saved path: ', saved_path)
        print('model id: ', model_id)
        print('mode name: ', model_name)
        print('model framework: ', model_framework)
        print('model serving engine: ', serving_engine)


    def __auto_select_client(self):
        # TODO according to the serving engine, select the right testing client.
        # print(self.model_info)
        return self.inspector


    def __stop_testing_container(self):
        """
        After testing, we should release the resources.
        """
        running_container = self.docker_client.containers.get(self.server_name)
        running_container.stop()
        print("successfully stop serving container: ", self.server_name)