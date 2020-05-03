"""
Author: huangyz0918
Dec: prolifing models.
Date: 03/05/2020
"""

from modelci.hub.manager import retrieve_model_by_name, retrieve_model_by_task
from modelci.persistence.bo import Framework, Engine
from modelci.metrics.benchmark.metric import BaseModelInspector


class Diagnoser(object):
    """
    Diagnoser class, call this to test model performance.
    """
    def __init__(self, inspector=None, server_name=None, model_info=None, model_path=None):
        """
        Init a diagnoser object

        @param wrapper:BaseDataWrapper
        @param inspector:BaseModelInspector
        """
        if isinstance(inspector, BaseModelInspector):
            self.inspector = inspector
        else:
            self.inspector = None
            raise TypeError("The inspector should be an instance of class BaseModelInspector!")

        self.server_name = server_name
        self.model_info = model_info
        self.model_path = model_path

    def diagnose(self, batch_size=None):
        """
        start diagnosing and profiling model.
        """
        if batch_size is not None:
            self.inspector.set_batch_size(batch_size)

        self.inspector.run_model(self.server_name) 

    def diagnose_all_batches(self):
        """
        run model tests in batch size = 1, 2, 4, 8, 16, 32, 64, 128
        """
        for size in [1, 2, 4, 8, 16, 32, 64, 128]:
            self.inspector.set_batch_size(size)
            self.inspector.run_model(self.server_name) 

    def __deploy_model(self, model_info):
        """
        deploy model here.
        TODO: auto select a free GPU device to test, make sure before testing, the GPU utilization is 0%.
        """
        if self.check_device_status():
            pass

    def check_device_status(self):
        """
        To check the server is good to deploy the model now.
        """
        #TODO: using monitor to see the result.
        return Ture

    def init_model_info(self, architecture_name, framework, engine):
        """
        init the model information before testing, should be called before calling diagnose.
        By model name and optionally filtered by model framework and(or) model engine
        """
        self.model_path, self.model_info = retrieve_model_by_name(architecture_name=architecture_name, 
                                                    framework=framework, engine=engine)

    def init_model_info(self, task):
        """
        init the model information before testing, should be called before calling diagnose
        By model task.
        """
        self.model_path, self.model_info = retrieve_model_by_task(task=task)


