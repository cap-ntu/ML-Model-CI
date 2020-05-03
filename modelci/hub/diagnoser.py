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
    def __init__(self, inspector:BaseModelInspector, server_name):
        """
        Init a diagnoser object

        @param wrapper:BaseDataWrapper
        @param inspector:BaseModelInspector
        """
        self.inspector = inspector
        self.server_name = server_name

        self.model_info = None
        self.model_path = None

    def diagnose(self):
        """
        start diagnosing and profiling model.
        TODO: auto select a free GPU device to test, make sure before testing, the GPU utilization is 0%.
        """
        self.inspector.run_model(self.server_name) 

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


