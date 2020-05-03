"""
Author: huangyz0918
Dec: prolifing models.
Date: 03/05/2020
"""

from modelci.metrics.benchmark.metric import BaseDataWrapper, BaseModelInspector

class Diagnoser(object):
    """
    Diagnoser class, call this to test model performance.
    """

    def __init__(self, inspector:BaseModelInspector, server_name):
        """
        Init a diagnoser object

        @param inspector:BaseModelInspector
        """
        self.inspector = inspector
        self.server_name

    def diagnose(self):
        """
        start diagnosing and profiling model.

        """
        inspector.run_model(server_name) 
