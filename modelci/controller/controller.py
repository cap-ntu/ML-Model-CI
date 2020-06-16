from modelci.hub.profiler import Profiler
from modelci.monitor.gpu_node_exporter import GPUNodeExporter

from threading import Thread
import time


class UtilMonitor(Thread):
    """
    Monitor for GPU Utilization
    """

    def __init__(self, profiler: Profiler, delay, util_level, memory_level):
        """
        Init the GPU Utilization Monitor Thread.

        :param delay: time period to get the information.
        :param util_level: The utilization level that trigger profiling.
        :param memory_level: The memory usage level that trigger profiling.
        :param profiler: The instance of model profiler.
        """
        super(UtilMonitor, self).__init__()
        self.stopped = False
        self.delay = delay
        self.start()
        self.available_device = []
        self.exporter = GPUNodeExporter()
        self.memory_level = memory_level
        self.util_level = util_level
        self.profiler = profiler

        if self.exporter is None:
            raise TypeError(
                'Failed to get GPU relative information from node exporter, please make sure the service has started.')

    def run(self):
        while not self.stopped:
            self.available_device = self.exporter.get_idle_gpu(util_level=self.util_level,
                                                               memory_level=self.memory_level)

            if not self.available_device:
                profiler.auto_diagnose(available_devices=self.available_device,
                                       batch_list=[8])  # default profiling batch size 8
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


def auto_model_profiling(model_info, server_name, device_util_thd=0.01, device_memory_thd=0.01, period=10):
    """
    Start model profiling automatically.

    :param model_info: model information object saved by ModelCI.
    :param server_name: serving docker container's name.
    :param device_util_thd: The utilization level that trigger profiling.
    :param device_memory_thd: The memory usage level that trigger profiling.
    :param period: time period to get the information.
    :return: None
    """
    profiler = Profiler(model_info=model_info, server_name=server_name)
    monitor = UtilMonitor(profiler, period, device_util_thd, device_memory_thd)
    monitor.start()


def auto_device_placement():
    raise NotImplementedError('Method `auto_device_placement` is not implemented.')
