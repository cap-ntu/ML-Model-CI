import collections
import time
from threading import Thread

import GPUtil

from modelci.hub import Profiler
from modelci.monitor import GPUNodeExporter


class UtilMonitor(Thread):
    """
    Monitor for GPU Utilization
    """

    def __init__(self, device, profiler: Profiler, delay, util_level, memory_level):
        """
        Init the GPU Utilization Monitor Thread.

        :param delay: time period to get the information.
        :param util_level: The utilization level that trigger profiling.
        :param memory_level: The memory usage level that trigger profiling.
        :param profiler: The instance of model profiler.
        :param device: GPU device to test.
        """
        super(UtilMonitor, self).__init__()
        self.stopped = False
        self.delay = delay
        self.memory_level = memory_level
        self.util_level = util_level
        self.profiler = profiler
        self.exporter = GPUNodeExporter()
        self.device = device

        if self.exporter is None:
            raise TypeError(
                'Failed to get GPU relative information from node exporter, please make sure the service has started.')

    def run(self):
        while not self.stopped:
            available_devices = self.exporter.get_idle_gpu(util_level=self.util_level,
                                                           memory_level=self.memory_level)

            if self.device.id in available_devices:
                self.profiler.auto_diagnose(available_devices=[self.device])

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

    different_kind_devices = collections.OrderedDict()
    for gpu in GPUtil.getGPUs():
        if gpu.name not in different_kind_devices:
            different_kind_devices[gpu.name] = gpu

    for device in list(different_kind_devices.values()):
        profiler = Profiler(model_info=model_info, server_name=server_name)
        monitor = UtilMonitor(device, profiler, period, device_util_thd, device_memory_thd)
        monitor.start()


def auto_device_placement():
    raise NotImplementedError('Method `auto_device_placement` is not implemented.')
