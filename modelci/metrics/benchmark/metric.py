"""
Author: huangyz0918
Author: Li Yuanming
Dec: Abstract class for data pre-processing, implement this class and try to meet
    the data format that model requires here.
Date: 26/04/2020
"""

import datetime
import time
from abc import ABCMeta, abstractmethod
from threading import Thread

import GPUtil
import numpy as np

from modelci.metrics.cadvisor.cadvisor import CAdvisor
from modelci.utils.misc import get_device


class BaseModelInspector(metaclass=ABCMeta):
    """A class for running the model inference with metrics testing. User can
    call the the method to run and test the model and return the tested
    latency and throughput.

    Args:
        batch_num: the number of batches you want to run
        batch_size: batch size you want
        repeat_data: data unit to repeat.
        asynchronous: running asynchronously, default is False.
        sla: SLA, default is 1 sec.
        percentile: The SLA percentile. Default is 95.
    """

    def __init__(self, repeat_data, batch_num=1, batch_size=1, asynchronous=False, percentile=95, sla=1.0):
        self.throughput_list = []
        self.latencies = []

        self.asynchronous = asynchronous
        self.percentile = percentile
        self.sla = sla

        self.batch_num = batch_num
        self.batch_size = batch_size

        self.raw_data = repeat_data
        self.processed_data = self.data_preprocess(self.raw_data)

        self.batches = self.__client_batch_request()

    @abstractmethod
    def data_preprocess(self, x):
        """Handle raw data, after preprocessing we can get the processed_data, which is using for benchmarking."""
        return x

    def set_batch_size(self, new_bs):
        """update the batch size here.

        Args:
            new_bs: new batch size you want to use.
        """
        self.batch_size = new_bs
        self.batches = self.__client_batch_request()

    def __client_batch_request(self):
        """Batching input data according to the specific batch size."""
        batches = []
        for i in range(self.batch_num):
            batch = []
            for j in range(self.batch_size):
                batch.append(self.processed_data)
            batches.append(batch)
        return batches

    @abstractmethod
    def check_model_status(self) -> bool:
        """Check the loading status for model."""
        raise NotImplementedError('Method `check_model_status` not implemented.')

    def run_model(self, server_name: str, device: str):
        """Running the benchmarking for the specific model on the specific server.

        Args:
            server_name (str): the container's name of Docker that serves the Deep Learning model.
            device (str): Device name. E.g.: cpu, cuda, cuda:1.
        """
        # reset the results
        self.throughput_list = []
        self.latencies = []

        cuda, device_num = get_device(device)

        # warm-up
        if self.batch_num > 10:
            warm_up_batches = self.batches[:10]
            for batch in warm_up_batches:
                self.start_infer_with_time(batch)
        else:
            raise ValueError("Not enough test values, try to make more testing data.")

        pass_start_time = time.time()
        for batch in self.batches:
            if self.asynchronous:
                ReqThread(self.__inference_callback, self.start_infer_with_time, batch).start()
            else:
                a_batch_latency = self.start_infer_with_time(batch)
                self.latencies.append(a_batch_latency)
                a_batch_throughput = self.batch_size / a_batch_latency
                self.throughput_list.append(a_batch_throughput)
                # TODO: replace printing with logging
                print(f' latency: {a_batch_latency:.4f} sec throughput: {a_batch_throughput:.4f} req/sec')

        while len(self.latencies) != len(self.batches):
            pass

        pass_end_time = time.time()
        all_data_latency = pass_end_time - pass_start_time
        all_data_throughput = (self.batch_size * self.batch_num) / (pass_end_time - pass_start_time)
        custom_percentile = np.percentile(self.latencies, self.percentile)

        # init CAdvisor 
        # FIXME: if the number of batch is really small, the GPU utilization will get a smaller value.
        # Usually, in order to increase accuracy, we need to increase the testing number of batches, try to make sure
        # the testing program can run over 1 minutes.
        cadvisor = CAdvisor()
        SLEEP_TIME = 15
        time.sleep(SLEEP_TIME)
        all_information = cadvisor.request_by_name(server_name)
        model_info = cadvisor.get_model_info(all_information)
        stats = model_info[list(model_info.keys())[0]]['stats']
        val_stats = [x for x in stats[-int(SLEEP_TIME + all_data_latency):] if
                     x['accelerators'][0]['duty_cycle'] is not 0]
        all_batch_avg_memory_total = sum([i['accelerators'][0]['memory_total'] for i in val_stats]) / len(val_stats)
        all_batch_avg_memory_used = sum([i['accelerators'][0]['memory_used'] for i in val_stats]) / len(val_stats)
        all_batch_avg_util = sum([i['accelerators'][0]['duty_cycle'] for i in val_stats]) / len(val_stats)
        memory_avg_usage_per = all_batch_avg_memory_used / all_batch_avg_memory_total

        if cuda:
            gpu_device = GPUtil.getGPUs()[device_num]
            device_name = gpu_device.name
            device_id = gpu_device.uuid
        else:
            # TODO: Get CPU name
            device_name = 'N.A.'
            device_id = 'cpu'

        return self.print_results(
            device_name, device_id, all_data_throughput, all_data_latency, custom_percentile,
            all_batch_avg_memory_total,
            all_batch_avg_memory_used, all_batch_avg_util, memory_avg_usage_per
        )

    def __inference_callback(self, a_batch_latency):
        """A callback function which handles the results of a asynchronous inference request.

        Args:
            a_batch_latency: The amount of required for the inference request to complete.
        """
        self.latencies.append(a_batch_latency)
        a_batch_throughput = self.batch_size / a_batch_latency
        self.throughput_list.append(a_batch_throughput)
        # print(" latency: {:.4f}".format(a_batch_latency), 'sec', " throughput: {:.4f}".format(a_batch_throughput), ' req/sec')

    def start_infer_with_time(self, batch_input):
        """Perform inference non-asynchronously, and return the total time.

        Args:
            batch_input: The batch data in the request.
        """
        self.make_request(batch_input)
        start_time = time.time()
        self.infer(batch_input)
        end_time = time.time()
        return end_time - start_time

    def make_request(self, input_batch):
        """Function for sub-class to implement before inferring, to create the `self.request` can be
            overridden if needed.
        """
        pass

    @abstractmethod
    def infer(self, input_batch):
        """Abstract function for sub-class to implement the detailed infer function.

        Args:
            input_batch: The batch data in the request.
        """
        pass

    def dump_result(self, path=None):
        """Export the testing results to local JSON file.

        Args:
            path: The path to save the results.

        TODO:
            save result as a dict or something with logger
        """
        pass

    def print_results(
            self,
            device_name,
            device_id,
            throughput,
            latency,
            custom_percentile,
            all_batch_avg_memory_total,
            all_batch_avg_memory_used,
            all_batch_avg_util,
            memory_avg_usage_per
    ):
        """Export the testing results to local JSON file.

        Args:
            throughput: The tested overall throughput for all batches.
            latency: The tested latency for all batches.
            custom_percentile: The custom percentile you want to check for latencies.
            all_batch_avg_memory_total: The capacity memory usages for the inference container.
            all_batch_avg_memory_used: Used memory amount of this inference for all batches.
            all_batch_avg_util: The average GPU utilization of inferring all batches.
            memory_avg_usage_per: The GPU memory usage percentile.
            device_name: The serving device's name, for example, RTX 2080Ti.
        TODO:
            replace printing with saving code in mongodb, or logging.
        """
        percentile_50 = np.percentile(self.latencies, 50)
        percentile_95 = np.percentile(self.latencies, 95)
        percentile_99 = np.percentile(self.latencies, 99)
        complete_time = datetime.datetime.now()

        print('\n')
        print(f'testing device: {device_name}')
        print(f'total batches: {len(self.batches)}, batch_size: {self.batch_size}')
        print(f'total latency: {latency} s')
        print(f'total throughput: {throughput} req/sec')
        print(f'50th-percentile latency: {percentile_50} s')
        print(f'95th-percentile latency: {percentile_95} s')
        print(f'99th-percentile latency: {percentile_99} s')
        # print(f'{self.percentile}th-percentile latency: {custom_percentile} s')
        print(f'total GPU memory: {all_batch_avg_memory_total} bytes')
        print(f'average GPU memory usage percentage: {memory_avg_usage_per:.4f}')
        print(f'average GPU memory used: {all_batch_avg_memory_used} bytes')
        print(f'average GPU utilization: {all_batch_avg_util:.4f}%')
        print(f'completed at {complete_time}')

        return {
            'device_name': device_name,
            'device_id': device_id,
            'total_batches': len(self.batches),
            'batch_size': self.batch_size,
            'total_latency': latency,
            'total_throughput': throughput,
            'latency': [latency / len(self.batches), percentile_50, percentile_95, percentile_99],
            'total_gpu_memory': all_batch_avg_memory_total,
            'gpu_memory_percentage': memory_avg_usage_per,
            'gpu_memory_used': all_batch_avg_memory_used,
            'gpu_utilization': all_batch_avg_util / 100,
            'completed_time': complete_time,
        }


class ReqThread(Thread):
    """Thread class for sending a request."""

    def __init__(self, callback, infer_method, batch_data):
        Thread.__init__(self)
        self.callback = callback
        self.batch_data = batch_data
        self.infer = infer_method

    def run(self):
        self.infer(self.batch_data)
        self.callback(time.thread_time())
