"""
Author: huangyz0918
Dec: Abstract class for data preprocessing, implement this class and try to meet
    the data format that model requires here.
Date: 26/04/2020
"""

import csv
import datetime
import requests
import numpy as np
import time
from dateutil import parser
from threading import Thread
from abc import ABCMeta, abstractmethod

from modelci.metrics.cadvisor.cadvisor import CAdvisor


class BaseModelInspector(metaclass=ABCMeta):
    """
    A class for running the model inference with metrics testing. User can 
    call the the method to run and test the model and return the tested 
    latency and throughput. 

    Parameters
    ----------
    @param: data wrapper, needed.
    @param asynchronous: runnning asynchronously, default is False.
    @param threads: number of threads while running the inference, default is cpu_count() * 24.
    @param sla: SLA, default is 1 sec.
    @param percentile: The SLA percentile. Default is 95.
    """
    def __init__(self, raw_data:list, batch_size=1, asynchronous=False, percentile=95, sla=1.0):
        self.throughputs = []
        self.latencies = []
    
        self.asynchronous = asynchronous
        self.percentile = percentile
        self.sla = sla
        self.batch_size = batch_size
        self.preprocess_time = self.start_preprocess()
        self.processed_data = raw_data
        self.batches = self.__client_batch_request()

        # TODO: replace printing with logging
        print(" - Data preprecess time: ", self.preprocess_time, '\n')

    def start_preprocess(self):
        start_time = time.time()
        self.data_preprocess()
        return time.time() - start_time

    @abstractmethod
    def data_preprocess(self):
        """
        Handle raw data, after preprocessing we can get the __test_data, which are used for benchmark testing.
        @param __raw_data: the raw data read by load_data method.
        """
        pass

    def set_batch_size(self, new_bs):
        """
        update the batch size here.
        """
        self.batch_size = new_bs
        self.batches = self.__client_batch_request()

    def __client_batch_request(self):
        """
        Batch all of the processed images before performing inference.
        """
        batches = []
        data_item_id = 0
        last_request = False
        while not last_request:
            input_batch = []
            for idx in range(self.batch_size):
                input_batch.append(self.processed_data[data_item_id])
                data_item_id = (data_item_id + 1) % len(self.processed_data)
                if data_item_id == 0:
                    last_request = True
            batches.append(input_batch)
        return batches

    def run_model(self, server_name):
        # reset the results
        self.throughputs = []
        self.latencies = []

        # warm-up
        if len(self.batches) > 10:
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
                a_batch_throughput =  self.batch_size / a_batch_latency
                self.throughputs.append(a_batch_throughput)
                # TODO: replace printing with logging
                print(" latency: {:.4f}".format(a_batch_latency), 'sec', " throughput: {:.4f}".format(a_batch_throughput), ' req/sec')

        while len(self.latencies) != len(self.batches):
            pass

        pass_end_time = time.time()
        all_data_latency = pass_end_time - pass_start_time
        all_data_throughput = len(self.processed_data) / (pass_end_time - pass_start_time)
        custom_percentile = np.percentile(self.latencies, self.percentile)

        # init CAdvisor
        cadvisor = CAdvisor()
        all_information_tfs = cadvisor.request_by_name(server_name)
        model_info_tfs = cadvisor.get_model_info(all_information_tfs)
        stats_tfs = model_info_tfs[list(model_info_tfs.keys())[0]]['stats']

        val_stats = [x for x in stats_tfs[-int(all_data_latency):] if x['accelerators'][0]['duty_cycle'] is not 0]
        all_batch_avg_memory_total = sum([i['accelerators'][0]['memory_total'] for i in val_stats]) / len(val_stats)
        all_batch_avg_memory_used = sum([i['accelerators'][0]['memory_used'] for i in val_stats]) / len(val_stats)
        all_batch_avg_util = sum([i['accelerators'][0]['duty_cycle'] for i in val_stats]) / len(val_stats)
        memory_avg_usage_per = all_batch_avg_memory_used / all_batch_avg_memory_total

        self.print_results(all_data_throughput, all_data_latency, custom_percentile, all_batch_avg_memory_total,
                           all_batch_avg_memory_used, all_batch_avg_util, memory_avg_usage_per)

    def __inference_callback(self, a_batch_latency):
        """
        A callback function which handles the results of a asynchronous inference request
        @param elapsed_time: The amount of required for the inference request to complete
        """
        self.latencies.append(a_batch_latency) 
        a_batch_throughput =  self.batch_size / a_batch_latency
        self.throughputs.append(a_batch_throughput)

        # TODO: replace printing with logging
        # print("a_batch_latency: {:.4f}".format(a_batch_latency), 'sec')
        # print("a_batch_throughput: {:.4f}".format(a_batch_throughput), ' req/sec')

    def start_infer_with_time(self, batch_input):
        """
        Perform inference non-asynchronosly, and return the total time.
        """
        self.setup_inference()
        start_time = time.time()
        self.infer(batch_input)
        end_time = time.time()
        return end_time - start_time

    def setup_inference(self):
        """
        function for sub-class to implement before infering,
        can be override if needed.
        """
        pass

    @abstractmethod
    def server_batch_request(self):
        '''
        sahould batch the data from server side, leave blank if don't.
        '''
        pass

    @abstractmethod
    def infer(self, input_batch):
        """
        Abstract function for sub-class to implement the detailed infer function.
        """
        pass

    # TODO: save result as a dict or something with logger
    def dump_result(self, path=None):
        pass

    # TODO: replace printing with logging
    def print_results(self, throughput, latiency, custom_percentile, all_batch_avg_memory_total, 
                        all_batch_avg_memory_used, all_batch_avg_util, memory_avg_usage_per):
        percentile_50 = np.percentile(self.latencies, 50)
        percentile_95 = np.percentile(self.latencies, 95)
        percentile_99 = np.percentile(self.latencies, 99)

        print(f'total batches: {len(self.batches)}')
        print(f'total latiency: {latiency} s')
        print(f'total throughput: {throughput} req/sec')
        print(f'50th-percentile latiency: {percentile_50} s')
        print(f'95th-percentile latiency: {percentile_95} s')
        print(f'99th-percentile latiency: {percentile_99} s')
        # print(f'{self.percentile}th-percentile latiency: {custom_percentile} s')
        print(f'total GPU memory: {all_batch_avg_memory_total} bytes')
        print('average GPU memory usage percentile: {:.4f}'.format(memory_avg_usage_per))
        print(f'average GPU memory used: {all_batch_avg_memory_used} bytes')
        print('average GPU utilization: {:.4f}%'.format(all_batch_avg_util))
        print(f'completed at {datetime.datetime.now()}')


class ReqThread(Thread):
    """
    Thread class for sending a request.
    """
    def __init__(self, callback, infer_mothod, batch_data):
        Thread.__init__(self)
        self.callback = callback
        self.batch_data = batch_data
        self.infer = infer_mothod
        
    def run(self):
        start_time = time.thread_time()
        self.infer(self.batch_data)
        self.callback(time.thread_time())