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
from time import time, sleep
from multiprocessing import Process, cpu_count, Pool
from abc import ABCMeta, abstractmethod


class BaseDataWrapper(metaclass=ABCMeta):
    """
    A class for pre-processing data that for inference, other model needs to implement this class
    and complete the data preprocessing method.
    """

    def __init__(self, model_info_url:str, raw_data:list, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = 1 # default: 1
        # The model information sent from the server url
        self.model_info = requests.get(model_info_url).json()
        # the raw data needs to be processed
        self.raw_data = raw_data
        # the processed data
        preprocess_start_time = time()
        self.processed_data = self.data_preprocess()
        preprocess_end_time = time()
        # the time we spent on the data preprocessing
        self.preprocess_time = preprocess_end_time - preprocess_start_time
        # The pre-batched set of images
        self.batches = self.__client_batch_request()

        # TODO: replace printing with logging
        print("DataWrapper - Data preprecess time: ", self.preprocess_time, '\n')

    @abstractmethod
    def data_preprocess(self):
        """
        Handle raw data, after preprocessing we can get the __test_data, which are used for benchmark testing.
        @param __raw_data: the raw data read by load_data method.
        """
        pass

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


class BaseModelInspector(metaclass=ABCMeta):
    """
    A class for running the model inference with metrics testing. User can 
    call the the method to run and test the model and return the tested 
    latency and throughput. 

    Parameters
    ----------
    @param data_wrapper: data wrapper, needed.
    @param asynchronous: runnning asynchronously, default is False.
    @param threads: number of threads while running the inference, default is cpu_count() * 24.
    @param sla: SLA, default is 1 sec.
    @param percentile: The SLA percentile. Default is 95.
    """
    def __init__(self, data_wrapper:BaseDataWrapper, threads=None, asynchronous=None, percentile=None,
                    sla=None):
        # check if the input data is an instance of BaseDataWrapper
        if isinstance(data_wrapper, BaseDataWrapper):
            self.data_wrapper = data_wrapper
        else:
            self.data_wrapper = None
            raise TypeError("The data wrapper should be an instance of class DataWrapper!")

        self.throughputs = []
        self.latencies = []

        if threads == None:
            self.threads = cpu_count() * 24
        else:
            self.threads = threads
    
        if asynchronous == None:
            self.asynchronous = False
        else:
            self.asynchronous = asynchronous

        self.pool = Pool(processes=self.threads)

        if percentile == None:
            self.percentile = 95
        else:
            self.percentile = percentile
            
        if sla == None:
            self.sla = 1.0
        else:
            self.sla = sla

    def run_model(self):
        # reset the results
        self.throughputs = []
        self.latencies = []

        # warm-up
        if len(self.data_wrapper.batches) > 10:
            warm_up_batches = self.data_wrapper.batches[:10]
            for batch in warm_up_batches:
                self.start_infer_with_time(batch)
        else:
            raise ValueError("Not enough test values, try to make more testing data.")

        pass_start_time = time()
        for batch in self.data_wrapper.batches:
            if self.asynchronous:
                self.pool.apply_async(self.start_infer_with_time, args=(batch,), callback=self.__inference_callback)
            else:
                a_batch_latency = self.start_infer_with_time(batch)
                self.latencies.append(a_batch_latency)
                a_batch_throughput =  self.data_wrapper.batch_size / a_batch_latency
                self.throughputs.append(a_batch_throughput)
                # TODO: replace printing with logging
                print(" latency: {:.4f}".format(a_batch_latency), 'sec', " throughput: {:.4f}".format(a_batch_throughput), ' req/sec')

        # TODO: Improve the asynchronous testing method, make sure the GPU can run in 100% and take about 75% throughput here.
        # FIXME: Fix asynchronous issue inside object, make sure all the requests have a response.
        # waitting until all the async requests have responses.
        while len(self.latencies) != len(self.data_wrapper.batches):
            pass

        pass_end_time = time()
        all_data_latency = pass_end_time - pass_start_time
        all_data_throughput = len(self.data_wrapper.processed_data) / (pass_end_time - pass_start_time)
        custom_percentile = np.percentile(self.latencies, self.percentile)

        self.print_results(all_data_throughput, all_data_latency, custom_percentile)
        # Remove processes from pool
        self.pool.close()
        self.pool.join()

    def __inference_callback(self, a_batch_latency):
        """
        A callback function which handles the results of a asynchronous inference request
        @param elapsed_time: The amount of required for the inference request to complete
        """
        self.latencies.append(a_batch_latency) 
        a_batch_throughput =  self.data_wrapper.batch_size / a_batch_latency
        self.throughputs.append(a_batch_throughput)

        # TODO: replace printing with logging
        print("a_batch_latency: {:.4f}".format(a_batch_latency), 'sec')
        print("a_batch_throughput: {:.4f}".format(a_batch_throughput), ' req/sec')

    def start_infer_with_time(self, batch_input):
        """
        Perform inference non-asynchronosly, and return the total time.
        """
        self.setup_inference()
        start_time = time()
        self.infer(batch_input)
        end_time = time()
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
    def print_results(self, throughput, latiency, custom_percentile):
        twenty_fifth_percentile = np.percentile(self.latencies, 25)
        fiftieth_percentile = np.percentile(self.latencies, 50)
        seventy_fifth_percentile = np.percentile(self.latencies, 75)

        print(f'total batches: {len(self.data_wrapper.batches)}')
        print(f'total latiency: {latiency} s')
        print(f'total throughput: {throughput} req/sec')
        print(f'25th-percentile latiency: {twenty_fifth_percentile} s')
        print(f'50th-percentile latiency: {fiftieth_percentile} s')
        print(f'75th-percentile latiency: {seventy_fifth_percentile} s')
        print(f'{self.percentile}th-percentile latiency: {custom_percentile} s')
        print(f'completed at {datetime.datetime.now()}')

    # A fix for multiprocessing inside class object, remove the 'self' from the states.
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)