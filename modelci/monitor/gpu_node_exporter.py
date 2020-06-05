import json
import requests
import re

from collections import defaultdict

DEFAULT_GPU_MONITOR_PORT = "http://localhost:9400/gpu/metrics"


class GPUNodeExporter(object):
    """A class for monitor GPU 
    """

    def __init__(self, service_url=None):
        """get gpu node exporter url
        Keyword Arguments:
            service_url {str} -- a url with port to export gpu info (default: {None})
        """
        if service_url == None:
            self.service_url = DEFAULT_GPU_MONITOR_PORT
        else:
            self.service_url = service_url

    def requst_all_gpu_info(self):
        """get all gpu info in a work
        Returns:
            dict -- {gpu_num: gpu_info}
        """
        info = requests.get(self.service_url)

        info_dict = defaultdict(list)

        info_list = self.__parse_text(info)

        for k, v in info_list:
            info_dict[k].append(v)

        # print("There {} GPUs in this worker.".format((len(dict(info_dict))))) # TODO replace with logging
        return dict(info_dict)

    def __parse_text(self, text_info):
        """format original text info from node exporter
        Arguments:
            text_info {str} -- information from node exporter
        Returns:
            defaultdict -- {gpu_num: list}
        """
        info_list = []
        for i in text_info.text.split("\n")[:-1]:
            if i[0] != '#':
                sample = i.replace('{', ',').replace('} ', ',').split(',')
                info_list.append((re.sub("[^0-9]", "", sample[1]), {sample[0]: sample[-1]}))

        return info_list

    def get_idle_gpu(self, util_level=0, memeory_level=0):
        """get idle gpu ids
        Keyword Arguments:
            util_level {int} -- if the GPU utilization larger than the value, the GPU is busy (default: {0})
            memeory_level {int} -- same with above (default: {0})
        Returns:
            list -- [gpu_id, ...]
        """
        info_dict = self.requst_all_gpu_info()
        idle_gpus = []
        for k, v in info_dict.items():
            # for temp in v:
            #     print(list(temp.values()))
            gpu_stat = {list(temp.keys())[0]: float(list(temp.values())[0]) for temp in v}
            total_memory = gpu_stat['dcgm_fb_used'] + gpu_stat['dcgm_fb_free']
            if gpu_stat['dcgm_gpu_utilization'] > util_level or gpu_stat['dcgm_fb_used'] / total_memory > memeory_level:
                pass
            else:
                idle_gpus.append(k)
        return idle_gpus

    def get_single_gpu_usage(self, gpu_num=1):
        raise NotImplementedError

    def get_gpu_type(self):
        raise NotImplementedError
