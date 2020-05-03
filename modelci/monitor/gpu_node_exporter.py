import json
import requests

DEFAULT_GPU_MONITOR_PORT = "http://localhost:9400/gpu/metrics"


class GPUNodeExporter(object):

    def __init__(self, service_url=None):
        if service_url == None:
            self.service_url = DEFAULT_GPU_MONITOR_PORT
        else:
            self.service_url = service_url

    def requst_all_gpu_info(self):
        info = requests.get(self.service_url)
        print(info.json())





if __name__ == '__main__':
    a = GPUNodeExporter()
    a.requst_all_gpu_info()

