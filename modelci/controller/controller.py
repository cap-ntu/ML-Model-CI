from modelci.hub.profiler import Profiler
from modelci.monitor.gpu_node_exporter import GPUNodeExporter

def profiler_callback(model_info, profiler=Profiler, device_util_thd=0, device_memory_thd=0, period=60):

    # After conversion, we will get a model object that contains mode info
    # use the model info to init a client and a profiler
    profiler = profiler(model_info=model_info)

    # Here we assume that each worker server only contains one kind of GPU

    # Use the monitor to get devices names (https://github.com/anderskm/gputil), IDs and utilization periodically.
    # idle_gpu = {'name':, 'id':, 'computation_util':, 'memory_util':,}
    gpu_collector = GPUNodeExporter()
    idle_gpu = gpu_collector()

    # if the utilization of any of them is lower than device_util_thd=0, device_memory_thd=0
    # 0）stop the period function
    # 1) pick one to profile 
    # 2) record the device name and profiling results to database

    profiler.auto_diagnose(idle_gpu['id'])

def auto_device_placement():
    raise NotImplementedError('Method `auto_device_placement` is not implemented.')


   
    

    
    















