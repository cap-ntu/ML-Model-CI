from pathlib import Path
from modelci.hub.manager import retrieve_model, register_model_from_yaml
from modelci.hub.converter import convert

import torch
import torchvision.models as models
import numpy as np


# Path(f"{str(Path.home())}/.modelci/ResNet50/pytorch-torchscript/image_classification").mkdir(parents=True, exist_ok=True)
# saving_path = Path(f"{str(Path.home())}/.modelci/ResNet50/pytorch-torchscript/image_classification/1.zip")

# origin_model = retrieve_model()[0]
# real_model = models.resnet50(pretrained=False)
# real_model.load_state_dict(torch.load(str(origin_model.saved_path)))
# convert(real_model, 'pytorch', 'torchscript', save_path=saving_path)

# register_model_from_yaml('./example/resnet50.yml')
from modelci.hub.client.torch_client import CVTorchClient
from modelci.hub.profiler import Profiler
from modelci.hub.deployer.dispatcher import serve
test_data_item = np.zeros((16,16,3), dtype=np.uint8)
batch_num = 100
batch_size = 32

# Obtain model info from `retrieve_models` API.
model_info = retrieve_model()[0]

# create a client
torch_client = CVTorchClient(test_data_item, model_info, batch_num, batch_size, asynchronous=False)

# init the profiler
# container = serve(model_info.saved_path, name = 'emm')

profiler = Profiler(model_info=model_info, server_name="pytorch-serving:latest", inspector=torch_client)

# start profiling model
profiler.diagnose(device='cpu')