#  Copyright (c) NTU_CAP 2021. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import unittest
import torch
import numpy 

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata

from pathlib import Path
from PIL import Image

import sys
sys.path.append('../')

# from modelci.types.bo import IOShape

class TestTVMConverter(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50')
        cls.img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
        cls.img_path = download_testdata(cls.img_url, "cat.png", module="data")
        cls.input_name = 'input_0'
        cls.input_shape = [1, 3, 224, 224]
        cls.opt_level = 3
        cls.test_path = Path("test_tvm")
        cls.processing='cpu'
        cls.override = False
        cls.dtype = "float32"

    def test_to_tvm1(self):
        """
        Convert a PyTorch nn.Module into TVM.
        """
        if self.test_path.with_suffix('.zip').exists():
            if not self.override:  # file exists yet override flag is not set
                print('Use cached model')
                return True

        model = self.model.eval()
        try:
            input_data = torch.randn(self.input_shape)
            traced_model = torch.jit.trace(model, input_data).eval()
            
            # image for testing 
            img = Image.open(self.img_path).resize((224, 224))

            # Preprocess the image and convert to tensor
            from torchvision import transforms

            my_preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            
            img = my_preprocess(img)
            img = img.numpy()
            img = numpy.expand_dims(img, 0)
            
            # Import graph to relay
            shape_list = [(self.input_name, img.shape)]

            # Facing an error while using IOShape
            # shape_list = [IOShape(shape=img.shape, dtype=self.dtype, name=self.input_name)]
            model, params = relay.frontend.from_pytorch(traced_model, shape_list)
                
            # Relay build
            target = "llvm"
            target_host = "llvm"

            if self.processing=='cpu':
                ctx = tvm.cpu(0)    
                    
            # TODO only provides CPU as of now, future work: GPU
            # else if processing=='gpu':
            #     ctx=tvm.gpu(0)

            with tvm.transform.PassContext(opt_level=self.opt_level):
                lib = relay.build(model, target=target, target_host=target_host, params=params)

            print('TVM format converted successfully')
            return True

        except:
            #TODO catch different types of error
            print("This model is not supported as TVM format")
            return False

if __name__ == '__main__':
    unittest.main()
