#!/usr/bin/python3
# -*- coding: utf-8 -*-
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

from pathlib import Path
from typing import Iterable, List, Optional, Callable

import tvm
import numpy 
import torch
import torch.jit
from torchvision import transforms
from modelci.utils import Logger

from tvm import relay
from tvm.contrib.download import download_testdata
from modelci.types.bo import IOShape

from pathlib import Path
from PIL import Image

logger = Logger('converter', welcome=False)


class TVMConverter(object):
    """Convert model to TVM format."""

    supported_framework = ["pytorch"]

    @staticmethod
    def from_pytorch(model: torch.nn.Module, save_path: Path, img_path, input_shape, input_name, processing, opt_level, override: bool = False):
        """Convert a PyTorch nn.Module into TorchScript.
        """
        if save_path.with_suffix('.zip').exists():
            if not override:  # file exists yet override flag is not set
                print('Use cached model')
                return True
        model.eval()
        try:
            input_data = torch.randn(input_shape)
            traced_model = torch.jit.trace(model, input_data).eval()
            
            # image for testing 
            img = Image.open(img_path).resize((224, 224))

            # Preprocess the image and convert to tensor

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
            shape_list = [(input_name, img.shape)]

            # Facing an import error while importing modelci.types.bo
            #  shape_list = [IOShape(shape=img.shape, dtype=self.dtype, name=self.input_name)]
            model, params = relay.frontend.from_pytorch(traced_model, shape_list)
                
            # Relay build
            target = "llvm"
            target_host = "llvm"

            if processing=='cpu':
                ctx = tvm.cpu(0)    
                    
            # TODO only provides CPU as of now, future work: GPU
            # else if processing=='gpu':
            #     ctx=tvm.gpu(0)

            with tvm.transform.PassContext(opt_level=opt_level):
                lib = relay.build(model, target=target, target_host=target_host, params=params)

            print('TVM format converted successfully')
            return True

        except:
            #TODO catch different types of error
            print("This model is not supported as TVM format")
            return False
