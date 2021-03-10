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

import torch
from pathlib import Path
from modelci.utils import Logger

logger = Logger('converter', welcome=False)


class TorchScriptConverter(object):
    supported_framework = ["pytorch"]

    @staticmethod
    def from_pytorch(model: torch.nn.Module, save_path: Path, override: bool = False):
        """Convert a PyTorch nn.Module into TorchScript.
        """
        if save_path.with_suffix('.zip').exists():
            if not override:  # file exist yet override flag is not set
                logger.info('Use cached model')
                return True
        model.eval()
        try:
            traced = torch.jit.script(model)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            traced.save(str(save_path.with_suffix('.zip')))
            logger.info('Torchscript format converted successfully')
            return True
        except Exception:
            # TODO catch different types of error
            logger.warning("This model is not supported as torchscript format")
            return False
