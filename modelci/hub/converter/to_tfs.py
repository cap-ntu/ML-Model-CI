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
import shutil
from pathlib import Path

from modelci.utils import Logger

logger = Logger('converter', welcome=False)


class TFSConverter(object):
    supported_framework = ["tensorflow"]

    @staticmethod
    def from_tensorflow(model, save_path: Path, override: bool = False):
        import tensorflow as tf

        if save_path.with_suffix('.zip').exists():
            if not override:  # file exist yet override flag is not set
                logger.info('Use cached model')
                return True

        tf.compat.v1.saved_model.save(model, str(save_path))
        shutil.make_archive(save_path, 'zip', root_dir=save_path.parent)

        return True
