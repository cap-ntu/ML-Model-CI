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

import onnx
import subprocess
import onnxmltools as onnxmltools
import os
import torch
import torch.jit
import torch.onnx
import tempfile
import tensorflow as tf
from modelci.utils import Logger

from modelci.types.type_conversion import model_data_type_to_torch, model_data_type_to_onnx
from onnx import optimizer
from tensorflow import keras
from modelci.types.bo import IOShape
logger = Logger('converter', welcome=False)


class ONNXConverter(object):
    """Convert model to ONNX format."""

    DEFAULT_OPSET = 10
    supported_framework = ["pytorch", "keras", "sklearn", "xgboost", "lightgbm", "tensorflow"]

    class _Wrapper(object):
        @staticmethod
        def save(converter: Callable[..., 'onnx.ModelProto']):
            def wrap(
                    *args,
                    save_path: Path = None,
                    optimize: bool = True,
                    override: bool = False,
                    **kwargs
            ) -> 'onnx.ModelProto':
                onnx_model = None
                save_path_with_ext = None

                if save_path is not None:
                    save_path = Path(save_path)
                    save_path_with_ext = save_path.with_suffix('.onnx')
                    if save_path_with_ext.exists() and not override:
                        # file exist yet override flag is not set
                        logger.info('Use cached model')
                        onnx_model = onnx.load(str(save_path))

                if onnx_model is None:
                    # otherwise, convert model
                    onnx_model = converter(*args, **kwargs)

                if optimize:
                    # optimize ONNX model
                    onnx_model = ONNXConverter.optim_onnx(onnx_model)

                if save_path_with_ext:
                    # save to disk
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    onnxmltools.utils.save_model(onnx_model, save_path_with_ext)

                return onnx_model

            return wrap

    @staticmethod
    def from_pytorch(
            model: torch.nn.Module,
            save_path: Path,
            inputs: Iterable[IOShape],
            outputs: Iterable[IOShape],
            model_input: Optional[List] = None,
            opset: int = 10,
            optimize: bool = True,
            override: bool = False,
    ):
        """Save a loaded model in ONNX.
            TODO: reuse inputs to pass model_input parameter later

        Arguments:
            model (nn.Module): PyTorch model.
            save_path (Path): ONNX saved path.
            inputs (Iterable[IOShape]): Model input shapes. Batch size is indicated at the dimension.
            outputs (Iterable[IOShape]): Model output shapes.
            model_input (Optional[List]) : Sample Model input data
            opset (int): ONNX op set version.
            optimize (bool): Flag to optimize ONNX network. Default to `True`.
            override (bool): Flag to override if the file with path to `save_path` has existed. Default to `False`.
        """
        if save_path.with_suffix('.onnx').exists():
            if not override:  # file exist yet override flag is not set
                logger.info('Use cached model')
                return True

        export_kwargs = dict()

        # assert batch size
        batch_sizes = list(map(lambda x: x.shape[0], inputs))
        if not all(batch_size == batch_sizes[0] for batch_size in batch_sizes):
            raise ValueError('batch size for inputs (i.e. the first dimensions of `input.shape` are not consistent.')
        batch_size = batch_sizes[0]

        if batch_size == -1:
            export_kwargs['dynamic_axes'] = {
                'input': {0: 'batch_size'},  # variable length axes
                'output': {0: 'batch_size'}
            }
            batch_size = 1
        else:
            assert batch_size > 0

        model.eval()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path_with_ext = save_path.with_suffix('.onnx')

        dummy_tensors, input_names, output_names = list(), list(), list()
        for input_ in inputs:
            dtype = model_data_type_to_torch(input_.dtype)
            dummy_tensors.append(torch.rand(batch_size, *input_.shape[1:], requires_grad=True, dtype=dtype))
            input_names.append(input_.name)
        for output_ in outputs:
            output_names.append(output_.name)
        if model_input is None:
            model_input = tuple(dummy_tensors)
        try:
            torch.onnx.export(
                model,  # model being run
                model_input,  # model input (or a tuple for multiple inputs)
                save_path_with_ext,  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=opset,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=input_names,  # the model's input names
                output_names=output_names,  # the model's output names
                keep_initializers_as_inputs=True,
                **export_kwargs
            )

            if optimize:
                onnx_model = onnx.load(str(save_path_with_ext))
                network = ONNXConverter.optim_onnx(onnx_model)
                onnx.save(network, str(save_path_with_ext))

            logger.info('ONNX format converted successfully')
            return True
        except Exception as e:
            # TODO catch different types of error
            logger.error('Unable to convert to ONNX format, reason:')
            logger.error(e)
            return False

    @staticmethod
    def from_keras(
            model: keras.models.Model,
            opset: int = DEFAULT_OPSET,
    ):
        """return a loaded model in ONNX.
            TODO: revise this function when tensorflow-onnx updated on pypi and use tf2onnx.convert.from_keras()

        Arguments:
            model : Keras model.
            opset (int): ONNX op set version.
        """
        import tf2onnx
        version = tf2onnx.version
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "temp_from_keras/")

            tf.saved_model.save(model, save_path)
            onnx_save = str(save_path)+'/last_tf2onnx_model.onnx'
            try:
                convertcmd = ['python', '-m', 'tf2onnx.convert', '--saved-model', save_path, '--output', onnx_save,
                            '--opset', str(opset)]
                subprocess.run(convertcmd)
                logger.info('ONNX format converted successfully'+'tf2onnx_version : '+str(version))
                return onnx.load(onnx_save)
            except Exception as e:
                logger.error('Unable to convert to ONNX format, reason:')
                logger.error(e)
                return False

    @staticmethod
    def from_tensorflow(
            saved_model_path: Path,
            opset: int = DEFAULT_OPSET,
    ):
        """return a loaded model in ONNX.
            TODO: revise this function when tensorflow-onnx updated on pypi and use tf2onnx.convert.from_keras()

        Arguments:
            saved_model_path : savedmodel path.
            opset (int): ONNX op set version.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "temp_from_tensorflow/")
            onnx_save = str(save_path)+'/tf_savedmodel.onnx'
            try:
                convertcmd = ['python', '-m', 'tf2onnx.convert', '--saved-model', saved_model_path, '--output', onnx_save,
                              '--opset', str(opset)]
                subprocess.run(convertcmd)
                logger.info('ONNX format converted successfully')
                return onnx.load(onnx_save)
            except Exception as e:
                logger.error('Unable to convert to ONNX format, reason:')
                logger.error(e)
                return False

    @staticmethod
    @_Wrapper.save
    def from_sklearn(
            model,
            inputs: Iterable[IOShape],
            opset: int = DEFAULT_OPSET,
    ):
        initial_type = ONNXConverter.convert_initial_type(inputs)
        return onnxmltools.convert_sklearn(model, initial_types=initial_type, target_opset=opset)

    @staticmethod
    @_Wrapper.save
    def from_xgboost(model, inputs: Iterable[IOShape], opset: int = DEFAULT_OPSET):
        initial_type = ONNXConverter.convert_initial_type(inputs)
        return onnxmltools.convert_xgboost(model, initial_types=initial_type, target_opset=opset)

    @staticmethod
    @_Wrapper.save
    def from_lightgbm(model, inputs: Iterable[IOShape], opset: int = DEFAULT_OPSET):
        initial_type = ONNXConverter.convert_initial_type(inputs)
        return onnxmltools.convert_lightgbm(model, initial_types=initial_type, target_opset=opset)

    @staticmethod
    def convert_initial_type(inputs: Iterable[IOShape]):
        # assert batch size
        batch_sizes = list(map(lambda x: x.shape[0], inputs))
        if not all(batch_size == batch_sizes[0] for batch_size in batch_sizes):
            raise ValueError(
                'batch size for inputs (i.e. the first dimensions of `input.shape` are not consistent.')
        batch_size = batch_sizes[0]

        if batch_size == -1:
            batch_size = None
        else:
            assert batch_size > 0

        initial_type = list()
        for input_ in inputs:
            initial_type.append((input_.name, model_data_type_to_onnx(input_.dtype)([batch_size, *input_.shape[1:]])))
        return initial_type

    @staticmethod
    def optim_onnx(model: onnx.ModelProto, verbose=False):
        """Optimize ONNX network"""
        logger.info("Begin Simplify ONNX Model ...")
        passes = [
            'eliminate_deadend',
            'eliminate_identity',
            'extract_constant_to_initializer',
            'eliminate_unused_initializer',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
            'fuse_matmul_add_bias_into_gemm'
        ]
        model = optimizer.optimize(model, passes)

        if verbose:
            for m in onnx.helper.printable_graph(model.graph).split("\n"):
                logger.debug(m)

        return model
