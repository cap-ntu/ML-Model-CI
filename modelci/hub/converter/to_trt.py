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
from typing import Union, List
import subprocess
import tempfile
import tensorflow as tf
import os
from betterproto import Casing
from google.protobuf import json_format

from modelci.hub.utils import parse_path, GiB, TensorRTPlatform
from modelci.types.bo import IOShape
from modelci.types.trtis_objects import (
    ModelConfig,
    ModelVersionPolicy,
    ModelOutput,
    ModelInput,
    ModelInstanceGroup,
    ModelInstanceGroupKind,
)
from modelci.utils import Logger
import tensorrt as trt


logger = Logger('converter', welcome=False)


class TRTConverter(object):
    supported_framework = ["onnx", "tfs","tensorflow"]


    @staticmethod
    def from_tensorflow(
            savedmodel_path: Path,
            shape: List,
            opset: int = 10,
    ):
        """
        This is the function to create the TensorRT engine
        Args:
        savedmodel_path : Path to savedmodel_file.

        shape : Shape of the input of the tensorflow model.
        """
        """
        This is the function to create the TensorRT engine
        Args:
           onnx_path : Path to onnx_file. 
           shape : Shape of the input of the ONNX file. 
       """
        tmpdir = tempfile.mkdtemp()

        onnx_save = str(os.path.join(tmpdir, "temp_from_tf/")) + '/tempmodel.onnx'
        convertcmd = ['python', '-m', 'tf2onnx.convert', '--saved-model', savedmodel_path, '--output', onnx_save,
                      '--opset', str(opset)]
        subprocess.run(convertcmd)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network,TRT_LOGGER) as parser:
            builder.max_workspace_size = (256 << 20)
            with open(onnx_save, 'rb') as model:
                parser.parse(model.read())
            network.get_input(0).shape = shape
            engine = builder.build_cuda_engine(network)
            return engine

    @staticmethod
    def from_onnx(
            onnx_path: Union[Path, str],
            save_path: Union[Path, str],
            inputs: List[IOShape],
            outputs: List[IOShape],
            int8_calibrator=None,
            create_model_config: bool = True,
            override: bool = False,
    ):
        """Takes an ONNX file and creates a TensorRT engine to run inference with
        From https://github.com/layerism/TensorRT-Inference-Server-Tutorial

        FIXME: bug exist: TRT 6.x.x does not support opset 10 used in ResNet50(ONNX).
        """
        #import tensorrt as trt

        if save_path.with_suffix('.plan').exists():
            if not override:  # file exist yet override flag is not set
                logger.info('Use cached model')
                return True

        onnx_path = Path(onnx_path)
        assert onnx_path.exists()

        save_path = Path(save_path)
        # get arch name
        arch_name = parse_path(save_path)['architecture']

        # trt serving model repository is different from others:
        # `<model-name>/<framework>-tensorrt/<version>/model.plan`
        save_path = save_path.with_suffix('')
        save_path.mkdir(parents=True, exist_ok=True)

        # Save TRT engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(trt_logger) as builder:
            with builder.create_network(
                    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network:
                with trt.OnnxParser(network, trt_logger) as parser:
                    builder.max_workspace_size = GiB(1)  # 1GB
                    builder.max_batch_size = 1
                    if int8_calibrator is not None:
                        builder.int8_mode = True
                        builder.int8_calibrator = int8_calibrator

                    print('Loading ONNX file from path {}...'.format(onnx_path))
                    with open(onnx_path, 'rb') as model:
                        parser.parse(model.read())
                    engine = builder.build_cuda_engine(network)

                    with open(save_path / 'model.plan', 'wb') as f:
                        f.write(engine.serialize())

        # create model configuration file
        if create_model_config:
            TRTConverter.generate_trt_config(
                save_path.parent,
                arch_name=arch_name,
                inputs=inputs,
                outputs=outputs
            )
        return True

    @staticmethod
    def from_tfs(
            tf_path: Union[Path, str],
            save_path: Union[Path, str],
            inputs: List[IOShape],
            outputs: List[IOShape],
            tf_version=2,
            max_batch_size: int = 32,
            max_workspace_size_bytes: int = 1 << 32,
            precision_mode: str = 'FP32',
            maximum_cached_engines: int = 100,
            create_model_config: bool = True,
            override: bool = False,
    ):
        """Convert TensorFlow SavedModel to TF-TRT SavedModel."""
        from tensorflow.python.compiler.tensorrt import trt_convert as trt

        if save_path.with_suffix('.zip').exists():
            if not override:  # file exist yet override flag is not set
                # TODO: add logging
                print('Use cached model')
                return True

        tf_path = Path(tf_path)
        save_path = Path(save_path)
        # get arch name
        arch_name = parse_path(save_path)['architecture']

        # TF SavedModel files should be contained in a directory
        # `~/.modelci/<model-name>/tensorflow-tfs/<version>/model.savedmodel`
        tf_saved_model_path = save_path / 'model.savedmodel'

        assert tf_path.exists()
        save_path.mkdir(parents=True, exist_ok=True)

        if tf_version == 1:
            converter = trt.TrtGraphConverter(
                input_saved_model_dir=str(tf_path),
                max_workspace_size_bytes=max_workspace_size_bytes,
                precision_mode=precision_mode,
                maximum_cached_engines=maximum_cached_engines
            )
        elif tf_version == 2:
            # conversion
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
            conversion_params = conversion_params._replace(
                max_workspace_size_bytes=max_workspace_size_bytes
            )
            conversion_params = conversion_params._replace(precision_mode=precision_mode)
            conversion_params = conversion_params._replace(
                maximum_cached_engines=maximum_cached_engines
            )

            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=str(tf_path),
                conversion_params=conversion_params
            )
        else:
            raise ValueError(f'tf_version expecting a value of `1` or `2`, but got {tf_version}')

        converter.convert()
        converter.save(str(tf_saved_model_path))

        # zip
        shutil.make_archive(save_path, 'zip', root_dir=save_path.parent)

        # create model configuration
        if create_model_config:
            TRTConverter.generate_trt_config(
                save_path.parent,
                arch_name=arch_name,
                platform=TensorRTPlatform.TENSORFLOW_SAVEDMODEL,
                inputs=inputs,
                outputs=outputs,
                max_batch_size=max_batch_size
            )

        return True

    @staticmethod
    def build_model_inputs(node_def: List[IOShape], remove_batch_dim=False):
        """Adopted from https://github.com/layerism/TensorRT-Inference-Server-Tutorial
        """

        model_inputs = []
        for node in node_def:
            # obtain name, dims, and data_type values
            if remove_batch_dim:
                dims = node.shape[1:]
            else:
                dims = node.shape

            model_input = ModelInput(
                name=node.name,
                data_type=node.dtype,
                dims=dims,
                format=node.format
            )

            # TODO: help to reshape

            model_inputs.append(model_input)

        return model_inputs

    @staticmethod
    def build_model_outputs(node_def: List[IOShape], remove_batch_dim=False):
        model_outputs = []
        for node in node_def:
            if remove_batch_dim:
                dims = node.shape[1:]
            else:
                dims = node.shape

            model_output = ModelOutput(
                name=node.name,
                data_type=node.dtype,
                dims=dims
            )

            # TODO: help to reshape

            model_outputs.append(model_output)

        return model_outputs

    @staticmethod
    def generate_trt_config(
            save_path: Path,
            inputs: List[IOShape],
            outputs: List[IOShape],
            arch_name: str = 'model',
            platform: TensorRTPlatform = TensorRTPlatform.TENSORRT_PLAN,
            max_batch_size: int = 32,
            instance_group: List[ModelInstanceGroup] = None
    ):
        """Generate and save TensorRT inference server model configuration file: `model.pbtxt`.

        see here for more detailed configuration
            https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/protobuf_api/model_config.proto.html
        Arguments:
            save_path (Path): Model saving path name, generated by `modelci.hub.utils.generate_path`.
            inputs (List[IOShape]): Input tensors shape definition.
            outputs: (List[IOShape]): Output tensors shape definition.
            arch_name (str): Model architecture name。
            platform (TensorRTPlatform): TensorRT platform name.
            max_batch_size (int): Maximum batch size. This will be activated only when the first dimension of the input
                shape is -1. Otherwise, the Default to 32, indicating the max batch size will be determined
                by the first dimension of the input shape. The batch size from input shape will be suppressed when
                there is a value applied to this argument.
            instance_group (List[ModelInstanceGroup]): Model instance group (workers) definition. Default is to
                create a single instance loading on the first available CUDA device.
        """
        from tensorrtserver.api import model_config_pb2

        # assert batch size
        batch_sizes = list(map(lambda x: x.shape[0], inputs))
        if not all(batch_size == batch_sizes[0] for batch_size in batch_sizes):
            raise ValueError('batch size for inputs (i.e. the first dimensions of `input.shape` are not consistent.')
        if batch_sizes[0] != -1:
            max_batch_size = 0
            remove_batch_dim = False
        else:
            remove_batch_dim = True

        inputs = TRTConverter.build_model_inputs(inputs, remove_batch_dim=remove_batch_dim)
        outputs = TRTConverter.build_model_outputs(outputs, remove_batch_dim=remove_batch_dim)

        if instance_group is None:
            instance_group = [ModelInstanceGroup(kind=ModelInstanceGroupKind.KIND_GPU, count=1, gpus=[0])]

        config = ModelConfig(
            name=str(arch_name),
            platform=platform.name.lower(),
            version_policy=ModelVersionPolicy(),
            max_batch_size=max_batch_size,
            input=inputs,
            output=outputs,
            instance_group=instance_group,
        )

        with open(str(save_path / 'config.pbtxt'), 'w') as cfg:
            # to dict
            config_dict = config.to_dict(casing=Casing.SNAKE)
            # to pbtxt format string
            model_config_message = model_config_pb2.ModelConfig()
            pbtxt_str = str(json_format.ParseDict(config_dict, model_config_message))
            cfg.write(pbtxt_str)
