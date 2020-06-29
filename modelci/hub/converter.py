import json
import re
import subprocess
from functools import reduce, partial
from pathlib import Path
from typing import Iterable, Union, List

import onnx
import torch
import torch.jit
import torch.onnx
from betterproto import Casing
from onnx import optimizer
from torch import nn

from modelci.types.bo import IOShape
from modelci.types.trtis_objects import (
    ModelConfig,
    ModelVersionPolicy,
    ModelOutput,
    ModelInput,
    ModelInstanceGroup,
    ModelInstanceGroupKind,
)
from ..hub.utils import GiB, parse_path, TensorRTPlatform


class TorchScriptConverter(object):
    @staticmethod
    def from_torch_module(model: nn.Module, save_path: Path, override: bool = False):
        """Save a loaded model in TorchScript."""
        if save_path.with_suffix('.zip').exists():
            if not override:  # file exist yet override flag is not set
                # TODO: add logging
                print('Use cached model')
                return True
        model.eval()
        traced = torch.jit.script(model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(save_path.with_suffix('.zip')))

        return True


class ONNXConverter(object):
    @staticmethod
    def from_torch_module(
            model: nn.Module,
            save_path: Path,
            inputs: Iterable[IOShape],
            opset: int = 10,
            optimize: bool = True,
            override: bool = False,
    ):
        """Save a loaded model in ONNX.

        Arguments:
            model (nn.Module): PyTorch model.
            save_path (Path): ONNX saved path.
            inputs (Iterable[IOShape]): Model input shapes.
            opset (int): ONNX op set version.
            optimize (bool): Flag to optimize ONNX network. Default to `True`.
            override (bool): Flag to override if the file with path to `save_path` has existed. Default to `False`.
        """
        if save_path.with_suffix('.onnx').exists():
            if not override:  # file exist yet override flag is not set
                # TODO: add logging
                print('Use cached model')
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

        dummy_tensors = list()
        for input_ in inputs:
            dummy_tensors.append(torch.rand(batch_size, *input_.shape[1:], requires_grad=True))

        torch.onnx.export(
            model,  # model being run
            tuple(dummy_tensors),  # model input (or a tuple for multiple inputs)
            save_path_with_ext,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=opset,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=['input'],  # the model's input names
            output_names=['output'],  # the model's output names
            keep_initializers_as_inputs=True,
            **export_kwargs
        )

        if optimize:
            network = optim_onnx(save_path_with_ext)
            onnx.save(network, str(save_path_with_ext))

        return True


def optim_onnx(onnx_path, verbose=True):
    """Optimize ONNX network
    """

    model = onnx.load(onnx_path)
    print("Begin Simplify ONNX Model ...")
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
            print(m)

    return model


class TFSConverter(object):
    @staticmethod
    def from_tf_model(model, save_path: Path, override: bool = False):
        import tensorflow as tf

        if save_path.with_suffix('.zip').exists():
            if not override:  # file exist yet override flag is not set
                # TODO: add logging
                print('Use cached model')
                return True

        tf.compat.v1.saved_model.save(model, str(save_path))
        subprocess.call(['zip', '-r', save_path.with_suffix('.zip'), save_path])

        return True


class TRTConverter(object):

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
        import tensorrt as trt

        if save_path.with_suffix('.plan').exists():
            if not override:  # file exist yet override flag is not set
                # TODO: add logging
                print('Use cached model')
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
    def from_saved_model(
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
        subprocess.call(['zip', '-r', save_path.with_suffix('.zip'), save_path])

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
            # to pretty JSON string
            json_str = json.dumps(config.to_dict(casing=Casing.SNAKE), indent=2)
            # to pbtxt format string
            pbtxt_str = TRTConverter._format_json(json_str)
            cfg.write(pbtxt_str)

    @staticmethod
    def _format_json(json_str: str):
        """Format pretty json string into the format of `config.pbtxt`.

        Arguments:
            json_str (str): JSON string in pretty format.
        """

        def remove_outer_parenthesis(pretty_json: str):
            # remove { ... }
            pretty_json = re.sub(r'{\n(.*)\n}', r'\1', pretty_json, flags=re.S | re.MULTILINE)
            # remove over indentation
            pretty_json = re.sub(r'(^\s{2})', '', pretty_json, flags=re.MULTILINE)

            return pretty_json

        def reformat_digit_list(pretty_json: str):
            """Reformat digit list where digits are wrapped by `"`.

            Examples:
                >>> pretty_json = '{' \
                ...     '  "name": "input0",' \
                ...     '  "data_type": "TYPE_FP16",' \
                ...     '  "dims": [' \
                ...     '    "0",' \
                ...     '    "2",' \
                ...     '    "3"' \
                ...     '  ]' \
                ...     '}'
                >>> reformat_digit_list(pretty_json)
                {
                  "name": "input0",
                  "data_type": "TYPE_FP16",
                  "dims": [ 0, 2, 3 ]
                }
            """

            def repl(match_obj):
                matched: str = match_obj.group(0)
                return matched.replace('"', '') \
                    .replace(' ', '') \
                    .replace(',', ', ') \
                    .replace('\n', '') \
                    .replace('[', '[ ').replace(']', ' ]')

            return re.sub(r'\[[^{}]*?\]', repl, pretty_json)

        def reformat_json_key(pretty_json: str):
            """Reformat key name by removing its wrapped quotation marks `"`, and remove ending `,` in each line.

            Examples:
                >>> pretty_json = '{' \
                ...     '  "name": "resnet50"' \
                ...     '  "input": []' \
                ...     '}'
                >>> reformat_json_key(pretty_json)
                {
                  name: "resnet50"
                  input: []
                }
            """
            pretty_json = re.sub(r'^(.*?),$', r'\1', pretty_json, flags=re.MULTILINE)
            return re.sub(r'"(.*?)":', r'\1:', pretty_json)

        def reformat_enum(pretty_json: str, keys: Iterable[str] = None):
            """Reformat values of the given enum keys by removing the quotation marks wrapped around the values.

            Arguments:
                pretty_json (str): JSON-like string, could be formatted by other formatter.
                keys (Iterable[str]): List of the key name of the enum to be formatted. Default to None.
            Raise:
                ValueError: Raised when `keys` is None or not a `Iterable` object.
            Examples:
                >>> pretty_json = '{' \
                ...    '  data_type: "TYPE_FP16"' \
                ...    '  kind: "KIND_GPU"' \
                ...    '}'
                >>> reformat_enum(pretty_json, enums=['data_type', 'kind'])
                {
                  data_type: TYPE_FP16
                  kind: KIND_GPU
                }
            """
            if keys is None:
                raise ValueError('Expecting `Keys` to be an iterable object, but got `None`')
            if not isinstance(keys, Iterable):
                raise ValueError(f'Expecting `Keys` to be an iterable object, but got `{type(keys)}`')

            for key in keys:
                pretty_json = re.sub(rf'({key}:\s*)"(.*?)"$', r'\1\2', pretty_json, flags=re.MULTILINE)
            return pretty_json

        def reformat_object_colon(pretty_json: str, keys: Iterable[str] = None):
            """Reformat by removing colon after given keys

            >>> pretty_json = '{' \
            ...     '  name: "resnet50"' \
            ...     '  input: [' \
            ...     '    {' \
            ...     '      name: "input0"' \
            ...     '      data_type: TYPE_FP16' \
            ...     '      dims: [ 0, 1, 2]' \
            ...     '    }' \
            ...     '  ],' \
            ...     '  output: []' \
            ...     '  instance_group: []' \
            ...     '}'
            >>> keys = ['input', 'output']
            >>> reformat_object_colon(pretty_json, keys)

            """
            if keys is None:
                raise ValueError('Expecting `Keys` to be an iterable object, but got `None`')
            if not isinstance(keys, Iterable):
                raise ValueError(f'Expecting `Keys` to be an iterable object, but got `{type(keys)}`')

            for key in keys:
                pretty_json = re.sub(rf"({key}):\s*", r'\1 ', pretty_json)
            return pretty_json

        # call the format functions in a sequence
        function_pipeline = [
            remove_outer_parenthesis,
            reformat_digit_list,
            reformat_json_key,
            partial(reformat_enum, keys=['data_type', 'kind', 'format']),
            partial(reformat_object_colon, keys=['input', 'output', 'instance_group'])
        ]

        return reduce(lambda x, func: func(x), function_pipeline, json_str)


def to_tvm(*args, **kwargs):
    raise NotImplementedError('Method `to_tvm` is not implemented.')
