import subprocess
from pathlib import Path
from typing import Iterable, Union, List, Sequence, Optional, Callable

import numpy as np
import onnx
import onnxmltools as onnxmltools
import torch
import torch.jit
import torch.onnx
from betterproto import Casing
from google.protobuf import json_format
from hummingbird.ml import constants as hb_constants
from hummingbird.ml.convert import _convert_xgboost, _convert_lightgbm, _convert_onnxml, _convert_sklearn  # noqa
from hummingbird.ml.operator_converters import constants as hb_op_constants
from hummingbird.ml.supported import xgb_operator_list, lgbm_operator_list, sklearn_operator_list
from onnx import optimizer
from tensorflow import keras

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
from ..types.type_conversion import model_data_type_to_torch, model_data_type_to_onnx
from ..utils import Logger

logger = Logger('converter', welcome=False)


class PyTorchConverter(object):
    hb_common_extra_config = {hb_constants.CONTAINER: False}

    @staticmethod
    def from_xgboost(
            model: Union.__getitem__(tuple(xgb_operator_list)),  # noqa
            inputs: Sequence[IOShape],
            device: str = 'cpu',
            extra_config: Optional[dict] = None,
    ) -> torch.nn.Module:
        """Convert PyTorch module from XGBoost"""
        # inputs for XGBoost should contains only 1 argument with 2 dim
        if not (len(inputs) == 1 and len(inputs[0].shape) == 2):
            raise RuntimeError(
                'XGboost does not support such input data for inference. The input data should contains only 1\n'
                'argument with exactly 2 dimensions.'
            )

        if extra_config is None:
            extra_config = dict()

        # assert batch size
        batch_size = inputs[0].shape[0]
        if batch_size == -1:
            batch_size = 1
        test_input = np.random.rand(batch_size, inputs[0].shape[1])

        extra_config_ = PyTorchConverter.hb_common_extra_config.copy()
        extra_config_.update(extra_config)

        return _convert_xgboost(
            model, 'torch', test_input=test_input, device=device, extra_config=extra_config_
        )

    @staticmethod
    def from_lightgbm(
            model: Union.__getitem__(tuple(lgbm_operator_list)),  # noqa
            inputs: Optional[Sequence[IOShape]] = None,
            device: str = 'cpu',
            extra_config: Optional[dict] = None
    ):
        if extra_config is None:
            extra_config = dict()

        extra_config_ = PyTorchConverter.hb_common_extra_config.copy()
        extra_config_.update(extra_config)

        return _convert_lightgbm(
            model, 'torch', test_input=None, device=device, extra_config=extra_config_
        )

    @staticmethod
    def from_sklearn(
            model: Union.__getitem__(tuple(sklearn_operator_list)),  # noqa
            device: str = 'cpu',
            extra_config: Optional[dict] = None,
    ):
        if extra_config is None:
            extra_config = dict()

        extra_config_ = PyTorchConverter.hb_common_extra_config.copy()
        extra_config_.update(extra_config)

        return _convert_sklearn(
            model, 'torch', test_input=None, device=device, extra_config=extra_config_
        )

    @staticmethod
    def from_onnx(
            model: onnx.ModelProto,
            opset: int = 10,
            device: str = 'cpu',
            extra_config: dict = None,
    ):
        if extra_config is None:
            extra_config = dict()
        inputs = {input_.name: input_ for input_ in model.graph.input}

        extra_config_ = PyTorchConverter.hb_common_extra_config.copy()
        extra_config_.update({
            hb_constants.ONNX_TARGET_OPSET: opset,
            hb_op_constants.ONNX_INPUTS: inputs,
            hb_op_constants.N_FEATURES: None
        })
        extra_config_.update(extra_config)

        return _convert_onnxml(model, 'torch', test_input=None, device=device, extra_config=extra_config_)

__all__ = ['TorchScriptConverter', 'TFSConverter', 'ONNXConverter', 'TRTConverter', 'to_tvm']

class PYTConverter(object):
    @staticmethod
    def from_torch_module(model: torch.nn.Module, save_path: Path, override: bool = False):
        """Save a PyTorch nn.Module by pickle
        """
        if save_path.with_suffix('.pth').exists():
            if not override:  # file exist yet override flag is not set
                logger.info('Use cached model')
                return True
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path_with_ext = save_path.with_suffix('.pth')
        torch.save(model, str(save_path_with_ext))

        return True

class TorchScriptConverter(object):
    @staticmethod
    def from_torch_module(model: torch.nn.Module, save_path: Path, override: bool = False):
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
        except:
            #TODO catch different types of error
            logger.warning("This model is not supported as torchscript format")
            return False


class ONNXConverter(object):
    """Convert model to ONNX format."""

    DEFAULT_OPSET = 10

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
    def from_torch_module(
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

        Arguments:
            model (nn.Module): PyTorch model.
            save_path (Path): ONNX saved path.
            inputs (Iterable[IOShape]): Model input shapes. Batch size is indicated at the dimension.
            outputs (Iterable[IOShape]): Model output shapes.
            model_input (Optional[List]) : Sample Model input data
            TODO reuse inputs to pass model_input parameter later
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
        except:
            #TODO catch different types of error
            logger.warning("This model is not supported as ONNX format")
            return False

    @staticmethod
    @_Wrapper.save
    def from_keras(
            model: keras.models.Model,
            opset: int = DEFAULT_OPSET,
    ):
        return onnxmltools.convert_keras(model, target_opset=opset)

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


class TFSConverter(object):
    @staticmethod
    def from_tf_model(model, save_path: Path, override: bool = False):
        import tensorflow as tf

        if save_path.with_suffix('.zip').exists():
            if not override:  # file exist yet override flag is not set
                logger.info('Use cached model')
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


def to_tvm(*args, **kwargs):
    raise NotImplementedError('Method `to_tvm` is not implemented.')
