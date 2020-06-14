import numpy as np
from PIL import Image

from modelci.hub.utils import TensorRTModelInputFormat
from modelci.types.trtis_objects import DataType


def preprocess(
        img: Image.Image,
        format: TensorRTModelInputFormat,
        dtype: DataType,
        c: int,
        h: int,
        w: int,
        scaling: str
):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.

    Arguments:
        img (Image.Image): Image object to be predicted.
        format (TensorRTModelInputFormat): Format of input tensor.
        dtype (DataType): Data type of input tensor.
        c (int): Channel size.
        h (int): Height size.
        w (int): Weight size.
        scaling (str): Image scaling algorithm. Supported one of `'INCELTION'`, `'VGG'` and `None`.
    """
    # np.set_printoptions(threshold='nan')

    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    typed = resized.astype(dtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 128) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=dtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=dtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == TensorRTModelInputFormat.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered
