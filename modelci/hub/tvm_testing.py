import tvm
import torch
from tvm import relay
from pathlib import Path
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy 
# import tensorflow as tf

def to_tvm(model: torch.nn.Module, save_path: Path, input_shape, img_path, input_name, opt_level, override=False):
    """
     Convert a PyTorch nn.Module into TVM.
    """
    if save_path.with_suffix('.zip').exists():
        if not override:  # file exist yet override flag is not set
            print('Use cached model')
            return True
    model = model.eval()
    try:
        input_data = torch.randn(input_shape)
        traced_model = torch.jit.trace(model, input_data).eval()
    
        # image for testing 
        img = Image.open(img_path).resize((224, 224))

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
        # img = tf.expand_dims(img, 0).numpy()
       
        # Import graph to relay
        shape_list = [(input_name, img.shape)]
        model, params = relay.frontend.from_pytorch(traced_model, shape_list)
        
        # Relay build
        target = "llvm"
        target_host = "llvm"
        ctx = tvm.cpu(0)    # TODO onlt provides CPU as of now, future work: GPU
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(model, target=target, target_host=target_host, params=params)

        print('TVM format converted successfully')
        return True

    except:
        #TODO catch different types of error
        print("This model is not supported as TVM format")
        return False

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50')
input_shape = [1, 3, 224, 224]
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
print("Image path:", img_path)
input_name = 'input0'
opt_level = 3
test_path = Path("test_tvm")
to_tvm(model=model, save_path=test_path, input_shape=input_shape, input_name=input_name, img_path = img_path, opt_level=opt_level)