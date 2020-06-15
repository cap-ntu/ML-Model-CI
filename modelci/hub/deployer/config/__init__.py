"""Default settings for deployer

1. Port
    | Engine Name | HTTP Port | gRPC Port | HTTP Port (GPU)  | gRPC Port (GPU) |
    |-------------|:---------:|:---------:|:----------------:|:---------------:|
    | ONNX        | 8001      | 8002      | 8010             | 8011            |
    | TorchScript | 8100      | 8101      | 8110             | 8111            |
    | TRT         | 8200      | 8201      | 8202 (Prometheus)| -               |
    | TFS         | 8501      | 8500      | 8510             | 8511            |
"""

ONNX_HTTP_PORT = 8001
ONNX_GRPC_PORT = 8002
ONNX_HTTP_PORT_GPU = 8010
ONNX_GRPC_PORT_GPU = 8011

TORCHSCRIPT_HTTP_PORT = 8100
TORCHSCRIPT_GRPC_PORT = 8101
TORCHSCRIPT_HTTP_PORT_GPU = 8110
TORCHSCRIPT_GRPC_PORT_GPU = 8111

TRT_HTTP_PORT = 8200
TRT_GRPC_PORT = 8201
TRT_PROMETHEUS_PORT = 8202

TFS_HTTP_PORT = 8501
TFS_GRPC_PORT = 8500
TFS_HTTP_PORT_GPU = 8510
TFS_GRPC_PORT_GPU = 8511
