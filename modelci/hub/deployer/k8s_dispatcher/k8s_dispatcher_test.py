import os
import pytest
from modelci.hub.deployer.k8s_dispatcher.k8s_dispatcher import serve

@pytest.mark.parametrize(
    "configuration, output_file_path",
    [('./modelci/hub/deployer/k8s_dispatcher/sample.config',
    './modelci/hub/deployer/k8s_dispatcher/sample_output.yaml')]
)
def test_generating_deployment_file(configuration, output_file_path):
    serve(configuration=configuration, output_file_path=output_file_path)
    assert os.path.exists('./modelci/hub/deployer/k8s_dispatcher/sample_output.yaml') == 1



