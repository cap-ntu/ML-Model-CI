<p align="center"> <img src="docs/img/iconv1.svg" width="230" alt="..."> </p>

<h1 align="center">
    Machine Learning Model CI - 中文简介
</h1>

<p align="center">
    <a href="https://www.python.org/downloads/release/python-370/" title="python version"><img src="https://img.shields.io/badge/Python-3.7%2B-blue.svg"></a>
    <a href="https://travis-ci.com/cap-ntu/ML-Model-CI" title="Build Status"><img src="https://travis-ci.com/cap-ntu/ML-Model-CI.svg?token=SvqJmaGbqAbwcc7DNkD2&branch=master"></a>
    <a href="https://app.fossa.com/projects/custom%2B8170%2Fgithub.com%2Fcap-ntu%2FML-Model-CI?ref=badge_shield" title="FOSSA Status"><img src="https://app.fossa.com/api/projects/custom%2B8170%2Fgithub.com%2Fcap-ntu%2FML-Model-CI.svg?type=shield"></a>
    <a href="https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cap-ntu/ML-Model-CI&amp;utm_campaign=Badge_Grade" title="Codacy Badge"><img src="https://app.codacy.com/project/badge/Grade/bfb9f8b11d634602acd8b67484a43318"></a>
    <a href="https://codebeat.co/a/yizheng-huang/projects/github-com-cap-ntu-ml-model-ci-master"><img alt="codebeat badge" src="https://codebeat.co/badges/343cc340-21c6-4d34-ae2c-48a48e2862ba" /></a>
    <a href="https://github.com/cap-ntu/ML-Model-CI/graphs/commit-activity" title="Maintenance"><img src="https://img.shields.io/badge/Maintained%3F-YES-yellow.svg"></a>
    <a href="https://gitter.im/ML-Model-CI/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge" title="Gitter"><img src="https://badges.gitter.im/ML-Model-CI/community.svg"></a>
</p>

<p align="center">
    <a href="README.md">English Version</a> •
    <a href="#系统简介">系统简介</a> •
    <a href="#简易安装">简易安装</a> •
    <a href="#快速使用">快速使用</a> •
    <a href="#更多例子">更多例子</a> •
    <a href="#详细教程">详细教程</a> •
    <a href="#加入我们">加入我们</a> •
    <a href="#文献引用">文献引用</a> •
    <a href="#版权许可">版权许可</a>
</p>

## 系统简介

Machine Learning Model CI 是一个**云上一站式机器学习模型和服务运维平台**，旨在解决模型训练完成后，到上线成为服务的”最后一公里问题“ -- 在训练得到模型和线上机器学习应用之间，构建了一个高度自动化的桥梁。

系统目前正处于快速迭代开发中，目前我们提供了如下功能，用户 1）可以注册模型到我们的系统，享受自动化的一揽子服务；2）也可以分别使用各个功能

1. **模型管家.** 该模块接受用户注册的原始训练模型，将其存储到一个中心化的数据库当中。并提供了若干API帮助用户在本地修改，检索，删除模型。
2. **模型转换.** 在收到用户的注册请求后，模型会被自动优化和转化为高性能的部署格式。目前支持的格式有Tensorflow SavedModel, ONNX, TorchScript, TensorRT。
3. **模型解析评估.** 为了保证高质量的线上服务，上线之前的模型需要大量的性能评估测试，一方面给模型性能调优提供参考，另一方面给线上服务设置提供参考。我们的评估模块可以对模型，硬件设施，软件设施进行基准评测，提供了p99延迟，吞吐等多维度的指标。
4. **模型分发上线.** 研究环境和生产环境一般是不同的，同时模型需要和模型推理服务引擎进行绑定进行服务。该模块将用户转换后的模型与各类引擎进行绑定，然后打包成docker服务，可以快速部署上线。
5. **流程控制调度.** 我们提供了一个调度器，一方面控制整个流程的自动化实施，另一方面会将各种模型转化、解析评估等任务，分发到较为空闲机器，提高集群的利用率，让整个流程更高效安全。

下面若干个功能正处于测试状态，马上会在下一个版本推出，读者可以到issue中和我们进行讨论。

- [ ] **模型优化.** 我们希望将模型量化、剪枝等加入到我们的自动化管道中。
- [ ] **模型可视化微调优** 我们希望用户可以零代码的查看和调优团队中的模型。

我们非常欢迎感兴趣的同学加入到我们的开发，请联系
> *huaizhen001 AT e.ntu.edu.sg*

## 简易安装

### pip安装
```shell script
# 确保安装依赖是最新版本
pip install -U setuptools requests
# 从github上自动下载安装
pip install git+https://github.com/cap-ntu/ML-Model-CI.git@master
```

### conda工作空间安装
**Note**
- 需要已经安装好conda和docker
- 需要`sudo`权限安装TensorRT （如需使用）

```shell script
git clone git@github.com:cap-ntu/ML-Model-CI.git
cd ML-Model-CI
bash scripts/install.sh
```

### Docker快速安装

![](https://img.shields.io/docker/pulls/mlmodelci/mlmodelci.svg) ![](https://img.shields.io/docker/image-size/mlmodelci/mlmodelci)

```shell script
docker pull mlmodelci/mlmodelci
```

## 快速使用
下面两幅图分别展示了我们系统的网页前台和整体的工作流程。接下来我们简要介绍如何快速上手系统
| Web frontend |   Workflow     |
|:------------:|:--------------:|
| <img src="https://i.loli.net/2020/12/10/4FsfciXjtPO12BQ.png" alt="drawing" width="500"/> | <img src="https://i.loli.net/2020/12/10/8IaeW9mS2NjQEYB.png" alt="drawing" width="500"/>    |

### 1. 启动MLModelCI中心服务

```shell script
modelci service init
```

### 2. 注册和发布模型

```python
from modelci.hub.manager import register_model
from modelci.types.bo import IOShape, Task, Metric

# 利用模型管家快速发布模型到系统中
register_model(
    'home/ResNet50/pytorch/1.zip',
    dataset='ImageNet',
    metric={Metric.ACC: 0.76},
    task=Task.IMAGE_CLASSIFICATION,
    inputs=[IOShape([-1, 3, 224, 224], float)],
    outputs=[IOShape([-1, 1000], float)],
    convert=True,
    profile=True
)
```

### 3. 自动转换模型

```python
from modelci.hub.converter import ONNXConverter
from modelci.types.bo import IOShape

# 系统会自动启动模型转换，用户也可以手工调用该函数
ONNXConverter.from_torch_module(
    '<path to torch model>', 
    '<path to export onnx model>', 
    inputs=[IOShape([-1, 3, 224, 224], float)],
)
```

### 4. 自动模型性能解析

```python
from modelci.hub.client.torch_client import CVTorchClient
from modelci.hub.profiler import Profiler

# 系统会利用空闲机器，对模型性能进行分析。
# 用户也可以单独调用该方法
test_data_item = ...
batch_num = ...
batch_size = ...
model_info = ...

torch_client = CVTorchClient(test_data_item, batch_num, batch_size, asynchronous=False)

profiler = Profiler(model_info=model_info, server_name='name of your server', inspector=torch_client)

profiler.diagnose('device name')
```

### 5. 模型部署上线

```python
from modelci.hub.deployer.dispatcher import serve
from modelci.hub.manager import retrieve_model
from modelci.types.bo import Framework, Engine

model_info = ...

# 获取模型信息，讲模型和模型服务系统绑定，并发布上线成为服务
model_info = retrieve_model(architecture_name='ResNet50', framework=Framework.PYTORCH, engine=Engine.TORCHSCRIPT)

serve(save_path=model_info[0].saved_path, device='cuda:0', name='torchscript-serving', batch_size=16) 
```

## 更多例子

- [安装MLModelCI，并发布图像分类模型](./example/notebook/image_classification_model_deployment.ipynb)  [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/cap-ntu/ML-Model-CI/blob/master/example/notebook/image_classification_model_deployment.ipynb)
- [注册并发布目标检测模型](./example/notebook/object_detection_model_deployment.ipynb)  [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/cap-ntu/ML-Model-CI/blob/master/example/notebook/object_detection_model_deployment.ipynb)
- [ ] 模型性能解析
- [ ] 模型上线服务
- [ ] 模型可视化编辑

## 详细教程
用户可以通过我们的文档，详细理解MLModelCI
- [注册并发布模型](./docs/tutorial/register.md)
- [团队模型管理](./docs/tutorial/housekeeper.md)
- [模型转化](./docs/tutorial/convert.md)
- [模型性能解析](./docs/tutorial/profile.md)
- [模型上线部署](./docs/tutorial/retrieve-and-deploy.md)
- [ ] 在一个团队中进行模型可视化编辑


## 加入我们

请参考 [代码贡献指南](CONTRIBUTING.md) 来加入我们的开发.

## 文献引用

如果您使用了我们系统的代码，我们非常感谢您可以引用我们的论文

```
@inproceedings{10.1145/3394171.3414535,
  author = {Zhang, Huaizheng and Li, Yuanming and Huang, Yizheng and Wen, Yonggang and Yin, Jianxiong and Guan, Kyle},
  title = {MLModelCI: An Automatic Cloud Platform for Efficient MLaaS},
  year = {2020},
  url = {https://doi.org/10.1145/3394171.3414535},
  doi = {10.1145/3394171.3414535},
  booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
  pages = {4453–4456},
  numpages = {4},
  location = {Seattle, WA, USA},
  series = {MM '20}
}
```

## 版权许可
```
   Copyright 2021 Nanyang Technological University, Singapore

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
