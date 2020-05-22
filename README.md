<p align="center"> <img src="docs/img/iconv1.svg" width="250" alt="..."> </p>

<h1 align="center">
    Machine Learning Model CI
</h1>

<p align="center">
    <a href="https://github.com/ellerbrock/open-source-badges/" title="Open Source Love"><img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103"></a>
    <a href="https://travis-ci.com/cap-ntu/ML-Model-CI" title="Build Status"><img src="https://travis-ci.com/cap-ntu/ML-Model-CI.svg?token=SvqJmaGbqAbwcc7DNkD2&branch=master"></a>
    <a href="https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cap-ntu/ML-Model-CI&amp;utm_campaign=Badge_Grade" title="Codacy Badge"><img src="https://app.codacy.com/project/badge/Grade/bfb9f8b11d634602acd8b67484a43318"></a>
    <a href="https://app.fossa.com/projects/custom%2B8170%2Fgithub.com%2Fcap-ntu%2FML-Model-CI?ref=badge_shield" title="FOSSA Status"><img src="https://app.fossa.com/api/projects/custom%2B8170%2Fgithub.com%2Fcap-ntu%2FML-Model-CI.svg?type=shield"></a>
    <a href="https://github.com/cap-ntu/ML-Model-CI/graphs/commit-activity" title="Maintenance"><img src="https://img.shields.io/badge/Maintained%3F-YES-yellow.svg"></a>
    <a href="https://gitter.im/ML-Model-CI/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge" title="Gitter"><img src="https://badges.gitter.im/ML-Model-CI/community.svg"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#tutorial">Tutorial</a> •
  <a href="#benchmark">Benchmark</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="#license">License</a>
</p>



## Features

- **Housekeeper** provides a refined management for model registration, deletion, update and selection.
- **Converter** is to convert models between frameworks.
- **Profiler** can diagnose a model and provide a detailed report about model performance in production environment.
- **Dispatcher** is to place models to devices and serve them automatically.
- **Controller** supports elastic test and deployment.

## Installation

- [ ] One-step Docker installation [[Refer]](/intergration/README.md)

#### Create Environment

```shell script
# create environment
conda env create -f environment.yml

# install PyTorch with specific CUDA version
conda install pytorch torchvision cudatoolkit=<YOUR_CUDA_VERSION> -c pytorch
pip install tensorflow-serving-api
```

#### Install Service

```shell script
sh scripts/start_service.sh
```

### Setup environment variables

#### Option1: Using setup script

```shell script
source scripts/setup_env.sh
```

#### Option2: Using IDE

1. Add `modelci/env-mongodb.env` as an EnvFile.
2. Set project root as source root.

## Quick Start


## Tutorial

Please go to [[Tutorial]](/register_convert_diagnose_deploy.md)

## Benchmark

- [ ] ModelZoo with model performance

## Roadmap

- [ ] diagnoser API
- [ ] add diagnose tutorial at modelci/hub/register_convert_diagnose_deploy.md
- [ ] restful API
- [ ] web frontend
- [ ] according to /intergration, create a setup shell script
- [ ] opensource stuff

Provide CURD for models

- [x] create -> register
- [ ] update
- [x] retrive
- [ ] delete

- [ ] cli toolkit (click package)
- [ ] intergrate k8s

## License
```
   Copyright 2020 Nanyang Technological University, Singapore

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