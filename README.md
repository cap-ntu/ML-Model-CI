<p align="center"> <img src="docs/img/modelci.png" alt="..."> </p>

# ML-Model-CI

[![Build Status](https://travis-ci.com/cap-ntu/ML-Model-CI.svg?token=SvqJmaGbqAbwcc7DNkD2&branch=master)](https://travis-ci.com/cap-ntu/ML-Model-CI)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/bfb9f8b11d634602acd8b67484a43318)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cap-ntu/ML-Model-CI&amp;utm_campaign=Badge_Grade)
[![FOSSA Status](https://app.fossa.com/api/projects/custom%2B8170%2Fgithub.com%2Fcap-ntu%2FML-Model-CI.svg?type=shield)](https://app.fossa.com/projects/custom%2B8170%2Fgithub.com%2Fcap-ntu%2FML-Model-CI?ref=badge_shield)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-yellow.svg)](https://github.com/cap-ntu/ML-Model-CI/graphs/commit-activity) 
[![GitHub stars](https://img.shields.io/github/stars/cap-ntu/ML-Model-CI.svg?style=social)](https://GitHub.com/cap-ntu/ML-Model-CI/stargazers/)

## Features

- **Housekeeper** provides a refined management for model registration, deletion, update and selection.
- **Converter** is to convert models between frameworks.
- **Profiler** can diagnose a model and provide a detailed report about model performance in production environment.
- **Dispatcher** is to place models to devices and serve them automatically.
- **Controller** supports elastic test and deployment.

## Quick Start

### Installation

- [ ] One-step Docker installation [[Refer]](/intergration/README.md)

#### Create environment

```shell script
# create environment
conda env create -f environment.yml

# install PyTorch with specific CUDA version
conda install pytorch torchvision cudatoolkit=<YOUR_CUDA_VERSION> -c pytorch
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

===============

- [ ] cli toolkit (click package)
- [ ] intergrate k8s
