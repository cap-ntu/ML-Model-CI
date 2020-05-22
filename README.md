<p align="center"> <img src="docs/img/iconv1.svg" width="230" alt="..."> </p>

<h1 align="center">
    Machine Learning Model CI
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
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#tutorial">Tutorial</a> •
  <a href="#benchmark">Benchmark</a> •
  <a href="#contributing">Contributing</a> •
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

### Docker 

```bash
docker pull xxx #TODO
```

Please refer to [here](/intergration/README.md) for more information.

### Command Line  

```bash
curl xxx.get_cli.sh #TODO
# or pip install modelci
```

## Quick Start

ModelCI offers a user-friendly interface for you to manage your model related workflows. 

## Tutorial

Let's go deeper, we have some tutorials here for your reference

- [Register Model in the Model Database]()
- [Converting Model to Different Frameworks]()
- [Profiling Model Automatically]()
- [Deploy Model in Specific Devices]()

## Benchmark

- [ ] ModelZoo with model performance

## Contributing

ModelCI welcomes your contributions! Please refer to [here](.github/CONTRIBUTING.md) to get start.

## Roadmap

- [x] profiler API
- [x] add diagnose tutorial at modelci/hub/register_convert_diagnose_deploy.md
- [ ] restful API
- [x] web frontend
- [ ] according to /intergration, create a setup shell script

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