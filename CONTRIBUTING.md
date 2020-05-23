# Contributing

ModelCI welcomes your contributions!

## Setup Environment

To contribute to the ModelCI, you need to setup the developing environment first, you can easily create the dev environments according to the following steps. All the scripts should be executed under the root project of this repo.

### Create Anaconda Environment

```shell script
# create environment
conda env create -f environment.yml

# install PyTorch with specific CUDA version
conda install pytorch torchvision cudatoolkit=<YOUR_CUDA_VERSION> -c pytorch
pip install tensorflow-serving-api
```

### Install Service

```shell script
sh scripts/start_service.sh
```

After installing the service we need, you must setup the environment variables to activate the service. You can run the script in command line, if you are using IDE to develop, you should add the EnvFile manually.


#### Option1: Using Script

```shell script
source scripts/setup_env.sh
```

#### Option2: Setup Manually while Using IDE

1. Add `modelci/env-mongodb.env` as an EnvFile.
2. Set project root as source root.


After these steps, try runing ModelCI locally and start making a difference!


## Coding Standards

### Unit Tests
[PyTest](https://docs.pytest.org/en/latest/) is used to execute tests. PyTest can be
installed from PyPI via `pip install pytest`. 

```bash
python -m pytest tests/
```

You can also provide the `-v` flag to `pytest` to see additional information about the
tests.


### Code Style

We have some static checks when you filing a PR or pushing commits to the project, please make sure you can pass all the tests and make sure the coding style meets our requirements.

