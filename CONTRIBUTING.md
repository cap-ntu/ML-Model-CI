# Contributing

MLModelCI welcomes your contributions!

## Setup Environment

To contribute to the MLModelCI, you need to setup the developing environment first, you can easily create the dev environments according to the following steps. All the scripts should be executed under the root project of this repo.

### Create Anaconda Environment

```shell script
# create environment
conda env create -f environment.yml

# install PyTorch with specific CUDA version
conda install pytorch torchvision cudatoolkit=<YOUR_CUDA_VERSION> -c pytorch
conda install pytorch-lightning -c conda-forge
pip install hummingbird-ml==0.2.1 torchviz==0.0.1
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


After these steps, try running MLModelCI locally and start making a difference!


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

## Contribution guide

### 1. Clone your fork and configure the upstream

Firstly, you have to fork this repo: <https://github.com/cap-ntu/ML-Model-CI> and create a local clone of this fork

```shell
git clone https://github.com/<YOUR-USERNAME>/ML-Model-CI.git
```

Then add the original repository as upstream

```shell
cd ML-Model-CI
git remote add upstream https://github.com/cap-ntu/ML-Model-CI.git
```

you can use the following command to verify that the remote is set

```shell
git remote -v
```

### 2. Sync the changes of the original repo

This is a must before you make any changes to your fork, or conflict may happen.

```shell
git fetch upstream
git checkout master
git merge upstream/master
git push origin master
```

or you can automate simplify this work by the following command

```shell
git branch --set-upstream-to=upstream/master master
git config --local remote.pushDefault origin
```

and then the workflow becomes:

```shell
git checkout master
git fetch
git merge
git push
```

### 3. Select (or open) an issue and create a new branch

Before you may any pull request, please correlate your branch with at least one [existing opened issue](https://github.com/cap-ntu/ML-Model-CI/issues) or open one with appropriate labels.

Then you can create a new branch, make sure the branch name is descriptive.

```shell
git checkout -b <NEW-BRANCH-NAME>
```

### 4. Make the change and commit

After making the necessary change, you can commit it to your local repository. There is a general rule for making a commit:

> Commits should be logical, atomic units of change

```shell
git add -A
git commit -m "<COMMIT-MESSAGE>"
```

### 5. Open a pull request

Then push the commit to your Github fork

```shell
git push -u origin <NEW-BRANCH-NAME>
```

You can make a pull request now, make sure to link the relevant issue in your pull request description.
Here are relevant docs which may help:

<https://docs.github.com/en/free-pro-team@latest/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue>

If your branch is out of date during the process, it cannot be merge automatically and you should fix your branch by merging the target branch into your branch.

### 6. Delete the branch

Once your pull request is merged into the original repository, you should delete the new branch and push the deletion to your fork

```shell
git branch -d <NEW-BRANCH-NAME>
git push -u origin master
git push --delete origin
```
