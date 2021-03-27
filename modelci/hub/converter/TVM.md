<h1 align="center">
    Setting Up TVM Through Docker
</h1>

## Introduction
TVM Installation using the TVM documentation can tend to be complicated. ML-Model-CI instead provides a Dockerfile enabling easy set up of TVM within a Docker environment. This README provides a step-by-step process for setting up the environment.

## Setting Up Docker 
1. The original Docker file used can be found at https://github.com/octoml/public-tvm-docker/blob/master/Dockerfile.cpu_base. The Dockerfile in our repository has been modified to suit the requirements of the TVM Conversion File.

2. Next, we create a container image using this Dockerfile. The command for the same is as follows (assuming ML-Model-CI is installed in your home directory):

```shell script
# create an image
docker build -f ML-Model-CI/docker/Dockerfile.cpu -t tvm_docker .
```
The parameters in this command can be explained as follows:
    1. The docker build command builds Docker images from a Dockerfile and a “context”. A build’s context is the set of files located in the specified PATH or URL. The build process can refer to any of the files in the context.
    2. The PATH specifies where to find the files for the “context” of the build on the Docker daemon. The above command will use the current directory as the build context (as specified by the **.**). The PATH is **.** .
    3. We specify a Dockerfile by providing its location: **ML-Model-CI/docker/Dockerfile.cpu**. 
    4. The **-t tvm_docker** is used to name the newly created docker image in a human readable form, such that in the future it can be referenced as ivm-docker. 

3. Next, we start a container using the image we just created. The command is as follows: 

```shell script
# start a container
docker run --name tvm_container -it tvm_docker
```

The parameters in this command can be explained as follows:
    1. The docker run command must specify an **image** to derive the container from. In this case, **tvm_docker** is the name of your image.
    2. Defining a name for your container can be a handy way to add meaning to a container. If you specify a name, you can use it when referencing the container within a Docker network. In this case, **—name tvm_container** does this for us, where our container name is **tvm_container**.
    3. The **-it** instructs Docker to allocate a pseudo-TTY connected to the container’s stdin; creating an interactive bash shell in the container.

4. Now, we must copy our files into this newly created container. The command is as follows:

```shell script
# copy ML-Model-CI contents into docker container
docker cp ML-Model-CI tvm_container:/root/
```
In this command, we copy the contents of the folder ML-Model-CI (present in our home directory) into the container: tvm_container’s root folder.

5. Finally, a new container has been set up successfully and is running with your files, now you can run your files within this docker container like any other shell.
