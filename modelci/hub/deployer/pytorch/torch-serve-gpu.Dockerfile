ARG CUDA="10.1"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-runtime-ubuntu16.04

# set built-time arguments
ARG CUDA

# set system environment
ENV CONDA_ROOT=/miniconda
ENV CONDA_PREFIX=${CONDA_ROOT}
ENV PATH=${CONDA_ROOT}/bin:${PATH}
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV MODEL_NAME='model'
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY ./environment.yml /content/

WORKDIR /content/

# install basics
RUN apt-get update -y \
 && apt-get install -y curl

# Install Miniconda
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -o /miniconda.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

# Create a Python environment
RUN conda env update --name base -f /content/environment.yml \
 && conda install -y pytorch cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya \
 && rm -rf ~/.cache/pip

RUN find ${CONDA_ROOT}/ -follow -type f -name '*.a' -delete 2> /dev/null; exit 0 \
 && find ${CONDA_ROOT}/ -follow -type f -name '*.pyc' -delete 2> /dev/null; exit 0 \
 && find ${CONDA_ROOT}/ -follow -type f -name '*.js.map' -delete 2> /dev/null; exit 0 \
 && find ${CONDA_ROOT}/lib/python*/site-packages/bokeh/server/static \
     -follow -type f -name '*.js' ! -name '*.min.js' -delete 2> /dev/null; exit 0

# Copy source
COPY . /content/

CMD python pytorch_serve.py ${MODEL_NAME}
