FROM nvcr.io/nvidia/tensorrt:19.10-py3

# set system environment
ENV CONDA_ROOT=/miniconda
ENV CONDA_PREFIX=${CONDA_ROOT}
ENV PATH=${CONDA_ROOT}/bin:${PATH}
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV TRITONIS_VERSION=1.8.0
ENV PYTHONPATH=/content/

COPY . /content

# Change all files EOF to LF
RUN find /content/scripts -type f -exec sed -i -e 's/^M$//' {} \;

RUN apt-get update -y \
 && apt-get install -y curl=7.58.0-2ubuntu3.8 zip=3.0-11build1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -o /miniconda.sh \
 && sh /miniconda.sh -b -p "${CONDA_ROOT}" \
 && rm /miniconda.sh

# Install Conda environment
RUN conda env update --name base -f /content/environment.yml \
 && conda install pytorch=1.5.0 torchvision cudatoolkit="${CUDA_VERSION}" -y -c pytorch \
 && conda install tensorflow-gpu=2.1.0 -y \
 && pip install tensorflow-serving-api==2.1.0

# Install TRTIS
RUN mkdir -p ~/tmp
WORKDIR /root/tmp
RUN curl -LJ https://github.com/NVIDIA/triton-inference-server/releases/download/v${TRITONIS_VERSION}/v${TRITONIS_VERSION}_ubuntu1804.clients.tar.gz \
    -o tritonis.clients.tar.gz \
 && tar xzf tritonis.clients.tar.gz \
 && pip install ~/tmp/python/tensorrtserver-${TRITONIS_VERSION}-py2.py3-none-linux_x86_64.whl

# Uninstall build dependency
RUN apt-get remove -y curl wget \
 && apt-get clean \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# remove cache
RUN conda clean -ya \
 && rm -rf ~/.cache/pip \
 && rm -rf ~/tmp

WORKDIR /content

ENTRYPOINT ["/bin/bash"]
