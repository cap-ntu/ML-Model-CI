ARG OS_VER=18.04

# Stage1: Compile
FROM ubuntu:${OS_VER} AS compile-image
ARG PYTHON_VER=3.7

# Install tvm  dependencies and python
WORKDIR /root
RUN apt-get update \
 && apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev libopenblas-dev ninja-build git llvm-10-dev wget\
  python${PYTHON_VER} python${PYTHON_VER}-venv python${PYTHON_VER}-dev python3-pip \
  && cd /tmp \
  && wget -q https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VER} get-pip.py

COPY . /content

# Use venv
ENV VIRTUAL_ENV=/venv
RUN python${PYTHON_VER} -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Build tvm
WORKDIR /root
RUN git clone https://github.com/apache/tvm tvm --recursive

WORKDIR /root/tvm
RUN mkdir -p build \
 && cp cmake/config.cmake build \
 && echo set\(USE_LLVM ON\) >> build/config.cmake \
 && echo set\(USE_GRAPH_RUNTIME ON\) >> build/config.cmake \
 && echo set\(USE_BLAS openblas\) >> build/config.cmake

WORKDIR /root/tvm/build
RUN cmake .. -G Ninja && ninja

WORKDIR /root/tvm/python
RUN pip install pip -U \
 && python setup.py install

# Install python dependencies
WORKDIR /content
RUN pip install .

# Stage2: Build
ARG OS_VER
FROM ubuntu:${OS_VER} AS build-image
ARG PYTHON_VER=3.7
WORKDIR /root
COPY --from=modelci-compile:cpu /venv /venv
RUN apt-get update \
 && apt-get install llvm-10 libopenblas-dev lsof libgl1-mesa-glx libglib2.0-0 python${PYTHON_VER}-distutils python${PYTHON_VER} python${PYTHON_VER}-venv python${PYTHON_VER}-dev -y\
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
ENV PATH="/venv/bin:$PATH"
CMD ["uvicorn", "modelci.app.main:app", "--host", "0.0.0.0", "--port", "8000"]