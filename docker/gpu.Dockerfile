# Stage1: Compile
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04 AS compile-image
# Install tvm
ARG PYTHON_VER=3.7
WORKDIR /root
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends\
 gcc \
 libtinfo-dev \
 zlib1g-dev \
 build-essential \
 cmake \
 libedit-dev \
 libxml2-dev \
 libopenblas-dev \
 ninja-build \
 git \
 wget \
 llvm-10-dev \
 python${PYTHON_VER} \
 python${PYTHON_VER}-venv \
 python${PYTHON_VER}-dev \
 python3-pip \
 && wget -q https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VER} get-pip.py

# Use venv
ENV VIRTUAL_ENV=/venv
RUN python3.7 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . /content
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
RUN pip install --no-cache-dir pip -U \
 && python setup.py install

WORKDIR /root
RUN cp /content/TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz . \
 && tar -xvzf TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz \
 && rm TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz

# Install python libs
WORKDIR /root/TensorRT-7.2.1.6/python
RUN pip install --no-cache-dir tensorrt-7.2.1.6-cp37-none-linux_x86_64.whl \
&& pip install --no-cache-dir 'pycuda>=2019.1.1' \
&& pip install --no-cache-dir /root/TensorRT-7.2.1.6/uff/uff-0.6.9-py2.py3-none-any.whl \
&& pip install --no-cache-dir /root/TensorRT-7.2.1.6/graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl \
&& pip install --no-cache-dir /root/TensorRT-7.2.1.6/onnx_graphsurgeon/onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl

# Set environment and working directory
ENV TRT_LIBPATH /root/TensorRT-7.2.1.6/lib
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_LIBPATH}"

# Install python dependencies
ENV CUDA_HOME "/usr/local/cuda"
WORKDIR /content
RUN pip install --no-cache-dir .

# Stage2: Build
FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04 AS build-image
ARG PYTHON_VER=3.7
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends\
 llvm-10 \
 libopenblas-dev \
 lsof \
 libgl1-mesa-glx \
 libglib2.0-0 \
 python${PYTHON_VER}-distutils \
 python${PYTHON_VER} \
 python${PYTHON_VER}-venv \
 python${PYTHON_VER}-dev -y\
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY --from=compile-image /root/TensorRT-7.2.1.6/lib /root/TensorRT-7.2.1.6/lib
COPY --from=compile-image /venv /venv
# Reference: https://github.com/tensorflow/tensorflow/issues/38194#issuecomment-629801937
RUN ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudart.so.10.2 /usr/lib/x86_64-linux-gnu/libcudart.so.10.1
# Set environment and working directory
ENV TRT_LIBPATH /root/TensorRT-7.2.1.6/lib
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_LIBPATH}:/usr/lib/x86_64-linux-gnu/"
ENV PATH="/venv/bin:$PATH"
CMD ["uvicorn",  "modelci.app.main:app", "--host", "0.0.0.0", "--port", "8000"]