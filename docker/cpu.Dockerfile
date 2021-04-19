# Stage1: Compile
FROM ubuntu:18.04 AS compile-image

# Install tvm  dependencies and python
WORKDIR /root

RUN apt-get update \
 && apt-get install -y --no-install-recommends\
 gcc=4:7.4.0-1ubuntu2.3 \
 libtinfo-dev=6.1-1ubuntu1.18.04 \
 zlib1g-dev=1:1.2.11.dfsg-0ubuntu2 \
 build-essential=12.4ubuntu1 \
 cmake=3.10.2-1ubuntu2.18.04.1 \
 libedit-dev=3.1-20170329-1 \
 libxml2-dev=2.9.4+dfsg1-6.1ubuntu1.3 \
 libopenblas-dev=0.2.20+ds-4 \
 ninja-build=1.8.2-1 \
 git=1:2.17.1-1ubuntu0.8 \
 llvm-10-dev=1:10.0.0-4ubuntu1~18.04.2 \
 wget=1.19.4-1ubuntu2.2 \
 python3.7=3.7.5-2~18.04.4 \
 python3.7-venv=3.7.5-2~18.04.4 \
 python3.7-dev=3.7.5-2~18.04.4 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN wget -q https://bootstrap.pypa.io/get-pip.py && python3.7 get-pip.py

COPY . /content

# Use venv
ENV VIRTUAL_ENV=/venv
RUN python3.7 -m venv $VIRTUAL_ENV
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
RUN pip install --no-cache-dir pip -U \
 && python setup.py install --no-cache-dir

# Install python dependencies
WORKDIR /content
RUN pip install --no-cache-dir .

# Stage2: Build
FROM ubuntu:18.04 AS build-image
COPY --from=compile-image /venv /venv

RUN apt-get update && apt-get install -y --no-install-recommends \
 llvm-10=1:10.0.0-4ubuntu1~18.04.2 \
 libopenblas-dev=0.2.20+ds-4 \
 lsof=4.89+dfsg-0.1 \
 libgl1-mesa-glx=20.0.8-0ubuntu1~18.04.1 \
 libglib2.0-0=2.56.4-0ubuntu0.18.04.8 \
 python3.7-distutils=3.7.5-2~18.04.4 \
 python3.7=3.7.5-2~18.04.4 \
 python3.7-venv=3.7.5-2~18.04.4 \
 python3.7-dev=3.7.5-2~18.04.4 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
ENV PATH="/venv/bin:$PATH"
CMD ["uvicorn",  "modelci.app.main:app", "--host", "0.0.0.0", "--port", "8000"]