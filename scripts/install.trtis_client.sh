#!/usr/bin/env bash

VERSION=1.8.0
UBUNTU_VERSION=$(lsb_release -sr | tr -d '.')
FILENAME=v"${VERSION}"_ubuntu"${UBUNTU_VERSION}".clients.tar.gz

mkdir -p ~/tmp
cd ~/tmp || return 1
mkdir -p tensorrtserver && cd tensorrtserver || return 1

# get package
if [ -f "${FILENAME}" ] ; then
  echo "Already downloaded at ${FILENAME}"
  tar xzf "${FILENAME}"
else
  wget https://github.com/NVIDIA/tensorrt-inference-server/releases/download/v"${VERSION}"/"${FILENAME}"
  tar xzf "${FILENAME}"
fi

# install
pip install python/tensorrtserver-${VERSION}-py2.py3-none-linux_x86_64.whl
