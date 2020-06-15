#!/usr/bin/env bash

VERSION=1.8.0
UBUNTU_VERSION=$(lsb_release -sr | tr -d '.')
FILENAME=v"${VERSION}"_ubuntu"${UBUNTU_VERSION}".clients.tar.gz

function download_file_and_un_tar() {
    wget https://github.com/NVIDIA/triton-inference-server/releases/download/v"${VERSION}"/"${FILENAME}"
    tar xzf "${FILENAME}"
}

mkdir -p ~/tmp
cd ~/tmp || return 1
mkdir -p tensorrtserver && cd tensorrtserver || return 1

# get package
if [ -f "${FILENAME}" ] ; then
  echo "Already downloaded at ${FILENAME}"
  tar xzf "${FILENAME}" || download_file_and_un_tar
else
    download_file_and_un_tar
fi

# install
pip install python/tensorrtserver-${VERSION}-py2.py3-none-linux_x86_64.whl
