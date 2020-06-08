#!/bin/bash

VERSION=0.18.1
DIR_NAME="node_exporter-${VERSION}.linux-amd64"
FILENAME="${DIR_NAME}.tar.gz"

# create temporary directory
mkdir -p ~/tmp && cd ~/tmp || exit 1
mkdir -p node_exporter && cd node_exporter || exit 1


# download and unzip
if [ -d "${DIR_NAME}" ]; then
    echo "${DIR_NAME} has been downloaded"
else
    echo "Start to download ${DIR_NAME}"
    wget "https://github.com/prometheus/node_exporter/releases/download/v${VERSION}/${FILENAME}"
    tar xvfz "${FILENAME}"
fi

# install
cd ${DIR_NAME} || exit
./node_exporter &
