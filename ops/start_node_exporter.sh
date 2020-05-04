#!/bin/bash

DIR="node_exporter-0.18.1.linux-amd64"

if [ -d "$DIR" ]; then
    echo "${DIR} has been dowloaded "
else
    echo "Start to dowload ${DIR} "
    wget https://github.com/prometheus/node_exporter/releases/download/v0.18.1/node_exporter-0.18.1.linux-amd64.tar.gz
    tar xvfz node_exporter-0.18.1.linux-amd64.tar.gz
fi

cd node_exporter-0.18.1.linux-amd64

./node_exporter