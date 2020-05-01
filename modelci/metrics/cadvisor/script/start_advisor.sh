#!/bin/bash

while test $# -gt 0; do
    case "$1" in
        -h|--help)
            echo "start_advisor.sh - attempt to start cAdvisor Docker container"
            echo " "
            echo "sh start_advisor.sh [options] [arguments]"
            echo " "
            echo "options:"
            echo "-h, --help                                show brief help"
            echo "-gpu, --gpu=false                         enable the GPU support"
            echo "-p, --port=8080                           specify the data publish port"
            echo "-n, --name=cadvisor                       specify the container's name"
            echo "-e, --env=/usr/lib/x86_64-linux-gnu       specify the LD_LIBRARY_PATH, only GPU supported"
            exit 0
            ;;
        -gpu|--gpu)
            shift
            gpu=true
            ;;
        -p|--port)
            shift
            port=$1
            shift
            ;;
        -n|--name)
            shift
            name=$1
            shift
            ;;
        -e|--env)
            shift
            env=$1
            shift
            ;;
        *)
            echo "$1 is not a recognized flag!"
            return 1;
            ;;
    esac
done  

gpu=${gpu:=false}
name=${name:=cadvisor}
port=${port:=8080}
env=${env:=/usr/lib/x86_64-linux-gnu}

echo "parameters: \n"
echo 'is_gpu_support: ' $gpu
echo 'port: ' $port
echo 'name: ' $name
echo 'env: ' $env
echo "\n"

if $gpu ; then
    echo 'starting GPU supported cAdvisor...'
    docker run -e LD_LIBRARY_PATH=$env --volume=$env:$env --volume=/:/rootfs:ro --volume=/var/run:/var/run:rw --volume=/sys:/sys:ro --volume=/var/lib/docker/:/var/lib/docker:ro --publish=$port:$port --detach=true --name=$name --privileged google/cadvisor:latest
else
    echo 'starting CPU only cAdvisor...'
    docker run --volume=/:/rootfs:ro --volume=/var/run:/var/run:rw --volume=/sys:/sys:ro --volume=/var/lib/docker/:/var/lib/docker:ro --publish=$port:$port --detach=true --name=$name google/cadvisor:latest
fi

echo "finished"