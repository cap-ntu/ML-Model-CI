#!/bin/bash
echo "$@"

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
            >&2 echo "$1 is not a recognized flag!"
            exit 1;
            ;;
    esac
done  

gpu=${gpu:-false}
name=${name:-cadvisor}
port=${port:-8080}
env=${env:-env-file/advisor.env}

# source environment
# shellcheck disable=SC2046
export $(grep -v '^#' "${env}" | xargs -d '\r\n')

printf "parameters: \n\n"
echo 'is_gpu_support: ' "${gpu}"
echo 'port: ' "${port}"
echo 'name: ' "${name}"
echo 'env: ' "${env}"
printf "\n\n"

if $gpu ; then
    echo 'starting GPU supported cAdvisor...'
    docker run -e "${env}" -d --rm --name="${name}" --privileged \
      -v="${LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}" \
      -v=/:/rootfs:ro --volume=/var/run:/var/run:rw \
      -v=/sys:/sys:ro --volume=/var/lib/docker/:/var/lib/docker:ro \
      -p="${port}:${port}" \
      google/cadvisor:latest || exit 1
else
    echo 'starting CPU only cAdvisor...'
    docker run -d --rm \
      -v=/:/rootfs:ro \
      -v=/var/run:/var/run:rw \
      -v=/sys:/sys:ro \
      -v=/var/lib/docker/:/var/lib/docker:ro \
      -p="${port}:${port}" \
      google/cadvisor:latest || exit 1
fi

echo "finished"