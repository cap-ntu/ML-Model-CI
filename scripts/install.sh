#!/usr/bin/env bash

now=$(date +'%Y%m%d-%H%M%S')
log_path="/tmp/modelci-install-${now}.log"
FLAG_ERROR=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

function script_execution() {
    if [[ "${redirect}" == all ]] ; then
      bash "${scripts_path}" "$@" &>> "${log_path}"
    elif [[ "${redirect}" == stdout ]] ; then
      bash "${scripts_path}" "$@" >> "${log_path}"
    else
      bash "${scripts_path}" "$@"
    fi
}

function error_capture() {
  local scripts_path=$1 && shift
  local redirect="${1:-all}" && shift

  if script_execution "$@" ; then
    echo -e "${GREEN}OK${NC}"
  else
    echo -e "${RED}FAIL${NC}"
    FLAG_ERROR=true
  fi
}

function info_echo() {
  printf "${CYAN}%s${NC}" "$1"
}

# Change all line ending to LF
find scripts/ -type f -exec sed -i -e "s/^M$//" {} \;

# Install Conda environment
info_echo "Installing Conda environment..."
conda >> /dev/null || exit 1

error_capture scripts/install.conda_env.sh all

# Activate conda
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate modelci

# Install Triton client APIs
info_echo "Installing Triton client API..."
error_capture scripts/install.trtis_client.sh all

# Generating proto
info_echo "Generating gRPC code..."
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. modelci/types/proto/service.proto

# Pull docker images
info_echo "Pulling Docker images..."
error_capture scripts/install.pull_docker_images.sh none "${log_path}"

# Start service
info_echo "Starting services..."
error_capture scripts/install.start_service.sh noen "${log_path}"

if "${FLAG_ERROR}" = true ; then
  echo -e "${YELLOW}Some installation step has failed. Please see full log at ${log_path}."
fi
