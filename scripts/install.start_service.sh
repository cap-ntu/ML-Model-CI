#!/usr/bin/env bash

# Get log path
NOW=$(date +'%Y%m%d-%H%M%S')
log_path="${1:-/tmp/modelci-install-${NOW}.log}"
flag_error=false

RED='\033[0;31m'
NC='\033[0m'

function error_caption() {
  local scripts_path=$1
  shift

  if bash "${scripts_path}" "$@" &>> "${log_path}" ; then
    printf '.'
  else
    # shellcheck disable=SC2059
    printf "${RED}x${NC}"
    flag_error=true
  fi
}

# start mongo db service
error_caption scripts/install.start_service.mongo.sh

# start node exporter
error_caption scripts/install.start_service.node_exporter.sh

# start cAdvisor
error_caption scripts/install.start_service.advisor.sh --gpu --name modelci.cadvisor

if "${flag_error}" = true ; then
    exit 1
fi
