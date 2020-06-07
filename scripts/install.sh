#!/usr/bin/env bash

NOW=$(date +'%Y%m%d-%H%M%S')
LOG_PATH="/tmp/modelci-${NOW}.log"
FLAG_ERROR=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

OK_MESSAGE="${GREEN}OK${NC}"
FAIL_MESSAGE="${RED}FAIL${NC}"

# Install Conda environment
printf "${CYAN}Installing Conda environment...${NC}"
if bash scripts/install_conda_env.sh &> "${LOG_PATH}" ; then
  echo -e "${OK_MESSAGE}"
else
  echo -e "${FAIL_MESSAGE}"
  FLAG_ERROR=true
fi

# Activate conda
source "${HOME}"/anaconda3/etc/profile.d/conda.sh
conda activate modelci

# Install TRTIS client APIs
printf "${CYAN}Installing TRTIS client API...${NC}"
if bash scripts/install_trtis_client.sh &>> "${LOG_PATH}" ; then
  echo -e "${OK_MESSAGE}"
else
  echo -e "${FAIL_MESSAGE}"
  FLAG_ERROR=true
fi

# Start MongoDB service
printf "${CYAN}Starting services...${NC}"
if bash scripts/start_service.sh &>> "${LOG_PATH}" ; then
  echo -e "${OK_MESSAGE}"
else
  echo -e "${FAIL_MESSAGE}"
  FLAG_ERROR=true
fi

if "${FLAG_ERROR}" = true ; then
  echo -e "${YELLOW}Some installation step has failed. Please see full log at ${LOG_PATH}."
fi
