#!/usr/bin/env source

# setup MongoDB env variables
# shellcheck disable=SC2046
export $(grep -v '^#' modelci/env-mongodb.env | xargs -d '\r\n')

# setup backend env variables
export $(grep -v '^#' modelci/env-backend.env | xargs -d '\r\n')

export PYTHONPATH="${PWD}"

# ref https://github.com/ContinuumIO/anaconda-issues/issues/11294
export KMP_INIT_AT_FORK=FALSE
