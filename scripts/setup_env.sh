#!/usr/bin/env source

# setup env variables
# shellcheck disable=SC2046
export $(grep -v '^#' modelci/env-mongodb.env | xargs -d '\r\n')

export PYTHONPATH="${PWD}"
