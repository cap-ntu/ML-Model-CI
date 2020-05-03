#!/usr/bin/env bash

# setup env variables
set -o allexport
source modelci/env-mongodb.env
set +o allexport

export PYTHONPATH="${PWD}"
