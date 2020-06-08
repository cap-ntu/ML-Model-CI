#!/usr/bin/env bash

kill -9 "$(lsof -t -i:9100)"
