#!/usr/bin/env bash
docker pull mongo
docker run --rm -d -p 27017:27017 --name modelci-mongo mongo
docker cp scripts/init_db.js modelci-mongo:/
docker exec modelci-mongo '/bin/bash' -c 'mongo < init_db.js'
