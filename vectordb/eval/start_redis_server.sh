#!/bin/bash

mkdir -p redis-install

curl -fsSL https://packages.redis.io/redis-stack/redis-stack-server-7.2.0-v9.monterey.arm64.zip -o redis-install/redis-stack-server.tar.gz
cd redis-install
tar -xvf redis-stack-server.tar.gz
./bin/redis-stack-server --daemonize yes

