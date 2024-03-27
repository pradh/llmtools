#!/bin/bash

mkdir -p /tmp/redis-install

curl -fsSL https://packages.redis.io/redis-stack/redis-stack-server-7.2.0-v9.monterey.arm64.zip -o /tmp/redis-install/redis-stack-server.tar.gz
cd /tmp/redis-install
tar -xvf redis-stack-server.tar.gz
./bin/redis-stack-server --daemonize yes

