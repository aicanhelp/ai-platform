#!/bin/bash

# 检查容器是否已存在
if docker ps -a --format "table {{.Names}}" | grep -q "^py3.12-dev$"; then
    echo "容器 py3.12-dev 已存在，直接连接到容器..."
else
    echo "创建新的 py3.12-dev 容器..."
    docker run -d --name py3.12-dev \
      -v $(pwd):/app \
      -w /app \
      python:3.12-slim \
      tail -f /dev/null
fi

# 连接到容器
docker exec -it py3.12-dev -- bash