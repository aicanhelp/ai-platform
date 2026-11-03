#!/bin/bash

sudo mkdir /etc/buildkit/
sudo sh -c 'echo "
debug = true
[registry.\"deploy.bocloud.k8s:40443\"]
  http = true
" > /etc/buildkit/buildkitd.toml'


docker run --privileged --rm tonistiigi/binfmt --uninstall qemu-* && \
docker run --privileged --rm tonistiigi/binfmt --install all && \
docker buildx rm mybuilder || true && \
docker buildx create --use --name mybuilder --driver-opt image=deploy.bocloud.k8s:40443/buildkit:master --driver-opt network=host --config /etc/buildkit/buildkitd.toml && \
docker buildx inspect mybuilder --bootstrap && \
docker buildx ls