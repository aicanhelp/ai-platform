# syntax=docker/dockerfile:1.11.1

#######################################################################
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#######################################################################

ARG BUILDER=golang
ARG BUILDER_VERSION=1.25.0
ARG BUILDER_IMAGE=cr.openfuyao.cn/openfuyao/$BUILDER:$BUILDER_VERSION

# stage 1: build
FROM ${BUILDER_IMAGE} AS build

ENV GOPROXY=${GOPROXY:-https://goproxy.cn,https://goproxy.io,proxy.golang.org,direct}

WORKDIR ${GOPATH}/src/app

COPY . .

RUN <<'EOF'
#!/bin/bash -xeu
export GITCOMMIT=$(git describe --dirty --always)
CGO_ENABLED=0 go build -v -ldflags="-s -w -X github.com/coredns/coredns/coremain.GitCommit=$GITCOMMIT" -o /home/nonroot/coredns
EOF

# stage 2: release
FROM scratch AS release
ENV PATH="/:${PATH}"
COPY --link --from=build --chmod=555 /home/nonroot/coredns ./coredns
EntryPoint ["/coredns"]