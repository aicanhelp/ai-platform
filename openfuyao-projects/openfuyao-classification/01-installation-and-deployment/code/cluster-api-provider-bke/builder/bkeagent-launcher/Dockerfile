# syntax=docker/dockerfile:latest

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
ARG BUILDER_VERSION=1.24.5
ARG BUILDER_IMAGE=cr.openfuyao.cn/openfuyao/builder/${BUILDER}:${BUILDER_VERSION}
ARG AGENT_VERSION=latest
ARG PKG=./cmd/bkeagent-launcher
ARG SRC=.

# stage 1: build
FROM ${BUILDER_IMAGE} AS build

# stage 2: agent
FROM cr.openfuyao.cn/openfuyao/bkeagent:${AGENT_VERSION} AS agent

# stage 3: final
FROM cgr.dev/chainguard/bash AS release

WORKDIR /workspace

COPY --link --from=build --chmod=555 /go/bin/app /bin/launcher
COPY --link --from=agent --chmod=555 /bkeagent ./bkeagent

CMD mkdir -p /etc/openFuyao/bkeagent/launcher
EXPOSE 8080