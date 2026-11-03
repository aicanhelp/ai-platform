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
ARG BUILDER_VERSION=1.24.5
ARG BUILDER_IMAGE=cr.openfuyao.cn/openfuyao/$BUILDER:$BUILDER_VERSION
ARG KUBE_BASE_IMAGE=registry.k8s.io/build-image/go-runner:v2.4.0-go1.24.2-bookworm.0

# stage 1: build
FROM ${BUILDER_IMAGE} AS build
USER 0
WORKDIR /go/src/app
COPY . .

ARG GOLDFLAGS
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked --mount=type=cache,target=/var/cache/apt,sharing=shared <<'EOF'
#!/bin/bash -xeu
apt-get update
apt-get install -y --no-install-recommends make rsync git

GOFLAGS="-mod=mod -trimpath"
go mod download
make WHAT="cmd/kube-scheduler vendor/k8s.io/component-base/logs/fy-kube-log-runner" GOLDFLAGS="${GOLDFLAGS}" GOFLAGS="${GOFLAGS}"
EOF

# stage 2: release
FROM scratch AS releas
COPY --link --from=build --chmod=555 /go/src/app/_output/local/go/bin/kube-scheduler /usr/local/bin/kube-scheduler
COPY --link --from=build --chmod=555 /go/src/app/_output/local/go/bin/fy-kube-log-runner /usr/local/bin/kube-log-runner