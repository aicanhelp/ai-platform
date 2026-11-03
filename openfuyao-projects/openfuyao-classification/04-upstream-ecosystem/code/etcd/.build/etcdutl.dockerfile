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
ARG BUILDER_IMAGE=cr.openfuyao.cn/openfuyao/builder/$BUILDER:$BUILDER_VERSION

ARG PKG=./server/main.go

# stage 1: build
FROM ${BUILDER_IMAGE} AS build
RUN cd ./etcdutl && \
    GOOS=$(go env GOOS) GOARCH=$(go env GOARCH) go build -trimpath -installsuffix=cgo -ldflags="all=-w -s -linkmode=external" -o /go/bin/etcdutl

# stage 2: release
FROM scratch AS release
ENV PATH="/:${PATH}"
COPY --link --from=build --chmod=555 /go/bin/etcdutl ./etcdutl
