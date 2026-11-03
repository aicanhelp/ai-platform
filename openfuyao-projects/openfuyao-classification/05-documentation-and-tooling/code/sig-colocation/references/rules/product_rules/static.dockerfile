# syntax=docker/dockerfile:latest

#######################################################################
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#######################################################################

ARG GO_VERSION=1.21.8
ARG GO_IMAGE=harbor.openfuyao.com/openfuyao/golang:${GO_VERSION}

# stage 1: build
FROM ${GO_IMAGE} AS build

# change this to the actual context directory, relative to repository root
ARG SRC=.
# change this to the actual package to build, relative to ${SRC}
ARG PKG=./...

WORKDIR /go/src/app
COPY ../${SRC} .

ARG GOPROXY=https://goproxy.io,direct
ARG CGO_ENABLED=1
ARG GOFLAGS='-buildvcs=false -buildmode=pie -tags=osusergo,netgo "-ldflags=-w -s -linkmode=external '"'"'-extldflags=-Wl,-z,now -static-pie'"'"'"'
ARG TARGETOS
ARG TARGETARCH

ENV GOOS=${TARGETOS}
ENV GOARCH=${TARGETARCH}

RUN --mount=type=cache,target=/go/pkg/mod \
    go build -o /go/bin/app ${PKG}

# stage 2: final
FROM scratch AS final

WORKDIR /

COPY --link --from=build /usr/share/zoneinfo /usr/share/zoneinfo
COPY --link --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --link --from=build --chmod=755 /go/bin/app entrypoint

ENTRYPOINT [ "./entrypoint" ]
