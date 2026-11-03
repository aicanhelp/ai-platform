# syntax=docker/dockerfile:1.11.1

#######################################################################
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#######################################################################

ARG BUILDER=golang
ARG BUILDER_VERSION=1.24.5-oe-24.03-lts-sp3
# Use the prebuilt openFuyao builder image to reduce build time.
ARG BUILDER_IMAGE=cr.openfuyao.cn/openfuyao/$BUILDER:$BUILDER_VERSION

# stage 1: build
FROM ${BUILDER_IMAGE} AS build
USER 0
WORKDIR /go/src/app

ARG GOPROXY="https://goproxy.cn,https://goproxy.io,https://proxy.golang.org,direct"
ENV GOPROXY="${GOPROXY}"

COPY go.mod go.sum ./

RUN --mount=type=cache,target=/go/pkg/mod,sharing=locked \
    go mod download

COPY . .

RUN --mount=type=cache,target=/go/pkg/mod,sharing=locked \
    --mount=type=cache,target=/root/.cache/go-build,sharing=locked \
    <<'EOF'
#!/bin/bash -xeu
export CGO_ENABLED=1
export CC=gcc
export CGO_CFLAGS="-Wall -fstack-protector-strong"
export CGO_LDFLAGS="-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none"
GOFLAGS="-buildvcs=false -buildmode=pie -trimpath -tags=osusergo,netgo -mod=readonly -modcacherw"
LDFLAGS='all=-w -s -buildid= -linkmode=external'
go build ${GOFLAGS} -ldflags="${LDFLAGS}" -o /home/nonroot/ub-network-bandwidth cmd/ub-network-bandwidth/main.go
EOF

# stage 2: release
FROM hub.oepkgs.net/openeuler/openeuler:24.03-lts-sp1 AS release
WORKDIR /
COPY --link --from=build --chmod=550 /home/nonroot/ub-network-bandwidth /usr/local/bin/ub-network-bandwidth
ENTRYPOINT ["/usr/local/bin/ub-network-bandwidth"]
