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
ARG OE_VERSION=22.03
ARG OE_IMAGE=openeuler/openeuler:${OE_VERSION}

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
ARG GOFLAGS='-buildvcs=false -buildmode=pie "-ldflags=-w -s -linkmode=external '"'"'-extldflags=-Wl,-z,now -pie'"'"'"'
ARG TARGETOS
ARG TARGETARCH

ENV GOOS=${TARGETOS}
ENV GOARCH=${TARGETARCH}

RUN --mount=type=cache,target=/go/pkg/mod \
    go build -o /go/bin/app ${PKG}

# stage 2: final
FROM ${OE_IMAGE} AS final

ARG USER=app
ARG HOME=/home/${USER}

WORKDIR ${HOME}

# execute the following instructions with root permission
USER 0

RUN chmod g=u "${HOME}" /etc/passwd

COPY --link --from=build --chmod=755 /go/bin/app go_entrypoint
COPY --link --chmod=755 <<EOF entrypoint
#!/bin/sh -e
if ! whoami > /dev/null 2>& 1; then
    if [ -w /etc/passwd ]; then
        echo "${USER}:x:\$(id -u):0::${HOME}:/sbin/nologin" >> /etc/passwd
    fi
fi
exec "\$@"
EOF

# default to a non-root user in runtime
USER 65532

ENTRYPOINT [ "./entrypoint" ]
CMD [ "go_entrypoint" ]
