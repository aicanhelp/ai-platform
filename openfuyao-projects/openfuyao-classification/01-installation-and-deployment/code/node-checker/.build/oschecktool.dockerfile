# syntax=docker/dockerfile:1.11.1

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

ARG BUILDER=golang
ARG BUILDER_VERSION=1.25.0
ARG BUILDER_IMAGE=cr.openfuyao.cn/openfuyao/$BUILDER:$BUILDER_VERSION

ARG PKG=./

# stage 1: build
FROM $BUILDER_IMAGE AS build

ENV GOPRIVATE=gopkg.openfuyao.cn
ENV GOPROXY=${GOPROXY:-https://goproxy.cn,https://goproxy.io,proxy.golang.org,direct}

WORKDIR /oschecktool

COPY . .

# 运行同目录下的build.sh，并将生成的的文件复制到根目录
RUN chmod +x .build/build.sh
RUN .build/build.sh

# stage 2: release
FROM scratch AS release

COPY --link --from=build --chmod=555 /oschecktool/output/*.tar.gz /



