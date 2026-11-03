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

# used for gitcode pipeline

ARG BUILDER=golang
ARG BUILDER_VERSION=1.25.10
ARG BUILDER_IMAGE=cr.openfuyao.cn/openfuyao/builder/$BUILDER:$BUILDER_VERSION
ARG CGO_ENABLED=1
ARG PKG=./cmd/ub-network-device-plugin

# stage 1: build
FROM $BUILDER_IMAGE AS build

# stage 2: release
FROM hub.oepkgs.net/openeuler/openeuler:24.03-lts-sp3 AS release

WORKDIR /

COPY --link --from=build /go/bin/app ub-network-device-plugin

RUN chmod +x ub-network-device-plugin

ENTRYPOINT [ "./ub-network-device-plugin" ]
