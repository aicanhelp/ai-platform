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

FROM golang:1.24.5 AS builder
ARG TARGETOS
ARG TARGETARCH

WORKDIR /workspace

COPY . .

RUN CGO_ENABLED=0 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH} go build -a -o kae-operator ./cmd

FROM alpine:3.20.0
WORKDIR /

COPY --from=builder /workspace/kae-operator /usr/local/bin/

RUN mkdir -p /opt/kae-operator/crds
COPY assets /opt/kae-operator/

# Install dependency: `kubectl`
RUN apk add --no-cache kubectl

# Add CRD resource into the image for helm upgrades
COPY charts/kae-operator/crds/device.openfuyao.com_kaepolicies_crd.yaml /opt/kae-operator/crds/device.openfuyao.com_kaepolicies_crd.yaml
COPY charts/kae-operator/charts/node-feature-discovery/crds/nfd-api-crds.yaml /opt/kae-operator/crds/nfd-api-crds.yaml

ENTRYPOINT ["/usr/local/bin/kae-operator"]
