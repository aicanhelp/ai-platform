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

FROM golang:1.24 AS builder
ARG TARGETOS
ARG TARGETARCH
ARG BUILDER_IMAGE=cr.openfuyao.cn/openfuyao/builder/${BUILDER}:${BUILDER_VERSION}
ARG ADM_VERSION=latest

WORKDIR /workspace

COPY . .


RUN go env -w GOPRIVATE=gitcode.com,gopkg.openfuyao.cn # 在此阶段设置 GOPRIVATE

RUN CGO_ENABLED=0 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH} go build -a -o installer-service ./cmd

# stage 2: adm - 不要显式设置 platform，Docker 会自动选择
FROM cr.openfuyao.cn/openfuyao/bkeadm:${ADM_VERSION} AS adm

FROM alpine:3.20.0

RUN apk add --no-cache git

# 这里不需要再设置 GOPRIVATE，因为它只需要在 builder 阶段
WORKDIR /

#COPY --link --from=builder/installer-service /workspace /usr/local/bin/installer-service
COPY --link --from=builder /workspace/installer-service /usr/local/bin/installer-service
COPY --link --from=adm --chmod=555 /bkeadm /usr/local/bin/bke

EXPOSE 9210

ENTRYPOINT ["/usr/local/bin/installer-service"]
