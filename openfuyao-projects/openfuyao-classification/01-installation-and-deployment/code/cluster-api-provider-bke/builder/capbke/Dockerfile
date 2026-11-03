# syntax=docker/dockerfile:latest

#######################################################################
# Copyright (c) 2024 Bocloud Technologies Co., Ltd.
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# cluster-api-provider-bke is licensed under Mulan PSL v2.
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
ARG BUILDER_IMAGE=cr.openfuyao.cn/openfuyao/builder/$BUILDER:$BUILDER_VERSION
ARG AGENT_VERSION=latest
ARG AGENT_IMAGE=cr.openfuyao.cn/openfuyao/bkeagent:$AGENT_VERSION

ARG COMMIT
ARG VERSION
ARG PLATFORMS
ARG TARGETPLATFORM
ARG SOURCE_DATE_EPOCH
ARG GOLDFLAGS="-X gopkg.openfuyao.cn/cluster-api-provider-bke/version.GitCommitID=$COMMIT -X gopkg.openfuyao.cn/cluster-api-provider-bke/version.Version=$VERSION -X gopkg.openfuyao.cn/cluster-api-provider-bke/version.Architecture=$TARGETPLATFORM -X gopkg.openfuyao.cn/cluster-api-provider-bke/version.BuildTime=$SOURCE_DATE_EPOCH"
ARG PKG=./cmd/capbke

FROM $BUILDER_IMAGE AS build

FROM --platform=linux/amd64 $AGENT_IMAGE AS agent_linux_amd64

FROM --platform=linux/arm64 $AGENT_IMAGE AS agent_linux_arm64

FROM cr.openfuyao.cn/openfuyao/base/static:latest AS release
COPY --link --from=build --chmod=555 /go/bin/app ./manager
COPY --link --from=agent_linux_amd64 --chmod=555 /bkeagent ./bkeagent_linux_amd64
COPY --link --from=agent_linux_arm64 --chmod=555 /bkeagent ./bkeagent_linux_arm64
COPY --link hack/bkeagent.service.tmpl ./
ENTRYPOINT ["/manager"]
