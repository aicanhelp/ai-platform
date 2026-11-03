# syntax=docker/dockerfile:latest

#######################################################################
# Copyright (c) 2024 Bocloud Technologies Co., Ltd.
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain n copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#######################################################################

ARG BUILDER=golang
ARG BUILDER_VERSION=1.24.5
ARG BUILDER_IMAGE=cr.openfuyao.cn/openfuyao/builder/$BUILDER:$BUILDER_VERSION

ARG COMMIT
ARG VERSION
ARG TARGETPLATFORM
ARG SOURCE_DATE_EPOCH
ARG GOLDFLAGS="-X gopkg.openfuyao.cn/bkeagent/version.GitCommitID=$COMMIT -X gopkg.openfuyao.cn/bkeagent/version.Version=$VERSION -X gopkg.openfuyao.cn/bkeagent/version.Architecture=$TARGETPLATFORM -X gopkg.openfuyao.cn/bkeagent/version.BuildTime=$SOURCE_DATE_EPOCH"

FROM $BUILDER_IMAGE AS build

FROM scratch AS release
COPY --link --from=build --chmod=555 /go/bin/app ./bkeagent
ENTRYPOINT ["/bkeagent"]
