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

ARG BUILDER_IMAGE=debian:trixie-slim
ARG BASE_IMAGE=alpine:3.20.3

FROM $BUILDER_IMAGE AS build
ARG COMMIT
ARG VERSION
ARG TARGETARCH
ARG SOURCE_DATE_EPOCH
RUN <<'EOF'
#!/bin/sh -xe
cat <<EOT > BUILD_INFO
🤯 Version=$VERSION
🤔 GitCommitId=$COMMIT
👉 Architecture=$TARGETARCH
⏲ BuildTime=$(date -u +'%FT%T.%3NZ' -d@$SOURCE_DATE_EPOCH)
EOT
EOF
FROM $BASE_IMAGE AS release
WORKDIR /workspace
COPY --link --from=build --chmod=444 /BUILD_INFO ./
COPY --link --chmod=444 kubernetes ./kubernetes
COPY --link --chmod=555 bkeinitscript ./bkeinitscript
ENTRYPOINT ["/bin/cat"]
