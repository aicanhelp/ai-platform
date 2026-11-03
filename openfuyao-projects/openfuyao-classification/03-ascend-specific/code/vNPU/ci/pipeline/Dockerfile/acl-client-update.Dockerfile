# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# 默认值在本地构建时保障不会报错，流水线构建时会被正确传参覆盖
ARG BASE_IMAGE=cr.openfuyao.cn/openfuyao/golang:1.25.0
ARG LOCAL_ARTIFACTS=false

# stage 1: build
FROM $BASE_IMAGE AS build
ARG LOCAL_ARTIFACTS
USER 0
WORKDIR /go/src/app
COPY . .
# LOCAL_ARTIFACTS=true 时跳过编译，复用 build context 中 ./ci/output/ 的本地构建产物
RUN if [ "${LOCAL_ARTIFACTS}" = "true" ]; then \
        echo "local artifacts, skip build"; \
    else \
        cd ./ci/pipeline/script && . /usr/local/Ascend/cann-8.5.1/set_env.sh && bash build_acl_client.sh; \
    fi

FROM ubuntu AS release
COPY --link --from=build --chmod=500 /go/src/app/ci/output/* /root/
USER root
ENTRYPOINT ["/root/acl-client-update.sh"]
