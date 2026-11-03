# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
ARG BUILDER_IMAGE=cr.openfuyao.cn/openfuyao/golang:1.25.0
ARG LOCAL_ARTIFACTS=false

# stage 1: build
FROM $BUILDER_IMAGE AS build
ARG LOCAL_ARTIFACTS
USER 0
WORKDIR /go/src/app
COPY . .
# LOCAL_ARTIFACTS=true 时跳过编译，复用 build context 中 ./ci/output/ 的本地构建产物
RUN if [ "${LOCAL_ARTIFACTS}" = "true" ]; then \
        echo "local artifacts, skip build"; \
    else \
        cd ./ci/pipeline/script && bash build_npu_device_plugin.sh; \
    fi

FROM ubuntu AS npu_device_plugin
COPY --link --from=build --chmod=500 /go/src/app/ci/output/npu-device-plugin /opt/xpu/bin/npu-device-plugin

# 运行时必须使用root用户，无法使用非root权限用户。
USER root:root
ENTRYPOINT ["/opt/xpu/bin/npu-device-plugin"]
