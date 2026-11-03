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
        cd ./ci/pipeline/script && bash build_xpu_exporter.sh; \
    fi

FROM ubuntu AS xpu_exporter
COPY --link --from=build --chmod=500 /go/src/app/ci/output/xpu-exporter /opt/xpu/bin/xpu-exporter
RUN sed -i -e 's/^APT/# APT/' -e 's/^DPkg/# DPkg/' /etc/apt/apt.conf.d/docker-clean && \
    apt-get -o Acquire::AllowInsecureRepositories=true update && \
    apt-get install -y -f --no-install-recommends --allow-unauthenticated openssl libcap2-bin
RUN echo "xpu:x:10001:10001:eXPUPoolService:/:/sbin/nologin" >> /etc/passwd \
    && echo "xpu:x:10001:" >> /etc/group \
    && echo "xpu:!:::::::" >> /etc/shadow \
    && chown xpu:xpu /opt/xpu/bin/xpu-exporter \
    && setcap CAP_DAC_OVERRIDE=ep /opt/xpu/bin/xpu-exporter

USER xpu:xpu
ENTRYPOINT ["/opt/xpu/bin/xpu-exporter"]