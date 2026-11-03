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
RUN if [ "${LOCAL_ARTIFACTS}" != "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends patch && rm -rf /var/lib/apt/lists/*; \
    fi
COPY . .
# LOCAL_ARTIFACTS=true 时跳过编译，复用 build context 中 ./ci/output/ 的本地构建产物
RUN if [ "${LOCAL_ARTIFACTS}" = "true" ]; then \
        echo "local artifacts, skip build"; \
    else \
        cd ./ci/pipeline/script && bash build_vc_webhook_manager.sh; \
    fi

FROM ubuntu AS vc-webhook-manager
COPY --link --from=build --chown=1000:1000 --chmod=500 /go/src/app/ci/output/vc-webhook-manager /vc-webhook-manager
COPY --link --from=build --chown=1000:1000 --chmod=500 /go/src/app/ci/output/kubectl /usr/local/bin/kubectl
COPY --link --from=build --chown=1000:1000 --chmod=500 /go/src/app/ci/output/gen-admission-secret.sh /gen-admission-secret.sh

RUN sed -i -e 's/^APT/# APT/' -e 's/^DPkg/# DPkg/' /etc/apt/apt.conf.d/docker-clean && \
    apt-get -o Acquire::AllowInsecureRepositories=true update && \
    apt-get install -y -f --no-install-recommends --allow-unauthenticated openssl
RUN chmod +x gen-admission-secret.sh && chmod +x /usr/local/bin/kubectl

ENTRYPOINT ["/vc-webhook-manager"]