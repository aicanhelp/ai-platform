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

FROM alpine:3.20.0 AS final

RUN mkdir /home/openfuyao-system
RUN mkdir /home/openfuyao-system/arm64-bin
RUN mkdir /home/openfuyao-system/amd64-bin

RUN apk add --no-cache wget

RUN cd /home/openfuyao-system/arm64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/mikefarah/yq/releases/download/v4.43.1/yq_linux_arm64
RUN cd /home/openfuyao-system/amd64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/mikefarah/yq/releases/download/v4.43.1/yq_linux_amd64

RUN cd /home/openfuyao-system/arm64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/jqlang/jq/releases/download/v1.7.1/jq-linux-arm64
RUN cd /home/openfuyao-system/amd64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/jqlang/jq/releases/download/v1.7.1/jq-linux-amd64

RUN cd /home/openfuyao-system/arm64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/helm/releases/download/v3.14.2/helm-v3.14.2-linux-arm64.tar.gz
RUN cd /home/openfuyao-system/amd64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/helm/releases/download/v3.14.2/helm-v3.14.2-linux-amd64.tar.gz

RUN cd /home/openfuyao-system/arm64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/cloudflare/cfssl/releases/download/v1.6.4/cfssl-certinfo_1.6.4_linux_arm64
RUN mv /home/openfuyao-system/arm64-bin/cfssl-certinfo_1.6.4_linux_arm64 /home/openfuyao-system/arm64-bin/cfssl-certinfo_linux_arm64
RUN cd /home/openfuyao-system/amd64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/cloudflare/cfssl/releases/download/v1.6.4/cfssl-certinfo_1.6.4_linux_amd64
RUN mv /home/openfuyao-system/amd64-bin/cfssl-certinfo_1.6.4_linux_amd64 /home/openfuyao-system/amd64-bin/cfssl-certinfo_linux_amd64

RUN cd /home/openfuyao-system/arm64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/cloudflare/cfssl/releases/download/v1.6.4/cfssl_1.6.4_linux_arm64
RUN mv /home/openfuyao-system/arm64-bin/cfssl_1.6.4_linux_arm64 /home/openfuyao-system/arm64-bin/cfssl_linux_arm64
RUN cd /home/openfuyao-system/amd64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/cloudflare/cfssl/releases/download/v1.6.4/cfssl_1.6.4_linux_amd64
RUN mv /home/openfuyao-system/amd64-bin/cfssl_1.6.4_linux_amd64 /home/openfuyao-system/amd64-bin/cfssl_linux_amd64

RUN cd /home/openfuyao-system/arm64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/cloudflare/cfssl/releases/download/v1.6.4/cfssljson_1.6.4_linux_arm64
RUN mv /home/openfuyao-system/arm64-bin/cfssljson_1.6.4_linux_arm64 /home/openfuyao-system/arm64-bin/cfssljson_linux_arm64
RUN cd /home/openfuyao-system/amd64-bin && wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/cloudflare/cfssl/releases/download/v1.6.4/cfssljson_1.6.4_linux_amd64
RUN mv /home/openfuyao-system/amd64-bin/cfssljson_1.6.4_linux_amd64 /home/openfuyao-system/amd64-bin/cfssljson_linux_amd64

COPY consts.sh /home/openfuyao-system/
COPY entrypoint.sh /home/openfuyao-system/
COPY install.sh /home/openfuyao-system/
COPY uninstall.sh /home/openfuyao-system/
COPY utils.sh /home/openfuyao-system/
COPY log.sh /home/openfuyao-system/
COPY preinstall.sh /home/openfuyao-system/
COPY postinstall.sh /home/openfuyao-system/


RUN mkdir /home/openfuyao-system/resource
COPY resource/ /home/openfuyao-system/resource/

RUN chmod +x /home/openfuyao-system/*.sh

USER root

CMD ["sh"]