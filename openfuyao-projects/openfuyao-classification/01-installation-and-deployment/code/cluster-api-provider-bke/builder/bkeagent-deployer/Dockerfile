# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#######################################################################

ARG AGENT_VERSION=latest
ARG AGENT_IMAGE=cr.openfuyao.cn/openfuyao/bkeagent:${AGENT_VERSION}

# =============== Stage 1: 提取 amd64 二进制 ===============
FROM --platform=linux/amd64 ${AGENT_IMAGE} AS agent_linux_amd64

# =============== Stage 2: 提取 arm64 二进制 ===============
FROM --platform=linux/arm64 ${AGENT_IMAGE} AS agent_linux_arm64

# =============== Stage 3: 最终 deployer 镜像（使用私仓 alpine） ===============
FROM alpine:3.18

# 安装必要依赖（包括 openssl、net-tools 和 socat 用于 HTTPS 健康检查）
RUN apk add --no-cache bash openssl net-tools socat

WORKDIR /artifacts

# 从两个提取 stage 拷贝对应架构的二进制
COPY --from=agent_linux_amd64 /bkeagent /artifacts/bkeagent_linux_amd64
COPY --from=agent_linux_arm64 /bkeagent /artifacts/bkeagent_linux_arm64

# 复制入口脚本（从项目根目录运行时的路径）
COPY builder/bkeagent-deployer/entrypoint.sh /entrypoint.sh

# 赋予执行权限
RUN chmod +x /entrypoint.sh /artifacts/bkeagent_*

ENTRYPOINT ["/entrypoint.sh"]