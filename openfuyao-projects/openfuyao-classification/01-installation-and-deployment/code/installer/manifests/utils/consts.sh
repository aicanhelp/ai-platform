#!/bin/bash
###############################################################
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
###############################################################

# openFuyao官方helm chart仓
FUYAO_REPO="https://harbor.openfuyao.com/chartrepo/openfuyao"
# openFuyao官方harbor仓
FUYAO_RGISTRY="harbor.openfuyao.com/openfuyao"
# openFuyao obs 根路由
FUYAO_OBS="https://openfuyao.obs.cn-north-4.myhuaweicloud.com"

OPENFUYAO="openFuyao"

BUSY_BOX_IMAGE_TAG="1.36.1"
OPENFUYAO_IMAGE_TAG="latest"
OPENFUYAO_CHART_VERSION="0.0.0-latest"

LOCAL_HARBOR_IMAGE_TAG="v2.7.0"
LOCAL_HARBOR_CHART_VERSION="1.11.4"

# 证书存放路径
BASE_CERT_PATH="/etc/ssl/certs"
# harbor证书存放路径
HARBOR_CERT_PATH="/etc/ssl/certs/openfuyao-harbor"
FUYAO_SYSTEM_CERT_PATH="/etc/ssl/certs/openfuyao-fuyao-system"
# harbor证书名称
HARBOR_CA_KEY="openfuyao-harbor-ca.key"
HARBOR_CA_CRT="openfuyao-harbor-ca.crt"
# harbor data volume
HARBOR_DATA_VOLUME="/data/openfuyao-harbor"
# harbor安装路径
HARBOR_INSTALL_PATH="/opt/openfuyao-offline-harbor"

# 导出的image存放路径
IMAGE_PATH="image"
# 下载的chart存放路径
CHART_PATH="chart"

# 离线部署使用harbor版本
HARBOR_VERSION="2.7.0"

# ingress-nginx 命名空间
INGRESS_NGINX_NAMESPACE="ingress-nginx"
