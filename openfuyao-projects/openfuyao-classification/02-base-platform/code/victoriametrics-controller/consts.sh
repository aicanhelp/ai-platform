#!/bin/bash
###############################################################
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
###############################################################

# openFuyao官方harbor仓
FUYAO_RGISTRY="cr.openfuyao.cn/openfuyao"

# openFuyao新官方镜像仓库
OEPKGS_REGISTRY="hub.oepkgs.net/openfuyao"

# 下载的image存放路径
IMAGE_PATH="image"

# 下载的chart存放路径
CHART_PATH="chart"

# victoriametrics chart name
VMS_CHART_NAME="victoria-metrics-k8s-stack"

VM_CLUSTER_IMAGE_TAG="v1.122.0-cluster"

# vm 安装的命名空间
VM_NAMESPACE="vmks"

# grafana 额外面板目录
GRAFANA_ADD_PANEL_DIR="victoria-metrics-k8s-stack/files/dashboards"

# grafana 额外面板配置文件列表
PANEL_LIST=(
    kubernetes-views-nodes-static.yaml
)