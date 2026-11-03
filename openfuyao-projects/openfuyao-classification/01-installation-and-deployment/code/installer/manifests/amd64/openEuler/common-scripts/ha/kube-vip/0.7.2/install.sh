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

source ../../utils/log.sh "kube-vip"
source ./consts.sh
source ./preinstall.sh

YAML_PATH="/etc/kubernetes/manifests/kube-vip.yaml"

function install_kube_vip () {
    sudo ctr image pull {{ .registry }}/kube-vip/kube-vip:v$VERSION
    sudo mkdir -p /etc/kubernetes/manifests/

    sudo ctr run --rm --net-host {{ .registry }}/kube-vip/kube-vip:v$VERSION vip /kube-vip manifest pod \
            --interface {{ .interface }} \
            --vip {{ .vip }} \
            --controlplane \
            --arp \
            --leaderElection | tee $YAML_PATH
}

install_kube_vip
# 离线部署时替换镜像仓地址
sudo sed -i "s|ghcr.io/kube-vip/kube-vip:v$VERSION|{{ .registry }}/kube-vip/kube-vip:v$VERSION|" ${YAML_PATH}
./postinstall.sh
