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

source ../../utils/log.sh "calico"
source ../../utils/consts.sh
source ./preinstall.sh
source ./common.sh

# 节点数量大于45个使用calico-typha.yaml
CALICO_WATERSHED=45

info_log "Start installing calico"

tar -xvf ./yaml/calico-yaml.tar.gz -C ./yaml
info_log "calico yaml files are extracted"

NODE_COUNT=$(kubectl get node | awk 'NR > 1' | wc -l)
if [ ${NODE_COUNT} -gt ${CALICO_WATERSHED} ]; then
     sudo sed -i "s|docker.io|{{ .registry }}|g" ./yaml/calico-yaml/calico-typha.yaml
    deal_with_calico_yaml ./yaml/calico-yaml/calico-typha.yaml 4993 4994 create
else
     sudo sed -i "s|docker.io|{{ .registry }}|g" ./yaml/calico-yaml/calico.yaml
    deal_with_calico_yaml ./yaml/calico-yaml/calico.yaml 4951 4952 create
fi
# 延迟5秒，防止执行下一操作安装失败
sleep 5

./postinstall.sh
