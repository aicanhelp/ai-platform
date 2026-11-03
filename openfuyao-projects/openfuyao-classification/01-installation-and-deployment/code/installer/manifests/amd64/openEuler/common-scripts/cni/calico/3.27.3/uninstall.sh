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
source ./common.sh

if ! kubectl get pod -n kube-system | grep -q "calico"; then
    info_log "calico has been uninstalled"
    exit 0
fi

CALICO_WATERSHED=45

info_log "Start uninstalling calico"

tar -xvf ./yaml/calico-yaml.tar.gz -C ./yaml
info_log "calico yaml files are extracted"

NODE_COUNT=$(kubectl get node | awk 'NR > 1' | wc -l)
if [ ${NODE_COUNT} -gt ${CALICO_WATERSHED} ]; then
    deal_with_calico_yaml ./yaml/calico-yaml/calico-typha.yaml 4993 4994 delete
else
    deal_with_calico_yaml ./yaml/calico-yaml/calico.yaml 4951 4952 delete
fi

sudo rm -rf /var/lib/calico
sudo rm -rf /opt/cni/bin/calico
sudo rm -rf /opt/cni/bin/flannel
sudo rm -rf /opt/cni/bin/calico-ipam

NICS=$(sudo ls /sys/class/net/)
STR="cali"

for NIC in $NICS; do
    if [[ "$NIC" =~ $STR ]]; then
       sudo ip link set dev ${NIC} down
       sudo ip link delete ${NIC}
    fi
done

info_log "Successfully uninstalled calico"