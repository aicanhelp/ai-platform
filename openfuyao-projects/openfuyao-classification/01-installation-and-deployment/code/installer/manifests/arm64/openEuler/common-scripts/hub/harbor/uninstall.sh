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

source ../../utils/log.sh "harbor"
source ../../utils/consts.sh

if [ -d "${HARBOR_INSTALL_PATH}/harbor" ]; then
    cd ${HARBOR_INSTALL_PATH}/harbor
    nerdctl compose down
fi
info_log "remove harbor network"
sudo ip link set dev br-e7effeff0dcd down
sudo ip link delete br-e7effeff0dcd
NICS=$(sudo ls /sys/class/net/)
STR="veth"

for NIC in $NICS; do
    if [[ "$NIC" =~ $STR ]]; then
       sudo ip link set dev ${NIC} down
       sudo ip link delete ${NIC}
    fi
done

sudo ip link set dev nerdctl0 down
sudo ip link delete nerdctl0

# 删除harbor证书和文件
sudo rm -rf /var/lib/nerdctl/
sudo rm -rf "${HARBOR_CERT_PATH}"
sudo rm -f "/etc/ssl/certs/${HARBOR_CA_CRT}"

# 删除harbor数据
info_log "remove harbor data"
sudo rm -rf "${HARBOR_DATA_VOLUME}"

rm -rf ${HARBOR_INSTALL_PATH}

info_log "Successfully uninstalled harbor"
