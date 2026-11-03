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

source ../../utils/log.sh "ascend-deployer"

WORKER_INFO="{{ .workerInfo }}"
MASTER_INFO="{{ .masterInfo }}"

info_log "install essential plugins for ascend ai"


###  unzip ascend-deployer
ARCH=$(uname -m)
case $ARCH in
    x86_64) ARCH_STANDARD="amd64";;
    aarch64) ARCH_STANDARD="arm64";;
esac

tar -zxf $ARCH_STANDARD-bin/ascend-deployer.tar.gz
inventory_file="ascend-deployer/ascend_deployer/inventory_file"

# 遍历输入数据中的每对用户名和密码
IFS=',' read -ra pairs <<< "$MASTER_INFO"
for pair in "${pairs[@]}"; do
  IFS=':' read -r serverip password <<< "$pair"
  # 使用 sed 替换文件中的用户名和密码
  sed -i "s/serverip/${serverip}/g" "$inventory_file"
  sed -i "s/password/${password}/g" "$inventory_file"
done


if [ -f "ascend-deployer/ascend_deployer/install.sh" ]; then
    echo "y"|bash ./ascend-deployer/ascend_deployer/install.sh --install-scene=dl --skip_check
fi