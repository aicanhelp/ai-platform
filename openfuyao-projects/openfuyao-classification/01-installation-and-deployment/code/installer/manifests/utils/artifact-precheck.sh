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

hash -r

# 判断是否安装containerd，同时获取containerd的版本
CONTAINERD_INSTALLED=0
if command -v containerd > /dev/null; then

    # 检查containerd的运行状态
    CONTAINERD_SERVICE="/usr/local/lib/systemd/system/containerd.service"
    if [ -f "${CONTAINERD_SERVICE}" ]; then
        if systemctl is-active containerd > /dev/null; then
            echo "containerd active"
        else
            systemctl start containerd
        fi

        sudo systemctl status containerd | grep "active (running)" > /dev/null
        if [ "$?" == "0" ]; then
            CONTAINERD_VERSION=`containerd -v | awk '{print $3}' | sed 's/^v//'`
            CONTAINERD_INSTALLED=1
        else
            exit 1
        fi
    fi
fi

# 判断runc
RUNC_INSTALLED=0
if command -v runc >/dev/null 2>&1; then
    RUNC_VERSION=`runc -v | awk 'NR<2 {print $3}'`
    RUNC_INSTALLED=1
fi

# 判断nerdctl
NERDCT_INSTALLED=0
if command -v nerdctl >/dev/null 2>&1; then
    NERDCTL_VERSION=`nerdctl -v | awk '{print $3}'`
    NERDCT_INSTALLED=1
fi

# 判断cni-plugins
CNI_PLUGINS_INSTALLED=0
if [ -d "/etc/cni/bin" ]; then
    if [ -f "/etc/cni/bin/ipvlan" ] && [ -f "/etc/cni/bin/bandwidth" ]; then
        CNI_PLUGINS_INSTALLED=1
    fi
fi

# 判断helm
HELM_INSTALLED=0
if command -v helm >/dev/null 2>&1; then
    HELM_INSTALLED=1
fi

json="{
  \"containerd\": {\"version\": \"${CONTAINERD_VERSION}\", \"installed\": \"${CONTAINERD_INSTALLED}\"},
  \"runc\": {\"version\": \"${RUNC_VERSION}\", \"installed\": \"${RUNC_INSTALLED}\"},
  \"cni-plugins\": {\"version\": \"NA\", \"installed\": \"${CNI_PLUGINS_INSTALLED}\"},
  \"helm\": {\"version\": \"NA\", \"installed\": \"${HELM_INSTALLED}\"},
  \"nerdctl\": {\"version\": \"${NERDCTL_VERSION}\", \"installed\": \"${NERDCT_INSTALLED}\"}
}"

# 将JSON内容输出到文件中
echo "$json" > artifact-precheck-res.json
