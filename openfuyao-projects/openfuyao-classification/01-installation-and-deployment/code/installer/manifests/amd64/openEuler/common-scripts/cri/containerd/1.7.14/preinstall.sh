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

source ../../utils/log.sh "containerd"
source ./consts.sh

if sudo which containerd >> "$OUTPUT"; then
    warning_log "containerd has been installed"

    if [[ "$(sudo containerd -v)" =~ $VERSION ]]; then
        info_log "The installed version of containerd is $VERSION"
    else
        error_log "containerd version conflict"
        fatal_log "$(sudo containerd -v)"
    fi

    CONTAINERD_SERVICE="/usr/local/lib/systemd/system/containerd.service"
    if [ -f "${CONTAINERD_SERVICE}" ]; then
        info_log "This computer is already installed containerd.service."
        # 检查containerd的运行状态
        if sudo systemctl is-active containerd >> "$OUTPUT"; then
            info_log "The status of the installed containerd is active"
        else
            sudo systemctl start containerd
        fi

        sudo systemctl status containerd | grep "active (running)" > /dev/null
        if [ "$?" == "0" ]; then
            info_log "The installed containerd is running normally"
        else
            error_log "The installed containerd runs abnormally"
            fatal_log "$(sudo systemctl status containerd)"
        fi
        exit 0
    else
        info_log "not install containerd.service."
    fi
fi
