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

source ../../utils/log.sh "kubernetes"

if sudo which kubelet >/dev/null 2>&1; then
    info_log "Successfully installed kubelet"
else
    fatal_log "Failed to install kubelet"
fi

if sudo which kubectl >/dev/null 2>&1; then
    info_log "Successfully installed kubectl"
else
    fatal_log "Failed to install kubectl"
fi

if sudo which kubeadm >/dev/null 2>&1; then
    info_log "Successfully installed kubeadm"
else
    fatal_log "Failed to install kubeadm"
fi

if sudo which crictl >/dev/null 2>&1; then
    info_log "Successfully installed cri-tools"
else
    fatal_log "Failed to install cri-tools"
fi
