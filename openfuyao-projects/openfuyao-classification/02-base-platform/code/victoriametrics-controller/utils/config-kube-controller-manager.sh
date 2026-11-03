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

source ./log.sh

info_log "Start configuring KubeControllerManager"

MANIFEST_FILE="/etc/kubernetes/manifests/kube-controller-manager.yaml"

# 一次性完成所有修改
yq -e '
  (.spec.containers[0].command[] | select(. == "--bind-address=127.0.0.1")) = "--bind-address=0.0.0.0" |
  .spec.containers[0].command += ["--tls-cert-file=/etc/kubernetes/pki/kube-controller-manager.crt"] |
  .spec.containers[0].command += ["--tls-private-key-file=/etc/kubernetes/pki/kube-controller-manager.key"]
' -i "$MANIFEST_FILE"

if [ $? -ne 0 ]; then
    error_log "Failed to edit kube-controller-manager bind-address"
    exit 1
fi

info_log "Finish configuring KubeControllerManager"