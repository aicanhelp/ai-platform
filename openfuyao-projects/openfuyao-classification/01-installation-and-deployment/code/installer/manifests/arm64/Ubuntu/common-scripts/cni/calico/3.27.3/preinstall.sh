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

if kubectl get pod -n kube-system | grep -q "calico"; then
    warning_log "calico may be installed"
    PODS=(
    calico-node
    calico-kube-controllers
    )
    for pod in ${PODS[@]}; do
        status_list=$(kubectl get pod -n kube-system | grep ${pod} | awk '{print $3}')
        for status in ${status_list[@]}; do
            if [ ${status} != "Running" ]; then
                error_log "calico has been installed, but some pod status is abnormal"
                fatal_log "$(kubectl get pod -n kube-system | grep ${pod})"
            fi
        done
    done
    exit 0
fi