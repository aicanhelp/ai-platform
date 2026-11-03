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
    info_log "Completing the installation of calico"
fi


COUNT=1
IS_OK=0

while [ $COUNT -le 120 ]
do
    info_log "waiting times $COUNT"
    status_list=$(kubectl get pod -n kube-system | awk 'NR > 1 {print $3}')
    IS_OK=0

    while read -r status; do
        if [ "${status}" == "Running" ] || [ "${status}" == "Completed" ]; then
            info_log "${status}"
        else
            info_log "kube-system namespace pod status is abnormal"
            sleep 5
            IS_OK=1
            break
        fi
    done <<< "$status_list"

    if [ $IS_OK -eq 0 ]; then
        info_log "kube-system namespace pod status is normal"
        break
    fi

    COUNT=$((COUNT + 1))
done

if [ $COUNT -eq 120 ] && [ $IS_OK -ne 0 ]; then
    fatal_log "Exiting abnormally after exceeding the waiting time. Please check the pod status or network situation in the kube system namespace"
fi

function delete_pod() {
    local target_ip=$1
    local pod_name=$2

    local ip_list=$(kubectl get pod -n kube-system -o wide | grep "$pod_name" | awk '{print $6}')

    while read -r pod_ip; do
       ip_part=$(echo "$pod_ip" | awk -F'.' '{print $1"."$2"."$3}')

       if echo "$target_ip" | grep -q "$ip_part"; then
           info_log "delete pod, the pod ip is $pod_ip"
           kubectl delete pod "$(kubectl get pod -n kube-system -o wide | grep $pod_ip | awk '{print $1}')" -n kube-system
       else
           info_log "pod ip normal"
       fi
    done <<< "$ip_list"
}

IP=$(ip addr show nerdctl0 | grep -w inet | awk '{print $2}' | cut -d '/' -f 1)
if [ -n "$IP" ]; then
    info_log "has nerdctl0 interface and get ip is ${IP}"
    delete_pod "${IP}" calico-kube-controllers
    delete_pod "${IP}" coredns
else
    info_log "not found nerdctl0 or nerdctl0 not ip"
fi
