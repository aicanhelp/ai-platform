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
source ./consts.sh

INSTALLED_KUBEADM=""
INSTALLED_KUBECTL=""
INSTALLED_KUBELET=""
INSTALLED_CRI_TOOLS=""

hash -r
KUBECTL_VERSION="Client Version: v$VERSION"
if sudo which kubectl >/dev/null 2>&1; then
    info_log "kubectl has been installed"
    if [[ $(kubectl version) =~ $KUBECTL_VERSION ]]; then
        info_log "The installed version of kubectl is $VERSION"
    else
        error_log "kubectl version conflict"
        error_log "The installed version of kubectl is $(kubectl version)"
        exit 1
    fi
    INSTALLED_KUBECTL="true"
fi

if sudo which kubeadm >/dev/null 2>&1; then
    info_log "kubeadm has been installed"
    if [[ $(sudo kubeadm version) =~ $VERSION ]]; then
        info_log "The installed version of kubeadm is $VERSION"
    else
        error_log "kubeadm version conflict"
        error_log "The installed version of kubeadm is $(sudo kubeadm version)"
        exit 1
    fi
    INSTALLED_KUBEADM="true"
fi

if sudo which kubelet >/dev/null 2>&1; then
    info_log "kubelet has been installed"
    if [[ $(sudo kubelet --version) =~ $VERSION ]]; then
        info_log "The installed version of kubelet is $VERSION"
    else
        error_log "kubelet version conflict"
        error_log "The installed version of kubelet is $(sudo kubelet --version)"
        exit 1
    fi
    INSTALLED_KUBELET="true"
fi

if sudo which crictl >/dev/null 2>&1; then
    info_log "cri-tools has been installed"
    if [[ $(sudo crictl -v) =~ ${CRICTL_VERSION} ]]; then
        info_log "The installed version of cri-tools is $VERSION"
    else
        error_log "cri-tools version conflict"
        error_log "The installed version of cri-tools is $(sudo crictl -v)"
        exit 1
    fi
    INSTALLED_CRI_TOOLS="true"
fi

if [ -z "$INSTALLED_KUBELET" ] && [ -z "$INSTALLED_KUBEADM" ] && [ -z "$INSTALLED_KUBECTL" ] && [ -z "$INSTALLED_CRI_TOOLS" ]; then
    if kubectl get nodes &> /dev/null && kubectl get pods -n kube-system &> /dev/null; then
        info_log "kubernetes has running"
        exit 0
    fi
fi
