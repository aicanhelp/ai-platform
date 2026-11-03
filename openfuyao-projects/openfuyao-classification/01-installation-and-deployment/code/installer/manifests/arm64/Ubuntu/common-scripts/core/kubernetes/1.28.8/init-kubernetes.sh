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
# 初始化kubernetes

source ../../utils/log.sh "init"

sudo kubectl get node > /dev/null
if [ "$?" == "0" ]; then
    info_log "kubernetes has been initialized"
else
    sudo kubeadm config print init-defaults --component-configs KubeProxyConfiguration > kubeadm.yaml
    sudo chmod +x kubeadm.yaml

    sudo sed -i "s|advertiseAddress: 1.2.3.4|advertiseAddress: {{ .host }}|" kubeadm.yaml
    sudo sed -i "s|bindPort: 6443|bindPort: {{ .apiserverBindPort }}|" kubeadm.yaml
    sudo sed -i "s|name: node|name: {{ .hostname }}|" kubeadm.yaml
    sudo sed -i "s|kubernetesVersion: 1.28.0|kubernetesVersion: {{ .kubernetes_version }}|" kubeadm.yaml
    sudo sed -i 's|metricsBindAddress: ""|metricsBindAddress: "{{ .metricsBindAddress }}"|' kubeadm.yaml
    sudo sed -i "s|imageRepository: registry.k8s.io|imageRepository: {{ .registry }}/kubernetes|" kubeadm.yaml
    sudo sed -i 's|certificatesDir: /etc/kubernetes/pki|certificatesDir: {{ .certDir }}|' kubeadm.yaml
    sudo sed -i 's|criSocket: unix:///var/run/containerd/containerd.sock|criSocket: {{ .criSocket }}|' kubeadm.yaml
    sudo sed -i 's|serviceSubnet: 10.96.0.0/12|serviceSubnet: {{ .serviceCidr }}|' kubeadm.yaml
    sudo sed -i 's|dnsDomain: cluster.local|dnsDomain: {{ .serviceDnsDomain }}|' kubeadm.yaml
    sudo sed -i 's|token: .*|token: {{ .token }}|' kubeadm.yaml
    sudo sed -i 's|ttl: 24h0m0s|ttl: {{ .tokenTtl }}|' kubeadm.yaml

    if [ "{{ .is_ha }}" == "true" ]; then
        sudo sed -i "22i \controlPlaneEndpoint: {{ .vip }}:{{ .apiserverBindPort }}" kubeadm.yaml
        sudo kubeadm init --config kubeadm.yaml --upload-certs >> "$OUTPUT"
    else
        sudo kubeadm init --config kubeadm.yaml >> "$OUTPUT"
    fi
fi

if [ ! -f "$HOME/.kube/config" ]; then
    mkdir -p $HOME/.kube
    sudo \cp -f /etc/kubernetes/admin.conf $HOME/.kube/config
    sudo chown "${SUDO_USER:-$(whoami)}":$(id -gn $SUDO_USER) $HOME/.kube/config
fi

if sudo which containerd; then
  sudo crictl config runtime-endpoint unix:///var/run/containerd/containerd.sock
fi

sudo systemctl status kubelet | grep "active (running)" > /dev/null
if [ "$?" == "0" ]; then
    info_log "kubelet has running"
else
    error_log "The kubelet status is abnormal"
    fatal_log "$(systemctl status kubelet)"
fi

kubectl get node > /dev/null
if [ "$?" == "0" ]; then
    info_log "Kubernetes initialization successful"
else
    fatal_log "Kubernetes initialization failed"
fi
