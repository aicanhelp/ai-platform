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

sudo kubeadm reset -f
sudo rm -rf $HOME/.kube
sudo rm -rf /etc/kubernetes/

sudo rm -rf /usr/bin/kube*
sudo rm -rf /usr/local/bin/kube*
sudo rm -rf /usr/local/bin/crictl

sudo rm -rf /etc/sysctl.d/k8s.conf
sudo rm -f /etc/systemd/system/kubelet.service
sudo rm -rf /etc/systemd/system/kubelet.service.d

sudo rm -rf /var/lib/etcd
sudo rm -rf /var/lib/kubelet

if [ "{{ .kubeProxyMode }}" == "ipvs" ]; then
    sudo ip link set dev kube-ipvs0 down
    sudo ip link delete kube-ipvs0
    sudo rm -f /etc/modules-load.d/ipvs.conf
    sudo rmmod ip_vs
    sudo rmmod ip_vs_rr
    sudo rmmod ip_vs_wrr
    sudo rmmod ip_vs_sh
    sudo rmmod nf_conntrack
fi
sudo sed -i '/swap.*swap/s/^#*//g' ./fstab
sudo mount -a
sudo setenforce 1
sudo sed -i 's/^SELINUX=permissive$/SELINUX=enforcing/' /etc/selinux/config

# 删除coredns 绑定的网卡
sudo ip link set dev tunl0 down
sudo ip link delete tunl0

info_log "Successfully uninstalled kubernetes"
