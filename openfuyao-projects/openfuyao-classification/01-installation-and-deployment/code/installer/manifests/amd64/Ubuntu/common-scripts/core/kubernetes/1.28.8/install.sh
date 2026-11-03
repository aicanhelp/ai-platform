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
source ../../utils/consts.sh

source ./preinstall.sh

ARCH=$(uname -m)
case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
esac

function exec_system_prepare () {
    sudo sed -i '/swap/s/^/#/g' /etc/fstab
    sudo mount -a
    sudo swapoff -a

    # 将 SELinux 设置为 permissive 模式（相当于将其禁用）
    sudo setenforce 0
    sudo sed -i 's/^SELINUX=enforcing$/SELINUX=permissive/' /etc/selinux/config
    modprobe ipip
    # 启动tunl0网卡
    sudo ip link set tunl0 up
}

function config_ipvs() {
    if [ "{{ .kubeProxyMode }}" != "ipvs" ]; then
        return
    fi
    cat <<EOF | sudo tee /etc/modules-load.d/ipvs.conf
    # Load IPVS at boot
    ip_vs
    ip_vs_rr
    ip_vs_wrr
    ip_vs_sh
    nf_conntrack
EOF

    sudo modprobe -- ip_vs
    sudo modprobe -- ip_vs_rr
    sudo modprobe -- ip_vs_wrr
    sudo modprobe -- ip_vs_sh
    sudo modprobe -- nf_conntrack
    # 确认内核模块加载成功
    sudo lsmod | grep -e ip_vs -e nf_conntrack >> "${OUTPUT}"
}

function install_kubernetes() {
    local install_dir="/usr/local/bin"
    # 安装cri-tools
    if [ -z "$INSTALLED_CRI_TOOLS" ]; then
        sudo tar Cxzvf ${install_dir} ./${ARCH}-bin/crictl-v${CRICTL_VERSION}-linux-${ARCH}.tar.gz
    fi
    if [ -z "$INSTALLED_KUBELET" ]; then
        sudo cp ./${ARCH}-bin/kubelet ${install_dir}
        sudo chmod +x ${install_dir}/kubelet
    fi
    if [ -z "$INSTALLED_KUBEADM" ]; then
        sudo cp ./${ARCH}-bin/kubeadm ${install_dir}
        sudo chmod +x ${install_dir}/kubeadm
    fi
    if [ -z "$INSTALLED_KUBECTL" ]; then
        sudo cp ./${ARCH}-bin/kubectl ${install_dir}
        sudo chmod +x ${install_dir}/kubectl
    fi
    
    local kubelet_service_folder="/etc/systemd/system/kubelet.service.d"
    if [ ! -d "${kubelet_service_folder}" ]; then
       sudo mkdir -p ${kubelet_service_folder}
    fi
    local kubelet_service_file="/etc/systemd/system/kubelet.service"
    if [ ! -f "${kubelet_service_file}" ]; then
        sudo sed "s:/usr/bin:${install_dir}:g" ./${ARCH}-bin/kubelet.service | sudo tee ${kubelet_service_file}
    fi
    local kubeadm_config="${kubelet_service_folder}/10-kubeadm.conf"
    if [ ! -f "${kubeadm_config}" ]; then
        sudo sed "s:/usr/bin:${install_dir}:g" ./${ARCH}-bin/10-kubeadm.conf | sudo tee ${kubeadm_config}
    fi
    
    if [ -z "$INSTALLED_CRI_TOOLS" ] || [ -z "$INSTALLED_CRI_TOOLS" ] || [ -z "$INSTALLED_CRI_TOOLS" ]; then
        sudo rm -rf $HOME/.kube
    fi

    sudo systemctl enable --now kubelet
}

exec_system_prepare
config_ipvs
install_kubernetes

./postinstall.sh
