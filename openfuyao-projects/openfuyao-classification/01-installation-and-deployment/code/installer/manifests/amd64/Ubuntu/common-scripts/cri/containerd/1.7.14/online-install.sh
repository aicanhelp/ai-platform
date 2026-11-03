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
source ../../utils/consts.sh
source ./consts.sh

ARCH=$(uname -m)
case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
esac

CONTAINERD_URL="${FUYAO_OBS}/containerd/containerd/releases/download/v${VERSION}/containerd-${VERSION}-linux-${ARCH}.tar.gz"
CONTAINERD_SERVICE_URL="${FUYAO_OBS}/containerd/containerd/releases/download/v${VERSION}/containerd.service"

# 配置网络桥接流量与ipv4转接
function net_bridge_call_iptables () {
    set +e
    cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

    sudo modprobe overlay
    sudo modprobe br_netfilter

    # 设置所需的sysctl参数，参数在重新启动后保持不变
    cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF

    # 应用 sysctl 参数而不重新启动
    sudo sysctl --system

    lsmod | grep br_netfilter
    lsmod | grep overlay

    sysctl net.bridge.bridge-nf-call-iptables net.bridge.bridge-nf-call-ip6tables net.ipv4.ip_forward

    # openEuler专属配置
    echo "1" >> /proc/sys/net/ipv4/ip_forward
    set -e
}


# 安装containerd
function exec_install_containerd () {
    curl -sOL ${CONTAINERD_URL}
    tar Cxzvf /usr/local containerd-${VERSION}-linux-${ARCH}.tar.gz
}


# 安装containerd的service
function install_container_service () {
    local path="/usr/local/lib/systemd/system"
    local file="containerd.service"
    mkdir -p $path
    if [ -f "$path/$file" ]; then
        return 0
    fi

    curl -sOL ${CONTAINERD_SERVICE_URL}
    cp ./$file $path/
    sudo systemctl daemon-reload
    sudo systemctl enable --now containerd
}

# 修改containerd配置
function modify_containerd_config () {
    mkdir -p /etc/containerd/
    containerd config default > /etc/containerd/config.toml
    # 使用systemd cgroup驱动
    sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
    sed -i 's/sandbox_image = "registry.k8s.io\/pause:3.8"/sandbox_image = "harbor.openfuyao.com\/openfuyao\/kubernetes\/pause:3.8"/' /etc/containerd/config.toml
    sudo systemctl enable containerd
    sudo systemctl restart containerd
}

net_bridge_call_iptables
exec_install_containerd
install_container_service
modify_containerd_config
