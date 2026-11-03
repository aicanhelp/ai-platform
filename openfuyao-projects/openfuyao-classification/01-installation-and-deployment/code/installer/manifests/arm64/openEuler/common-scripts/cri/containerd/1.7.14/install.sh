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

# 配置网络桥接流量与ipv4转接
function net_bridge_call_iptables () {
    set +e
    # 取自kubernetes官方文档
    # https://v1-28.docs.kubernetes.io/zh-cn/docs/setup/production-environment/container-runtimes/
    cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

    sudo modprobe br_netfilter
    sudo modprobe overlay

    # 设置所需的sysctl参数，参数在重新启动后保持不变
    cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF
    # 应用sysctl参数
    sudo sysctl --system

    sudo lsmod | grep overlay >> "$OUTPUT"
    sudo lsmod | grep br_netfilter >> "$OUTPUT"

    sudo sysctl net.bridge.bridge-nf-call-iptables net.bridge.bridge-nf-call-ip6tables net.ipv4.ip_forward
    sudo systemctl stop firewalld
    sudo systemctl disable firewalld

    # openEuler专属配置
    sudo echo "1" >> /proc/sys/net/ipv4/ip_forward
    set -e
}

# 安装containerd
function exec_install_containerd () {
    # 通过前置检查，则肯定没有版本冲突，简单检查没有则安装
    if sudo which containerd; then
        return 0
    fi
    sudo tar Cxzvf /usr/local ./${ARCH}-bin/containerd-${VERSION}-linux-${ARCH}.tar.gz >> "$OUTPUT"
}

# 安装containerd的service
function install_container_service () {
    info_log "start install container_service"
    local path="/usr/local/lib/systemd/system"
    local file="containerd.service"
    sudo mkdir -p $path
    if [ -f "$path/$file" ]; then
        warning_log "This computer is already installed containerd.service."
        return 0
    fi

    sudo cp ./${ARCH}-bin/$file $path/
    sudo systemctl daemon-reload
    sudo systemctl enable --now containerd
    info_log "Successfully installed container_service"
}

# 修改containerd配置
function modify_containerd_config () {
    info_log "start modify containerd config"
    local config_file="/etc/containerd/config.toml"
    if [ ! -f "${config_file}" ]; then
        sudo mkdir -p /etc/containerd/
        sudo containerd config default > ${config_file}
        sudo chmod 600 ${config_file}
        sudo chmod 700 /etc/containerd/
    fi

    local need_restart=0
    # 使用systemd cgroup驱动
    if grep "SystemdCgroup = false" ${config_file}; then
        need_restart=1
        sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' ${config_file}
    fi

    # 设置containerd的根目录
    if ! cat ${config_file} | grep "root = \"/var/lib/containerd\"" | grep "{{ .rootPath }}"; then
        need_restart=1
        sudo mkdir -p {{ .rootPath }}
        sudo sed -i 's|root = "/var/lib/containerd"|root = "{{ .rootPath }}"|' ${config_file}
    fi
    # 设置containerd的状态目录
    if ! cat ${config_file} | grep "root = \"/var/lib/containerd\"" | grep "{{ .statePath }}"; then
        need_restart=1
        sudo mkdir -p {{ .statePath }}
        sudo sed -i 's|state = "/run/containerd"|state = "{{ .statePath }}"|' ${config_file}
    fi

    # 修改沙箱镜像
    # 场景：在线安装
    if grep "sandbox_image = \"registry.k8s.io/pause:3.8\"" ${config_file}; then
        need_restart=1
        sudo sed -i 's|sandbox_image = "registry.k8s.io/pause:3.8"|sandbox_image = "{{ .registry }}/kubernetes/pause:3.9"|' ${config_file}
    fi
    # 场景：离线安装，导出制品节点改为集群内节点
    if grep "sandbox_image = \"harbor.openfuyao.com/openfuyao/kubernetes/pause:3.8\"" ${config_file}; then
        need_restart=1
        sudo sed -i 's|sandbox_image = "harbor.openfuyao.com/openfuyao/kubernetes/pause:3.8"|sandbox_image = "{{ .registry }}/kubernetes/pause:3.9"|' ${config_file}
    fi
    # 场景：离线安装，镜像节点为集群内节点
    if grep "sandbox_image = \"harbor.openfuyao.com/openfuyao/kubernetes/pause:3.9\"" ${config_file}; then
        need_restart=1
        sudo sed -i 's|sandbox_image = "harbor.openfuyao.com/openfuyao/kubernetes/pause:3.9"|sandbox_image = "{{ .registry }}/kubernetes/pause:3.9"|' ${config_file}
    fi

    {{ if ne .registry "${FUYAO_RGISTRY}" }}
    if [ "$(sed -n '167p' /etc/containerd/config.toml)" != "        [plugins.\"io.containerd.grpc.v1.cri\".registry.configs.\"{{ .registry }}\".tls]" ]; then
        need_restart=1
        sudo sed -i "167 i\        [plugins.\"io.containerd.grpc.v1.cri\".registry.configs.\"{{ .registry }}\".tls]" ${config_file}
    fi
    if [ "$(sed -n '168p' /etc/containerd/config.toml)" != "          ca_file = \"${BASE_CERT_PATH}/${HARBOR_CA_CRT}\"" ]; then
        need_restart=1
        sudo sed -i "168 i\          ca_file = \"${BASE_CERT_PATH}/${HARBOR_CA_CRT}\"" ${config_file}
    fi
    {{ end }}

    if [ ${need_restart} == 1 ]; then
        sudo systemctl enable containerd
        sudo systemctl restart containerd
    fi
    if [ -d "${HARBOR_INSTALL_PATH}/harbor" ] && sudo which nerdctl; then
        info_log "restart harbor"
        local abs_path=$(pwd)
        cd ${HARBOR_INSTALL_PATH}/harbor
        nerdctl compose down
        ./install-harbor.sh
        cd ${abs_path}
        info_log "Finished restart harbor"
    fi
    info_log "Complete container configuration modifications"
}

info_log "start install containerd"

net_bridge_call_iptables
exec_install_containerd
install_container_service
modify_containerd_config

source ./postinstall.sh
