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

set -e

function check_ca_cert() {
    if [ -f /etc/debian_version ]; then
      if [ ! -e "/etc/ssl/certs/ca-certificates.crt" ]; then
        sudo apt-get update
        sudo apt-get install ca-certificates
      fi
    fi
}

# 节点资源检查 至少需要2核CPU、4G内存
function check_resource() {
    mkdir -p /var/log/fuyao-installer
    local fuyao_out="/var/log/fuyao-installer/shell.out"

    local core_count
    local socket_count
    core_count=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')
    socket_count=$(lscpu | grep "Socket(s):" | awk '{print $2}')
    local cpu_count
    cpu_count=$(($core_count * $socket_count))
    if [ $cpu_count -lt 2 ]; then
        echo "The number of CPUs is $cpu_count, and the minimum requirement for the number of CPUs is 2" >> ${fuyao_out}
        exit 1
    fi

    local memory
    memory=$(free -g | awk '/^Mem:/{print $2}')
    if [ $memory -lt 4 ]; then
        echo "Memory size is ${memory}GB, and the minimum memory requirement is 4GB" >> ${fuyao_out}
        exit 1
    fi
}

function check_tools() {
    local tool_name
    tool_name=$1

    if command -v "${tool_name}" &> /dev/null; then
        return
    fi

    if [ "`cat /etc/os-release | grep openEuler`" ]; then
        sudo yum install -y "${tool_name}"
    elif [ -f /etc/debian_version ]; then
        sudo apt-get update
        sudo apt-get install -y "${tool_name}"
    elif [ -f /etc/redhat-release ]; then
        sudo yum install -y "${tool_name}"
    fi
    echo $tool_name
}

check_resource
check_ca_cert

declare -a TOOLS=(curl tar bc expect)
for tool in "${TOOLS[@]}"
do
  check_tools "$tool"
done


