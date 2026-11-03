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

sudo systemctl stop containerd
sudo systemctl disable containerd
sudo rm -rf /usr/local/lib/systemd/system/containerd.service
sudo rm -rf /opt/containerd
sudo rm -rf /var/lib/containerd
sudo rm -rf /usr/local/bin/containerd*
sudo rm -f /usr/local/bin/ctr
sudo rm -rf /run/containerd
sudo rm -rf /etc/containerd/

sudo rm -f /etc/modules-load.d/k8s.conf
sudo rmmod overlay
sudo rmmod br_netfilter
sudo rm -f /etc/sysctl.d/k8s.conf
sudo sysctl -p
sudo systemctl enable firewalld
sudo systemctl start firewalld

info_log "Successfully uninstalled containerd"
