#!/bin/bash
# ******************************************************************
# Copyright (c) 2025 Bocloud Technologies Co., Ltd.
# cluster-api-provider-bke is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************

sudo bke reset
sudo bke init --runtime docker --confirm true

# 判断当前用户是否为root
if [ $(id -u) != "0" ]; then
  # 获取home目录
  home=$(eval echo ~${SUDO_USER})
  sudo cp /etc/rancher/k3s/k3s.yaml $home/.kube/config
fi
