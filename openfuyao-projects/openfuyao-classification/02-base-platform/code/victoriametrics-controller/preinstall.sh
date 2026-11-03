#!/bin/bash
###############################################################
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
###############################################################

source ./log.sh

# 检查vmks命名空间是否存在
if kubectl get ns vmks > /dev/null 2>&1; then
    # 检查VictoriaMetrics的核心组件是否已存在
    if kubectl get deployment -n vmks | grep -q "victoria-metrics" || \
       kubectl get statefulset -n vmks | grep -q "victoria-metrics" || \
       kubectl get pods -n vmks | grep -q "victoria-metrics"; then
        info_log "VictoriaMetrics 组件已在 vmks 命名空间中安装，跳过安装"
        exit 0
    else
        info_log "vmks 命名空间存在但未安装 VictoriaMetrics，继续安装"
    fi
else
    info_log "vmks 命名空间不存在，开始安装 VictoriaMetrics"
fi
