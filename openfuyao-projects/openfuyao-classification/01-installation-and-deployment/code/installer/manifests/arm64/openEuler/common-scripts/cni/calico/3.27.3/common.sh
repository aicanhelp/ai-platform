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

source ../../utils/log.sh "calico"
source ../../utils/consts.sh

function deal_with_calico_yaml() {
    local yaml_file=$1
    local up_count=$2
    local down_count=$3
    local operate=$4

    sudo sed -i "${up_count}i\            - name: IP_AUTODETECTION_METHOD" ${yaml_file}
    sudo sed -i "${down_count}i\              value: \"skip-interface=nerdctl.*\"" ${yaml_file}
    kubectl ${operate} -f ${yaml_file}
    info_log "deal with calico yaml success"
}