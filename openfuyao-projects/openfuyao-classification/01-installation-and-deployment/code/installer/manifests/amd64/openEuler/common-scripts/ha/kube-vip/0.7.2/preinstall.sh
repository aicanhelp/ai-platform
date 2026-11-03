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

source ../../utils/log.sh "kube-vip"

VIP={{ .vip }}
# ip 是否合法
VALID_CHECK=$(echo "${VIP}" | awk -F. '$1<=255&&$2<=255&&$3<=255&&$4<=255 {print "yes"}')
if echo "${VIP}" | grep -E "^([0-9]{1,3}\.){3}[0-9]{1,3}$" >/dev/null; then
    if [ "${VALID_CHECK}" == "yes" ]; then
        info_log "vip is valid"
    else
        fatal_log "error: ${VIP} is invalid, please check the input vip"
    fi
else
    fatal_log "error: ${VIP} is invalid, please check the input vip"
fi

# ip 不能ping通
if ping -c 1 -w 1 "${VIP}" >/dev/null; then
    warning_log "${VIP} may be invalid"
fi

# 判断已存在的vip是否相同
YAML_PATH="/etc/kubernetes/manifests/kube-vip.yaml"
if [ -f "$YAML_PATH" ]; then
    if grep -q "${VIP}" "${YAML_PATH}"; then
        warning_log "kube-vip has been installed"
        exit 0
    else
        fatal_log "Please check the ${YAML_PATH} files, the VIP in the file is different from the VIP provided"
    fi
fi
