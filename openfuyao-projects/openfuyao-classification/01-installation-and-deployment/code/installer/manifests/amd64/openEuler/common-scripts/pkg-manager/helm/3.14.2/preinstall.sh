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

source ../../utils/log.sh "helm"
source ./consts.sh

if sudo which helm >> "$OUTPUT" 2>&1; then
    warning_log "Helm has been installed"

    if [[ $(helm version) =~ $VERSION ]]; then
        info_log "The installed version of helm is $VERSION"
    else
        error_log "helm version conflict"
        fatal_log "The installed version of helm is $(helm version)"
    fi
    exit 0
fi