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

source ../../utils/log.sh "runc"
source ./consts.sh

if sudo which runc >/dev/null 2>&1; then
    warning_log "runc has been installed"

    if [[ $(sudo runc -v) =~ $VERSION ]]; then
        info_log "The installed version of runc is $VERSION"
    else
        error_log "runc version conflict"
        fatal_log "The installed version of runc is $(sudo runc -v)"
    fi
    exit 0
fi
