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

source ../../utils/log.sh "nerdctl"
source ./consts.sh

if sudo which nerdctl; then
    warning_log "nerdctl has been installed"

    if [[ $(sudo nerdctl version) =~ $VERSION ]]; then
        info_log "The installed version of nerdctl is $VERSION"
        exit 0
    else
        error_log "nerdctl version conflict"
        fatal_log "$(sudo nerdctl version)"
    fi
fi
