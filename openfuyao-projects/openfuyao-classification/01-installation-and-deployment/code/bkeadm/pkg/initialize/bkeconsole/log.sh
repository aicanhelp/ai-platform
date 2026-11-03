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

if [ "${LOG_SH_LOADED:-}" = "1" ]; then
  return 0
fi
LOG_SH_LOADED=1

LOG_PATH="/var/log/openFuyao/openfuyao-system-controller"
INSTALLER_LOG="${LOG_PATH}/openfuyao-system-controller.log"

function _log() {
	local prefix="$1"
	shift
	echo "$(date +"[%Y-%m-%d %H:%M:%S,%N]") [${prefix}] $*"

	if [ ! -f ${INSTALLER_LOG} ]; then
	    mkdir -p ${LOG_PATH}
	    touch ${INSTALLER_LOG}
	fi
  echo "$(date +"[%Y-%m-%d %H:%M:%S,%N]") [${prefix}] $*" >> ${INSTALLER_LOG}
}

function info_log () {
    _log "INFO" "$*"
}

function warning_log () {
    _log "WARNING" "$*"
}

function error_log () {
    _log "ERROR" "$*"
}

function fatal_log () {
    _log "FATAL" "$*"
    exit 1
}
