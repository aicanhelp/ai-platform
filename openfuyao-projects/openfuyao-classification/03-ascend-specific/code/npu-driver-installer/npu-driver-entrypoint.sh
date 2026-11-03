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

DRIVER_INSTALL_EXEC_DIR="openFuyao-npu-driver"
HOST_INSTALL_DIR="/mnt/opt/${DRIVER_INSTALL_EXEC_DIR}"
HOST_SCRIPT_PATH="/opt/${DRIVER_INSTALL_EXEC_DIR}/npu-driver-operate.sh"

VERSION="${DRIVER_VERSION}"
HOST_SOURCE_PATH="${HOST_DRIVER_SOURCE_PATH}"
INSTALL_MODE="auto"

function _log() {
	local prefix="$1"
	shift
	echo "$(date +"[%Y-%m-%d %H:%M:%S,%N]") [${prefix}] $*"
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

function parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                INSTALL_MODE="$2"; shift 2 ;;
            --version)
                VERSION="$2"; shift 2 ;;
            *)
                warning_log "Unknown argument ignored: $1"
                shift ;;
        esac
    done
}


function validate_env() {
    if [ -z "$VERSION" ]; then
        fatal_log "Missing Driver Version. Please set 'env: DRIVER_VERSION' or use '--version'."
    fi

    if [ ! -d "/home/npu-driver-files" ]; then
        fatal_log "Container source directory '/home/npu-driver-files' missing."
    fi

    info_log "Config: Version=[$VERSION], Mode=[$INSTALL_MODE], HostSource=[${HOST_SOURCE_PATH:-'N/A'}]"
}

function prepare_host_env() {
    info_log "Preparing host directory: ${HOST_INSTALL_DIR}"

    rm -rf "${HOST_INSTALL_DIR}"
    mkdir -p "${HOST_INSTALL_DIR}"


    cp -r /home/npu-driver-files/* "${HOST_INSTALL_DIR}/" || fatal_log "Failed to copy scripts to host."

    chmod +x "${HOST_INSTALL_DIR}"/*.sh
}

function trigger_host_installation() {
    info_log "Switching context to Host PID 1..."

    nsenter -i/proc/1/ns/ipc -m/proc/1/ns/mnt -n/proc/1/ns/net \
        bash "${HOST_SCRIPT_PATH}" \
        --work-dir "${DRIVER_INSTALL_EXEC_DIR}" \
        --version "${VERSION}" \
        --mode "${INSTALL_MODE}" \
        --offline-path "${HOST_SOURCE_PATH}"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        info_log "Host installation succeeded."
        rm -rf "${HOST_INSTALL_DIR}"
    else
        fatal_log "Host installation failed with code $exit_code."
    fi
}

function main() {
    parse_args "$@"
    validate_env
    prepare_host_env
    trigger_host_installation
}

main "$@"