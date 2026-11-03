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

component="${1:-all}"
zip_path=/opt
mount_prefix=/mnt
MINDIOttp_INSTALL_EXEC_DIR="openFuyao-npu-mindiottp"
MINDIOACP_INSTALL_EXEC_DIR="openFuyao-npu-mindioacp"

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

function copy_to_host() {
    local install_exec_dir=$1
    info_log "Copying to host directory: $zip_path/$install_exec_dir"
    nsenter -i/proc/1/ns/ipc -m/proc/1/ns/mnt -n/proc/1/ns/net mkdir -p "$zip_path/$install_exec_dir"
    if [ $? -ne 0 ]; then
        fatal_log "Failed to create directory on host"
    fi

    cp -r /tmp/mindio/* "$mount_prefix/$zip_path/$install_exec_dir"
    if [ $? -ne 0 ]; then
        fatal_log "Failed to copy files to host"
    fi
    info_log "Copy to host successful"
}


function install_whl_package() {
    local component=$1
    local install_exec_dir=$2
    local whl_file=""
    local host_whl_file=""

    info_log "Looking for $component WHL package (name contains '$component') in $mount_prefix/$zip_path/$install_exec_dir"
        whl_file=$(ls "$mount_prefix/$zip_path/$install_exec_dir"/*.whl 2>/dev/null |
                   while read -r file; do
                       if basename "$file" | grep -qi "$component"; then
                           echo "$file"
                           break
                       fi
                   done)

    if [ -z "$whl_file" ] || [ ! -f "$whl_file" ]; then
        fatal_log "No WHL package found for $component (name must contain '$component') in $mount_prefix/$zip_path/$install_exec_dir"
    fi
    info_log "Found $component WHL package: $whl_file"
    host_whl_file=$(echo "$whl_file" | sed -E "s|^${mount_prefix}||; s|^//|/|")
    if [[ ! "$host_whl_file" =~ ^/ ]]; then
            host_whl_file="/$host_whl_file"
    fi
    info_log "Installing $component WHL package: $host_whl_file"
    nsenter -i/proc/1/ns/ipc -m/proc/1/ns/mnt -n/proc/1/ns/net bash -c "python3 -m pip install '$host_whl_file' --force-reinstall --no-index"
    if [ $? -ne 0 ]; then
        fatal_log "Failed to install $component WHL package: $host_whl_file"
    fi
    info_log "$component WHL package installed successfully"
}


function component_install() {
    local component=$1
    local install_exec_dir=""

    if [ "$component" = "ttp" ]; then
        install_exec_dir=$MINDIOttp_INSTALL_EXEC_DIR
    elif [ "$component" = "acp" ]; then
        install_exec_dir=$MINDIOACP_INSTALL_EXEC_DIR
    else
        fatal_log "Unknown component: $component"
    fi

    info_log "Starting installation for $component..."
    copy_to_host "$install_exec_dir"
    install_whl_package "$component" "$install_exec_dir"

    info_log "$component installation completed"
}


if [ "$component" = "ttp" ]; then
        component_install "ttp"
elif [ "$component" = "acp" ]; then
        component_install "acp"
else
        component_install "ttp"
        component_install "acp"
fi