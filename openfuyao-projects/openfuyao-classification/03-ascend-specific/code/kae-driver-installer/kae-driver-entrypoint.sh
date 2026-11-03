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

DRIVER_INSTALL_EXEC_DIR="openFuyao-kae-driver"

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

# 拷贝脚本文件到宿主机
function copy_to_host() {
    info_log "copy kae-install to host"
    # 避免宿主机有同名路径，在宿主机上创建带有时间戳的目录
    nsenter -i/proc/1/ns/ipc -m/proc/1/ns/mnt -n/proc/1/ns/net mkdir -p "/opt/${DRIVER_INSTALL_EXEC_DIR}"
    if [ $? -ne 0 ]; then
        fatal_log "failed to create directory on host"
    fi

    cp -r /home/kae-driver/* /mnt/opt/${DRIVER_INSTALL_EXEC_DIR}
    if [ $? -ne 0 ]; then
        fatal_log "failed to copy kae install files to host"
    fi
    info_log "copy kae-install to host success"
}

function remove_tmp_files() {
    info_log "remove temporary files"
    rm -rf /mnt/opt/${DRIVER_INSTALL_EXEC_DIR}
    if [ $? -ne 0 ]; then
        error_log "failed to remove temporary files"
    fi
}

function install_kae_driver() {
    info_log "install kae-driver"
    copy_to_host

    info_log "Install Policy: $INSTALL_POLICY"
    nsenter -i/proc/1/ns/ipc -m/proc/1/ns/mnt -n/proc/1/ns/net bash /opt/${DRIVER_INSTALL_EXEC_DIR}/kae-driver-operate.sh "$INSTALL_POLICY" "$DRIVER_INSTALL_EXEC_DIR" "install"
    if [ $? -ne 0 ]; then
        fatal_log "install failed"
    fi

    remove_tmp_files
}

function uninstall_kae_driver() {
    info_log "uninstall kae-driver"
    copy_to_host

    nsenter -i/proc/1/ns/ipc -m/proc/1/ns/mnt -n/proc/1/ns/net bash /opt/${DRIVER_INSTALL_EXEC_DIR}/kae-driver-operate.sh "$INSTALL_POLICY" "$DRIVER_INSTALL_EXEC_DIR" "uninstall"
    if [ $? -ne 0 ]; then
        fatal_log "install failed"
    fi

    remove_tmp_files
}

function usage() {
    echo "see source file"
}

# get-opt 参数
main(){
    while true
    do
        case "$1" in
        -o|--operate)  # 操作类型
            local operate="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
        shift
        break
        ;;
        *)
        echo "$1 is not option, please use -h to view help"
        shift
        break
        ;;
        esac
        shift
    done

    if [ "${operate}" == "uninstall" ];then
        info_log "uninstall kae-driver operation"
        uninstall_kae_driver
    elif [ "${operate}" == "install" ]; then
        info_log "install kae-driver operation"
        install_kae_driver
    else
        info_log "without operation, use default install operation"
        install_kae_driver
    fi
}

ENTRYPOINT_SHELL=$(getopt -n kae-driver-entrypoint.sh -o o:h --long operate:,help -- "$@")
[ $? -ne 0 ] && exit 1
eval set -- "$ENTRYPOINT_SHELL"
main $@
