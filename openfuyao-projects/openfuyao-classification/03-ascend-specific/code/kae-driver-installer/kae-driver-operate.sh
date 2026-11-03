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

INSTALL_POLICY=$1
DRIVER_INSTALL_EXEC_DIR=$2
OPERATE_TYPE=$3

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

function install_basic_packages () {
    info_log "Install dependency packages"
    yum clean all
    yum makecache

    for pkg in perl bzip2 gcc make zlib zlib-devel pcre pcre-devel perl-ExtUtils-Embed openssl openssl-devel net-tools* automake autoconf libtool git patch numactl-devel kernel-devel-$(uname -r); do
        yum -y install "$pkg" || fatal_log "Failed to install $pkg"
    done
    info_log "Install dependency packages success"
}

function exec_install_kae_driver () {
    cd /opt/$DRIVER_INSTALL_EXEC_DIR/KAE || fatal_log "Failed to change directory to /opt/$DRIVER_INSTALL_EXEC_DIR/KAE"
    install_basic_packages

    info_log "Install KAE driver"
    sh build.sh all
    if [ $? -ne 0 ]; then
        fatal_log "Failed to install KAE driver"
    fi
    info_log "success install KAE driver"
}

function uninstall_kae_driver () {
    info_log "Uninstall KAE driver"
    cd /opt/$DRIVER_INSTALL_EXEC_DIR/KAE || fatal_log "Failed to change directory to /opt/$DRIVER_INSTALL_EXEC_DIR/KAE"

    sh build.sh cleanup
    if [ $? -ne 0 ]; then
        fatal_log "Failed to clean up KAE driver"
    fi
}

function grep_utils() {
    local check_list=$1

    local result=1
    for item in $check_list; do
        local target="$(echo $item | cut -d':' -f1)"
        info_log "Check $target in $item"
        local target_path="$(echo $item | cut -d':' -f2)"
        info_log "Check $target_path in $item"
        # 检查路径是否存在
        if [ ! -d "$target_path" ]; then
            continue
        fi

        if ls -al "$target_path" | grep -q "$target"; then
            info_log "find $target in $target_path"
            result=0
        fi
    done
    info_log "Check $check_list success with result $result"
    return $result
}

function check_dev_path() {
    info_log "check dev path"
    local check_dev="hisi_hpre:/dev hisi_sec2:/dev hisi_zip:/dev"
    # 检查通过返回0，检查不通过返回1
    grep_utils "$check_dev"
    return $?
}

function check_modprobe_path() {
    info_log "check modprobe path"
    local check_modprobe="hisi_hpre.conf:/etc/modprobe.d hisi_sec2.conf:/etc/modprobe.d hisi_zip.conf:/etc/modprobe.d"
    # 检查通过返回0，检查不通过返回1
    grep_utils "$check_modprobe"
    return $?
}

function kae_driver_post_install () {
    check_dev_path
    local check_dev=$?

    check_modprobe_path
    local check_modprobe=$?

    if [ $check_dev -gt 0 ] || [ $check_modprobe -gt 0 ]; then
        fatal_log "KAE driver post install failed, please check the installation environment"
    fi
    info_log "KAE driver post install success"
}

# check kae driver has been installed
function kae_driver_pre_install () {
    check_dev_path
    local check_dev=$?

    check_modprobe_path
    local check_modprobe=$?

    #两项检查均通过，返回0，否则返回1
    if [ $check_dev -eq 0 ] && [ $check_modprobe -eq 0 ]; then
        info_log "KAE driver pre install success"
        return 0
    fi
    warning_log "KAE driver pre install failed, we will try to uninstall KAE driver and reinstall it"
    return 1
}

function install_kae_driver_always () {
    uninstall_kae_driver
    exec_install_kae_driver
    kae_driver_post_install
}

function install_kae_driver_if_not_present () {
    kae_driver_pre_install
    if [ $? -eq 0 ]; then
       return 0
    fi

    uninstall_kae_driver
    exec_install_kae_driver
    kae_driver_post_install
}

function install_kae_driver () {
    # 根据 INSTALL_POLICY 进行不同的操作
    if [ "$(echo "$INSTALL_POLICY" | tr '[:upper:]' '[:lower:]')" = "always" ]; then
        install_kae_driver_always
    elif [ "$(echo "$INSTALL_POLICY" | tr '[:upper:]' '[:lower:]')" = "ifnotpresent" ]; then
        install_kae_driver_if_not_present
    else
        warning_log "Invalid INSTALL_POLICY, use default policy IfNotPresent"
        install_kae_driver_if_not_present
    fi
}

if [ "$OPERATE_TYPE" = "uninstall" ]; then
    uninstall_kae_driver
elif [ "$OPERATE_TYPE" = "install" ]; then
    install_kae_driver
else
    error_log "Invalid operate type"
fi
