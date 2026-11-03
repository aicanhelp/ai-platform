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

COMPONENT=$1

LOG_PATH="/var/log/fuyao-installer"
mkdir -p ${LOG_PATH}
INSTALLER_LOG="${LOG_PATH}/shell.log"

# 脚本命令输出内容
OUTPUT="${LOG_PATH}/shell.out"

if [ -z "$COMPONENT" ]
then
    COMPONENT=$(pwd)
fi

function info_log () {
    echo "$(date +"[%Y-%m-%d %H:%M:%S,%N]") [$COMPONENT] [INFO] $1" >> ${INSTALLER_LOG}
}

function warning_log () {
    echo "$(date +"[%Y-%m-%d %H:%M:%S,%N]") [$COMPONENT] [WARNING] $1" >> ${INSTALLER_LOG}
}

function error_log () {
    echo "$(date +"[%Y-%m-%d %H:%M:%S,%N]") [$COMPONENT] [ERROR] $1" >> ${INSTALLER_LOG}
}

function fatal_log () {
    echo "$(date +"[%Y-%m-%d %H:%M:%S,%N]") [$COMPONENT] [FATAL] $1" >> ${INSTALLER_LOG}
    echo "$1"
    exit 1
}
