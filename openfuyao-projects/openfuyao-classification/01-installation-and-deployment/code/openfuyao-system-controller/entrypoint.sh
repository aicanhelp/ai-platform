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

# 安装执行路径
BASE_INSTALL_EXEC_DIR="/opt"
SUB_EXEC_DIR="openFuyao/openfuyao-system-install"
INSTALL_EXEC_DIR="$BASE_INSTALL_EXEC_DIR/$SUB_EXEC_DIR"

# 环境变量
# OPENFUYAO_REGISTRY

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
    info_log "copy install files to host"
    # 避免宿主机有同名路径，在宿主机上创建带有时间戳的目录
    nsenter -i/proc/1/ns/ipc -m/proc/1/ns/mnt -n/proc/1/ns/net mkdir -p "${INSTALL_EXEC_DIR}"
    if [ $? -ne 0 ]; then
        fatal_log "failed to create directory on host"
    fi

    cp -rf /home/openfuyao-system/* /mnt/opt/${SUB_EXEC_DIR}
    if [ $? -ne 0 ]; then
        fatal_log "failed to copy install files to host"
    fi
    info_log "copy install files to host success"
}

function remove_tmp_files() {
    info_log "remove temporary files"
    rm -rf /mnt/opt/${SUB_EXEC_DIR}
    if [ $? -ne 0 ]; then
        error_log "failed to remove temporary files"
    fi
}

function install_openfuyao_system() {
    info_log "Starting installation of openfuyao-system"

    # 复制文件到目标主机
    info_log "Copying installation files to host"
    copy_to_host || {
        fatal_log "Failed to copy installation files to host"
        return 1
    }

    # 设置默认值（防止空变量导致参数解析错误）
    local registry_arg=""
    if [ -n "${OPENFUYAO_REGISTRY}" ]; then
        registry_arg="-r \"${OPENFUYAO_REGISTRY}\""
    fi

    # 构建安装命令（使用数组或Here Document提高可读性）
    info_log "Executing installation script in host namespace"

    # 方法1：使用Here Document（推荐，可读性好）
    if ! nsenter -i/proc/1/ns/ipc -m/proc/1/ns/mnt -n/proc/1/ns/net bash <<EOF
set -e  # 遇到错误立即退出
cd "${INSTALL_EXEC_DIR}" && ./install.sh \
    ${registry_arg} \
    --enableHttps="${ENABLE_HTTPS:-false}" \
    --repo="${HELM_REPORITORY_URL:-}" \
    --harborAdminPassword="${HARBOR_ADMIN_PASSWORD:-}" \
    --harborRegistryPassword="${HARBOR_REGISTRY_PASSWORD:-}" \
    --harborDatabasePassword="${HARBOR_DATABASE_PASSWORD:-}" \
    --harborRegistryPvSize="${HARBOR_REGISTRY_PV_SIZE:-100Gi}" \
    --harborJobservicePvSize="${HARBOR_JOBSERVICE_PV_SIZE:-10Gi}" \
    --harborJobservicePvcSize="${HARBOR_JOBSERVICE_PVC_SIZE:-10Gi}" \
    --harborDatabasePvSize="${HARBOR_DATABASE_PV_SIZE:-100Gi}" \
    --harborDatabasePvcSize="${HARBOR_DATABASE_PVC_SIZE:-100Gi}" \
    --harborRegistryPvcSize="${HARBOR_REGISTRY_PVC_SIZE:-100Gi}" \
    --harborChartmuseumPvSize="${HARBOR_CHARTMUSEUM_PV_SIZE:-50Gi}" \
    --harborChartmuseumPvcSize="${HARBOR_CHARTMUSEUM_PVC_SIZE:-50Gi}" \
    --harborRedisPvSize="${HARBOR_REDIS_PV_SIZE:-10Gi}" \
    --harborRedisPvcSize="${HARBOR_REDIS_PVC_SIZE:-10Gi}" \
    --oauthCertsExpirationTime="${OAUTH_CERTS_EXPIRATION_TIME:-87600h}"
EOF
    then
        fatal_log "Failed to install openfuyao-system"
        remove_tmp_files
        return 1
    fi

    info_log "Successfully installed openfuyao-system"

    # 清理临时文件
    remove_tmp_files
    return 0
}

function uninstall_openfuyao_system() {
    info_log "start uninstall openfuyao-system"

    copy_to_host || {
        fatal_log "failed to copy installation files to host"
        return 1
    }

    local registry_cmd=""
    if [ -n "${OPENFUYAO_REGISTRY}" ]; then
        registry_cmd="-r \"${OPENFUYAO_REGISTRY}\""
    fi

    if ! nsenter -i/proc/1/ns/ipc -m/proc/1/ns/mnt -n/proc/1/ns/net bash -c "cd ${INSTALL_EXEC_DIR} && ./uninstall.sh ${registry_cmd}"; then
        fatal_log "failed to uninstall openfuyao-system"
        remove_tmp_files
        return 1
    fi

    remove_tmp_files
    return 0
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
        info_log "uninstall openfuyao-system"
        uninstall_openfuyao_system
    elif [ "${operate}" == "install" ]; then
        info_log "install openfuyao-system"
        install_openfuyao_system
    else
        info_log "without operation, use default install operation"
        install_openfuyao_system
    fi
}

ENTRYPOINT_SHELL=$(getopt -n entrypoint.sh -o o:h --long operate:,help -- "$@")
[ $? -ne 0 ] && exit 1
eval set -- "$ENTRYPOINT_SHELL"
main "$@"