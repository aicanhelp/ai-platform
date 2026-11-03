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
SUB_EXEC_DIR="openFuyao/victoriametric-install"
INSTALL_EXEC_DIR="$BASE_INSTALL_EXEC_DIR/$SUB_EXEC_DIR"

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
    nsenter -i/proc/1/ns/ipc \
            -m/proc/1/ns/mnt \
            -n/proc/1/ns/net \
            mkdir -p "${INSTALL_EXEC_DIR}"
    if [ $? -ne 0 ]; then
        fatal_log "failed to create directory on host"
    fi

    cp -rf /home/victoriametric/* /mnt/opt/${SUB_EXEC_DIR}
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

function install_victoriametric() {
    info_log "start install victoriametric"
    copy_to_host

    # 执行安装脚本
    nsenter -i/proc/1/ns/ipc \
            -m/proc/1/ns/mnt \
            -n/proc/1/ns/net \
            bash -c "cd ${INSTALL_EXEC_DIR} && \
            ./install.sh -r ${OPENFUYAO_REGISTRY} \
            --version=${VERSION} \
            --useVMSingle=${useVMSingle} \
            --vmSingleStorageSize=${vmSingleStorageSize} \
            --vmAgentAllowStatefulSet=${vmAgentAllowStatefulSet} \
            --vmAgentCpuCount=${vmAgentCpuCount} \
            --vmAgentMemorySize=${vmAgentMemorySize} \
            --vmAgentStorageSize=${vmAgentStorageSize} \
            --vmAgentReplicaCount=${vmAgentReplicaCount} \
            --vmAgentShareCount=${vmAgentShareCount} \
            --vmAgentScrapeInterval=${vmAgentScrapeInterval} \
            --vmSelectCpuCount=${vmSelectCpuCount} \
            --vmSelectMemorySize=${vmSelectMemorySize} \
            --vmSelectStorageSize=${vmSelectStorageSize} \
            --vmSelectReplicaCount=${vmSelectReplicaCount} \
            --vmStorageCPUCount=${vmStorageCPUCount} \
            --vmStorageMemorySize=${vmStorageMemorySize} \
            --vmStorageStorageSize=${vmStorageStorageSize} \
            --vmStorageReplicaCount=${vmStorageReplicaCount} \
            --vmInsertReplicaCount=${vmInsertReplicaCount} \
            --vmInsertCpuCount=${vmInsertCpuCount} \
            --vmInsertMemorySize=${vmInsertMemorySize} \
            --vmAlertReplicaCount=${vmAlertReplicaCount} \
            --vmAlertCpuCount=${vmAlertCpuCount} \
            --vmAlertMemorySize=${vmAlertMemorySize} \
            --vmAlertManagerReplicaCount=${vmAlertManagerReplicaCount} \
            --vmAlertManagerCpuCount=${vmAlertManagerCpuCount} \
            --vmAlertManagerMemorySize=${vmAlertManagerMemorySize} \
            --vmClusterRetentionPeriod=${vmClusterRetentionPeriod} \
            --vmClusterReplicationFactor=${vmClusterReplicationFactor} \
            --grafanaNodePort=${grafanaNodePort} \
            --kubeStateMetricsAutoSharding=${kubeStateMetricsAutoSharding} \
            --kubeStateMetricsReplicaCount=${kubeStateMetricsReplicaCount} \
            --kubeStateMetricsCpuCount=${kubeStateMetricsCpuCount} \
            --kubeStateMetricsMemorySize=${kubeStateMetricsMemorySize}"   
    if [ $? -ne 0 ]; then 
        fatal_log "failed to installed victoriametric"
    fi

    remove_tmp_files
}

function uninstall_victoriametric() {
    info_log "start uninstall victoriametric"
    copy_to_host

    nsenter -i/proc/1/ns/ipc \
            -m/proc/1/ns/mnt \
            -n/proc/1/ns/net \
            bash -c "cd ${INSTALL_EXEC_DIR} && ./uninstall.sh -r ${OPENFUYAO_REGISTRY}"
    if [ $? -ne 0 ]; then
        fatal_log "failed to uninstall victoriametric"
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
        info_log "uninstall victoriametric"
        uninstall_victoriametric
    elif [ "${operate}" == "install" ]; then
        info_log "install victoriametric"
        install_victoriametric
    else
        info_log "without operation, use default install operation"
        install_victoriametric
    fi
}

ENTRYPOINT_SHELL=$(getopt -n entrypoint.sh -o o:h --long operate:,help -- "$@")
[ $? -ne 0 ] && exit 1
eval set -- "$ENTRYPOINT_SHELL"
main "$@"
