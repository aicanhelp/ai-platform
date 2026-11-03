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

source ./log.sh
source ./consts.sh
source ./preinstall.sh
source ./utils/param-modification.sh
source ./utils/certificates-secrets.sh
source ./utils/create-pv.sh

VM_VALUES_YAML="${CHART_PATH}/${VMS_CHART_NAME}/values.yaml"

OS=$(echo `uname`|tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
esac

function setup_values_yaml_config() {
    local vmClusterReplicationFactor=${vmClusterReplicationFactor}
    local vmStorageReplicaCount=${vmStorageReplicaCount}
    local vmSelectReplicaCount=${vmSelectReplicaCount}
    local vmInsertReplicaCount=${vmInsertReplicaCount}
    local vmClusterRetentionPeriod=${vmClusterRetentionPeriod}
    local vmStorageStorageSize=${vmStorageStorageSize}
    local vmStorageMemorySize=${vmStorageMemorySize}
    local vmStorageCPUCount=${vmStorageCPUCount}
    local vmSelectMemorySize=${vmSelectMemorySize}
    local vmSelectCpuCount=${vmSelectCpuCount}
    local vmSelectStorageSize=${vmSelectStorageSize}
    local vmInsertMemorySize=${vmInsertMemorySize}
    local vmInsertCpuCount=${vmInsertCpuCount}
    local vmAgentReplicaCount=${vmAgentReplicaCount}
    local vmAgentShareCount=${vmAgentShareCount}
    local vmAgentStorageSize=${vmAgentStorageSize}
    local vmAgentMemorySize=${vmAgentMemorySize}
    local vmAgentCpuCount=${vmAgentCpuCount}
    local grafanaNodePort=${grafanaNodePort}
    local kubeStateMetricsAutoSharding=${kubeStateMetricsAutoSharding}
    local kubeStateMetricsReplicaCount=${kubeStateMetricsReplicaCount}
    local kubeStateMetricsMemorySize=${kubeStateMetricsMemorySize}
    local kubeStateMetricsCpuCount=${kubeStateMetricsCpuCount}
    local vmAgentScrapeInterval=${vmAgentScrapeInterval}
    local vmAlertMemorySize=${vmAlertMemorySize}
    local vmAlertCpuCount=${vmAlertCpuCount}
    local vmAlertManagerMemorySize=${vmAlertManagerMemorySize}
    local vmAlertManagerCpuCount=${vmAlertManagerCpuCount}
    local vmAlertReplicaCount=${vmAlertReplicaCount}
    local vmAlertManagerReplicaCount=${vmAlertManagerReplicaCount}
    local vmSingleStorageSize=${vmSingleStorageSize}
    local vmAgentAllowStatefulSet=${vmAgentAllowStatefulSet}
    local useVMSingle=${useVMSingle}
    if [ "$IS_ONLINE" = "false" ]; then
        local imageRepo=${OPENFUYAO_REGISTRY}
    else
        local imageRepo=${OEPKGS_REGISTRY}
    fi
    
    modify_values_yaml
}

function setup_create_pv_config() {
    local vmAgentReplicaCount=${vmAgentReplicaCount}
    local vmAgentShareCount=${vmAgentShareCount}
    local vmAgentAllowStatefulSet=${vmAgentAllowStatefulSet}
    local vmAgentStorageSize=${vmAgentStorageSize}
    local vmSelectStorageSize=${vmSelectStorageSize}
    local vmStorageStorageSize=${vmStorageStorageSize}
    local vmSingleStorageSize=${vmSingleStorageSize}
    local deployVMAgentNodeNames=${deployVMAgentNodeNames}
    local deployVMSelectNodeNames=${deployVMSelectNodeNames}
    local deployVMStorageNodeNames=${deployVMStorageNodeNames}
    local DeployVMSingleNodeNames=${DeployVMSingleNodeNames}
    local useVMSingle=${useVMSingle}

    create_pv
}

function setup_configure_certificates_secrets_config() {
    local etcdNodeIP=${etcdNodeIP}
    local etcdDataNodeIP=${etcdDataNodeIP}
    local etcdEventsNodeIP=${etcdEventsNodeIP}
    local controllerManagerIP=${controllerManagerIP}
    local kubeSchedulerIP=${kubeSchedulerIP}

    configure_certificates_secrets
}  

function download_charts_with_retry() {
    local chart_name=$1
    local chart_version=$2

    if [ ! -d "${CHART_PATH}" ]; then
        mkdir -p "${CHART_PATH}"
    fi

    local cur_path=$(pwd)
    cd "${CHART_PATH}"  || fatal_log "Failed to change directory to ${CHART_PATH}"
    if [ -f "${chart_name}-${chart_version}.tgz" ]; then
        info_log "${chart_name}-${chart_version}.tgz already exists, skip downloading"
        cd "${cur_path}" || fatal_log "Failed to change directory to ${cur_path}"
        return
    fi
    info_log "Downloading ${chart_name} chart"
    
    # 添加helm仓库
    helm repo add vm https://victoriametrics.github.io/helm-charts/
    helm repo update
    
    local attempts=0
    while [ $attempts -lt 3 ]; do

        if [ "$IS_ONLINE" = "false" ]; then
            helm fetch "${chart_name}" --repo "${OPENFUYAO_REPO}" --version "${chart_version}"
        else
            helm pull vm/"${chart_name}" --version "${chart_version}"
        fi
        
        if [ $? -eq 0 ]; then
            info_log "Successfully downloaded ${chart_name} chart"
            break
        else
            ((attempts++))
            sleep 2
        fi
    done

    if [ $attempts -eq 3 ]; then
        fatal_log "Failed to download $chart_name after 3 attempts."
    fi

    cd "${cur_path}" || fatal_log "Failed to change directory to ${cur_path}"
    return
}

function add_grafana_dashboard() {
    local cur_path=$(pwd)
    local dashboard_dir="${CHART_PATH}/${GRAFANA_ADD_PANEL_DIR}"

    if [ ! -d "${dashboard_dir}" ]; then
        mkdir -p "${dashboard_dir}"
    fi
    
    for panel in "${PANEL_LIST[@]}"; do
        cp "./resources/${panel}" "${dashboard_dir}/${panel}" || fatal_log "Failed to copy ${panel} to ${dashboard_dir}"
        info_log "Successfully copy ${panel} to ${dashboard_dir}"
    done

    info_log "Successfully add extra grafana dashboards"

    cd "${cur_path}" || fatal_log "Failed to change directory to ${cur_path}"
    return
}


# 安装victoria metrics
function install_victoria_metrics() {
    
    info_log "Start installing victoriametrics"

    download_charts_with_retry ${VMS_CHART_NAME} ${VERSION}
    tar -zxvf ${CHART_PATH}/${VMS_CHART_NAME}-${VERSION}.tgz -C ${CHART_PATH}

    # 添加grafana面板配置文件
    add_grafana_dashboard

    # 修改values.yaml文件
    setup_values_yaml_config

    helm install vm ${CHART_PATH}/${VMS_CHART_NAME} -f ${VM_VALUES_YAML} -n ${VM_NAMESPACE} --create-namespace
}

function install_yq() {
    if command -v yq >/dev/null 2>&1; then
        info_log "yq already installed"
        return
    fi

    info_log "Start installing yq"
    sudo mv -f ./${ARCH}-bin/yq_linux_${ARCH} /usr/local/bin/yq
    sudo chmod +x /usr/local/bin/yq
    info_log "success install yq"
}

function install_helm() {
    if command -v helm >/dev/null 2>&1; then
        info_log "helm already installed"
        return
    fi

    info_log "Start installing helm"
    sudo tar -zxvf ./${ARCH}-bin/helm-v3.14.2-"${OS}"-${ARCH}.tar.gz
    sudo mv "${OS}"-$ARCH/helm /usr/local/bin/helm
    info_log "success install helm"
}


# get-opt 参数
main() {
    while true
    do
        case "$1" in
        -r|--registry)  # 镜像仓库地址
            OPENFUYAO_REGISTRY="$2"
            info_log "registry is ${OPENFUYAO_REGISTRY}"
            shift
            ;;
        --version)
            VERSION="$2"
            info_log "version is ${VERSION}"
            shift
            ;;
        --useVMSingle)
            useVMSingle="$2"
            info_log "useVMSingle is ${useVMSingle}"
            shift
            ;;
        --vmSingleStorageSize)
            vmSingleStorageSize="$2"
            info_log "vmSingleStorageSize is ${vmSingleStorageSize}"
            shift
            ;;
        --vmAgentAllowStatefulSet)
            vmAgentAllowStatefulSet="$2"
            info_log "vmAgentAllowStatefulSet is ${vmAgentAllowStatefulSet}"
            shift
            ;;
        --vmAgentCpuCount)
            vmAgentCpuCount="$2"
            info_log "vmAgentCpuCount is ${vmAgentCpuCount}"
            shift
            ;;
        --vmAgentMemorySize)
            vmAgentMemorySize="$2"
            info_log "vmAgentMemorySize is ${vmAgentMemorySize}"
            shift
            ;;
        --vmAgentStorageSize)
            vmAgentStorageSize="$2"
            info_log "vmAgentStorageSize is ${vmAgentStorageSize}"
            shift
            ;;
        --vmAgentReplicaCount)
            vmAgentReplicaCount="$2"
            info_log "vmAgentReplicaCount is ${vmAgentReplicaCount}"
            shift
            ;;
        --vmAgentShareCount)
            vmAgentShareCount="$2"
            info_log "vmAgentShareCount is ${vmAgentShareCount}"
            shift
            ;;
        --vmAgentScrapeInterval)
            vmAgentScrapeInterval="$2"
            info_log "vmAgentScrapeInterval is ${vmAgentScrapeInterval}"
            shift
            ;;
        --vmSelectCpuCount)
            vmSelectCpuCount="$2"
            info_log "vmSelectCpuCount is ${vmSelectCpuCount}"
            shift
            ;;
        --vmSelectMemorySize)
            vmSelectMemorySize="$2"
            info_log "vmSelectMemorySize is ${vmSelectMemorySize}"
            shift
            ;;
        --vmSelectStorageSize)
            vmSelectStorageSize="$2"
            info_log "vmSelectStorageSize is ${vmSelectStorageSize}"
            shift
            ;; 
        --vmSelectReplicaCount)
            vmSelectReplicaCount="$2"
            info_log "vmSelectReplicaCount is ${vmSelectReplicaCount}"
            shift
            ;;
        --vmStorageCPUCount)
            vmStorageCPUCount="$2"
            info_log "vmStorageCPUCount is ${vmStorageCPUCount}"
            shift
            ;;
        --vmStorageMemorySize)
            vmStorageMemorySize="$2"
            info_log "vmStorageMemorySize is ${vmStorageMemorySize}"
            shift
            ;;
        --vmStorageStorageSize)
            vmStorageStorageSize="$2"
            info_log "vmStorageStorageSize is ${vmStorageStorageSize}"
            shift
            ;;
        --vmStorageReplicaCount)
            vmStorageReplicaCount="$2"
            info_log "vmStorageReplicaCount is ${vmStorageReplicaCount}"
            shift
            ;;
        --vmInsertReplicaCount)
            vmInsertReplicaCount="$2"
            info_log "vmInsertReplicaCount is ${vmInsertReplicaCount}"
            shift
            ;;
        --vmInsertCpuCount)
            vmInsertCpuCount="$2"
            info_log "vmInsertCpuCount is ${vmInsertCpuCount}"
            shift
            ;;
        --vmInsertMemorySize)
            vmInsertMemorySize="$2"
            info_log "vmInsertMemorySize is ${vmInsertMemorySize}" 
            shift
            ;;
        --vmAlertReplicaCount)
            vmAlertReplicaCount="$2"
            info_log "vmAlertReplicaCount is ${vmAlertReplicaCount}"
            shift
            ;;
        --vmAlertCpuCount)
            vmAlertCpuCount="$2"
            info_log "vmAlertCpuCount is ${vmAlertCpuCount}"
            shift
            ;;
        --vmAlertMemorySize)
            vmAlertMemorySize="$2"
            info_log "vmAlertMemorySize is ${vmAlertMemorySize}"
            shift
            ;;
        --vmAlertManagerReplicaCount)
            vmAlertManagerReplicaCount="$2"
            info_log "vmAlertManagerReplicaCount is ${vmAlertManagerReplicaCount}"
            shift
            ;;
        --vmAlertManagerCpuCount)
            vmAlertManagerCpuCount="$2"
            info_log "vmAlertManagerCpuCount is ${vmAlertManagerCpuCount}"
            shift
            ;;
        --vmAlertManagerMemorySize)
            vmAlertManagerMemorySize="$2"
            info_log "vmAlertManagerMemorySize is ${vmAlertManagerMemorySize}"
            shift
            ;;
        --vmClusterRetentionPeriod)
            vmClusterRetentionPeriod="$2"
            info_log "vmClusterRetentionPeriod is ${vmClusterRetentionPeriod}"
            shift
            ;;
        --vmClusterReplicationFactor)
            vmClusterReplicationFactor="$2"
            info_log "vmClusterReplicationFactor is ${vmClusterReplicationFactor}"
            shift
            ;;
        --grafanaNodePort)
            grafanaNodePort="$2"
            info_log "grafanaNodePort is ${grafanaNodePort}"
            shift
            ;;
        --kubeStateMetricsAutoSharding)
            kubeStateMetricsAutoSharding="$2"
            info_log "kubeStateMetricsAutoSharding is ${kubeStateMetricsAutoSharding}"
            shift
            ;;
        --kubeStateMetricsReplicaCount)
            kubeStateMetricsReplicaCount="$2"
            info_log "kubeStateMetricsReplicaCount is ${kubeStateMetricsReplicaCount}"
            shift
            ;;
        --kubeStateMetricsCpuCount)
            kubeStateMetricsCpuCount="$2"
            info_log "kubeStateMetricsCpuCount is ${kubeStateMetricsCpuCount}"
            shift
            ;;
        --kubeStateMetricsMemorySize)
            kubeStateMetricsMemorySize="$2"
            info_log "kubeStateMetricsMemorySize is ${kubeStateMetricsMemorySize}"
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

    if [ -z "${REGISTRY}" ];then
        REGISTRY="${FUYAO_RGISTRY}"
    fi

    # 根据REGISTRY参数确定在线安装还是离线安装，在线：cr.openfuyao.cn/openfuyao  离线：deploy.bocloud.k8s:40443/kubernetes
    substring="cr.openfuyao.cn"
    if [[ "${REGISTRY}" == *"${substring}"* ]]; then
        echo "online deploy"
        IS_ONLINE="true"
    else
        echo "offline deploy"
        IS_ONLINE="false"
    fi

    REGISTRY="${REGISTRY%/}"


    if [ -z "${OPENFUYAO_REPO}" ];then
        OPENFUYAO_REPO="cr.openfuyao.cn"
    fi

    install_yq
    install_helm
    # 安装victoriametrics
    install_victoria_metrics
    # 创建pv
    setup_create_pv_config
    # 配置证书和secret
    setup_configure_certificates_secrets_config

    # 完成安装后操作
    info_log "successfully installed victoriametrics"
    ./postinstall.sh
}

INSTALL_SHELL=$(getopt -n install.sh -o r: --long registry:, \
version:,useVMSingle:vmSingleStorageSize:, \
vmAgentAllowStatefulSet:,vmAgentCpuCount:,vmAgentMemorySize:,vmAgentStorageSize:,vmAgentReplicaCount:,vmAgentShareCount:,vmAgentScrapeInterval:, \
vmSelectCpuCount:,vmSelectMemorySize:,vmSelectStorageSize:,vmSelectReplicaCount:, \
vmStorageCPUCount:,vmStorageMemorySize:,vmStorageStorageSize:,vmStorageReplicaCount:, \
vmInsertReplicaCount:,vmInsertCpuCount:,vmInsertMemorySize:, \
vmAlertReplicaCount:,vmAlertCpuCount:,vmAlertMemorySize:, \
vmAlertManagerReplicaCount:,vmAlertManagerCpuCount:,vmAlertManagerMemorySize:, \
vmClusterRetentionPeriod:,vmClusterReplicationFactor:, \
grafanaNodePort:,kubeStateMetricsAutoSharding:,kubeStateMetricsReplicaCount:,kubeStateMetricsCpuCount:,kubeStateMetricsMemorySize:, \
help -- "$@")
[ $? -ne 0 ] && fatal_log "failed to parse command line options"
eval set -- "$INSTALL_SHELL"
main "$@"