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

source ../../utils/log.sh "openfuyao-system"
source ./consts.sh
source ../../utils/consts.sh

{{ if .http_proxy }}
export HTTP_PROXY="{{ .http_proxy }}"
{{ end }}
{{ if .https_proxy }}
export HTTPS_PROXY="{{ .htts_proxy }}"
{{ end }}
{{ if .no_proxy }}
export NO_PROXY="{{ .no_proxy }}"
{{ end }}

CFSSL_VERSION="1.6.4"
JQ_VERSION="1.7.1"

function download_with_retry() {
    local urls=("$@")
    for url in "${urls[@]}"; do
        retry=0
        while [ $retry -lt 3 ]; do
            sudo -E curl -sOL $url
            if [ $? -eq 0 ]; then
                info_log "File downloaded successfully: $(basename $url)"
                if [ -f $(basename $url) ]; then
                    info_log "File $(basename $url) saved to local directory."
                    break
                else
                    info_log "File $(basename $url) not saved to local directory."
                fi
            else
                info_log "Download failed. Retrying..."
                ((retry++))
            fi
        done

        if [ $retry -eq 3 ]; then
            fatal_log "Failed to download file: $(basename $url) after 3 attempts. Exiting program."
        fi
    done
}

# 执行下载
function exec_download_bin() {
    local arch=$1
    
    info_log "Start downloading the ${arch} architecture binary files."

    local urls=(
        "${FUYAO_OBS}/jqlang/jq/releases/download/v${JQ_VERSION}/jq-linux-${arch}"
        "${FUYAO_OBS}/cloudflare/cfssl/releases/download/v${CFSSL_VERSION}/cfssl-certinfo_${CFSSL_VERSION}_linux_${arch}"
        "${FUYAO_OBS}/cloudflare/cfssl/releases/download/v${CFSSL_VERSION}/cfssljson_${CFSSL_VERSION}_linux_${arch}"
        "${FUYAO_OBS}/cloudflare/cfssl/releases/download/v${CFSSL_VERSION}/cfssl_${CFSSL_VERSION}_linux_${arch}"
    )
    download_with_retry "${urls[@]}"

    mv jq-linux-${arch} jq
    mv cfssl-certinfo_${CFSSL_VERSION}_linux_${arch} cfssl-certinfo
    mv cfssljson_${CFSSL_VERSION}_linux_${arch} cfssljson
    mv cfssl_${CFSSL_VERSION}_linux_${arch} cfssl
}

# 下载二进制文件
function download_bin() {
    local arch=$1

    local arch_path="${arch}-bin"
    sudo mkdir -p ${arch_path}
    cd ${arch_path}

    exec_download_bin ${arch}

    cd ../
}

# 下载镜像文件
function exec_download_image() {
    sudo mkdir -p ${IMAGE_PATH} && cd ${IMAGE_PATH}
    # 下载镜像
    images=(
        # openfuyao-system
        ${FUYAO_RGISTRY}/console-service:"${CONSOLE_SERVICE_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/console-website:"${CONSOLE_WEBSITE_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/monitoring-service:"${MONITORING_SERVICE_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/oauth-webhook:"${OAUTH_WEBHOOK_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/oauth-server:"${OAUTH_SERVER_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/application-management-service:"${APPLICATION_MANAGEMENT_SERVICE_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/marketplace-service:"${MARKETPLACE_SERVICE_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/plugin-management-service:"${PLUGIN_MANAGEMENT_SERVICE_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/oauth-proxy:"${OAUTH_PROXY_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/user-management-operator:"${USER_MANAGEMENT_OPERATOR_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/web-terminal-service:"${WEB_TERMINAL_SERVICE_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/kubectl-openfuyao:"${KUBECTL_OPENFUYAO_IMAGE_TAG}"
        # prometheus 镜像
        ${FUYAO_RGISTRY}/prometheus/alertmanager:v0.27.0
        ${FUYAO_RGISTRY}/prometheus/blackbox-exporter:v0.25.0
        ${FUYAO_RGISTRY}/prometheus/node-exporter:v1.8.1
        ${FUYAO_RGISTRY}/prometheus/prometheus:v2.52.0
        ${FUYAO_RGISTRY}/prometheus-operator/prometheus-operator:v0.74.0
        ${FUYAO_RGISTRY}/prometheus-operator/prometheus-config-reloader:v0.74.0
        ${FUYAO_RGISTRY}/brancz/kube-rbac-proxy:v0.18.0
        ${FUYAO_RGISTRY}/kube-state-metrics/kube-state-metrics:v2.12.0
        ${FUYAO_RGISTRY}/jimmidyson/configmap-reload:v0.13.0

        ${FUYAO_RGISTRY}/metrics-server/metrics-server:v0.7.1
        ${FUYAO_RGISTRY}/busybox/busybox:1.36.1

        # harbor 镜像
        ${FUYAO_RGISTRY}/harbor/harbor-exporter:${HARBOR_IMAGE_TAG}
        ${FUYAO_RGISTRY}/harbor/redis-photon:${HARBOR_IMAGE_TAG}
        ${FUYAO_RGISTRY}/harbor/harbor-db:${HARBOR_IMAGE_TAG}
        ${FUYAO_RGISTRY}/harbor/chartmuseum-photon:${HARBOR_IMAGE_TAG}
        ${FUYAO_RGISTRY}/harbor/harbor-registryctl:${HARBOR_IMAGE_TAG}
        ${FUYAO_RGISTRY}/harbor/registry-photon:${HARBOR_IMAGE_TAG}
        ${FUYAO_RGISTRY}/harbor/harbor-jobservice:${HARBOR_IMAGE_TAG}
        ${FUYAO_RGISTRY}/harbor/harbor-core:${HARBOR_IMAGE_TAG}
        ${FUYAO_RGISTRY}/harbor/harbor-portal:${HARBOR_IMAGE_TAG}
        ${FUYAO_RGISTRY}/harbor/nginx-photon:${HARBOR_IMAGE_TAG}
    )
    for image_name in "${images[@]}"; do
        local attempts=0
        while [ $attempts -lt 3 ]; do
            sudo nerdctl pull $image_name --platform=linux/arm64,linux/amd64
            if [ $? -eq 0 ]; then
                local key=$(echo $image_name | awk -F '/' '{print $NF}' | awk -F ':' '{print $1}')
                sudo nerdctl save -o $key.tar $image_name --platform=linux/arm64,linux/amd64
                break
            else
                ((attempts++))
            fi
        done

        if [ $attempts -eq 3 ]; then
            fatal_log "Failed to pull $image_name after 3 attempts."
        fi

        # 检查保存到本地的镜像是否存在
        local key=$(echo $image_name | awk -F '/' '{print $NF}' | awk -F ':' '{print $1}')
        if [ -f "$key.tar" ]; then
            info_log "Image $key saved to $key.tar successfully."
        else
            fatal_log "Failed to save image $key to $key.tar."
        fi
    done

    cd ../
}

# 下载chart
function exec_download_chart() {
    sudo mkdir -p  ${CHART_PATH} && cd ${CHART_PATH}

    # 使用关联数组，key为chart名称，value为chart版本
    declare -A chart_map
    chart_map["${OAUTH_WEBHOOK_CHART_NAME}"]="${OAUTH_WEBHOOK_CHART_VERSION}"
    chart_map["${OAUTH_SERVER_CHART_NAME}"]="${OAUTH_SERVER_CHART_VERSION}"
    chart_map["${HARBOR_CHART_NAME}"]="${LOCAL_HARBOR_CHART_VERSION}"
    chart_map["${CONSOLE_WEBSITE_CHART_NAME}"]="${CONSOLE_WEBSITE_CHART_VERSION}"
    chart_map["${MONITORING_SERVICE_CHART_NAME}"]="${MONITORING_SERVICE_CHART_VERSION}"
    chart_map["${CONSOLE_SERVICE_CHART_NAME}"]="${CONSOLE_SERVICE_CHART_VERSION}"
    chart_map["${MARKETPLACE_SERVICE_CHART_NAME}"]="${MARKETPLACE_SERVICE_CHART_VERSION}"
    chart_map["${APPLICATION_MANAGEMENT_SERVICE_CHART_NAME}"]="${APPLICATION_MANAGEMENT_SERVICE_CHART_VERSION}"
    chart_map["${PLUGIN_MANAGEMENT_SERVICE_CHART_NAME}"]="${PLUGIN_MANAGEMENT_SERVICE_CHART_VERSION}"
    chart_map["${USER_MANAGEMENT_OPERATOR_CHART_NAME}"]="${USER_MANAGEMENT_OPERATOR_CHART_VERSION}"
    chart_map["${WEB_TERMINAL_SERVICE_CHART_NAME}"]="${WEB_TERMINAL_SERVICE_CHART_VERSION}"

    for chart_name in "${!chart_map[@]}"; do
        info_log "Start downloading chart ${chart_name}:${chart_map[$chart_name]}"
        local attempts=0
        while [ $attempts -lt 3 ]; do
            helm fetch ${chart_name} --repo ${FUYAO_REPO} --version "${chart_map[$chart_name]}"
            if [ $? -eq 0 ]; then
                break
            else
                ((attempts++))
                sleep 2
            fi
        done

        if [ $attempts -eq 3 ]; then
            fatal_log "Failed to download $chart_name after 3 attempts."
        fi

        # 检查保存到本地的chart是否存在
        local chart_file="${chart_name}-${chart_map[$chart_name]}.tgz"
        if [ -f "$chart_file" ]; then
            info_log "Chart $chart_file saved to $chart_file successfully."
        else
            fatal_log "Failed to save chart $chart_file to $chart_file."
        fi
    done

    info_log "Download chart successfully"
    cd ../
}

# 下载yaml文件
function exec_download_yaml() {
    # 下载yaml文件
    local yaml_path="yaml"
    sudo mkdir ${yaml_path} && cd ${yaml_path}

    local urls=(
        "${FUYAO_OBS}/kubernetes-sigs/metrics-server/releases/v0.7.1/metrics-server-yaml.tar.gz"
        "${FUYAO_OBS}/prometheus-operator/kube-prometheus/releases/main/kube-prometheus-yaml.tar.gz"
    )
    download_with_retry "${urls[@]}"
    cd ../
}

function usage() {
    echo "see source file"
}

# get-opt 参数
main(){
    while true
    do
        case "$1" in
        -i|--image)  # 镜像
            local download_image="$2"
            shift
            ;;
        -p|--path)  # 下载的根路径 eg: ./fuyao-openEuler-22.03-offline-artifact/cri/containerd
            local base_path="$2"
            shift
            ;;
        -c|--chart) # chart
            local download_chart="$2"
            shift
            ;;
        -y|--yaml) # yaml
            local download_yaml="$2"
            shift
            ;;
        -b|--binary)
            local download_binary="$2"
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

    if [ -z "${base_path}" ];then
        echo "base_path need have value"
        exit 1
    fi
    # 创建存储文件夹
    sudo mkdir -p ${base_path}
    cd ${base_path}
    # 卸载无需下载二进制文件
    if [ -n "${download_binary}" ]; then
        download_bin "arm64"
        download_bin "amd64"
    fi

    # 在线安装无需提前下载镜像
    if [ -n "${download_image}" ]; then
        exec_download_image
    fi

    # 在线安装无需提前下载chat
    if [ -n "${download_chart}" ]; then
        exec_download_chart
    fi

    exec_download_yaml
    exit 0
}

DOWNLOAD_SHELL=$(getopt -n download.sh -o p:i:c:y:b:h --long path:,image:,chart:,yaml:,binary:,help -- "$@")
[ $? -ne 0 ] && exit 1
eval set -- "$DOWNLOAD_SHELL"
main $@
