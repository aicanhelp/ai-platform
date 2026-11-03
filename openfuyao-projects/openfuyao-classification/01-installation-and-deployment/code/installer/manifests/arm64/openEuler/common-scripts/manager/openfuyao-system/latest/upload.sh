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

MAX_RETRIES=3

function upload_image() {
    # load tag push
    IMAGES=(
        # openfuyao-system 镜像
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
    )

    for pullimageName in "${IMAGES[@]}"; do
        # load
        key=$(echo "$pullimageName" | awk -F '/' '{print $NF}' | awk -F ':' '{print $1}')
        sudo nerdctl load -i $key.tar --platform=linux/arm64,linux/amd64
        NAME_TAG=$(echo "$pullimageName" | awk -F '/' '{print $NF}')
        NAME="${REGISTRY}/${NAME_TAG}"
        # tag
        sudo nerdctl tag "${pullimageName}" "${NAME}"

        retry=0
        success=false

        while [ $retry -lt $MAX_RETRIES ]; do
            # push
            sudo nerdctl push "${NAME}" --platform=linux/arm64,linux/amd64
            if [ $? -eq 0 ]; then
                info_log "Upload successful for ${NAME}"
                success=true
                break
            else
                info_log "Upload failed for ${NAME}. Retrying..."
                ((retry++))
            fi
        done

        if [ $retry -eq $MAX_RETRIES ] && [ $success = false ]; then
            fatal_log "Failed to upload ${NAME} after $MAX_RETRIES attempts. Terminating upload process."
        fi
    done

    IMAGES=(
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
        ${FUYAO_RGISTRY}/harbor/harbor-exporter:"${HARBOR_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/harbor/redis-photon:"${HARBOR_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/harbor/harbor-db:"${HARBOR_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/harbor/chartmuseum-photon:"${HARBOR_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/harbor/harbor-registryctl:"${HARBOR_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/harbor/registry-photon:"${HARBOR_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/harbor/harbor-jobservice:"${HARBOR_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/harbor/harbor-core:"${HARBOR_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/harbor/harbor-portal:"${HARBOR_IMAGE_TAG}"
        ${FUYAO_RGISTRY}/harbor/nginx-photon:"${HARBOR_IMAGE_TAG}"
    )

    for pullimageName in "${IMAGES[@]}"; do
        # load
        key=$(echo "$pullimageName" | awk -F '/' '{print $NF}' | awk -F ':' '{print $1}')
        sudo nerdctl load -i $key.tar --platform=linux/arm64,linux/amd64
        NAME_TAG=$(echo "$pullimageName" | awk -F '/' '{print $NF}')
        COMPONENT=$(echo "${pullimageName}" | awk -F '/' '{print $3}')
        NAME="${REGISTRY}/${COMPONENT}/${NAME_TAG}"
        # tag
        sudo nerdctl tag "${pullimageName}" "${NAME}"

        retry=0
        success=false

        while [ $retry -lt $MAX_RETRIES ]; do
            # push
            sudo nerdctl push "${NAME}" --platform=linux/arm64,linux/amd64
            if [ $? -eq 0 ]; then
                info_log "Upload successful for ${NAME}"
                success=true
                break
            else
                info_log "Upload failed for ${NAME}. Retrying..."
                ((retry++))
            fi
        done

        if [ $retry -eq $MAX_RETRIES ] && [ $success = false ]; then
            fatal_log "Failed to upload ${NAME} after $MAX_RETRIES attempts. Terminating upload process."
        fi
    done
}

function usage() {
    echo "see source file"
}

# get-opt 参数
main(){
    while true
    do
        case "$1" in
        -r|--registry)  # 镜像仓库地址
            local registry_url="$2"
            shift
            ;;
        -p|--path)  # 镜像存放路径
            local base_path="$2"
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

    if [ -z "${base_path}" ] || [ -z "${registry_url}" ];then
        fatal_log "path or registry is invalid"
    fi

    cd ${base_path} || fatal_log "image path not exist"
    export REGISTRY=${registry_url}
    upload_image
    exit 0
}

UPLOAD_SHELL=$(getopt -n upload.sh -o p:r:h --long path:,registry:,help -- "$@")
[ $? -ne 0 ] && exit 1
eval set -- "$UPLOAD_SHELL"
main $@
