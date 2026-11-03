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
source ../../utils/log.sh "ingress-nginx"
source ../../utils/consts.sh
source ./consts.sh

MAX_RETRIES=3

function upload_image() {
    # load tag push
    images=(
        ${FUYAO_RGISTRY}/ingress-nginx/kube-webhook-certgen:v${CERTGEN_VERSION}
        ${FUYAO_RGISTRY}/ingress-nginx/controller:v${VERSION}
    )

    for pullimageName in "${images[@]}"; do
        # load
        key=$(echo "$pullimageName" | awk -F '/' '{print $4}' | awk -F ':' '{print $1}')
        sudo nerdctl load -i $key.tar --platform=linux/arm64,linux/amd64
        NAME_TAG=$(echo "$pullimageName" | awk -F '/' '{print $4}')
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
