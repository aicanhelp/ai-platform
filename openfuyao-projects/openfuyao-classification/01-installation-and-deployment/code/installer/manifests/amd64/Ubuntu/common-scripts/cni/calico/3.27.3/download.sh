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

source ../../utils/consts.sh
source ../../utils/log.sh "calico"
source ./consts.sh

{{ if .http_proxy }}
export HTTP_PROXY="{{ .http_proxy }}"
{{ end }}
{{ if .https_proxy }}
export HTTPS_PROXY="{{ .htts_proxy }}"
{{ end }}
{{ if .no_proxy }}
export NO_PROXY="{{ .no_proxy }}"
{{ end }}

# 下载镜像文件
function download_image() {
    local image_path="image"
    sudo mkdir ${image_path} && cd ${image_path}

    local images=(
        ${FUYAO_RGISTRY}/calico/kube-controllers:v${VERSION}
        ${FUYAO_RGISTRY}/calico/node:v${VERSION}
        ${FUYAO_RGISTRY}/calico/cni:v${VERSION}
        ${FUYAO_RGISTRY}/calico/typha:v${VERSION}
    )

    for image_name in "${images[@]}"; do
        local attempts=0
        while [ $attempts -lt 3 ]; do
            sudo nerdctl pull $image_name --platform=linux/arm64,linux/amd64
            if [ $? -eq 0 ]; then
                local key=$(echo $image_name | awk -F '/' '{print $4}' | awk -F ':' '{print $1}')
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
        local key=$(echo $image_name | awk -F '/' '{print $4}' | awk -F ':' '{print $1}')
        if [ -f "$key.tar" ]; then
            info_log "Image $key saved to $key.tar successfully."
        else
            fatal_log "Failed to save image $key to $key.tar."
        fi
    done

    cd ../
}

# 下载yaml文件
function exec_download_yaml() {
    # 下载yaml文件
    local yaml_path="yaml"
    sudo mkdir ${yaml_path} && cd ${yaml_path}

    local urls=(
        "${FUYAO_OBS}/projectcalico/calico/releases/${VERSION}/calico-yaml.tar.gz"
    )

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
        fatal_log "base_path need have value"
    fi
    sudo mkdir -p ${base_path}
    cd ${base_path}

    # 在线安装无需提前下载镜像
    if [ -n "${download_image}" ]; then
        download_image
    fi

    if [ -n "${download_yaml}" ]; then
        exec_download_yaml
    fi

    exit 0
}

DOWNLOAD_SHELL=$(getopt -n download.sh -o p:i:c:y:b:h --long path:,image:,chart:,yaml:,binary:,help -- "$@")
[ $? -ne 0 ] && exit 1
eval set -- "$DOWNLOAD_SHELL"
main $@
