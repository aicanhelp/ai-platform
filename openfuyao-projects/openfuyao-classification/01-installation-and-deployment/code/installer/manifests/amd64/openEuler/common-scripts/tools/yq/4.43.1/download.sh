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

source ../../utils/log.sh "yq"
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

VERSION="4.43.1"

# 执行下载
function exec_download_bin() {
    local arch=$1

    local urls=(
        "${FUYAO_OBS}/mikefarah/yq/releases/download/v${VERSION}/yq_linux_${arch}"
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
    mv yq_linux_${arch} yq
    info_log "The ${arch} architecture binary files have been downloaded"
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

    exit 0
}

DOWNLOAD_SHELL=$(getopt -n download.sh -o p:i:c:y:b:h --long path:,image:,chart:,yaml:,binary:,help -- "$@")
[ $? -ne 0 ] && exit 1
eval set -- "$DOWNLOAD_SHELL"
main $@
