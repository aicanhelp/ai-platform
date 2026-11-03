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

source ../../utils/log.sh "cni-plugins"
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

# 执行下载
function exec_download_bin() {
    local arch=$1
    local cni_url="${FUYAO_OBS}/containernetworking/plugins/releases/download/v${VERSION}/cni-plugins-linux-${arch}-v${VERSION}.tgz"

    local attempts=0
    while [ $attempts -lt 3 ]; do
        sudo -E curl -sOL ${cni_url}
        if [ $? -eq 0 ]; then
            info_log "The ${arch} architecture binary files have been downloaded"
            break
        else
            ((attempts++))
        fi
    done

    if [ $attempts -eq 3 ]; then
        fatal_log "Failed to download the file after 3 attempts."
    fi

    # 检查本地是否有对应文件
    if [ -f "cni-plugins-linux-${arch}-v${VERSION}.tgz" ]; then
        info_log "File cni-plugins-linux-${arch}-v${VERSION}.tgz downloaded successfully."
    else
        fatal_log "Failed to download file cni-plugins-linux-${arch}-v${VERSION}.tgz."
    fi
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

    sudo mkdir -p ${base_path}
    cd ${base_path}
    if [ -n "${download_binary}" ];then
        download_bin "arm64"
        download_bin "amd64"
    fi

    exit 0
}

DOWNLOAD_SHELL=$(getopt -n download.sh -o p:i:c:y:b:h --long path:,image:,chart:,yaml:,binary:,help -- "$@")
[ $? -ne 0 ] && exit 1
eval set -- "$DOWNLOAD_SHELL"
main $@
