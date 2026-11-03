#!/bin/bash

#
# /*
#  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  * openFuyao is licensed under Mulan PSL v2.
#  * You can use this software according to the terms and conditions of the Mulan PSL v2.
#  * You may obtain a copy of Mulan PSL v2 at:
#  *          http://license.coscl.org.cn/MulanPSL2
#  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
#  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
#  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#  * See the Mulan PSL v2 for more details.
#  */
#

set -ex

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
source ${SCRIPT_DIR}/_common_utils.sh

VOLCANO_BASE_VERSION='1.15.0'
VOLCANO_PLUGIN_NAME=volcano-vxpu
GCC_PATH=/usr/bin/gcc
DATE=$(date "+%Y-%m-%d %H:%M:%S")

function init_volcano_env() {
    VOLCANO_TOP_DIR=${CI_DIR}/volcano_build/volcano
    VOLCANO_BASE_PATH=${VOLCANO_TOP_DIR}/pkg/scheduler/plugins/volcano-xpu-plugin/
}

function prepare_volcano() {
    local ci_dir=$1
    local top_dir=$2
    local volcano_top_dir=${ci_dir}/volcano_build/volcano

    rm -rf ${ci_dir}/volcano_build && mkdir -p ${ci_dir}/volcano_build
    rm -rf ${top_dir}/third_party/volcano_tmp
    
    cp -P --remove-destination -rf ${top_dir}/third_party/volcano ${top_dir}/third_party/volcano_tmp
    cd ${top_dir}/third_party/volcano_tmp
    patch -p1 < ${ci_dir}/patches/*.patch

    cp -P --remove-destination -rf ${top_dir}/third_party/volcano_tmp ${volcano_top_dir}
    cd ${volcano_top_dir}

    cp -P --remove-destination -rf ${top_dir}/volcano-xpu-plugin ${volcano_top_dir}/pkg/scheduler/plugins

    rm -rf ${top_dir}/third_party/volcano_tmp
}

function build_volcano() {
    local volcano_top_dir=$1
    local volcano_base_path=${volcano_top_dir}/pkg/scheduler/plugins/volcano-xpu-plugin/
    local volcano_pkg_path=${volcano_top_dir}/pkg

    export GO111MODULE=on
    export GOTOOLCHAIN=auto

    cd "${volcano_top_dir}"
    go mod tidy

    mkdir -p "${volcano_base_path}"/output
    cd "${volcano_base_path}"/output

    for name in controller-manager scheduler webhook-manager; do \
      CGO_CFLAGS="-fstack-protector-strong -D_FORTIFY_SOURCE=2 -O2 -fPIC -ftrapv" \
      CGO_CPPFLAGS="-fstack-protector-strong -D_FORTIFY_SOURCE=2 -O2 -fPIC -ftrapv" \
      CC=${GCC_PATH} CGO_ENABLED=1 \
      go build -mod=mod -buildmode=pie -ldflags "-s -linkmode=external -extldflags=-Wl,-z,now
      -X '${volcano_pkg_path}/version.Built=${DATE}' -X '${volcano_pkg_path}/version.Version=${VOLCANO_BASE_VERSION}'" \
      -o vc-$name "${volcano_top_dir}"/cmd/$name
    done

    CGO_CFLAGS="-fstack-protector-strong -D_FORTIFY_SOURCE=2 -O2 -fPIC -ftrapv" \
    CGO_CPPFLAGS="-fstack-protector-strong -D_FORTIFY_SOURCE=2 -O2 -fPIC -ftrapv" \
    CC=${GCC_PATH} CGO_ENABLED=1 \
    go build -mod=mod -buildmode=plugin -ldflags "-s -linkmode=external -extldflags=-Wl,-z,now" \
    -o "${VOLCANO_PLUGIN_NAME}".so "${volcano_base_path}"
}

function compile_volcano() {
    local ci_dir=$1
    local top_dir=$2
    local volcano_top_dir=${ci_dir}/volcano_build/volcano

    echo "compile volcano start"
    prepare_volcano "${ci_dir}" "${top_dir}"
    build_volcano "${volcano_top_dir}"
    echo "compile volcano end"
}
