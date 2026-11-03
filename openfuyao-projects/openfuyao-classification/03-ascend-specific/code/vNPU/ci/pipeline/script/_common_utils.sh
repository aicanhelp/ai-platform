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

function init_build_env() {
    WORK_DIR=$1
    CI_DIR=$(dirname "$(dirname "${WORK_DIR}")")
    TOP_DIR=$(dirname "${CI_DIR}")
    DEST_DIR=${CI_DIR}/output/
    mkdir -p "${DEST_DIR}"
}

function copy_to_dest() {
    cp -P --remove-destination -r "$@" "${DEST_DIR}"
}

function strip_gotest_codes() {
    if [ ! -d "$1" ]; then
        echo "Error: Directory '$1' does not exist."
        return
    fi
    cd "$1"
    find . -name *_test.go | xargs rm -rf
    sed -i '/gomonkey/d' go.mod
    go mod tidy
}

function prepare_cann_env() {
    local ci_dir=$1
    local top_dir=$2
    local work_dir=$3

    rm -rf ${ci_dir}/build && mkdir -p ${ci_dir}/build
    mkdir -p ${top_dir}/XPU_symbols/
}

function compile_client() {
    local ci_dir=$1
    local top_dir=$2
    local vcann_rt_path="${ci_dir}/build/ubs-virt/ubs-virt-enpu/vcann-rt"
    local ubs_virt_src="${top_dir}/third_party/ubs-virt"

    cd "${ci_dir}/build/"
    cp -r "${ubs_virt_src}" .

    cd "${vcann_rt_path}"
    chmod +x make_build.sh && ./make_build.sh
}

function compile_xpu_client_tool() {
    local top_dir=$1
    cd ${top_dir}/xpu-device-plugin/
    export CGO_LDFLAGS_ALLOW='-Wl,--unresolved-symbols=ignore-in-object-files'
    export CGO_CFLAGS='-D_FORTIFY_SOURCE=2 -O2'
    go build \
        -tags vnpu \
        -buildmode=pie \
        -ldflags="-s -w -linkmode 'external' -extldflags '-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack'" \
        -o xpu-client-tool \
        ./client
}

function strip_symbols() {
    local ci_dir=$1
    cd ${ci_dir}/build/ubs-virt/ubs-virt-enpu/vcann-rt/build
    objcopy --only-keep-debug libvruntime.so libvruntime.so.sym
    objcopy --strip-all libvruntime.so

    objcopy --only-keep-debug enpu-monitor enpu-monitor.sym
    objcopy --strip-all enpu-monitor
}

function compile_device_plugin() {
    local top_dir=$1
    strip_gotest_codes "${top_dir}/xpu-device-plugin/"
    cd ${top_dir}/xpu-device-plugin/

    export CGO_LDFLAGS_ALLOW='-Wl,--unresolved-symbols=ignore-in-object-files'
    export CGO_CFLAGS='-fcommon'
    export CGO_CPPFLAGS='-fcommon'
    go build \
        -tags vnpu \
        -buildmode=pie \
        -ldflags="-s -w -linkmode 'external' -extldflags '-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack'" \
        -o npu-device-plugin \
        ./cmd
}

function compile_xpu_exporter() {
    local top_dir=$1
    strip_gotest_codes "${top_dir}/xpu-exporter/"
    cd ${top_dir}/xpu-exporter/

    rm -f ./xpu-exporter

    export CGO_LDFLAGS_ALLOW='-Wl,--unresolved-symbols=ignore-in-object-files'
    go build \
        -buildmode=pie \
        -ldflags="-s -w -linkmode 'external' -extldflags '-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack' \
        -X huawei.com/npu-exporter/v6/versions.BuildName=xpu-exporter \
        -X huawei.com/xpu-exporter/versions.BuildVersion=v1.0.RC2" \
        -o xpu-exporter \
        ./cmd/xpu-exporter
}
