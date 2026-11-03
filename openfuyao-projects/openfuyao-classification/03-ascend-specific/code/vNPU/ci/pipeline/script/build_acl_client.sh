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

WORK_DIR=$(cd $(dirname $0); pwd)
source ${WORK_DIR}/_common_utils.sh
init_build_env "${WORK_DIR}"

function prepare() {
    prepare_cann_env "${CI_DIR}" "${TOP_DIR}" "${WORK_DIR}"
}

function copy_outputs() {
    local build_dir="${CI_DIR}/build/ubs-virt/ubs-virt-enpu/vcann-rt/build"

    copy_to_dest \
        ${build_dir}/libvruntime.so \
        ${TOP_DIR}/client_update/ld.so.preload \
        ${TOP_DIR}/client_update/systemd-detect-virt \
        ${build_dir}/enpu-monitor \
        ${TOP_DIR}/client_update/acl-client-update.sh \
        ${TOP_DIR}/xpu-device-plugin/xpu-client-tool

    cp -P --remove-destination -r ${build_dir}/*.sym ${TOP_DIR}/XPU_symbols/
    cd ${TOP_DIR} && tar -czvf XPU_symbols.tar.gz XPU_symbols
    copy_to_dest ${TOP_DIR}/XPU_symbols.tar.gz

    echo "acl_client build complete. Outputs in ${DEST_DIR}"
}

function main() {
    prepare
    compile_client "${CI_DIR}" "${TOP_DIR}"
    compile_xpu_client_tool "${TOP_DIR}"
    strip_symbols "${CI_DIR}"
    copy_outputs
}

main "$@"
