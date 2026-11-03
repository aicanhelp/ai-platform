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

set -e

WORK_DIR=$(cd $(dirname $0); pwd)
source ${WORK_DIR}/_common_utils.sh
init_build_env "${WORK_DIR}"

function copy_outputs() {
    copy_to_dest ${TOP_DIR}/xpu-device-plugin/npu-device-plugin
    echo "npu-device-plugin build complete. Outputs in ${DEST_DIR}"
}

function main() {
    compile_device_plugin "${TOP_DIR}"
    copy_outputs
}

main "$@"
