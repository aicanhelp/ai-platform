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
source ${WORK_DIR}/_volcano_common.sh
init_build_env "${WORK_DIR}"
init_volcano_env

function copy_outputs() {
    copy_to_dest ${VOLCANO_BASE_PATH}/output/vc-controller-manager
    echo "vc-controller-manager build complete. Outputs in ${DEST_DIR}"
}

function main() {
    compile_volcano "${CI_DIR}" "${TOP_DIR}"
    copy_outputs
}

main "$@"
