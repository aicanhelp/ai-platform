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

# ==================== Configuration ====================

ROOT_PATH="/root"
WORK_PATH="/opt/xpu"
WORK_LIB_PATH="${WORK_PATH}/lib"
WORK_BIN_PATH="${WORK_PATH}/bin"
FILE_MODE=555
DIR_MODE=555
HEAL_INTERVAL=5

# File registry: destination -> source
# To add a new file, add one line here
declare -A FILES=(
    ["${WORK_BIN_PATH}/enpu-monitor"]="${ROOT_PATH}/enpu-monitor"
    ["${WORK_BIN_PATH}/xpu-client-tool"]="${ROOT_PATH}/xpu-client-tool"
    ["${WORK_LIB_PATH}/libvruntime.so"]="${ROOT_PATH}/libvruntime.so"
    ["${WORK_LIB_PATH}/ld.so.preload"]="${ROOT_PATH}/ld.so.preload"
    ["${WORK_BIN_PATH}/systemd-detect-virt"]="${ROOT_PATH}/systemd-detect-virt"
)

# Symlink registry: link_path -> target_name
# To add a new symlink, add one line here
declare -A SYMLINKS=(
    ["${WORK_BIN_PATH}/xpu-monitor"]="enpu-monitor"
)

# ==================== Functions ====================

# Sync file from source to destination using MD5 verification
sync_file() {
    local src="$1" dest="$2"

    if [ ! -f "${src}" ]; then
        echo "${src} missing!"
        return 1
    fi

    if [ ! -f "${dest}" ]; then
        install -m ${FILE_MODE} "${src}" "${dest}"
        echo "$(basename "${dest}") is installed"
        return 0
    fi

    local src_md5 dest_md5
    src_md5=$(md5sum "${src}" | awk '{ print $1 }')
    dest_md5=$(md5sum "${dest}" | awk '{ print $1 }')
    if [ "${src_md5}" != "${dest_md5}" ]; then
        install -m ${FILE_MODE} "${src}" "${dest}"
        echo "$(basename "${dest}") is restored"
    fi
}

# Check and restore a symlink if broken
sync_symlink() {
    local link_path="$1" target="$2"

    if [ "$(readlink "${link_path}")" != "${target}" ]; then
        ln -fs "${target}" "${link_path}"
        echo "$(basename "${link_path}") is restored"
    fi
}

# ==================== Main ====================

# Step 1: Create directories
mkdir -m ${DIR_MODE} -p "${WORK_LIB_PATH}" "${WORK_BIN_PATH}"

# Step 2: Install all files
for dest in "${!FILES[@]}"; do
    sync_file "${FILES[${dest}]}" "${dest}"
done

# Step 3: Create all symlinks
for link in "${!SYMLINKS[@]}"; do
    ln -fs "${SYMLINKS[${link}]}" "${link}"
done

echo "client file initialization completed"

# Step 4: Self-healing daemon loop
while true; do
    for dest in "${!FILES[@]}"; do
        sync_file "${FILES[${dest}]}" "${dest}"
    done

    for link in "${!SYMLINKS[@]}"; do
        sync_symlink "${link}" "${SYMLINKS[${link}]}"
    done

    echo "files under watch"
    sleep "${HEAL_INTERVAL}"
done
