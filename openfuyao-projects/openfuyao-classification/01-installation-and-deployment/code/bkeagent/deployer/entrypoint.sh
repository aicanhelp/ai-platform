#!/bin/sh
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain n copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#######################################################################

set -eu

ARCH=${HOST_ARCH:-$(uname -m)}
case "$ARCH" in
  x86_64|amd64)   SRC=/artifacts/bkeagent_linux_amd64 ;;
  aarch64|arm64)  SRC=/artifacts/bkeagent_linux_arm64 ;;
  *) echo "[deployer] unsupported arch: $ARCH"; exit 1 ;;
esac

HOST_BIN_DIR=${HOST_BIN_DIR:-/etc/bkeagent/bin}
DEST="${HOST_BIN_DIR}/bkeagent"

mkdir -p "${HOST_BIN_DIR}"

if [ ! -f "${DEST}" ] || ! cmp -s "${SRC}" "${DEST}"; then
  cp -f "${SRC}" "${DEST}"
  chmod +x "${DEST}"
  echo "[deployer] installed ${SRC} -> ${DEST}"
else
  echo "[deployer] host already has same binary, skip"
fi

"${DEST}" -v || true
tail -f /dev/null
