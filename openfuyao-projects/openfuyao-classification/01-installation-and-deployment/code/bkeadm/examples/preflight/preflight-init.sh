#!/usr/bin/env bash
# Copyright (c) 2026 Bocloud Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BKE_BIN="${BKE_BIN:-${ROOT}/bke}"
HOST_IP="${BKE_HOST_IP:?set BKE_HOST_IP to the host IP used by bke init}"
ONLINE_IMAGE="${BKE_ONLINE_IMAGE:?set BKE_ONLINE_IMAGE to the installation image}"
REPORT="${BKE_INIT_REPORT:-/tmp/bke-preflight-init.txt}"

if [[ "$(id -u)" -eq 0 ]]; then
  RUN=( )
else
  RUN=(sudo)
fi

"${RUN[@]}" "${BKE_BIN}" preflight init \
  --hostIP "${HOST_IP}" \
  --onlineImage "${ONLINE_IMAGE}" \
  --output text \
  --output-file "${REPORT}" \
  "$@"
