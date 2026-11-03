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
KUBECONFIG_PATH="${BKE_KUBECONFIG:?set BKE_KUBECONFIG to the management-cluster kubeconfig}"
CLUSTER_FILE="${BKE_CLUSTER_FILE:-${ROOT}/examples/preflight/cluster.yaml}"
NODES_FILE="${BKE_NODES_FILE:-${ROOT}/examples/preflight/nodes.yaml}"
REPORT="${BKE_CLUSTER_REPORT:-/tmp/bke-preflight-cluster-create.txt}"
MAX_CONCURRENCY="${BKE_MAX_CONCURRENCY:-16}"

if [[ "$(id -u)" -eq 0 ]]; then
  RUN=( )
else
  RUN=(sudo)
fi

"${RUN[@]}" "${BKE_BIN}" \
  --kubeconfig "${KUBECONFIG_PATH}" \
  preflight cluster-create \
  --file "${CLUSTER_FILE}" \
  --nodes "${NODES_FILE}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --output text \
  --output-file "${REPORT}" \
  "$@"
