# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_PATH="${SCRIPT_DIR}/auto_aiperf.conf"
CONFIG_PATH="${1:-${AUTO_AIPERF_CONFIG:-${DEFAULT_CONFIG_PATH}}}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "配置文件不存在: ${CONFIG_PATH}" >&2
  echo "用法: $0 [config_file_path]" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "配置项未设置: ${name}" >&2
    exit 1
  fi
}

require_var "NAMESPACE"
require_var "RELEASE_NAME"
require_var "CHART_DIR"
require_var "PROJECT_DIR"
require_var "ARTIFACTS_DIR"
require_var "BASE_RESULT_DIR_NAME"
require_var "AIPERF_PROCESS_PATTERN"
require_var "POLL_INTERVAL_SECONDS"
require_var "POST_UNINSTALL_WAIT_SECONDS"
require_var "PODS_READY_TIMEOUT_SECONDS"
require_var "LOOP_INTERVAL_SECONDS"
require_var "GATEWAY_SERVICE_NAME"
require_var "GATEWAY_SERVICE_PORT"
require_var "PYTHON_BIN"
require_var "DECODE_COLLECTOR_SCRIPT"
require_var "DECODE_COST_IDLE_TIMEOUT"

if ! declare -p AIPERF_CMD >/dev/null 2>&1; then
  echo "配置项 AIPERF_CMD 未定义（需在配置文件中定义为 bash 数组）" >&2
  exit 1
fi

if [[ "$(declare -p AIPERF_CMD 2>/dev/null)" != declare\ -a* ]]; then
  echo "配置项 AIPERF_CMD 类型错误（必须是 bash 数组）" >&2
  exit 1
fi

if (( ${#AIPERF_CMD[@]} == 0 )); then
  echo "配置项 AIPERF_CMD 不能为空（需为 bash 数组）" >&2
  exit 1
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

wait_for_no_running_aiperf() {
  log "检查是否有正在运行的 aiperf 进程..."
  while pgrep -fa "${AIPERF_PROCESS_PATTERN}" >/dev/null 2>&1; do
    log "检测到 aiperf 仍在运行，${POLL_INTERVAL_SECONDS}s 后重试。"
    sleep "${POLL_INTERVAL_SECONDS}"
  done
  log "当前没有运行中的 aiperf profile 进程。"
}

reinstall_infernex() {
  log "开始卸载 Helm release: ${RELEASE_NAME} (namespace: ${NAMESPACE})"
  local did_uninstall=false
  if helm -n "${NAMESPACE}" status "${RELEASE_NAME}" >/dev/null 2>&1; then
    helm uninstall "${RELEASE_NAME}" -n "${NAMESPACE}"
    did_uninstall=true
  else
    log "未发现 release ${RELEASE_NAME}，跳过卸载。"
  fi

  if [[ "${did_uninstall}" == "true" ]]; then
    log "卸载完成，等待 ${POST_UNINSTALL_WAIT_SECONDS}s 让资源释放..."
    sleep "${POST_UNINSTALL_WAIT_SECONDS}"
  fi

  log "安装 Helm release: ${RELEASE_NAME}"
  helm install "${RELEASE_NAME}" "${CHART_DIR}" -n "${NAMESPACE}"
}

wait_for_pods_running() {
  log "等待 ${NAMESPACE} 命名空间 Pod 全部满足 Running 且 READY 完全就绪..."
  local timeout_seconds="${PODS_READY_TIMEOUT_SECONDS}"
  local elapsed=0

  while true; do
    mapfile -t pod_lines < <(kubectl -n "${NAMESPACE}" get pods --no-headers 2>/dev/null || true)

    if [[ ${#pod_lines[@]} -eq 0 ]]; then
      log "命名空间下暂未发现 Pod，继续等待。"
    else
      local all_ready_and_running=true
      local not_ready_details=()

      local line
      for line in "${pod_lines[@]}"; do
        local pod_name ready status
        pod_name="$(awk '{print $1}' <<<"${line}")"
        ready="$(awk '{print $2}' <<<"${line}")"
        status="$(awk '{print $3}' <<<"${line}")"

        local ready_ok=false
        if [[ "${ready}" =~ ^([0-9]+)/([0-9]+)$ ]]; then
          local ready_num total_num
          ready_num="${BASH_REMATCH[1]}"
          total_num="${BASH_REMATCH[2]}"
          if [[ "${ready_num}" == "${total_num}" ]]; then
            ready_ok=true
          fi
        fi

        if [[ "${status}" != "Running" || "${ready_ok}" != "true" ]]; then
          all_ready_and_running=false
          not_ready_details+=("${pod_name}:ready=${ready},status=${status}")
        fi
      done

      if [[ "${all_ready_and_running}" == "true" ]]; then
        log "所有 Pod 已满足 Running 且 READY 完全就绪。"
        return 0
      fi

      log "仍有 Pod 未满足条件: ${not_ready_details[*]}"
    fi

    sleep "${POLL_INTERVAL_SECONDS}"
    elapsed=$((elapsed + POLL_INTERVAL_SECONDS))
    if (( elapsed >= timeout_seconds )); then
      log "等待 Pod Running 超时（${timeout_seconds}s）。"
      kubectl -n "${NAMESPACE}" get pods
      return 1
    fi
  done
}

get_gateway_cluster_ip() {
  local ip
  ip="$(kubectl -n "${NAMESPACE}" get svc "${GATEWAY_SERVICE_NAME}" -o jsonpath='{.spec.clusterIP}')"
  if [[ -z "${ip}" ]]; then
    log "未获取到 ${GATEWAY_SERVICE_NAME} 的 CLUSTER-IP。"
    return 1
  fi
  printf '%s' "${ip}"
}

archive_result_dir() {
  mkdir -p "${ARTIFACTS_DIR}"
  local src_dir="${ARTIFACTS_DIR}/${BASE_RESULT_DIR_NAME}"

  if [[ ! -d "${src_dir}" ]]; then
    log "未找到结果目录 ${src_dir}，跳过归档。"
    return 0
  fi

  local ts target_dir
  ts="$(date '+%Y%m%d_%H%M%S')"
  target_dir="${ARTIFACTS_DIR}/${BASE_RESULT_DIR_NAME}_${ts}"

  mv "${src_dir}" "${target_dir}"
  log "结果已归档: ${target_dir}"
}

run_aiperf_once() {
  cd "${PROJECT_DIR}"
  local gateway_ip
  gateway_ip="$(get_gateway_cluster_ip)"
  log "获取到 ${GATEWAY_SERVICE_NAME} CLUSTER-IP: ${gateway_ip}"
  log "开始执行 aiperf profile..."

  "${AIPERF_CMD[@]}" --url "${gateway_ip}:${GATEWAY_SERVICE_PORT}"
}

# Pod 全部就绪后启动 decode cost 采集（后台），再跑 aiperf；aiperf 结束后 wait 采集进程（靠 -t 空闲退出）
run_aiperf_with_collector() {
  cd "${PROJECT_DIR}"
  mkdir -p "${ARTIFACTS_DIR}"

  if [[ ! -f "${DECODE_COLLECTOR_SCRIPT}" ]]; then
    log "未找到 ${DECODE_COLLECTOR_SCRIPT}，跳过 decode 采集，仅执行 aiperf。"
    run_aiperf_once || return $?
    return 0
  fi

  local round_ts decode_log decode_crash_dir decode_pid decode_rc

  round_ts="$(date '+%Y%m%d_%H%M%S')"
  decode_log="${ARTIFACTS_DIR}/decode_cost_${round_ts}.log"
  decode_crash_dir="${ARTIFACTS_DIR}/decode_crash_logs_${round_ts}"

  log "启动 decode cost 采集: ${PYTHON_BIN} ${DECODE_COLLECTOR_SCRIPT} -n ${NAMESPACE} -t ${DECODE_COST_IDLE_TIMEOUT} -o ${decode_crash_dir}"
  log "采集脚本 stdout/stderr -> ${decode_log}"

  "${PYTHON_BIN}" "${DECODE_COLLECTOR_SCRIPT}" \
    -n "${NAMESPACE}" \
    -t "${DECODE_COST_IDLE_TIMEOUT}" \
    -o "${decode_crash_dir}" >>"${decode_log}" 2>&1 &
  decode_pid=$!

  local aiperf_rc=0
  run_aiperf_once || aiperf_rc=$?

  log "等待 decode cost 采集进程结束 (pid=${decode_pid})..."
  set +e
  wait "${decode_pid}"
  decode_rc=$?
  set -e
  log "$(basename "${DECODE_COLLECTOR_SCRIPT}") 退出码: ${decode_rc}"

  return "${aiperf_rc}"
}

main_loop() {
  cd "${PROJECT_DIR}"
  wait_for_no_running_aiperf

  while true; do
    reinstall_infernex
    wait_for_pods_running

    if run_aiperf_with_collector; then
      log "aiperf 执行完成。"
    else
      log "aiperf 执行失败（退出码: $?）。"
    fi

    archive_result_dir
    log "本轮流程结束，${LOOP_INTERVAL_SECONDS}s 后开始下一轮。按 Ctrl+C 停止。"
    sleep "${LOOP_INTERVAL_SECONDS}"
  done
}

main_loop
