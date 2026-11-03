#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#######################################################################
set -euo pipefail

LOG_FILE="/var/log/openFuyao/bkeagent-update.log"
SERVICE_NAME="bkeagent"
CURRENT_BIN="/usr/local/bin/bkeagent"
BACKUP_COUNT=3               # 保留最近 N 个备份
HEALTH_CHECK_URL=""          # 可选：如 "http://localhost:8080/health"
HEALTH_CHECK_TIMEOUT=5       # 健康检查超时（秒）
MAX_START_WAIT=15            # 最大等待启动时间（秒）

log() {
  local level=${2:-INFO}
  echo "$(date +'%Y-%m-%d %H:%M:%S') [$level] update.sh: $1" | tee -a "$LOG_FILE"
}

error_exit() {
  log "$1" "ERROR"
  exit 1
}

# 健康检查：支持 HTTP 或仅进程检查
health_check() {
  # 检查 systemd 状态
  if ! systemctl is-active --quiet "$SERVICE_NAME"; then
    return 1
  fi

  # 检查进程是否存在
  if ! pgrep -f "$SERVICE_NAME" > /dev/null 2>&1; then
    return 1
  fi

  # 可选：HTTP 健康探测
  if [[ -n "${HEALTH_CHECK_URL}" ]]; then
    if ! timeout "$HEALTH_CHECK_TIMEOUT" curl -sf --max-time "$HEALTH_CHECK_TIMEOUT" "$HEALTH_CHECK_URL" > /dev/null 2>&1; then
      return 1
    fi
  fi

  return 0
}

# 回滚函数
rollback() {
  local backup_bin="$1"
  log "Initiating rollback to $backup_bin..." "WARN"

  # 停止当前（可能异常的）服务
  systemctl stop "$SERVICE_NAME" || true

  # 恢复备份
  if [[ -f "$backup_bin" ]]; then
    cp -f "$backup_bin" "$CURRENT_BIN"
    chmod +x "$CURRENT_BIN"
    log "Restored binary from $backup_bin"
  else
    log "Backup file not found, cannot rollback!" "ERROR"
    return 1
  fi

  # 重启服务
  if systemctl start "$SERVICE_NAME"; then
    # 等待回滚后服务恢复
    for i in $(seq 1 10); do
      if health_check; then
        log "Rollback succeeded. Service is healthy."
        return 0
      fi
      sleep 1
    done
    log "Rollback started but service not healthy after 10s." "WARN"
  else
    log "Failed to start service after rollback!" "ERROR"
  fi
}

# 清理旧备份（保留最近 BACKUP_COUNT 个）
cleanup_old_backups() {
  local backups=()
  while IFS= read -r -d '' file; do
    backups+=("$file")
  done < <(find "$(dirname "$CURRENT_BIN")" -maxdepth 1 -name "$(basename "$CURRENT_BIN").bak.*" -print0 2>/dev/null | sort -rzV)

  # 删除超出保留数量的旧备份
  if [[ ${#backups[@]} -gt $BACKUP_COUNT ]]; then
    for ((i=BACKUP_COUNT; i<${#backups[@]}; i++)); do
      rm -f "${backups[i]}"
      log "Removed old backup: ${backups[i]}"
    done
  fi
}

# 初始化日志
touch "$LOG_FILE"
log "=== Starting bkeagent update ==="

# 参数校验
NEW_BIN_PATH="$1"
if [[ -z "$NEW_BIN_PATH" ]] || [[ ! -f "$NEW_BIN_PATH" ]]; then
  error_exit "Usage: $0 <new_binary_path>. Provided path is empty or not a file."
fi

# 验证新二进制可执行
if ! "$NEW_BIN_PATH" -v >/dev/null 2>&1; then
  error_exit "New binary is not valid or cannot run '-v'. Aborting."
fi

log "New binary validated: $NEW_BIN_PATH"

# 创建备份
BACKUP_BIN="${CURRENT_BIN}.bak.$(date +%s)"
if [[ -f "$CURRENT_BIN" ]]; then
  cp -f "$CURRENT_BIN" "$BACKUP_BIN"
  chmod +x "$BACKUP_BIN"
  log "Backed up current binary to $BACKUP_BIN"
else
  log "No existing binary found at $CURRENT_BIN, skipping backup." "WARN"
  BACKUP_BIN=""
fi

# 停止服务
log "Stopping $SERVICE_NAME..."
systemctl stop "$SERVICE_NAME" || true

# 替换二进制
log "Replacing binary..."
cp -f "$NEW_BIN_PATH" "$CURRENT_BIN"
chmod +x "$CURRENT_BIN"

# 启动服务
log "Starting $SERVICE_NAME with new binary..."
if ! systemctl start "$SERVICE_NAME"; then
  log "Failed to start service. Triggering rollback..." "ERROR"
  [[ -n "$BACKUP_BIN" ]] && rollback "$BACKUP_BIN"
  error_exit "Update failed and rollback completed (or attempted)."
fi

# 等待并验证健康状态
log "Waiting for service to become healthy (max ${MAX_START_WAIT}s)..."
healthy=false
for i in $(seq 1 "$MAX_START_WAIT"); do
  if health_check; then
    healthy=true
    break
  fi
  sleep 1
done

if [[ "$healthy" == true ]]; then
  log "Update successful! Service is healthy."
  # 清理临时文件
  rm -f "$NEW_BIN_PATH"
  log "Cleaned up temporary binary: $NEW_BIN_PATH"
else
  log "Service not healthy after ${MAX_START_WAIT}s. Triggering rollback..." "ERROR"
  if [[ -n "$BACKUP_BIN" ]]; then
    rollback "$BACKUP_BIN"
  else
    log "No backup available. Service may be broken!" "CRITICAL"
  fi
  error_exit "Update failed due to health check timeout."
fi

# 清理旧备份
cleanup_old_backups

log "=== Update process completed successfully ==="