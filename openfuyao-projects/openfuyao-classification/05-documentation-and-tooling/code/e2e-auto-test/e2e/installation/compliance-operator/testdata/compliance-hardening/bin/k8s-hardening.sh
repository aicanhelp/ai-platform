#!/usr/bin/env bash
# k8s-hardening.sh — K8s CIS/STIG 安全加固与回退工具主入口
# 对外命令：cluster-harden / cluster-rollback
# 内部命令（cluster 模式 SSH 到各节点调用）：harden / rollback（不对外暴露，不生成报告/不落盘日志）
# 依赖：bash 4+, yq, jq, sshpass（entry_node）; 各节点需 yq/jq

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB_DIR="$PROJECT_DIR/lib"
CONFIG_DIR="$PROJECT_DIR/config"

# shellcheck source=../lib/common.sh
source "$LIB_DIR/common.sh"

_load_all() {
  local lib
  for lib in runtime_detect backup manifest_ops kubelet_config_ops file_perms rules health_check cluster; do
    # fail-fast：缺失库文件立即报错，而不是等到函数调用时才报 command not found
    [[ -f "$LIB_DIR/$lib.sh" ]] || die "缺少库文件：$LIB_DIR/$lib.sh"
    # shellcheck disable=SC1090
    source "$LIB_DIR/$lib.sh"
  done
}

VERSION="0.1.0"

usage() {
  cat <<EOF
K8s CIS/STIG 安全加固与回退工具 v$VERSION

用法: k8s-hardening.sh <command> config/cluster.yaml [options]

命令:
  cluster-harden    一键全集群加固（--dry-run 仅预览，不加则执行；master 串行 + worker 并行）
  cluster-rollback  一键全集群回退（可选 --backup-subdir 指定统一备份）；
                    差量恢复：仅恢复与备份内容有差异的文件，未改配置时不重启不等待

选项:
  --dry-run                仅预览变更不执行（cluster-harden 可用）
  -y, --yes                跳过交互确认
  --backup-subdir <时间戳>  cluster-rollback 专用：指定回退到哪次统一备份。
                           cluster-harden 时 entry_node 会统一下发时间戳，所有节点
                           备份到同一子目录；回退指定它即可让全部节点回退到同一状态。
                           不指定则各节点回退到各自最新备份。
  -v, --verbose            详细日志
  -h, --help               帮助
  --version                版本

示例:
  # 预览全集群变更（不执行）
  k8s-hardening.sh cluster-harden config/cluster.yaml --dry-run

  # 一键加固
  k8s-hardening.sh cluster-harden config/cluster.yaml -y

  # 一键回退（各节点回退到各自最新备份）
  k8s-hardening.sh cluster-rollback config/cluster.yaml -y

  # 一键回退到指定统一备份（加固日志会打印"本次集群加固备份时间戳：xxx"）
  k8s-hardening.sh cluster-rollback config/cluster.yaml -y --backup-subdir 20260823-165152

报告: 加固完成后在 entry_node 生成集群统一报告
  /var/lib/compliance-hardening/reports/cluster-harden-report-<时间戳>.md
  /var/lib/compliance-hardening/reports/cluster-harden-report-<时间戳>.json
  报告含各节点状态（已加固/无需加固（已合规）/失败）与逐规则明细。
EOF
}

# _print_result_rows — 输出 RESULT_ROWS 机器协议块（entry_node 解析生成集群报告）
# 行本身不带日志前缀；协议格式唯一出处，勿在调用点重复展开
_print_result_rows() {
  log_info "输出结果数据（${#RESULT_ROWS[@]} 行，供集群报告收集）："
  printf 'RESULT_ROWS_START\n'
  local row
  for row in "${RESULT_ROWS[@]:-}"; do
    [[ -z "$row" ]] && continue
    printf '%s\n' "$row"
  done
  printf 'RESULT_ROWS_END\n'
}

# ---------- 内部命令（cluster 模式 SSH 到各节点调用，不对外暴露）----------
# harden --internal：不生成报告、不落盘日志，结果通过 RESULT_ROWS 全局变量+stdout 传回
cmd_harden_internal() {
  local groups=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run) DRY_RUN=1; shift ;;
      --group)
        [[ -n "${2:-}" && "$2" != -* ]] || die "--group 需要组 ID 参数（如 g1 或 g1,g3）"
        groups="$2"; shift 2 ;;
      --internal) shift ;;  # 内部标记，无实际作用
      --backup-dir) BACKUP_DIR="$2"; export BACKUP_DIR; shift 2 ;;
      --backup-subdir) export CURRENT_BACKUP_SUBDIR="${2:-}"; shift 2 ;;
      --skip-preview) SKIP_PREVIEW=1; export SKIP_PREVIEW; shift ;;
      -y|--yes) YES=1; export YES; shift ;;
      *) shift ;;
    esac
  done

  # 依赖检查前置：缺 yq/jq 时立即报错，不必先加载全部库
  require_cmd yq jq
  _load_all
  require_root
  _init_rules

  log_info "=== 开始安全加固 ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log_warn "预览模式：仅显示变更，不落盘、不备份、不执行"
  fi
  detect_all

  if [[ "${SKIP_PREVIEW:-0}" -eq 1 ]]; then
    # 集群模式：用户已在 entry_node 确认过全集群预览，节点内不再重复打印详细清单，
    # 仅静默统计待整改数（维持"无待整改项则跳过"门控）
    log_info "=== 变更预览（基于 g1-g6 规则清单）==="
    harden_preview "$groups" 1
    log_info "待整改 ${PREVIEW_PENDING_COUNT:-0} 项（详细预览已在确认阶段展示，此处不再重复）"
  else
    log_info "=== 变更预览（基于 g1-g6 规则清单）==="
    harden_preview "$groups"
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log_info "=== 预览完成（未执行任何变更）==="
    return 0
  fi

  if [[ "${PREVIEW_PENDING_COUNT:-0}" -eq 0 ]]; then
    log_info "无待整改项（全部已合规或高危跳过），跳过加固执行。"
    # 仍确保统一时间戳备份存在（内容=当前合规状态，changes.json 为空）：
    # 保证 cluster-harden 的统一备份在所有节点都存在，后续 --backup-subdir
    # 回退时该节点差量恢复自然无差异、无需重启，而不是"备份目录不存在"失败
    # （集群模式阶段 0 已预建则跳过；standalone 或预备份缺失时此处补建）
    backup_all_if_needed
    # 输出全部规则的 RESULT_ROWS（skipped/manual/error 状态）供报告收集——
    # 否则集群报告会显示"无结果数据，可能节点加固失败"，无法体现节点已合规
    rules_record_status "$groups"
    _print_result_rows
    # 专用退出码 3：本节点无待整改项——entry_node 据此跳过组后等待（Pod 重建/健康检查）
    # （cluster_harden_node 只区分 rc=0/非0，3 与 0 同样视为"节点完成"）
    return 3
  fi

  backup_all_if_needed
  harden_run "$groups"
  _print_result_rows
  log_info "=== 加固完成 ==="
}

cmd_rollback_internal() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --backup-subdir) export ROLLBACK_SUBDIR="${2:-}"; shift 2 ;;
      --internal) shift ;;
      --backup-dir) BACKUP_DIR="$2"; export BACKUP_DIR; shift 2 ;;
      -y|--yes) YES=1; export YES; shift ;;
      *) shift ;;
    esac
  done

  require_cmd yq jq
  _load_all
  require_root
  log_info "=== 开始回退 ==="
  rollback_run
  log_info "=== 回退完成 ==="
}

# prebackup - 集群阶段 0 预备份：仅创建统一时间戳备份（含加固起点标记），不做任何加固。
# 供 entry_node 在加固任何节点之前对全部节点预建备份，使中断轮/失败轮也留下
# 全集群可用的统一回退目标（--backup-subdir）。
cmd_prebackup_internal() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --backup-subdir) export CURRENT_BACKUP_SUBDIR="${2:-}"; shift 2 ;;
      --internal) shift ;;
      --backup-dir) BACKUP_DIR="$2"; export BACKUP_DIR; shift 2 ;;
      *) shift ;;
    esac
  done

  require_cmd yq jq
  _load_all
  require_root
  if [[ -z "${CURRENT_BACKUP_SUBDIR:-}" ]]; then
    die "prebackup 需要统一时间戳：--backup-subdir <ts>"
  fi
  log_info "=== 预备份（不做任何加固）==="
  if [[ -f "$BACKUP_DIR/$CURRENT_BACKUP_SUBDIR/changes.json" ]]; then
    log_info "统一备份 $CURRENT_BACKUP_SUBDIR 已存在，跳过"
    return 0
  fi
  backup_all
  # 机器协议行（entry_node 解析；与节点侧 RESULT_ROWS 同类的简洁输出）
  printf 'PREBACKUP_DONE ts=%s\n' "$CURRENT_BACKUP_SUBDIR"
  log_info "=== 预备份完成 ==="
}

# ---------- 全集群模式子命令 ----------
cmd_cluster_harden() {
  local cluster_cfg=""
  local dryrun_only=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run) dryrun_only=1; shift ;;
      -y|--yes) YES=1; export YES; shift ;;
      -v|--verbose) VERBOSE=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) cluster_cfg="$1"; shift ;;
    esac
  done
  [[ -z "$cluster_cfg" ]] && die "用法: cluster-harden <cluster.yaml> [--dry-run] [-y]"
  require_cmd yq jq sshpass
  _load_all
  # 先 load_config 获取 log_dir，再设 LOG_FILE 落盘日志
  cluster_load_config "$cluster_cfg"
  local ts; ts="$(now_ts)"
  LOG_FILE="$CLUSTER_LOG_DIR/cluster-harden-${ts}.log"
  log_init
  cluster_harden_all "$dryrun_only"
}

cmd_cluster_rollback() {
  local cluster_cfg=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -y|--yes) YES=1; export YES; shift ;;
      -v|--verbose) VERBOSE=1; shift ;;
      --backup-subdir) export CLUSTER_ROLLBACK_SUBDIR="${2:-}"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) cluster_cfg="$1"; shift ;;
    esac
  done
  [[ -z "$cluster_cfg" ]] && die "用法: cluster-rollback <cluster.yaml> [-y] [--backup-subdir <时间戳>]
  --backup-subdir 指定回退到哪个统一备份子目录（cluster-harden 时 entry_node 统一下发的时间戳），
                 让所有节点回退到同一次备份。未指定则各节点回退到各自最新备份。"
  require_cmd yq jq sshpass
  _load_all
  cluster_load_config "$cluster_cfg"
  local ts; ts="$(now_ts)"
  LOG_FILE="$CLUSTER_LOG_DIR/cluster-rollback-${ts}.log"
  log_init
  cluster_rollback_all
}

_init_rules() {
  RULES_FILE="${RULES_FILE:-$CONFIG_DIR/hardening-rules.yaml}"
  [[ -f "$RULES_FILE" ]] || die "规则清单不存在：$RULES_FILE"
  yq eval '.' "$RULES_FILE" >/dev/null 2>&1 || die "规则清单 YAML 解析失败：$RULES_FILE"
  log_debug "规则清单：$RULES_FILE"
}

# ---------- main ----------
main() {
  [[ $# -eq 0 ]] && { usage; exit 1; }
  local cmd="$1"; shift
  case "$cmd" in
    # 对外命令
    cluster-harden)   cmd_cluster_harden "$@" ;;
    cluster-rollback) cmd_cluster_rollback "$@" ;;
    # 内部命令（cluster 模式 SSH 到各节点调用，不对外暴露）
    harden)   cmd_harden_internal "$@" ;;
    rollback) cmd_rollback_internal "$@" ;;
    prebackup) cmd_prebackup_internal "$@" ;;
    -h|--help|help) usage; exit 0 ;;
    --version) echo "k8s-hardening v$VERSION"; exit 0 ;;
    *) echo "未知命令：$cmd（仅支持 cluster-harden / cluster-rollback）" >&2; usage; exit 1 ;;
  esac
}

main "$@"
