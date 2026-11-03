# F5 3Master 管理集群系统级集成测试执行说明

本文档说明如何执行 **f5_system_integration_test.go** 中的两个 F5 系统级集成测试，以及如何使用 **run_f5_sit_soak.sh** 进行多轮长稳测试。

## 1. 测试范围

测试套件名称：

  F5管理集群系统级集成测试

| 用例 | 验证目标 |
| --- | --- |
| F5管理集群_SIT_1 | 部署 F5/HAProxy 入口的3Master 管理集群，同时重启三台 Master，验证真实重启、审计规则持久化、关键路径、节点、Pod 和 BKECluster 恢复 |
| F5管理集群_SIT_2 | 复用 SIT_1 恢复后的集群，对三台 Master 执行 bke reset --all --mount，验证 containerd 清理，并使用相同配置重新部署 |

同时执行两个用例时，整体流程如下：

    配置 HAProxy
      -> 部署3Master 管理集群
      -> SIT_1 写入审计规则
      -> 三台 Master 同时重启
      -> 等待 boot ID 全部变化
      -> 验证 auditd、关键路径、节点、Pod、BC
      -> SIT_2 复用当前集群
      -> 三台 Master 执行 bke reset
      -> 删除引导侧 BC、BN 和 namespace
      -> 验证 /var/lib/containerd 已清空
      -> 使用原配置重新部署
      -> AfterAll 统一清理

## 2. 环境拓扑

| 角色 | 环境变量 | 要求 |
| --- | --- | --- |
| 引导节点 | GUIDE_NODE_* | 已执行 bke init，引导集群正常，具备 kubectl、bke 和集群创建能力 |
| Master 1 | TEST_NODE1_* | 麒麟 V10、root SSH、auditd 可用 |
| Master 2 | TEST_NODE2_* | 麒麟 V10、root SSH、auditd 可用 |
| Master 3 | TEST_NODE3_* | 麒麟 V10、root SSH、auditd 可用 |
| F5 模拟节点 | TEST_NODE4_* | 独立 HAProxy 节点，监听 36443 并代理3个 Master 的 6443 |

前三个测试节点自动设置为 master/node 和 etcd。第四个节点只作为 HAProxy/F5 模拟节点，不加入 Kubernetes 集群。

## 3. 前置检查

引导节点：

    kubectl get nodes
    kubectl get pods -A
    kubectl get bc -A
    which bke

每台 Master：

    systemctl is-active auditd
    auditctl -s
    cat /proc/cmdline
    which bke

要求 auditd 为 active、auditctl -s 显示 enabled 1、内核启动参数不包含 audit=0，且 root 用户的 PATH 中可以找到 bke。

HAProxy/F5 模拟节点：

    which haproxy
    systemctl status haproxy --no-pager
    ss -lntp | grep :36443 || true

测试会自动生成 HAProxy 配置。集群部署前，三个 Master 的 6443 尚未监听是正常现象，不能将后端端口暂时不可达作为 HAProxy 配置失败。

## 4. 执行方式

进入部署机项目的 e2e 目录：

    cd /root/zyh/e2e-auto-test/e2e
    chmod +x run_f5_sit_soak.sh

默认执行10轮：

    ./run_f5_sit_soak.sh

执行20轮或120轮：

    ./run_f5_sit_soak.sh --rounds 20
    ./run_f5_sit_soak.sh --rounds 120

自定义单轮超时：

    ./run_f5_sit_soak.sh --rounds 20 --timeout 150m

只执行 SIT_1：

    ./run_f5_sit_soak.sh \
      --rounds 20 \
      --focus "F5管理集群_SIT_1"

只执行 SIT_2：

    ./run_f5_sit_soak.sh \
      --rounds 20 \
      --focus "F5管理集群_SIT_2"

单独执行 SIT_2 时，测试仍会先部署一套新的3Master 管理集群。

失败后不保留现场：

    ./run_f5_sit_soak.sh --rounds 20 --no-keep-on-failure

不使用循环脚本，直接执行一次：

    go test ./installation -v -timeout 120m \
      --ginkgo.focus="F5 3Master 管理集群系统级集成测试"

## 5. 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| --rounds | 10 | 循环轮数，必须是正整数 |
| --timeout | 120m | 每轮 go test 总超时 |
| --focus | 完整 F5 SIT 套件名称 | Ginkgo 聚焦文本，可用于只运行一个用例 |
| --no-keep-on-failure | 未启用 | 真实失败后仍执行清理；默认行为是保留现场 |
| -h、--help | - | 显示脚本帮助 |

## 6. 长稳脚本完整内容

脚本可放置位置：e2e/run_f5_sit_soak.sh。

    #!/usr/bin/env bash
    set -uo pipefail
    rounds=10
    timeout="120m"
    focus="F5 3Master 管理集群系统级集成测试"
    keep_on_failure=1
    
    usage() {
      cat <<USAGE
    Usage: ./run_f5_sit_soak.sh [options]
    
    Options:
      --rounds N             Run N rounds. Default: 10
      --timeout DURATION     go test timeout. Default: 120m
      --focus TEXT           Ginkgo focus text. Default: F5 3Master 管理集群系统级集成测试
      --no-keep-on-failure   Allow test cleanup even when a round fails
      -h, --help             Show this help
    
    Environment switches passed to the test:
      F5_SIT_KEEP_ENV_ON_FAILURE=1
    USAGE
    }
    
    require_option_value() {
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "$1 requires a value" >&2
        exit 2
      fi
    }
    
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --rounds)
          require_option_value "$@"
          rounds="$2"
          shift 2
          ;;
        --timeout)
          require_option_value "$@"
          timeout="$2"
          shift 2
          ;;
        --focus)
          require_option_value "$@"
          focus="$2"
          shift 2
          ;;
        --no-keep-on-failure)
          keep_on_failure=0
          shift
          ;;
        -h|--help)
          usage
          exit 0
          ;;
        *)
          echo "Unknown option: $1" >&2
          usage >&2
          exit 2
          ;;
      esac
    done
    
    if ! [[ "$rounds" =~ ^[0-9]+$ ]] || [[ "$rounds" -lt 1 ]]; then
      echo "--rounds must be a positive integer" >&2
      exit 2
    fi
    
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir"
    
    if [[ -f .env ]]; then
      set -a
      # shellcheck disable=SC1091
      source ./.env
      set +a
    fi
    
    timestamp="$(date +%Y%m%d-%H%M%S)"
    artifact_root="${E2E_ARTIFACTS_DIR:-e2e-artifacts/f5-sit-soak/$timestamp}"
    mkdir -p "$artifact_root"
    export E2E_ARTIFACTS_DIR="$artifact_root"
    
    if [[ "$keep_on_failure" -eq 1 ]]; then
      export F5_SIT_KEEP_ENV_ON_FAILURE=1
    else
      unset F5_SIT_KEEP_ENV_ON_FAILURE
    fi
    
    summary_file="$artifact_root/summary.log"
    {
      echo "F5 SIT soak started at $(date '+%F %T')"
      echo "rounds=$rounds"
      echo "timeout=$timeout"
      echo "focus=$focus"
      echo "keep_on_failure=$keep_on_failure"
      echo "artifact_root=$artifact_root"
      echo
    } | tee "$summary_file"
    
    for round in $(seq 1 "$rounds"); do
      log_file="$artifact_root/round-${round}.log"
      echo "===== Round $round/$rounds started at $(date '+%F %T') =====" | tee -a "$summary_file"
    
      set +e
      go test ./installation -v -timeout "$timeout" --ginkgo.focus="$focus" 2>&1 | tee "$log_file"
      status=${PIPESTATUS[0]}
      set -e
    
      if [[ "$status" -ne 0 ]]; then
        echo "Round $round failed with exit code $status at $(date '+%F %T')" | tee -a "$summary_file"
        echo "Log: $log_file" | tee -a "$summary_file"
        if [[ "$keep_on_failure" -eq 1 ]]; then
          echo "F5_SIT_KEEP_ENV_ON_FAILURE=1 was set. The test should preserve cluster, HAProxy, configs, and audit rules on failure." | tee -a "$summary_file"
        fi
        exit "$status"
      fi
    
      echo "Round $round passed at $(date '+%F %T')" | tee -a "$summary_file"
      echo | tee -a "$summary_file"
    done
    
    echo "All $rounds rounds passed at $(date '+%F %T')" | tee -a "$summary_file"
## 7. 日志与产物

默认产物目录：

    e2e/e2e-artifacts/f5-sit-soak/<启动时间>/

| 文件 | 内容 |
| --- | --- |
| summary.log | 总轮数、参数、每轮开始/结束时间和最终结果 |
| round-N.log | 第 N 轮完整 go test 和 Ginkgo 输出 |
| f5_system_integration_test_* | 失败时收集的 controller-manager 等诊断日志 |

自定义产物目录：

    export E2E_ARTIFACTS_DIR=/root/f5-sit-artifacts/$(date +%Y%m%d-%H%M%S)
    ./run_f5_sit_soak.sh --rounds 20

