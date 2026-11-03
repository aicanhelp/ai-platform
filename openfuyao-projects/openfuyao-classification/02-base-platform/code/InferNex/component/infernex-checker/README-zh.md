
<div align="center">
  <h1 align="center">
      InferNex-Checker
  </h1>

  <p><b>InferNex部署前的环境校验工具，提前发现潜在问题，降低部署失败风险</b></p>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex_checker.md)
[![License](https://img.shields.io/badge/License-Mulan_PSL_v2-blue.svg)](../../LICENSE)

</div>

<hr>

## Updates

[26-09] 新增快慢卡检测能力；支持`--enable-connectivity-check`可选网络连通性测试标志与`--dns-check-image`离线场景DNS校验参数；DNS校验改为每节点创建临时Pod验证。

## Overview
InferNex环境校验工具（infernex-checker）是一个独立的CLI（Command-Line Interface，命令行界面）工具，用于对InferNex部署环境进行系统性验证。该工具通过检测硬件、K8s集群及业务配置与环境匹配的关键环节，帮助用户在部署前识别环境问题、在部署失败后快速定位原因，有效降低部署失败风险并提升问题排查效率。

**系统性环境校验**：覆盖硬件层（NPU驱动、固件、资源、网络连通性）、Kubernetes层（CoreDNS、节点状态、资源配额）及业务配置层（存储权限、版本兼容性）等关键环节。

**灵活的执行模式**：支持全量校验或按层级执行（硬件/K8s/业务配置可单独选择），适配不同部署阶段的校验需求。

**结构化校验报告**：提供终端彩色输出和JSON文件双重报告形式，包含详细错误描述、修复建议及资源使用建议，便于问题追踪和自动化处理。

## Quick Start

### Prerequisites

**支持平台**：

| 平台 | 下载链接 |
|------|---------|
| Linux AMD64 | [下载](https://static.openfuyao.cn/openFuyao/infernex/releases/download/latest/bin/linux/amd64/infernex-checker) |
| Linux ARM64 | [下载](https://static.openfuyao.cn/openFuyao/infernex/releases/download/latest/bin/linux/arm64/infernex-checker) |

**软件版本**：

Kubernetes v1.33.0+

**网络要求**：

运行主机可访问Kubernetes API Server

运行主机可通过SSH连接所有目标节点

**权限要求**：

Kubernetes集群访问权限（kubeconfig配置文件）

目标节点SSH登录权限（用户名+密码或密钥）

### Installation

1. 下载二进制文件。

```bash
# 下载二进制文件（以 AMD64 为例）
wget https://static.openfuyao.cn/openFuyao/infernex/releases/download/latest/bin/linux/amd64/infernex-checker

# 添加执行权限
chmod +x infernex-checker

# 验证安装
./infernex-checker --help
```

2. 准备配置文件。

创建`nodes.yaml`，填写目标节点信息：

```yaml
nodes:
  - name: n2                   # 节点名称（需与K8s集群中的节点名一致）
    ip: 192.168.1.10           # 节点IP
    port: 22                   # SSH端口
    user: root                 # SSH用户
    password: "your-password"  # 密码认证（与privateKeyPath二选一）
```

## Usage

**全量校验**：

按硬件层 → K8s层 → 业务配置与环境层顺序执行：

```bash
infernex-checker all --nodes nodes.yaml --values values.yaml --output result.json
```

**参数说明**：
- `--nodes`：节点信息文件路径（必填）
- `--values`：InferNex部署配置文件路径（必填）
- `--kubeconfig`：Kubernetes配置文件路径（可选，默认`~/.kube/config`）
- `--dns-check-image`：DNS校验临时Pod使用的镜像地址（可选，默认为busybox:1.36）。离线部署或集群节点无法拉取公网镜像时，需提前将含nslookup命令的轻量镜像（如busybox）导入私有镜像仓库，并通过此参数指定私有仓库地址，例如--dns-check-image private-registry.example.com/library/busybox:1.36
- `--enable-connectivity-check`：启用网络连通性检查（可选，默认关闭）
- `--output`：JSON结果文件输出路径（可选）
- `--log`：日志文件路径（可选，默认`./infernex-checker.log`）

**分层校验**：

```bash
# 仅硬件层
infernex-checker hardware --nodes nodes.yaml

# 若需启用完整的连通性测试

infernex-checker hardware --nodes nodes.yaml --enable-connectivity-check

# 仅K8s层
infernex-checker k8s --nodes nodes.yaml --dns-check-image busybox:1.36

# 仅业务配置与环境层
infernex-checker config-env --nodes nodes.yaml --values values.yaml
```

## Output

**终端输出**：

```
=== InferNex Checker ===

[Hardware Layer - Single-Node Check: node-01]
  ✅ H-01  NPU driver and firmware installed
  ✅ H-02  Ascend Device Plugin Running and NPU resource registered
  ✅ H-03  NPU model: Ascend910B4
  ℹ️ H-04  NPU available: 8/8
  ✅ H-05  Key host files and directories are complete
  ✅ H-06  hccn.conf configuration is correct
  ✅ H-07  Step 1: NIC TLS switch states are consistent (on)
  ⏭️ H-07  Step 2: Connectivity check skipped (use --enable-connectivity-check to enable)
  ⚠️ H-08  Slow NPU card detected: rank 0 computation time significantly higher than other ranks
    → Check NPU card health status via npu-smi info (temperature, ECC errors, etc.)

[Hardware Layer - Single-Node Check: node-02]
  ✅ H-01  NPU driver and firmware installed
  ✅ H-02  Ascend Device Plugin Running and NPU resource registered
  ⚠️ H-03  NPU model is non-910 series: Ascend310P3
    → Only 910-series supports H-06~H-10, will skip these checks
  ℹ️ H-04  NPU available: 8/8
  ✅ H-05  Key host files and directories are complete
  ⏭️ H-06~H-10 skipped  Only 910-series supports HCCS/RoCE checks

[Hardware Layer - Cross-Node Check] (910-series nodes: node-01)
  ℹ️ H-09  Cross-node communication metrics (1 node): node-01(avg Transit: 2.34ms, 8 ranks)
  ⏭️ H-10  Only 1 910-series node(s) passed single-node checks, skipping cross-node check (at least 2 required)

[K8s Layer]
  ✅ K-01  CoreDNS Running, all nodes DNS resolution is normal
  ✅ K-02  node-01 Ready
  ✅ K-02  node-02 Ready
  ✅ K-03  node-01 has no taints
  ✅ K-03  node-02 has no taints
  ℹ️ K-04  node-01  Allocatable: CPU 48, Memory 256Gi | Remaining: CPU 40, Memory 200Gi
  ℹ️ K-04  node-02  Allocatable: CPU 48, Memory 256Gi | Remaining: CPU 40, Memory 200Gi

[Config-Env Layer: node-01]
  ✅ B-01  /home/llm_cache directory exists and is writable
  ✅ B-02  Driver 24.1.RC2 is compatible with image v0.13.0

[Config-Env Layer: node-02]
  ✅ B-01  /home/llm_cache directory exists and is writable
  ✅ B-02  Driver 24.1.RC2 is compatible with image v0.13.0

────────────────────────────────
Result: 19 passed, 0 failed, 2 warning, 2 skipped, 5 info

⚠️ Warning items:
  H-03  node-02  NPU model is non-910 series: Ascend310P3
  H-08  node-01  Slow NPU card detected: rank 0 computation time significantly higher than other ranks

⏭️ Skipped checks:
  H-07  node-01  Step 2: Connectivity check skipped (use --enable-connectivity-check to enable)
  H-10  Only 1 910-series node(s) passed single-node checks, skipping cross-node check (at least 2 required)

ℹ️ Info items:
  H-04  node-01  NPU available: 8/8
  H-04  node-02  NPU available: 8/8
  H-09  Cross-node communication metrics (1 node): node-01(avg Transit: 2.34ms, 8 ranks)
  K-04  node-01  Allocatable: CPU 48, Memory 256Gi | Remaining: CPU 40, Memory 200Gi
  K-04  node-02  Allocatable: CPU 48, Memory 256Gi | Remaining: CPU 40, Memory 200Gi
```

**JSON输出**：

结构化JSON文件包含详细的校验结果、错误描述和修复建议，便于自动化处理和问题追踪。使用`--output`参数指定输出路径。
