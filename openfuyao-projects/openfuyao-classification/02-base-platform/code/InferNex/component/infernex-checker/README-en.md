
<div align="center">
  <h1 align="center">
      InferNex-Checker
  </h1>

  <p><b>Pre-deployment environment validation tool for InferNex, identifying potential issues in advance to reduce deployment failure risks</b></p>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/en/ai_inference_infernex/user_guide/ai_inference_infernex_checker.md)
[![License](https://img.shields.io/badge/License-Mulan_PSL_v2-blue.svg)](../../LICENSE)

</div>

<hr>

## Updates

[26-09] Add slow NPU card detection capability; support `--enable-connectivity-check` flag for optional network connectivity tests and `--dns-check-image` parameter for offline DNS validation; improve DNS validation by creating temporary Pod on each node for verification.

## Overview
InferNex-Checker (infernex-checker) is a standalone CLI (Command-Line Interface) tool designed for systematic validation of InferNex deployment environments. By inspecting critical aspects of hardware, Kubernetes clusters, and business configuration-environment compatibility, it helps users identify environmental issues before deployment and quickly pinpoint causes after deployment failures, effectively reducing deployment failure risks and improving troubleshooting efficiency.

**Systematic Environment Validation**: Covers key aspects across the hardware layer (NPU driver, firmware, resources, network connectivity), Kubernetes layer (CoreDNS, node status, resource quotas), and business configuration layer (storage permissions, version compatibility).

**Flexible Execution Modes**: Supports full validation or layer-by-layer execution (hardware/K8s/business configuration can be selected individually), adapting to validation needs at different deployment stages.

**Structured Validation Reports**: Provides dual report formats — terminal colored output and JSON files — including detailed error descriptions, remediation suggestions, and resource usage recommendations, facilitating issue tracking and automated processing.

## Quick Start

### Prerequisites

**Supported Platforms**:

| Platform | Download Link |
|------|---------|
| Linux AMD64 | [Download](https://static.openfuyao.cn/openFuyao/infernex/releases/download/latest/bin/linux/amd64/infernex-checker) |
| Linux ARM64 | [Download](https://static.openfuyao.cn/openFuyao/infernex/releases/download/latest/bin/linux/arm64/infernex-checker) |

**Software Version**:

Kubernetes v1.33.0+

**Network Requirements**:

The host running the tool must be able to access the Kubernetes API Server

The host running the tool must be able to connect to all target nodes via SSH

**Permission Requirements**:

Kubernetes cluster access permissions (kubeconfig configuration file)

Target node SSH login permissions (username + password or key)

### Installation

1. Download the binary file.

```bash
# Download the binary file (AMD64 example)
wget https://static.openfuyao.cn/openFuyao/infernex/releases/download/latest/bin/linux/amd64/infernex-checker

# Add execute permission
chmod +x infernex-checker

# Verify installation
./infernex-checker --help
```

2. Prepare the configuration file.

Create `nodes.yaml` and fill in target node information:

```yaml
nodes:
  - name: n2                   # Node name (must match the node name in the K8s cluster)
    ip: 192.168.1.10           # Node IP
    port: 22                   # SSH port
    user: root                 # SSH user
    password: "your-password"  # Password authentication (choose one between password and privateKeyPath)
```

## Usage

**Full Validation**:

Execute in order: hardware layer → K8s layer → business configuration and environment layer:

```bash
infernex-checker all --nodes nodes.yaml --values values.yaml --output result.json
```

**Parameter Description**:
- `--nodes`: Node information file path (required)
- `--values`: InferNex deployment configuration file path (required)
- `--kubeconfig`: Kubernetes configuration file path (optional, default `~/.kube/config`)
- `--dns-check-image`: Image address for DNS validation temporary Pod (optional, default `busybox:1.36`). For offline deployments or when cluster nodes cannot pull public images, you must pre-import a lightweight image with the `nslookup` command (such as busybox) into your private registry and specify the private registry address via this parameter, e.g., `--dns-check-image private-registry.example.com/library/busybox:1.36`
- `--enable-connectivity-check`: Enable network connectivity checks (optional, disabled by default)
- `--output`: JSON result file output path (optional)
- `--log`: Log file path (optional, default `./infernex-checker.log`)

**Layer-by-layer Validation**:

```bash
# Hardware layer only
infernex-checker hardware --nodes nodes.yaml

# To enable full connectivity testing

infernex-checker hardware --nodes nodes.yaml --enable-connectivity-check

# K8s layer only
infernex-checker k8s --nodes nodes.yaml --dns-check-image busybox:1.36

# Business configuration and environment layer only
infernex-checker config-env --nodes nodes.yaml --values values.yaml
```

## Output

**Terminal Output**:

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

**JSON Output**:

The structured JSON file contains detailed validation results, error descriptions, and remediation suggestions, facilitating automated processing and issue tracking. Use the `--output` parameter to specify the output path.