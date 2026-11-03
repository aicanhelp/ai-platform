
<div align="center">
  <h1 align="center">
      InferNex Bridge
  </h1>

  <p><b>InferNex 接入 KServe 的适配层，支持双 CRD 声明式部署</b></p>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex_bridge.md)
[![License](https://img.shields.io/badge/License-Mulan_PSL_v2-blue.svg)](../../LICENSE)
[![Helm](https://img.shields.io/badge/Helm-chart-00a1d6.svg)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex_bridge.md#%E5%BC%80%E5%A7%8B%E5%AE%89%E8%A3%85)
[![KServe](https://img.shields.io/badge/KServe-LLMInferenceService-326ce5.svg)](https://kserve.github.io/website/docs/0.18/install/llmisvc-install)

</div>

<hr>

## Overview

InferNex Bridge 是 InferNex 接入 KServe 的**适配层**（Controller + Webhook），根据 `LLMInferenceService` (KServe 的 LLM 推理 CRD) / `InferNexService`(InferNex Bridge 原生推理 CRD) 声明，自动部署并调和 InferNex 推理套件。

提供两种部署形式：

- **KServe + InferNex Bridge**：推理引擎 / Hermes Router 由 KServe 部署，增强组件由 InferNex Bridge 部署。
- **InferNex Bridge**：由 InferNex Bridge 统一编排推理引擎、Hermes Router 及增强组件（Mooncake KVCache、cache-indexer、proxy-server、Elastic-Scaler、Tidal Controller、ResourceScalingGroup、Eagle-Eye Hardware Monitor、Eagle-Eye Hardware Diagnosis、Eagle-Eye Network Performance Exporter）。

与 [InferNex 主 chart](../../README.md) 的一键 Helm 全栈安装为不同入口。

增强组件能力范围与 [InferNex 集成部署用户手册](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex.md) 一致。架构与设计依据见 [OFEP-0040 InferNex 接入 KServe 适配层提案](https://gitcode.com/openFuyao/ofep/blob/main/ofeps/sig-ai-inference/0040-ofep-Infernex%E6%8E%A5%E5%85%A5kserve%E9%80%82%E9%85%8D%E5%B1%82.md)。

## Specification

InferNex Bridge 面向昇腾（Ascend）NPU 全栈推理，支持 `LLMInferenceService` 与 `InferNexService`双入口。版本兼容、Webhook 补丁行为、默认镜像及责任边界等详见 **[InferNex Bridge 技术规格说明文档](docs/InferNex-Bridge-Technical-Specification.md)**。

### Version compatibility

| InferNex-Bridge | KServe LLMISVC 控制器 | InferNex |
| --- | --- | --- |
| 26.6.0 | 0.17.0、0.18.0、0.19.0 | 26.6.0 |
| 26.9.0 / master（`0.0.0-latest`） | 0.18.0、0.19.0 | 26.9.0 / master |

当前主线（master / `0.0.0-latest`，自 26.9.0 起）不再支持 KServe 0.17.0 / InferNex 26.6.0（0.17.0 的 `kserve-config-llm-scheduler` 与旧版 Hermes Router 参数冲突，26.9.0 起尽量不再由 Mutating Webhook 改写该 Config，故无法继续适配）。完整说明见 [技术规格 §1](docs/InferNex-Bridge-Technical-Specification.md#1-部署规格)。

KServe 安装前提见 [LLMInferenceService Prerequisites](https://kserve.github.io/website/docs/0.18/install/llmisvc-install#prerequisites)；InferNex 整体部署规格见 [InferNex 规格 #42](https://gitcode.com/openFuyao/InferNex/issues/42)。

## Quick Start

### Prerequisites

- Kubernetes 集群，`kubectl`、Helm v3+。
- **推理集群**：NPU Operator、LWS 等，详见 [InferNex 用户手册 — 前提条件](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex.md#%E5%89%8D%E6%8F%90%E6%9D%A1%E4%BB%B6)。
- **KServe**：须安装 [KServe](https://kserve.github.io/website/docs/0.18/install/llmisvc-install)（LLMISVC 控制器版本 0.18.0–0.19.0）。
- **网关访问**：Envoy Gateway 及 Gateway API、GIE 相关 CRD。

### Binary Deployment

以命名空间 `infernex-bridge-system`、release 名称 `infernex-bridge` 为例，执行：

```bash
helm upgrade --install infernex-bridge oci://cr.openfuyao.cn/charts/infernex-bridge \
  --version 0.0.0-latest \
  -n infernex-bridge-system --create-namespace --wait --timeout 10m
```

其中 `0.0.0-latest` 需替换为具体 Chart 版本。

### Source Deployment

1. 从仓库拉取项目。

   ```bash
   git clone https://gitcode.com/openFuyao/InferNex.git
   ```

2. 安装 InferNex Bridge。

   以命名空间 `infernex-bridge-system`、release 名称 `infernex-bridge` 为例，在 `InferNex/component/InferNex-Bridge` 目录下执行如下命令：

   ```bash
   cd InferNex/component/InferNex-Bridge
   helm upgrade --install infernex-bridge ./chart/infernex-bridge \
     -n infernex-bridge-system --create-namespace --wait --timeout 10m
   ```

### Verify Deployment

1. 确认 Controller Pod 与 Service 已就绪。

   ```bash
   kubectl get pods,svc -n infernex-bridge-system
   ```

2. 确认 Mutating / Validating Webhook 已注册。

   ```bash
   kubectl get mutatingwebhookconfiguration,validatingwebhookconfiguration | grep infernex-bridge
   ```

3. 等待 Webhook 与证书就绪，而后部署 InferNex。

安装参数、卸载及 InferNex 部署详见 [InferNex Bridge 用户手册](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex_bridge.md)。

### Deploy InferNex

InferNex Bridge 安装并验证通过后，**在 `config/examples/` 下选取示例 YAML，用 `kubectl apply` 提交即可部署 InferNex**。

1. **选择示例文件**

   **LLMISVC**

   | 模式 | 目录 | 说明 |
   | --- | --- | --- |
   | Aggregate | [llmisvc/aggregate/](config/examples/llmisvc/aggregate/) | `ag-01-*.yaml` .. `ag-03-*.yaml` |
   | Disaggregated (PD) | [llmisvc/disaggregated/](config/examples/llmisvc/disaggregated/) | `pd-01-*.yaml` .. `pd-05-*.yaml` |

   示例含 `huggingface-download` initContainer，部署时从 Hugging Face Hub 拉取模型（如 `ag-01-single-node-single-card.yaml`）。

   **InferNexService（推理引擎模板与 llmisvc 同 Spec ID 对齐）**

   | 模式 | 目录 | 说明 |
   | --- | --- | --- |
   | Aggregate | [insvc/aggregate/](config/examples/insvc/aggregate/) | `ag-01-*.yaml` .. `ag-03-*.yaml` |
   | Disaggregated (PD) | [insvc/disaggregated/](config/examples/insvc/disaggregated/) | `pd-01-*.yaml` .. `pd-05-*.yaml` |

   示例含 Mooncake initContainer、`huggingface-download` 与完整 `vllm serve` 启动参数（与对应 llmisvc 示例一致）；`InferNexService.spec.model` + `baseRefs` 指向 `InferNexServiceConfig` 中的推理引擎模板。

   - **KServe `LLMInferenceService` + InferNex Bridge**：选 `llmisvc/` 下对应 Spec ID YAML（含 `infernex.io/runtime: "true"`）。
   - **InferNex Bridge 直管 `InferNexService`**：选 `insvc/` 下同 Spec ID YAML。

2. **提交 YAML**

   ```bash
   cd InferNex/component/InferNex-Bridge/config/examples/insvc/aggregate
   kubectl apply -f ag-01-single-node-single-card.yaml -n kserve
   ```

   将路径与文件名替换为目标场景。
