
<div align="center">
  <h1 align="center">
      InferNex Bridge
  </h1>

  <p><b>Adaptation layer for integrating InferNex with KServe, supporting dual CRD declarative deployment</b></p>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex_bridge.md)
[![License](https://img.shields.io/badge/License-Mulan_PSL_v2-blue.svg)](../../LICENSE)
[![Helm](https://img.shields.io/badge/Helm-chart-00a1d6.svg)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex_bridge.md#%E5%BC%80%E5%A7%8B%E5%AE%89%E8%A3%85)
[![KServe](https://img.shields.io/badge/KServe-LLMInferenceService-326ce5.svg)](https://kserve.github.io/website/docs/0.18/install/llmisvc-install)

</div>

<hr>

## Overview

InferNex Bridge is the **adaptation layer** (Controller + Webhook) for integrating InferNex with KServe. Based on `LLMInferenceService` (KServe's LLM inference CRD) / `InferNexService` (InferNex Bridge's native inference CRD) declarations, it automatically deploys and reconciles the InferNex inference suite.

Two deployment modes are available:

- **KServe + InferNex Bridge**: The inference engine / Hermes Router is deployed by KServe, while enhancement components are deployed by InferNex Bridge.
- **InferNex Bridge**: InferNex Bridge uniformly orchestrates the inference engine, Hermes Router, and enhancement components (Mooncake KVCache, cache-indexer, proxy-server, Elastic-Scaler, Tidal Controller, ResourceScalingGroup, Eagle-Eye Hardware Monitor, Eagle-Eye Hardware Diagnosis, Eagle-Eye Network Performance Exporter).

This provides a different entry point from the one-click Helm full-stack installation via the [InferNex main chart](../../README.md).

The scope of enhancement component capabilities is consistent with the [AI Inference Integrated Deployment](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/en/ai_inference_infernex/user_guide/ai_inference_infernex.md). For architecture and design rationale, see [OFEP-0040 InferNex KServe Adaptation Layer Proposal](https://gitcode.com/openFuyao/ofep/blob/main/ofeps/sig-ai-inference/0040-ofep-Infernex%E6%8E%A5%E5%85%A5kserve%E9%80%82%E9%85%8D%E5%B1%82.md).

## Specification

InferNex Bridge targets Ascend NPU full-stack inference, supporting dual entry points via `LLMInferenceService` and `InferNexService`. For version compatibility, Webhook patch behavior, default images, and responsibility boundaries, see **[InferNex Bridge Technical Specification](docs/InferNex-Bridge-Technical-Specification.md)**.

### Version compatibility

| InferNex-Bridge | KServe LLMISVC controller | InferNex |
| --- | --- | --- |
| 26.6.0 | 0.17.0, 0.18.0, 0.19.0 | 26.6.0 |
| 26.9.0 / master (`0.0.0-latest`) | 0.18.0, 0.19.0 | 26.9.0 / master |

The current master / `0.0.0-latest` line (since 26.9.0) no longer supports KServe 0.17.0 / InferNex 26.6.0 (`kserve-config-llm-scheduler` on 0.17.0 conflicted with older Hermes Router parameters; from 26.9.0 the Mutating Webhook largely stops rewriting that Config, so that path is no longer viable). Full rationale: [Technical Specification §1](docs/InferNex-Bridge-Technical-Specification.md#1-部署规格).

For KServe installation prerequisites, see [LLMInferenceService Prerequisites](https://kserve.github.io/website/docs/0.18/install/llmisvc-install#prerequisites); for InferNex overall deployment specifications, see [InferNex Specification #42](https://gitcode.com/openFuyao/InferNex/issues/42).

## Quick Start

### Prerequisites

- Kubernetes cluster, `kubectl`, Helm v3+.
- **Inference cluster**: NPU Operator, LWS, etc. See [InferNex User Guide — Prerequisites](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex.md#%E5%89%8D%E6%8F%90%E6%9D%A1%E4%BB%B6).
- **KServe**: [KServe](https://kserve.github.io/website/docs/0.18/install/llmisvc-install) must be installed (LLMISVC controller version 0.18.0–0.19.0).
- **Gateway access**: Envoy Gateway, Gateway API, and GIE-related CRDs.

### Binary Deployment

Using namespace `infernex-bridge-system` and release name `infernex-bridge` as an example, execute:

```bash
helm upgrade --install infernex-bridge oci://cr.openfuyao.cn/charts/infernex-bridge \
  --version 0.0.0-latest \
  -n infernex-bridge-system --create-namespace --wait --timeout 10m
```

Replace `0.0.0-latest` with the specific Chart version.

### Source Deployment

1. Clone the project from the repository.

   ```bash
   git clone https://gitcode.com/openFuyao/InferNex.git
   ```

2. Install InferNex Bridge.

   Using namespace `infernex-bridge-system` and release name `infernex-bridge` as an example, execute the following command in the `InferNex/component/InferNex-Bridge` directory:

   ```bash
   cd InferNex/component/InferNex-Bridge
   helm upgrade --install infernex-bridge ./chart/infernex-bridge \
     -n infernex-bridge-system --create-namespace --wait --timeout 10m
   ```

### Verify Deployment

1. Confirm that the Controller Pod and Service are ready.

   ```bash
   kubectl get pods,svc -n infernex-bridge-system
   ```

2. Confirm that the Mutating / Validating Webhooks are registered.

   ```bash
   kubectl get mutatingwebhookconfiguration,validatingwebhookconfiguration | grep infernex-bridge
   ```

3. Wait for Webhooks and certificates to be ready, then deploy InferNex.

For installation parameters, uninstallation, and InferNex deployment, see the [AI Inference InferNex Bridge](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/en/ai_inference_infernex/user_guide/ai_inference_infernex_bridge.md).

### Deploy InferNex

After InferNex Bridge is installed and verified, **select an example YAML from `config/examples/` and submit it with `kubectl apply` to deploy InferNex**.

1. **Select an Example File**

   **LLMISVC**

   | Mode | Directory | Description |
   | --- | --- | --- |
   | Aggregate | [llmisvc/aggregate/](config/examples/llmisvc/aggregate/) | `ag-01-*.yaml` .. `ag-03-*.yaml` |
   | Disaggregated (PD) | [llmisvc/disaggregated/](config/examples/llmisvc/disaggregated/) | `pd-01-*.yaml` .. `pd-05-*.yaml` |

   Examples include the `huggingface-download` initContainer, which pulls models from Hugging Face Hub during deployment (e.g., `ag-01-single-node-single-card.yaml`).

   **InferNexService (inference engine templates aligned with the same Spec IDs as llmisvc)**

   | Mode | Directory | Description |
   | --- | --- | --- |
   | Aggregate | [insvc/aggregate/](config/examples/insvc/aggregate/) | `ag-01-*.yaml` .. `ag-03-*.yaml` |
   | Disaggregated (PD) | [insvc/disaggregated/](config/examples/insvc/disaggregated/) | `pd-01-*.yaml` .. `pd-05-*.yaml` |

   Examples include the Mooncake initContainer, `huggingface-download`, and complete `vllm serve` startup parameters (consistent with the corresponding llmisvc examples); `InferNexService.spec.model` + `baseRefs` point to the inference engine templates in `InferNexServiceConfig`.

   - **KServe `LLMInferenceService` + InferNex Bridge**: Select the YAML with the corresponding Spec ID under `llmisvc/` (contains `infernex.io/runtime: "true"`).
   - **InferNex Bridge directly managing `InferNexService`**: Select the YAML with the same Spec ID under `insvc/`.

2. **Submit the YAML**

   ```bash
   cd InferNex/component/InferNex-Bridge/config/examples/insvc/aggregate
   kubectl apply -f ag-01-single-node-single-card.yaml -n kserve
   ```

   Replace the path and filename with your target scenario.
