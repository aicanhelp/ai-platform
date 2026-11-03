
<div align="center">
  <h1 align="center">
      InferNex
  </h1>

  <p><b>End-to-end one-click integrated deployment for the openFuyao AI inference service framework</b></p>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex.md)
[![License](https://img.shields.io/badge/License-Mulan_PSL_v2-blue.svg)](./LICENSE)
[![Helm](https://img.shields.io/badge/Helm-chart-00a1d6.svg)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex.md#%E5%BC%80%E5%A7%8B%E5%AE%89%E8%A3%85)
[![GIE](https://img.shields.io/badge/Gateway_API_Inference_Extension-v1.1.0-orange.svg)](https://github.com/kubernetes-sigs/gateway-api-inference-extension)

</div>

<hr>

## Updates
 - [26-09] InferNex default inference engine upgraded to vLLM-Ascend v0.23.0, with reference configs for DeepSeek-V4-Flash / GLM-5.2; InferNex-Bridge Mutating Webhook no longer patches the `kserve-config-llm-scheduler` LLMInferenceServiceConfig; Hermes-router delivered Decode-first D/PD disaggregated scheduling with tokenizer/KServe path alignment, cache-indexer supports Decode-side L1 KVCache discovery, and proxy-server supports D/PD Header orchestration; eagle-eye added end-to-end distributed link tracing, and infernex-checker enhanced optional connectivity checks and slow-card analysis.
 - [26-06] The inference backend switched to LeaderWorkerSet (LWS) deployment orchestration, natively supporting multi-DP coordination; PD-Orchestrator's elastic-scaler added the APA scaling algorithm, supporting multi-metric scaling; Hermes-router added routing strategies based on compute saturation and latency prediction; cache-indexer implemented L3-level KV-aware perception, collaborating with Mooncake to support global KVCache indexing; eagle-eye added weight distribution and Lingqu network dynamic metric collection; InferNex added a Helm pre-deployment validation tool, covering NPU driver, hardware resources, and network communication environment checks to identify deployment risks in advance.
 - [26-05] Added InferNex-Bridge component, compatible with KServe integration into the InferNex inference suite, supporting dual CRD declarative deployment via LLMInferenceService and InferNexService, with the adaptation layer automatically completing orchestration and routing integration.
 - [26-03] Added PD-Orchestrator component, supporting dynamic PD group scaling; intelligent routing added disaster recovery capabilities, including automatic traffic switching, fault awareness, and request retry; inference backend component refactored and updated, supporting configuration of different vLLM inference engine versions and non-Hugging Face models.

<details>
<summary>More</summary>

 - [25-12] Added inference observability sub-component; intelligent routing implemented gateway plugin based on the GIE framework.
 - [25-09] Released the AI inference integrated deployment alpha version! Supporting features such as intelligent routing Hermes-router with KVCache aware and other strategies, xPyD disaggregated inference engine, global KVCache metadata management, and Mooncake distributed KVCache management system integration.

</details>

## Overview
This project is built on mainstream LLM inference technology stacks and the K8s official project GIE (Gateway API Inference Extension), integrating the following K8s-native, high-performance, and scalable sub-features, aiming to improve inference throughput and reduce latency, providing efficient and reliable technical support for AI service deployment.

**Intelligent Routing System (Hermes-router)**: A gateway plugin implemented based on the GIE framework, featuring dynamic request distribution and load balancing capabilities; supports multi-dimensional perception capabilities including diverse compute load awareness, KV hit awareness, request pressure awareness, request length awareness, and semantic awareness. Users can leverage built-in strategy extensions (KVCache aware strategy, PD long/short request bucketing strategy) to achieve optimal node routing for inference requests.

**xPyD Disaggregated Inference Engine**: An AI inference backend built on the high-performance vLLM inference engine, supporting xPyD architecture, inference node auto-discovery (Proxy Server), Mooncake KVCache storage, and flexible multi-instance deployment; the inference engine is deployed via LeaderWorkerSet (LWS), natively supporting multi-DP coordination, and DP load balancing strategies can be configured through `dataParallelSize` and `dataParallelSizeLocal`.

**PD-Orchestrator**: An elastic orchestration component for PD disaggregated scenarios, integrating tidal algorithms, scaling decision frameworks, and dynamic PD group scaling capabilities; supports metric-driven and event-driven resource scaling, group-level and intra-group proportional or custom strategy scaling, and exposes extensibility for user-defined decision and resource management logic.

**Distributed KVCache Management**: Utilizes the Mooncake Hccl Transfer Engine for high-speed KVCache transfer between PD nodes.

**Global KVCache Index (cache-indexer)**: Based on vLLM's KV Event mechanism and providing RESTful interfaces, builds a distributed global KVCache metadata prefix tree, enabling routing KV-aware perception capabilities for efficient utilization of global KVCache resources.

**Inference Observability System (eagle-eye)**: Based on Prometheus standard data collection and reporting formats, provides key observability metrics for inference scenarios, covering business runtime metrics, system runtime metrics, and resource health metrics; through NATS asynchronous message queue publish-subscribe mode, provides real-time (millisecond-level) observability of key business runtime metrics, supporting near-real-time decision capabilities for critical acceleration modules.

## Dependent Components

This integrated deployment solution includes the following sub-components and their version information:

| Component | Version | Optional | Description |
|---------|------|------|------|
| inference-backend | latest | Required | Inference engine backend based on vllm/vllm-ascend |
| pd-orchestrator | latest | Required | PD dynamic scaling component |
| Hermes-router | latest | Optional | Intelligent routing system |
| cache-indexer | latest | Optional | Distributed global KVCache metadata management component |
| eagle-eye | latest | Optional | Observability system |
| vLLM-Ascend | v0.23.0 | Optional | Inference engine framework. Users can configure other versions of vLLM-Ascend images. |
| Mooncake | 0.3.11 | Optional | Distributed KVCache management. Changes with vLLM-Ascend image version. |

> Note: Some components are optional and must be enabled through configuration. For detailed configuration, refer to the [Configuration](#configuration) section.

## Quick Start

### Prerequisites
- Kubernetes v1.29.0 and above (v1.33.0 and above recommended).
- [npu-operator](https://gitcode.com/openFuyao/npu-operator) component installed.
- [LWS](https://lws.sigs.k8s.io/docs/installation/) component installed.
- Metrics server v0.8.0 and above must be installed in the cluster.

### Binary Deployment

1. Pull the project installation package.
    ```bash
    helm pull oci://cr.openfuyao.cn/charts/infernex --version xxx
    ```
    Replace `xxx` with the specific project installation package version, such as `0.0.0-latest`. The pulled package is in compressed format.

2. Decompress the installation package.
    ```bash
    tar -xzvf infernex-xxx.tgz
    ```
    Replace `xxx` with the specific project installation package version, such as `0.0.0-latest`.

3. Install and deploy.

    Using namespace `ai-inference` and release name `infernex` as an example, execute the following command in the directory at the same level as `infernex`:
   ```bash
   helm install -n ai-inference infernex ./infernex
   ```

### Source Deployment

1. Clone the project from the repository.

   ```bash
   git clone https://gitcode.com/openFuyao/InferNex.git
   ```

2. Install and deploy.

   Using namespace `ai-inference` and release name `infernex` as an example, execute the following command in the directory at the same level as `InferNex`:
   ```bash
   cd InferNex/charts/infernex
   helm dependency build
   helm install -n ai-inference infernex .
   ```

### Configuration
For detailed configuration instructions, refer to the [Configuring AI Inference Integrated Deployment](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex.md#%E9%85%8D%E7%BD%AEai%E6%8E%A8%E7%90%86%E9%9B%86%E6%88%90%E9%83%A8%E7%BD%AE) section of the InferNex User Guide.


## Performance

- **Fixed-length system prompt reuse scenario**: This scenario simulates a workload with fixed-length system prompts where KVCache can be reused. The dataset contains 120 types of 8k-length system prompts, each repeated 4 times, totaling 480 requests. Request times follow a Poisson distribution with concurrency of 8. Using random routing as the performance baseline, after enabling InferNex optimization capabilities, aggregate deployment shows an average TTFT reduction of approximately 54% and an average TPS improvement of approximately 20%, demonstrating significant performance gains.
- **Multi-turn conversation scenario**: This scenario simulates multiple users engaging in continuous conversations with an LLM. A single session contains multiple rounds of requests, with 120 independent users generating 480 requests total. The first round requests 16k tokens, each round returns 128 tokens, and subsequent rounds append 1k tokens for 4 rounds total, averaging approximately 17.5k per request. Request times follow a Poisson distribution with concurrency of 8. Using random routing as the performance baseline, after enabling InferNex optimization capabilities, aggregate deployment shows an average TTFT reduction of approximately 60% and an average TPS improvement of approximately 44%, demonstrating significant performance gains.
- **Inference observability system**: Pod resource consumption CPU <20m, MEM 3000~3500M; second-level reporting publisher average **collection duration <10ms**, subscriber average log **receiving latency <1ms**. For details, see the [eagle-eye Performance Test Report](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/reports/performance/eagle-eye%E6%80%A7%E8%83%BD%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-v26.03.md).

> For detailed performance data and comparative analysis of each optimization strategy, see the [InferNex Performance Test Report](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/reports/performance/InferNex%E6%95%B4%E4%BD%93%E6%80%A7%E8%83%BD%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-v26.06.md).

## Specification Sheet

Each **InferNex major release** publishes the **theoretical supported version ranges** and **verified combinations** for subcomponents. The **core concept** is to use the **InferNex release version as the anchor point**, declaring the support relative to this anchor for inference engines, smart router, pd orchestrators, observability, accelerators, Kubernetes, and other dependencies in the specification sheet; prioritize this table when performing integrated deployments.

> In the table, **Yes** under "Verified" means effective validation has been completed on the corresponding combination; **No** means **not yet verified** (may still work, but not committed as accepted). **All component versions below are theoretical support ranges**, unless otherwise noted in remarks.

### Subcomponent Specification

The following lists **software components** that the InferNex release depends on or integrates with, and their support versions relative to the InferNex anchor version (baseline: **InferNex 26.9.0**).
v26.09 removes vllm-ascend versions earlier than 0.13.0; those versions are no longer recommended.

| Component | Version | Verified | Remarks |
|------|------|----------|------|
| Inference engine (vllm-ascend) | v0.23.0 | Yes | Default version; updated in v26.09 |
| Inference engine (vllm-ascend) | v0.22.1rc1 | No | |
| Inference engine (vllm-ascend) | v0.21.0rc1 | No | |
| Inference engine (vllm-ascend) | v0.20.2rc1 | No | |
| Inference engine (vllm-ascend) | v0.19.1rc1 | Yes | |
| Inference engine (vllm-ascend) | v0.18.0 | Yes | |
| Inference engine (vllm-ascend) | v0.17.0rc1 | No | |
| Inference engine (vllm-ascend) | v0.16.0rc1 | No | |
| Inference engine (vllm-ascend) | v0.15.0rc1 | No | |
| Inference engine (vllm-ascend) | v0.14.0rc1 | Yes | |
| Inference engine (vllm-ascend) | v0.13.0 | Yes | |
| Open-source gateway (Istio) | 1.29.0 | No | |
| Open-source gateway (Istio) | 1.28.0 | Yes | |
| Intelligent routing (Hermes-router) | 26.9.0 | Yes | Default version |
| Intelligent routing (Hermes-router) | 0.21.0 | Yes | |
| cache-indexer | 26.9.0 | Yes | Default version |
| cache-indexer | 0.21.1 | Yes | |
| PD-Orchestrator | 26.9.0 | Yes | Default version |
| PD-Orchestrator | 0.21.1 | Yes | |
| PD-Orchestrator | 0.21.0 | Yes | |
| PD-Orchestrator | 0.20.0 | Yes | |
| eagle-eye | 26.9.0 | Yes | Default version; added end-to-end distributed tracing in v26.09 |
| eagle-eye | 0.22.0 | Yes | |
| eagle-eye | 0.21.0 | Yes | |
| infernex-checker | 26.9.0 | Yes | Added single-node / cross-node slow-card detection in v26.09 |

### Hardware Prerequisite

The following lists **inference accelerator hardware** targeted by InferNex and their verification status (orthogonal to component, environment, and model specifications; can be combined).

| Hardware Model | Verified | Remarks |
|----------|----------|------|
| Ascend 910B4 | Yes | Default chart target hardware for 26.9.0 |
| Ascend 910B3 | Yes | |
| Ascend 310P | No | |

### Environment Prerequisite

The following lists **environments of cluster and platform** required to run InferNex (including the LWS dependency).

| Environment Item | Version / Requirement | Verified | Remarks |
|--------|-------------|----------|------|
| Kubernetes | 1.34.0 | Yes | |
| Kubernetes | 1.33.0 | Yes | |
| Kubernetes | 1.29.0 | No | |
| LeaderWorkerSet (LWS Operator) | v0.8.0 | Yes | Prerequisite for InferNex 26.9.0 (LWS); chart uses `leaderworkerset.x-k8s.io/v1`; see [LWS official documentation](https://lws.sigs.k8s.io/docs/installation/) |

### Model Supported

The following covers the model support scope under **default chart/values.yaml**; sorted by use case for cross-reference with component, hardware, and environment specifications. MoE models typically require multi-DP configuration such as `dataParallelSize` / `dataParallelSizeLocal`.

> Note: Model spec validations below use Atlas 800I A2 machines with NPU cards 910B3 (64G) / 910B4 (32G).

| Category | Model | Download | Deployment Spec | Deployment Example | Remarks |
|------|------|------|----------|------|------|
| Default dense baseline | Qwen3-8B | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Single node, 2x 910B4, prefill tp1 dp1, decode tp1 dp1 | [values.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/charts/infernex/values.yaml) | Default values.yaml deployment for 26.6.0 |
| Default dense baseline | Qwen3-8B | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Single node, single 910B4, aggregated tp1 dp1 | [Qwen3-8B-vLLM-aggregated-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/Qwen3-8B-vLLM-aggregated-random.yaml) | |
| Basic MoE | Qwen3-Coder-30B-A3B (and `-Instruct` variant) | [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | Single node, 8x 910B4, prefill tp2 dp2, decode tp2 dp2 | [Qwen3-Coder-30B-A3B-Instruct-vLLM-pd-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/Qwen3-Coder-30B-A3B-Instruct-vLLM-pd-random.yaml) | Recommended starter MoE model for 26.6.0 |
| Mainstream MoE large model | MiniMax-M2.7-w8a8-QuaRot | [MiniMax-M2.7-w8a8-QuaRot](https://www.modelscope.ai/models/vllm-ascend/MiniMax-M2.7-w8a8-QuaRot) | 4 nodes, 32x 910B3, prefill tp8 dp2, decode tp8 dp2 | [Minimax-m2.7-vLLM-pd-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/Minimax-m2.7-vLLM-pd-random.yaml) | |
| Mainstream MoE large model | GLM-5.1-w4a8 | [GLM-5.1-w4a8](https://modelers.cn/models/Eco-Tech/GLM-5.1-w4a8) | 4 nodes, 32x 910B3, prefill tp8 dp2, decode tp2 dp8 | [GLM-5.1-w4a8-vLLM-pd-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/GLM-5.1-w4a8-vLLM-pd-random.yaml) | vllm-ascend uses nightly-main-0606 image |
| Mainstream MoE large model | GLM-5.2-w8a8 | [GLM-5.2-w8a8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w8a8) | 4 nodes, 32x 910B3, aggregated tp8 dp4 | [GLM-5.2-w8a8-vLLM-aggregated-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/GLM-5.2-w8a8-vLLM-aggregated-random.yaml) | vllm-ascend uses glm5.2-openeuler image |
| Mainstream MoE large model | DeepSeek-V4-Flash-w8a8-mtp | [DeepSeek-V4.1-Flash-w8a8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8) | 4 nodes, 32x 910B4, prefill tp8 dp2, decode tp8 dp2 | [DeepSeek-V4-Flash-w8a8-mtp-vLLM-pd-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/DeepSeek-V4-Flash-w8a8-mtp-vLLM-pd-random.yaml) | vllm-ascend v0.23.0 |

### Multi-Component Integration Specification

The following lists companion component versions declared by InferNex **26.9.0** under the **default chart/values combination**. It complements the previous «Subcomponent Specification» section: this section describes the **release default integration snapshot**, while the previous one describes **per-component verifiable version boundaries**. If this conflicts with the official release notes, the release notes take precedence.

| Name | InferNex 26.9.0 default config |
|------|------|
| Inference engine (vllm-ascend) | v0.23.0 |
| Open-source gateway (Istio) | 1.28.0 |
| Intelligent routing (Hermes-router) | 26.9.0 |
| cache-indexer | 26.9.0 |
| PD-Orchestrator | 26.9.0 |
| eagle-eye | 26.9.0 |
| Kubernetes | 1.34.0 |
| Default hardware | 910B4 |
| Default dense model | Qwen3-8B |
| Verified baseline MoE model | Qwen3-Coder-30B-A3B-Instruct |

## Components

- **InferNex-Checker**: InferNex pre-deployment validation tool, checking hardware, K8s clusters, and configuration environments before install to identify deployment risks in advance.
- **InferNex-Bridge**: Adaptation layer for integrating InferNex with KServe, supporting dual CRD declarative deployment of InferNex via `LLMInferenceService` / `InferNexService`. See [Component README](component/InferNex-Bridge/README.md).

## Roadmap
 - [26-06] Hermes-router intelligent routing supports perception and scheduling based on instance resource saturation, and secondary scheduling among instances under PD disaggregated architecture.
 - [26-06] InferNex-Deployer improves the continuous integration pipeline, adding production-grade validation for large model PD disaggregated and multi-instance deployment.
 - [26-06] Elastic-scaler focuses on workload distribution acceleration (weights, images, process startup), supporting high-performance elastic scenarios, while complementing event signal-driven scaling and compute-aware strategies.
 - [26-X] Eagle-Eye near-real-time observability will extend dynamic network resource metrics, broaden hardware health and sub-health perception, adapt to A5 generation specifications, and advance error code standardization.
 - [26-X] KVCacheX covers Cache-indexer / conductor and Lingqu enablement-related directions; plans to incorporate iterative enhancements such as DSA and Hybrid Attention KV offloading capabilities.
 - [26-X] Plan KServe integration adaptation for unified management of different types of inference Serving (predictive, LLM, etc.) and compute stack traffic (InferNex, llm-d, etc.); collaborate with the vLLM-ascend community to publish recommended deployment cases based on InferNex.
