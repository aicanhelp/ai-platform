<div align="center">
  <h1 align="center">
      Hermes-router
  </h1>

  [![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_hermes_routing/user_guide/ai_inference_hermes_routing.md)
   [![License](https://img.shields.io/badge/License-Mulan_PSL_v2-blue.svg)](./LICENSE)
   [![Go Version](https://img.shields.io/badge/Go-1.25.0-blue.svg)](https://golang.org/)
   [![GIE](https://img.shields.io/badge/Gateway_API_Inference_Extension-v1.5.0-orange.svg)](https://github.com/kubernetes-sigs/gateway-api-inference-extension)
</div>
<br/>

Hermes-router is an intelligent routing EPP (Endpoint Picker) component built on the K8s GIE framework, supporting multiple advanced routing strategies such as KV cache aware, prediction, bucket, and decode-first. By routing inference requests to the most suitable backend service instances, it optimizes the efficiency and performance of large language model (LLM) inference services.

## Latest Updates 🔥
- [2026-09] Added the Decode-first routing strategy (`decode-first`) for PD disaggregated deployments. It selects a Decode instance first, then decides whether to run Prefill: a high enough prefix-cache hit takes the Decode-only (D) path; otherwise the request takes the full Prefill-Decode (PD) path.
- [2026-06] Added routing strategies based on compute saturation and latency prediction (`aggregate-prediction`, `pd-prediction`), significantly reducing tail latency under high NPU load scenarios.
- [2026-03] Added disaster recovery capabilities for production scenarios, including automatic traffic switching, fault awareness, and inference request retry.

<details>
<summary>More</summary>

- [2025-12] Refactored based on K8s GIE (Gateway API Inference Extension), now supporting integration with open-source gateways such as Istio. Added 3 routing strategies for PD disaggregated architecture: bucket scheduling strategy pd-bucket, random scheduling strategy pd-random, and multi-factor KVCache aware strategy pd-kv-cache-aware. The new architecture enables developers to quickly create new routing strategies.
- [2025-09] v25.09 initial version released, implementing high-performance routing strategies based on Gin, introducing 2 routing strategies for aggregate architecture: multi-factor KVCache aware strategy kv-cache-aware and random routing strategy.

</details>

## Key Features

- Designed following the K8s GIE framework, natively supporting the K8s gateway system. It can integrate with various open-source gateways such as Istio, and can be added to existing gateway clusters as a pluggable component to enhance AI inference performance.
- Provides multiple innovative routing strategies, supporting various inference backend architectures such as aggregate/PD disaggregated.
   - kv-cache-aware (PD aggregate/disaggregate): A multi-factor KV Cache aware routing strategy supporting custom scoring functions, significantly improving inference performance in high KV Cache reuse scenarios (tool/agent, multi-turn conversations, etc.).
   - pd-bucket (PD disaggregate): A bucket scheduling strategy supporting custom parameters, improving inference throughput in long/short request and medium-to-high concurrency scenarios.
   - prediction (PD aggregate/disaggregate): A multi-factor routing strategy based on compute saturation and latency prediction, collecting real-time compute saturation metrics and combining TTFT/TPOT latency prediction data for scoring, significantly reducing tail latency under high NPU load scenarios.
   - decode-first (PD disaggregate): For PD disaggregated deployments, selects a Decode instance first, then chooses the Decode-only path or the full PD path based on prefix cache hits.
- Dynamic inference service discovery, supporting adding/removing inference backends at runtime for flexible routing scheduling.
- Provides disaster recovery capabilities based on open-source gateways such as Istio.

## Overall Architecture

1. **Service Discovery**: Hermes-router discovers matching backend model service Pods through InferencePool resources;
2. **Model Matching**: When a request reaches the Gateway, the Gateway forwards the request to InferencePool via HTTPRoute;
3. **Routing Decision**: InferencePool calls Hermes-router as the EPP, completing routing decisions through the coordinated effort of three plugin layers — Data Layer, Request Control Layer, and Scheduling Layer (see the plugin system below);
4. **Request Forwarding**: The Gateway forwards the request to the selected backend Pod based on Hermes-router's decision;
5. **Request Retry** (optional): The Gateway detects an abnormal request and reselects a backend Pod to send the request.

### Plugin System

GIE v1.5.0 divides plugins into three layers, each with independent responsibilities and individual extensibility:

- **Data Layer**: Periodically collects metrics from external systems (e.g., NPU exporter), parsing raw data into structured endpoint attributes for upper-layer plugins to consume;

- **Request Control Layer**: Preprocesses requests before routing decisions:
  - **PreRequest Plugin**: Executes pre-processing operations when a request arrives (e.g., PD role header injection);
  - **DataProducer Plugin**: Generates request-level data for endpoints (tokenization, prefix cache matching, inflight counting, prediction feature extraction, etc.);

- **Scheduling Layer**: Executes routing decisions based on the above data:
  - **Filter Plugin**: Filters endpoints that do not meet conditions (e.g., Prefill/Decode role, prefix cache hit rate);
  - **Scorer Plugin**: Performs multi-factor scoring on candidate endpoints (KV Cache, compute saturation, latency prediction, etc.);
  - **Picker Plugin**: Selects the optimal endpoint or endpoint group based on scoring results;
  - **ProfileHandler**: After Decode scoring selects an instance, Decode-first decides whether to continue scheduling the PD profile.

## Quick Start

During deployment, choose one of the following two modes based on your access method:

- **standalone**: Built-in Envoy sidecar, no additional Gateway required, suitable for scenarios with existing inference backends that only need routing layer integration;
- **gateway integration mode (`charts/hermes-router`)**: Integrated via Gateway, HTTPRoute, and InferencePool, suitable for unified inclusion into gateway systems such as Istio.

This section demonstrates the former, i.e., deploying Hermes-router in **standalone** mode and enabling the **PD KVCache Aware** routing strategy.

### Scenario Description

This example uses standalone mode because it includes a built-in Envoy sidecar, making it easy to quickly validate routing capabilities on top of existing inference backends. If you need to uniformly manage HTTPRoute and InferencePool through gateways such as Istio, use `charts/hermes-router` instead.

In PD KVCache Aware routing mode, Hermes-router distinguishes Prefill/Decode roles and queries the KV Cache manager cache-indexer in real time to obtain prefix cache hit information for each Prefill endpoint, prioritizing routing requests to instances with higher hit rates, thereby reducing redundant computation and improving inference efficiency.

### Prerequisites

Before deployment, confirm that the vLLM inference backend meets the following conditions:

- **Listening port matches `inferencePool.targetPorts`**: InferencePool forwards traffic to the port configured in `targetPorts` (8000 in the example). Ensure the vLLM Pod listens on the same port, or modify `inferencePool.targetPorts` according to the actual port;
- **Pod labeled with PD role**: PD KVCache Aware strategy depends on `openfuyao.com/pdRole` to distinguish Prefill/Decode roles, and on `openfuyao.com/pdGroupID` to pair Prefill-Decode into groups. Label names can be customized via `routing.pd.pdLabelName` / `routing.pd.pdGroupLabelName`, but vLLM Pods must have corresponding label values;
- **Pod labels match `inferencePool.modelServers.matchLabels`**: InferencePool discovers candidate Pods through this selector. The selector and Pod labels must match exactly;
- **cache-indexer service deployed and accessible**: KV Cache aware routing queries cache-indexer during the request control phase to obtain prefix cache hit information. The correct service address must be configured in `routing.cacheIndexer.address`.

### Start Deployment

1. **Navigate to the Repository Root Directory**
   ```bash
   cd /path/to/hermes-router
   ```

2. **Prepare Configuration**

   `charts/standalone` has PD KVCache Aware routing enabled by default. Simply override the following parameters for your actual environment and create `my-values.yaml`:
   ```yaml
   inferenceExtension:
     routing:
       cacheIndexer:
         address: http://<cache-indexer-service>:8080  # Replace with actual address

   inferencePool:
     targetPorts:
       - number: 8000
     modelServerType: vllm
     modelServers:
       matchLabels:
         openfuyao.com/model: qwen-qwen3-8b  # Keep consistent with labels on vLLM Pods
   ```

3. **Deploy Hermes-router**
   ```bash
   helm dependency update charts/standalone

   helm install hermes-standalone charts/standalone \
     -n <namespace> --create-namespace \
     -f my-values.yaml
   ```

   To enable disaster recovery capabilities, see [Enabling Disaster Recovery in Standalone Deployment](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_hermes_routing/user_guide/ai_inference_hermes_routing.md#%E7%8B%AC%E7%8B%81%E9%83%A8%E7%BD%AE%E6%97%B6%E5%BC%80%E5%90%AF%E5%AE%B9%E7%81%BE)

4. **Verify Deployment** (optional)
   ```bash
   # Check InferencePool status
   kubectl get inferencepool -n <namespace>

   # Check Hermes-router Pod status
   kubectl get pods -n <namespace> -l inferencepool=hermes-standalone-epp

   # Check vLLM Pod labels
   kubectl get pods -n <namespace> -l openfuyao.com/pdRole
   ```

5. **Send a Test Request** (optional)
   ```bash
   # Get EPP Pod IP (Envoy sidecar listens on port 8081)
   EPP_POD_IP=$(kubectl get pod -n <namespace> \
     -l inferencepool=hermes-standalone-epp \
     -o jsonpath='{.items[0].status.podIP}')

   # Send inference request
   curl -X POST http://${EPP_POD_IP}:8081/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen3-8B",
       "prompt": "Hello, how are you?",
       "max_tokens": 100
     }'
   ```

## Security Notes

### TLS Certificate Verification in Istio Gateway Integration Mode

When deploying with `charts/hermes-router` using `provider.name: istio`, the Chart creates an Istio `DestinationRule` for the EPP service with the following default TLS policy:

```yaml
trafficPolicy:
  tls:
    mode: SIMPLE
    insecureSkipVerify: true
```

This default value **skips verification of the EPP server certificate**, primarily to accommodate scenarios using self-signed certificates or incomplete PKI configuration within the cluster, lowering the configuration barrier for initial integration.

**Security Risk**: When verification is skipped, the TLS connection between Gateway and EPP cannot confirm the peer's identity, posing risks such as man-in-the-middle (MITM) attacks and service impersonation. Therefore, this default configuration **is not suitable for production environments with security compliance requirements**.

**Production Environment Recommendations**:

- Configure valid certificates issued by a trusted CA (or internal cluster CA) for the EPP service, and ensure the Gateway side trusts this CA;
- Disable certificate skip verification (`insecureSkipVerify: false`) and enable full TLS identity verification;
- Before making changes, evaluate the existing configuration of Istio, mTLS, and certificate distribution (e.g., cert-manager) within the cluster to avoid conflicts with global security policies.

### Related Documentation

For more project details, see:
- [Hermes-router Developer Guide](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_hermes_routing/developer_guide/ai_inference_hermes_routing_plugin_in_development_guide.md)
- [Hermes-router User Guide](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_hermes_routing/user_guide/ai_inference_hermes_routing.md)
