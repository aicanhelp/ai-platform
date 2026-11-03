<div align="center">
  <h1 align="center">
      cache-indexer
  </h1>

   [![License](https://img.shields.io/badge/License-Mulan_PSL_v2-blue.svg)](./LICENSE)
   [![Go Version](https://img.shields.io/badge/Go-1.24-blue.svg)](https://golang.org/)
   [![oFEP](https://img.shields.io/badge/oFEP-0053-orange.svg)](https://gitcode.com/openFuyao/ofep/blob/main/ofeps/sig-ai-inference/0053-ofep-cache-indexer重构增强.md)
</div>
<br/>

cache-indexer is a global KVCache indexing component for large language model (LLM) inference. It maintains two tiers of KVCache indexes in a unified way — L1 (the local HBM of an inference instance) and L3 (the memory/SSD cache of an inference instance) — and answers how well an inference request hits the KVCache of candidate inference instances, so that the routing plane can schedule requests more accurately on a global, request-level basis.

## Latest Updates 🔥

- [2026-09] Added Decode-node KVCache awareness: decode Pods are now included in the L1 index discovery path; L1 ingest is compatible with the map-encoded KV events of vLLM 0.24.0 and later versions, avoiding missing cache indexes.
- [2026-06] Rebuilt in Go for Kubernetes scenarios: added two-tier L1/L3 KVCache indexes, dynamic service discovery and unified hit-rate query. L1 now supports vLLM and L3 now supports Mooncake.
- [2025-09] First release in Python/FastAPI, supporting L1 KVCache hit-rate query (available in the `release-0.21.x` branch).

## Key Features

- **L1 view**: Subscribes to the ZMQ KV-events of vLLM and maintains a real-time view of the KVCache blocks held in the local HBM of each inference instance, which is used to compute the L1 prefix hit rate; both prefill and decode instances are covered by the collection.
- **L3 view**: Periodically polls the Mooncake Master for block replica information and maintains a view of the KVCache held in the memory/SSD of each inference instance, which is used to compute the L3 prefix hit rate.
- **Unified KVCache block computation mechanism**: Replicates the block hash computation logic of vLLM on the Go side and keeps parameters such as `token_ids`, `block_size`, `cache_salt` and `PYTHONHASHSEED` consistent, so that the results computed on the request side align with vLLM events and Mooncake keys.
- **Dynamic service discovery**: Periodically discovers vLLM Pods (including prefill, decode and aggregate) and Mooncake Master Pods based on Kubernetes labels, automatically switches subscription and polling targets, and resolves the Mooncake `transport_endpoint` through `/get_all_segments`.

## Overall Architecture

@startuml
skinparam shadowing false
skinparam componentStyle rectangle
skinparam packageStyle rectangle
skinparam defaultTextAlignment center
title cache-indexer Overall Architecture
top to bottom direction

actor "User" as USER

package "K8s" {
    [Gateway + Hermes-router EPP Pod] as ENTRY
    [Kubernetes API Server] as K8S
    [cache-indexer Pod] as CI #c2e0c6
    [Mooncake Master Pod] as MC

    package "server (PD disaggregated)" as PKG_PD {
        [Leader Pod] as PROXY
        package "Prefill Pod" as PPod{
            [Prefill Engine] as PREFILL
            [Mooncake Client] as PREFILLMC
        }
        package "Decode Pod" as DPod{
            [Decode Engine] as DECODE
            [Mooncake Client] as DECODEMC
        }
    }
}

USER --> ENTRY : OpenAI Request
ENTRY --> CI : POST /kv-cache/hit-rate

ENTRY --> PROXY
PROXY --> PREFILL

PREFILL --> CI : ZMQ KV events
CI --> MC : GET /get_all_keys

PREFILL --> PREFILLMC
DECODE --> DECODEMC

PREFILLMC --> MC
DECODEMC --> MC

K8S --> ENTRY : Gateway / backend discovery
K8S <-- CI : List Pods(namespace)

USER -[hidden]down- K8S
K8S -[hidden]down- ENTRY
ENTRY -[hidden]down- CI
CI -[hidden]right- PKG_PD
CI -[hidden]down- MC
PKG_PD -[hidden]down- MC
PPod -[hidden]right- DPod
PREFILL -[hidden]right- DECODE
PREFILL -[hidden]down- PREFILLMC
DECODE -[hidden]down- DECODEMC
@enduml

cache-indexer sits on the routing decision path of the inference service. The main flow is as follows:

1. **Service discovery**: cache-indexer lists vLLM Pods and Mooncake Master Pods with specific labels through the Kubernetes API Server and builds a discovery snapshot.
2. **L1 collection**: starts one ZMQ SUB connection per vLLM Pod, subscribing to its `BlockStored` / `BlockRemoved` / `AllBlocksCleared` events and writing them into the L1 index.
3. **L3 collection**: periodically calls `GET /get_all_keys` + `GET /query_key` on the Mooncake Master to fetch the block replica list and writes it into the L3 index.
4. **Routing query**: the router (for example the hermes-router EPP) calls `POST /kv-cache/hit-rate`, and cache-indexer computes the longest contiguous prefix hit on each of the two indexes and returns the result.

## Version Matrix

| Component Category | Currently Verified Versions | Notes |
|----------|----------------|------|
| Inference engine (L1) | `vLLM 0.13.0`, `vLLM 0.18.0` | Provides the L1 KVCache; prefix caching and KV events must be enabled, and the same block computation parameters must be used as cache-indexer |
| Distributed cache management system (L3) | `Mooncake 0.3.7` | Provides the L3 KVCache (memory + SSD); must expose `/get_all_keys`, `/query_key` and `/get_all_segments` |
| Kubernetes | `1.28+` | Used to discover vLLM and Mooncake Master Pods; requires `get/list/watch` permissions on `pods` |

## Quick Start
### Dependency Component Configuration
For cache-indexer to run correctly, the inference engine and the distributed cache system must be configured in a specific way. For configuration notes, see [User Guide - Parameter Configuration](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/en/ai_inference_cache_indexer/user_guide/ai_inference_cache-indexer.md)

### Installation

#### Standalone Deployment

First, pull the Helm Chart package of cache-indexer:

```bash
helm pull oci://cr.openfuyao.cn/charts/cache-indexer --version 0.0.0-latest
```

Then extract the downloaded Chart package:

```bash
tar -xzvf cache-indexer-0.0.0-latest.tgz
```

Finally, run the install command:

```bash
helm -n <namespace> install <helm-name> ./cache-indexer
```

Where:

- `<namespace>`: the namespace where cache-indexer is installed, which is also the default service discovery scope.
- `<helm-name>`: the Helm release name, which mainly affects the names of resources such as Deployment, ConfigMap and ServiceAccount.
- `--create-namespace` (optional): use it when creating a new namespace.

The chart automatically injects the current namespace into `POD_NAMESPACE` through the downward API, and the default external Service name remains `cache-indexer-service`.

If no inference instance or distributed cache management system has been deployed yet, both the L1 and L3 hit rates will be `0`.

#### InferNex Integrated Deployment

[InferNex](https://gitcode.com/openFuyao/InferNex) is an AI inference service software suite provided by the openFuyao community. It has integrated cache-indexer and supports end-to-end one-click deployment. For the deployment procedure, see [User Guide - InferNex Integrated Deployment](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/en/ai_inference_cache_indexer/user_guide/ai_inference_cache-indexer.md)

### Verifying the Deployment

After deployment, first check whether the Pod, Service and health check endpoint are working:

```bash
kubectl -n <namespace> get pod -l app.kubernetes.io/name=cache-indexer
kubectl -n <namespace> get svc cache-indexer-service
kubectl -n <namespace> port-forward svc/cache-indexer-service 8080:8080
curl http://127.0.0.1:8080/healthz
```

### Sending a Request

```bash
curl -X POST http://127.0.0.1:8080/kv-cache/hit-rate \
  -H 'Content-Type: application/json' \
  -d '{
    "server_ip": ["this.is.pod.ip"],
    "body": {
      "token_ids": [9707, 0, 358, 3900, 374, 1234, 5678, 9012],
      "cache_salt": "",
      "block_size": 128
    }
  }'
```

Example response:

```json
{
  "server_score_list": [
    {
      "server_ip": "this.is.pod.ip",
      "l1_hit_ratio": 1.0,
      "l3_hit_ratio": 1.0
    }
  ],
  "message": "",
  "status": 0
}
```

## Related Documentation

For more project details, see:

- [oFEP-0053: cache-indexer Refactoring and Enhancement](https://gitcode.com/openFuyao/ofep/blob/main/ofeps/sig-ai-inference/0053-ofep-cache-indexer重构增强.md)
- [User Guide](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/en/ai_inference_cache_indexer/user_guide/ai_inference_cache-indexer.md)
- [API Reference](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/en/ai_inference_cache_indexer/api_reference/ai_inference_cache-indexer_api_reference.yaml)

## License

This project is licensed under the [Mulan PSL v2](./LICENSE) license.
