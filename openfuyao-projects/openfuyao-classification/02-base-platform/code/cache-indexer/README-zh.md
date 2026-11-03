<div align="center">
  <h1 align="center">
      cache-indexer
  </h1>

   [![License](https://img.shields.io/badge/License-Mulan_PSL_v2-blue.svg)](./LICENSE)
   [![Go Version](https://img.shields.io/badge/Go-1.24-blue.svg)](https://golang.org/)
   [![oFEP](https://img.shields.io/badge/oFEP-0053-orange.svg)](https://gitcode.com/openFuyao/ofep/blob/main/ofeps/sig-ai-inference/0053-ofep-cache-indexer重构增强.md)
</div>
<br/>

cache-indexer 是面向大语言模型（LLM）推理场景的全局 KVCache 索引组件，统一维护两层 KVCache 索引——L1（推理实例本地HBM）与 L3（推理实例内存/SSD缓存），支持查询推理请求对候选推理实例的 KVCache 命中情况，支持路由进行更准确的全局请求级调度。

## 最新进展 🔥

- [2026-09] 新增 Decode 节点 KV Cache 感知能力，将 decode Pod 纳入 L1 索引发现链路；L1 ingest 兼容 vLLM 0.24.0 及后续版本的 map 编码 KV 事件，避免缓存索引缺失。
- [2026-06] 基于 Go 面向 K8s 场景重构，新增 L1/L3 双层 KVCache 索引、动态服务发现与统一命中率查询，L1 现已支持 vLLM、L3现已支持 Mooncake。
- [2025-09] Python/FastAPI 首版本发布，支持 L1 KVCache 命中率查询（位于 `release-0.21.x` 分支）。

## 关键特性

- **L1 视图**：订阅 vLLM 的 ZMQ KV-event，实时维护推理实例本地 HBM 中的 KVCache block 视图，用于计算 L1 前缀命中率；prefill 与 decode 实例均纳入采集范围。
- **L3 视图**：周期轮询 Mooncake Master 的 block 副本信息，维护推理实例内存/SSD 中的 KVCache 视图，用于计算 L3 前缀命中率。
- **统一 KVCache block 计算机制**：在 Go 侧复刻与 vLLM 一致的 block hash 计算逻辑，通过 `token_ids`、`block_size`、`cache_salt` 和 `PYTHONHASHSEED` 等参数保持一致，确保请求侧计算结果与 vLLM 事件和 Mooncake key 对齐。
- **动态服务发现**：基于 K8s label 周期发现 vLLM Pod（含 prefill、decode、aggregate）与 Mooncake Master Pod，自动切换订阅目标、轮询目标，并通过 `/get_all_segments` 反查 Mooncake `transport_endpoint`。

## 整体架构

@startuml
skinparam shadowing false
skinparam componentStyle rectangle
skinparam packageStyle rectangle
skinparam defaultTextAlignment center
title cache-indexer 总体架构
top to bottom direction

actor "User" as USER

package "K8s" {
    [Gateway + Hermes-router EPP Pod] as ENTRY
    [Kubernetes API Server] as K8S
    [cache-indexer Pod] as CI #c2e0c6
    [Mooncake Master Pod] as MC

    package "server（PD disaggregated）" as PKG_PD {
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

K8S --> ENTRY : 网关/后端发现
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

cache-indexer 在推理服务中处于路由决策路径上，主链路如下：

1. **服务发现**：cache-indexer 通过 K8s API Server 列举带特定 label 的 vLLM Pod 与 Mooncake Master Pod，构造发现快照。
2. **L1 采集**：对每个 vLLM Pod 启动一路 ZMQ SUB，订阅其 `BlockStored` / `BlockRemoved` / `AllBlocksCleared` 事件，写入 L1 索引。
3. **L3 采集**：对 Mooncake Master 周期调用 `GET /get_all_keys` + `GET /query_key` 拉取 block 副本列表，写入 L3 索引。
4. **路由查询**：路由器（如 hermes-router EPP）调用 `POST /kv-cache/hit-rate`，cache-indexer 在两张索引上各算一次最长连续前缀命中并返回。

## 版本规格

| 组件类别 | 当前已验证版本 | 说明 |
|----------|----------------|------|
| 推理引擎（L1） | `vLLM 0.13.0`、`vLLM 0.18.0` | 提供 L1 KVCache；需开启 prefix caching 与 KV events，并与 cache-indexer 使用一致的 block 计算参数 |
| 分布式缓存管理系统（L3） | `Mooncake 0.3.7` | 提供 L3 KVCache（内存 + SSD）；需暴露 `/get_all_keys`、`/query_key`、`/get_all_segments` |
| Kubernetes | `1.28+` | 用于服务发现 vLLM 与 Mooncake Master Pod；需具备 `pods` 的 `get/list/watch` 权限 |

## 快速开始
### 依赖组件配置
cache-indexer的正确运行要求推理引擎与分布式缓存系统进行特定配置，配置注意事项见[用户指南-参数配置说明](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_cache_indexer/user_guide/ai_inference_cache-indexer.md#%E5%8F%82%E6%95%B0%E9%85%8D%E7%BD%AE%E8%AF%B4%E6%98%8E)

### 安装部署

#### 独立部署

先拉取 cache-indexer 的 Helm Chart 包：

```bash
helm pull oci://cr.openfuyao.cn/charts/cache-indexer --version 0.0.0-latest
```

再解压下载后的 Chart 包：

```bash
tar -xzvf cache-indexer-0.0.0-latest.tgz
```

最后执行安装命令：

```bash
helm -n <namespace> install <helm-name> ./cache-indexer
```

其中：

- `<namespace>`：cache-indexer 的安装 namespace，也是默认服务发现范围。
- `<helm-name>`：Helm release 名称，主要影响 Deployment、ConfigMap、ServiceAccount 等资源名。
- `--create-namespace`（可选）：新创建命名空间时使用。

chart 会通过 downward API 自动把当前 namespace 注入 `POD_NAMESPACE`，默认对外 Service 名仍为 `cache-indexer-service`。

如果此时尚未部署推理实例和分布式缓存管理系统，L1 与 L3 命中率都会是 `0`。

#### InferNex 集成部署

[InferNex](https://gitcode.com/openFuyao/InferNex) 是 openFuyao 社区提供的 AI 推理服务软件套件，现已集成 cache-indexer，可端到端一键部署，部署方法详见[用户指南-InferNex集成部署](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_cache_indexer/user_guide/ai_inference_cache-indexer.md#infernex%E9%9B%86%E6%88%90%E9%83%A8%E7%BD%B2)

### 验证部署

部署完成后，可先确认 Pod、Service 和健康检查接口是否正常：

```bash
kubectl -n <namespace> get pod -l app.kubernetes.io/name=cache-indexer
kubectl -n <namespace> get svc cache-indexer-service
kubectl -n <namespace> port-forward svc/cache-indexer-service 8080:8080
curl http://127.0.0.1:8080/healthz
```

### 发送请求

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

返回示例：

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

## 相关文档

更多项目细节见：

- [oFEP-0053：cache-indexer 重构增强](https://gitcode.com/openFuyao/ofep/blob/main/ofeps/sig-ai-inference/0053-ofep-cache-indexer重构增强.md)
- [用户手册](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_cache_indexer/user_guide/ai_inference_cache-indexer.md)
- [API参考](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_cache_indexer/api_reference/ai_inference_cache-indexer_api_reference.yaml)

## 许可

本项目采用 [Mulan PSL v2](./LICENSE) 许可。