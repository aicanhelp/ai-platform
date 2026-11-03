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

Hermes-router 是一个基于 K8s GIE 框架构建的智能路由 EPP（Endpoint Picker）组件，支持 kv cache aware、prediction、bucket、decode-first 等多种高级路由策略，通过将推理请求路由到最合适的后端服务实例，优化大语言模型（LLM）推理服务的效率和性能。

## 最新进展 🔥
- [2026-09] PD 分离架构新增 Decode-first 路由策略（`decode-first`）。先选 Decode 实例，再决定是否执行 Prefill：prefix cache 命中足够高时走纯 Decode（D 路径），否则走完整 Prefill-Decode（PD 路径）。
- [2026-06] 新增基于算力饱和度与时延预测的路由策略（`aggregate-prediction`、`pd-prediction`），在 NPU 高负载场景下显著降低尾延迟。
- [2026-03] 新增生产场景容灾能力，包含自动切流、故障感知以及推理请求重试。

<details>
<summary>More</summary>

- [2025-12] 基于 K8s GIE（Gateway API Inference Extension）重构，现支持与 Istio 等开源网关集成，新增3种适用于PD分离架构的路由策略：分桶调度策略 pd-bucket、随机调度策略 pd-random 以及多因素KVCache感知策略 pd-kv-cache-aware。新架构支持开发者快速创建新的路由策略。
- [2025-09] v25.09 首版本发布，基于 Gin 实现高性能路由策略，推出2种适用于聚合架构的路由策略：多因素KVCache感知策略 kv-cache-aware、随机路由策略。

</details>

## 关键特性

- 设计遵循 K8s GIE 框架，天然支持 K8s 网关体系，可集成 Istio 等多种开源网关，支持以可插拔组件形式加入已有网关的集群，提升AI推理性能。
- 提供多种创新路由策略，支持 聚合/PD分离等多种推理后端架构。
   - kv-cache-aware(PD aggregate/disaggregate)：支持自定义得分函数的多因素KV Cache感知路由策略，在KV Cache高重复率场景（工具/智能体、多轮对话等）显著提升推理性能。
   - pd-bucket (PD disaggregate)：支持自定义参数的分桶调度策略，在长短请求、中高并发场景提升推理吞吐量。
   - prediction (PD aggregate/disaggregate)：基于算力饱和度与时延预测的多因素路由策略，实时采集算力饱和度指标并结合 TTFT/TPOT 时延预测数据评分，在 NPU 高负载场景下显著降低尾延迟。
   - decode-first (PD disaggregate)：面向 PD 分离架构，先选 Decode 实例，再按 prefix cache 命中情况决定走纯 Decode 还是完整 PD。
- 动态推理服务发现，支持在运行时新增/删除推理后端，灵活进行路由调度。
- 基于 Istio 等开源网关提供容灾能力。

## 整体架构

1. **服务发现**：Hermes-router 通过 InferencePool 资源发现匹配的后端模型服务 Pod；
2. **模型匹配**：当请求到达 Gateway 时，Gateway 通过 HTTPRoute 将请求转发到 InferencePool；
3. **路由决策**：InferencePool 调用 Hermes-router 作为 EPP，经由数据层、请求控制层、调度层三层插件协同完成路由决策（详见下方插件体系）；
4. **请求转发**：Gateway 根据 Hermes-router 的决策结果，将请求转发到选中的后端 Pod；
5. **请求重试**（可选）：Gateway 检测到请求异常，重新选取后端 Pod 发送请求。

### 插件体系

GIE v1.5.0 将插件分为三个层次，各层职责独立、可单独扩展：

- **数据层（Data Layer）**：周期性从外部系统（如 NPU exporter）采集指标，将原始数据解析为结构化 endpoint 属性，供上层插件消费；

- **请求控制层（Request Control Layer）**：在路由决策前对请求进行预处理：
  - **PreRequest 插件**：请求到达时执行前置操作（如 PD 角色 header 注入）；
  - **DataProducer 插件**：为 endpoint 生成请求级数据（分词、prefix cache 匹配、inflight 计数、预测特征提取等）；

- **调度层（Scheduling Layer）**：基于上述数据执行路由决策：
  - **Filter 插件**：过滤不符合条件的 endpoint（如 Prefill/Decode 角色、prefix cache 命中率）；
  - **Scorer 插件**：对候选 endpoint 进行多因素评分（KV Cache、算力饱和度、时延预测等）；
  - **Picker 插件**：根据评分结果选出最优 endpoint 或 endpoint 组；
  - **ProfileHandler**：Decode-first 在 Decode 评分选出实例之后，决定是否继续调度 PD profile。

## 快速开始

部署时可按接入方式选择以下两种模式：

- **standalone**：内置 Envoy sidecar，无需额外 Gateway，适合已有推理后端、只需接入路由层的场景；
- **gateway 集成模式（`charts/hermes-router`）**：通过 Gateway、HTTPRoute 和 InferencePool 接入，适合统一纳入 Istio 等网关体系。

本节演示前者，即以 **standalone** 模式部署 Hermes-router，并启用 **PD KVCache Aware** 路由策略。

### 场景说明

本示例选择 standalone 模式，是因为它内置 Envoy sidecar，便于在已有推理后端之上快速验证路由能力。如需通过 Istio 等网关统一管理 HTTPRoute 和 InferencePool，请改用 `charts/hermes-router`。

在 PD KVCache Aware 路由模式下，Hermes-router 会区分 Prefill/Decode 角色，并实时查询 KV Cache 管理器 cache-indexer，获取各 Prefill endpoint 的 prefix cache 命中情况，优先将请求路由到命中率更高的实例，从而减少重复计算，提升推理效率。

### 前置条件

部署前，请确认 vLLM 推理后端满足以下条件：

- **监听端口与 `inferencePool.targetPorts` 一致**：InferencePool 会将流量转发到 `targetPorts` 中配置的端口（示例中为 8000），确保 vLLM Pod 监听同一端口，或按实际端口修改 `inferencePool.targetPorts`；
- **Pod 打上 PD 角色标签**：PD KVCache Aware 策略依赖 `openfuyao.com/pdRole` 区分 Prefill/Decode 角色、依赖 `openfuyao.com/pdGroupID` 将 Prefill-Decode 配对成组；标签名可通过 `routing.pd.pdLabelName` / `routing.pd.pdGroupLabelName` 自定义，但 vLLM Pod 上需打上对应的标签值；
- **Pod 标签与 `inferencePool.modelServers.matchLabels` 一致**：InferencePool 通过该 selector 发现候选 Pod，selector 与 Pod 上的标签需完全匹配；
- **cache-indexer 服务已部署并可访问**：KV Cache 感知路由在请求控制阶段查询 cache-indexer 获取 prefix cache 命中信息，需在 `routing.cacheIndexer.address` 中配置正确的服务地址。

### 开始部署

1. **进入仓库根目录**
   ```bash
   cd /path/to/hermes-router
   ```

2. **准备配置**

   `charts/standalone` 默认已启用 PD KVCache Aware 路由，只需按实际环境覆盖以下参数，创建 `my-values.yaml`：
   ```yaml
   inferenceExtension:
     routing:
       cacheIndexer:
         address: http://<cache-indexer-service>:8080  # 替换为实际地址

   inferencePool:
     targetPorts:
       - number: 8000
     modelServerType: vllm
     modelServers:
       matchLabels:
         openfuyao.com/model: qwen-qwen3-8b  # 与 vLLM Pod 上的标签保持一致
   ```

3. **部署 Hermes-router**
   ```bash
   helm dependency update charts/standalone

   helm install hermes-standalone charts/standalone \
     -n <namespace> --create-namespace \
     -f my-values.yaml
   ```

   如需开启容灾能力，详见[独立部署时开启容灾](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_hermes_routing/user_guide/ai_inference_hermes_routing.md#%E7%8B%AC%E7%AB%8B%E9%83%A8%E7%BD%B2%E6%97%B6%E5%BC%80%E5%90%AF%E5%AE%B9%E7%81%BE)

4. **验证部署**（可选）
   ```bash
   # 检查 InferencePool 状态
   kubectl get inferencepool -n <namespace>

   # 检查 Hermes-router Pod 状态
   kubectl get pods -n <namespace> -l inferencepool=hermes-standalone-epp

   # 检查 vLLM Pod 标签
   kubectl get pods -n <namespace> -l openfuyao.com/pdRole
   ```

5. **发送测试请求**（可选）
   ```bash
   # 获取 EPP Pod IP（Envoy sidecar 监听 8081 端口）
   EPP_POD_IP=$(kubectl get pod -n <namespace> \
     -l inferencepool=hermes-standalone-epp \
     -o jsonpath='{.items[0].status.podIP}')

   # 发送推理请求
   curl -X POST http://${EPP_POD_IP}:8081/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen3-8B",
       "prompt": "Hello, how are you?",
       "max_tokens": 100
     }'
   ```

## 安全说明

### Istio 网关集成模式下的 TLS 证书校验

使用 `charts/hermes-router` 并以 `provider.name: istio` 部署时，Chart 会为 EPP 服务创建一条 Istio `DestinationRule`，其中 TLS 策略默认为：

```yaml
trafficPolicy:
  tls:
    mode: SIMPLE
    insecureSkipVerify: true
```

该默认值会**跳过对 EPP 服务端证书的校验**，主要目的是兼容集群内使用自签名证书或未完整配置 PKI 的场景，降低初次接入时的配置门槛。

**安全风险**：在跳过校验的情况下，Gateway 到 EPP 之间的 TLS 连接无法确认对端身份，存在中间人攻击（MITM）、服务冒充等风险。因此该默认配置**不适合直接用于对安全合规有要求的生产环境**。

**生产环境建议**：

- 为 EPP 服务配置由可信 CA（或集群内部 CA）签发的有效证书，并确保 Gateway 侧信任该 CA；
- 关闭证书跳过校验（`insecureSkipVerify: false`），启用完整的 TLS 身份验证；
- 在变更前评估集群内 Istio、mTLS 及证书分发（如 cert-manager）的现有配置，避免与全局安全策略冲突。

### 相关文档

更多项目细节请见：
- [Hermes-router 开发指南](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_hermes_routing/developer_guide/ai_inference_hermes_routing_plugin_in_development_guide.md)
- [Hermes-router 用户手册](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_hermes_routing/user_guide/ai_inference_hermes_routing.md)
