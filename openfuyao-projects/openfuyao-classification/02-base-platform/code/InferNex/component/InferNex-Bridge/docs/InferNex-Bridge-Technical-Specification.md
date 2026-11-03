# InferNex-Bridge 技术规格说明文档

## 0 文档性质

本文件为 InferNex-Bridge 的技术规格与责任边界契约，说明当前主线（master / `0.0.0-latest`）与 KServe 的兼容范围、昇腾全栈推理能力及 KServe 链路上的 Mutating Webhook 适配行为。设计依据见提案 [OFEP-0040-InferNex 接入 KServe 适配层](https://gitcode.com/openFuyao/ofep/blob/main/ofeps/sig-ai-inference/0040-ofep-Infernex%E6%8E%A5%E5%85%A5kserve%E9%80%82%E9%85%8D%E5%B1%82.md)。集群安装、`InferNexService` 入口配置及日常运维见 [InferNex Bridge 用户手册](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex_bridge.md)。

---

## 1 部署规格

本规格按 **InferNex-Bridge 版本线** 给出与 KServe、InferNex 的支持矩阵。当前文档对应 **InferNex-Bridge master / `0.0.0-latest`**（能力对齐 26.9.0 起）。

| InferNex-Bridge | KServe LLMISVC 控制器 | InferNex |
| --- | --- | --- |
| 26.6.0 | 0.17.0、0.18.0、0.19.0 | 26.6.0 |
| 26.9.0 / master（`0.0.0-latest`） | 0.18.0、0.19.0 | 26.9.0 / master |

**当前主线（master / `0.0.0-latest`，自 26.9.0 起）不再支持 KServe 0.17.0 与 InferNex 26.6.0。**

原因：KServe 0.17.0 的 `kserve-config-llm-scheduler` 与 InferNex 26.6.0 时期 Hermes Router 存在参数冲突，当时依赖 Bridge Mutating Webhook 改写该 Config 做适配。自 26.9.0 起，Hermes Router 尽量做到无需 Webhook 再改 KServe 模板即可适配；而 0.17.0 该 Config 中仍有写死参数，失去 Webhook patch 后无法继续兼容。故当前主线仅支持 KServe 0.18.0–0.19.0，并与 InferNex master / 26.9.0 起能力对齐。若仍需 KServe 0.17.0 / InferNex 26.6.0，请使用 InferNex-Bridge 26.6.0。

KServe 侧安装及集群环境前提见 [LLMInferenceService Prerequisites](https://kserve.github.io/website/docs/0.18/install/llmisvc-install#prerequisites)（0.19 与 0.18 一致，替换文档版本号即可）。

InferNex 整体部署规格见 [InferNex 规格 #42](https://gitcode.com/openFuyao/InferNex/issues/42)。默认组件镜像见附录 A。

---

## 2 部署场景

InferNex-Bridge 面向昇腾（Ascend）NPU 全栈推理场景，支持在 `LLMInferenceService`（`infernex.io/runtime=true`）与 `InferNexService`（无 `spec.sourceRef`）双入口下部署 InferNex。不限定具体的 TP/DP 组合、节点分布或编排方式，覆盖聚合（Aggregate）与 PD 分离（Disaggregated）等常见部署形态，用户可按业务需要配置并行规模与拓扑。

对应示例位于 [config/examples/llmisvc](https://gitcode.com/openFuyao/InferNex/tree/master/component/InferNex-Bridge/config/examples/llmisvc) 与 [config/examples/insvc](https://gitcode.com/openFuyao/InferNex/tree/master/component/InferNex-Bridge/config/examples/insvc)。

---

## 3 Mutating Admission Webhook 补丁说明

InferNex-Bridge 在 KServe 链路上注册 Mutating Admission Webhook（类型名：`LLMInferenceService` Mutating）。用户创建或更新带 InferNex 运行时标签的 `LLMInferenceService` 时触发，不修改 `LLMInferenceService` 对象本身，而是对集群中 KServe 预设的 `LLMInferenceServiceConfig` 模板做一次性适配补丁，使推理后端与路由符合 InferNex 运行要求。

### 3.1 触发条件与修改对象

Mutating Admission Webhook 在创建或更新带有 `infernex.io/runtime=true` 标签的 `LLMInferenceService`（v1alpha1/v1alpha2）时起作用。KServe 预设 `LLMInferenceServiceConfig` 须安装在 `kserve` 命名空间（见第 1 节 KServe 安装前提），否则 Webhook 无法完成修改。

实际修改的是已安装在集群中 KServe `LLMInferenceServiceConfig` 预设模板，依次在 `kserve` 命名空间中查找下列 6 个 Config，并进行补丁修改：

- `kserve-config-llm-template`
- `kserve-config-llm-worker-data-parallel`
- `kserve-config-llm-decode-template`
- `kserve-config-llm-decode-worker-data-parallel`
- `kserve-config-llm-prefill-template`
- `kserve-config-llm-prefill-worker-data-parallel`

每个 Config 首次修改成功后写入注解 `infernex.io/llm-inference-service-config-mutated=infernex`；已带该注解的 Config 不会再次执行修改。升级 Bridge 后若需重新补丁（例如在调整 initContainer 清理逻辑之前已修改过的集群），须删除该注解。

### 3.2 修改内容

补丁对象仍为 3.1 节所列 6 个 `LLMInferenceServiceConfig`。以下摘录自 KServe 预设 `kserve/config/llmisvcconfig/config-llm-decode-template.yaml`（[上游仓库](https://github.com/kserve/kserve/tree/master/config/llmisvcconfig)），展示补丁前后关键字段差异（省略无关字段）。

清空 KServe 预设里的 llm-d initContainer。`capabilities.drop: ALL` 予以保留（不再剥离），使推理 workload 兼容 restricted Pod Security Admission；NPU 设备访问改由 pod 级 `supplementalGroups` 注入（单独实现），不再依赖剥 drop ALL 来获取 DAC_OVERRIDE。

`securityContext.capabilities.drop`：workload 容器的 securityContext 补丁前后保持不变，`drop: [ALL]` 等字段原样保留（`spec.template`、`spec.prefill.template`、`spec.worker`、`spec.prefill.worker` 路径上的容器均不再被改动 securityContext）。

`initContainers`

补丁前（decode 模板 `spec.template.initContainers`）：

```yaml
initContainers:
  - name: llm-d-routing-sidecar
    image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.7.1
    command:
      - /app/pd-sidecar
      - --port=8000
      - --vllm-port=8001
    # ... llm-d PD 路由 sidecar 其余配置
```

补丁后：

```yaml
initContainers: []
```

### 3.3 影响

#### 3.3.1 安全问题注意

KServe 预设 workload 容器通常带有 `capabilities.drop: [ALL]`，有利于在 restricted Pod Security Admission 下运行。Ascend / Hermes 在 NPU 环境下仍需访问设备节点，Bridge **保留** workload 上的 `drop: ALL`，并通过 Pod 级 **`supplementalGroups`** 注入（单独实现）授予设备访问能力，**不再**通过剥离 `drop: ALL` 获取 `DAC_OVERRIDE`。若集群有等保、强隔离等合规要求，可在 `LLMInferenceService` 或 `LLMInferenceServiceConfig` 里自行调整安全上下文。

> 后续规划：openFuyao 社区将持续评估在 restricted PSA 下更细粒度的能力白名单方案，在兼容性与安全性之间取得更好平衡。

#### 3.3.2 sidecar 清理说明

KServe 默认会在 decode 等 Workload Pod 里加 llm-d 路由 sidecar（`initContainers`），用于 PD 分离等场景的请求转发。InferNex 不走 llm-d 链路，补丁会清空这些 initContainer。补丁后 KServe 预设不会再自动挂载 llm-d sidecar；若业务仍依赖 LLM-D 的 PD 路由等能力，须在 `LLMInferenceService` 或 `LLMInferenceServiceConfig` 里自行写回 initContainer，并按 3.1 节重新触发补丁。

> 后续规划：已向 KServe 提交 Issue 跟踪该兼容性诉求，后续版本将尽可能避免由 Webhook 修改该模板，改为通过其他机制实现平滑兼容。

---

## 附录 A：默认镜像

InferNex master / `0.0.0-latest` 线随 Bridge Chart 交付的默认组件镜像如下。


| 组件                                     | 版本/标签   | 镜像                                                                 |
| -------------------------------------- | ------- | ------------------------------------------------------------------ |
| vLLM-Ascend                            | v0.23.0 | `quay.io/ascend/vllm-ascend`                             |
| mooncake-master                        | v0.23.0 | `quay.io/ascend/vllm-ascend`                             |
| mooncake-metadata                      | 8.6.1   | `hub.oepkgs.net/openfuyao/redis`                                   |
| hermes-router                          | latest  | `cr.openfuyao.cn/openfuyao/hermes-router`                          |
| cache-indexer                          | latest  | `cr.openfuyao.cn/openfuyao/cache-indexer`                          |
| proxy-server                           | latest  | `cr.openfuyao.cn/openfuyao/proxy-server`                           |
| elastic-scaler                         | latest  | `cr.openfuyao.cn/openfuyao/elastic-scaler`                         |
| tidal                                  | latest  | `cr.openfuyao.cn/openfuyao/tidal`                                  |
| resource-scaling-group                 | latest  | `cr.openfuyao.cn/openfuyao/resource-scaling-group`                 |
| eagle-eye-hardware-monitor             | latest  | `cr.openfuyao.cn/openfuyao/eagle-eye-hardware-monitor`             |
| eagle-eye-hardware-diagnosis           | latest  | `cr.openfuyao.cn/openfuyao/eagle-eye-hardware-diagnosis`           |
| eagle-eye-network-performance-exporter | latest  | `cr.openfuyao.cn/openfuyao/eagle-eye-network-performance-exporter` |


