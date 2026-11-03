
<div align="center">
  <h1 align="center">
      InferNex
  </h1>

  <p><b>提供openFuyao AI推理服务化框架的端到端一键式集成部署</b></p>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex.md)
[![License](https://img.shields.io/badge/License-Mulan_PSL_v2-blue.svg)](./LICENSE)
[![Helm](https://img.shields.io/badge/Helm-chart-00a1d6.svg)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex.md#%E5%BC%80%E5%A7%8B%E5%AE%89%E8%A3%85)
[![GIE](https://img.shields.io/badge/Gateway_API_Inference_Extension-v1.5.0--rc.2-orange.svg)](https://github.com/kubernetes-sigs/gateway-api-inference-extension)

</div>

<hr>

## Updates
 - [26-09] InferNex 默认推理引擎升级至 vLLM-Ascend v0.23.0，补充 DeepSeek-V4-Flash / GLM-5.2 等参考配置；InferNex-Bridge 的 Mutating Webhook 不再对 `kserve-config-llm-scheduler` LLMInferenceServiceConfig 做 patch；Hermes-router 落地 Decode-first D/PD 分离调度并完善 tokenizer 与 KServe 路径对齐，cache-indexer 支持 Decode 侧 L1 KVCache 发现，proxy-server 支持 D/PD Header 编排；eagle-eye 新增端到端分布式链路追踪，infernex-checker 增强可选连通性检查与慢卡分析能力。
 - [26-06] 推理后端切换为 LeaderWorkerSet（LWS）部署编排，原生支持多 DP 协同；PD-Orchestrator 的 elastic-scaler 新增 APA 扩缩算法，支持多样指标扩缩；Hermes-router 新增基于算力饱和度与时延预测的路由策略；cache-indexer 实现 L3 级 KV-aware 感知，与 Mooncake 联动支撑全局 KVCache 索引；eagle-eye 新增权重分发及灵衢网络动态指标获取；InferNex 新增 Helm 部署前置校验工具，覆盖 NPU 驱动、硬件资源及网络通信等环境检查，提前发现部署风险。
 - [26-05] 新增 InferNex-Bridge 组件，兼容 KServe 接入 InferNex 推理套件，支持 LLMInferenceService 与 InferNexService 双 CRD 声明式部署，适配层自动完成编排与路由打通。
 - [26-03] 新增PD-Orchestrator组件，支持动态PD组扩缩容；智能路由新增容灾能力，包含自动切流、故障感知以及请求重试；推理后端组件重构更新，支持配置不同版本vLLM推理引擎、非huggingface模型。

<details>
<summary>More</summary>

 - [25-12] 新增推理可观测子组件；智能路由基于GIE基础框架实现网关插件。
 - [25-09] 发布AI推理集成部署alpha版本！支持KVCache aware等策略的智能路由Hermes-router、xPyD分离模式推理引擎、全局KVCache元数据管理、Mooncake分布式KVCache管理体系集成等特性。

</details>

## Overview
本项目基于主流LLM推理技术栈及K8s官方项目GIE（Gateway API Inference Extension）构建，集成了以下K8s原生的高性能、可扩展子特性，旨在提升推理吞吐量并降低时延，为AI服务部署提供高效、可靠的技术支撑。

**智能路由系统（Hermes-router）**：基于GIE基础框架实现的网关插件，具备动态请求分发与负载均衡能力；支持多样化算力负载感知、KV命中感知、请求压力感知、请求长度感知、语义感知等多维度感知能力，用户可利用内置的策略扩展（KVCache aware策略、PD长短请求分桶策略）实现推理请求的最佳节点路由。

**xPyD分离推理引擎**：基于 vLLM 高性能推理引擎构建的 AI 推理后端，支持 xPyD 架构、推理节点自动发现（Proxy Server）、Mooncake KVCache 存储及多实例灵活部署；推理引擎由 LeaderWorkerSet（LWS）承载部署，原生支持多 DP 协同，并可通过 `dataParallelSize` 与 `dataParallelSizeLocal` 配置 DP 负载均衡策略。

**PD-Orchestrator**：面向PD分离场景的弹性编排组件，集成潮汐算法、扩缩容决策框架与动态PD组扩缩容能力，支持指标与事件驱动的资源伸缩、成组及组内按比例或自定义策略伸缩，并开放用户自定义决策与资源管理逻辑扩展。

**分布式KVCache管理**：利用Mooncake HCCL Transfer Engine实现PD节点间KVCache的高速传输。

**全局KVCache索引（cache-indexer）**：基于vLLM的KV Event机制并提供RESTful接口，构建分布式全局KVCache元数据前缀树，赋能路由KV-aware感知能力，实现全局KVCache资源的高效利用。

**推理可观测体系（eagle-eye）**：基于Prometheus标准数据采集和上报格式，提供推理场景的关键观测指标，覆盖业务运行时指标、系统运行时指标和资源健康度指标；通过NATS异步消息队列发布订阅模式，提供关键业务运行态指标的实时（毫秒级）可观测能力，支撑关键加速模块的近实时决策能力。

## Dependent Components

本集成部署方案包含以下子组件及其版本信息：

| 组件名称 | 版本 | 可选 | 说明 |
|---------|------|------|------|
| inference-backend | latest | 必选 | 基于vllm/vllm-ascend的推理引擎后端 |
| pd-orchestrator | latest | 必选 | PD动态扩缩容组件 |
| Hermes-router | latest | 可选 | 智能路由系统 |
| cache-indexer | latest | 可选 |分布式全局KVCache元数据管理组件 |
| eagle-eye | latest | 可选 |可观测体系 |
| vLLM-Ascend | v0.23.0 | 可选 | 推理引擎框架。用户可配置其他版本vLLM-Ascend镜像。 |
| Mooncake | 0.3.11 | 可选 | 分布式KVCache管理。随vLLM-Ascend镜像版本变动。 |

> 注：部分组件为可选组件，需通过配置启用。详细配置请参考[详细配置](#configuration)章节。

## Quick Start

### Prerequisites
- Kubernetes v1.29.0及以上版本（推荐 v1.33.0及以上）。
- 已安装[npu-operator](https://gitcode.com/openFuyao/npu-operator)组件。
- 已安装[LWS](https://lws.sigs.k8s.io/docs/installation/)组件。
- 集群中需要安装metrics server v0.8.0及以上版本。

### Binary Deployment

1. 拉取项目安装包。
    ```bash
    helm pull oci://cr.openfuyao.cn/charts/infernex --version xxx
    ```
    其中`xxx`需替换为具体项目安装包版本，如`0.0.0-latest`。拉取得到的安装包为压缩包形式。

2. 解压安装包。
    ```bash
    tar -xzvf infernex-xxx.tgz
    ```
    其中`xxx`需替换为具体项目安装包版本，如`0.0.0-latest`。

3. 安装部署。

    以命名空间`ai-inference`、release名称`infernex`为例，在`infernex`同级目录下执行如下命令：
   ```bash
   helm install -n ai-inference infernex ./infernex
   ```

### Source Deployment

1. 从仓库拉取项目。

   ```bash
   git clone https://gitcode.com/openFuyao/InferNex.git
   ```

2. 安装部署。

   以命名空间`ai-inference`、release名称`infernex`为例，在`InferNex`同级目录下执行如下命令：
   ```bash
   cd InferNex/charts/infernex
   helm dependency build
   helm install -n ai-inference infernex .
   ```

### Configuration
详细配置说明请参考InferNex用户手册的[配置AI推理集成部署](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_infernex/user_guide/ai_inference_infernex.md#%E9%85%8D%E7%BD%AEai%E6%8E%A8%E7%90%86%E9%9B%86%E6%88%90%E9%83%A8%E7%BD%B2)章节。


## Performance

- **定长系统提示词复用场景**：该场景模拟系统提示词固定长度且 KVCache 可复用的负载，数据集包含 120 种长度为 8k 的系统提示词，每种重复 4 次，共 480 条请求，请求时间遵循泊松分布，并发 8。以随机路由为性能基线，开启 InferNex 优化能力后，聚合部署 TTFT 平均降幅约 54%、TPS 平均提升约 20%，性能显著提升。
- **多轮对话场景**：该场景模拟多用户与 LLM 持续对话，单个会话包含多轮请求，120 个独立用户共 480 条请求，首轮请求 16k token、每轮返回 128 token，后续每轮追加 1k token 共 4 轮，平均请求长度约 17.5k，请求时间遵循泊松分布，并发 8。以随机路由为性能基线，开启 InferNex 优化能力后，聚合部署 TTFT 平均降幅约 60%、TPS 平均提升约 44%，性能显著提升。
- **推理可观测体系**：Pod资源占用CPU <20m，MEM 3000~3500M；秒级上报功能发布方平均**采集时长<10ms**，订阅方平均日志**接收时延<1ms**。详细可见[eagle-eye性能测试报告](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/reports/performance/eagle-eye%E6%80%A7%E8%83%BD%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-v26.03.md)。

> 各优化策略的详细性能数据及对比分析，详见 [InferNex 性能测试报告](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/reports/performance/InferNex%E6%95%B4%E4%BD%93%E6%80%A7%E8%83%BD%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-v26.06.md)。


## Specification Sheet

每个 **InferNex 大版本**发布时，会同步给出该版本下各配套组件的**理论支持版本范围**及**已验证组合**说明。**核心理念**是：以 **InferNex 发行版本为锚点**，其余配套如推理引擎、智能路由、编排器、观测、芯片与 Kubernetes 等在规格表中声明其相对该锚点的支持关系；集成部署时以本表为优先参照。

> 表中「验证情况」为 **是** 表示已在对应组合上完成过有效验证；为 **否** 表示**暂未验证**（仍可能可用，但不作为已验收承诺）。**以下组件版本皆为理论支持范围**，除非备注另有说明。

### Subcomponent Specification

下列为 InferNex 发行版所依赖或集成的**软件组件**及其相对锚点版本的支持边界（以 **InferNex 26.9.0** 为基线列举）。
v26.09版本删除了vllm-ascend 0.13.0 之前的版本，不再建议使用。

| 组件                      | 版本       | 验证情况 | 备注                                                 |
| ------------------------- | ---------- | -------- | ---------------------------------------------------- |
| 推理引擎（vllm-ascend）   | v0.23.0    | 是       | 默认版本提供，v26.09版本更新                         |
| 推理引擎（vllm-ascend）   | v0.22.1rc1 | 否       |                                                      |
| 推理引擎（vllm-ascend）   | v0.21.0rc1 | 否       |                                                      |
| 推理引擎（vllm-ascend）   | v0.20.2rc1 | 否       |                                                      |
| 推理引擎（vllm-ascend）   | v0.19.1rc1 | 是       |                                                      |
| 推理引擎（vllm-ascend）   | v0.18.0    | 是       |                                                      |
| 推理引擎（vllm-ascend）   | v0.17.0rc1 | 否       |                                                      |
| 推理引擎（vllm-ascend）   | v0.16.0rc1 | 否       |                                                      |
| 推理引擎（vllm-ascend）   | v0.15.0rc1 | 否       |                                                      |
| 推理引擎（vllm-ascend）   | v0.14.0rc1 | 是       |                                                      |
| 推理引擎（vllm-ascend）   | v0.13.0    | 是       |                                                      |
| 开源网关（Istio）         | 1.29.0     | 否       |                                                      |
| 开源网关（Istio）         | 1.28.0     | 是       |                                                      |
| 智能路由（Hermes-router） | 26.9.0    | 是       | 默认版本提供                                         |
| 智能路由（Hermes-router） | 0.21.0     | 是       |                                                      |
| cache-indexer             | 26.9.0    | 是       | 默认版本提供                                         |
| cache-indexer             | 0.21.1     | 是       |                                                      |
| PD-Orchestrator           | 26.9.0    | 是       | 默认版本提供                                         |
| PD-Orchestrator           | 0.21.1     | 是       |                                                      |
| PD-Orchestrator           | 0.21.0     | 是       |                                                      |
| PD-Orchestrator           | 0.20.0     | 是       |                                                      |
| eagle-eye                 | 26.9.0    | 是       | 默认版本提供；新增端到端分布式链路追踪，v26.09版本更新 |
| eagle-eye                 | 0.22.0     | 是       |                                                      |
| eagle-eye                 | 0.21.0     | 是       |                                                      |
| infernex-checker          | 26.9.0    | 是       | 新增单机/跨节点慢卡检测，v26.09版本更新              |

### Hardware Prerequisite

下列为 InferNex 所面向的**推理加速硬件**型号及验证情况（与组件、环境、模型规格正交，可独立组合）。

| 硬件型号   | 验证情况 | 备注                        |
| ---------- | -------- | --------------------------- |
| 昇腾 910B4 | 是       | 26.9.0 默认 chart 目标硬件 |
| 昇腾 910B3 | 是       |                             |
| 昇腾 310P  | 否       |                             |

### Environment Prerequisite

下列为运行 InferNex 所需的**集群与平台环境**（含 LWS 依赖）。

| 环境项                          | 版本 / 要求 | 验证情况 | 备注                                                         |
| ------------------------------- | ----------- | -------- | ------------------------------------------------------------ |
| Kubernetes                      | 1.34.0      | 是       |                                                              |
| Kubernetes                      | 1.33.0      | 是       |                                                              |
| Kubernetes                      | 1.29.0      | 否       |                                                              |
| LeaderWorkerSet（LWS Operator） | v0.8.0      | 是       | InferNex 26.9.0（LWS）前置依赖；chart 使用 `leaderworkerset.x-k8s.io/v1`；安装见 https://lws.sigs.k8s.io/docs/installation/ |

### Model Supported

下列为 InferNex 在**默认 chart/values.yaml**下覆盖的模型支持范围；按用途分层，便于与组件、硬件、环境规格对照。MoE 类模型通常需配合 `dataParallelSize` / `dataParallelSizeLocal` 等多 DP 配置。

> 注意：以下模型规格验证中机器指 Atlas 800I A2机器，NPU卡910B3（64G）/910B4（32G）。

| 类型 | 模型 | 下载方式 | 部署规格 | 部署示例 |备注 |
|------|------|------|----------|------|------|
| 默认稠密基线 | Qwen3-8B | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 单机2卡910B4，prefill tp1 dp1, decode tp1 dp1 | [values.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/charts/infernex/values.yaml) | 26.6.0 默认 values.yaml 部署 |
| 默认稠密基线 | Qwen3-8B | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 单机单卡910B4，aggregated tp1 dp1 | [Qwen3-8B-vLLM-aggregated-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/Qwen3-8B-vLLM-aggregated-random.yaml) | |
| 基础 MoE | Qwen3-Coder-30B-A3B（及 `-Instruct` 变体） | [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 单机8卡910B4，prefill tp2 dp2, decode tp2 dp2 | [Qwen3-Coder-30B-A3B-Instruct-vLLM-pd-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/Qwen3-Coder-30B-A3B-Instruct-vLLM-pd-random.yaml) | 26.6.0 推荐起步MoE模型 |
| 主流 MoE 大模型 | MiniMax-M2.7-w8a8-QuaRot | [MiniMax-M2.7-w8a8-QuaRot](https://www.modelscope.ai/models/vllm-ascend/MiniMax-M2.7-w8a8-QuaRot) | 4机32卡910B3，prefill tp8 dp2, decode tp8 dp2 | [Minimax-m2.7-vLLM-pd-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/Minimax-m2.7-vLLM-pd-random.yaml) | |
| 主流 MoE 大模型 | GLM-5.1-w4a8 | [GLM-5.1-w4a8](https://modelers.cn/models/Eco-Tech/GLM-5.1-w4a8) | 4机32卡910B3，prefill tp8 dp2, decode tp2 dp8 | [GLM-5.1-w4a8-vLLM-pd-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/GLM-5.1-w4a8-vLLM-pd-random.yaml) | vllm-ascend 使用 nightly-main-0606 镜像| 
| 主流 MoE 大模型 | GLM-5.2-w8a8 | [GLM-5.2-w8a8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w8a8) | 4机32卡910B3，aggregated tp8 dp4 | [GLM-5.2-w8a8-vLLM-aggregated-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/GLM-5.2-w8a8-vLLM-aggregated-random.yaml) | vllm-ascend 使用 glm5.2-openeuler 镜像 |
| 主流 MoE 大模型 | DeepSeek-V4-Flash-w8a8-mtp | [DeepSeek-V4.1-Flash-w8a8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8) | 4机32卡910B4，prefill tp8 dp2, decode tp8 dp2 | [DeepSeek-V4-Flash-w8a8-mtp-vLLM-pd-random.yaml](https://gitcode.com/openFuyao/InferNex/blob/master/examples/DeepSeek-V4-Flash-w8a8-mtp-vLLM-pd-random.yaml) | vllm-ascend v0.23.0 |

### Multi-Component Integration Specification

下列为 InferNex 26.9.0 在**默认 chart/values 组合**下声明的配套组件版本，与上一节「Subcomponent Specification」互补：前者描述**发行默认集成快照**，后者描述**单组件可验证版本边界**。若与正式发行说明不一致，以发行说明为准。

| 名称                      | InferNex 26.9.0 默认配置     |
| ------------------------- | ---------------------------- |
| 推理引擎（vllm-ascend）   | v0.23.0                      |
| 开源网关（Istio）         | 1.28.0                       |
| 智能路由（Hermes-router） | 26.9.0                       |
| cache-indexer             | 26.9.0                       |
| PD-Orchestrator           | 26.9.0                       |
| eagle-eye                 | 26.9.0                       |
| Kubernetes                | 1.34.0                       |
| 默认硬件                  | 910B4                        |
| 默认稠密模型              | Qwen3-8B                     |
| 验证基线 MoE 模型         | Qwen3-Coder-30B-A3B-Instruct |


## Components

- **InferNex-Checker**：InferNex 前置校验工具，在 install 前检查硬件、K8s 集群及配置环境，提前发现部署风险。
- **InferNex-Bridge**：InferNex 接入 KServe 的适配层，支持 `LLMInferenceService` / `InferNexService` 双 CRD 声明式部署 InferNex，详见 [组件 README](component/InferNex-Bridge/README-zh.md)。

## Roadmap
 - [26-06] Hermes-router 智能路由支持基于实例资源饱和状态的感知与调度、PD 分离架构下的实例间二级调度。
 - [26-06] InferNex-Deployer 完善持续集成链路，增加强大模型 PD 分离与多实例部署的生产级验证。
 - [26-06] Elastic-scaler 侧重作业分发加速（权重、镜像、进程启动），支撑高性能弹性场景，同时补齐事件信号驱动的伸缩与算力感知策略。
 - [26-X] Eagle-Eye 近实时可观测将扩展动态网络资源指标，延伸硬件健康与亚健康感知，适配 A5 代际规格并推进错误码标准化。
 - [26-X] KVCacheX 覆盖 Cache-indexer / conductor 与灵衢使能等相关方向；规划纳入 DSA 与 Hybrid Attention KV offloading 等能力的迭代增强。
 - [26-X] 规划 KServe 对接适配，便于统一管理 predictive、LLM 等不同类型推理 Serving 及 InferNex、llm-d 等算力栈流量；并与 vLLM-ascend 社区协同发布基于 InferNex 的推荐部署案例。
