---
# 提案标题
title: Infernex 接入 KServe 适配层
# 提案编号
ofep-number: 0040
# 提案作者
authors:
  - "@xinhongchen"
# 提案主导SIG
owning-sig: sig-ai-inference
# 提案协作SIG
participating-sigs:
# 提案状态(初步草案|准备开始实现|已在SIG中实现并合并|提案被暂缓|提案被否决|作者主动撤回提案|被另一个oFEP取代)
status: provisional
# 创建日期
creation-date: 2026-03-24
# 评审人
reviewers:
  - foxbit
# 批准人
approvers:
# 相关联的其他oFEP
see-also:
# 当前oFEP替代了哪些已有提案
replaces:
# 当前oFEP被哪些提案替代，当status为replaced时需要填写此字段
replaced-by:
# 此oFEP当前开发阶段
stage: alpha
# 最近一次推进的版本
latest-milestone: "v0.1.0"
# 各阶段目标版本
milestone:
  alpha: "v0.1.0"
feature-gates:
  - name: 
# 是否支持关闭该功能
disable-supported: true
# 该功能引入的监控指标
metrics:
---
# oFEP-0040：InferNex 接入 KServe 适配层

- [发布签核清单](#发布签核清单)
- [概括](#概括)
- [动机](#动机)
  - [目标](#目标)
  - [非目标](#非目标)
- [提案](#提案)
  - [用户故事](#用户故事)
  - [注释/限制/注意事项](#注释限制注意事项)
  - [风险与缓解措施](#风险与缓解措施)
- [设计细节](#设计细节)
  - [逻辑视图](#逻辑视图)
  - [部署视图](#部署视图)
  - [运行视图](#运行视图)
  - [开发详解](#开发详解)
    - [1. KServe llmisvc 可覆盖范围（穿刺验证）](#1-kserve-llmisvc-可覆盖范围穿刺验证)
    - [2. Webhook 设计](#2-webhook-设计)
    - [3. InferNexServiceController Reconcile 概要](#3-infernexservicecontroller-reconcile-概要)
    - [4. InferNexService CRD 设计](#4-infernexservice-crd-设计)
      - [API 定义](#api-定义)
      - [字段说明](#字段说明)
      - [使用示例](#使用示例)
- [测试计划](#测试计划)
  - [单元测试](#单元测试)
  - [集成测试](#集成测试)
  - [e2e 测试](#e2e-测试)

## 发布签核清单

标记有（R）的项目*在达到里程碑/发布*之前是必需的。
- [ ]（R）发布里程碑中的增强问题，链接到 [openfuyao/ofep] 中的 oFEP 目录
- [ ] (R) oFEP 审批者已批准 oFEP 状态为“可实施”
- [ ] (R) 设计细节已适当记录
- [ ]（R）测试计划已到位，并考虑了 SIG 架构和 SIG 测试的输入（包括测试重构）
  - [ ] 针对所有 Beta API 操作（端点）进行 e2e 测试
  - [ ] (R) 确保 GA e2e 测试满足一致性测试的要求
  - [ ] (R) GA e2e 测试至少需要两周时间才能证明测试结果无 flake（不稳定或偶发失败）
- [ ] (R) 毕业标准已设定
  - [ ] (R) 所有 GA 端点必须通过[一致性测试]
- [ ] (R) 生产准备情况审查完成
- [ ] (R) 生产准备情况审查已获批准
- [ ] “实施历史”部分已更新里程碑
- [ ] 面向用户的文档已在 [openfuyao/docs] 创建，以便发布到 [openfuyao.cn]
- [ ] 支持文档 - 例如，额外的设计文档、邮件列表讨论/SIG 会议链接、相关 PR/问题、发行说明

<!--
**注意：**此清单是迭代的，每次考虑将此增强功能作为里程碑时都应进行审查和更新。
-->

- [openfuyao.cn](https://openfuyao.cn/)
- [openfuyao/ofep](https://gitcode.com/openfuyao/ofep)
- [openfuyao/docs](https://gitcode.com/openfuyao/docs)


## 概括

本提案旨在为 openFuyao AI 推理服务框架（InferNex）与 KServe 提供兼容接入，提出一套以 Webhook 与 `InferNexService` CRD 为核心的适配层，该适配层称为 **InferNex Bridge**。InferNex 面向基于 NPU 的推理栈；用户仍以 `LLMInferenceService`（KServe 上用于声明大模型推理服务的 CR，下文简称 `llmisvc`）为使用入口，由 KServe 的 `llmisvc` 控制器负责推理后端与路由网关；由 `InferNexService` 控制器负责 proxy-server（仅在 P/D 模式下存在）、Mooncake、cache-indexer、eagle-eye、PD-Orchestrator 等其余 InferNex 组件。

该方案通过 Validating 准入 Webhook 对启用 InferNex 的 `llmisvc` 做配置语义校验，校验通过则放行，不修改资源对象配置；通过约定标签 `infernex.io/runtime: true` 由用户在 `llmisvc` YAML 中声明，由 `InferNexService` 控制器监听带约定标签的 `llmisvc` 并创建/维护 `InferNexService`，在此基础上编排补充资源。核心设计包括：

- **Webhook 校验与放行**：主要通过 Validating Webhook 检查配置合法性，并识别 `LLMInferenceService` 是否带有 InferNex 标识，从而确定是否需要拉起关联组件；同时通过 Mutating Webhook 对目标 `LLMInferenceServiceConfig` 做一次性模板改写（如清理 `drop: ALL`、清空 decode 侧 `initContainers`），并通过注解门闩保证幂等。
- **InferNexService 主动适配**：`llmisvc` 控制器负责推理后端与路由网关；`InferNexService` 控制器负责 proxy-server（仅在 P/D 模式下存在）、Mooncake、cache-indexer、eagle-eye、PD-Orchestrator 等其余 InferNex 组件。

## 动机

近年来，大语言模型推理在业务侧落地加速，模型服务平台逐步走向多引擎、多集群与异构算力并存的格局，不同技术栈间的协同与资源差异正成为平台能力演进的关键挑战。

通过将 InferNex、llm-d 等异构技术栈统一纳入 KServe 的声明式部署规范，平台能够将底层差异化的引擎能力抽象为标准化的资源对象，这一适配逻辑实现了不同技术栈在统一路由平面下的接入，有效地屏蔽了单集群内异构资源的定义差异，并为后续通过请求级路由实现算力资源的深度感知与转发预留了空间。

### 目标

1. **支持 KServe** `llmisvc` **控制器与本提案 Webhook、**`InferNexService` **控制器协同拉起面向 NPU 的 InferNex**：三者共同协作，在 `llmisvc` 带 InferNex 约定标签时，于 NPU 场景下完成 InferNex 的协同部署。
2. **支持可选的** `InferNexService` **直部署**：除与 `llmisvc` 协同外，有需要的用户也可直接提交 `InferNexService`（例如无 `sourceRef` 的独立部署），仅由 InferNex 控制面完成声明与编排。

### 非目标

1. **不替代 KServe 对** `llmisvc` **的核心语义**：不侵入式修改上游 KServe 提供的 `LLMInferenceService` CRD；对未打 InferNex 约定标签的 `llmisvc`，适配层不对其改写或抢占 reconcile，行为与原生 KServe 一致。
2. **不将** `InferNexService` **作为用户的强制迁移目标**：默认仍以 `LLMInferenceService` 为入口；有需要的用户亦可仅通过 `InferNexService` 声明部署。

## 提案

### 用户故事

#### 故事 1
作为用户，我希望使用 KServe 的 `LLMInferenceService`（`llmisvc`）来拉起面向 NPU 的 `vllm-ascend` 推理后端。
<!-- ```plantuml -->
@startuml
left to right direction
skinparam componentStyle rectangle

actor "用户" as User

rectangle "KServe" {
  (通过 LLMInferenceService 拉起 vLLM Ascend（NPU）推理后端) as UC1
  (创建 LLMInferenceService) as UC2
  (部署 vLLM Ascend（NPU）推理后端) as UC3
}

User --> UC1
UC1 ..> UC2 : <<include>>
UC2 ..> UC3 : <<include>>
@enduml
<!-- ``` -->

#### 故事 2
作为用户，我希望通过 KServe 的 `llmisvc` 拉起面向 NPU 的 openFuyao AI 推理服务框架（InferNex）；当我在 `llmisvc` 上打上对应 label 后，平台会自动创建/关联 `InfernexService`，并把 InferNex 相关组件一并纳入部署。
<!-- ```plantuml -->
@startuml
left to right direction
skinparam componentStyle rectangle

actor "用户" as User

rectangle "KServe" {
  (通过 LLMInferenceService 拉起 InferNex 推理服务框架) as KS1
  (创建 LLMInferenceService（llmisvc）并打标) as KS2
  (部署推理后端与路由网关) as KS3
}

rectangle "InferNexService" {
  (创建/关联 InfernexService) as IX1
  (部署 InferNex 组件能力) as IX2
}

User --> KS1
KS1 ..> KS2 : <<include>>
KS2 ..> KS3 : <<include>>
KS1 ..> IX1 : <<include>>
IX1 ..> IX2 : <<include>>
@enduml
<!-- ``` -->

#### 故事 3
作为高级用户/平台研发，我希望直接提交 `InfernexService` 来拉起 InferNex；并可按需选择启用哪些组件，最终获得推理服务。
<!-- ```plantuml -->
@startuml
left to right direction
skinparam componentStyle rectangle

actor "高级用户" as User

rectangle "InferNexService" {
  (通过 InferNexService 拉起 InferNex 推理服务框架) as UC1
  (选择启用的 InferNex 组件) as UC2
  (创建 InfernexService CR) as UC3
  (部署 InferNex 推理服务框架) as UC4
}

User --> UC1
UC1 ..> UC2 : <<include>>
UC1 ..> UC3 : <<include>>
UC3 ..> UC4 : <<include>>
@enduml
<!-- ``` -->

### 注释/限制/注意事项

1. `infernex.io/runtime=true`表示推理运行时为 InferNex，是控制面切换的前提标记。
2. MutatingWebhook 会在准入阶段对目标 `LLMInferenceServiceConfig` 执行一次性模板改写并写入 annotation 门闩，相关影响与约束见下文“约束”章节。

### 约束

通过 InferNex-Bridge 部署 InferNex 后，MutatingWebhook 会对 KServe 的 `LLMInferenceServiceConfig` 模板进行相关的注入：

- 对 prefill / decode 的 template 去掉 `securityContext.capabilities.drop` 中的 `ALL`。
- 清空 decode 侧 `initContainers`，不再保留原先通过 init 注入 sidecar 的启动链路。

影响：

- 对 InferNex 适配路径本身不构成额外兼容性问题，但会影响后续 `llm-d` 推理后端的部署，主要原因是 decode 侧不再注入原有 sidecar 后，原依赖 sidecar 的流量转发、启动探测或链路观测行为可能缺失。
- 去掉 `drop: ALL` 后，容器能力基线上移，可能触发与既有安全策略不一致的运行差异。


## 设计细节

### 逻辑视图

逻辑视图展示了InferNex 接入 KServe 适配层在 K8s 环境下的工作关系。系统结构分为用户层、KServe 控制面、InferNex 控制面以及业务运行层，主要通过两个阶段来实现 InferNex 的部署：

1. **准入校验阶段**：用户提交 LLMInferenceService 时，声明 `infernex.io/runtime=true` 的标签， Validating Webhook 可对带 InferNex 约定标签的对象做校验。
2. **LLMInferenceService 和 InferNexService 控制阶段**：LLMInferenceService Controller 利用 K8s 的 Watch 和 Reconcile 机制，通过监听资源变化来配置 Istio Gateway 流量入口，并拉起 Hermes-Router 和推理后端。与此同时，InferNexServiceController 通过Watch机制监听带标签的 `llmisvc` 资源对象，自动创建 InferNexService (CR)，作为整个适配层的配置核心。

<!-- ```plantuml -->

@startuml
top to bottom direction
skinparam packageStyle rectangle
skinparam monochrome true
skinparam shadowing false

actor "用户/客户端" as USER

node "Kubernetes 集群" as K8S {

  ' --- KServe 控制面 ---
  frame "KServe 控制面" as NS_KSERVE {
    node "LLMInferenceService (CR)" as LLMISVC
    node "KServe Controller" as KCTRL
  }

  ' --- InferNexService 控制面 ---
  frame "InferNexService 控制面" as NS_INFER {
    node "InferNexService Controller" as I_CTRL
    node "InferNexService (CR)" as INFSVC
    node "InferNexService Webhook\n(Validating\Mutating)" as I_WEB
  }

  ' --- 业务运行层 (Data Plane) ---
  frame "业务运行层 (InferNex)" as NS_RUNTIME {
    node "Istio Gateway\n(流量入口)" as GATEWAY
    node "Hermes-Router\n(智能路由)" as HERMES
    node "Proxy-Server\n(P/D 分流转发)" as PROXY
    node "InferNex 推理后端\n(vLLM-ascend Pods)" as BACKEND
    
    node "PD-Orchestrator (弹性伸缩)" as PDO
    node "Cache-Indexer (KV 索引)" as INDEXER
    node "Eagle-Eye (性能监控)" as EAGLE
    
    database "Mooncake Store\n(KVCache 池)" as MOONCAKE
  }
}

' --- 逻辑流开始 ---

' 1. 准入阶段
USER --> LLMISVC : 1. Apply (YAML)
I_WEB .> LLMISVC : 2. Validating\n(准入校验)

' 2. InferNex 控制面初始化 (提前至第 3 步)
LLMISVC <- I_CTRL : 3. Watch (感知新资源)
I_CTRL --> INFSVC : 3. Create/Reconcile
INFSVC .> LLMISVC : SourceRef

' 3. KServe 基础基础设施部署
LLMISVC <-- KCTRL : 4. Watch
KCTRL --> GATEWAY : 5. 配置网关路由
KCTRL --> HERMES : 6. Reconcile \n(Deploy)
KCTRL --> BACKEND : 6. Reconcile\n (Deploy)

' 4. InferNex 组件补齐 (依赖于后端状态或配置)
I_CTRL --> PROXY : 7. Reconcile \n(Deploy)\n(PD 模式)
I_CTRL --> PDO : 8. Reconcile \n(Deploy)
I_CTRL --> INDEXER : 8. Reconcile \n(Deploy)
I_CTRL --down-> EAGLE : 8. Reconcile \n(Deploy)

' 5. 数据链路

GATEWAY --> HERMES
HERMES --> PROXY
PROXY --> BACKEND
BACKEND <--> MOONCAKE : KV Cache 读写
INDEXER ..> BACKEND : 缓存管理
PDO ..> BACKEND : 策略驱动

@enduml

<!-- ``` -->

### 部署视图

部署视图展示了系统在 Kubernetes 集群中的物理部署结构。系统运行依赖于两个核心部分的协同：一是外部 KServe 基础组件，包括部署在集群中的 `LLMInferenceService` 控制器及其关联的 CRD 资源；二是InferNex接入KServe的适配层。

适配层以独立 Pod 形式运行，内部集成了 InferNexService Controller 与 Webhook。Webhook 同时包含 Validating 与 Mutating 两类准入逻辑：前者通过 `Webhook Service` 与集群准入配置对接，对 `llmisvc` 做校验与放行；后者按需对目标 `LLMInferenceServiceConfig` 做模板改写。Controller 则配合 `InferNexService` CRD 监听资源变化。三者共同工作，在不侵入 KServe 原生逻辑的前提下，实现对 InferNex 推理服务框架的组件部署。

<!-- ```plantuml -->

@startuml
' --- 基础设置 ---
skinparam monochrome true
skinparam shadowing false
skinparam packageStyle rectangle
left to right direction

' --- 外部环境 (制品来源) ---
cloud "制品库 (Registry)" as REGISTRY {
    artifact "infernex-manager-image" as IMG_INF
    artifact "kserve-controller-image" as IMG_KS
    
    ' 新增部署形式：Chart
    artifact "infernex-bridge-chart" as CHART_INF <<Helm Chart>>
    artifact "kserve-chart" as CHART_KS <<Helm Chart>>
}

' --- Kubernetes 集群控制面 ---
node "Node" as CP_NODE {

    ' 1. KServe 部署层
    frame "Namespace: kserve" as NS_K {
        package "KServe Deployment" <<Deployment>> {
            node "LLMInferenceService Manager Pod" <<Pod>> as K_POD {
                artifact "LLMInferenceService Controller Manager"
            }
        }
        database "LLMInferenceService (CRD)" as CRD_LLM <<K8s API Resource>>
    }
    
    ' 2. InferNex-Bridge 部署层
    frame "Namespace: infernex-bridge-system" as NS_I {
        package "InferNexService Deployment" <<Deployment>> {
            node "InferNexService Manager Pod" <<Pod>> as I_POD {
                artifact "InferNexService Validating\Mutating Webhook"
                artifact "InferNexService Controller Manager"
            }
        }
        
        ' 基础设施 CRD 组
        folder "InferNex CRDs" {
            database "InferNexService (CRD)" as CRD_INF
        }

        interface "Webhook Service" as I_SVC
    }
    
    artifact "ValidatingWebhookConfiguration" as WEB_CONF
}

' --- 部署与逻辑关联 ---

' 镜像交付路径
IMG_KS ..> K_POD : 1. 运行
IMG_INF ..> I_POD : 1. 运行

' Chart 部署路径 (新增)
CHART_KS ..> NS_K : 2. 部署/安装
CHART_INF ..> NS_I : 2. 部署/安装

' 准入拦截链路
WEB_CONF --> I_SVC : 拦截 llmisvc 请求
I_SVC --> I_POD : 路由至 Webhook 逻辑

' 控制与观测链路
I_POD ..> CRD_INF : 维护 CR 状态
I_POD ..> CRD_LLM : 监听 llmisvc 变更 (Watch)

' KServe 自身的逻辑
K_POD ..> CRD_LLM : 处理资源定义

' 布局微调
NS_K -[hidden]right- NS_I

@enduml

<!-- ``` -->


### 运行视图

运行视图通过时序图展示了系统从用户提交推理请求到适配层组件全面就绪的完整运行时流程。流程包括两个主要阶段：

1. **资源提交与准入校验阶段**：用户提交带有 InferNex 约定标签（如 `infernex.io/runtime: true`）的 `LLMInferenceService`（llmisvc）至 API Server；InferNex **Validating Webhook** 先对请求做静态校验（Allowed/Denied，不修改对象），随后按需进入 **Mutating Webhook**（例如对目标 `LLMInferenceServiceConfig` 做一次性模板改写与幂等注解门闩）；最终持久化本次提交的 `LLMInferenceService`，再由 KServe 与 `InferNexService` 控制面按标签与 spec 继续处理。
2. **组件并行部署阶段**：系统进入异步调和流程，KServe 核心控制器与InferNexService 控制器 通过 Watch 机制并发感知资源变化。KServe 负责拉起 Hermes-Router 与 vLLM-Ascend 推理后端；与此同时，InferNex 适配器自动创建并绑定 InferNexService CR，并以并行方式按需下发增强组件指令，包括 PD-Orchestrator、Cache-Indexer、Eagle-Eye及 Mooncake；若识别为 PD 模式，则额外拉起 Proxy-Server 以实现预填充与解码的分流转发，最终确保适配层与推理引擎达到最终一致性的就绪状态。
<!-- ```plantuml -->
@startuml
skinparam monochrome true
skinparam shadowing false
skinparam sequence {
    MessageAlignment center
}

' --- 参与者定义 ---
participant "用户" as User
participant "K8s API Server" as API
participant "etcd" as etcd
participant "InferNex ValidatingWebhook" as V_Webhook
participant "InferNex MutatingWebhook" as M_Webhook
participant "LLMInferenceService Controller" as K_Ctrl
participant "InferNexService Controller" as I_Ctrl
participant "InferNex (Runtime)" as INF

== [阶段 1] LLMInferenceService 提交与请求处理 ==

User -> API : Apply LLMInferenceService CR\n(含 infernex.io/runtime: true)
API -> V_Webhook : Validating Admission Call
Note right of V_Webhook: 校验 spec / 约定标签
V_Webhook -->> API : Allowed / Denied
API -> M_Webhook : Mutating Admission Call 
Note right of M_Webhook: 目标对象改写 + 幂等注解门闩
M_Webhook -->> API : Patch / NoPatch
API -> etcd : 持久化 LLMInferenceService
etcd -->> API : 确认存储
API -->> User : 响应已创建 (Created)

== [阶段 2] 部署 InferNex 组件 ==

par KServe 控制面部署
    K_Ctrl -> API : Watch LLMInferenceService
    par
        K_Ctrl -> INF : Deploy Hermes-Router
    else
        K_Ctrl -> INF : Deploy vLLM-Backend
    end
    
else InferNex 适配器部署
    I_Ctrl -> API : Watch LLMInferenceService
    I_Ctrl -> API : Create & Bind InferNexService (CR)
    API -> etcd : 持久化 InferNexService
    
    par
        I_Ctrl -> INF : Deploy PD-Orchestrator
    else
        I_Ctrl -> INF : Deploy Cache-Indexer
    else
        I_Ctrl -> INF : Deploy Eagle-Eye
    else
        I_Ctrl -> INF : Deploy Mooncake
    else
        alt is PD Mode
            I_Ctrl -> INF : Deploy Proxy-Server
        end
    end
end

@enduml
<!-- ``` -->
### 开发详解

本提案提出在不更换用户入口的前提下，通过Webhook + InferNexService建立适配层，用户继续提交 `llmisvc`，由 `llmisvc` 控制器负责推理后端与路由网关；在打上约定标签后，由 `InferNexService` 控制器补齐 proxy-server（仅在 P/D 模式下存在）、Mooncake、cache-indexer、eagle-eye、PD-Orchestrator 等其余 InferNex 组件，从而在沿用 KServe 使用习惯的同时，落地面向 NPU 的 InferNex 能力。

#### 1. KServe `LLMInferenceService` 可覆盖组件

基于穿刺验证，下表归纳 **在仅使用 `LLMInferenceService`（`llmisvc`）时**，各类 InferNex 相关能力能否由 KServe 侧声明并拉起，作为后文 `InferNexService` CRD 的引入。

| InferNex 相关组件/能力                                       | 能否仅用 `LLMInferenceService` 拉起 |
| ------------------------------------------------------------ | ----------------------------------- |
| **推理后端工作负载**                                         | 部分                                |
| ├─ 聚合模式                                                  | 是                                  |
| ├─ P/D 模式 prefill/decode 工作负载                          | 部分（缺少proxy-server组件） |
| └─ 并行策略（TP/PP/DP/EP）                                   | 是                                  |
| **智能网关路由与调度**                                       | 是                                  |
| ├─ Hermes Router / EPP 配置                                  | 是                                  |
| ├─ Gateway / HTTPRoute / InferencePool                     | 是                                  |
| 独立 Proxy Server（P/D 必选）                                | 否                                  |
| Cache Indexer                                                | 否                                  |
| Mooncake（Master / MetadataServer 等）                       | 否                                  |
| PD Orchestrator（elasticScaler / tidal / resourceScalingGroup） | 否                                  |
| EagleEye                                                     | 否                                  |

#### 2. Webhook 设计

**约定标签**

`infernex.io/runtime: true`：表示推理运行时为 InferNex，从而走 InferNex 适配路径。适配层 ValidatingWebhook 仅对带有该 runtime 标识的 `LLMInferenceService`（`llmisvc`）做下文「InferNex 增量校验」；未带该标识的 `LLMInferenceService` 直接放行，不与 KServe 通用准入重复处理。

**ValidatingWebhook**

对带上述 runtime 标识的 `llmisvc` 的 spec 做 InferNex 相关合法校验：通过则 `Allowed`，失败则 `Denied` 并返回可读错误信息。与 KServe 侧及既有 `llmisvc` Webhook 已覆盖的通用校验不重复；本层只补 InferNex 路径上需对齐的三块：`model`、按 aggregate / P/D 区分的推理后端拓扑、以及 InferNex 路由（Hermes Router）与后端的引用是否闭合。


**MutatingWebhook**

在 Create / Update 准入阶段，对需按 InferNex 规则调整的 `LLMInferenceServiceConfig` 改写其 spec 中的 prefill / decode 模版。主要做两类变更，便于 NPU 场景下 InferNex 与 vLLM-Ascend 等进程按权限启动推理后端以及避免sidecar的空转：

1. 去掉 `securityContext.capabilities.drop` 中的 `ALL`：`LLMInferenceServiceConfig` 里 prefill 与 decode 两侧 template 若配置了 `securityContext.capabilities.drop: ["ALL"]`，MutatingWebhook 应删除该 drop 列表或移除其中对 `ALL` 的条目，避免与 InferNex 的推理后端因为权限问题无法正常部署。
2. 清空 decode 侧 `initContainers`：decode 模版中的 `initContainers` 应置空或整段删除，用于去掉原先通过 init 注入的 sidecar ，避免进程空转。

改写完成后，须在`LLMInferenceServiceConfig`对象 `metadata.annotations` 写入注解（ `infernex.io/llm-inference-service-config-mutated: "infernex"`）：后续准入若已存在该注解，则不再对同一对象重复 patch，避免二次改写及与控制面反复写回冲突；若使用者需要重新套用规则，须由人工删除注解或调整版本策后再提交 Update。

主要涉及的配置文件与改写点如下：

| 配置文件 | 去掉 `securityContext.capabilities.drop: ["ALL"]` | 清空 `decode.initContainers` |
| --- | --- | --- |
| `outsidefiles/kserve_llmisvc/llmisvcconfig/config-llm-template.yaml` | 是（`template`） | 否 |
| `outsidefiles/kserve_llmisvc/llmisvcconfig/config-llm-worker-data-parallel.yaml` | 是（`worker`） | 否 |
| `outsidefiles/kserve_llmisvc/llmisvcconfig/config-llm-prefill-template.yaml` | 是（`prefill.template`） | 否 |
| `outsidefiles/kserve_llmisvc/llmisvcconfig/config-llm-prefill-worker-data-parallel.yaml` | 是（`prefill.worker`） | 否 |
| `outsidefiles/kserve_llmisvc/llmisvcconfig/config-llm-decode-template.yaml` | 是（`decode.template`） | 是 |
| `outsidefiles/kserve_llmisvc/llmisvcconfig/config-llm-decode-worker-data-parallel.yaml` | 是（`decode.worker`） | 是 |



#### 3. InferNexServiceController Reconcile 概要

控制器以 `For(InferNexService)` 为主入口，并对带 runtime 标签的 `LLMInferenceService` 建立 `Watches`，将事件映射到同名 `InferNexService`。当队列 key 对应的 `InferNexService` 不存在时，先尝试依据同名 `llmisvc` 创建 `InferNexService` 资源。进入调和后先判定模式（aggregate / pd），再按 `baseRefs` 或模式默认模板名加载并合并 `InferNexServiceConfig`，最后下发补充组件、清理非期望资源并更新状态。

<!-- ```plantuml -->
@startuml
skinparam shadowing false
skinparam monochrome true
autonumber

participant "K8s API Server" as API
participant "InferNex Controller" as Ctrl

== 1. 创建触发阶段 ==

API -> Ctrl: LLMInferenceService (带标签) 创建事件
Ctrl -> API: 获取/对比同名 InferNexService
alt 不存在
    Ctrl -> API: 创建 InferNexService (SourceRef 指向 LLMInferenceService)
end

== 2. 组件拉起阶段 ==

Ctrl -> API: 监听到 InferNexService 创建
Ctrl -> Ctrl: 计算配套组件清单 \n (Proxy Server\Mooncake\Cache Indexer\PD Orchestrator\EagleEye)
Ctrl -> API: 下发配套组件资源 (Owner 指向 InferNexService)
API -> Ctrl: 组件就绪状态反馈
Ctrl -> API: 更新 InferNexService Status 为 Ready

== 3. 删除阶段 ==

API -> Ctrl: LLMInferenceService 发生删除事件
Ctrl -> API: 主动执行删除 InferNexService

Ctrl -> API: 删除/收敛不再期望的附属组件 \n (Proxy Server\Mooncake\Cache Indexer\PD Orchestrator\EagleEye)

@enduml
<!-- ``` -->

Reconcile 主链（示意）：

监听资源在 controller 注册阶段（`SetupWithManager`）定义，当前同步监听两类资源：

- `For(InferNexService)`：`InferNexService` 自身变更触发调和
- `Watches(LLMInferenceService)`：`llmisvc` 事件映射到同名同 namespace 的 `InferNexService` 调和键；调和起始会校验 `sourceRef`，若源 `llmisvc` 不存在或已不带 runtime 标签，则主动删除关联 `InferNexService`

```go
func (r *InferNexServiceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// 1) Get(InferNexService)；NotFound 时尝试从同名 llmisvc 补建 InferNexService
	// 2) buildEffectiveSpec:
	//    - computeServingMode 计算 aggregate/pd
	//    - spec.baseRefs 为空时回退默认模板名（infernex-default-aggregate / infernex-default-pd）
	//    - 从 TemplateNamespace（控制器配置）加载模板并合并；为空时直接报错
	// 3) buildDesiredComponents，逐个 reconcile Deployment/Service，并清理多余 Deployment/Service/RBAC
	// 4) computeInferNexServiceStatus 后写回 status
	return ctrl.Result{}, nil
}
```

模板前置：

- 在控制器模板 namespace 预先安装 aggregate 默认模板
- 在控制器模板 namespace 预先安装 pd 默认模板
- 再创建 `InferNexService`（或由 `llmisvc` 触发自动补建），调和时执行模板合并

#### 4. InferNexService / InferNexServiceConfig CRD 设计

`InferNexService` 是 InferNex 控制面的主对象，用于声明与编排 InferNex 侧组件。其 `spec.sourceRef` 用于可选引用 `llmisvc`。  
`InferNexServiceConfig` 是模板对象（类似 KServe 的 `LLMInferenceServiceConfig`），用于预置组件模板并由 `InferNexService` 通过引用进行合并。

##### 与 `llmisvc` 协同时的职责边界（重要）

推理后端、Hermes Router、Gateway / HTTPRoute / InferencePool 等由 KServe `llmisvc` 控制器（及 `LLMInferenceServiceConfig`）负责编排；InferNex Bridge 只负责 `llmisvc` 无法单独覆盖的补充组件（例如 PD 模式下由 InferNex 提供的 proxy-server、Mooncake、cache-indexer、eagle-eye、PD-Orchestrator 等，具体以实现为准）。

因此，当 `spec.sourceRef` 指向某 `LLMInferenceService` 时：

- `sourceRef` 表示关联与生命周期联动（例如同名 `InferNexService` 与 `llmisvc` 配对、删除/摘标签时的协调），以及控制器对 `llmisvc` 的只读观测（例如判断是否 P/D、是否应为 InferNex 侧 proxy-server 下发资源）。
- 不把被引用 `llmisvc` 的 `spec` 合并进 `InferNexService` 的有效调和视图来 再次拉起 与 KServe 相同的推理引擎或 Router；否则两套控制面会对同一类工作负载重复 reconcile，集成不合理。
- `InferNexService.spec` 中与 KServe 重叠的块（`model`、`engine`、`intelligentGatewayRouting`）保留在 CRD 中，主要用于无 `sourceRef` 的直部署路径，或文档/工具链完整性；在以 `llmisvc` 为主入口的协同路径上，Bridge 实现应忽略这些字段用于部署与 `llmisvc` 重复的工作负载（仅部署补充组件 + 模板合并所允许的 InferNex 独占片段，例如 PD 下的 `proxyServer`）。

补充组件的默认值与模板合并 以 `InferNexService` CR 与 `InferNexServiceConfig` 为准，不以 `llmisvc` 作为 `InferNexService` 的隐式合并源。

##### API 定义


InferNexService 是面向 InferNex 组件编排的主 CRD：当用户以 `llmisvc` 为入口并启用 InferNex 约定标签时，用于补齐 `llmisvc` 之外的 InferNex 组件；用户也可以直接提交该 CR 完整声明 InferNex 部署。  
InferNexServiceConfig 用于承载可复用默认模板（如组件开关、副本、模板片段），控制器在调和 `InferNexService` 时按需读取并合并。

```yaml
apiVersion: infernex.infernex.io/v1alpha1
kind: InferNexService
metadata:
  name: <name>
  # 删除保护：由控制器写入/摘除。可采用单一 finalizer 在 Reconcile 内按组件串行清理；
  # 或按组件粒度多条（示例键名以实现为准），删除时逐条回收对应工作负载后再摘除。
  finalizers:
    - infernex.io/infernexservice
    # 可选示例（按组件）：infernex.io/finalizer.proxy-server、infernex.io/finalizer.cache-indexer 等
spec:
  # 可选：引用 KServe LLMInferenceService（关联与观测；不把 llmisvc.spec 合并为隐式默认值，见上文「职责边界」）
  sourceRef:
    apiVersion: serving.kserve.io/v1alpha1
    kind: LLMInferenceService
    name: <llmisvc-name>
    namespace: <namespace>

  # 可选：按顺序引用多个 InferNexServiceConfig（仅 name；查找 namespace 由控制器 TemplateNamespace 决定）
  baseRefs:
    - name: <infernexserviceconfig-name-a>
    - name: <infernexserviceconfig-name-b>

  # 无 sourceRef 的直部署路径下通常必填；有 sourceRef 时由实现决定是否要求（补充组件若需模型元数据可显式填或走模板）
  model:
    uri: <model-uri>
    name: <model-name>

  # 二选一：engine.aggregate 或 engine.pd
  engine:
    aggregate:
      replicas: <int>
      parallelism: {}
      template: {}
      worker: {}
    pd:
      prefill:
        replicas: <int>
        parallelism: {}
        template: {}
        worker: {}
      decode:
        replicas: <int>
        parallelism: {}
        template: {}
        worker: {}
      proxyServer:
        enabled: <bool>
        replicas: <int>
        servicePort: <int>
        template: {}

  intelligentGatewayRouting:
    router:
      enabled: <bool>
      replicas: <int>
      servicePort: <int>
      template: {}
    httpRoute:
      refs:
        - name: <httpRoute-name>
      spec: {}
    gateway:
      refs:
        - name: <gateway-name>
      spec: {}
    inferencePool:
      refs:
        - name: <inferencePool-name>
      spec: {}
  components:
    cacheIndexer:
      enabled: <bool>
      replicas: <int>
      servicePort: <int>
      template: {}
      serviceDiscovery:
        labelSelector: <string>
        refreshInterval: <int>

    mooncake:
      enabled: <bool>
      master:
        enabled: <bool>
        replicas: <int>
        template: {}
      metadataServer:
        enabled: <bool>
        replicas: <int>
        template: {}

    pdOrchestrator:
      elasticScaler:
        enabled: <bool>
        replicas: <int>
        template: {}
      tidal:
        enabled: <bool>
        replicas: <int>
        template: {}
      resourceScalingGroup:
        enabled: <bool>
        replicas: <int>
        template: {}

    eagleEye:
      hardwareMonitor:
        enabled: <bool>
        replicas: <int>
        template: {}
      hardwareDiagnosis:
        enabled: <bool>
        replicas: <int>
        template: {}
# status 子资源：只读；汇总「KServe/llmisvc 不负责拉起」的 InferNex 补充组件就绪情况，
# 以及有效运行模式与整体是否可服务（与 llmisvc 侧 backend/router 状态互补观测）。
status:
  # 有效引擎形态：aggregate | pd（可由合并后的 spec.engine 解析；若存在 sourceRef，也可由控制器观测 llmisvc 得到，与实现一致）
  mode: <aggregate | pd>
  # 本 CR 所编排的 InferNex 补充面是否整体就绪（各必选子组件均满足时可 True）
  ready: <bool>
  observedGeneration: <int>
  # 按组件粒度就绪（proxy-server、cache-indexer、mooncake、pd-orchestrator、eagle-eye 等；未启用可省略或标记为 Skipped）
  components:
    proxyServer:
      ready: <bool>
      message: <string>
    cacheIndexer:
      ready: <bool>
      message: <string>
    mooncake:
      ready: <bool>
      message: <string>
    pdOrchestrator:
      ready: <bool>
      message: <string>
    eagleEye:
      ready: <bool>
      message: <string>
```

`InferNexServiceConfig`（模板对象）示意：

```yaml
apiVersion: infernex.infernex.io/v1alpha1
kind: InferNexServiceConfig
metadata:
  name: infernex-default-pd
spec:
  # 与 InferNexService.spec 结构保持一致，便于 merge
  engine:
    pd:
      proxyServer:
        enabled: true
        replicas: 1
  components:
    cacheIndexer:
      enabled: true
      replicas: 1
```

##### 字段说明

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| metadata.finalizers | []string | 由 InferNexServiceController 维护，用于删除时阻塞 `InferNexService` 直至附属资源回收完毕。可实现为单一 `infernex.io/infernexservice` 在 Reconcile 内按组件顺序清理；或按组件多条 finalizer（如 proxy-server、cache-indexer、mooncake、pd-orchestrator、eagle-eye）。 |
| spec.sourceRef | TypedObjectReference | 可选；指向 KServe `LLMInferenceService`，用于 关联、生命周期联动与只读观测；不表示把 `llmisvc.spec` 合并进本对象作为隐式默认部署语义（见「职责边界」）。 |
| spec.baseRefs | []NamedRef | 可选；仅含 `name`。按顺序加载 `InferNexServiceConfig` 并合并 `spec.components` / `spec.engine`（见 `mergeSpecFromTemplate`）。模板查询 namespace 取控制器 `TemplateNamespace`（来自 `INFERNEX_TEMPLATE_NAMESPACE` 或 `POD_NAMESPACE`）；若为空则报错。若为空列表，控制器按模式选用默认名：`infernex-default-aggregate`（aggregate）或 `infernex-default-pd`（pd）。 |
| spec.model | LLMModelSpec | 在无 `sourceRef` 的直部署路径下通常必填（至少 `uri`）；有 `sourceRef` 时可选，是否必填由实现约定（补充组件若需模型元数据可显式填写或来自 `InferNexServiceConfig`）。 |
| spec.engine | InferenceEngineSpec | 可选；直部署路径下声明推理引擎形态与工作负载；有 `sourceRef` 且以 `llmisvc` 为主入口时，Bridge 不应据此部署与 KServe 重复的推理工作负载（仅可编排 InferNex 独占片段如 `proxyServer`，以实现为准）。`aggregate` 与 `pd` 二选一。 |
| spec.engine.aggregate | InferenceEngineWorkloadSpec | 可选；聚合/单路径模式工作负载（副本、并行、模板、worker）；与 `llmisvc` 协同时由 KServe 负责，不在 Bridge 重复拉起。 |
| spec.engine.pd | PDModeServingSpec | 可选；P/D 模式工作负载集合；prefill/decode 由 KServe 负责；proxyServer 可为 InferNex 侧由 Bridge 编排。 |
| spec.engine.pd.proxyServer | ComponentSpec | 可选；P/D 模式下 InferNex 提供的 proxy-server（分流/转发）；是否与 `llmisvc` 协同由实现根据 `sourceRef` 与 `llmisvc` 观测结果决定。 |
| spec.intelligentGatewayRouting | IntelligentGatewayRoutingSpec | 可选；直部署路径或与 `llmisvc` 协同时的占位/扩展；以 `llmisvc` 为主入口时 Router/Gateway 由 KServe 编排，Bridge 不应重复部署 Hermes/Gateway API 工作负载。** |
| spec.intelligentGatewayRouting.router | ComponentSpec | 可选；与 `spec.intelligentGatewayRouting` 同上；协同路径下通常由 `llmisvc`/`LLMInferenceServiceConfig` 声明。 |
| spec.components | InfernexComponentsSpec | 可选；InferNex 侧独立组件集合。 |
| spec.components.cacheIndexer | CacheIndexerComponentSpec | 可选；cache-indexer 组件及其 serviceDiscovery 配置。 |
| spec.components.mooncake | MooncakeComponentSpec | 可选；mooncake master/metadata-server 组件编排。 |
| spec.components.pdOrchestrator | PDOrchestratorComponentSpec | 可选；P/D 编排组件组（含 elasticScaler/tidal/resourceScalingGroup 等）。 |
| spec.components.eagleEye | EagleEyeComponentSpec | 可选；观测组件组（hardwareMonitor/hardwareDiagnosis）。 |
| status | InferNexServiceStatus | 只读子资源；不重复描述已由 llmisvc/KServe 拉起的主推理与主路由就绪情况，而聚焦 InferNex 补充组件（proxy、Mooncake、cache-indexer、PD-Orchestrator、EagleEye 等）的就绪与摘要信息。 |
| status.mode | string | `aggregate` 或 `pd`；可与合并后的 `spec.engine` 一致，或在存在 `sourceRef` 时由控制器观测 `llmisvc` 得到，便于观测与告警。 |
| status.ready | bool | 在当前 spec 下，本 CR 所负责的 InferNex 补充面是否整体就绪（各必选子组件均满足时为 True；未启用的组件可视为 Skipped 不计入阻塞）。 |
| status.observedGeneration | int64 | 已调和完成的 `metadata.generation`，用于判断 spec 变更是否已被控制器观测。 |
| status.components | map 或结构化子字段 | 按组件的就绪位与简短 message（如 Deployment 未就绪原因）；字段名与 `spec.components` / P/D 下 proxy 等对应，以实现定义为准。 |
| status.conditions | []Condition | 可选；标准 Condition（如 `Ready`、`SourceAvailable`、`SpecValid`），与 `status.ready` / 各 `components` 互为补充。 |
| InferNexServiceConfig 合并 | — | `InferNexServiceConfig` 不单独触发 reconcile；仅在调和 `InferNexService` 时被读取。 |
##### 使用示例

本提案在集群侧会通过 InferNexServiceController 维护 `InferNexService` 对象：当用户创建 `llmisvc` 并打上 `infernex.io/runtime: true` 等约定标签后，控制器会创建/填充一个 `InferNexService`，用于补齐 InferNex 侧组件；用户也可以直接提交 `InferNexService` 自行声明完整部署。下面给出一个 `InferNexService` 示例，便于理解该 CR 能表达的能力边界。

`sourceRef` 的行为约定如下：

- `spec.sourceRef` 非空：与对应 `llmisvc` 建立关联，控制器可只读访问该 `llmisvc`（例如判断 P/D、删除联动）；不将 `llmisvc.spec` 作为隐式合并源去默认填充 `InferNexService.spec` 的 `model` / `engine` / `intelligentGatewayRouting`，以免与 KServe 对推理引擎与 Router 的 reconcile 重复。补充组件的配置以 `InferNexService` + `InferNexServiceConfig` 模板合并为准。
- `spec.sourceRef` 为空：不关联 `llmisvc`，为直部署路径，完全以 `InferNexService.spec`（及 Config 模板）驱动；此时 `spec.model` 等字段按直部署语义使用。

`baseRefs` 的行为约定如下：

- `spec.baseRefs` 非空：按 `name` 依次 `Get` 各 `InferNexServiceConfig` 并将 `spec.components` / `spec.engine` 按顺序合并进有效 spec（后者填补前者空缺）。

<details>
<summary>点击展开查看 InferNexService YAML配置示例（P/D + 路由 + 组件开关）</summary>

下列示例用于说明 CRD 可表达的能力边界。**若以 `llmisvc` 为主入口**：推理与 Hermes/Gateway 等应在 `llmisvc` / `LLMInferenceServiceConfig` 中声明；`InferNexService` 侧通过 `sourceRef` 关联即可，勿指望从 `llmisvc` 继承 这些字段到本 CR 再 reconcile，以免与 KServe 重复拉起。

```yaml
apiVersion: infernex.infernex.io/v1alpha1
kind: InferNexService
metadata:
  name: qwen3-8b
spec:
  sourceRef:
    apiVersion: serving.kserve.io/v1alpha1
    kind: LLMInferenceService
    name: qwen3-8b
  # 运行模式由 engine 下字段决定：填 engine.pd=PD；填 engine.aggregate=聚合/单路径
  model:
    uri: hf://Qwen/Qwen2.5-7B-Instruct # sourceRef 为空时必填；有 sourceRef 时可省略由 llmisvc 提供
    name: qwen3-8b
  engine:
    pd:
      decode:
        replicas: 2
        parallelism:
          data: 2
          pipeline: 1
        template:
          containers:
            - name: vllm-decode
              image: quay.io/ascend/vllm-ascend:v0.13.0
        worker:
          containers:
            - name: vllm-decode-worker
              image: quay.io/ascend/vllm-ascend:v0.13.0
      prefill:
        replicas: 1
        parallelism:
          tensor: 2
        template:
          containers:
            - name: vllm-prefill
              image: quay.io/ascend/vllm-ascend:v0.13.0
      proxyServer:
        enabled: true
        replicas: 2
        servicePort: 8000
        template:
          containers:
            - name: proxy-server
              image: cr.openfuyao.cn/openfuyao/proxy-server:latest
  intelligentGatewayRouting:
    router:
      enabled: true
      replicas: 2
      template:
        containers:
          - name: router
            image: cr.openfuyao.cn/openfuyao/hermes-router:latest
    httpRoute:
      refs:
        - name: qwen3-8b-route
    gateway:
      refs:
        - name: inference-gateway
    inferencePool:
      refs:
        - name: qwen3-8b-pool
  components:
    cacheIndexer:
      enabled: true
      replicas: 1
      servicePort: 8080
      template:
        containers:
          - name: cache-indexer
            image: cr.openfuyao.cn/openfuyao/cache-indexer:latest
      serviceDiscovery:
        labelSelector: openfuyao.com/engine=vllm
        refreshInterval: 10
    mooncake:
      enabled: true
      master:
        enabled: true
        template:
          containers:
            - name: mooncake-master
              image: hub.oepkgs.net/openfuyao/ascend/vllm-ascend:v0.13.0
      metadataServer:
        enabled: true
        template:
          containers:
            - name: redis
              image: docker.io/library/redis:7-alpine
    pdOrchestrator:
      # 仅包含三个子组件；无顶层 enabled/replicas/template（与 CRD 一致）
      elasticScaler:
        enabled: true
        replicas: 1
        template:
          containers:
            - name: elastic-scaler
              image: cr.openfuyao.cn/openfuyao/elastic-scaler:latest
      tidal:
        enabled: true
        replicas: 1
        template:
          containers:
            - name: tidal
              image: cr.openfuyao.cn/openfuyao/tidal:latest
      resourceScalingGroup:
        enabled: true
        replicas: 1
        template:
          containers:
            - name: resource-scaling-group
              image: cr.openfuyao.cn/openfuyao/resource-scaling-group:latest
    eagleEye:
      hardwareMonitor:
        enabled: true
        replicas: 1
        template:
          containers:
            - name: eagle-eye-hardware-monitor
              image: cr.openfuyao.cn/openfuyao/eagle-eye-hardware-monitor:latest
      hardwareDiagnosis:
        enabled: true
        replicas: 1
        template:
          containers:
            - name: eagle-eye-hardware-diagnosis
              image: cr.openfuyao.cn/openfuyao/eagle-eye-hardware-diagnosis:latest
```
</details>

#### 5. 用户直接提交 `LLMInferenceService` 用例（聚合 / P/D）

本节给出用户直接提交 KServe `LLMInferenceService`（`llmisvc`）的 YAML 用例。适配层在识别到 `infernex.io/runtime: true` 等约定标签后，会由 InferNexServiceController 自动创建/维护 `InferNexService` 以补齐 InferNex 侧组件。

##### 聚合模式
<details>
<summary>点击展开查看 LLMInferenceService 和 LLMInferenceServiceConfig YAML配置示例</summary>

```yaml
---
# LLMInferenceServiceConfig - 聚合模式模板
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceServiceConfig
metadata:
  name: vllm-ascend-with-hermes-and-gateway
  namespace: kserve
spec:
  template:
    containers:
      # ===== vLLM-Ascend 主容器 =====
      - name: main
        image: quay.io/ascend/vllm-ascend:v0.13.0
        imagePullPolicy: IfNotPresent
        ports:
          - name: http
            containerPort: 8000
            protocol: TCP
        command:
          - python3
          - -m
          - vllm.entrypoints.openai.api_server
        args:
          - --model
          - /mnt/models
          - --port
          - "8000"
          - --host
          - "0.0.0.0"
          - --trust-remote-code
        env:
          - name: HOME
            value: /home
          - name: VLLM_LOGGING_LEVEL
            value: INFO
          - name: HF_HUB_CACHE
            value: /models
          # vLLM 优化环境变量
          - name: VLLM_USE_V1
            value: "1"
          - name: GLOO_SOCKET_IFNAME
            value: "eth0"
          - name: HCCL_SOCKET_IFNAME
            value: "eth0"
          # HCCL 优化
          - name: HCCL_WHITELIST_DISABLE
            value: "1"
          - name: HCCL_COMM_ID
            value: "127.0.0.1:5555"
        resources:
          limits:
            cpu: "8"
            memory: 64Gi
            huawei.com/Ascend910: "1"
          requests:
            cpu: "4"
            memory: 32Gi
            huawei.com/Ascend910: "1"
        securityContext:
          privileged: false
          runAsUser: 1000
          runAsGroup: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false
          runAsNonRoot: false
        volumeMounts:
          - mountPath: /home
            name: home
          - mountPath: /dev/shm
            name: dshm
          - mountPath: /models
            name: model-cache
          - mountPath: /root/.cache
            name: root-cache
          - mountPath: /etc/ssl/certs
            name: tls-certs
            readOnly: true
          - mountPath: /etc/passwd
            name: host-passwd
            readOnly: true
          - mountPath: /etc/group
            name: host-group
            readOnly: true
          # === 昇腾必要挂载 ===
          - mountPath: /usr/local/dcmi
            name: dcmi
          - mountPath: /usr/local/bin/npu-smi
            name: npu-smi
          - mountPath: /usr/local/Ascend/driver/lib64
            name: ascend-lib64
          - mountPath: /usr/local/Ascend/driver/version.info
            name: ascend-version
          - mountPath: /etc/ascend_install.info
            name: ascend-installinfo
          - mountPath: /usr/bin/hccn_tool
            name: hccn-tool
          - mountPath: /etc/hccn.conf
            name: hccn-conf

    volumes:
      - name: host-passwd
        hostPath:
          path: /etc/passwd
          type: File
      - name: host-group
        hostPath:
          path: /etc/group
          type: File
      - name: home
        emptyDir: {}
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 24Gi
      - name: model-cache
        emptyDir: {}
      - name: root-cache
        hostPath:
          path: /root/.cache
          type: Directory
      - name: tls-certs
        secret:
          secretName: "{{ ChildName .ObjectMeta.Name `-kserve-self-signed-certs` }}"
      # === 昇腾卷 ===
      - name: dcmi
        hostPath:
          path: /usr/local/dcmi
          type: Directory
      - name: npu-smi
        hostPath:
          path: /usr/local/bin/npu-smi
          type: File
      - name: ascend-lib64
        hostPath:
          path: /usr/local/Ascend/driver/lib64
          type: Directory
      - name: ascend-version
        hostPath:
          path: /usr/local/Ascend/driver/version.info
          type: File
      - name: ascend-installinfo
        hostPath:
          path: /etc/ascend_install.info
          type: File
      - name: hccn-tool
        hostPath:
          path: /usr/bin/hccn_tool
          type: File
      - name: hccn-conf
        hostPath:
          path: /etc/hccn.conf
          type: File

  # ===== Router 配置 =====
  router:
    gateway: {}
    route: {}
    scheduler:
      pool:
        spec:  # 使用 spec 而不是 ref
          selector:
            app.kubernetes.io/name: qwen3-8b-ascend-single
            app.kubernetes.io/component: llminferenceservice-workload
          targetPortNumber: 8000
          extensionRef:
            name: qwen3-8b-ascend-single-epp-service
      template:
        containers:
          # ===== Hermes Router 容器 =====
          - name: main
            image: cr.openfuyao.cn/openfuyao/hermes-router:latest
            imagePullPolicy: IfNotPresent
            ports:
              - containerPort: 9002
                name: grpc
                protocol: TCP
              - containerPort: 9003
                name: grpc-health
                protocol: TCP
              - containerPort: 9090
                name: metrics
                protocol: TCP
            args:
              - "--pool-name"
              - "qwen3-8b-ascend-single-inference-pool"
              - "--pool-namespace"
              - "kserve"
              - "--zap-encoder"
              - "json"
              - "--config-text"
              - |
                apiVersion: inference.networking.x-k8s.io/v1alpha1
                kind: EndpointPickerConfig
                plugins:
                  - type: picker-min-random
                schedulingProfiles:
                  - name: default
                    plugins:
                      - pluginRef: picker-min-random
              - "--v"
              - "4"
              - "--tracing"
              - "false"
              - "--metrics-endpoint-auth"
              - "false"
            volumeMounts:
              - mountPath: /etc/ssl/certs
                name: tls-certs
                readOnly: true
            resources:
              requests:
                cpu: 256m
                memory: 500Mi
            securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              runAsNonRoot: true
              capabilities:
                drop:
                  - ALL
              seccompProfile:
                type: RuntimeDefault
        volumes:
          - name: tls-certs
            secret:
              secretName: "{{ ChildName .ObjectMeta.Name `-kserve-self-signed-certs` }}"
        dnsPolicy: ClusterFirst
        restartPolicy: Always
        terminationGracePeriodSeconds: 30

---
# LLMInferenceService
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: qwen3-8b-ascend-single
  namespace: kserve
  labels:
    infernex.io/runtime: true
spec:
  model:
    uri: hf://Qwen/Qwen3-8B
    name: Qwen/Qwen3-8B

  replicas: 1

  baseRefs:
    - name: vllm-ascend-with-hermes-and-gateway
```
</details>

##### P/D 模式
<details>
<summary>点击展开查看 LLMInferenceService 和 LLMInferenceServiceConfig YAML配置示例</summary>

```yaml
---
# LLMInferenceServiceConfig - 1p1d P/D 模式（1 Prefill + 1 Decode）
# 与 Single-Node 保持一致：镜像、env、volumes、resources、Router。P/D 路由由独立 Proxy Server 做，不用 KServe sidecar。
# - 显式关闭 sidecar：template.initContainers: []，覆盖控制器合并的 llm-d-routing-sidecar
# - Decode / Prefill 均监听 8000，由单独部署的 Proxy Server 做 prefill/decode 分流
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceServiceConfig
metadata:
  name: vllm-ascend-pd-1p1d
  namespace: kserve
spec:
  # ===== Decode 侧：用空转 sidecar 覆盖默认 sidecar，main 监听 8000 =====
  template:
    containers:
      # ===== vLLM-Ascend 主容器（decode）=====
      - name: main
        image: quay.io/ascend/vllm-ascend:v0.13.0
        imagePullPolicy: IfNotPresent
        ports:
          - name: http
            containerPort: 8000
            protocol: TCP
        command:
          - python3
          - -m
          - vllm.entrypoints.openai.api_server
        args:
          - --model
          - /mnt/models
          - --port
          - "8000"
          - --host
          - "0.0.0.0"
          - --trust-remote-code
          - --no-enable-prefix-caching
          - --tensor-parallel-size
          - "1"
          - --kv-transfer-config
          - '{"engine_id": "$(POD_NAME)","kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_buffer_device":"npu","kv_port":"20002","kv_connector_extra_config":{"decode":{"dp_size":1,"tp_size":1},"prefill":{"dp_size":1,"tp_size":1},"use_ascend_direct":true}}'
        env:
          - name: HOME
            value: /home
          - name: VLLM_LOGGING_LEVEL
            value: INFO
          - name: HF_HUB_CACHE
            value: /models
          - name: VLLM_USE_V1
            value: "1"
          - name: GLOO_SOCKET_IFNAME
            value: "eth0"
          - name: HCCL_SOCKET_IFNAME
            value: "eth0"
          - name: HCCL_WHITELIST_DISABLE
            value: "1"
          - name: HCCL_COMM_ID
            value: "127.0.0.1:5555"
          - name: POD_NAME
            valueFrom:
              fieldRef:
                fieldPath: metadata.name
        resources:
          limits:
            cpu: "8"
            memory: 64Gi
            huawei.com/Ascend910: "1"
          requests:
            cpu: "4"
            memory: 32Gi
            huawei.com/Ascend910: "1"
        securityContext:
          privileged: false
          runAsUser: 0
          runAsGroup: 0
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false
          runAsNonRoot: false
        livenessProbe:
          httpGet: { path: /health, port: 8000, scheme: HTTP }
          initialDelaySeconds: 120
          periodSeconds: 10
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet: { path: /health, port: 8000, scheme: HTTP }
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 60
        startupProbe:
          httpGet: { path: /health, port: 8000, scheme: HTTP }
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 10
          failureThreshold: 60

  # ===== Prefill 侧：1 副本，与 Single-Node 主容器一致（端口 8000）=====
  prefill:
    replicas: 1

  # ===== Router 配置（与 Single-Node 一致，仅名称改为 1p1d 对应服务）=====
  router:
    gateway: {}
    route: {}
    scheduler:
      pool:
        spec:
          selector:
            app.kubernetes.io/name: qwen3-8b-ascend-1p1d
            app.kubernetes.io/part-of: llminferenceservice
          targetPortNumber: 8000
          extensionRef:
            name: qwen3-8b-ascend-1p1d-epp-service

---
# LLMInferenceService - 1p1d
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: qwen3-8b-ascend-1p1d
  namespace: kserve
  labels:
    infernex.io/runtime: true
spec:
  model:
    uri: hf://Qwen/Qwen3-8B
    name: Qwen/Qwen3-8B

  replicas: 1

  prefill:
    replicas: 1

  baseRefs:
    - name: vllm-ascend-pd-1p1d
```
</details>

#### 6. 安全设计

本提案采用以下最小安全原则：

- 权限最小化： InferNexService 仅保留基础调和所需的 RBAC 权限；而针对 pd-orchestrator 下属的 Tidal、ElasticScaler 及 RSG 控制器，采取按需分配策略，随组件动态创建/回收专属的 ServiceAccount 与 RBAC 资源，确保各子控制器权限相互隔离且不被长期滥用。

## 测试计划

\[ \] 我/我们理解，相关组件的所有者可能会要求更新已有的测试，以便在提交实现该增强功能所需的更改之前，使代码达到足够稳固的质量标准。

测试大类方向（用例细节在实现阶段再拆；此处只定范围）：

| 测试大类 | 范围要点 |
| --- | --- |
| KServe 与 InferNex-Bridge 协同 | 聚合 / P/D、有或无 `sourceRef`、直接提交 `InferNexService` 等组合下，`llmisvc` 与 `InferNexService`、KServe 与 InferNex 控制面分工正确、无资源争抢。 |
| InferNex-Bridge 独立拉起 | 不依赖 `llmisvc` 入口，仅通过 `InferNexService` 在聚合、P/D 等模式下完整拉起与验证；与带 `sourceRef` 的协同路径对照，覆盖独立 spec 下的 engine 与组件编排。 |
| Webhook | Validating（`llmisvc` 增量规则）与 Mutating（`LLMInferenceServiceConfig` 模版改写、注解门闩防二次 patch）在允许、拒绝、幂等等行为上符合设计。 |
| 状态与可观测 | `InferNexService` 的 `status.mode`、`ready`、`components`、`conditions`、`observedGeneration`；全就绪、部分就绪、失败、源不可用等；与 `llmisvc` 侧状态互补（不重复断言 KServe 已负责的主推理/主路由）。 |
| 运行时与端到端推理 | 拉起并进入就绪后：各 InferNex 补充组件（如 proxy-server、Mooncake、cache-indexer、PD-Orchestrator 等）实际工作正常；与 KServe 推理入口串联，完成经网关/路由到后端的端到端推理。 |
| 生命周期与清理 | `llmisvc` 与同名 `InferNexService` 的创建、更新、删除联动；runtime 标签变更、摘标签等边界；以及资源归属原则应按“谁拉起、谁负责”验证：由 KServe `llmisvc` 控制器拉起的推理后端、Router、Gateway/HTTPRoute 等资源由 KServe 负责生命周期；由 `InferNexService` 控制器拉起的资源（包括补充组件，以及直部署路径下由 `InferNexService` 拉起的推理后端/路由资源）由 `InferNexService` 负责生命周期与回收。需验证一侧源对象删除或失效时，仅回收本侧负责的资源，不误删另一侧控制面已创建资源。 |


##### 单元测试

- Webhook（Validating / Mutating）、InferNexServiceController 核心逻辑建议达到约定覆盖率，即大于80% ：准入分支、sourceRef 合并与覆盖、engine 互斥、plan/apply 与 status/condition 更新、finalizer 分支等。

##### 集成测试

- 在集群或 envtest 中覆盖上述大类中的关键组合：聚合与 P/D、多并行策略、`llmisvc` 触发后 InferNex 组件补齐、路由与推理链路、status 各形态（就绪、部分失败、条件变更）及 Mutating 对 Config 的一次性改写。

##### e2e 测试

- 从集成用例中抽取端到端路径并实现自动化，覆盖多模式组合下的可用性、删除与回收。

