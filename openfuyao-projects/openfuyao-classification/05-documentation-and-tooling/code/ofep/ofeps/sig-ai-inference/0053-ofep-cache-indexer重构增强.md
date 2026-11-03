---
title: oFEP cache-indexer重构增强
ofep-number: 53
authors:
    - "@qinwenzh"
owning-sig: "si-ai-inference"
participating-sigs: []
status: provisional
creation-date: 2026-04-23
reviewers:
    - TBD
approvers:
    - TBD
see-also: []
replaces: []
replaced-by: []
stage: alpha
latest-milestone: ""
milestone:
    alpha: ""
    beta: ""
    stable: ""
feature-gates:
disable-supported: true
metrics: []
---

# oFEP-0053：cache-indexer重构增强

## 1. 发布签核清单

- [ ] (R) 发布里程碑中的增强问题，链接到 [openfuyao/ofep] 中的 oFEP 目录
- [ ] (R) oFEP 审批者已批准 oFEP 状态为"可实施"
- [ ] (R) 设计细节已适当记录
- [ ] (R) 测试计划已到位
- [ ] (R) 毕业标准已设定
- [ ] (R) 生产准备情况审查完成
- [ ] 实施历史已更新里程碑
- [ ] 面向用户的文档已创建

## 2. 摘要

当前 cache-indexer 为 Python/FastAPI 实现，仅支持 L1。内部以"每 server 一棵前缀树" + 进程内 tokenizer 加载 messages 转 token 序列的方式维护 vLLM 本地 KV 缓存视图，对外暴露 `/match_sort`。

本提案目标是将 cache-indexer 转为更适合 K8s 场景的 Go 实现，并同时维护 L1 与 L2 两层视图，对外提供统一的 KVCache 命中率查询接口 `/kv-cache/hit-rate`：

- L1：推理引擎本地 NPU/GPU 显存中的 KVCache。本提案面向 vLLM 设计，通过订阅 vLLM ZMQ KV event获取，设计上保留扩展接口。
- L2：已下沉到分布式缓存管理器内存中的 KVCache。本提案面向 Mooncake 设计，通过轮询 Mooncake HTTP API 获取，设计上保留扩展接口。

L1/L2 在数据来源、key 形态、删除语义上严格解耦，仅在查询接口层并列返回。同时 tokenizer 责任上移到上游，cache-indexer 只接收 `token_ids`。

## 3. 动机

现有版本存在以下问题：

1. **仅支持查询 L1 KVCache 缓存命中率**：路由决策只能依赖各 vLLM 实例本地命中，无法考虑已下沉到内存的跨实例可复用 block。
2. **原前缀树索引设计不适配高频事件与并发查询**：现有 Python 版按 server 维护 `IndexTree` 森林，`TreeNode` 直接存储每个 block 的 `token_ids` 列表，并在写入/删除时通过递归查找父节点或目标节点维护树结构；查询时还要先按 `block_size` 切分请求 token 序列，再在多棵树上做基于 token 内容的 DFS 前缀匹配。该设计一方面需要长期持有大量按 block 切分后的 token 序列，内存占用高；另一方面写入、删除、查询都依赖树遍历，难以在高频 ZMQ 事件 + 跨 server 并发查询场景下稳定控制吞吐与尾延迟。
3. **tokenizer 内置带来热点与扩缩容负担**：请求并发升高时，进程内 tokenizer 会成为 cache-indexer 的 CPU 热点；而 cache-indexer 本身是有状态服务，K8s 场景下扩缩容需要搬迁和重建索引，代价高、不适合靠水平扩容来吸收 tokenizer 带来的额外负载，因此应将 tokenizer 责任上移到无状态上游。

### 3.1 目标

1. 补齐 L2 能力：接入 Mooncake Master，补齐跨实例可复用 block 的命中率查询能力；
2. 统一查询键：复用 vLLM `hash_block_tokens`，统一生成 `BlockHash`；
3. 动态发现服务：提供 K8s Pod 级发现，驱动 L1/L2 目标自动更新；
4. 收敛并发边界：明确模块边界，使用 Go 并发替代 Python GIL；

### 3.2 非目标

1. 不在 cache-indexer 内做 tokenizer，`token_ids` 由上游对齐；不再提供 `messages` 输入形式。
2. 不做 L1/L2 合并存储：查询键统一为 `BlockHash`，但 L1/L2 索引、删除来源、入库语义保持独立。
3. 不替代 vLLM 或 Mooncake 自身的缓存管理逻辑，独立管理一份全局 KVCache 索引信息。
4. 不与与旧 `/match_sort` 请求体按字段同构：输入从 `messages` 改为 `token_ids`，响应从单一分数扩为 L1/L2 并列。

## 4. 提案

### 4.1 用户故事

- 作为用户，我希望根据推理请求对应的 token ID 列表 `token_ids` 查询每个候选 server 的 L1/L2 命中比例，用于路由决策。


@startuml
title cache-indexer use case
left to right direction
skinparam shadowing false
skinparam packageStyle rectangle

actor "用户" as Router

rectangle "cache-indexer" {
        usecase "查询推理请求在各个后端上的L1/L3 KVCache命中率" as UC1
}

Router --> UC1
@enduml


### 4.2 风险与缓解措施

1. **风险**：cache-indexer 与 vLLM 进程之间 `BlockHash` 不一致，导致查询全部 miss。`vllm.v1.core.kv_cache_utils.init_none_hash` 在 `PYTHONHASHSEED` 未设置时回退到 `os.urandom` 生成 `NONE_HASH` 种子，导致每个进程启动时拿到的 `NONE_HASH` 都是不同的随机值；同时 vLLM 默认的 `sha256/pickle` 路径不具备稳定跨语言语义；`--prefix-caching-hash-algo` 不一致同样会破坏位级一致性。

    **缓解**：[§5.10](#510-必要约束) 硬约束——cache-indexer 与所有目标 vLLM Pod 必须共享同一固定 `PYTHONHASHSEED`，并显式固定 `--prefix-caching-hash-algo=sha256_cbor` 或 `xxhash_cbor`；启动期校验；。
2. **风险**：vLLM 升级 `hash_block_tokens` / `--prefix-caching-hash-algo` 默认值后 cache-indexer 未同步，整体 miss。

   **缓解**：锁定 vLLM 版本范围；在发布说明列出必须同步的 vLLM commit；CI 自动比对。
3. **风险**：K8s 轮询发现存在感知延迟，导致已下线 Pod 的索引残留。

   **缓解**：discovery diff 触发 unsubscribe 并清理对应 server 的 L1 索引；L2 由 `/get_all_keys` diff 自然收敛；保留固定刷新间隔配置。

## 5. 设计细节
### 5.1 统一查询接口

```text
POST /kv-cache/hit-rate
```

请求体：

```json
{
  "server_ip": ["10.250.49.aa", "10.250.49.bb"],
  "body": {
    "token_ids": [151644, 8948, 198, 872, 25, 9707, 11, 1879, 0],
    "cache_salt": "",
    "block_size": 128
  }
}
```

请求体参数说明：

| 参数名 | 描述 | 示例 |
|--------|------|------|
| `server_ip` | 候选 server 的 IP 列表，作为查询主键，由路由层传入 | `["10.250.49.34", "10.250.49.35"]` |
| `body.token_ids` | 已由上游 tokenizer 完成编码的 token ID 序列，cache-indexer 不再做 tokenize | `[151644, 8948, 198, 872, ...]` |
| `body.cache_salt` | 与对应 vLLM 推理请求实际使用的 `cache_salt` 一致的多租户隔离盐值，作为 `BlockHash` 计算输入（vLLM `hash_block_tokens` 的 `extra_keys` 首位）；若对应 vLLM 请求未设置该字段，则这里传空串 | `""` |
| `body.block_size` | 与 vLLM `--block-size` 一致的 KV block 大小（token 数），写入侧/查询侧必须一致 | `128` |

响应体：

```json
{
  "server_score_list": [
    {
      "server_ip": "10.250.49.aa",
      "l1_hit_ratio": 0.75,
      "l3_hit_ratio": 0.50,
    },
    {
      "server_ip": "10.250.49.bb",
      "l1_hit_ratio": 0.25,
      "l3_hit_ratio": 0.50,
    }
  ],
  "message": "",
  "status": 0
}
```

响应体参数说明：

| 参数名 | 描述 | 示例 |
|--------|------|------|
| `server_score_list[].server_ip` | 与请求 `server_ip` 一一对应的 server IP | `"10.250.49.34"` |
| `server_score_list[].l1_hit_ratio` | L1（vLLM 本地 GPU/HBM）最长连续前缀命中 block 数 / `total_blocks`，范围 `[0,1]` | `0.75` |
| `server_score_list[].l3_hit_ratio` | L2（Mooncake 内存）最长连续前缀命中 block 数 / `total_blocks`，范围 `[0,1]` | `0.50` |
| `message` | 业务侧错误/提示信息，正常返回为空串 | `""` |
| `status` | 业务状态码，`0` 表示成功，非 0 表示业务错误（HTTP 状态码仍按标准语义返回） | `0` |

此外，cache-indexer 提供两个健康检查端点用于 K8s liveness/readiness probe：

- `GET /healthz` — 返回 `{"status": "ok"}`，用于 liveness probe
- `GET /readyz` — 返回 `{"status": "ok"}`，用于 readiness probe

### 5.2 总体架构

下图是推理服务整体架构，重点展示cache-indexer与其他推理组件的关系：

- **请求主链路**：用户请求先进入网关入口，再由其中的 Hermes-router EPP 调用 cache-indexer 查询所有可用 server 的 L1/L2 命中信息，完成调度后将请求转发到 PD Leader/Proxy Pod，并进入 Prefill 阶段。
- **cache-indexer**：cache-indexer 通过 Kubernetes API Server 动态更新 server pod 列表；异步订阅 Prefill Pod 的 ZMQ KV event 构建 L1；轮询 Mooncake Master 构建 L2。


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


### 5.3 逻辑视图

下图为cache-indexer，组件间通过 Go interface 暴露的依赖方向


@startuml
title cache-indexer 逻辑视图
skinparam componentStyle rectangle
skinparam shadowing false
skinparam nodesep 12
skinparam ranksep 18
skinparam component {
    BackgroundColor<<query>> #C8E6C9
    BackgroundColor<<index>> #BBDEFB
    BackgroundColor<<ingest>> #FFE0B2
    BackgroundColor<<discovery>> #E1BEE7
}
left to right direction

' 左侧：查询面（内部垂直）
package "查询面" as PKG_QUERY {
    [HTTP Handler] as HTTP <<query>>
    [ScoreService] as SCORE <<query>>
    [BlockKeyBuilder] as BKB <<query>>
    HTTP -[hidden]down- SCORE
    SCORE -[hidden]down- BKB
}

' 右侧：服务发现面（内部垂直）
package "服务发现面" as PKG_DISC {
    [DiscoveryManager] as DM <<discovery>>
    [DiscoveryReconciler] as REC <<discovery>>
    DM -[hidden]down- REC
}

' 右侧：采集面（内部水平）
package "采集面" as PKG_INGEST {
    [L1SubscriptionManager] as L1SUB <<ingest>>
    [MooncakePollTargetUpdater] as L3POLL <<ingest>>
    L1SUB -[hidden]right- L3POLL
}

' 右侧：索引面（内部水平）
package "索引面" as PKG_INDEX {
    [L1Indexer] as L1 <<index>>
    [L3Indexer] as L3 <<index>>
    L1 -[hidden]right- L3
}

' 查询面内部
HTTP  ..> SCORE : ScoreServers
SCORE ..> BKB   : BuildFromRequest

' 查询面 -> 索引面
SCORE ..> PKG_INDEX : L1/L3 MatchPrefix

' 服务发现面内部
DM ..> REC : Reconcile

' 服务发现面 -> 采集面
REC ..> L1SUB  : Subscribe /\nUnsubscribe
REC ..> L3POLL : UpdateTarget

' 服务发现面 -> 索引面（下线清理）
REC ..> L1 : ClearServer

' 采集面 -> 索引面（写入）
L1SUB  ..> L1 : UpsertStoredEvent /\nRemoveBlocks
L3POLL ..> L3 : ApplyReplicaSnapshot /\nRemoveReplicaKeys

' 强制右侧三个 package 自上而下排列
PKG_DISC   -[hidden]down- PKG_INGEST
PKG_INGEST -[hidden]down- PKG_INDEX
@enduml


模块边界硬约束：

1. **查询面**：查询路径严格单向：`HTTP Handler → Service.Compute → BlockKeyBuilder.BuildFromRequest → L1Indexer/L3Indexer.MatchPrefix`。HTTP 层不直连 Indexer/Builder；查询侧与索引侧只通过 `MatchPrefix` 交互，统一键类型仅为 `BlockHash`。
2. **服务发现面**：K8s 访问全部收敛到 Discovery。仅 `Manager` 持 K8s client，输出 `Snapshot` 给 `Reconciler.Reconcile`；`L1SubscriptionManager` 与 `MooncakePollTargetUpdater` 作为下游订阅者，由 Reconciler 反向驱动。
3. **采集面**：采集写入不经过 `BlockKeyBuilder`。L1 由 `L1SubscriptionManager` 归一化 ZMQ `block_hashes` 后直调 `L1Indexer.UpsertStoredEvent` / `RemoveBlocks`；L3 由 `MooncakePollTargetUpdater` 取 Mooncake key 末段为 `BlockHash` 后直调 `L3Indexer.ApplyReplicaSnapshot` / `RemoveReplicaKeys`。
4. **索引面**：`L1Indexer` 与 `L3Indexer` 互不可见，禁止跨层读/写/删。下线清理保持非对称：L1 由 `Reconciler` 显式调 `L1Indexer.ClearServer`；L2 不暴露 `ClearServer`，通过 `MooncakePollTargetUpdater.UpdateTarget` 更新。

接口定义见 [§5.7.4](#574-接口)。
### 5.4 端到端时序图

上游路由（hermes-router/EPP）发起一次 `POST /kv-cache/hit-rate`，cache-indexer 复算 key、并行查询 L1/L2、组装并返回。L1/L2 索引由 [§5.5.1](#551-时序图) / [§5.6.1](#561-时序图) 的后台流持续填充。

1. HTTP 层只入 `Service`，不直连任何 Indexer / Builder（[§5.3](#53-逻辑视图) 硬约束 #2）。
2. L1/L2 查询 key 统一为 `BlockHash`，由同一 `BlockKeyBuilder` 产出；其内部按 vLLM `get_request_block_hasher` 的语义在 Go 中复刻 `BlockHash` 计算流程，并显式固定 `sha256_cbor` 或 `xxhash_cbor` 路径，与事件侧 / Mooncake key 末段位级一致（[§5.10](#510-必要约束)）。
3. L2 查询前，`Service` 把候选 vLLM Pod 解析为同 pod Mooncake client 的 `transport_endpoint_`；该值通过 Mooncake Master `/get_all_segments` 反查得到。若解析失败，则该 server 的 L2 命中直接按 0。
4. `server_score_list` 严格按入参 `server_ip` 顺序返回，不在 cache-indexer 内排序（[§5.1](#51-统一查询接口)）。


@startuml
!theme plain
title cache-indexer 查询主链路（/kv-cache/hit-rate）

actor "Router\n(hermes-router/EPP)" as Router
participant "HTTP Handler\n/kv-cache/hit-rate" as HTTP
participant "ScoreService" as SCORE
participant "BlockKeyBuilder" as BKB
participant "DiscoveryRegistry" as DISC
participant "L1Indexer" as L1
participant "L3Indexer" as L3

Router -> HTTP: POST /kv-cache/hit-rate\n{server_ip[], body{model, token_ids,\n cache_salt, block_size}}
activate HTTP
HTTP -> SCORE: ScoreServers(modelCtx, serverIPs, tokenIDs)
activate SCORE

SCORE -> BKB: BuildFromRequest(modelCtx, tokenIDs)
BKB --> SCORE: []BlockHash\n（按 vLLM get_request_block_hasher 语义\n在 Go 中复刻 + cache_salt）

loop for serverIP in 入参顺序
    SCORE -> L1: MatchPrefix(modelCtx, serverIP, blockKeys)
    L1 --> SCORE: MatchResult\n(matched_blocks)
    SCORE -> DISC: ResolveTransportEndpoint(serverIP)
    alt endpoint resolved
        DISC --> SCORE: transportEndpoint
        SCORE -> L3: MatchPrefix(modelCtx, transportEndpoint, blockKeys)
        L3 --> SCORE: MatchResult\n(matched_blocks)
    else endpoint missing
        DISC --> SCORE: not found
        SCORE -> SCORE: L2 matched_blocks = 0
    end
    SCORE -> SCORE: 组装 ServerScore\n(l1/l3_hit_ratio, l1/l3_matched_blocks,\n total_blocks)
end

SCORE --> HTTP: []ServerScore
deactivate SCORE
HTTP --> Router: 200 OK\n{server_score_list[], message:"", status:0}
deactivate HTTP

@enduml


### 5.5 L1 设计

约定：设候选 server 为 $s$，请求 token 序列经 `BlockKeyBuilder` 切分为 $N$ 个 block，得到键序列 $K = (k_1, \dots, k_N)$，$k_i \in \text{BlockHash}$。

本节首先介绍 L1 时序图以明确整体流程，而后对订阅、索引结构、写入、计算命中、删除几个关键能力进行介绍。

#### 5.5.1 时序图

L1 走 ZMQ 事件订阅，`L1SubscriptionManager` 收到 `KVEventBatch` 后按事件类型分派给 `L1Indexer`。


@startuml
!theme plain
title cache-indexer L1 数据采集时序

participant "vLLM Pod\n(ZMQ PUB)" as VLLM
participant "L1SubscriptionManager" as L1SUB
participant "L1Indexer" as L1

VLLM -> L1SUB: ZMQ KVEventBatch
activate L1SUB

alt BlockStored
    L1SUB -> L1SUB: normalize block_hashes\n(bytes|int) → []BlockHash
    L1SUB -> L1: UpsertStoredEvent(event)
    note right of L1
      更新 l1KeyIndex / l1ServerIndex
      （以 BlockHash 为键）
      仅 medium=="GPU" 计入命中
    end note
else BlockRemoved
    L1SUB -> L1SUB: normalize block_hashes\n→ []BlockHash
    L1SUB -> L1: RemoveBlocks(event)
    note right of L1: 仅删当前 server 的 L1 项
else AllBlocksCleared
    L1SUB -> L1: RemoveBlocks(全量)
    note right of L1: 清当前 server 全部 L1 项
end

deactivate L1SUB
@enduml


#### 5.5.2 订阅

由 `Reconciler` 通过 `L1SubscriptionManager.Subscribe / Unsubscribe` 驱动 ZMQ SUB 端点上下线（详见 [§5.7](#57-k8s-动态服务发现)）。每次收到 `KVEventBatch` 后，按下面顺序处理：

1. 先按事件类型分流，区分 `BlockStored`、`BlockRemoved`、`AllBlocksCleared`。
2. 对 `BlockStored` 和 `BlockRemoved`，先把事件里的 `block_hashes` 归一化为统一 hex 形式的 `BlockHash`。
3. 把归一化后的结果连同 `serverIP`、`medium`、`group_idx` 等字段交给 `L1Indexer.UpsertStoredEvent` 或 `RemoveBlocks`。
4. 对 `AllBlocksCleared`，直接触发当前 server 的全量删除路径。

补充说明：

- `BlockStored` 关键字段包括 `block_hashes`、`parent_block_hash`、`token_ids`、`block_size`、`medium`、`lora_id` / `lora_name`、`extra_keys`、`group_idx`；`BlockRemoved` 包括 `block_hashes`、`medium`、`group_idx`；`AllBlocksCleared` 仅表示当前 server 全量清空。
- `block_hashes` 在 vLLM 里是 `list[ExternalBlockHash]`，长度 $N$ 为本批 block 数；每个元素只表示一个 block 的 hash，不携带 token 内容。
- 同一 `BlockStored` 事件中，`token_ids` 长度恰为 $N \times \text{block\_size}$，与 `block_hashes` 按下标一一对应；单个 hash 由该 block 的 token、`parent_block_hash` 和 `extra_keys[i]` 共同计算，首块 `parent_block_hash=None` 对应 `init_none_hash` 种子。
- 归一化规则如下：

$$
	ext{block\_hashes} \in \{\text{bytes} \mid \text{int}\} \xrightarrow{\text{normalize}} \text{BlockHash} \in \text{hex}
$$

- `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=True` 时，每个元素是截断后的 `int`，归一化时按大端写回 8 字节再转 16 位 hex；为 `False` 时，每个元素是完整摘要 `bytes`，归一化时直接 `bytes.hex()`。
- 查询侧 `BlockKeyBuilder` 必须与事件侧 `block_hashes` 规格严格对齐，保证两边得到的 `BlockHash` 完全一致。本提案默认采用 `True` 分支，并在 [§5.10](#510-必要约束) 中统一约束。
- vLLM `BlockPool` 把 `medium` 硬编码为 `MEDIUM_GPU = "GPU"`，vllm-ascend 未重写 `BlockPool`，故 NPU 后端事件 `medium` 仍为 `"GPU"`。

#### 5.5.3 索引结构

L1 用两张内存 map 互为倒排，一张以 block 为主键、一张以 server 为主键，同一份 `L1ServerState` 被两边共享指针指向，一处更新即两处可见：

```go
// 主键：BlockHash —— 回答“这个 block 当前在哪些 server 上？”
// 用于查询、全局存在性判断、回收空 entry
l1KeyIndex    map[BlockHash]*L1KeyEntry
// 主键：serverIP —— 回答“这个 server 当前持有哪些 block？”
// 用于按 server 写入 / 删除 / 整体清理（ClearServer）
l1ServerIndex map[string]map[BlockHash]*L1ServerState

type L1KeyEntry struct {
    ServerStates map[string]*L1ServerState // serverIP -> 与 l1ServerIndex 同一指针
}
type L1ServerState struct {
    MediumSet     map[string]struct{}
    DPRankSet     map[int64]struct{}
    GroupIndexSet map[int64]struct{}
}
```

变量说明：

| 变量名 | 类型 | 含义 |
|--------|------|------|
| `l1KeyIndex` | `map[BlockHash]*L1KeyEntry` | 以 `BlockHash` 为键的正向索引，回答“这个 block 在哪些 server 上” |
| `l1ServerIndex` | `map[string]map[BlockHash]*L1ServerState` | 以 `serverIP` 为键的反向索引，回答“这个 server 持有哪些 block” |
| `L1KeyEntry` | `struct` | 单个 `BlockHash` 的聚合条目 |
| `ServerStates` | `map[string]*L1ServerState` | 某个 block 在各个 `serverIP` 上的状态表 |
| `L1ServerState` | `struct` | 单个 `(serverIP, BlockHash)` 组合上的状态 |
| `MediumSet` | `map[string]struct{}` | 该 block 在该 server 上出现过的介质集合 |
| `DPRankSet` | `map[int64]struct{}` | 该 block 关联过的 DP rank 集合 |
| `GroupIndexSet` | `map[int64]struct{}` | 该 block 关联过的 `group_idx` 集合 |

两张表的分工与典型读路径：

| 路径 | 入口 map | 用途 |
|------|----------|------|
| `MatchPrefix(serverIP, keys)` | `l1ServerIndex[serverIP]` | 按 server 逐 key lookup，算最长连续命中前缀 |
| `RemoveBlocks` / `ClearServer` | `l1ServerIndex[serverIP]` | 直接定位该 server 持有的 block 集合做删除 |
| 全局存在性 / 回收 | `l1KeyIndex[k]` | 当 `ServerStates` 空时把 `k` 从 `l1KeyIndex` 删除 |

> 之所以不只用一张 `l1ServerIndex`：删除某 block 的最后一个 server 时需要 O(1) 判定该 block 是否还存在于全局，否则得遍历所有 server。`l1KeyIndex` 就是这张正向索引。

#### 5.5.4 写入（`UpsertStoredEvent`）

输入归一化后的 `L1StoredEvent` 后，按下面顺序写入：

1. 先按 `ModelContext = (model, lora, block_size, cache_salt)` 定位到对应上下文。
2. 对事件里的每个 `BlockHash` $k$，在 `l1ServerIndex[s][k]` 上合并本次事件携带的 `medium`、`dp_rank`、`group_idx`；若 entry 不存在则新建 `L1ServerState`。
3. 把同一份 `*L1ServerState` 指针同步写入 `l1KeyIndex[k].ServerStates[s]`，保持两张倒排一致。

补充说明：

- `L1ServerState` 内部三张 set 分别记录 `medium`、`dp_rank`、`group_idx` 的并集，不重复保存同一值。
- `medium == "GPU"` 视为 L1 HBM 存在标记，是后续命中判定的唯一依据。

#### 5.5.5 计算命中（`MatchPrefix`）

查询侧给定候选 server $s$（Pod IP）和该请求的键序列 $K = (k_1, \dots, k_N)$（$N$ = 请求被 `BlockKeyBuilder` 切成的 block 数，$k_i$ 是第 $i$ 个 block 的 `BlockHash`）。

计算时按下面顺序执行：

1. 先定位 `l1ServerIndex[s]`，取得候选 server 当前持有的 block 视图。
2. 从 $k_1$ 开始顺序检查每个 `BlockHash` 是否存在，且对应状态里的 `MediumSet` 是否包含 `"GPU"`。
3. 连续命中则继续，直到首个未命中位置为止。

单 block 命中判定如下：在 $s$ 上存在且 `medium` 含 `"GPU"`。

$$
\text{hit}_1(s, i) \iff k_i \in l1ServerIndex[s] \,\land\, \text{GPU} \in l1ServerIndex[s][k_i].\text{MediumSet}
$$

最长连续命中前缀长度 $m_1(s)$（从 $k_1$ 起连续命中的 block 个数，中间任一个未命中即截断）：

$$
m_1(s) = \max\{ m \in [0, N] \mid \forall i \le m,\ \text{hit}_1(s, i) \}
$$

L1 命中率 = $m_1(s) / N$，写入 `MatchResult.MatchedBlocks = m_1(s)`。

补充说明：

- L1 命中判定同时依赖“存在性”和 `MediumSet` 中是否包含 `"GPU"`；仅存在但不含 `"GPU"` 不计命中。

#### 5.5.6 删除（`RemoveBlocks` / clear）

删除时按下面顺序执行：

1. 若事件是 `BlockRemoved`，则从 `l1ServerIndex[s]` 删除事件携带的 `BlockHash` 集合。
2. 若事件是 `AllBlocksCleared` 或外部触发 `ClearServer`，则清空 `l1ServerIndex[s]` 整个子表。
3. 完成后反向同步 `l1KeyIndex`：在 `l1KeyIndex[k].ServerStates` 中删掉对应 $s$；若 `ServerStates` 变空，则把该 `k` 整条回收。

补充说明：

- 删除路径只作用于 L1，不可触达 L2（[§5.3](#53-逻辑视图) 硬约束 #1）。

### 5.6 L2 设计

约定沿用 [§5.5](#55-l1-设计)：候选 server $s$、键序列 $K = (k_1, \dots, k_N)$。

L2 面向 Mooncake 分布式缓存管理的内存层 KVCache，通过周期轮询 Mooncake Master 获取 block 副本信息并维护索引。L3（SSD 层）当前暂不支持。

本节首先介绍 L2 时序图以明确整体流程，而后对采集、索引结构、写入、计算命中、删除几个关键能力进行介绍。

#### 5.6.1 时序图

L2 走 HTTP 周期轮询（拉），`MooncakePollTargetUpdater` 内嵌的 poller 按 `PollInterval` 拉 Mooncake Master 全量 key 并 diff。


@startuml
!theme plain
title cache-indexer L2 数据采集时序

participant "Mooncake Poller" as L3POLL
participant "Mooncake Master" as MC
participant "L3Indexer" as L3

loop 每 PollInterval
    L3POLL -> MC: GET /get_all_keys
    activate L3POLL
    MC --> L3POLL: keys
    L3POLL -> L3POLL: 与上一轮 diff\n→ removed / refresh_set
    opt refresh_set 非空
        L3POLL -> MC: GET /batch_query_keys(refresh_set)
        MC --> L3POLL: replicas
        L3POLL -> L3POLL: 截取 key 末段 → BlockHash\n读取 values[].transport_endpoint_
        L3POLL -> L3: ApplyReplicaSnapshot(records)
    end
    opt removed 非空
        L3POLL -> L3: RemoveReplicaKeys(keys)
    end
    deactivate L3POLL
end
@enduml


#### 5.6.2 采集与键归一化

`MooncakePollTargetUpdater.UpdateTarget` 控轮询目标。每轮采集按下面顺序执行：

1. 调用 `GET /get_all_keys`，获取当前 Mooncake Master 上的全量 key 集合。
2. 对每个 key 直接按 `@` 右切取末段 hex，归一化为 `BlockHash`；同时与上一轮结果做 diff，得到 `removed` 与 `refresh_set`。
3. 对 `refresh_set` 调用 `GET /batch_query_keys(refresh_set)`，读取每个 key 的副本详情，并从 `values[].transport_endpoint_` 提取 block 当前所在 Mooncake client 的真实位置，转成 `L3ReplicaRecord` 写入 `ApplyReplicaSnapshot`。
4. 对 `removed` 调用 `RemoveReplicaKeys`，回收本轮已不存在的 key。

补充说明：

- Mooncake key 形如 `Qwen3-8B@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@<hex>`，末段 hex 即 vLLM `hash_block_tokens` 产物，因此无须额外映射即可得到 `BlockHash`：

$$
	ext{BlockHash} = \text{rsplit}(\text{mooncakeKey}, \text{"@"}, 1)[-1]
$$

- `GET /get_all_keys` 只说明“有哪些 block”，**不包含** block 当前所在 Mooncake client 的真实位置；真实位置必须靠 `GET /batch_query_keys` 返回的 `values[].transport_endpoint_` 获取。
- `/batch_query_keys` 的 `data[key].values[]` 中，每个 element 都是一个 memory replica 描述，至少包含 `size_`、`buffer_address_`、`protocol_`、`transport_endpoint_`。
- `transport_endpoint_` 形如 `host:port`，代表当前 block 所在 Mooncake client 的本地传输端点；该值由 Mooncake client 启动配置的 `MOONCAKE_LOCAL_HOSTNAME` 决定。本设计将 `transport_endpoint_` 作为 L2 索引的 server 归属维度，原始字段从 `L3ReplicaRecord.TransportEndpoint` 流入（见 [§5.12](#512-核心数据模型)）。

#### 5.6.3 索引结构

L3 与 L1 同形（两张 map 互为倒排，共享 `*L3ServerState` 指针），仅状态字段不同；不暴露 `ClearServer`，server 整体下线靠 poller diff 收敛：

```go
l3KeyIndex    map[BlockHash]*L3KeyEntry                          // BlockHash -> 该 block 在哪些 transport_endpoint 上有副本
l3ServerIndex map[TransportEndpoint]map[BlockHash]*L3ServerState // transport_endpoint -> 该 endpoint 持有的 block

type L3KeyEntry struct {
    ServerStates map[TransportEndpoint]*L3ServerState
}
type L3ServerState struct {
    ReplicaTypeSet map[string]struct{}
}
```

变量说明：

| 变量名 | 类型 | 含义 |
|--------|------|------|
| `l3KeyIndex` | `map[BlockHash]*L3KeyEntry` | 以 `BlockHash` 为键的正向索引，回答“这个 block 在哪些 endpoint 上有副本” |
| `l3ServerIndex` | `map[TransportEndpoint]map[BlockHash]*L3ServerState` | 以 `TransportEndpoint` 为键的反向索引，回答“这个 endpoint 持有哪些 block” |
| `L3KeyEntry` | `struct` | 单个 `BlockHash` 的聚合条目 |
| `ServerStates` | `map[TransportEndpoint]*L3ServerState` | 某个 block 在各个 `TransportEndpoint` 上的状态表 |
| `L3ServerState` | `struct` | 单个 `(TransportEndpoint, BlockHash)` 组合上的状态 |
| `ReplicaTypeSet` | `map[string]struct{}` | 该 `(TransportEndpoint, BlockHash)` 已写入过的 `ReplicaType` 字符串集合 |

#### 5.6.4 写入（`ApplyReplicaSnapshot`）

输入 `[]L3ReplicaRecord` 后，逐条按下面顺序写入：

1. 从当前 record 取出 `e = record.TransportEndpoint` 与 `k = record.BlockKey`。
2. 在 `l3ServerIndex[e][k]` 上合并本条携带的 `ReplicaType`；若 entry 不存在则新建 `L3ServerState`。
3. 把同一份 `*L3ServerState` 指针同步写入 `l3KeyIndex[k].ServerStates[e]`，保持两张倒排一致。

补充说明：

- `e` 必须取 `record.TransportEndpoint`，**禁止**用 vLLM pod IP 直接作为 key；否则同一 vLLM pod 与其本地 Mooncake client 的 endpoint 不一致时，会把实际存在的 L2 副本查成 0 命中。
- 写入语义与 L1 对称，区别只在于 L2 状态只维护 `ReplicaTypeSet`。

#### 5.6.5 计算命中（`MatchPrefix`）

查询侧给定候选 vLLM server `serverIP`（pod 维度）和键序列 $K = (k_1, \dots, k_N)$（同 L1，$N$ = block 数，$k_i$ = 第 $i$ 个 block 的 `BlockHash`）。

计算时按下面顺序执行：

1. 先经 `Registry` 解析该 `serverIP` 对应的 Mooncake client endpoint，得到 $e = \text{ResolveTransportEndpoint}(serverIP)$。
2. 若解析失败（同 pod 内未部署 Mooncake client，或映射缺失），则该 server 的 L2 命中直接判 0。
3. 若解析成功，则从 $k_1$ 开始顺序检查 `l3ServerIndex[e]` 中是否存在对应 `BlockHash`；连续命中则继续，直到首个未命中位置为止。

单 block 命中判定如下。L2 不看 `medium`，仅看存在性：

$$
    	ext{hit}_3(serverIP, i) \iff k_i \in l3ServerIndex[\text{ResolveTransportEndpoint}(serverIP)]
$$

最长连续命中前缀长度 $m_3(s)$（同 L1，中间断即截）：

$$
m_3(s) = \max\{ m \in [0, N] \mid \forall i \le m,\ \text{hit}_3(s, i) \}
$$

L2 命中率 = $m_3(s) / N$，写入 `MatchResult.MatchedBlocks = m_3(s)`。

补充说明：

- L2 命中判定只看 block 是否存在于 `l3ServerIndex[e]`，不再区分 `medium`。
- 这里的 endpoint 解析是 L2 查询前置条件；因此查询侧按 `serverIP` 入参打分，索引侧仍按 `transport_endpoint_` 存储，两者通过 `Registry` 衔接。

#### 5.6.6 删除（`RemoveReplicaKeys`）

- 从 `l3ServerIndex[s]` 删除本轮 diff 得到的 `BlockHash` 集合；`l3KeyIndex` 侧同步调整，`ServerStates` 空则整条回收。
- 不暴露 `ClearServer`：server 整体下线靠 `MooncakePollTargetUpdater.UpdateTarget` 切目标 + 下一轮 poller diff 自然收敛（[§5.3](#53-逻辑视图) 硬约束 #5）。
- 不可触达 L1（[§5.3](#53-逻辑视图) 硬约束 #1）。

### 5.7 K8s 动态服务发现

该模块目标是发现两类服务的Pod：
- vLLM Pod
- Mooncake Master Pod

这两种服务 Pod 分别喂给 L1 订阅器与 L2 轮询器。
> Pod 级而非 Service 级，是因为下游需要 Pod IP + 容器端口。

#### 5.7.1 时序图

`Manager` 是唯一直连 K8s 的模块，通过 `Reconciler` 把 snapshot diff 翻译成对下游的副作用调用，使下游实际状态向期望状态收敛。


@startuml
!theme plain
title cache-indexer 服务发现时序（5.7）

participant "K8s API Server" as K8S
participant "DiscoveryManager" as DM
participant "DiscoveryReconciler" as REC
participant "L1SubscriptionManager" as L1SUB
participant "MooncakePollTargetUpdater" as L3UPD
participant "L1Indexer" as L1

loop 每 RefreshInterval
    DM -> K8S: List Pods(namespace)
    activate DM
    K8S --> DM: PodList
    DM -> DM: 过滤 phase==Running && Ready\n按硬标签筛选：\n openfuyao.com/engine=vllm\n + pdRole∈{prefill,aggregate}\n openfuyao.com/kvmanager=mooncake
    DM -> DM: 解析端口（zmq-pub / http-api /\n mooncake HTTP+RPC，含回退）
    DM -> DM: 生成 Snapshot\n与上一轮 diff
    DM -> REC: Reconcile(prev, next)
    activate REC

    alt 新增 vLLM Pod
        REC -> L1SUB: Subscribe(VLLMPodEndpoint)
    else 下线 vLLM Pod
        REC -> L1SUB: Unsubscribe(podIP)
        REC -> L1: ClearServer(modelCtx, podIP)
        note right of L1: L1 显式清理；\nL2 不经 Reconciler，\n由 poller diff 自然收敛
    end

    opt Mooncake Master 端点变化
        REC -> L3UPD: UpdateTarget(MooncakeMasterEndpoint)
    end

    deactivate REC
    deactivate DM
end

@enduml


#### 5.7.2 配置要求

```go
type DiscoveryConfig struct {
    Namespace       string
    RefreshInterval time.Duration

    VLLM struct {
        ZMQPortName string // 默认 zmq-pub
    }

    MooncakeMaster struct {
        HTTPPortName string
        RPCPortName  string
        HTTPPort     int32 // 兜底，建议默认 9003
        RPCPort      int32
    }
}
```

标签发现规则：

1. vLLM Pod 必须同时满足：
   - `openfuyao.com/engine=vllm`（当前唯一合法取值为`vLLM`）；
   - `openfuyao.com/pdRole` 取值为 `prefill` 或 `aggregate`（仅这两类参与 L1 订阅；`decode` 等其他取值忽略）。
2. Mooncake Master Pod 必须满足：
   - `openfuyao.com/kvmanager=mooncake`（当前唯一合法取值为 `mooncake`）。

不满足上述标签组合的 Pod 一律不进入 `Snapshot`。

同 pod Mooncake client 的 `local_hostname` / `transport_endpoint_` 发现通过 Mooncake Master `/get_all_segments` 反查实现：cache-indexer 调用该接口获取所有 Mooncake client 的 `transport_endpoint_` 列表及其 IP 信息，与 vLLM Pod IP 交叉匹配得到同 pod 的 endpoint；若匹配失败则记为 unresolved，L2 查询对该 Pod 直接判 0 命中。

部署协同要求：

- Mooncake Master 必须支持 `/get_all_segments` 接口，该接口返回的 segment 信息需包含 client IP 与 `transport_endpoint_` 的映射关系。

#### 5.7.3 数据结构

```go
type PodIdentity struct {
    Namespace, Name, UID, ResourceVersion string
}

type MooncakeClientEndpoint struct {
    TransportEndpoint TransportEndpoint
    Host              string
    TransferPort      int32
    Resolved          bool
    Source            string // master-segment | unresolved
}

type VLLMPodEndpoint struct {
    Identity       PodIdentity
    PodIP          string
    ZMQPort        int32
    Labels         map[string]string
    MooncakeClient MooncakeClientEndpoint
    Ready          bool
    LastSeenAt     time.Time
}

type MooncakeMasterEndpoint struct {
    Identity   PodIdentity
    PodIP      string
    HTTPPort   int32
    RPCPort    int32
    Labels     map[string]string
    Ready      bool
    LastSeenAt time.Time
}

type Snapshot struct {
    Namespace         string
    Version           uint64
    RefreshedAt       time.Time
    VLLMPods          map[string]*VLLMPodEndpoint // key: PodIP
    MooncakeMasterPod *MooncakeMasterEndpoint
}
```

`Registry` 在内存中至少维护两张只读快照索引：

```go
byServerIP          map[string]*VLLMPodEndpoint
serverToMooncakeEP  map[string]TransportEndpoint // 仅收录 MooncakeClient.Resolved=true 的 pod
```

这样查询路径拿到 `serverIP` 后，可 O(1) 解析出同 pod 的 Mooncake client endpoint；若 `serverToMooncakeEP` 无记录，则该 server 的 L2 命中直接按 0 处理。

#### 5.7.4 接口

涉及 query / index / ingest / discovery 的关键接口定义统一见 [§5.8](#58-关键接口定义)。

`Reconcile` 调谐 snapshot diff：

1. 新发现的 vLLM pod → L1 ZMQ subscribe。
2. 下线的 vLLM pod → unsubscribe + `L1Indexer.ClearServer(podIP)`。
3. mooncake master pod 变化 → 更新 L2 poller 目标。
4. 任一 vLLM pod 的 `MooncakeClient.TransportEndpoint` 变化 → `Registry.Update(snapshot)` 原子替换 `serverIP -> transport_endpoint` 映射；该变化**不**触发 L2 全量重建，后续查询自然按新映射命中。

#### 5.7.5 轮询步骤

按固定刷新间隔轮询 K8s API Server，不接入 informer/watch：

1. `List Pods(namespace)`。
2. 过滤 `phase == Running` 且 `Ready == true`。
3. 按 label selector OR 语义筛 vLLM pods。
4. 从容器端口解析 `zmq-pub` / `http-api`。
5. 对每个 vLLM pod，调用 Mooncake Master `/get_all_segments` 获取 transport_endpoint 到 IP 的映射，交叉匹配得到同 pod Mooncake client endpoint；若匹配失败则记为 unresolved。
6. 按 mooncake master selector 找 master pod，解析 HTTP/RPC 端口。
7. 生成 `Snapshot`，与旧 snapshot diff，交 `Reconciler`。

#### 5.7.6 端口解析规则

1. vLLM ZMQ：优先按 `cfg.VLLM.ZMQPortName`（默认 `zmq-pub`）。
2. vLLM HTTP：存在 `http-api` 则记录，否则 0。
4. Mooncake HTTP：优先按 `HTTPPortName`，回退 `HTTPPort`（默认 9003）。
5. Mooncake RPC：优先按 `RPCPortName`，回退 `RPCPort`。


### 5.8 关键接口定义

本小节统一汇总全文关键接口定义。

```go
// ===== Query Plane =====

type Service interface {
    Compute(modelCtx ModelContext, serverIPs []string, tokenIDs []int32) ([]ServerScore, error)
}

// BlockKeyBuilder 复用 vLLM 自身的 hash_block_tokens 流程（canonical CBOR + hash algo + init_none_hash + cache_salt），
// 为 L1/L2 共同产出与 vLLM ZMQ event / Mooncake key 位级一致的 BlockHash 序列。
// 依赖：cache-indexer 与 vLLM 共享同一固定 PYTHONHASHSEED，且 vLLM 显式固定
// --prefix-caching-hash-algo=sha256_cbor 或 xxhash_cbor（§5.10）。
type BlockKeyBuilder interface {
    BuildFromRequest(modelCtx ModelContext, tokenIDs []int32) ([]BlockHash, error)
}

// ===== Index Plane =====

type L1Indexer interface {
    UpsertStoredEvent(event *L1StoredEvent) error
    RemoveBlocks(event *L1RemovedEvent) error
    ClearServer(modelCtx ModelContext, serverIP string) error
    MatchPrefix(modelCtx ModelContext, serverIP string, keys []BlockHash) (*MatchResult, error)
}

type L3Indexer interface {
    // endpoint 是 Mooncake client `transport_endpoint_`（`local_hostname`，host:port），
    // 不是 vLLM pod IP；查询端需经 Registry 从 vllmPodID 解析得到。
    ApplyReplicaSnapshot(modelCtx ModelContext, endpoint TransportEndpoint, records []L3ReplicaRecord) error
    RemoveReplicaKeys(modelCtx ModelContext, endpoint TransportEndpoint, keys []BlockHash) error
    MatchPrefix(modelCtx ModelContext, endpoint TransportEndpoint, keys []BlockHash) (*MatchResult, error)
}

// ===== Ingest Plane =====
// 由 Discovery 通过 EndpointSubscriber 驱动上下线。

type L1SubscriptionManager interface {
    Subscribe(endpoint *VLLMPodEndpoint) error
    Unsubscribe(podIP string) error
}

type MooncakePollTargetUpdater interface {
    UpdateTarget(endpoint *MooncakeMasterEndpoint) error
}

// ===== Discovery Plane =====

type Registry interface {
    Update(snapshot *Snapshot) error
    Snapshot() *Snapshot
    // 解析候选 vLLM pod 对应的同 pod 内 Mooncake client transport_endpoint_，
    // 缺失时返回 ok=false（该 server 的 L2 命中按 0 处理）。
    ResolveTransportEndpoint(serverIP string) (endpoint TransportEndpoint, ok bool)
    ResolveVLLMPod(serverIP string) (*VLLMPodEndpoint, bool)
}

type Manager interface {
    Start(ctx context.Context) error
    Stop() error
    Snapshot() *Snapshot
}

type EndpointSubscriber interface {
    OnVLLMAdded(endpoint *VLLMPodEndpoint) error
    OnVLLMRemoved(podIP string) error
    OnMooncakeMasterChanged(endpoint *MooncakeMasterEndpoint) error
}

type Reconciler interface {
    Reconcile(prev, next *Snapshot) error
}
```

### 5.9 关键设计决策

| # | 决策 | 替代方案 | 选择原因 |
|---|------|---------|---------|
| 1 | 查询侧按 vLLM `get_request_block_hasher` 的语义在 Go 中复刻 `BlockHash` 计算流程，并强制固定 `sha256_cbor` 或 `xxhash_cbor` + 共享 `PYTHONHASHSEED` | 自创前缀 key + 事件句柄反查表 / 沿用默认 `sha256/pickle` | 与 vLLM 位级对齐，与 Mooncake key 末段同形；同时避免让实现误解为 Go 直接调用 Python 函数，并规避 pickle 路径跨语言不稳 |
| 2 | L2 直接取 Mooncake key 末段作 `BlockHash` | 额外补一层映射 | Mooncake 已暴露可复用 hex，无需额外状态 |
| 3 | L1 / L2 独立存储与删除，仅查询键类型统一 | 合并存储 / 互跨删除 | 删除语义不同，隔离避免跨层副作用 |
| 4 | K8s Pod 级轮询发现 | Service 级静态注入 / informer | 下游需 Pod IP + 容器端口；一期低复杂度 |
| 5 | tokenizer 外取（上游传 `token_ids`） | 内置 tokenizer | 避免 cache-indexer 与 vLLM 因 tokenizer 版本/配置不同而切出不同的 `token_ids`，同时把 tokenize 的 CPU 开销留在无状态上游 |

### 5.10 必要约束

1. **共享 `PYTHONHASHSEED`（强约束）**：cache-indexer 进程与集群中所有 vLLM 实例必须使用同一固定 `PYTHONHASHSEED`。该值在 vLLM v1 路径下只用于派生首块 `NONE_HASH`；若未设置则会 fallback 到 `os.urandom`，产生跨进程随机种子，导致首块开始整条 `BlockHash` 链位级不一致。Go 侧必须读取同一环境变量字符串，并按同一规则派生 `NONE_HASH`。
2. **共享 hash 算法与序列化路径（强约束）**：vLLM 必须显式启用 `--prefix-caching-hash-algo=sha256_cbor` 或 `xxhash_cbor`；禁止依赖默认 `sha256/pickle` 或 `xxhash/pickle` 路径。Go 侧 `BlockKeyBuilder` 必须按 canonical CBOR 复现相同输入三元组 `(parent_block_hash, token_ids, extra_keys)`，否则仅共享 `PYTHONHASHSEED` 仍不足以保证跨语言一致。
3. **tokenizer 责任在上游**：`token_ids` 与目标 vLLM 的 tokenizer / chat template 必须一致。
4. **block_size 必须一致**：写入与查询使用同一 `block_size`，否则 `BlockHash` 序列不一致。
5. **`cache_salt` / `extra_keys` 透传**：调用方传入的 `cache_salt` 必须与对应 vLLM 推理请求实际使用的 `cache_salt` 一致（进 `extra_keys` 首位计算）；若对应 vLLM 请求未设置该字段，则这里也必须为空。其他进入 `extra_keys` 的字段口径也必须与 vLLM 保持锁步。
6. **vLLM Pod 与本地 Mooncake client `transport_endpoint_` 映射必须可解析（强约束）**：L2 命中按 Mooncake client 的 `transport_endpoint_`（即客户端启动时传入的 `local_hostname`，形如 `host:port`）做归属，而 vLLM 请求路由按 pod 走。cache-indexer 通过 Mooncake Master `/get_all_segments` 反查 vLLM Pod IP 到 `transport_endpoint_` 的映射；因此 Mooncake Master 必须暴露 `/get_all_segments` 接口，且返回的 segment 信息需包含 client IP。若反查失败，则该 pod 的 L2 命中恒为 0。

### 5.11 DFX

- **性能**：写入 / 查询均 O(blocks)。`ContextData.mu` 为 `sync.RWMutex`：多个查询可同时持有读锁，不会彼此串行阻塞；只有写入 L1/L2 索引时才占用写锁并短暂独占。
- **可靠性**：Discovery diff 驱动上下线与索引清理；ZMQ SUB 断连后由 `L1SubscriptionManager` 指数退避重连；Mooncake poller 失败跳轮且不丢定 `K_{prev}`。
- **可服务性**：Prometheus 指标包括：`cache_indexer_l1_keys_total{server}`、`cache_indexer_l2_keys_total{server}`、`cache_indexer_query_latency_seconds`、`cache_indexer_l1_event_lag_seconds`、`cache_indexer_l2_poll_duration_seconds`、`cache_indexer_discovery_pods{role}`。
- **可测试性**：查询 / 采集 / 发现三层按 interface 隔离；CI 固定同一组输入，分别用 Python vLLM 与 Go `BlockKeyBuilder` 计算 block hash，并逐个比较输出 hex 是否完全一致（[§5.13](#513-测试计划)）。
- **兼容性**：HTTP 查询套路留存与旧 Python 版路径一致（仅请求体 / 响应体变更，见 [§3.2](#32-非目标) #4）。

### 5.12 核心数据模型

本提案主要类型定义如下：

```go
type BlockHash string

type ModelContext struct {
    ModelName string
    LoraName  string
    BlockSize int64
    CacheSalt string
}

type L1ServerState struct {
    MediumSet      map[string]struct{}
    DPRankSet      map[int64]struct{}
    GroupIndexSet  map[int64]struct{}
}

type L3ServerState struct {
    ReplicaTypeSet map[string]struct{}
}

type L1KeyEntry struct{ ServerStates map[string]*L1ServerState }
type L3KeyEntry struct{ ServerStates map[TransportEndpoint]*L3ServerState }

// 一个 ModelContext 对应一份 ContextData
type ContextData struct {
    mu sync.RWMutex
    // L1：两张互为倒排、共享 *L1ServerState 指针
    l1KeyIndex    map[BlockHash]*L1KeyEntry
    l1ServerIndex map[string]map[BlockHash]*L1ServerState
    // L2：同 L1 同形；第一维 key 是 Mooncake client 的 transport_endpoint_，
    // 不是 vLLM pod IP，由 Registry 在查询阶段从 serverIP 解析得到。
    l3KeyIndex    map[BlockHash]*L3KeyEntry
    l3ServerIndex map[TransportEndpoint]map[BlockHash]*L3ServerState
}

type TransportEndpoint string // 形如 "host:port"，对应 Mooncake client 启动 local_hostname
```

采集 / 查询 / 命中结果各自独立建模，避免与索引内部类型耦合：

```go
type RequestKeyView struct {
    ModelContext ModelContext
    BlockKeys    []BlockHash
    TotalBlocks  int
}

type L1StoredEvent struct {
    ModelContext     ModelContext
    ServerIP         string
    BlockHashes      []BlockHash
    ParentBlockHash  *BlockHash
    TokenIDs         []int32
    BlockSize        int64
    Medium           string
    DPRank           int64
    GroupIndex       *int64
    EventTimestampMs int64
}

type L1RemovedEvent struct {
    ModelContext     ModelContext
    ServerIP         string
    BlockHashes      []BlockHash
    Medium           string
    GroupIndex       *int64
    EventTimestampMs int64
}

type L3ReplicaRecord struct {
    ModelContext      ModelContext
    ServerIP          string
    BlockKey          BlockHash
    RawMooncakeKey    string
    ReplicaType       string
    TransportEndpoint TransportEndpoint
}

type MatchResult struct {
    MatchedBlocks int
    MatchedTokens int
    Keys          []BlockHash
}

type ServerScore struct {
    ServerIP        string
    L1HitRatio      float64
    L3HitRatio      float64
    L1MatchedBlocks int
    L3MatchedBlocks int
    TotalBlocks     int
}
```


### 5.13 测试计划

#### 5.13.1 单元测试
- `BlockKeyBuilder`：固定一组 `token_ids`、`block_size`、`cache_salt`、`PYTHONHASHSEED`、`prefix_caching_hash_algo=sha256_cbor`（或 `xxhash_cbor`）输入，分别调用 Go `BlockKeyBuilder` 与 Python 侧 vLLM 计算 `BlockHash` 序列，断言每个位置的 hex 结果完全相同；只要有一个 block 的长度或内容不同，测试即失败。
- `L1Indexer`：BlockStored/Removed/AllBlocksCleared 状态机。
- `L3Indexer`：snapshot 应用与 diff 删除。
- discovery：label OR 筛选、端口解析回退。

#### 5.13.2 集成测试
- mock vLLM ZMQ event source + mock Mooncake HTTP，跑端到端 L1/L2 命中查询。
- discovery 上下线触发 subscribe/unsubscribe + 索引清理。

#### 5.13.3 e2e 测试
- 真实 vLLM + Mooncake 集群下的 L1/L2 命中率比对。

### 5.14 升级 / 降级策略

- HTTP 路径与请求/响应体均与旧 Python 版（`/match_sort` + `messages`）不兼容，不能原地替换。
- 部署侧须为 cache-indexer Pod 与所有 vLLM Pod 同时设置同一固定 `PYTHONHASHSEED`（[§5.10](#510-必要约束) #1）；建议从同一 ConfigMap 注入。
- vLLM 部署侧须显式固定 `--prefix-caching-hash-algo=sha256_cbor`；若选择 `xxhash_cbor`，还需同步保证运行环境具备 `xxhash` 依赖。未显式切到 `*_cbor` 路径时，cache-indexer 不应宣称跨语言 `BlockHash` 可复现。
- Mooncake Master 须暴露 `/get_all_segments` 接口，cache-indexer 据此反查 vLLM Pod IP 到 transport_endpoint 的映射；若反查失败则该 pod 的 L2 命中为 0。
- 启动期可用固定输入分别向本进程与目标 vLLM 对账 `BlockHash` 结果；只要返回的 hex 不完全一致，就拒绝对外服务。
- 回滚：若要回滚至旧 Python 版，需要使用旧版 Hermes-router，并且重新部署推理服务。

### 5.15 版本倾斜策略

- cache-indexer 与 vLLM 锁步：每个 cache-indexer release 明确声明兼容的 vLLM 版本范围（错调则 `BlockHash` 不同型）。锁定范围依据：`hash_block_tokens`、`init_none_hash`、`get_request_block_hasher`、`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` 默认值。
- cache-indexer 与 Mooncake 锁步：Mooncake key 格式与 `/get_all_keys` / `/batch_query_keys` 返回结构变更需项发布。
- N−1 / N+1 滑动：HTTP 请求 / 响应体仅后向兼容（不取除字段、不重定义语义）；新字段默认可选。

## 6. 生产可用性审查

- **可观测**：Prometheus 指标见 [§5.11](#511-dfx)；访问日志含 `serverIP` / `model` / `block_count` / `l1_matched` / `l2_matched`。
- **依赖**：K8s API Server（List Pods）、vLLM ZMQ PUB、Mooncake Master HTTP。任一快速失败：查询仍然可以返回（可能 0% 命中）。
- **扩缩容**：一期单副本（状态全部在内存）；后续能按 `ModelContext` 分片中转成多副本。
- **运营必查项**：`PYTHONHASHSEED` 与 vLLM 对齐；`--prefix-caching-hash-algo` 已显式固定为 `sha256_cbor` 或 `xxhash_cbor`；`block_size` / `cache_salt` / `extra_keys` 口径一致；Mooncake Master 暴露 `/get_all_segments` 且返回 client IP 映射；Mooncake key 末段 hex 与事件 hex 同口径（[§5.5.2](#552-订阅)）。
- 详细 PRR 问卷待补。

## 7. 实施历史

- 2026-04-23：初稿。

## 8. 缺点

1. K8s 发现为定时轮询，存在最大 `RefreshInterval` 的感知延迟。
2. `BlockHash` 计算跟随 vLLM，升级需锁步（见 [§5.15](#515-版本倾斜策略)）。
3. `PYTHONHASHSEED` 运营约束：任何 vLLM Pod 未对齐 seed 均会静默 0% 命中（[§5.10](#510-必要约束) #1 / [§5.14](#514-升级--降级策略)）。
4. 索引全内存；进程重启期间需重订阅 + 重轮询恢复，期间命中率偏低。

## 9. 替代方案

详见 [§5.9](#59-关键设计决策)。被否决的主要路线：自创前缀 key + 事件句柄反查表；L1/L2 合并存储；以 K8s Service + Endpoints 取代 Pod 级发现；cache-indexer 内置 tokenizer。

## 10. 所需基础设施

无新增基础设施需求。

## 附录

### A. 专有名词解释

| 名词 | 解释 |
|------|------|
| `server` | 本文中的 `server` 统一指一个推理后端实例，即一个 vLLM Pod；在索引与查询路径中通常以 Pod IP 作为主键。 |
| `L1` | 指 vLLM 实例本地的一级 KVCache 视图，数据来源是 vLLM 通过 ZMQ 发布的 KV event，表示该实例当前驻留在本地 GPU/HBM 中的 block。 |
| `L2` | 指 Mooncake 提供的内存层 KVCache 视图，数据来源是 cache-indexer 轮询 Mooncake Master 获得的 key 与副本信息，表示已下沉到 Mooncake 内存、可跨实例复用的 block。 |
| `L3` | 指 Mooncake 提供的 SSD 层 KVCache 视图，当前 cache-indexer 暂不支持。 |
| `tokenizer` | 指把原始文本或消息按模型规则切分为 token 序列的组件。本文方案中 tokenizer 责任上移到上游，cache-indexer 不再内置 tokenizer。 |
| `token_ids` | 指经过 tokenizer 编码后的 token ID 整数序列，是 cache-indexer 查询接口的核心输入，用于按 `block_size` 切分并计算 `BlockHash` 序列。 |
| `block_size` | 指每个 block 包含的 token 数，也是 `token_ids` 切分粒度与 `BlockHash` 递推计算时必须保持一致的参数。 |
| `block` | 指按 `block_size` 将连续 `token_ids` 切分后得到的 KVCache 基本管理单元。L1/L2 的命中统计、存储与删除语义都以 block 为粒度。 |
| `Block Hash` | 指按 vLLM `hash_block_tokens` 规则为单个 block 计算出的统一哈希标识；在本文代码类型定义中写作 `BlockHash`，并作为查询、索引、删除语义中的核心键类型。 |