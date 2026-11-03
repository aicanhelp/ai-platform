# AdmissionWebhook 与 Validator 模块设计

## 1. 模块定位与职责

| 模块 | 所处层次 | 触发时机 | 主要职责 |
|------|----------|----------|----------|
| **AdmissionWebhook** | 外部接口层 | ElasticScaler 的 **Create / Update** 请求到达 API Server 时 | 准入校验：在写入 etcd 前拦截不合法或语义冲突的 spec，保证集群内 CR 数据一致、合法。 |
| **Validator** | 协调工具层 | Reconciler 每次 **Reconcile** 开始时 | 控制器侧的 CR 语义校验：对跨字段约束、策略组合合法性等复杂规则进行校验，为后续决策与执行提供合法输入。 |

两者校验目标一致（保证 spec 合法、无冲突），但**时机与上下文不同**：Webhook 仅能拿到当前请求中的对象，Validator 可结合集群内已有资源（如 targetRef 是否存在、是否被其他 ElasticScaler 管理）做更丰富的校验。

---

## 2. 校验维度总览

以下校验维度在两个模块中的分工建议如下（具体规则见后文）：

- **必填与类型**：Webhook + Validator 均做。
- **Trigger 单一性**：Webhook 必须做；Validator 建议做（防御性）。
- **按 trigger.type 的配置合法性**：Webhook + Validator 均做，Validator 可更细（如算法是否注册）。
- **minReplicas / maxReplicas**：Webhook + Validator 均做。
- **targetRef 格式与存在性**：Webhook 做格式与 GVK 支持性；Validator 可做存在性、唯一管理权等。
- **MetricSpec 与指标配置**：Webhook + Validator 按 type 分别校验。

---

## 3. AdmissionWebhook 模块设计

### 3.1 接口与职责

- **ValidatingAdmissionWebhook**：仅做校验，不修改对象；拒绝非法 Create/Update。
- **MutatingAdmissionWebhook**（可选）：在写入前做默认值填充或规范化，例如 `trigger.type` 根据已填写的 trigger 子字段推断、`minReplicas` 默认 1 等，以降低用户配置错误。

建议：**校验逻辑以 Validating 为主**；若引入 Mutating，只做“默认值/规范化”，不改变用户明确填写的语义。

### 3.2 校验流程概览

```
请求进入 → 反序列化 ElasticScaler → 校验 metadata / spec 必填
  → 校验 targetRef
  → 校验 trigger 单一性与 type 与子配置一致
  → 按 trigger.type 分支校验（metricsTrigger / stateTrigger / signalTrigger）
  → 校验 minReplicas / maxReplicas
  → 返回 Allow / Deny(+ 原因)
```

### 3.3 通用校验（与 trigger 类型无关）

| 校验项 | 规则 | 失败说明 |
|--------|------|----------|
| **targetRef 必填** | `spec.targetRef` 非空 | 必须指定要扩缩的目标资源。 |
| **targetRef 格式** | `apiVersion`、`kind`、`name` 非空；`namespace` 可选（空则视为与 ElasticScaler 同 namespace） | 引用格式不完整。 |
| **targetRef GVK 支持** | 仅允许框架已支持的 GVK（如 `apps/v1 Deployment`、自定义 Tide 等），可配置白名单 | 避免引用无法被 ResourceHandler 处理的资源。 |
| **trigger 必填** | `spec.trigger` 非空 | 必须指定触发方式。 |
| **trigger.type 必填** | `spec.trigger.type` 为 `MetricsTrigger` / `StateTrigger` / `SignalTrigger` 之一 | 必须明确触发类型。 |
| **minReplicas / maxReplicas** | `minReplicas ≥ 0`（允许 0）；`maxReplicas ≥ 1`；且 `minReplicas ≤ maxReplicas` | 副本数范围非法。 |

### 3.4 按 trigger.type 的配置校验

#### 3.4.1 type = MetricsTrigger

| 校验项 | 规则 | 失败说明 |
|--------|------|----------|
| **仅配置 metricsTrigger** | `trigger.stateTrigger`、`trigger.signalTrigger` 均为空或未设置 | 保证“仅使用一种 trigger 策略”，避免语义冲突。 |
| **metricsTrigger 必填** | `spec.trigger.metricsTrigger` 非空 | 指标驱动必须提供指标配置。 |
| **scalingAlgorithm** | 非空；若 Webhook 侧可访问注册表，可校验算法已注册，否则交给 Validator | 算法名为空或未注册。 |
| **metrics 非空** | `metricsTrigger.metrics` 至少一条 | 指标驱动至少需要一个指标。 |
| **每条 MetricSpec** | 见下文 [3.5 MetricSpec 校验](#35-metricspec-校验) | 按 type 校验 resource / external / custom / externalServer。 |

#### 3.4.2 type = StateTrigger

| 校验项 | 规则 | 失败说明 |
|--------|------|----------|
| **仅配置 stateTrigger** | `trigger.metricsTrigger`、`trigger.signalTrigger` 均为空或未设置 | 同上，单一 trigger。 |
| **stateTrigger** | 可为空对象 `{}`，无额外必填字段 | 仅做存在性与“唯一性”校验。 |
| **targetRef 类型** | 通常为带 status（如 desiredReplicas）的 CR（如 Tide），可与 GVK 白名单结合 | 状态驱动需从目标资源 status 读副本数。 |

#### 3.4.3 type = SignalTrigger

| 校验项 | 规则 | 失败说明 |
|--------|------|----------|
| **仅配置 signalTrigger** | `trigger.metricsTrigger`、`trigger.stateTrigger` 均为空或未设置 | 同上，单一 trigger。 |
| **signalTrigger 必填** | `spec.trigger.signalTrigger` 非空 | 信号驱动必须提供信号配置。 |
| **endpoint** | 非空；格式为合法 URL（http/https，可选的 host:port/path） | 用于接收 scale 信号的 endpoint 必填且合法。 |
| **minReplicas** | 建议允许 0（scale-to-zero） | 信号场景常需 minReplicas=0。 |

### 3.5 MetricSpec 校验

适用于 `trigger.metricsTrigger.metrics[]` 中每一项。按 `type` 分支：

| MetricSpec.type | 必填子字段 | 校验要点 |
|-----------------|------------|----------|
| **Resource** | `resource` | `metricsName` 非空；`target` 非空；`target.type` 为 Utilization / Value / AverageValue 之一；数值类 target ≥ 0。 |
| **External** | `external` | `metricsName` 非空；`target` 非空；`selector` 可选；数值类 target ≥ 0。 |
| **Custom** | `custom` | `metricsName` 非空；`target` 非空；`selector` 非空（与 custom 指标绑定所需）；数值类 target ≥ 0。 |
| **ExternalServer** | `externalServer` | `metricsName` 非空；`target` 非空；`endpoint` 非空且为合法 URL；`protocol` 在允许枚举内（http/https）。 |

- 同一项中：**仅允许填写与 `type` 对应的子字段**（例如 type=Resource 时只填 `resource`，不填 `external` / `externalServer` / `custom`），否则拒绝或在校验说明中明确“禁止混用”。

补充校验：
- **ExternalServer.protocol**：允许枚举为 `http` 或 `https`；未填写时按 CRD 默认视为 `https`；显式填写 `http` 或 `https` 均允许。
- **ExternalServer.query**：字段为可选；若提供则必须为非空字符串（不做语法级校验，避免 webhook 侧引入重依赖）。

### 3.6 Webhook 实现要点

- **仅处理 ElasticScaler**：根据 `req.Kind` 只对 ElasticScaler 的 Create/Update 做校验，其他资源直接 Allow。
- **失败响应**：使用 `admission.Denied(err.Error())` 或带字段路径的 `status.Details`，便于用户定位。
- **性能**：不做集群内 List/Watch，仅基于请求体；若需“targetRef 是否存在”“唯一管理权”等，放在 Validator。
- **版本**：若支持多版本 CRD，在反序列化时统一到内部版本再校验，保证规则一致。

---

## 4. Validator 模块设计

### 4.1 接口与职责

```go
// Validator 负责控制器侧 ElasticScaler spec 的语义校验。
type Validator interface {
	// Validate 校验 ElasticScaler 的 spec（及与集群状态的约束）。
	// 若校验失败返回 error，Reconciler 将不执行扩缩逻辑并更新 status 为 ValidationFailed。
	Validate(ctx context.Context, es *autoscalingv1alpha1.ElasticScaler) error
}
```

- **输入**：`context`（可带 logger、client）+ 当前 ElasticScaler 对象（含 spec 与现有 status）。
- **输出**：`nil` 表示通过；`error` 表示未通过，Reconciler 写 Condition 并可选在 status 中记录原因。

### 4.2 校验流程概览

```
Validate(es)
  → 通用校验（与 Webhook 规则对齐：targetRef、trigger、min/max replicas）
  → 按 trigger.type 分支校验（与 Webhook 对齐，并可做更细规则）
  → 集群态校验：targetRef 是否存在、是否已被其他 ElasticScaler 管理
  → 可选：MetricsTrigger 时校验 scalingAlgorithm 是否已注册
  → 返回 nil / error
```

### 4.3 通用校验（与 Webhook 对齐）

- 与 [3.3 通用校验](#33-通用校验与-trigger-类型无关) 保持一致，避免 Webhook 通过后到控制器侧仍出现“基础字段不合法”的情况。
- 可复用同一套“结构校验”函数，供 Webhook 与 Validator 共同调用（例如放在 `pkg/validation`）。

### 4.4 按 trigger.type 的配置校验（增强）

- **MetricsTrigger**  
  - 与 Webhook 相同的 metricsTrigger / metrics / MetricSpec 规则。  
  - **算法已注册**：`scalingAlgorithm` 必须在 `AlgorithmRegistry` 中存在，否则返回“algorithm not registered”类错误。  
  - 若框架支持“指标来源可用性”的预检，可在此做可选检查（如 ExternalServer endpoint 是否可达），失败时返回明确错误而非在后续采集阶段才报错。
  - **MetricSpec.type=Custom**：若 custom 字段解析失败或必要字段缺失，返回明确错误。
  - **MetricSpec.type=ExternalServer**：若 `externalServer.query` 提供但为空字符串，拒绝（与 Webhook 对齐）。

- **StateTrigger**  
  - 与 Webhook 一致；可额外校验 targetRef 指向的资源是否具有 status（如 `status.replicas` / `status.desiredReplicas`），若 CRD 已知则校验字段存在性。

- **SignalTrigger**  
  - 与 Webhook 一致；可对 endpoint 做格式或协议白名单校验。

### 4.5 跨字段与集群态校验（Validator 独有）

| 校验项 | 规则 | 说明 |
|--------|------|------|
| **targetRef 资源存在** | 根据 `targetRef` 在集群中 Get 对应资源，存在则通过 | 避免指向不存在的资源。 |
| **ElasticScaler 与 targetRef 一一对应** | 查询集群中是否已有其他 ElasticScaler 的 `spec.targetRef` 与当前对象的 targetRef 一致（同 namespace、同 name、同 GVK） | 防止多 CR 管理同一 targetRef，导致扩缩冲突。|
| **minReplicas / maxReplicas 与业务语义** | 例如 SignalTrigger 下 minReplicas=0 是否允许（策略上可配置） | 可按需求在 Validator 中加策略级校验。 |
| **默认值等价处理** | 若 `minReplicas` 为空，应在语义校验阶段等价视为默认 `1`；若 `maxReplicas` 未显式填写，应等价视为默认 `10` | 便于在不依赖 Mutating Webhook 的情况下仍保证一致语义。 |

### 4.6 Validator 实现要点

- **幂等**：同一 spec 多次 Validate 结果一致。
- **可观测性**：校验失败时打日志并写 Condition/Reason/Message，便于排查。
- **性能**：仅对当前 ElasticScaler 做必要 Get/List（如查 targetRef、查其他 ElasticScaler），避免全集群扫描。
- **与 Webhook 共享规则**：建议将“结构校验”“按 type 的配置校验”“MetricSpec 校验”抽成公共函数，Webhook 与 Validator 都调用，保证行为一致且易维护。

---

## 5. 校验规则对照表（按配置汇总）

便于实现与测试时对照，下表按“配置场景”汇总“应校验项”及“负责模块”。

| 配置场景 | 校验项 | AdmissionWebhook | Validator |
|----------|--------|------------------|-----------|
| 所有 | targetRef 必填、格式、GVK 支持 | ✓ | ✓ |
| 所有 | trigger 必填，trigger.type 必填且枚举正确 | ✓ | ✓ |
| 所有 | 仅一种 trigger 子配置与 type 一致 | ✓ | ✓ |
| 所有 | minReplicas ≥ 0（允许 0），maxReplicas ≥ 1，且 min ≤ max | ✓ | ✓ |
| MetricsTrigger | metricsTrigger 非空；scalingAlgorithm 非空；metrics 非空 | ✓ | ✓ |
| MetricsTrigger | scalingAlgorithm 已注册 | 可选 | ✓ |
| MetricsTrigger | 每条 MetricSpec 按 Resource/External/Custom/ExternalServer 校验 | ✓ | ✓ |
| MetricsTrigger | ExternalServer.query 若提供则非空字符串 | ✓ | ✓ |
| StateTrigger | 仅 stateTrigger 配置；targetRef 类型适合状态驱动 | ✓ | ✓ |
| SignalTrigger | signalTrigger 非空；endpoint 非空且合法 URL | ✓ | ✓ |
| 集群态 | targetRef 对应资源存在 | — | ✓ |
| 集群态 | 无其他 ElasticScaler 管理同一 targetRef | — | ✓ |

---

## 6. 建议的代码结构（参考）

```
pkg/
  validation/
    spec.go          # 通用 spec 校验（targetRef, trigger, min/max）
    trigger.go      # 按 trigger.type 分支及 metricsTrigger/stateTrigger/signalTrigger
    metrics.go      # MetricSpec 与 Resource/External/ExternalServer
    doc.go          # 包注释与校验规则说明
  webhook/
    admission.go    # ValidatingAdmissionWebhook 实现，调用 validation 包
  validator/
    validator.go    # Validator 实现，调用 validation 包 + 集群态校验（client）
```

Webhook 与 Validator 均依赖同一套 `validation` 包，可保证“同一种配置”在两处得到相同的基础校验结果；Validator 在此基础上增加“算法注册”“targetRef 存在性”“唯一管理权”等需要集群状态的校验。
