---
title: Omni 支持动态 UDAF
ofep-number: 0020
authors:
  - TBD
owning-sig: sig-bigdata
participating-sigs:
status: provisional
creation-date: 2026-07-24
reviewers:
  - TBD
approvers:
  - TBD
see-also:
replaces:
replaced-by:
stage: alpha
latest-milestone: TBD
milestone:
  alpha: TBD
  beta: TBD
  stable: TBD
feature-gates:
  - name: spark.gluten.sql.columnar.backend.omni.udfLibraryPaths
    components:
      - Gluten Omni Backend
      - OmniOperator
disable-supported: true
metrics:
---

# oFEP-0020：Omni 支持动态 UDAF

<!-- toc -->
- [发布签核清单](#发布签核清单)
- [概括](#概括)
- [动机](#动机)
  - [目标](#目标)
  - [非目标](#非目标)
- [提案](#提案)
  - [用户故事（可选）](#用户故事可选)
  - [注释/限制/注意事项（可选）](#注释限制注意事项可选)
  - [风险与缓解措施](#风险与缓解措施)
- [设计细节](#设计细节)
  - [测试计划](#测试计划)
  - [毕业标准](#毕业标准)
  - [升级/降级策略](#升级降级策略)
  - [版本倾斜策略](#版本倾斜策略)
- [生产可用性审查](#生产可用性审查)
- [实施历史](#实施历史)
- [缺点](#缺点)
- [替代方案](#替代方案)
- [所需基础设施（可选）](#所需基础设施可选)
<!-- /toc -->

## 发布签核清单

标记有（R）的项目*在达到里程碑/发布*之前是必需的。
- [ ]（R）发布里程碑中的增强问题，链接到 [openfuyao/ofep] 中的 oFEP 目录
- [ ] (R) oFEP 审批者已批准 oFEP 状态为“可实施”
- [ ] (R) 设计细节已适当记录
- [ ]（R）测试计划已到位
- [ ] (R) 毕业标准已设定
- [ ] (R) 生产准备情况审查完成
- [ ] “实施历史”部分已更新里程碑
- [ ] 面向用户的文档已在 [openfuyao/docs] 创建

- [openfuyao.cn](https://openfuyao.cn/)
- [openfuyao/ofep](https://gitcode.com/openfuyao/ofep)
- [openfuyao/docs](https://gitcode.com/openfuyao/docs)

## 概括

本提案在现有动态 UDF 框架上扩展，使 Gluten Omni 后端支持通过 `.so` 插件动态加载 **UDAF（用户自定义聚合函数）**。采用与 Velox 后端一致的「Java 类名绑定 + Native 实现」模式：Driver 注册签名用于计划校验与转换，Executor 加载同一插件并在运行时按 Partial / PartialMerge / Final / Complete 四阶段执行聚合，从而让 Hive/Spark 自定义聚合可下推到 Omni 原生执行。

## 动机

Omni 当前不支持 UDAF。SQL 中出现 Hive/Scala 自定义聚合时只能 fallback 到 Spark 行式执行，聚合阶段无法享受 Omni 加速，也与 Velox 已具备的动态 UDAF 能力不对齐。

业务侧大量存量 Hive UDAF 无法短期全部改写为内置函数，需要一条与现有 `udfLibraryPaths` 一致的插件化路径。

### 目标

1. 支持 Hive UDAF / Scala UDAF 的动态加载与下推
2. 复用动态 UDF 的配置、`.so` 加载与 JNI 初始化流程
3. 规划期可识别已注册 UDAF；未注册时安全 fallback 到 Spark
4. 运行时支持四阶段聚合（Partial / PartialMerge / Final / Complete）
5. 提供清晰的插件实现约束（JVM 类 + Native `.so`，以 Java 全限定类名绑定）

### 非目标

1. 不自动把纯 JVM UDAF 编译/翻译成 Native 实现
2. 不在本提案内覆盖全部复杂类型与全部 spill 边界场景的完备性证明（可作为后续增强）
3. 不改变 Spark 侧 `CREATE TEMPORARY FUNCTION` / `ADD JAR` 的使用方式

## 提案

扩展 Omni 动态函数框架以同时承载 UDAF：

```text
用户配置 .so + 注册 JVM UDAF
        │
        ▼
Driver: 加载插件 → 注册签名到 UDFResolver
        │
        ▼
计划转换: HiveUDAF → Substrait Aggregate → Omni AggregationNode
        │
        ▼
Executor: UdafRegistry 创建 Aggregator → 四阶段执行
```

成功标准：配置并注册 UDAF 后，含该聚合的 SQL 可下推 Omni 且结果正确；未配置时可 fallback，不影响其他查询。

### 用户故事（可选）

#### 故事 1

作为数据分析用户，我按现有方式 `ADD JAR` 并 `CREATE TEMPORARY FUNCTION` 注册自定义聚合，同时配置 Omni UDAF `.so` 后，即可在 SQL 中直接使用该函数并获得 Omni 加速。

示例：

```sql
ADD JAR 'file:///opt/hive-udf-bench-1.0.0.jar';
CREATE TEMPORARY FUNCTION my_double_avg AS 'com.example.hive.MyDoubleAvg';

SELECT k, my_double_avg(CAST(x AS DOUBLE))
FROM t
GROUP BY k;
```

#### 故事 2

作为平台运维，我只需在提交参数中增加：

```text
spark.gluten.sql.columnar.backend.omni.udfLibraryPaths=file:///xxx/libmyudaf.so
```

即可在集群中启用该 UDAF 插件；去掉配置或未注册时，查询自动 fallback，不影响集群可用性。

### 注释/限制/注意事项（可选）

| 项 | 说明 |
| -- | -- |
| 支持范围 | Hive UDAF、Scala UDAF 动态加载与下推 |
| 插件形态 | 必须同时提供 JVM 实现（jar）与 Native 实现（`.so`） |
| 绑定键 | Java 全限定类名（如 `com.example.hive.MyDoubleAvg`） |
| 配置项 | 与动态 UDF 共用 `udfLibraryPaths` |
| 聚合阶段 | 插件需正确实现四阶段 `inputRaw` / `outputPartial` 组合 |

### 风险与缓解措施

| 风险 | 缓解 |
| -- | -- |
| 未注册 UDAF 被错误下推 | 规划期 `UDAFNames` 校验，未命中则 fallback |
| `.so` / JVM 签名不一致 | 以类名与 `UdafEntry` 双向约束；类型不匹配尽早抛错 |
| 多阶段聚合中间结果错误 | 覆盖 Partial/Merge/Final/Complete 与 group/非 group 测试 |
| 插件 ABI 误用 | 文档明确元数据接口与 Aggregator 接口约束 |

## 设计细节

### 1. 启动加载与注册

**Driver / Executor 共同：**

1. `resolveUdfConf` 解析 `udfLibraryPaths`（绝对路径 / Yarn files / 目录扫描）
2. `UdfLoader::loadUdfLibraries` → `dlopen`
3. 调用 `.so` 导出的 `registerUdf()`，向 `UdafRegistry` 注册 creator

**仅 Driver：**

1. JNI `registerFunctionSignatures()`
2. 存在 `intermediateType` 的条目注册为 UDAF：`UDAFNames` + `UDAFMap`

`.so` 需导出：

| 接口 | 作用 |
| -- | -- |
| `getNumUdaf()` | UDAF 条目数 |
| `getUdafEntries(...)` | 名称、参数/返回/intermediate 类型 |
| `registerUdf()` | 注册运行时 Aggregator 工厂 |

### 2. SQL 规划与校验（Driver）

Spark 物理计划中的 `HiveUDAFFunction` 由 `OmniHashAggregateExecTransformer` 接管时：

- 若类名在 `UDFResolver.UDAFNames` 中 → 允许下推
- 否则 → fallback 到 Spark

通过 `OmniAggregateFunctionsBuilder` 生成 Substrait 聚合函数：

- 函数名 = Java 全限定类名
- phase = `INITIAL_TO_INTERMEDIATE` / `INTERMEDIATE_TO_RESULT` 等
- 按阶段选择原始输入类型或 intermediate 类型

### 3. Substrait → Omni 计划（Executor）

`SubstraitToOmniPlanConverter` 解析 AggregateRel：

1. 识别已注册 UDAF → `OMNI_AGGREGATION_TYPE_UDAF`
2. 解析 UDAF 名与 intermediate 类型
3. 构造带 `aggUdafNames` 的 `AggregationNode`

### 4. 运行时执行（Executor）

```text
AggregationNode(aggUdafNames)
    → DynamicUdafAggregatorFactory
    → UdafRegistry::create(className)
    → TypedAggregator（插件实现）
    → ProcessGroup / ExtractValues*
```

四阶段语义：

| 模式 | inputRaw | outputPartial | 输入 | 输出 |
| -- | -- | -- | -- | -- |
| Partial | true | true | 原始列 | intermediate |
| PartialMerge | false | true | intermediate | intermediate |
| Final | false | false | intermediate | final |
| Complete | true | false | 原始列 | final |

### 测试计划

[x] 理解：实现前提交前需补齐相关组件测试。

##### 先决条件测试更新

复用并扩展现有动态 UDF 加载、聚合算子相关测试。

##### 单元测试

- UDAF 签名注册 / 未注册判定
- Substrait 函数名与阶段映射
- Aggregator 四阶段输入输出

##### 集成测试

- group by / 无 group by
- Partial / Merge / Final 多阶段正确性
- 配置加载成功；未加载时 fallback
- 与内置聚合混用场景

##### e2e 测试

- 端到端 SQL（全局聚合、分组聚合）结果与 Spark 一致
- 性能基线后续专项评估（本阶段以能力打通为主）

### 毕业标准

#### Alpha

- 动态加载与下推通路打通
- 典型 Hive UDAF 示例可用
- 未注册 fallback 可靠

#### Beta

- 多阶段聚合与常见类型稳定
- 插件约束文档与示例齐全
- 建议具备 spill 相关能力或明确限制说明

#### GA

- 生产验证通过，监控与排障文档齐全

### 升级/降级策略

- 升级：无配置时行为不变；配置 `.so` 后按需启用
- 降级：移除 `udfLibraryPaths` 或回退版本后，UDAF 查询回退 Spark 执行

### 版本倾斜策略

Driver 与 Executor 必须加载同一套 UDAF 插件与兼容运行时；版本或 `.so` 不一致时可能导致注册失败或执行异常，应统一发布与分发。

## 生产可用性审查

### 功能启用和回滚

###### 如何在实时集群中启用/禁用此功能？

- [x] **其他机制**
  - 配置 `udfLibraryPaths` 指向 UDAF `.so`；去掉配置即禁用下推
  - 需在作业/集群启动时生效，无需单独控制面停机

###### 启用该功能会改变任何默认行为吗？

默认无插件时不改变行为。配置并注册后，原会 fallback 的 UDAF 可能改为 Omni 执行，语义应保持一致。

###### 该功能一旦启用，是否可以禁用？

可以。移除库路径或停止注册后，相关聚合回退 Spark。

###### 如果该功能之前已回滚，现在我们重新启用它会发生什么情况？

重新加载插件并注册后，符合条件的 UDAF 再次下推。

###### 是否有任何针对功能启用/禁用的测试？

有：加载成功下推 vs 未加载 fallback 对比测试。

### 推出、升级和回滚规划

###### 部署或回滚为何会失败？这会影响正在运行的工作负载吗？

常见失败：`.so` 缺失、ABI 不匹配、签名不一致。影响范围限于使用该 UDAF 的新作业；可通过去掉配置快速回退。

###### 哪些具体指标应该通知回滚？

聚合结果不正确、任务失败率上升、Native 聚合阶段异常日志增多。

###### 升级和回滚测试了吗？

Alpha 阶段以配置开关与版本回退手工验证为主。

### 监控要求

###### 操作员如何确定该功能是否正在被工作负载使用？

- Driver 日志中的 `Registered UDAF: ...`
- 执行计划中聚合是否落在 Omni HashAggregate，而非 Spark fallback

###### 使用此功能的人如何知道它适用于他们的实例？

- [x] 其他：explain / Spark UI 显示 Omni 聚合算子；日志可见 UDAF 注册信息

### 依赖项

- Gluten Omni 后端、OmniOperator、动态 UDF 加载框架
- 用户提供的 JVM UDAF jar 与 Native `.so`
- 不新增集群级外部服务

### 可扩展性

不引入新 K8s/API 对象。每个 group 持有独立 UDAF state，内存随 cardinality 增长；大数据 GROUP BY 场景建议插件实现 spill 相关接口。

### 故障排除

| 现象 | 排查 |
| -- | -- |
| 仍 fallback | 检查 `udfLibraryPaths`、`.so` 是否加载、类名是否一致、是否 `CREATE FUNCTION` |
| Native 执行失败 | 检查四阶段实现、类型与 NULL 处理、intermediate schema |
| 结果不一致 | 对比 Spark 执行；重点查 PartialMerge / Final |

## 实施历史

- 2026-07-24：提案初稿提交（provisional）

## 缺点

- 要求用户同时维护 JVM 与 Native 两套实现，接入成本高于纯 JVM UDAF
- 插件 ABI 与四阶段语义有学习成本
- 复杂类型 / spill 完备性需要持续补齐

## 替代方案

1. **继续整段 fallback 到 Spark（现状）**：零开发成本，但自定义聚合无法加速
2. **仅支持内置聚合、不开放插件**：可控性更好，但无法覆盖存量 Hive UDAF
3. **只在 JVM 列式框架执行 UDAF**：实现可能更简单，但无法复用 Omni Native 聚合与现有动态 UDF 体系

故选择与 Velox 对齐的「类名绑定 + `.so` 插件」方案。

## 所需基础设施（可选）

无额外基础设施需求；建议在文档仓库补充 UDAF 插件模板与示例。
