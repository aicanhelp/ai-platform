---
title: Omni 支持 UDF 表达式部分回退（Partial Project）
ofep-number: 0010
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
  - name: spark.gluten.sql.columnar.partial.project
    components:
      - Gluten Omni Backend
      - OmniOperator
disable-supported: true
metrics:
  - omni_partial_project_total_time
  - omni_partial_project_udf_eval_time
  - omni_partial_project_output_write_time
---

# oFEP-0010：Omni 支持 UDF 表达式部分回退（Partial Project）

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

本提案为 Gluten Omni 后端引入 **Partial Project（部分表达式回退）**：当 Project 算子中混有无法下推的表达式（典型为 ScalaUDF / Hive UDF）时，不再整段回退到 Spark 行式执行，而是只把不可下推子表达式放到 JVM 侧物化，其余可下推表达式继续走 Omni 原生列式路径，从而减少行列转换与列式收益损失。

## 动机

在现有行为下，Project 中只要出现一个不可下推 UDF，整段 Project 往往会 fallback 到 Spark：

- 已可下推的普通投影一并损失
- 需要引入 C2R / R2C，额外开销大

大数据查询中这类“少量 UDF + 大量普通表达式”很常见，需要一种更细粒度的回退能力。

### 目标

1. 支持 Project 中 ScalaUDF / Hive UDF 的部分回退，其余表达式继续 Omni Native 执行
2. 主路径保持 `executeColumnar()` 列式传递，避免整段 Project 回退带来的 C2R/R2C
3. 提供配置开关 `spark.gluten.sql.columnar.partial.project`，可随时关闭
4. 结果与整段回退语义一致；在典型场景下相对整段回退有可测量的性能收益

### 非目标

1. 不覆盖 Aggregate / UDAF 场景的部分回退（另案处理）
2. 不承诺支持全部复杂类型 UDF；不支持类型直接校验失败并回退
3. 不改变 Spark UDF 本身语义与注册方式

## 提案

引入优化规则与新算子，将一个物理 Project 拆成两级：

1. **先执行** `OmniColumnarPartialProjectExec`：在 JVM 侧执行 UDF 等不可下推表达式，输出 `_SparkPartialProjectN` 列
2. **后执行** `ProjectExecTransformer`：对合并后的 ColumnarBatch 执行剩余可下推投影

成功标准：开启特性后计划中出现 `OmniColumnarPartialProjectExec`，SQL 结果与整段回退一致，端到端耗时优于整段回退。

### 用户故事（可选）

#### 故事 1

作为使用 Gluten Omni 的数据分析用户，我在 SQL 中混用普通投影与少量 Hive/Scala UDF 时，希望只有 UDF 在 JVM 执行，其余投影仍走 Omni 加速，查询不必因单个 UDF 整段降速。

#### 故事 2

作为运维人员，我希望通过配置开关关闭 Partial Project；关闭后行为回退到原有整段 Project fallback，不影响集群稳定性。

### 注释/限制/注意事项（可选）

| 项 | 说明 |
| -- | -- |
| 触发条件 | 单个 Project 的 `projectList` 中存在无法下推表达式（如 UDF） |
| 配置 | `spark.gluten.sql.columnar.partial.project` |
| 表达式范围 | 首期验证 ScalaUDF / Hive UDF |
| 数据形态 | 算子间仍用 Omni 列式 batch；UDF 内部按行 `eval` |

### 风险与缓解措施

| 风险 | 缓解 |
| -- | -- |
| 拆分后校验失败导致错误计划 | `splitIfValid`：两侧 `doValidate` 均通过才改写，否则保持原计划 |
| UDF 输入列缺失 / 类型不支持 | 算子内做列索引与类型校验，失败则不启用部分回退 |
| AQE 生成新 Project 未覆盖 | 规则同时挂到 `QueryStagePrep` 与 `PostTransform` |

## 设计细节

### 总体流程

```text
原计划: ProjectExec(expr..., udf(...))
              │
              ▼  OmniPartialProjectRule
后执行: ProjectExecTransformer(expr..., _SparkPartialProjectN)
              │
先执行: OmniColumnarPartialProjectExec(cols..., udf(...))
              │
              ▼
        ColumnarBatch（全程列式主路径）
```

对比：

| 维度 | Partial Project | 整段 Project 回退 |
| -- | -- | -- |
| UDF 执行 | JVM（局部） | Spark 行式（整段） |
| 其余投影 | Omni Native | 一并损失 |
| 主路径数据形态 | 列式 batch | 需要 C2R & R2C |

### 关键改动

1. **优化规则 `OmniPartialProjectRule`**
   - 匹配 `ProjectExec` / `ProjectExecTransformer`
   - `projectListNeedsPartialSplit` 判断是否需要拆分
   - 上游已是 `OmniColumnarPartialProjectExec` 则跳过
   - `splitIfValid`：新建两级算子并分别 `doValidate`

2. **新算子 `OmniColumnarPartialProjectExec`**
   - `create`：识别并拆分表达式
   - `doExecuteColumnar`：列式输入 → 按行喂给 UDF → 结果写回 OmniColumnVector
   - `doValidate`：检查开关、是否挂载 UDF、输入列是否可得、类型是否支持

3. **指标**
   - 算子总耗时、UDF eval 耗时、结果写回耗时

### 测试计划

[x] 理解：实现前提交前需补齐相关组件测试。

##### 先决条件测试更新

补充 Project / UDF 相关已有用例，作为拆分前后对照基线。

##### 单元测试

- 规则命中 / 不命中 / 重复应用
- 校验失败不改写计划
- 开关关闭时不注入算子

##### 集成测试

- 同一 SQL：开启 Partial vs 整段回退，结果一致
- 执行计划包含 `OmniColumnarPartialProjectExec`
- NULL、单/多分区、AQE 开/关

##### e2e 测试

- 百万 / 千万行规模 Project+UDF 场景，对比端到端耗时（至少 3 轮）

### 毕业标准

#### Alpha

- 功能默认关闭，配置可开
- ScalaUDF / Hive UDF 基本场景通过
- 单元与集成测试就绪

#### Beta

- 覆盖 AQE、多分区、常见类型
- 指标可观测，无已知正确性缺陷

#### GA

- 生产验证通过，文档齐全，开关行为稳定

### 升级/降级策略

- 升级：默认关闭，不影响存量作业；按需打开
- 降级：关闭配置或回退版本后，恢复整段 Project fallback

### 版本倾斜策略

特性仅影响 Driver 计划改写与 Executor 算子执行，Driver/Executor 需同版本部署；版本不一致时以关闭开关或统一版本规避。

## 生产可用性审查

### 功能启用和回滚

###### 如何在实时集群中启用/禁用此功能？

- [x] **其他机制**
  - 配置：`spark.gluten.sql.columnar.partial.project=true/false`
  - 无需控制面或节点停机，重启相关 Spark 作业即可

###### 启用该功能会改变任何默认行为吗？

默认关闭；开启后仅对含不可下推表达式的 Project 改写计划，不改变 SQL 语义。

###### 该功能一旦启用，是否可以禁用？

可以。关闭配置后新作业不再注入 Partial Project。

###### 如果该功能之前已回滚，现在我们重新启用它会发生什么情况？

重新按规则拆分 Project，行为与首次启用一致。

###### 是否有任何针对功能启用/禁用的测试？

有：开关开/关下的计划与结果对比测试。

### 推出、升级和回滚规划

###### 部署或回滚为何会失败？这会影响正在运行的工作负载吗？

主要风险是拆分校验不完善导致计划异常；运行中作业不受配置变更影响，仅新提交作业生效。

###### 哪些具体指标应该通知回滚？

结果不一致、算子失败率升高、或 UDF eval / 总耗时异常劣化。

###### 升级和回滚测试了吗？

Alpha 阶段以开关切换与版本回退手工验证为主。

### 监控要求

###### 操作员如何确定该功能是否正在被工作负载使用？

查看执行计划是否出现 `OmniColumnarPartialProjectExec`，并结合算子指标。

###### 使用此功能的人如何知道它适用于他们的实例？

- [x] 其他：Spark UI / explain 中可见新算子与相关 metrics

### 依赖项

依赖 Gluten Omni 后端与现有列式执行框架；不新增集群级外部服务。

### 可扩展性

不引入新 API 类型；额外开销主要来自 UDF 行式 eval，规模随 UDF 调用次数增长，可通过只拆必要表达式控制。

### 故障排除

- 计划未出现新算子：检查开关、是否确有不可下推 UDF、校验日志
- 结果异常：对比整段回退结果；检查 UDF 类型与空值处理
- 性能回退：查看 UDF eval 占比，评估是否值得开启

## 实施历史

- 2026-07-24：提案初稿提交（provisional）

## 缺点

- 引入额外算子与规则，增加计划复杂度
- UDF 仍在 JVM 执行，收益取决于“可下推表达式占比”
- 复杂类型支持有限，部分场景仍会整段回退

## 替代方案

1. **整段 Project 回退（现状）**：实现简单，但性能损失大，故不作为目标方案
2. **将 UDF 全部改写为 Native**：收益更大，但开发成本高、覆盖慢，不在本提案范围

## 所需基础设施（可选）

无额外基础设施需求。
