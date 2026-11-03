---
title: 【安全】支持compliance-operator扫描能力
ofep-number: 0003
authors:
  - "@regandy"
owning-sig: security-committee
status: provisional
creation-date: 2026-04-29
reviewers:
approvers:
see-also: 
replaces:
---
# oFEP-0003: 【安全】支持compliance-operator扫描能力

<!-- toc -->
- [发布签核清单](#release-signoff-checklist)
- [摘要](#summary)
- [动机](#motivation)
  - [目标](#goals)
  - [非目标](#non-goals)
- [提案](#proposal)
  - [用户故事（可选）](#user-stories-optional)
    - [故事 1](#story-1)
    - [故事 2](#story-2)
  - [注释/约束/警告（可选）](#notesconstraintscaveats-optional)
  - [风险与缓解措施](#risks-and-mitigations)
- [设计细节](#design-details)
  - [测试计划](#test-plan)
      - [先决条件测试更新](#prerequisite-testing-updates)
      - [单元测试](#unit-tests)
      - [集成测试](#integration-tests)
      - [e2e 测试](#e2e-tests)
  - [毕业标准](#graduation-criteria)
  - [升级/降级策略](#upgrade--downgrade-strategy)
  - [版本倾斜策略](#version-skew-strategy)
- [生产准备情况评审问卷](#production-readiness-review-questionnaire)
  - [功能启用和回滚](#feature-enablement-and-rollback)
  - [推出、升级和回滚规划](#rollout-upgrade-and-rollback-planning)
  - [监控要求](#monitoring-requirements)
  - [依赖项](#dependencies)
  - [可扩展性](#scalability)
  - [故障排除](#troubleshooting)
- [实施历史](#implementation-history)
- [缺点](#drawbacks)
- [替代方案](#alternatives)
- [所需基础设施（可选）](#infrastructure-needed-optional)
<!-- /toc -->

## 发布签核清单

标记有（R）的项目在达到里程碑/发布之前是必需的。
- [ ]（R）发布里程碑中的增强问题，链接到 openfuyao/ofep 中的 oFEP 目录
- [ ] (R) oFEP 审批者已批准 oFEP 状态为“可实施”
- [ ] (R) 设计细节已适当记录
- [ ]（R）测试计划已到位，并考虑了 SIG 架构和 SIG 测试的输入
 [ ] 控制平面和worker节点都需支持扫描
- [ ] (R) 毕业标准已设定
- [ ] (R) 生产准备情况审查完成
- [ ] (R) 生产准备情况审查已获批准
- [ ] “实施历史”部分已更新里程碑

## 概括

本提案定义一个合规扫描 compliance-operator，用 CRD 统一控制扫描任务下发、节点选择、kube-bench/oscap 命令执行以及扫描结果回收，并提供“按 scanId 精准导出历史报告”的用户体验：用户先查看扫描历史获取 scanId，再通过子资源式命令携带 scanId 导出对应的 HTML 报告。

关键特性：
- 单个 ScannerTask CR 长期存在，触发扫描后自动追加历史记录，并仅保留最近 5 次。
- 扫描结果以 JSON 形态保存到 etcd（满足“不能用 PVC/hostPath 存档”的约束），导出时由服务端动态生成 HTML（不存 HTML 到 etcd）。
- 支持 kubectl 原生命令，导出指定历史版本报告。

## 动机

### 目标

- 使用 CRD，由 Compliance-Operator 统一控制：
  - 给哪些节点下发扫描任务（Job/Pod）
  - kube-bench /OSCAP的启动命令行与参数
  - Pod 的必要主机目录挂载
  - 结果回收与汇总
- 结果必须上报到 Operator，并存入 etcd；kubectl get 可查看扫描简报（时间、状态、scanId）。
- 用户导出报告时必须显式指定 scanId，避免覆盖与歧义；支持保留最近 5 次历史。
- HTML 报告动态生成，UI 完全由 Operator 控制，不将 HTML 持久化到 etcd。

### 非目标

- 不在本提案中定义完整的CIS、 STIG 规则内容与评估体系；仅定义运行框架与数据模型。
- 不在本提案中支持无限历史保留；历史保留固定为最近 5 次。
- 不承诺将完整原始扫描产物（例如所有中间文件）持久化；仅保留扫描结果所需数据。
- 不提供webUI，全命令行操作。

## 提案

用户视角核心流程("scan“、”cluster-scan"只是示例，具体在开发时确定)：

```bash
# 1) 查看历史列表（拿到 scanId）
kubectl get scan cluster-scan -o wide

# 2) 指定 scanId 导出历史报告
kubectl get --subresource=report scan cluster-scan?scanId=20250522-112233 > report.html
```

总体方案：
- 定义 Cluster-scan CRD：ScannerTask（shortName: scan）。
- 部署CR即触发扫描，二次扫描可以通过注解触发扫描，Operator 生成唯一 scanId 并启动扫描。
- Operator 在目标节点创建一次性 Job/Pod 运行 kube-bench/OSCAP（镜像封装 kube-bench/OSCAP 即可，命令行参数由 Operator 注入）。
  - kube-bench扫描CIS K8S规则。
  - oscap扫描STIG K8S规则。
- Pod 将结果上报给 Operator；Operator 将结果以 JSON 形态持久化到 etcd，并维护最多 5 条历史索引。
- 报告导出采用子资源式 API（示例）：`.../scannertasks/<name>/report?scanId=<id>`，由服务端根据 scanId 查找数据并生成 HTML 返回。
- 报告的html可以参考灵雀云的UI呈现。

### 用户故事（可选）

#### 故事 1

安全管理员在生产集群定期触发一次全量扫描，观察到历史列表中保留最近 5 次 scanId；遇到审计需要时，指定某次 scanId 导出报告用于存档。

#### 故事 2

平台运维只需维护一个固定的 ScannerTask CR（cluster-scan），通过注解触发扫描即可；不需要为每次扫描创建新 CR 或手工收集各节点报告。

### 注释/限制/注意事项（可选）

- 结果存入 etcd 需要关注对象大小限制；建议将 ScannerTask.status 中仅保存“简报 + 索引”。详细结果也只是存结果相关内容，检查项的描述、修正方法等文本，不要存在ETCD。
- 扫描 Pod 涉及主机目录挂载。

  

### 风险与缓解措施

- 风险：扫描结果体量大导致 etcd 对象超限或性能压力。
  - 缓解：将大报告拆分存入独立资源；限制历史条数（5）；对 JSON 进行裁剪（仅保留必要字段）。
- 风险：扫描 Pod 需要 hostPath，存在安全风险。
  - 缓解：最小权限运行、只读挂载
- 风险：报告导出端点被滥用导致敏感信息泄露。
  - 缓解：不单独提供导出API，只支持后端命令行，通过kubectl导出

## 设计细节

### API 与数据模型

ScannerTask（Cluster-scoped）核心字段建议（示例，实际字段明以开发为主）：
- spec.scanType: control-plane | node | all
- status.lastScanStatus: Pending | Running | Completed | Failed
- status.scanHistory（最多 5 条）：
  - scanId：唯一 ID（时间戳/随机串）
  - scanTime：扫描时间
  - state：状态
  - summary：摘要
  - reportRef：指向存储完整 JSON 的资源引用（可选，但建议）

用户实例 CR（长期只维护 1 个，避免每次扫描都部署CR，因为内容都基本一致）：

```yaml
apiVersion: security.example.com/v1
kind: ScannerTask
metadata:
  name: cluster-scan
spec:
  scanType: all
```

### 触发机制

- 使用注解触发：`scan.example.com/trigger=true`，Operator 收到后生成 scanId 并启动扫描，然后清理注解。
- 保留“单 CR 永久使用”的体验：触发不依赖创建新 CR。

### 扫描执行与结果回收

- Operator 根据 scanType 与策略选择目标节点。
- 每节点创建一次性 Job/Pod 运行 kube-bench：
  - 镜像：封装 kube-bench /oscap即可
  - 命令行：由 Operator 注入
  - hostPath 挂载：由 Operator 统一定义（根据扫描所需目录）
- Pod 将结果（JSON）上报给 Operator，Operator 汇总并写入 etcd。

### 报告导出（按 scanId）

- 导出接口语义：必须携带 scanId；未携带返回 400；scanId 不存在返回 404。
- 返回内容：`Content-Type: text/html; charset=utf-8`，HTML 由服务端按模板渲染生成。

### 测试计划

[ ] 我/我们理解，相关组件所有者可能会要求在实现前补充或重构测试。

##### 先决条件测试更新

- 明确 kube-bench/oscap 扫描所需挂载目录清单与最小权限策略。

##### 单元测试

- scanId 生成与历史保留（最多 5 条）逻辑。
- 报告导出参数校验与 scanId 精准匹配逻辑。

##### 集成测试

- Operator 创建 Job/Pod、收集回传结果、写入状态/报告资源的链路。

  

##### e2e 测试

- 触发扫描、查看历史列表、导出指定 scanId 报告的端到端验证。
- 触发超过 5 次扫描后，旧报告被淘汰的行为验证。

### 毕业标准

- Alpha（v26.06）：CRD + Operator 完成扫描触发、结果回收、历史保留（5）与导出指定 scanId 报告的完整链路，生产验证充分。
- Beta（v26.09）：完成不合规项整改，生产验证充分，
- Stable（v27.03）：生产验证充分，形成运维手册与安全评审结论。

### 升级/降级策略

- 升级：新增 CRD 与 Operator 部署，不影响现有工作负载；默认不自动触发扫描。
- 降级：下线 Operator 后，历史数据仍保留在 etcd（CR 对象仍在）；可按需清理。

### 版本倾斜策略



## 生产可用性审查

### 功能启用和回滚

###### 如何在实时集群中启用/禁用此功能？

- [ ] 其他机制
  - 机制：部署/卸载 compliance-operator。
  - 启用/禁用是否导致控制平面停机：否。
  - 启用/禁用是否导致节点停机或重新部署：否。

###### 启用该功能会改变任何默认行为吗？

- 不会自动触发扫描，仅在用户显式触发时执行。

###### 该功能一旦启用，是否可以禁用（回滚启用）？

- 可以，通过卸载 Operator 或关闭调度能力停止后续扫描。扫描功能很快，停止后就没有运行中的pod了。

### 推出、升级和回滚规划

###### 部署或回滚为何会失败？这会影响正在运行的工作负载吗？

- 可能因 RBAC/特权不足导致扫描 Pod 创建失败；不会影响业务工作负载，但会影响扫描可用性。

###### 哪些具体指标应该通知回滚？

- 不用回滚，不影响业务pod运行。

### 监控要求

###### 操作员如何确定该功能是否正在被工作负载使用？

- 人工触发的，执行完就停止了。可以查看运行中的job

###### 使用此功能的人如何知道它适用于他们的实例？

- [ ] 通过查看历史

### 依赖项

###### 此功能是否依赖于集群中运行的任何特定服务？

- Kubernetes apiserver/etcd：用于存储 CR 与报告数据。

  

### 可扩展性

###### 启用/使用此功能会导致任何新的 API 调用吗？

- 会：触发扫描时创建 Job/Pod、更新状态/报告资源

###### 启用/使用此功能是否会导致现有 API 对象的大小或数量增加？

- 会：每次扫描产生 1 条历史记录，可能产生 1 个报告资源；历史最多保留 5 条/5 个报告资源。

### 故障排除

###### 如果 API 服务器和/或 etcd 不可用，此功能如何反应？

- 无法写入状态与报告数据，扫描结果无法持久化；应在日志与事件中给出明确错误。

###### 其他已知故障模式有哪些？

- 

###### 如果未满足 SLO，应采取哪些步骤来确定问题？

- 先看 Operator 日志与事件，再看扫描 Pod 状态与节点资源，再看报告写入与导出端点的错误码统计。

## 实施历史

- 2026-04-28：形成方案初稿（provisional）。

## 缺点

- 扫描需要 hostPath能力
- 结果存入 etcd 需要处理对象大小与性能影响。

## 替代方案

- 每次扫描创建独立 CR（每个 scanId 一个 CR）：实现简单，但不符合“单 CR 永久使用”的体验要求。
- 将报告存入 PVC/hostPath：不符合“结果必须存入 etcd”的约束。
- 全部使用oscap，不引入kube-bench。此方案好处是可以归一化对三方件的依赖，便于维护，坏处是需要额外实现在stig存在而不在cis中存在的规则的扫描脚本的开发。或者是反过来全依赖kube-bench，那就需要额外开发stig相关的规则。

## 所需基础设施（可选）

- 若采用聚合 API 方式提供 report 子资源端点，需要新增 APIService 配置与对应服务部署能力（可与 Operator 同进程实现）。