# MCO（众核抗干扰隔离与调度优化）端到端自动化测试

## 概述

本目录为 MCO（many-core-orchestrator）的 e2e 自动化测试，当前覆盖四类场景：

- **Kata 部署阶段**：Kata 自动化部署与节点预检。
- **指标采集分析阶段**：mco-collector 与 mco-analyzer 后端集成测试。
- **调度插件 Helm 模板测试**：Volcano 调度插件的 Helm template 渲染验证。
- **调度插件集群集成测试**：调度插件 Filter/Score 逻辑与全链路部署验证。

基于 Ginkgo v2 + Gomega + kubernetes client-go + Helm CLI，遵循仓库 `e2e/` 下统一的测试规范。

## 目录结构

```
many-core-orchestrator/
├── mco_suite_test.go                  # Suite 入口（BeforeSuite 初始化客户端/上下文）
├── helm_template_test.go              # Kata Helm template 渲染测试（KATA-IT-001~004）
├── mco_test.go                        # Kata 集群级集成测试（KATA-IT-006/008~012）
├── collector_test.go                  # mco-collector 指标采集、上报、降级与 DFX 测试
├── analyzer_test.go                   # mco-analyzer 计算、CR 写入、metrics 与热更新测试
├── scheduler_helm_template_test.go    # 调度插件 Helm template 渲染测试（SCHED-IT-001~005）
├── scheduler_filter_test.go           # 调度插件 Filter/Score 测试（SCHED-IT-006~012）
├── scheduler_fullpath_test.go         # 调度插件全链路测试（SCHED-IT-014~017）
├── utils/
│   ├── const.go                       # 常量（helm 配置、OCI 地址、标签键、RuntimeClass 名称）
│   └── mco_functions.go               # 工具函数（helm 操作、CR 操作、Pod 创建/等待等）
└── README.md                          # 本文件
```

## Chart 来源

Chart 已上传至远程 OCI 仓库，默认通过 OCI URL 安装：

```
oci://cr.openfuyao.cn/charts/many-core-orchestrator  版本 0.0.0-latest
```

代码中通过 `GetChartPath()` 返回 OCI URL，`HelmInstallLocalChart` 和 `HelmTemplateLocal` 自动检测 OCI 前缀并附加 `--version` 参数。如需覆盖：

- `MCO_CHART_PATH`：设为本地路径或其他 OCI URL
- `MCO_CHART_VERSION`：覆盖 chart 版本号

## 标签体系

| 标签 | 含义 |
|------|------|
| `many-core-orchestrator` | 特性级标签，本目录所有 Describe 均携带 |
| `helm-template` | Helm template 渲染测试，无需集群 |
| `virtual-machine` | 可在虚拟机上运行的测试 |
| `physical-machine` | 需要 KVM 物理机集群的测试 |
| `with-workload-cluster` | 需要工作负载集群 |
| `scheduler` | 调度插件相关测试 |
| `scheduler-filter` | 调度插件 Filter 集成测试 |
| `scheduler-fullpath` | 调度插件全链路集成测试 |
| `collector` | Collector 集成测试 |
| `analyzer` | Analyzer 集成测试 |

---

## 一、Kata 部署阶段

### 虚拟机可运行（标签 `helm-template` + `virtual-machine`）

仅做 `helm template` 本地渲染验证，不需要集群连接：

| 用例编号 | 用例名称 | 说明 |
|---------|---------|------|
| KATA-IT-001 | 默认值渲染——全量资源 | 检查 DaemonSet/ConfigMap/SA/CR/CRB/RuntimeClass/initContainer |
| KATA-IT-002 | 关闭 kata-deploy 子 chart | `kata-deploy.enabled=false` 后无 kata 相关资源 |
| KATA-IT-003 | 关闭 preflight | 无 initContainer、无 preflight ConfigMap、无 `/dev/kvm` volume |
| KATA-IT-004 | 禁用单个 shim | 禁用 `qemu-runtime-rs` 后无对应 RuntimeClass，其他 shim 不受影响 |

### 需要物理机（标签 `physical-machine` + `with-workload-cluster`）

| 用例编号 | 用例名称 | 说明 |
|---------|---------|------|
| KATA-IT-006 | 无 KVM 节点预检失败 | 检查 `kata-ready=false` 节点的 initContainer 日志含 fail；全 KVM 集群时 Skip |
| KATA-IT-007 | 混合节点预检 | 混合 KVM/非KVM 集群中，分别验证 kata-ready=true/false 节点的 initContainer 日志；非混合集群时 Skip |
| KATA-IT-008 | Helm 安装 kata-deploy | BeforeAll 中 helm install + 等待 DaemonSet Ready |
| KATA-IT-009 | RuntimeClass 注册验证 | 检查 kata-qemu/kata-clh 存在，handler 字段不为空 |
| KATA-IT-010 | Kata Pod 创建与运行 | 创建 `runtimeClassName: kata-qemu` 的 busybox Pod，验证 Running 且无错误事件 |
| KATA-IT-011 | Kata 运行时实际校验 | SSH 登录 Kata Pod 所在节点，检查 kata-runtime/kata-proxy/qemu 等进程，确认 Pod 实际运行在 Kata VM 中 |
| KATA-IT-012 | 卸载与资源清理 | helm uninstall 后 DaemonSet 消失，runc Pod 不受影响 |

### Kata 阶段部署参数

`KataPhaseHelmSets` 自动注入：

```bash
--set collector.enabled=false
--set analyzer.enabled=false
--set mcoPlugin.enabled=false
```

---

## 二、指标采集分析阶段

### Collector 用例（标签 `collector` + `with-workload-cluster`）

| 用例编号 | 用例名称 | 说明 |
|---------|---------|------|
| mco-collector_FIT_1 | Collector 部署与健康验证 | DaemonSet 副本、Pod Running/Ready、无重启、日志健康 |
| mco-collector_FIT_2 | Collector 采集核心宿主机指标 | `/metrics` 暴露 LLC、BIO、PSI、snapshot timestamp 等核心指标 |
| mco-collector_FIT_3 | Collector 按节点标签暴露指标 | 每个 Collector Pod 暴露的指标带有对应 `node` 标签 |
| mco-collector_FIT_4 | Collector 向 Analyzer 流式上报快照 | 重启 Collector Pod 后 gRPC 重连，snapshot 字段完整且间隔约 1s |
| mco-collector_FIT_6 | Collector 指标端点不依赖 Analyzer | Analyzer 缩容后 `/metrics` 仍可访问且 timestamp 持续递增 |
| mco-collector_FIT_7 | Collector 识别 PSI 可用路径 | PSI 可用节点的指标和日志为实际采样值 |
| mco-collector_FIT_8 | Collector 执行 PSI 不可用降级采集 | PSI 不可用节点的降级指标、日志和可用性状态 |
| mco-collector_DFX_1 | Collector 长时间断链本地缓存与恢复 | Analyzer 不可用 >60s，本地 JSONL 缓存写入，恢复后自动重连 |
| mco-collector_DFX_3 | Collector 权限最小化验证 | 去掉 SYS_ADMIN 后 bcc/perf 采集器降级，恢复后回到 ok |

### Analyzer 用例（标签 `analyzer` + `with-workload-cluster`）

| 用例编号 | 用例名称 | 说明 |
|---------|---------|------|
| mco-analyzer_FIT_1 | Analyzer 部署与 CR 初始化 | Deployment/Pod/Service/gRPC/metrics 端口和 CR 初始化 |
| mco-analyzer_FIT_2 | Analyzer 识别缓存污染干扰 | 注入 MetricSnapshot，验证缓存污染检测和 CR 状态 |
| mco-analyzer_FIT_3 | Analyzer 识别 IO 队列拥塞干扰 | 注入高 BIO 延迟快照，验证 IO 拥塞检测和干扰等级 |
| mco-analyzer_FIT_4 | Analyzer 识别内存压力干扰 | 注入内存压力快照，验证内存压力检测和干扰等级 |
| mco-analyzer_FIT_5 | Analyzer 按权重计算干扰等级 | 权重计算、窗口平滑和单点毛刺数据处理 |
| mco-analyzer_FIT_6 | Analyzer 写入集群级 CR 状态 | CR `status.nodes` 包含 sequence、metrics、availability 等结构 |
| mco-analyzer_FIT_7 | Analyzer 重启后快速恢复 CR 更新 | 重启后旧数据保留并继续写入新 snapshot |
| mco-analyzer_FIT_8 | Analyzer 暴露计算产出指标 | `/metrics` 中干扰等级和 CR 更新相关指标 |
| mco-analyzer_FIT_9 | Analyzer 执行 PSI 不可用降级评估 | PSI 降级路径下 availability、降级值和检测结果 |
| mco-analyzer_FIT_10 | Analyzer 热更新 DetectionRuleConfig | 修改规则后 hash 更新、干扰等级变化并恢复默认 |

### 采集分析阶段部署参数

`CollectorPhaseHelmSets` 自动注入：

```bash
--set kata-deploy.enabled=false
--set mcoPlugin.enabled=false
```

---

## 三、调度插件测试

### Helm 模板测试（标签 `scheduler` + `helm-template` + `virtual-machine`）

仅做 `helm template` 本地渲染验证，不需要集群连接：

| 用例编号 | 用例名称 | 说明 |
|---------|---------|------|
| SCHED-IT-001 | 默认全量渲染 | 包含 mco-volcano-scheduler Deployment（`--plugins-dir`）、ConfigMap、SA/CR/CRB |
| SCHED-IT-002 | 关闭 mcoPlugin | `mcoPlugin.enabled=false` 后无 MCO 调度器资源 |
| SCHED-IT-003 | 一站式安装模式 | 包含 Volcano 标准资源 + mco-volcano-scheduler，默认 scheduler 被禁用 |
| SCHED-IT-004 | 接入已有 Volcano 模式 | 不包含 Volcano CRD/admission/controllers，包含 mco-volcano-scheduler |
| SCHED-IT-005 | ConfigMap 自动生成 | tiers[0] 含 mco-plugin + priority/gang/conformance，tiers[1] 含 overcommit/drf/predicates |

### Filter/Score 集成测试（标签 `scheduler-filter` + `with-workload-cluster`）

需要至少 2 个节点的集群，通过 `kubectl patch` 注入 CR 数据验证调度行为：

| 用例编号 | 用例名称 | 说明 |
|---------|---------|------|
| SCHED-IT-006 | 数据源不可用时降级调度 | 删除 CR 后 Pod 仍可正常调度，调度器不阻塞 |
| SCHED-IT-007 | Filter_Kata 双标签运行时检查 | 节点 A 缺少 kata-ready 标签被排除，Pod 调度到双标签节点 B |
| SCHED-IT-008 | Filter_PSI Memory Full 硬排除 | 节点 A psiMemoryFull=5.0 被排除，Pod 调度到节点 B |
| SCHED-IT-009 | Filter_PSI IO 硬排除 | 节点 A psiIoSome=30.0 被排除，Pod 调度到节点 B |
| SCHED-IT-010 | Filter_PSI 不可用时跳过约束 | 节点 A psiAvailable=false 时 PSI 约束被跳过，Pod 可调度到 A |
| SCHED-IT-011 | Score_干扰等级评分 | 节点 A interferenceLevel=0.2 得分高，Pod 优先调度到 A |
| SCHED-IT-012 | Score_数据不完整时中性分回退 | 节点 A status=PARTIAL 得中性分 50，Pod 调度到正常评分节点 B |

### 全链路集成测试（标签 `scheduler-fullpath` + `with-workload-cluster`）

Ordered 测试，按顺序执行安装→验证→卸载：

| 用例编号 | 用例名称 | 说明 |
|---------|---------|------|
| SCHED-IT-014 | Helm 安装部署 | helm install 成功，Pod Running，日志无 panic/fatal |
| SCHED-IT-015 | 调度器接管 | Volcano Job 由 mco-volcano-scheduler 正常调度 |
| SCHED-IT-016 | Filter/Score 决策日志验证 | 日志包含 filter/score 决策记录 |
| SCHED-IT-017 | 卸载与资源清理 | Deployment/ConfigMap/SA/ClusterRole/CRB 被正确清理 |

### 调度插件部署参数

`SchedulerPhaseHelmSets` 自动注入：

```bash
--set volcano.enabled=true
--set mcoPlugin.enabled=true
--set kata-deploy.enabled=false
--set collector.enabled=false
--set analyzer.enabled=false
```

---

## 环境配置

1. 复制 `e2e/.env.template` 为 `e2e/.env`
2. 确保 `KUBECONFIG` 指向集群的 kubeconfig（默认 `~/.kube/config`）
3. 如需覆盖 chart 来源：
   - `MCO_CHART_PATH`：chart 路径或 OCI URL（默认 `oci://cr.openfuyao.cn/charts/many-core-orchestrator`）
   - `MCO_CHART_VERSION`：chart 版本号（默认 `0.0.0-latest`）

## 运行测试

```bash
cd e2e

# Kata Helm template 渲染测试（虚拟机可跑）
ginkgo -v --label-filter="helm-template && virtual-machine" ./many-core-orchestrator

# Kata 物理机集群测试
ginkgo -v --label-filter="physical-machine" ./many-core-orchestrator

# Collector/Analyzer 集成测试
ginkgo -v --label-filter="collector || analyzer" ./many-core-orchestrator

# 调度插件 Helm template 渲染测试（虚拟机可跑）
ginkgo -v --label-filter="scheduler && helm-template" ./many-core-orchestrator

# 调度插件 Filter 集成测试（需集群）
ginkgo -v --label-filter="scheduler-filter" ./many-core-orchestrator

# 调度插件全链路集成测试（需集群）
ginkgo -v --label-filter="scheduler-fullpath" ./many-core-orchestrator

# 全部调度插件测试
ginkgo -v --label-filter="scheduler" ./many-core-orchestrator

# 全部测试
ginkgo -v ./many-core-orchestrator

# 跳过临时禁用的测试
ginkgo -v --skip-label="skip-temporarily" ./many-core-orchestrator
```
