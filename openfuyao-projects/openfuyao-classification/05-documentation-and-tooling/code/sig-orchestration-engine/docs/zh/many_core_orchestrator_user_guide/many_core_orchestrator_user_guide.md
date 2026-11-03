# Many-Core Orchestrator（MCO）

## 特性介绍

Many-Core Orchestrator（MCO，众核调度编排系统）面向众核混部集群，提供宿主机干扰指标采集、节点干扰分析、Volcano干扰感知调度，以及可选的Kata Containers虚拟机级隔离能力。它的目标是在不修改应用代码的前提下，降低在线业务在I/O、LLC缓存和内存压力混部场景下的长尾延迟抖动，并提高节点安全混部密度。

### 应用场景

MCO适合在线延迟敏感业务和离线吞吐型业务共用同一批众核节点的场景，例如Redis、实时推理、搜索服务与批处理、全量查询、I/O压测任务混部。系统重点感知以下干扰信号：

- LLC cache miss rate与LLC occupancy，用于识别缓存污染。
- block I/O latency P95/P99，用于识别块设备队列拥塞。
- PSI I/O与PSI memory，用于识别资源等待压力。
- PSI不可用时，使用MemAvailable ratio与page scan rate做降级评估。

MCO不会自动判断业务类型，也不会强制把某类Pod切换到Kata。是否使用Kata由用户在Pod中设置`runtimeClassName`决定。

### 能力范围

MCO由四类核心组件组成：

**表1** 核心组件

| 组件 | 部署形态 | 作用 |
|---|---|---|
| Collector | DaemonSet | 在每个节点采集LLC、I/O、PSI、内存回收等宿主机指标，并通过gRPC上报Analyzer。 |
| Analyzer | Deployment | 对Collector样本做窗口平滑、规则检测和干扰等级计算，并写入`NodeInterferenceReport`。 |
| MCO Volcano Plugin | Volcano scheduler插件 | 在Filter阶段排除高压力或不满足Kata条件的节点，在Score阶段优先选择低干扰节点。 |
| Kata Deploy | DaemonSet子chart | 检查节点虚拟化能力，安装/注册Kata runtime handler，并创建RuntimeClass。 |

MCO使用两个CRD：

**表2** CRD说明

| CRD | Scope | 简写 | 说明 |
|---|---|---|---|
| `NodeInterferenceReport` | Cluster | `nir` | 保存所有节点的最新干扰等级、指标快照、检测原因和采样状态。默认对象名为`cluster`。 |
| `DetectionRuleConfig` | Namespaced | `drc` | 保存干扰检测阈值、权重、惩罚系数和PSI降级规则。默认在`mco-system`命名空间，名称为`cluster`。 |

### 亮点特征

- **低改造接入**：用户只需在Pod中设置`runtimeClassName`即可接入Kata虚拟机隔离，无需修改应用代码或引入Sidecar。
- **微架构级干扰感知**：突破传统调度器仅基于CPU/内存余量的局限，直接采集LLC miss rate、block I/O P99 latency、PSI等底层微架构指标，识别Cgroups无法隔离的硬件级资源竞争。
- **调度 + 隔离双层防护**：干扰感知调度从选节点阶段规避高干扰节点；Kata虚拟机级隔离从物理层面切断离线业务对在线业务的串扰。两者可独立或组合使用。
- **故障安全降级**：Analyzer不可用或干扰报告过期时，调度插件自动回退为基础评分，不阻断正常业务部署。
- **可观测性强**：Collector和Analyzer各自暴露Prometheus`/metrics`端点，干扰报告通过CR对外呈现，调度插件提供过滤原因与评分分布等指标，便于持续调优。

### 基本概念

**数据链路：**

```text
Collector -> Analyzer -> NodeInterferenceReport CR -> Volcano mco-plugin -> Pod调度落点
```

- **Collector** 以DaemonSet形态在每个节点运行，通过eBPF/perf采集宿主机微架构指标，经gRPC流式推送至Analyzer。
- **Analyzer** 接收所有节点的指标快照，做窗口平滑后依据`DetectionRuleConfig`中的阈值与权重进行干扰检测和等级计算，最终将所有节点的干扰报告汇总写入一份集群级`NodeInterferenceReport`CR（名称固定为`cluster`）。
- **MCO Plugin** 作为Volcano调度器插件，在Filter阶段排除高压力或不满足Kata条件的节点，在Score阶段按`score = 100 × (1 - interferenceLevel)`评分，低干扰节点得分更高。
- **Kata Deploy** 自动检查节点虚拟化条件，安装Kata运行时并注册RuntimeClass，完成后为节点打上`mco.many-core.io/kata-ready=true`标签。

**干扰等级计算：**

Analyzer将干扰归纳为三类维度——缓存污染（LLC miss rate × LLC occupancy）、内存压力（PSI memory some/full）和I/O队列拥塞（bio latency P99 + PSI I/O some），分别归一化至 [0, 1] 区间后按权重加权求和，并叠加动态惩罚系数（复合干扰时放大权重），最终输出节点干扰等级$L \in [0, 1]$。PSI不可用时自动切换至降级路径，使用MemAvailable比例和page scan rate替代PSI指标。

### 实现原理

1. **指标采集**：Collector每秒采集一次宿主机指标（LLC miss/occupancy、bio latency P95/P99、PSI I/O/memory），通过gRPC流推送至Analyzer。
2. **干扰分析**：Analyzer对窗口内（默认 10s）的样本取均值平滑，基于`DetectionRuleConfig`CR中的阈值判定是否触发缓存污染、I/O 队列拥塞、内存压力三类检测，计算归一化干扰指数并加权求和得出干扰等级。
3. **报告写入**：Analyzer定期（默认 1s）将各节点的干扰等级、原始指标快照、检测明细写入`NodeInterferenceReport/cluster`CR的`status.nodes`中。
4. **调度决策**：MCO Plugin通过Kubernetes informer侦听`NodeInterferenceReport/cluster`，在Filter阶段排除PSI memory full > 0 或 PSI I/O some 超过硬约束阈值的节点；在Score阶段按`score = 100 × (1 - interferenceLevel)`评分。
5. **Kata隔离**：用户在Pod中设置`runtimeClassName: kata-*`后，调度插件确保Pod只落到Kata就绪节点，容器运行时以独立Guest OS运行，从物理层面切断LLC缓存污染和I/O队列串扰。

### 与相关特性的关系

**表3** 与相关特性的关系

| 相关特性 | 关系 |
|---|---|
| **Volcano** | MCO Plugin作为Volcano scheduler的插件运行，依赖Volcano的Session/Plugin机制。MCO Chart可随带安装Volcano。 |
| **Kata Containers** | MCO通过kata-deploy自动部署和注册Kata运行时，用户在Pod中按需选择。Kata隔离是可选的增强能力。 |
| **Kubernetes调度器** | MCO不修改或替换kube-scheduler，仅在Volcano调度链路中增强Filter/Score逻辑。未被Volcano接管的Pod不受影响。 |
| **Prometheus** | Collector和Analyzer各自暴露`/metrics`端点，通过ServiceMonitor或手动配置scrape job接入Prometheus。 |

## 安装

### 前提条件

- Kubernetes集群，建议v1.33及以上版本。
- Helm 3.x。
- 如需干扰感知调度，需要Volcano 1.9.0。Chart可随MCO一起安装Volcano。
- 如需Kata隔离，节点需要支持KVM虚拟化，并允许kata-deploy完成运行时注册。
- Collector依赖宿主机能力采集指标，默认会挂载`/sys`、`/lib/modules`、`/usr/src` 和`/var/log/mco-collector`，并使用`perf`与`biolatency`命令。
- ServiceMonitor依赖集群已安装Prometheus Operator或兼容的ServiceMonitor CRD。如果集群未配置Prometheus，请关闭`collector.serviceMonitor.enabled`和`analyzer.serviceMonitor.enabled`。

### 开始安装

拉取Chart（所有场景共用）：

```bash
helm pull oci://cr.openfuyao.cn/charts/many-core-orchestrator --version 0.0.0-latest
```

以下按场景选择安装命令。Chart默认启用全部组件，`--set`仅列出需要**覆盖缺省值**的关键参数。

#### 场景一：全功能部署（推荐）

部署采集、分析、调度插件、Kata隔离全套组件：

```bash
helm install mco many-core-orchestrator-0.0.0-latest.tgz
```

>![image](figures/icon-note.gif)**说明：**
> - 默认会同时安装Volcano（含CRD、controller、admission），并禁用其自带scheduler，由MCO scheduler接管。
> - 如果集群没有Prometheus Operator，请追加关闭ServiceMonitor：
```bash
helm install mco many-core-orchestrator-0.0.0-latest.tgz \
  --set collector.serviceMonitor.enabled=false \
  --set analyzer.serviceMonitor.enabled=false
```

#### 场景二：只部署Kata隔离

```bash
helm install mco many-core-orchestrator-0.0.0-latest.tgz \
  --set collector.enabled=false \
  --set analyzer.enabled=false \
  --set mcoPlugin.enabled=false
```

#### 场景三：只部署采集与分析（不做调度干预）

```bash
helm install mco many-core-orchestrator-0.0.0-latest.tgz \
  --set kata-deploy.enabled=false \
  --set mcoPlugin.enabled=false
```

#### 场景四：接入已有Volcano集群

如果集群已安装Volcano，只部署MCO组件并注册调度插件。

>![image](figures/icon-notice.gif)**注意：**<br />
> - 此场景下必须同时设置`volcano.enabled=false`（不重复安装Volcano子Chart）和`mcoPlugin.manageSchedulerConfig=false`（不渲染ConfigMap，保留已有配置）。**如果遗漏`volcano.enabled=false`（默认 `true`），Chart将重复安装Volcano子Chart并覆盖已有 `volcano-scheduler-configmap`，可能导致集群冲突和原有调度配置丢失。**
>
> - `Collector`、`Analyzer`、`Kata Deploy`、`MCO Plugin`默认均为启用状态，无需额外`--set`。

##### 对原有Volcano组件的影响

接入MCO会停用集群已有的核心调度器`volcano-scheduler`，改由MCO部署的`mco-volcano-scheduler`接管全部Volcano调度。各组件变更与影响如下。

**表4** 接入MCO对原有Volcano组件的影响

| 组件 | 变更内容 | 影响 |
|---|---|---|
| `volcano-scheduler`（Deployment） | 缩容至`replicas=0`，对象保留不删除 | 调度能力暂停。缩容后到`mco-volcano-scheduler`就绪前，新提交的Volcano Job/PodGroup将无法被调度，处于Pending状态。 |
| `mco-volcano-scheduler`（Deployment） | MCO新建，使用独立ServiceAccount`mco-scheduler` | 接管全部Volcano调度。就绪前集群无可用调度器，需等待其Running后业务才能恢复调度。 |
| `volcano-scheduler-configmap` | 由用户手动编辑注册`mco-plugin` | 若原调度器仍在运行时编辑，可能读取到未注册插件的配置；务必先停用原调度器再编辑。 |
| Volcano CRD、controller、admission | 不变更 | `volcano.enabled=false`仅跳过Volcano子Chart安装，已有CRD、controller、admission保持原状继续运行。 |
| `volcano-scheduler`相关Service、RBAC | 不变更 | 原`volcano-scheduler-service`、ClusterRole/ClusterRoleBinding等对象保留，但不再有Pod匹配，Service Endpoints为空。 |

>![image](figures/icon-notice.gif)**注意：**<br />
> - 原`volcano-scheduler`仅被缩容，未删除Deployment对象。如需回退，将其`replicas`恢复为原值并卸载MCO即可，但需同步从ConfigMap中移除`mco-plugin`配置段，避免原调度器因无法加载插件而启动失败。

##### 操作步骤

1. **停用原有Volcano scheduler**（避免两个调度器同时运行导致冲突）。
   ```bash
   kubectl scale deploy -n volcano-system volcano-scheduler --replicas=0
   ```
   若集群存在多个Volcano调度器实例（如多副本或HA部署），需将所有副本缩容至0，确保仅`mco-volcano-scheduler`参与调度，否则会出现重复调度或调度冲突。

2. **在已有volcano-scheduler-configmap中手动注册mco-plugin**。
   参考[调整调度插件参数](#调整调度插件参数)中的ConfigMap格式，在`tiers[0].plugins`中（`conformance`之后）插入`mco-plugin`段。
   ```bash
   kubectl edit configmap volcano-scheduler-configmap -n volcano-system
   ```

3. **安装MCO**（不重复安装Volcano，不托管scheduler config）。
   ```bash
   helm install mco many-core-orchestrator-0.0.0-latest.tgz \
     --set volcano.enabled=false \
     --set mcoPlugin.manageSchedulerConfig=false
   ```
   MCO会部署`mco-volcano-scheduler`（替代原`volcano-scheduler`），启动时即读取已编辑好的ConfigMap，无需额外重启。

如果先安装MCO再编辑ConfigMap，需执行如下命令重启`mco-volcano-scheduler`使配置生效。
```bash
kubectl rollout restart deploy mco-volcano-scheduler -n volcano-system
```
原`volcano-scheduler`保持缩容状态（`replicas=0`），由MCO部署的`mco-volcano-scheduler`接管所有Volcano调度。

#### 部署场景速查

**表5** 部署场景速查

| 场景 | Collector | Analyzer | Kata | 调度插件 | Volcano |
|---|---|---|---|---|---|
| 全功能（默认） | ✅ | ✅ | ✅ | ✅ | ✅ |
| 仅Kata隔离 | ❌ | ❌ | ✅ | ❌ | ✅ |
| 仅采集分析 | ✅ | ✅ | ❌ | ❌ | ✅ |
| 接入已有Volcano | ✅ | ✅ | ✅ | ✅ | ❌ |

>![image](figures/icon-notice.gif)**注意：**<br /> 
> - ✅：表示MCO Chart默认部署该组件。
> - ❌：表示MCO Chart默认不部署该组件。
> - 场景四中 Volcano=❌ 表示MCO Chart不安装Volcano子chart，但集群中**必须已有可用的Volcano**（含CRD、controller、admission），否则调度插件无法工作。

### 部署后检查

查看组件状态：

```bash
kubectl get pods -n mco-system
kubectl get pods -n volcano-system
```
> 正常则Collector、Analyzer Pod均为Running状态，Volcano scheduler Pod为Running状态。

查看CRD：

```bash
kubectl get crd | grep mco.many-core.io
```
> 正常则返回`nodeinterferencereports.mco.many-core.io`和`detectionruleconfigs.mco.many-core.io`两个CRD。

查看节点干扰报告：

```bash
kubectl get nir cluster -o yaml
```
> 正常则`status.nodes`中包含各节点的`interferenceLevel`、`metrics`和`detections`等字段。

查看检测规则：

```bash
kubectl get drc cluster -n mco-system -o yaml
```
> 正常则返回默认的检测阈值、权重和降级参数配置。

查看Kata节点能力标签：

```bash
kubectl get nodes -L katacontainers.io/kata-runtime,mco.many-core.io/kata-ready
```
> 正常则物理机节点标签值为true。

## 使用干扰感知调度与Kata隔离

### 前提条件

- 已完成MCO全功能部署（场景一）或已接入已有Volcano集群（场景四），确保Collector、Analyzer、MCO Plugin均正常运行。
- 如需使用Kata隔离，目标节点需具备KVM虚拟化能力且`mco.many-core.io/kata-ready=true`。
- 待调度的Pod需通过Volcano调度（即由Volcano scheduler接管），否则MCO Plugin不会参与调度决策。

### 背景信息

部署MCO后，系统的核心工作流程如下：

1. **Collector→Analyzer→CR** 链路持续运行，`NodeInterferenceReport/cluster`中每个节点每 1s 更新一次干扰等级（0.0~1.0）。
2. 当通过Volcano调度的Pod进入调度流程时，MCO Plugin自动介入Filter和Score阶段。
3. 用户可通过`kubectl get nir cluster`随时查看各节点干扰状态，通过`kubectl edit drc cluster -n mco-system`调整检测规则。

关键对象：
- `NodeInterferenceReport/cluster`：集群级干扰报告，`status.nodes`以节点名为key。
- `DetectionRuleConfig/mco-system/cluster`：检测规则，Analyzer热加载，变更即时生效。

### 使用限制

MCO的降级策略以“不阻断业务调度”为原则，以下限制需在使用前了解：

- Analyzer不可用或`NodeInterferenceReport`超过TTL未刷新时，调度插件返回中性分`50`，节点不会被排除，但干扰感知能力暂时失效。
- 某节点没有报告时，该节点不因缺少干扰数据被过滤。
- PSI不可用时，调度插件跳过PSI硬约束（`psiMemoryFull`和 `psiIoSome`过滤）；Analyzer使用内存可用比例、page scan rate与bio latency继续计算干扰等级。
- Kata部署失败时，节点会缺少`mco.many-core.io/kata-ready=true`，只影响请求`runtimeClassName: kata-*`的Pod，不影响普通runc Pod。
- 检测阈值和权重均为基于实验或经验的初始值，不同机型的LLC、内存带宽、I/O栈能力差异会影响同一阈值的含义，建议按机型分别校准。

### 操作步骤

#### 启用干扰感知调度

部署MCO调度插件后，通过Volcano调度的Pod会自动参与干扰感知调度。插件行为分为两步：

1. **Filter**：如果节点PSI memory full或PSI I/O some超过硬约束阈值，则排除该节点；如果Pod请求Kata runtime，则节点必须同时具备`katacontainers.io/kata-runtime=true`和`mco.many-core.io/kata-ready=true`。
2. **Score**：读取`NodeInterferenceReport/cluster`中该节点的`interferenceLevel`，按`score = 100 * (1 - interferenceLevel)`评分。干扰等级越低，得分越高。

当CR数据缺失、过期或节点报告状态不是`ok`时，插件返回中性分`50`，不阻断正常调度。

#### 使用Kata虚拟机隔离

用户通过Pod的 `runtimeClassName` 选择Kata运行时。通常建议把离线高吞吐、高I/O或缓存扫描型负载放入Kata，以降低其对同节点在线业务的串扰。

示例：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: etl-batch
  labels:
    mco.many-core.io/workload-type: offline
spec:
  runtimeClassName: kata-qemu
  containers:
  - name: job
    image: example/etl:latest
```

`mco.many-core.io/workload-type`是预留标签，当前不会影响调度决策，可用于业务分类和后续策略扩展。

可用RuntimeClass由kata-deploy注册，常见名称包括`kata-qemu`、`kata-clh`等，具体以集群实际对象为准：

```bash
kubectl get runtimeclass
```

#### 查看干扰报告

`NodeInterferenceReport/cluster`的`status.nodes`以节点名为key，保存每个节点的最新状态。核心字段如下：

**表6** 干扰报告核心字段说明

| 字段 | 说明 |
|---|---|
| `interferenceLevel` | 干扰等级，范围 0.0 到 1.0，越高表示节点越拥塞。 |
| `psiAvailable` | 当前节点PSI文件是否可用。 |
| `status` | 报告状态，正常为`ok`。 |
| `metrics.llcMissRate` | LLC miss rate。 |
| `metrics.llcOccupancy` | LLC occupancy。 |
| `metrics.bioLatencyP95Ms` | block I/O P95 延迟，单位ms。 |
| `metrics.bioLatencyP99Ms` | block I/O P99 延迟，单位ms。 |
| `metrics.psiIoSome` | PSI I/O some avg10。 |
| `metrics.psiIoFull` | PSI I/O full avg10。 |
| `metrics.psiMemorySome` | PSI memory some avg10。 |
| `metrics.psiMemoryFull` | PSI memory full avg10。 |
| `metrics.memAvailableRatio` | 可用内存比例，用于PSI不可用时的降级判断。 |
| `metrics.pageScanRate` | 页面回收速率，单位pages/s。 |
| `detections` | Analyzer命中的检测类型和解释。 |
| `availability` | 各采集器/指标的可用性、状态和失败原因。 |
| `sequence` | 节点样本序号，用于判断报告是否刷新。 |
| `sampleCount` | 当前平滑窗口中的样本数。 |
| `windowSeconds` | 平滑窗口长度。 |

执行如下命令，快速查看节点干扰等级：

```bash
kubectl get nir cluster -o jsonpath='{.status.nodes}'
```

查看某个节点的详细报告：

```bash
kubectl get nir cluster -o yaml
```

#### 调整检测规则

默认规则由Helm渲染为`DetectionRuleConfig/mco-system/cluster`。Analyzer通过informer侦听该对象，变更后自动热加载；解析失败或对象被删除时保留上一版有效规则。

以下为默认配置（同时也是`kubectl edit drc cluster -n mco-system`看到的完整结构）：

```yaml
spec:
  # —— 三维度权重（总和应接近 1.0） ——
  weights:
    cache: 0.30        # LLC 缓存污染权重
    memory: 0.30       # 内存压力权重
    io: 0.40           # I/O 拥塞权重（通常影响最大）

  # —— 各维度饱和上限（归一化分母） ——
  saturation:
    cacheProduct: 0.175
    psiMemory: 50
    bioLatencyP99Ms: 100
    psiIo: 20

  # —— 检测触发阈值（超过即标记对应干扰类型） ——
  thresholds:
    cacheMissRate: 0.20
    llcOccupancy: 0.70
    bioLatencyP99Ms: 50      # 常调：I/O敏感业务可降低
    psiIo: 20                # 常调：与bioLatencyP99Ms配合
    psiMemorySome: 10
    psiMemoryFull: 0
    ioPenaltyP99Ms: 50       # 动态权重触发条件
    ioPenaltyPsiIo: 20
    memIoPenaltyMemory: 50
    memIoPenaltyPsiIo: 20

  # —— 动态惩罚系数（复合干扰时放大权重） ——
  penalties:
    ioMultiplier: 1.2
    memIoMultiplier: 1.1

  # —— PSI不可用时的降级内存评估 ——
  degradedMemory:
    memAvailableRatioThreshold: 0.30
    pageScanRateThreshold: 10000
```

**表7** 常用调优场景

| 场景 | 调整字段 | 建议方向 |
|---|---|---|
| 在线业务对I/O延迟极度敏感 | `thresholds.bioLatencyP99Ms`、`thresholds.psiIo` | 降低阈值（如 P99→30ms），让Analyzer更早标记I/O拥塞 |
| 节点内存充裕、PSI memory频繁误报 | `thresholds.psiMemorySome` | 适当提高（如 10→20） |
| 某机型LLC偏小、cache miss基线偏高 | `thresholds.cacheMissRate` | 按机型基线校准（如 0.20→0.30） |
| I/O与内存叠加干扰需更激进规避 | `penalties.ioMultiplier`、`penalties.memIoMultiplier` | 适度提高惩罚系数 |

**修改方式：**

```bash
# 方式一：Helm 升级（推荐，保持GitOps可追溯）
helm upgrade mco many-core-orchestrator-0.0.0-latest.tgz \
  --set analyzer.detectionRules.thresholds.bioLatencyP99Ms=30 \
  --set analyzer.detectionRules.thresholds.psiIo=15

# 方式二：直接编辑CR（即时生效，适合调试）
kubectl edit drc cluster -n mco-system
```

#### 调整调度插件参数

插件行为由Volcano`volcano-scheduler-configmap`中的`mco-plugin`配置段控制。MCO Chart托管配置时，通过`.Values.mcoPlugin.config`渲染。接入已有Volcano时，直接编辑ConfigMap注册插件。

**ConfigMap格式（`kubectl edit configmap volcano-scheduler-configmap -n volcano-system`看到的实际结构）**

```yaml
data:
  volcano-scheduler.conf: |
    actions: "enqueue, allocate, backfill"
    tiers:
    - plugins:
      - name: priority
      - name: gang
      - name: conformance
      - name: mco-plugin
        arguments:
          mco-plugin:
            crName: "cluster"               # 固定，不建议修改
            crStatusTTLSeconds: 30           # CR缓存TTL
            hardConstraints:
              psiMemoryFullAvg10: 0          # memory full硬过滤阈值
              psiIoSomeAvg10Threshold: 20    # I/O some硬过滤阈值
            fallbackWhenCROutdated: true     # CR过期时降级（当前始终生效）
    - plugins:
      - name: overcommit
      - name: drf
      - name: predicates
      - name: proportion
      - name: nodeorder
      - name: binpack
```

**Values.yaml对应路径（Helm`--set`方式使用）**

```yaml
mcoPlugin:
  config:
    crName: "cluster"
    crStatusTTLSeconds: 30
    hardConstraints:
      psiMemoryFullAvg10: 0.0
      psiIoSomeAvg10Threshold: 20.0
    fallbackWhenCROutdated: true
```

>![image](figures/icon-notice.gif)**注意：**<br />
> - `crName` 虽然可配，但插件informer当前硬编码只处理名为`cluster`的`NodeInterferenceReport`，不建议修改。
> - 接入已有Volcano时（`manageSchedulerConfig=false`），需按 **ConfigMap格式** 手动在tiers中注册`mco-plugin`，而非通过values.yaml。

**表8** 调度插件常用调参

| 参数 | 缺省值 | Helm `--set` 路径 | ConfigMap路径 | 说明 |
|---|---:|---|---|---|
| `crStatusTTLSeconds` | `30` | `mcoPlugin.config.crStatusTTLSeconds` | `arguments.mco-plugin.crStatusTTLSeconds` | CR缓存过期秒数。Collector/Analyzer抖动时可调大。 |
| `psiMemoryFullAvg10` | `0.0` | `mcoPlugin.config.hardConstraints.psiMemoryFullAvg10` | `arguments.mco-plugin.hardConstraints.psiMemoryFullAvg10` | 大于该值的节点在Filter阶段直接排除。 |
| `psiIoSomeAvg10Threshold` | `20.0` | `mcoPlugin.config.hardConstraints.psiIoSomeAvg10Threshold` | `arguments.mco-plugin.hardConstraints.psiIoSomeAvg10Threshold` | PSI I/O硬过滤阈值。I/O敏感业务可降低（如 10.0）。 |

修改示例：

```bash
# 方式一：Helm升级（推荐，Chart托管ConfigMap时使用）
helm upgrade mco many-core-orchestrator-0.0.0-latest.tgz \
  --reuse-values \
  --set mcoPlugin.config.crStatusTTLSeconds=60 \
  --set mcoPlugin.config.hardConstraints.psiIoSomeAvg10Threshold=15.0

# 方式二：直接编辑ConfigMap（需重启scheduler Pod生效）
kubectl edit configmap volcano-scheduler-configmap -n volcano-system
kubectl rollout restart deploy mco-volcano-scheduler -n volcano-system
```

### 后续操作

部署并启用MCO后，建议通过以下方式持续观测系统状态：

- 监控`mco_node_interference_level{node}`指标，观察干扰等级与业务P99延迟的相关性。
- 通过`mco_scheduler_filter_rejections{node,reason}`和`mco_scheduler_score_distribution{node,dimension}`指标了解调度决策。
- 定期查看`NodeInterferenceReport/cluster`中`detections`字段，了解命中干扰类型。

### 相关操作

- **卸载MCO**：`helm uninstall mco`。
- **升级MCO**：`helm upgrade mco many-core-orchestrator-0.0.0-latest.tgz --reuse-values`。
- **禁用特定组件**：通过Helm`--set`参数关闭Collector、Analyzer、Kata或调度插件，参见[开始安装](#开始安装)中的场景选择。
- **临时禁用干扰感知**：将调度插件配置中的`psiIoSomeAvg10Threshold`设为一个极大值（如100.0），使I/O硬过滤基本不触发；PSI memory full硬过滤默认阈值为0.0（即仅当PSI memory full > 0时触发），可维持缺省值。

## 附录

### 关键Helm参数

**表9** 关键Helm参数

| 参数 | 缺省值 | 说明 |
|---|---|---|
| `global.imageRegistry` | `cr.openfuyao.cn/openfuyao` | 全局镜像仓库。 |
| `global.systemNamespace` | `mco-system` | MCO组件命名空间。 |
| `volcano.enabled` | `true` | 是否安装Volcano子chart。 |
| `kata-deploy.enabled` | `true` | 是否安装Kata Deploy子chart。 |
| `analyzer.enabled` | `true` | 是否部署Analyzer。 |
| `analyzer.grpc.port` | `50051` | Analyzer gRPC服务端口。 |
| `analyzer.metrics.port` | `9878` | Analyzer metrics端口。 |
| `analyzer.smoothWindow` | `10s` | Analyzer样本平滑窗口。 |
| `analyzer.crFlushInterval` | `1s` | Analyzer写入`NodeInterferenceReport`的间隔。 |
| `collector.enabled` | `true` | 是否部署Collector。 |
| `collector.collectInterval` | `1s` | Collector采集间隔。 |
| `collector.sampleDuration` | `1s` | 单次采样持续时间。 |
| `collector.metricsPort` | `9877` | Collector metrics端口。 |
| `collector.collectors.llcMissRate` | `true` | 是否采集LLC miss rate。 |
| `collector.collectors.llcOccupancy` | `true` | 是否采集LLC occupancy。 |
| `collector.collectors.ioLatency` | `true` | 是否采集block I/O latency。 |
| `collector.collectors.psi` | `true` | 是否采集PSI。 |
| `collector.commands.perf` | `/usr/bin/perf` | Helm部署时注入的perf命令路径；程序未通过Chart注入时默认使用`perf`。 |
| `collector.commands.biolatency` | `/usr/share/bcc/tools/biolatency` | Helm部署时注入的biolatency命令路径；程序未通过Chart注入时默认使用`biolatency`。 |
| `mcoPlugin.enabled` | `true` | 是否启用MCO调度插件。 |
| `mcoPlugin.manageSchedulerConfig` | `true` | 是否由Chart渲染`volcano-scheduler-configmap`。接入已有Volcano时必须设为`false`，否则会覆盖已有配置。 |

### 环境变量

**表10** Collector环境变量

Collector主要环境变量：

| 环境变量 | 缺省值 | 说明 |
|---|---|---|
| `NODE_NAME` | hostname | 节点名，Chart中来自`spec.nodeName`。 |
| `ANALYZER_ADDR` | `mco-analyzer.mco-system.svc.cluster.local:50051` | Analyzer gRPC地址。 |
| `COLLECT_INTERVAL` | `1s` | 采集周期。 |
| `SAMPLE_DURATION` | 同`COLLECT_INTERVAL` | 单次采样持续时间。 |
| `CACHE_PATH` | `/var/log/mco-collector/metrics.jsonl` | 本地JSONL缓存路径。 |
| `PERF_CMD` | `perf` | perf命令路径。 |
| `BIOLATENCY_CMD` | `biolatency` | biolatency命令路径。 |
| `COLLECTOR_METRICS_PORT` | `9877` | metrics端口。 |
| `METRICS_ADDR` | 空 | Collector自身`/metrics`HTTP服务侦听地址；设置后优先于`COLLECTOR_METRICS_PORT`。 |
| `GRPC_TIMEOUT` | `3s` | Collector上报Analyzer的gRPC超时时间。 |
| `COLLECTOR_LLC_MISS_RATE_ENABLED` | `true` | 是否启用LLC miss rate采集。 |
| `COLLECTOR_LLC_OCCUPANCY_ENABLED` | `true` | 是否启用LLC occupancy采集。 |
| `COLLECTOR_IO_LATENCY_ENABLED` | `true` | 是否启用block I/O latency采集。 |
| `COLLECTOR_PSI_ENABLED` | `true` | 是否启用PSI采集。 |

**表11** Analyzer环境变量

Analyzer主要环境变量：

| 环境变量 | 缺省值 | 说明 |
|---|---|---|
| `GRPC_ADDR` | `:50051` | gRPC侦听地址。 |
| `METRICS_ADDR` | `:9878` | Analyzer自身`/metrics`HTTP服务侦听地址。 |
| `SMOOTH_WINDOW` | `10s` | 平滑窗口。 |
| `REPORT_TTL` | `60s` | Analyzer报告TTL。 |
| `CR_FLUSH_INTERVAL` | `1s` | CR写入间隔。 |
| `DETECTION_RULES_NAMESPACE` | `mco-system` | 检测规则所在命名空间。 |
| `DETECTION_RULES_NAME` | `cluster` | 检测规则对象名。 |
| `DETECTION_RULES_RESYNC_INTERVAL` | `30s` | `DetectionRuleConfig`informer resync间隔。 |
| `DETECTION_RULES_GROUP` | `mco.many-core.io` | 检测规则CRD API group。 |
| `DETECTION_RULES_VERSION` | `v1alpha1` | 检测规则CRD API version。 |
| `DETECTION_RULES_RESOURCE` | `detectionruleconfigs` | 检测规则CRD resource名称。 |
| `ENABLE_CR_UPDATE` | `true` | 是否写入 `NodeInterferenceReport`。 |
| `CR_NAME` | `cluster` | 干扰报告对象名。 |
| `CR_GROUP` | `mco.many-core.io` | 干扰报告CRD API group。 |
| `CR_VERSION` | `v1alpha1` | 干扰报告CRD API version。 |
| `CR_RESOURCE` | `nodeinterferencereports` | 干扰报告CRD resource名称。 |

### 可观测性指标

Collector默认在`9877`暴露`/metrics`：

**表12** Collector可观测性指标

| 指标 | 说明 |
|---|---|
| `mco_llc_miss_rate_raw{node}` | LLC miss rate。 |
| `mco_llc_occupancy_raw{node}` | LLC occupancy。 |
| `mco_bio_latency_p95_ms_raw{node}` | block I/O P95 延迟。 |
| `mco_bio_latency_p99_ms_raw{node}` | block I/O P99 延迟。 |
| `mco_psi_io_some_avg10_raw{node}` | PSI I/O some avg10。 |
| `mco_psi_memory_some_avg10_raw{node}` | PSI memory some avg10。 |
| `mco_mem_available_ratio_raw{node}` | 可用内存比例。 |
| `mco_pgscan_rate_pages_per_second_raw{node}` | 页面回收速率。 |
| `mco_collector_psi_available{node}` | PSI是否可用，1 表示可用。 |
| `mco_snapshot_status{node}` | 采样快照状态，1 表示正常。 |
| `mco_cache_references_raw{node}` | perf采集到的 cache references原始值。 |
| `mco_cache_misses_raw{node}` | perf采集到的 cache misses原始值。 |
| `mco_mem_total_kb_raw{node}` | `/proc/meminfo`中的MemTotal。 |
| `mco_mem_available_kb_raw{node}` | `/proc/meminfo`中的MemAvailable。 |
| `mco_bio_samples_raw{node}` | biolatency直方图样本数。 |
| `mco_psi_io_full_avg10_raw{node}` | PSI I/O full avg10。 |
| `mco_psi_memory_full_avg10_raw{node}` | PSI memory full avg10。 |
| `mco_collector_status{node,collector}` | 单个采集器状态，1 表示正常。 |
| `mco_snapshot_timestamp_ms{node}` | 最新采样快照 Unix毫秒时间戳。 |

Analyzer默认在`9878`暴露`/metrics`：

**表13** Analyzer可观测性指标

| 指标 | 说明 |
|---|---|
| `mco_node_interference_level{node}` | Analyzer计算后的节点干扰等级。 |
| `mco_analyzer_report_generation_duration_seconds` | 报告生成耗时。 |
| `mco_analyzer_cr_updates_total{status}` | CR更新次数。 |
| `mco_analyzer_collector_samples_total{node}` | 接收到的节点样本数。 |
| `mco_analyzer_grpc_requests_total{method,status}` | gRPC请求统计。 |
| `mco_analyzer_rule_reload_total{status}` | 规则重载次数。 |
| `mco_analyzer_cr_last_success_timestamp_seconds` | 最近一次成功写入CR的时间戳。 |
| `mco_analyzer_invalid_metric_values_total{node,metric,reason}` | Analyzer清洗非法指标值的计数（reason: nan/inf）。 |
| `mco_analyzer_rule_config_info{hash,path}` | 当前生效检测规则配置的信息（path为CR资源路径）。 |
| `mco_analyzer_rule_last_reload_timestamp_seconds` | 最近一次规则成功加载的时间戳。 |

Chart默认支持ServiceMonitor，可由Prometheus Operator自动发现。

如果集群没有安装Prometheus Operator或ServiceMonitor CRD，需要关闭相关配置：

```bash
helm upgrade mco many-core-orchestrator-0.0.0-latest.tgz \
  --set collector.serviceMonitor.enabled=false \
  --set analyzer.serviceMonitor.enabled=false
```

### 构建镜像

```bash
docker build -f build/analyzer.Dockerfile -t mco-analyzer:latest .
docker build -f build/collector.Dockerfile -t mco-collector:latest .
docker build -f build/scheduler.Dockerfile -t mco-volcano-scheduler:latest .
```

也可以使用 Makefile：

```bash
make build-analyzer
make build-collector
make build-plugin-so
make build-scheduler
```

开发验证：

```bash
go test ./...
```

### 最佳实践

- 将在线业务保留在默认runc或既有运行时，将高I/O、高内存带宽、缓存扫描型离线任务显式设置为`runtimeClassName: kata-*`。
- 初始上线时保守调整`psiIoSomeAvg10Threshold`和`bioLatencyP99Ms`，先观察`mco_node_interference_level`与业务P99的相关性。
- 对不同机型分别校准阈值。LLC、内存带宽、I/O 栈能力差异会影响同一阈值的含义。
- 保持`fallbackWhenCROutdated=true`，避免监控链路异常影响业务创建。
- 通过`detections`字段区分cache pollution、I/O queue contention和memory pressure，再决定是调度规避、阈值调整，还是把离线负载迁入Kata。

## FAQ

1. Pod没有使用MCO插件调度。

   ```bash
   kubectl get configmap volcano-scheduler-configmap -n volcano-system -o yaml
   kubectl logs -n volcano-system deploy/mco-volcano-scheduler
   ```

   确认配置中包含 `mco-plugin`，并且使用的是包含插件的scheduler镜像。

2. 没有生成干扰报告。

   ```bash
   kubectl logs -n mco-system deploy/mco-analyzer
   kubectl logs -n mco-system ds/mco-collector
   kubectl get nir cluster -o yaml
   ```

   重点检查Collector是否能连接`ANALYZER_ADDR`，Analyzer是否有写`nodeinterferencereports/status`的权限。

3. PSI指标不可用。

   ```bash
   kubectl get nir cluster -o yaml
   ```

   查看目标节点的`psiAvailable`和`availability`字段。PSI不可用时系统仍会降级计算，但PSI相关Filter不会生效。

4. Kata Pod无法调度。

   ```bash
   kubectl get runtimeclass
   kubectl get nodes -L katacontainers.io/kata-runtime,mco.many-core.io/kata-ready
   kubectl describe pod <pod-name> -n <namespace>
   ```

   请求`runtimeClassName: kata-*`的Pod只会调度到同时具有`katacontainers.io/kata-runtime=true`和`mco.many-core.io/kata-ready=true`的节点。

5. Collector指标异常。

   ```bash
   kubectl logs -n mco-system ds/mco-collector
   kubectl get ds mco-collector -n mco-system -o yaml
   ```

   重点检查宿主机是否具备`perf`、`biolatency`、内核模块目录和所需采集权限。

   I/O延迟指标由eBPF方式采集，通常通过`biolatency`工具完成。如果I/O指标采集失败，请重点检查节点是否安装了与当前运行内核版本匹配的kernel header，例如确认`/lib/modules/$(uname -r)/build` 是否存在。缺少匹配的kernel header时，eBPF程序可能无法编译或加载。

   **配置方法：**

   ```bash
   yum install kernel-devel-$(uname -r)
   ```

   安装后验证以下两个路径出现：
   ```bash
   ls -l /lib/modules/$(uname -r)/build
   ls -ld /usr/src/kernels/$(uname -r) 
   ```

   并且以下三个命令返回版本匹配：
   ```bash
   uname -r
   rpm -qa | grep kernel-devel
   ls -ld /lib/modules/$(uname -r)/build
   ```

   最后重启collector即可。

6. PSI指标不可用但希望启用。

   ```bash
   kubectl get nir cluster -o yaml
   ```

   如果节点报告中`psiAvailable: false`，请检查节点内核是否支持PSI，并确认启动参数已开启PSI。常见做法是在节点内核启动参数中加入`psi=1`，重启节点后再确认`/proc/pressure/cpu`、`/proc/pressure/memory`、`/proc/pressure/io` 是否存在。