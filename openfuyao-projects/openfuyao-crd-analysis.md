# openFuyao 自研 CRD 分析与插件框架关系

> 本文基于只读分析生成，未修改任何业务代码。

## 1. 范围约定

本文只统计 **openFuyao 自研/自行发布**的 CRD，即由仓库内 openFuyao 各服务或组件自己定义的 CRD。

**不统计的 CRD**：
- Kubernetes 原生资源与原生扩展（如 `Node`、`Pod`、`Service` 等）
- 上游/第三方生态随 chart 引入的 CRD，例如：
  - Volcano、Istio、Gateway API、Karmada、Calico、CDI
  - Ray / KubeRay、kube-prometheus-stack
  - node-feature-discovery（作为子 chart 引入）
  - kwok（用于测试）
  - mooncake、Moonshot 等外部项目
- 仅作为子 chart 依赖打包、但非 openFuyao 自研的 CRD

判断依据：CRD 的 Go types 文件是否在 openFuyao 自有模块内，或 CRD YAML 是否由 openFuyao 自有模块发布。

---

## 2. 全量自研 CRD 一览

当前仓库内识别出的 openFuyao 自研 CRD 共 **28 个 Kind**。按 apiVersion 分组列出：

| 序号 | Kind | apiVersion | Scope | Go 类型定义文件 | 部署用 CRD / YAML 文件 | 所属模块 | 类型 |
|---|---|---|---|---|---|---|---|
| 1 | `Command` | `bkeagent.bocloud.com/v1beta1` | Namespaced | `bkeagent/api/v1beta1/command_types.go` | `bkeagent/config/crd/bases/bkeagent.bocloud.com_commands.yaml` | bkeagent | 平台共享 |
| 2 | `BKECluster` | `bke.bocloud.com/v1beta1` | Namespaced | `cluster-api-provider-bke/api/capbke/v1beta1/bkecluster_types.go` | `cluster-api-provider-bke/config/crd/bases/bke.bocloud.com_bkeclusters.yaml` | cluster-api-provider-bke | 平台共享 |
| 3 | `BKEClusterTemplate` | `bke.bocloud.com/v1beta1` | Namespaced | `cluster-api-provider-bke/api/capbke/v1beta1/bkeclustertemplate_types.go` | `cluster-api-provider-bke/config/crd/bases/bke.bocloud.com_bkeclustertemplates.yaml` | cluster-api-provider-bke | 平台共享 |
| 4 | `BKEMachine` | `bke.bocloud.com/v1beta1` | Namespaced | `cluster-api-provider-bke/api/capbke/v1beta1/bkemachine_types.go` | `cluster-api-provider-bke/config/crd/bases/bke.bocloud.com_bkemachines.yaml` | cluster-api-provider-bke | 平台共享 |
| 5 | `BKEMachineTemplate` | `bke.bocloud.com/v1beta1` | Namespaced | `cluster-api-provider-bke/api/capbke/v1beta1/bkemachinetemplate_types.go` | `cluster-api-provider-bke/config/crd/bases/bke.bocloud.com_bkemachinetemplates.yaml` | cluster-api-provider-bke | 平台共享 |
| 6 | `BKENode` | `bke.bocloud.com/v1beta1` | Namespaced | `cluster-api-provider-bke/api/bkecommon/v1beta1/bkenode_types.go` | `cluster-api-provider-bke/config/crd/bases/bke.bocloud.com_bkenodes.yaml` | cluster-api-provider-bke | 平台共享 |
| 7 | `ContainerdConfig` | `bke.bocloud.com/v1beta1` | Namespaced | `cluster-api-provider-bke/api/capbke/v1beta1/containerdconfig_types.go` | `cluster-api-provider-bke/config/crd/bases/bke.bocloud.com_containerdconfigs.yaml` | cluster-api-provider-bke | 平台共享 |
| 8 | `KubeletConfig` | `bke.bocloud.com/v1beta1` | Namespaced | `cluster-api-provider-bke/api/bkecommon/v1beta1/kubeletconfig_types.go` | `cluster-api-provider-bke/config/crd/bases/bke.bocloud.com_kubeletconfigs.yaml` | cluster-api-provider-bke | 平台共享 |
| 9 | `ClusterVersion` | `config.openfuyao.com/v1alpha1` | Namespaced | `cluster-api-provider-bke/api/v1alpha1/clusterversion_types.go` | `cluster-api-provider-bke/config/crd/bases/config.openfuyao.com_clusterversions.yaml` | cluster-api-provider-bke | 平台共享 |
| 10 | `ComponentVersion` | `config.openfuyao.com/v1alpha1` | Namespaced | `cluster-api-provider-bke/api/v1alpha1/componentversion_types.go` | `cluster-api-provider-bke/config/crd/bases/config.openfuyao.com_componentversions.yaml` | cluster-api-provider-bke | 平台共享 |
| 11 | `ReleaseImage` | `config.openfuyao.com/v1alpha1` | Namespaced | `cluster-api-provider-bke/api/v1alpha1/releaseimage_types.go` | `cluster-api-provider-bke/config/crd/bases/config.openfuyao.com_releaseimages.yaml` | cluster-api-provider-bke | 平台共享 |
| 12 | `UpgradePath` | `config.openfuyao.com/v1alpha1` | Cluster | `cluster-api-provider-bke/api/v1alpha1/upgradepath_types.go` | `cluster-api-provider-bke/config/crd/bases/config.openfuyao.com_upgradepaths.yaml` | cluster-api-provider-bke | 平台共享 |
| 13 | `TenantCluster` | `clusters-computing-operator.openfuyao.com/v1alpha1` | Namespaced | `clusters-computing-operator/api/v1alpha1/tenantcluster_types.go` | `clusters-computing-operator/config/crd/bases/clusters-computing-operator.openfuyao.com_tenantclusters.yaml` | clusters-computing-operator | 私有 |
| 14 | `TenantManagedCluster` | `clusters-computing-operator.openfuyao.com/v1alpha1` | Namespaced | `clusters-computing-operator/api/v1alpha1/tenantmanagedcluster_types.go` | `clusters-computing-operator/config/crd/bases/clusters-computing-operator.openfuyao.com_tenantmanagedclusters.yaml` | clusters-computing-operator | 私有 |
| 15 | `ContainerCheckpoint` | `colocation.openfuyao.com/v1` | Namespaced | `colocation-management/pkg/apis/v1/containercheckpoint_types.go` | `colocation-management/charts/colocation-management/templates/crds/colocation.openfuyao.com_containercheckpoints.yaml` | colocation-management | 私有 |
| 16 | `ColocationWorkloadMatchRule` | `colocation.openfuyao.com/v1` | 未发布独立 YAML | `colocation-service/pkg/apis/colocation/v1/colocationworkloadmatchrule_types.go` | 仅有 Go 类型，未找到独立 CRD YAML | colocation-service | 私有 |
| 17 | `ConsolePlugin` | `console.openfuyao.com/v1beta1` | Cluster | `console-service/pkg/plugin/consoleplugin_type.go`<br>`plugin-management-service/pkg/plugin/consoleplugin_type.go` | `console-service/charts/console-service/crds/consoleplugins.crd.yaml`<br>`plugin-management-service/charts/plugin-management-service/crds/consoleplugins.crd.yaml` | console-service / plugin-management-service | **插件框架共享** |
| 18 | `HelmChartRepository` | `console.openfuyao.com/v1beta1` | Cluster | 无 Go types，YAML 内定义 | `marketplace-service/charts/marketplace-service/crds/helmchartrepository.crd.yaml` | marketplace-service | **插件框架** |
| 19 | `Dashboard` | `monitoring-dashboard.openfuyao.com/v1beta1` | Namespaced | 无 Go types，代码内使用结构 | `monitoring-dashboard-service/charts/monitoring-dashboard-service/crds/dashboard.crd.yaml` | monitoring-dashboard-service | 私有 |
| 20 | `Widget` | `monitoring-dashboard.openfuyao.com/v1beta1` | Namespaced | 无 Go types，代码内使用结构 | `monitoring-dashboard-service/charts/monitoring-dashboard-service/crds/widget.crd.yaml` | monitoring-dashboard-service | 私有 |
| 21 | `Cluster` | `multicluster.openfuyao.com/v1beta1` | Cluster | `multi-cluster-service/api/v1beta1/cluster_types.go` | `multi-cluster-service/config/crd/bases/multicluster.openfuyao.com_clusters.yaml` | multi-cluster-service | 私有 |
| 22 | `NPUClusterPolicy` | `npu.openfuyao.com/v1` | Cluster | `npu-operator/api/v1/npuclusterpolicy_types.go` | `npu-operator/config/crd/bases/npu.openfuyao.com_npuclusterpolicies.yaml` | npu-operator | 私有 |
| 23 | `AscendJob` | `mindxdl.gitee.com/v1` | Namespaced | 无 Go types，仅有 YAML | `npu-operator/charts/npu-operator/crds/ascendjobs.crd.yaml` | npu-operator（部署托管） | 非自研，托管部署 |
| 24 | `KAEPolicy` | `device.openfuyao.com/v1` | Cluster | `kae-operator/api/v1/kaepolicy_types.go` | `kae-operator/config/crd/bases/device.openfuyao.com_kaepolicies.yaml` | kae-operator | 私有 |
| 25 | `WebterminalTemplate` | `terminal.openfuyao.com/v1beta1` | Namespaced | `web-terminal-service/api/v1beta1/webterminaltemplate_types.go` | `web-terminal-service/config/crd/bases/terminal.openfuyao.com_webterminaltemplates.yaml` | web-terminal-service | 私有 |
| 26 | `InferNexService` | `infernex.infernex.io/v1alpha1` | Namespaced | `InferNex/component/InferNex-Bridge/api/v1alpha1/infernexservice_types.go` | `InferNex/component/InferNex-Bridge/config/crd/bases/infernex.infernex.io_infernexservices.yaml` | InferNex-Bridge | 私有 |
| 27 | `InferNexServiceConfig` | `infernex.infernex.io/v1alpha1` | Namespaced | `InferNex/component/InferNex-Bridge/api/v1alpha1/infernexserviceconfig_types.go` | `InferNex/component/InferNex-Bridge/config/crd/bases/infernex.infernex.io_infernexserviceconfigs.yaml` | InferNex-Bridge | 私有 |
| 28 | `User` | `auth.openfuyao.com/v1beta1` | Cluster | `installer/pkg/executor/auth/user_types.go` | 未找到对应 CRD YAML | installer | 私有（未发布） |

**说明**：
- `bkecommon` 中同样存在 `Command`、`BKECluster`、`ContainerdConfig`、`KubeletConfig`、`BKENode` 等类型的复制定义，用于共享 Go 结构；实际的 CRD 定义与发布方为 `cluster-api-provider-bke`。
- `AscendJob` 的 group 是 `mindxdl.gitee.com`，说明它不是 openFuyao 自研，只是由 `npu-operator` 负责随 chart 安装部署。
- `User` 只有 Go 类型文件，未找到对应 CRD YAML，可能尚未正式发布。
- `ColocationWorkloadMatchRule` 同样只有 Go 类型，未找到发布用 CRD YAML。

---

## 3. 按领域分类

### 3.1 插件 / UI 扩展层

| Kind | 所属模块 | 说明 |
|---|---|---|
| `ConsolePlugin` | console-service / plugin-management-service | 控制台 UI 插件注册点 |
| `HelmChartRepository` | marketplace-service | 应用市场 Helm 仓库源 |

### 3.2 集群生命周期管理

| Kind | 所属模块 | 说明 |
|---|---|---|
| `BKECluster` / `BKEClusterTemplate` / `BKEMachine` / `BKEMachineTemplate` / `BKENode` / `ContainerdConfig` / `KubeletConfig` / `Command` | cluster-api-provider-bke / bkeagent / bkecommon | 集群、节点、运行时、命令下发的声明式管理 |
| `ClusterVersion` / `ComponentVersion` / `ReleaseImage` / `UpgradePath` | cluster-api-provider-bke | 集群版本、组件版本、发布镜像、升级路径管理 |

### 3.3 多集群管理

| Kind | 所属模块 | 说明 |
|---|---|---|
| `TenantCluster` / `TenantManagedCluster` | clusters-computing-operator | 多租赁/多集群抽象 |
| `Cluster` | multi-cluster-service | 纳管集群元数据 |

### 3.4 异构算力（当前均为各厂商 operator 私有）

| Kind | 所属模块 | 说明 |
|---|---|---|
| `NPUClusterPolicy` | npu-operator | 昇腾 NPU 全栈组件集群策略声明 |
| `KAEPolicy` | kae-operator | KAE（鲲鹏加速引擎）策略声明 |

### 3.5 共置 / 混部

| Kind | 所属模块 | 说明 |
|---|---|---|
| `ContainerCheckpoint` | colocation-management | 容器快照/检查点 |
| `ColocationWorkloadMatchRule` | colocation-service | 混部负载匹配规则 |

### 3.6 可观测

| Kind | 所属模块 | 说明 |
|---|---|---|
| `Dashboard` / `Widget` | monitoring-dashboard-service | 可观测大盘与组件 |

### 3.7 运维 / 扩展

| Kind | 所属模块 | 说明 |
|---|---|---|
| `WebterminalTemplate` | web-terminal-service | Web 终端模板 |

### 3.8 推理服务

| Kind | 所属模块 | 说明 |
|---|---|---|
| `InferNexService` / `InferNexServiceConfig` | InferNex-Bridge | 推理服务管理 |

### 3.9 认证与安装

| Kind | 所属模块 | 说明 |
|---|---|---|
| `User` | installer | 认证用户类型，尚未找到独立 CRD YAML |

---

## 4. 插件框架 CRD 专项分析

openFuyao 的插件/可插拔架构目前由两个 CRD 支撑：

### 4.1 `ConsolePlugin`

- **apiVersion**：`console.openfuyao.com/v1beta1`
- **Scope**：Cluster（`console-service/charts/.../consoleplugins.crd.yaml` 中声明）
- **定义者**：两个服务都发布了同一个 CRD
  - `console-service/pkg/plugin/consoleplugin_type.go`
  - `plugin-management-service/pkg/plugin/consoleplugin_type.go`
- **生产者 / 创建者**：
  - 各微前端子应用的 chart 模板，例如：
    - `multi-cluster-website`
    - `colocation-website`
    - `installer-website`
    - `logging-website`
    - `monitoring-dashboard-website`
    - `oauth-proxy`
    - `ray-website`
    - `volcano-config-website`
- **消费者**：
  - `plugin-management-service`：提供 REST API 管理 `ConsolePlugin` 实例（`plugin-management-service/pkg/api/consoleplugin/v1beta1/handler.go`）
  - `bke-console-website` / `console-website`：读取 `ConsolePlugin` 列表，渲染顶部导航和侧边栏（`bke-console-website/src/components/NavBar.jsx`、`Menu.jsx`；`consolePluginStore.js`）
- **作用**：定义一个 UI 插件在 openFuyao 控制台中的挂载位置（`Nav` 或 `Side`）、显示名称、后端 Service 等。
- **局限**：它只解决 **UI 注册** 问题，不解决组件安装后的运行期生命周期管理问题。

### 4.2 `HelmChartRepository`

- **apiVersion**：`console.openfuyao.com/v1beta1`
- **Scope**：Cluster
- **定义者**：`marketplace-service/charts/marketplace-service/crds/helmchartrepository.crd.yaml`
- **生产者 / 消费者**：`marketplace-service`
- **作用**：声明一个 Helm Chart 仓库源，供应用市场发现、索引、展示组件包。
- **局限**：它只解决 **包发现 / 包来源** 问题，不定义组件安装后如何声明预期状态、如何调和、如何回写状态。

### 4.3 当前插件框架的能力边界

**已覆盖**：
- 组件包的发现与安装入口（`HelmChartRepository`）
- 组件 UI 的注册与渲染（`ConsolePlugin`）

**未覆盖**：
- 组件安装后，其内部子服务 / 子 DaemonSet / 配置 / 证书等资源如何统一声明和调和。
- 组件的启用、禁用、升级、回滚、状态聚合等运行期语义。
- 后台组件与硬件厂商驱动（如 `npu-operator`、`kae-operator`）如何被纳入统一生命周期模型。

---

## 5. `npu-operator` 与插件框架的关系

### 5.1 `npu-operator` 自研 / 托管的 CRD

| Kind | apiVersion | Scope | 自研 or 托管 |
|---|---|---|---|
| `NPUClusterPolicy` | `npu.openfuyao.com/v1` | Cluster | 自研，单例 CRD（`metadata.name` 必须为 `cluster`） |
| `AscendJob` | `mindxdl.gitee.com/v1` | Namespaced | 托管部署，非自研 |

### 5.2 关键事实（基于代码扫描，非推断）

1. **`npu-operator` 不创建 `ConsolePlugin` 实例**
   - 在 `npu-operator/` 全目录搜索 `ConsolePlugin`、`consoleplugins`、`console.openfuyao.com`，无结果。
   - 仓库中 `kind: ConsolePlugin` 实例仅出现在前端 website 的 chart 模板中（如 `multi-cluster-website`、`installer-website` 等），未出现在 `npu-operator` 的 chart 或示例中。

2. **`npu-operator` 不使用 `HelmChartRepository`**
   - 在 `npu-operator/` 中搜索 `HelmChartRepository`、`helmchartrepositories`、`console.openfuyao.com`，无结果。

3. **插件管理服务不读写 `NPUClusterPolicy`**
   - 在 `plugin-management-service`、`marketplace-service`、`console-service` 中搜索 `NPUClusterPolicy`、`npu.openfuyao.com`、`npuclusterpolicies`，无结果。

### 5.3 结论

`NPUClusterPolicy` 是 `npu-operator` 的**私有 CRD**，用于声明昇腾 NPU 全栈组件（Driver、Runtime、Device Plugin、Volcano、Trainer、Exporter 等）的集群级策略与生命周期。

它与 openFuyao 当前的插件市场 / 控制台插件框架**没有直接的 CRD 引用或消费关系**。如果 `npu-operator` 未来需要在 console 中暴露 UI，需要由独立的前端组件（例如尚未存在的 `npu-website`）来创建 `ConsolePlugin` 实例。

---

## 6. 平台共享 CRD vs 私有 CRD

| 类别 | 包含的 CRD | 说明 |
|---|---|---|
| **平台级 / 共享 CRD** | `ConsolePlugin`、`HelmChartRepository`、BKE 系列（`BKECluster`、`BKEMachine`、`BKENode`、`Command`、`ContainerdConfig`、`KubeletConfig` 等）、`ClusterVersion` / `ComponentVersion` / `ReleaseImage` / `UpgradePath` | 被多个服务共同发布或消费，通常定义在平台基础能力模块中 |
| **Operator / 服务私有 CRD** | `NPUClusterPolicy`、`KAEPolicy`、`TenantCluster`、`TenantManagedCluster`、`Cluster`、`ContainerCheckpoint`、`ColocationWorkloadMatchRule`、`Dashboard`、`Widget`、`WebterminalTemplate`、`InferNexService`、`InferNexServiceConfig` 等 | 仅由所属 operator 自己定义、自己消费，不跨服务共享 |
| **未正式发布** | `User`、`ColocationWorkloadMatchRule` | 仅有 Go 类型，未找到 CRD YAML 发布文件 |

---

## 7. 当前缺失与建议新增的 CRD

为了把 openFuyao 的插件体系从“包分发 + UI 注册”扩展为“通用组件生命周期管理 + 跨厂商异构算力抽象”，建议新增以下 CRD。

### 7.1 A 层：通用组件生命周期管理

| 候选 CRD | apiVersion 建议 | 作用 |
|---|---|---|
| `ClusterComponentPolicy` | `component.openfuyao.com/v1beta1` 或 `system.openfuyao.com/v1beta1` | 声明一个组件套件的整体部署策略。包含：vendor、version、runtimeClass、一组可启用/禁用的 `Component`，以及每个 Component 的镜像、命令、环境变量、资源、Placement 等通用字段。 |
| `ClusterComponent`（可选） | 同上 | 单一组件的运行期声明。若策略粒度需要拆分，可作为 `ClusterComponentPolicy` 的下级资源。 |

设计目标：
- 让 `npu-operator`、`kae-operator` 等 vendor-specific 的 operator 从“既定义 CRD 又实现调和”变为“实现具体的渲染/校验逻辑”，而通用框架负责统一 CRD 与 Reconcile 流程。
- 组件包安装后，由通用 controller 自动创建 `ConsolePlugin` 实例，打通 UI 注册。

### 7.2 B 层：跨厂商异构算力抽象

| 候选 CRD | apiVersion 建议 | 作用 |
|---|---|---|
| `DeviceProfile` 或 `AcceleratorProfile` | `accelerator.openfuyao.com/v1beta1` | 抽象不同厂商加速卡的能力模型，例如算力单元、显存、带宽、拓扑结构、健康检查方式等。 |
| `NodeDeviceInventory` | `accelerator.openfuyao.com/v1beta1` | 每个节点一张“异构设备清单”，汇总该节点上各厂商加速设备的逻辑容量与健康状态。 |

设计目标：
- 屏蔽昇腾、昆仑芯、海光、摩尔线程、平头哥等厂商的私有资源名差异。
- 向上层调度器（如 Volcano）、推理服务（如 InferNex）、训练平台提供统一的节点标签与 resource name 约定。

### 7.3 现有 CRD 继续复用

| CRD | 在新框架中的角色 |
|---|---|
| `ConsolePlugin` | 保留。由 A 层通用组件控制器或组件包的 post-install 钩子自动创建，无需各组件自己写 UI 集成代码。 |
| `HelmChartRepository` | 保留。继续作为应用市场包发现机制，作为 A 层组件包的来源。 |

---

## 8. 后续落地建议

1. **通用框架 CRD 应定义在平台侧，不能放在 vendor operator 里**
   - `ClusterComponentPolicy` / `ClusterComponent` 应该由新的 `component-management-operator` 或 `openfuyao-system-controller` 定义，而不是由 `npu-operator` 定义。
   - 只有当 CRD 由平台定义后，各 vendor operator 才能以相同模式接入。

2. **参考 `npu-operator` 的既有实现，迁移通用骨架**
   - `npu-operator/api/v1/component_types.go` 中的 `ComponentWorkloadSpec` 是一个很好的通用组件配置抽象。
   - `npu-operator` 的 Reconciler 流程（读取 CR → 校验 → 渲染 assets → apply/delete → 回写 status）可以直接泛化。
   - 迁移时，把昇腾-specific 的字段（如 `Ascend Driver`、`OCIRuntime`、`MindIO`、`vNPU` 等）保留在 `NPUClusterPolicy` 或 `ClusterComponentPolicy` 的 vendor-specific 扩展字段中。

3. **A 层与 B 层解耦**
   - 先落地 A 层（通用组件生命周期），让各厂商 operator 能统一接入 openFuyao 插件体系。
   - 等 A 层稳定后，再定义 B 层最小公约集（统一节点标签、统一 extended resource 名、`DeviceProfile` 结构），逐步让各厂商设备插件兼容。

4. **UI 注册自动化**
   - 在 `ClusterComponentPolicy` 中增加 `consolePlugin` 字段。
   - 通用控制器在组件 Ready 后，自动创建对应的 `ConsolePlugin` 实例，结束当前“前端组件手动创建 ConsolePlugin”的分散模式。

---

## 9. 附录：主要 CRD 文件索引

| CRD | Go 类型文件 | CRD YAML 文件 |
|---|---|---|
| `ConsolePlugin` | `console-service/pkg/plugin/consoleplugin_type.go`<br>`plugin-management-service/pkg/plugin/consoleplugin_type.go` | `console-service/charts/console-service/crds/consoleplugins.crd.yaml`<br>`plugin-management-service/charts/plugin-management-service/crds/consoleplugins.crd.yaml` |
| `HelmChartRepository` | 无 Go types，仅在 YAML 中定义 | `marketplace-service/charts/marketplace-service/crds/helmchartrepository.crd.yaml` |
| `NPUClusterPolicy` | `npu-operator/api/v1/npuclusterpolicy_types.go` | `npu-operator/config/crd/bases/npu.openfuyao.com_npuclusterpolicies.yaml` |
| `KAEPolicy` | `kae-operator/api/v1/kaepolicy_types.go` | `kae-operator/config/crd/bases/device.openfuyao.com_kaepolicies.yaml` |
| `TenantCluster` | `clusters-computing-operator/api/v1alpha1/tenantcluster_types.go` | `clusters-computing-operator/config/crd/bases/clusters-computing-operator.openfuyao.com_tenantclusters.yaml` |
| `TenantManagedCluster` | `clusters-computing-operator/api/v1alpha1/tenantmanagedcluster_types.go` | `clusters-computing-operator/config/crd/bases/clusters-computing-operator.openfuyao.com_tenantmanagedclusters.yaml` |
| `Cluster` | `multi-cluster-service/api/v1beta1/cluster_types.go` | `multi-cluster-service/config/crd/bases/multicluster.openfuyao.com_clusters.yaml` |
| `ContainerCheckpoint` | `colocation-management/pkg/apis/v1/containercheckpoint_types.go` | `colocation-management/charts/colocation-management/templates/crds/colocation.openfuyao.com_containercheckpoints.yaml` |
| `WebterminalTemplate` | `web-terminal-service/api/v1beta1/webterminaltemplate_types.go` | `web-terminal-service/config/crd/bases/terminal.openfuyao.com_webterminaltemplates.yaml` |
| `InferNexService` / `InferNexServiceConfig` | `InferNex/component/InferNex-Bridge/api/v1alpha1/infernexservice_types.go`<br>`InferNex/component/InferNex-Bridge/api/v1alpha1/infernexserviceconfig_types.go` | `InferNex/component/InferNex-Bridge/config/crd/bases/infernex.infernex.io_infernexservices.yaml`<br>`InferNex/component/InferNex-Bridge/config/crd/bases/infernex.infernex.io_infernexserviceconfigs.yaml` |
| `Dashboard` / `Widget` | 无 Go types | `monitoring-dashboard-service/charts/monitoring-dashboard-service/crds/dashboard.crd.yaml`<br>`monitoring-dashboard-service/charts/monitoring-dashboard-service/crds/widget.crd.yaml` |
| `User` | `installer/pkg/executor/auth/user_types.go` | 未找到独立 CRD YAML |
| `ColocationWorkloadMatchRule` | `colocation-service/pkg/apis/colocation/v1/colocationworkloadmatchrule_types.go` | 未找到独立 CRD YAML |

---

*本文档基于对 `openfuyao-projects` 仓库内 Go 类型、CRD YAML、chart 模板和控制器代码的只读扫描生成，未修改任何业务代码。*
