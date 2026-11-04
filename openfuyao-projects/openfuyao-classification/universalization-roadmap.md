# openFuyao 三层解耦与通用化改造路线图

## 背景

当前 openFuyao 仓库把三层代码放在一个 repo 中：

1. **通用 K8s 部署工具**：`bkeadm` / `cluster-api-provider-bke` / `bkeagent` / `installer-*` 等，负责安装、升级、维护 K8s 集群本身；
2. **通用 openFuyao 管理平台**：控制台、集群/节点管理、组件参考库、组件安装部署、应用市场、Volcano 增强、NUMA 感知、Ray、日志/监控等，作为可选平台层运行在 K8s 之上；
3. **设备接入框架**：昇腾 NPU / KAE / vNPU 相关 operator 与驱动的通用抽象层，供各类国产芯片复用。

三个层次目前存在较多耦合：

- 安装器默认绑定 openFuyao 制品源（`cr.openfuyao.cn`、华为云 OBS）和默认控制台；
- 基础平台服务之间通过 openFuyao 特定 namespace / label / oauth 耦合；
- 昇腾设备直接定义私有 CRD，其它国产芯片无法复用其接入模式。

本路线图的目标是让三层各自成为可独立使用、可替换的模块。

## 三个目标

| 目标 | 含义 | 对应分类 |
|---|---|---|
| 通用 K8s 部署工具 | 引导控制台/CLI 可以独立部署任意 K8s 发行版及任意上层平台，不默认绑定 openFuyao 控制台、镜像源或管理面。 | 01 + 04 |
| 通用 openFuyao 管理平台 | 控制台、集群/节点/组件管理、组件参考库、Volcano / Ray / 监控等可以独立打包、独立部署，作为平台层运行在 K8s 之上。 | 02 |
| 设备接入框架 | 昇腾相关代码和设备接入模式抽象成通用接口，寒武纪、海光、燧原等可复制实现。 | 03 |

## 改造思路总览

### 第一层：通用 K8s 部署工具

把 `bkeadm` / `installer-service` / `cluster-api-provider-bke` 从“openFuyao 专用引导工具”变成“通用 K8s 集群部署工具”。该工具只负责把裸机/虚拟机变成可用的 K8s 集群，不强制附带 openFuyao 管理面。

关键动作：

1. **制品源外部化**：registry / yum / chart / binary 源由 CRD/配置文件决定，不再是硬编码。
2. **默认 addon 中立化**：默认只安装 `kubeproxy`、`calico`、`coredns` 等 K8s 基础组件；openFuyao 管理面变为**可选** addon，由第二层统一部署。
3. **addon 注册机制 CRD 化**：用 `ClusterComponent` / `ComponentCatalog` / `ComponentSource` 替代 `bke-manifests` 的硬编码目录映射。
4. **认证与控制台可替换**：`oauth-*` / 引导控制台默认关闭或支持外部 OIDC；部署工具本身不预装认证面。
5. **命名与品牌解耦**：提供纯 CLI 模式，可在不拉取 openFuyao 控制台的情况下完成裸 K8s 集群部署。
6. **支持超大规模单体算力集群的部分控制面部署架构**：通用部署工具生成的 K8s 控制面不局限于单 etcd / 单 apiserver，应支持：
   - 多 etcd 拆分（如 pods / events / data）并配置 `--etcd-servers-overrides`；
   - 多 apiserver 实例 + 入口七层按 API 资源分组负载均衡（例如 `/api/v1/pods`、`/api/v1/nodes` 分别路由到不同后端组）；
   - kubelet / apiserver 关键调优参数（如 `--kube-api-qps`、`--max-requests-inflight`、`status update frequency`）在部署阶段可配置；
   - 使单 K8s 集群在不变更上游 K8s 源码的前提下，具备覆盖万级节点规模的部署能力。

### 第二层：通用 openFuyao 管理平台

把 openFuyao 平台能力拆成可独立 Helm 安装的子系统。该平台作为可选层运行在 `bkeadm` 部署出的 K8s 之上，提供集群/节点管理、组件参考库、组件安装部署、可观测、异构调度、AI 负载等企业级能力。

1. **核心控制台框架**：`console-service` + `console-website` + `plugin-management-service` + `ConsolePlugin` CRD，作为整个管理平台的入口和扩展框架。
2. **组件参考库与组件部署**：统一分类管理平台组件（如 volcano-ext、monitoring、logging、ray 等），用户可选安装/卸载/升级。
3. **应用市场框架**：`marketplace-service` + `HelmChartRepository` CRD 可单独使用。
4. **用户与认证**：把 `user-management-operator` 和 `oauth-*` 拆成可选依赖，支持外部 IDP。
5. **调度与计算增强**：`volcano-ext` + `volcano-config-service` + `node-feature-discovery` 可独立成“NUMA/异构调度套件”。
6. **AI workload**：`ray-service` + `ray-website` 已几乎独立，只需补齐可选的依赖开关。
7. **可观测性**：`monitoring-*`、`logging-*`、`victoriametrics-controller` 可打包为独立可观测套件。

### 第三层：设备接入框架

为国产芯片建立通用接入框架，并保留昇腾作为首个实现：

1. **设备模型 CRD**：
   - `DeviceClass`：设备类型（NPU、KAE、DSA 等）。
   - `DeviceProfile`：某款芯片的驱动、运行时、资源名、虚拟化能力、监控规则。
   - `NodeDeviceInventory`：节点上实际设备清单。
2. **生命周期接口**：
   - 驱动安装、设备插件、DRA 插件、节点打标规则统一抽象。
   - 厂商 operator 只负责把物理设备映射到通用 CRD，平台统一完成调度、UI、监控。
3. **调度扩展**：
   - `volcano-config-service` 的 `NodeConfig` / `Oenuma` 支持任意厂商拓扑 CRD 或 gRPC 接口。
4. **Console 扩展点**：
   - 各厂商子前端通过通用 `ConsolePlugin` 注册，平台提供统一的设备视图表单框架。

## 阶段建议

| 阶段 | 工作 | 产出 |
|---|---|---|
| 阶段一 | 文档与 CRD 设计 | 新增/修改 CRD 提案、独立部署清单、接口契约 |
| 阶段二 | 通用 K8s 部署工具通用化 | 修改 `bkeadm`、`installer-service`、`cluster-api-provider-bke` 的默认配置与扩展机制，支持原生 K8s 与大规模控制面部署架构 |
| 阶段三 | 通用 openFuyao 管理平台解耦 | 为每个 02 组件提供独立 `values.yaml`、可选依赖开关、`openfuyao-management-platform` Helm 元 chart |
| 阶段四 | 设备抽象 | 在 02 层新增 `device-*` 通用 operator / CRD；把 03 的昇腾实现改造成该框架的实现 |
| 阶段五 | 验证与多厂商示例 | 用昇腾做回归，新增一个最小国产芯片（或虚拟设备）示例实现 |

## 文档清单

- `01-installation-and-deployment/universalization.md`：通用 K8s 部署工具详细设计
- `02-base-platform/universalization.md`：通用 openFuyao 管理平台详细设计
- `03-ascend-specific/universalization.md`：设备接入框架与国产芯片扩展详细设计
