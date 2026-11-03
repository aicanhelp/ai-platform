# 02 通用基础平台详细分析

## 1. 控制台与插件框架

### 1.1 控制台组件

- `console-service`：控制台后端，提供 REST API、用户会话、聚合各子服务。
- `console-website`：基于 `openInula` + `qiankun` 的微前端宿主，负责加载子应用。
- `plugin-management-service`：管理 `ConsolePlugin` CR 的生命周期，子应用注册与挂载。

### 1.2 真正的插件 CRD

openFuyao 自研的插件/扩展 CRD 只有两个：

| Group | Kind | 作用 |
|---|---|---|
| `console.openfuyao.com/v1beta1` | `ConsolePlugin` | 子前端注册项，声明子应用 URL、入口脚本、菜单位置等。 |
| `console.openfuyao.com/v1beta1` | `HelmChartRepository` | Helm 源注册，用于应用市场。 |

- `ConsolePlugin` 由具体子前端创建，控制台/ `plugin-management-service` 消费。
- `HelmChartRepository` 由 `marketplace-service` 消费。
- 控制台本身**不感知**昇腾等厂商的私有 CRD，也没有昇腾专有页面。

### 1.3 子前端清单（通用）

所有通过 `qiankun` 挂载到控制台的子前端均属于通用基础平台：

- `console-website`（host）
- `installer-website`
- `bke-console-website`（安装阶段 host）
- `multi-cluster-website`
- `ray-website`
- `volcano-config-website`
- `monitoring-dashboard-website`
- `logging-website`
- `colocation-website`

> 注：`installer-website` 和 `bke-console-website` 虽然与安装相关，但本质是通用子前端能力，因此也在控制台框架内出现；其应用代码已按用途归入 `01-installation-and-deployment`。 `bke-console-website` 实际上归属于安装阶段 UI，这里不再重复复制。

## 2. 应用市场与扩展

- `marketplace-service`：应用市场后端，管理 chart 源、应用模板、发布。
- `application-management-service`：已部署应用的实例生命周期管理。
- `victoriametrics-controller`：监控指标相关组件，也作为 addon 被安装器部署。
- `monitoring-service` / `monitoring-dashboard-service` / `monitoring-dashboard-website`：可观测能力，不是监控昇腾专用。

## 3. 多集群与网络增强

- `multi-cluster-service` / `multi-cluster-website`：多集群纳管与操作。
- `hermes-router`：跨集群/入口路由代理。
- `ub-network-bandwidth-operator` / `ub-network-device-plugin`：网络卸载/带宽管理，通用节点网络增强。
- `cache-indexer`：平台内部缓存索引能力。

## 4. 节点发现（node-feature-discovery）

`node-feature-discovery` 是 `kubernetes-sigs/node-feature-discovery` 的集成/Fork。它提供通用能力：

- 读取节点 CPU、内核、PCI、USB、自定义规则等；
- 通过 `NodeFeature` CRD 或内置规则给节点打标签/注解；
- 支持 `NodeFeatureRule` 自定义标签规则。

openFuyao 中它被多处作为子 chart 引用：
- `npu-operator/charts/npu-operator/Chart.yaml`
- `kae-operator/charts/kae-operator/Chart.yaml`
- `colocation-website/charts/colocation-package/Chart.yaml`

`npu-operator` 的 `values.yaml` 中默认 `nfd.nodefeaturerules: false`，只在启用昇腾规则时生成额外规则文件。  
因此：
- **NFD 上游代码本身属于通用基础平台**；
- **昇腾-specific 的 NFD 规则**保留到 `03-ascend-specific` 分析。

## 5. Volcano 与 NUMA 调度

### 5.1 核心组件

- `volcano-ext`：Volcano 调度器扩展（fork v1.10.0）。
- `volcano-config-service`：通过 K8s API 维护 `volcano-scheduler-configmap`、`NodeConfig` CR、`Oenuma` CR。
- `volcano-config-website`：配置控制台子前端。

### 5.2 关键 CRD / 配置对象

| 对象 | 类型 | 说明 |
|---|---|---|
| `Numatopology` | `nodeinfo.volcano.sh/v1alpha1` | Volcano 自身 NUMA 拓扑 CRD。`volcano-ext/pkg/scheduler/plugins/numaaware/numaaware.go` 读取它。 |
| `NodeConfig` | `volcano.sh/v1` | 由 `volcano-config-service` 生成/管理，用于配置节点 kubelet / 调度参数。 |
| `Oenuma` | `resource.sop.huawei.com/v1alpha1` | `volcano-config-service` 仅读写其 `Spec.Updateenable` 作为开关；NUMA 核心字段当前无人填充。 |

### 5.3 NUMA 数据流

1. `numaadj` DaemonSet 修改节点 kubelet 配置（ `/etc/kubernetes/kubelet.conf` ）。
2. `volcano-config-service` 的 NodeConfig 控制器保持 `NodeConfig` 对象与目标节点 kubelet 配置一致。
3. `volcano-ext` 的 `numa-aware` 插件读取 `Numatopology` 进行 NUMA 感知调度决策。
4. `numa-exporter`（与 `volcano-config-service` 同镜像）暴露 Prometheus metrics。

> **结论**：NUMA 感知调度是平台通用能力，不依赖昇腾。

## 6. Ray 集成

### 6.1 组件

- `ray-service`：REST 后端，管理 `RayCluster` / `RayJob` / `RayService` 三种 KubeRay 原生 CRD。
- `ray-website`：`qiankun` 子前端，需要控制台 host 才能运行。
- `ray`：通用目录/库。

### 6.2 关键设计

- 后端不修改 KubeRay CRD 结构；
- 通过 `kuberaycrd-createby=openfuyao.com` 标签标记由平台创建的资源；
- 提供模板/草稿机制（本地 SQLite `kuberay_crd` 表）；
- 中文状态映射、启动/停止/重启语义；
- 无 `console-service` / `marketplace-service` / Volcano / NPU 的代码依赖（`go.mod` 独立）。

### 6.3 可嵌入与独立使用

- Grafana 默认 `openFuyao: true` 模式把 ingress 根路径设为 `/rest/ray/grafana` 以嵌入控制台；
- 独立使用时可改为 `false` 并设置自己的 `domain` / `nodePort`；
- 默认命名空间 `vcjob`（可在 chart `values.yaml` 中配置）。

## 7. 日志、机密容器与其他通用增强

- `logging-operator` / `logging-website`：平台统一日志。
- `confidential-containers-deployment`：安全容器部署。
- `colocation-management` / `colocation-service` / `colocation-website`：混部协同调度。
- `clusters-computing-operator`：集群计算资源抽象。
- `InferNex`：AI 推理框架。
- `mooncake`：分布式推理通信/调度增强。
- `web-terminal-service`：控制台内 Web 终端。

## 8. 通用化方向

1. 把 `ConsolePlugin` / `HelmChartRepository` 作为通用插件框架的核心；
2. 将 `node-feature-discovery` 提升为平台级通用设备发现底座；
3. 将 `volcano-ext` + `volcano-config-service` 作为通用 NUMA/异构调度能力；
4. 将 `ray-service` 的模板/标签/状态语义抽象成通用 AI workload 控制器模式。
