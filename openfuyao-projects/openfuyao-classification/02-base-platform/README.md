# 02 openFuyao 通用基础平台

本目录收录与任何单一厂商芯片无关的 openFuyao 平台能力，包括控制台、插件框架、应用市场、NUMA / Volcano 调度增强、Ray 集成、日志/监控/多集群等。

## 包含组件

| 目录 | 说明 |
|---|---|
| `application-management-service` | 应用生命周期管理后端。 |
| `cache-indexer` | 平台内部缓存/索引组件。 |
| `clusters-computing-operator` | 通用集群计算管理 operator。 |
| `colocation-management` / `colocation-service` / `colocation-website` | 混部/协同调度管理。 |
| `confidential-containers-deployment` | 机密容器部署能力。 |
| `console-service` | 控制台后端。 |
| `console-website` | 控制台前端（含 qiankun 微前端 host）。 |
| `hermes-router` | 平台网络路由器/代理。 |
| `InferNex` | AI 推理服务框架。 |
| `logging-operator` / `logging-website` | 日志采集与管理。 |
| `marketplace-service` | 应用市场后端。 |
| `monitoring-service` / `monitoring-dashboard-service` / `monitoring-dashboard-website` | 监控与可观测大盘。 |
| `mooncake` | 分布式 AI 推理通信/调度框架。 |
| `multi-cluster-service` / `multi-cluster-website` | 多集群管理。 |
| `node-feature-discovery` | 上游通用节点打标组件，被多个组件作为子 chart 引入。 |
| `plugin-management-service` | 控制台插件生命周期管理。 |
| `ray` | Ray 通用目录/库。 |
| `ray-service` | 基于 KubeRay 的 Ray 集群生命周期 REST 服务。 |
| `ray-website` | Ray 控制台子前端。 |
| `ub-network-bandwidth-operator` / `ub-network-device-plugin` | 通用网络增强组件。 |
| `ub-enable-quickstart` | UB 能力快速入口。 |
| `victoriametrics-controller` | 可观测指标存储/控制组件。 |
| `volcano-config-service` | Volcano 调度配置管理服务。 |
| `volcano-config-website` | Volcano 配置控制台子前端。 |
| `volcano-ext` | Volcano 调度器扩展，含 `numa-aware` 插件。 |
| `web-terminal-service` | 控制台内置 Web 终端。 |

## 边界说明

- 这些组件不定义、不消费任何昇腾专有 CRD（如 `NPUClusterPolicy`、`KAEPolicy`）。
- `node-feature-discovery` 虽是上游项目，但它提供**通用节点发现/打标**能力，是平台异构算力接入的底座之一。昇腾专有规则（`nodefeaturerules.yaml`）保留在 `03-ascend-specific`。
- `volcano-ext` 的 `numa-aware` 插件、Volcano 的 `Numatopology` CRD 是通用 NUMA 感知调度能力；`Oenuma` CRD 的 NUMA 数据字段在当前仓库中尚未被任何组件填充，仅被 `volcano-config-service` 作为开关读写。
- `ray-service` 只操作 KubeRay 原生 CRD（`RayCluster`、`RayJob`、`RayService`），无其他 openFuyao 服务依赖。
