# 01 安装部署相关

本目录收录所有用于“把 openFuyao 管理面部署到目标集群”以及“节点生命周期初始化”的组件。

## 包含组件

| 目录 | 说明 |
|---|---|
| `bkeadm` | 引导 CLI，负责启动首节点、K3s 引导集群、本地 registry / repo、Cluster API、provider-bke 和可选控制台。 |
| `bkeagent` | 节点 agent，运行在每个目标节点上，执行由 `cluster-api-provider-bke` 下发的 `Command` CR。 |
| `bkecommon` | 安装器通用库代码。 |
| `bke-console-website` | 安装阶段可选部署的初始控制台前端。 |
| `bke-manifests` | 组件包仓库，`BKECluster.spec.clusterConfig.addons` 中的每一项都对应这里的一个 YAML 包目录。 |
| `cluster-api-provider-bke` | openFuyao 自研的 CAPI 基础设施 provider，把 `BKECluster` / `BKENode` 翻译成节点可执行的 `Command` CR。 |
| `env-check` | 安装前环境检查工具。 |
| `installer` | 早期/辅助安装器相关目录。 |
| `installer-service` | 安装器 REST / WebSocket API 服务，调用 K8s API 生成 `BKECluster` 与 `BKENode`。 |
| `installer-website` | 安装器控制台子前端，供用户创建/纳管集群。 |
| `node-checker` | 节点状态检查组件。 |
| `oauth-proxy` | 安装阶段拉起的认证代理。 |
| `oauth-server` | 安装阶段拉起的认证服务。 |
| `oauth-webhook` | 安装阶段拉起的认证 Webhook。 |
| `openfuyao-system-controller` | 在目标集群中部署/卸载 openFuyao 管理面的 addon 包，是安装流程的“最后一公里”。 |

## 边界说明

- 这些组件使用 `bke.bocloud.com/v1beta1`、`bkeagent.bocloud.com/v1beta1`、`config.openfuyao.com/v1alpha1` 这类 openFuyao 自研安装器 CRD。
- 它们不直接参与芯片调度、设备发现等运行时逻辑，但负责把运行这些逻辑所需的 chart 和 manifest 分发到目标集群。
- `installer-website`、`bke-console-website` 与控制台插件框架无关：前者是安装/引导阶段 UI，后者是通用控制台的 UI 扩展机制。
