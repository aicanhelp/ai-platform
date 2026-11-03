# 01 安装部署类组件详细分析

## 1. 总体定位

openFuyao 的安装部署体系不是单一工具，而是“引导 + 生命周期 + 节点执行 + 组件包”四层结构：

1. **bkeadm**：裸机引导，生成首节点 K3s 控制面、本地制品仓库和 Cluster API 基础设施。
2. **cluster-api-provider-bke**：CAPI 控制器，监听 `BKECluster` / `BKENode` 并生成 `Command` CR。
3. **bkeagent**：节点执行器，读取 `Command` CR 并顺序完成 kubeadm / containerd / 网络等初始化。
4. **bke-manifests + openfuyao-system-controller**：在集群就绪后 apply 平台所需 addon 包，完成管理面交付。

## 2. 主要 CRD

| Group | Kind | 作用 |
|---|---|---|
| `bke.bocloud.com/v1beta1` | `BKECluster` | 目标集群元数据与 addon 配置。 |
| `bke.bocloud.com/v1beta1` | `BKENode` | 目标节点清单与角色。 |
| `bke.bocloud.com/v1beta1` | `BKEMachine` / `BKEMachineTemplate` | 节点运行时模板。 |
| `bke.bocloud.com/v1beta1` | `BKEClusterTemplate` | 集群模板。 |
| `bke.bocloud.com/v1beta1` | `ContainerdConfig` | containerd 运行时配置。 |
| `bkeagent.bocloud.com/v1beta1` | `Command` | 节点执行命令序列。 |
| `config.openfuyao.com/v1alpha1` | `ClusterVersion` / `ComponentVersion` / `ReleaseImage` / `UpgradePath` | 版本与升级元数据。 |

## 3. 关键流程

### 3.1 首节点引导

`bkeadm init` 会：
- 检查环境与端口；
- 拉起本地 registry、yum repo、chart repo；
- 部署 K3s 引导集群；
- 安装 Cluster API 与 `cluster-api-provider-bke`；
- 可选通过 `installConsole.sh` 部署 `bke-console-*`。

源码入口：`bkeadm/pkg/initialize/initialize.go`。

### 3.2 目标集群创建

`installer-service` 暴露 `/rest/v1/...` 接口，获取用户输入后生成 `BKECluster` 与 `BKENode`。
默认 `BKECluster` 见 `installer-service/pkg/installer/cluster_create.go`。

### 3.3 节点执行

`cluster-api-provider-bke/pkg/command/` 负责把节点生命周期转换为 `Command` CR；
`bkeagent/api/v1beta1/command_types.go` 定义命令类型：
- `BuiltIn`：kubeadm / containerd 等内置步骤；
- `Shell`：自定义 shell 脚本；
- `Kubernetes`：挂载 ConfigMap / Secret 到节点执行。

### 3.4 Addon 分发

`BKECluster.spec.clusterConfig.addons` 默认包含：
`kubeproxy`、`calico`、`coredns`、`openfuyao-system-controller`。  
`installer-service/pkg/installer/cluster_create.go` 的 `mergeAddons` 允许前端增删改 addon。

每个 addon 名称映射到 `bke-manifests/kubernetes/<name>/<version>/<name>.yaml`。
当前 `bke-manifests/kubernetes/` 目录包含：

- `bkeagent`
- `bkeagent-deployer`
- `calico`
- `cluster-api`
- `clusterextra`
- `coredns`
- `kubeproxy`
- `minio`
- `nodelocaldns`
- `numa-affinity-package`
- `openfuyao-system-controller`
- `redis`
- `victoriametrics-controller`

其中 `openfuyao-system-controller` 把 openFuyao 管理面（console-service、marketplace-service 等）作为 chart 部署。

## 4. 扩展性评估

当前安装器扩展是**目录 + 默认列表**模式：
- 新增组件需要在 `bke-manifests/kubernetes/` 新建包目录；
- 需要在 `installer-service` 代码或默认配置中注册 addon 名称；
- 没有通过 CRD 注册通用组件的能力。

这与控制台的 `ConsolePlugin` / `HelmChartRepository` 插件框架是两套体系。
通用化方向：定义 `ClusterComponentPolicy` / `ComponentCatalog` 等 CRD，让安装器通过 CRD 而非硬编码目录发现组件。

## 5. 独立使用可能性

- `cluster-api-provider-bke` + `bkeagent` 相对独立，是完整的 CAPI provider；
- `installer-service` 只依赖 K8s API，可单独部署；
- `bkeadm` 默认硬编码 `cr.openfuyao.cn` 镜像源和华为云 OBS 地址，并可选安装 openFuyao 控制台，想作为通用 K8s 发行版装器需要替换制品源和默认 addon。
