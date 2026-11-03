# 集群卸载总览

本文是openFuyao集群卸载的统一入口，说明**删除顺序**、**影响范围**、**操作节点**及各命令会删除哪些内容。具体步骤见下文各子文档。

> ![危险](./figures/icon-danger.gif) **危险：**<br />
> 卸载与删除为高危操作，执行前请确认目标集群、命名空间与资源名称，并完成必要备份。误删后可能无法通过管理面继续纳管或卸载其他集群。

## 推荐删除顺序

请按以下顺序卸载，**不要**在仍有业务/管理集群依赖时先卸载引导集群：

```text
业务集群 → 管理集群（若有） → 引导集群
```
**表1** 推荐删除顺序

| 顺序 | 对象 | 原因 |
|:---|:---|:---|
| 1 | 业务集群 | 依赖引导集群或管理集群上的`BKECluster`（`bc`）与管理面进行删除与清理。 |
| 2 | 管理集群 | 若环境中部署了管理集群，须在卸载引导集群前完成；且须同时处理**引导集群**与**管理集群自身**上的 `bc`（安装后bkeagent的侦听会从引导集群切换到管理集群）。 |
| 3 | 引导集群 | 最后执行；卸载后引导节点管理面不可用。 |

> ![注意](./figures/icon-caution.gif) **注意：**<br />
> **不能**在业务/管理集群尚未卸载时直接删除引导集群。卸载引导集群后：
>
> - openFuyao**引导节点管理面将无法登录**。
> - 无法再通过管理面或引导节点上的 `bc` 对业务/管理集群执行卸载。
> - 残留的业务/管理集群节点需另行登录各节点手工清理，风险高。

## 操作节点与入口

**表2** 卸载操作节点与入口

| 卸载对象 | Web（前台） | 命令行（后台） |
|:---|:---|:---|
| 业务集群 | **引导集群部署**：`https://<引导节点IP地址>:30010`；**管理集群部署**：`https://<管理集群IP地址>:31616`（管理面左侧导航选择“集群生命周期管理”） | 在创建该业务集群的管理面所在集群（引导或管理集群）上，对目标`bc`执行编辑/删除。 |
| 管理集群 | **两步 Web**：1.管理集群管理面`https://<管理集群IP>:31616`删除；2.引导节点管理面`https://<引导节点IP>:30010`删除。 | **先**在管理集群上删除本地`bc`，**再**在引导节点删除对应`bc`。|
| 引导集群 | **不支持**Web页面删除。 | 仅在**引导节点**执行`bke reset`等命令。 |

## 删除方式与清理范围

### 1. 业务/管理集群（通过`BKECluster`）

在引导或管理集群上编辑目标`bc`，配置说明如下：

**表3** BKECluster卸载配置说明

| 配置 | 含义 | 实际效果 |
|:---|:---|:---|
| `metadata.annotations["bke.bocloud.com/ignore-target-cluster-delete"]` | 是否忽略对**目标集群节点**的清理。 | 默认为`"true"`：只删除API上的CR等对象，**不**reset目标节点。设为 `"false"`时才会继续清理目标集群节点。 |
| `spec.reset: true` | 触发删除 | 对该`bc`发起删除，进入卸载调谐。 |
| `bke.bocloud.com/deep-restore-node` | 深度恢复 | 默认为`"true"`：清理目标节点时除Kubernetes相关外，还会清理容器运行时（如containerd/docker）等。 |
| `bke.bocloud.com/ignore-namespace-delete` | 是否保留`bc`所在命名空间 | 默认为`"true"`不删namespace；仅当为`"false"`且该namespace下无其他`bc`时才删除。 |

**完整卸载目标集群（推荐后台步骤）**需同时满足：将`ignore-target-cluster-delete`设为`"false"`，并设置`spec.reset: true`。仅删除API/`bc`对象而不清理节点时，可保持默认注解为`"true"`。

> ![说明](./figures/icon-note.gif) **说明：**<br />
> **管理集群**安装时节点代理bkeagent前期侦听引导集群，之后会切换为侦听管理集群自身，因此卸载管理集群时须分别处理引导集群与管理集群两侧（前台两侧列表删除，或后台两侧`bc`），详见具体指导文档。

卸载过程中，控制器会按依赖**逆序**清理addon、Kubernetes组件、etcd等；失败时`bc`可能进入终止失败类状态，需结合`kubectl describe bc`排查。

### 2. 引导集群（`bke reset`）

**表4** bke reset命令清理范围

| 命令 | 会删除 / 清理的内容 |
|:---|:---|
| `bke reset` | 清理本地Kubernetes/k3s过渡集群相关容器与目录。 |
| `bke reset --mount` | 删除解压/挂载相关目录；并清理容器服务、NTP、软件源等配置（单独使用时不要求`--all` 确认）。 |
| `bke reset --all` | 在普通reset基础上进一步清空节点容器服务、NTP、容器运行时、网络及相关残留；未加`--confirm`时会交互确认。 |
| `bke reset --all --mount` | 上述`--all`与`--mount`的组合；还会移除BKE管理的系统配置并恢复时区等（以实际版本行为为准）。 |

> ![说明](./figures/icon-note.gif) **说明：**<br />
> - `bke reset`**不会**自动解绑高可用场景下已绑定的虚拟IP（VIP）；若节点上仍残留VIP，需手工`ip addr del`，参见[FAQ](../cluster_installation_guide/appendix/FAQ.md)。
> - `rm -rf /bke`、`rm -f /usr/local/bin/bke` 为**可选**清理步骤，用于删除安装介质目录与`bke`命令；**非必须**。未执行也可认为引导服务已重置，但再次安装时可能需重新下载工具与介质。

参数细节见[命令参考](../cluster_installation_guide/appendix/command_reference.md)与[bkeadm 配置参数](../cluster_installation_guide/appendix/bkeadm_configuration.md)。

## 卸载成功示意（如何判断干净）

### 前台（Web）

- “集群生命周期管理”列表中对应集群已被移除。
- **管理集群**：管理集群管理面（`:31616`）无法再登录或访问；引导节点管理面（`:30010`）列表中已无该管理集群。

### 后台（命令行）

- 业务集群：创建侧集群上`kubectl get bc -A`已无对应`BKECluster`。
- 管理集群：原管理集群API/管理面已不可用；在**引导节点**上`kubectl get bc -A`已无该管理集群。
- 引导集群：`bke reset --all --mount`正常结束，本地k3s/Kubernetes相关进程与容器已清理；（可选清理后）`/bke`与`bke`命令不存在。

## 文档导航

**表5** 集群卸载文档导航

| 场景 | 文档 |
|:---|:---|
| 前台删除业务集群 | [通过前台（Web页面）删除业务集群](./delete_service_cluster_web.md) |
| 后台删除业务集群 | [通过后台（命令行）删除业务集群](./delete_service_cluster_command.md) |
| 前台删除管理集群 | [通过前台（Web页面）删除管理集群](./delete_management_cluster_web.md) |
| 后台删除管理集群 | [通过后台（命令行）删除管理集群](./delete_management_cluster_command.md) |
| 后台删除引导集群 | [通过后台（命令行）删除引导集群](./delete_bootstrap_cluster_command.md) |
