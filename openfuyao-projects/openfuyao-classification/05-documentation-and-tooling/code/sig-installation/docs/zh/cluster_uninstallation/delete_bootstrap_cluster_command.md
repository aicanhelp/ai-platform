# 通过后台（命令行）删除引导集群

本文介绍在**引导节点**通过命令行卸载引导集群。引导集群**不支持**通过openFuyao管理面Web删除。

## 前提条件

- 使用具备`sudo`权限的账号登录**引导节点**。
- **必须**已完成业务集群、管理集群的卸载（或确认可接受无法再通过管理面卸载它们）。参见[集群卸载总览](./overview.md)。

> ![危险](./figures/icon-danger.gif) **危险：**<br />
> 卸载引导集群后：
>
> - openFuyao**引导节点管理面无法登录**。
> - **无法**再通过管理面或引导节点上的`bc`卸载业务/管理集群。
> - 若业务/管理集群尚未卸载，将失去统一卸载入口，只能登录各节点手工清理。

## 操作步骤

1. 执行如下命令，重置引导节点。

   ```bash
   bke reset --all --mount
   ```

   > ![说明](./figures/icon-note.gif) **说明：**<br />
   > 未加`--confirm` 时，`--all`会在执行前要求确认。跳过确认可使用`bke reset --all --mount --confirm`。

2. （可选）删除BKE安装介质目录与bke命令。

   ```bash
   rm -rf /bke
   rm -f /usr/local/bin/bke
   ```

   > ![说明](./figures/icon-note.gif) **说明：**<br />
   > 步骤2仅为进一步清理磁盘与命令入口，**不是**卸载成功的必要条件。若需保留介质以便再次`bke init`，可跳过本步。

## 删除内容说明

| 步骤 / 命令 | 删除内容 |
|:---|:---|
| `bke reset`（含于 `--all`） | 本地Kubernetes/k3s过渡集群相关容器与目录。 |
| `--all` | 进一步清理容器服务、NTP、容器运行时、网络及相关残留。 |
| `--mount` | 删除解压/挂载相关目录；与 `--all` 同用时还会处理BKE管理系统配置、时区等。 |
| `rm -rf /bke`（可选） | 安装/解压工作目录（介质、缓存等）。 |
| `rm -f /usr/local/bin/bke`（可选） | `bke` / bkeadm命令。 |

**不会**自动处理：节点上网卡上残留的VIP（需手工解绑，见[FAQ](../cluster_installation_guide/appendix/FAQ.md)）。

## 卸载成功示意

满足以下情况可认为引导集群已卸载干净（步骤2是否执行不影响“服务已重置”的判断）：

- `bke reset --all --mount`正常结束。
- 无法再通过 `kubectl get pod -A`访问原引导集群，或本地k3s/Kubernetes相关进程与容器已不存在。
- （若执行了可选步骤）`ls /bke`失败，`which bke`无输出。

## 后续说明

- 再次使用需重新下载 BKE 工具并执行引导节点初始化，参见[在线安装引导集群](../cluster_installation_guide/bootstrap_cluster_installation/online_bootstrap_cluster_installation.md)或[离线安装引导集群](../cluster_installation_guide/bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md)。
- 更多`bke reset`参数见[命令参考](../cluster_installation_guide/appendix/command_reference.md)与[bkeadm 配置参数](../cluster_installation_guide/appendix/bkeadm_configuration.md)。
