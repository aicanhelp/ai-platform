# 通过后台（命令行）删除管理集群

本文介绍卸载管理集群。管理集群安装过程中，节点代理bkeagent **前期侦听引导集群**，创建完成后会**切换为侦听管理集群自身**，因此管理集群相关的`BKECluster`（`bc`）会同时存在于：

1. **引导集群**（创建时下发的对象）。
2. **管理集群自身**（侦听切换后的实际调谐入口）。

彻底卸载时，**两处`bc`都需要处理**，缺一不可。

## 前提条件

- 使用具备`sudo`权限的账号操作，并具备目标集群的`kubectl`访问能力。
- 可分别登录**引导节点**与**管理集群控制节点**。
- 管理集群上纳管的业务集群应已卸载。
- 已阅读[集群卸载总览](./overview.md)。

## 操作步骤

### 步骤 1：在管理集群上删除本地`bc`（清理目标节点）

侦听已切换到管理集群后，对节点的 reset/清理由**管理集群自身**上的`bc`调谐。在管理集群控制节点执行：

```bash
# kubeconfig 指向管理集群
kubectl get bc -A
kubectl edit bc -n <命名空间> <名称>
```

彻底清理管理集群节点时，修改为如下配置：

```yaml
metadata:
  annotations:
    bke.bocloud.com/ignore-target-cluster-delete: "false"
spec:
  reset: true
```

保存后可短暂观察进度；卸载推进后管理集群API往往不可用，属于正常预期现象，无需再以管理集群上`kubectl get bc`结果作为成功依据，继续步骤2。

### 步骤 2：在引导集群上删除对应`bc`

在**引导节点**上（kubeconfig 指向引导集群）再次处理同名/对应的管理集群`bc`，清理引导集群侧残留对象：

```bash
# kubeconfig 指向引导集群
kubectl get bc -A
kubectl edit bc -n <命名空间> <名称>
```

将`spec.reset`设为`true`（若仅清理API对象、节点已在步骤1清理，可将`ignore-target-cluster-delete`保持默认`"true"`，避免对已清理环境重复 reset）：

```yaml
metadata:
  annotations:
    bke.bocloud.com/ignore-target-cluster-delete: "true"   # 节点已在步骤1清理时可保持默认
spec:
  reset: true
```

或直接删除：

```bash
kubectl delete bc -n <命名空间> <名称>
```

> ![说明](./figures/icon-note.gif) **说明：**<br />
> - 步骤1、步骤2的命名空间与名称以各集群上`kubectl get bc -A`的实际输出为准，可能一致也可能因部署命名而不同，请分别核对。
> - 若侦听尚未切换、仍仅由引导集群调谐，可先在引导集群上按业务集群相同方式设置`ignore-target-cluster-delete: "false"` 与 `spec.reset: true`；切换完成后仍须确认管理集群本地是否还有 `bc`并清理。

## 删除内容说明

| 位置 | 作用 |
|:---|:---|
| 管理集群上的`bc`（步骤1） | 在侦听已切换的前提下，驱动对管理集群节点的卸载/reset（需`ignore-target-cluster-delete: "false"`才会清理节点）。 |
| 引导集群上的`bc`（步骤2） | 删除引导集群上的对应CR，避免残留。 |

## 卸载成功示意

- 原管理集群API/管理面已不可用。
- 在**引导节点**上：`kubectl get bc -A`已无该管理集群。

## 后续说明

- 注解与`bkeagent-listener`说明见[cluster-api-provider-bke 配置参数](../cluster_installation_guide/appendix/cluster-api-provider-bke_configuration.md)。
- 其他卸载方式与删除顺序见[集群卸载总览](./overview.md)。
