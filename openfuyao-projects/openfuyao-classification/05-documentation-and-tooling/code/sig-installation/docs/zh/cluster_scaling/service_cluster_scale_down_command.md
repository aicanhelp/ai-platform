# 通过后台（命令行）缩容业务集群

集群缩容是指将已有的工作节点（Worker Node）从业务集群中移除，以释放计算资源。通过命令行操作Kubernetes自定义资源（CR），用户可以在后台完成业务集群的缩容操作。

## 前提条件

- 待缩容的业务集群状态为“健康”。
- 需在创建该集群时使用的引导节点或管理集群上进行缩容操作。
- 待删除的节点为工作节点（Worker Node），且节点上无关键业务运行，或已做好Pod迁移准备。

> ![输入图片说明](figures/icon-notice.gif) **须知：**
> - 集群处于非健康状态时进行缩容操作可能会出现错误。请在集群状态为“健康”时进行操作。
> - 待删除节点上的Pod将被强制驱逐或删除，请确保业务可接受节点下线。
> - 集群缩容操作必须在创建该集群时使用的引导节点或管理集群上进行，否则无法管理目标集群。

## 使用限制

- 缩容时仅支持删除工作节点（Worker Node），不支持手动删除Master节点。通过Webhook校验，手动删除Master角色的`BKENode`资源会被拒绝，只有控制器发起的删除或节点处于Failed状态时才允许删除。
- 删除节点后将无法恢复，请谨慎操作。
- 节点处于`Pending`（部署中）或`Upgrading`（升级中）状态时不允许删除。
- 当前仅支持IPv4地址的节点。

## 操作步骤

### 1. 查看集群节点信息

确认待缩容集群的名称和命名空间，并查看当前集群的节点信息，确定需要删除的节点。

```bash
# 查看所有BKECluster资源，确认集群名称和命名空间（namespace与集群名称通常一致）
kubectl get bkecluster -A

# 查看集群当前的BKENode资源，关注Name和State字段
# 将<bke-cluster>替换为实际的集群命名空间
kubectl get bn -n bke-cluster
```

输出示例：

```text
NAME             STATE    IP              ROLE     HOSTNAME
bke-cluster-m1   Ready    192.168.200.1   master   m1
bke-cluster-n1   Ready    192.168.200.2   node     n1
bke-cluster-n2   Ready    192.168.200.3   node     n2
```

> ![输入图片说明](figures/icon-note.gif) **说明：**
> - `ROLE`为`master`的节点为Master节点，不可删除。
> - `ROLE`为`node`的节点为Worker节点，可执行缩容删除操作。
> - 请记录待删除节点的`BKENode`资源名称（NAME列），后续步骤将使用。

### 2. 删除BKENode资源

执行以下命令删除待缩容节点对应的`BKENode`资源，触发缩容流程。

**单个节点删除：**

```bash
# 将bke-cluster替换为实际的集群命名空间，bke-cluster-n1替换为实际的BKENode资源名称
kubectl delete bn -n bke-cluster bke-cluster-n1
```

**批量删除：**

```bash
# 同时删除多个节点
kubectl delete bn -n bke-cluster bke-cluster-n1 bke-cluster-n2
```

> ![输入图片说明](figures/icon-note.gif) **说明：**
> - 删除`BKENode`资源后，命令会立即返回，缩容操作在后台异步执行。
> - 控制器会自动处理节点Drain、Pod驱逐、kubeadm reset等操作，无需手动干预。

### 3. 查看缩容进度

删除`BKENode`资源后，cluster-api-provider-bke控制器会自动开始缩容流程。可通过以下命令查看缩容进度。

```bash
# 查看BKECluster状态，ClusterStatus字段显示集群状态
kubectl get bkecluster -n bke-cluster

# 查看剩余BKENode资源，确认待删除节点是否已移除
kubectl get bn -n bke-cluster

# 查看集群事件，了解缩容详细进度
kubectl describe bkecluster bke-cluster -n bke-cluster
```

缩容过程中，集群状态（ClusterStatus）变化如下：

| 集群状态 | 说明 |
|:----- |:----------- |
| ScalingWorkerNodesDown | 工作节点缩容进行中，包括Pod驱逐、节点Drain、kubeadm reset等操作。 |
| Ready | 缩容完成，节点已成功移除，集群处于健康状态。 |
| WorkerScalingDownFailed | 缩容失败，需排查原因。 |

> ![输入图片说明](figures/icon-note.gif) **说明：**
> - 缩容过程是异步的，删除`BKENode`资源后命令会立即返回，缩容操作在后台异步执行。
> - 缩容时间视节点上运行的Pod数量和网络情况而定，通常每个节点需要3~10分钟。
> - Pod驱逐阶段，如果节点上存在无法驱逐的Pod（如DaemonSet Pod、本地存储Pod等），可能会延长缩容时间。无法驱逐的Pod将在节点删除后被强制清理。

### 4. 验证缩容结果

缩容完成后，验证节点是否已从集群中移除。

```bash
# 查看管理集群（引导节点）中BKENode资源，确认待删除节点已不在列表中
kubectl get bn -n bke-cluster

# 查看业务集群节点列表，确认节点已移除
kubectl get nodes
```

待删除的节点不再出现在节点列表中，且集群状态恢复为`Ready`，表示缩容成功。

> ![输入图片说明](figures/icon-notice.gif) **须知：**
> 如果集群状态长时间停留在`ScalingWorkerNodesDown`，或显示为`WorkerScalingDownFailed`，请参考[集群部署问题定位指导](../cluster_installation_guide/appendix/cluster_deployment_issue_diagnosis_guide.md)排查原因。常见问题包括节点上存在Pod无法驱逐、节点失联、节点处于异常状态等。

## 后续操作

完成集群缩容后，如需对集群进行扩容操作，请参见[通过后台（命令行）扩容业务集群](./service_cluster_scale_up_command.md)。
