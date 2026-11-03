# 通过后台（命令行）扩容业务集群

集群扩容是指在已有业务集群的基础上，向集群中新增工作节点（Worker Node），以提升集群的计算资源和调度能力。通过命令行操作Kubernetes自定义资源（CR），用户可以在后台完成业务集群的扩容操作。

## 前提条件

- 待扩容的业务集群状态为“健康”。
- 需在创建该集群时使用的引导节点或管理集群上进行扩容操作。
- 已准备好待加入集群的新节点，且节点满足以下要求：
  - 节点可以连通管理集群网络。
  - 节点可以使用root用户通过SSH登录。
  - 节点为裸机操作系统，未安装任何docker与Kubernetes组件。
  - 节点的IP地址未被任何已有集群使用。
  - 节点的时间与管理集群时间差不超过10秒。

> ![输入图片说明](figures/icon-notice.gif) **须知：**
> - 集群处于非健康状态时进行扩容操作可能会出现错误。请在集群状态为“健康”时进行操作。
> - 集群扩容操作必须在创建该集群时使用的引导节点或管理集群上进行，否则无法管理目标集群。

## 使用限制

- 扩容时仅支持添加工作节点（Worker Node）。通过Webhook校验，已有集群中不允许新增Master节点（创建`BKENode`资源时role包含master会被拒绝）。
- 待加入节点的IP地址必须在所有集群中唯一，否则`BKENode`创建时会被Webhook校验拦截。
- 节点的role创建后不可更改，不支持将工作节点变更为Master节点。
- 当前仅支持IPv4地址的节点。

## 操作步骤

### 1. 查看集群信息

确认待扩容集群的名称和命名空间，并查看当前集群的节点信息。

```bash
# 查看所有BKECluster资源，确认集群名称和命名空间（namespace与集群名称通常一致）
kubectl get bkecluster -A

# 查看集群当前的BKENode资源
# 将<bke-cluster>替换为实际的集群命名空间
kubectl get bn -n bke-cluster
```

### 2. 编写节点配置文件

创建新节点的`BKENode`资源配置文件（如`newNode.yaml`）。

```yaml
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: bke-cluster-n1          # 节点名称，需在集群内唯一
  namespace: bke-cluster         # 集群命名空间，与BKECluster的namespace一致
  labels:
    cluster.x-k8s.io/cluster-name: bke-cluster   # 关联集群名称，与BKECluster的name一致
spec:
  hostname: n1                   # 节点hostname
  ip: 192.168.200.101            # 修改为实际节点IP
  password: '<密码>'          # 修改为实际节点密码，Webhook会自动AES加密
  port: "22"                     # SSH端口，默认为22
  role:
  - node                         # 扩容时role必须为node（工作节点）
  username: root                 # SSH用户名，默认为root
```

**表 1** BKENode配置参数说明

| 参数 | 说明 |
|:---------- |:------- |
| metadata.name | BKENode资源名称，需在集群命名空间内唯一。建议使用“集群名称-节点序号”格式。 |
| metadata.namespace | 命名空间，需与待扩容的BKECluster所在namespace一致。 |
| metadata.labels | 必须包含`cluster.x-k8s.io/cluster-name`标签，值为目标BKECluster的名称，用于关联节点与集群。 |
| spec.hostname | 节点的hostname，将作为Kubernetes节点的名称。 |
| spec.ip | 节点的IP地址，需为有效的IPv4地址，且在所有集群中唯一。 |
| spec.port | SSH登录端口号，默认为“22”。 |
| spec.username | SSH登录用户名，默认为root。 |
| spec.password | SSH登录密码。创建时填写密码，Webhook会自动进行AES加密存储。 |
| spec.role | 节点角色。扩容时必须设置为`node`（工作节点）。不支持在已有集群中新增master角色节点。 |

> ![输入图片说明](figures/icon-note.gif) **说明：**
> - 如果需要一次扩容多个节点，可以在同一个YAML文件中使用`---`分隔多个`BKENode`资源定义，或者分别创建多个配置文件。
> - `metadata.labels`中的`cluster.x-k8s.io/cluster-name`值必须与`BKECluster`的`metadata.name`一致，且`metadata.namespace`必须与`BKECluster`的`metadata.namespace`一致，否则节点无法被正确关联到目标集群。

### 3. 创建BKENode资源

执行以下命令创建`BKENode`资源，触发扩容流程。

```bash
kubectl apply -f newNode.yaml
```

如果一次扩容多个节点，可将多个节点配置写入同一文件：

```bash
kubectl apply -f newNodes.yaml
```

### 4. 查看扩容进度

创建`BKENode`资源后，cluster-api-provider-bke控制器会自动开始扩容流程。可通过以下命令查看扩容进度。

```bash
# 查看BKENode资源状态，State字段显示节点状态
kubectl get bn -n bke-cluster

# 查看BKECluster状态，ClusterStatus字段显示集群状态
kubectl get bkecluster -n bke-cluster

# 查看节点详情，关注State和Message字段
kubectl describe bn bke-cluster-n1 -n bke-cluster
```

扩容过程中，节点状态（State）变化如下：

| 节点状态 | 说明 |
|:----- |:----------- |
| Initializing | bkeagent正在推送到节点，环境初始化中。 |
| BootStrapping | 节点正在执行kubeadm join，加入集群中。 |
| NotReady | 节点已加入集群，但尚未通过健康检查。 |
| Ready | 节点已就绪，扩容成功。 |

集群状态（ClusterStatus）变化如下：

| 集群状态 | 前端显示 | 说明 |
|:----- |:----- |:----------- |
| Initializing | 安装中 | 推送bkeagent代理到新节点、节点环境初始化（安装containerd等组件）、后置脚本处理阶段。 |
| ScalingWorkerNodesUp | 扩容中 | 工作节点正在执行kubeadm join，加入业务集群。 |
| Ready | 健康 | 扩容完成，集群处于健康状态。 |
| WorkerScalingUpFailed | 扩缩容失败 | 扩容过程中出现错误，节点未能成功加入集群。 |

> ![输入图片说明](figures/icon-note.gif) **说明：**
> - 扩容过程是异步的，创建`BKENode`资源后命令会立即返回，扩容操作在后台异步执行。
> - 扩容过程中，集群状态会在`Initializing`（安装中）和`ScalingWorkerNodesUp`（扩容中）之间切换，属于正常现象。
> - 扩容过程中可以通过`kubectl describe bkecluster <集群名称> -n <命名空间>`查看事件（Events）了解详细进度。
> - 扩容时间视节点网络情况和环境配置而定，通常每个节点需要3~10分钟。

### 5. 验证扩容结果

扩容完成后，在目标业务集群中验证新节点是否已成功加入。

```bash
# 查看管理集群（引导节点）中BKENode资源，确认节点已在列表中
kubectl get bn -n bke-cluster

# 查看业务集群节点列表
kubectl get nodes
```

新增节点的状态应为`Ready`，表示扩容成功。

> ![输入图片说明](figures/icon-notice.gif) **须知：**
> 如果节点状态长时间停留在`Initializing`或`BootStrapping`，或集群状态显示为`WorkerScalingUpFailed`，请参考[集群部署问题定位指导](../cluster_installation_guide/appendix/cluster_deployment_issue_diagnosis_guide.md)排查原因。常见问题包括节点网络不通、SSH密码错误、时间不同步等。

## 后续操作

完成集群扩容后，如需对集群进行缩容操作，请参见[通过后台（命令行）缩容业务集群](./service_cluster_scale_down_command.md)。
