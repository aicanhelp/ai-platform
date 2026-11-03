# 一、现状

BKE中在管理集群创建业务集群的过程中管理集群将admin权限的kubeconfig传给业务集群，业务集群用该kubeconfig监听管理集群的apiserver下发的command，但该kubeconfig权限过大，需要进行权限最小化。

# 二、BKEAgent 使用管理集群 KubeConfig 情况分析

## 概述

该章节梳理了 bkeagent 项目中，使用管理集群 localKubeConfig 访问管理集群的所有操作，以便后续创建基于客户端证书的最小权限认证和 RBAC 配置。

**认证方式**: 使用客户端证书（Client Certificate）而非 ServiceAccount Token

- **原因**: ServiceAccount Token 需要传送到业务集群节点，无法自动更新，且业务集群无法挂载管理集群的 ServiceAccount
- **方案**: 生成专用的客户端证书，通过证书的 CN (Common Name)  字段绑定 RBAC 权限

## 1、业务集群 KubeConfig 的获取和使用

### 1.1 KubeConfig 来源

- **位置**: `/etc/bkeagent/config` (在节点上)
- **来源**: 管理集群通过 `ensureBKEAgent.go` 将 localKubeConfig 推送到业务集群各节点
- **用途**: bkeagent 使用此 kubeconfig 连接管理集群的 API Server

### 1.2 客户端创建

- **文件**: `bkeagent/utils/clientutil/client.go`
- **函数**: `NewKubernetesClient(path string)`
- **创建对象**:
  - `kubernetes.Clientset` (标准 Kubernetes API)
  - `dynamic.Interface` (动态客户端，用于 CRD)

## 2、业务集群操作管理集群API 操作清单

### 2.1 Secret 操作

#### 2.1.1 读取 Secret (Get)

**位置和操作**:

```go
// bkeagent/utils/clientutil/client.go
func ClientSetFromManagerClusterSecret(nsName ...string) (*kubernetes.Clientset, error)
// bkeagent/utils/pkiutil/secret.go
func StoreClusterAPICert(secret *corev1.Secret, pkiPath string) error
// bkeagent/pkg/job/builtin/kubeadm/certs/certs.go
func (cp *CertPlugin) getCertFromSecret(name, saveTo string) error
```

**资源**:

- Secret (特定命名空间)
- 命名空间: 由 BKECluster 的 namespace 决定
- Secret 名称模式: `{clusterName}-{certName}` 或 `{clusterName}-kubeconfig`

**权限需求**:

- `secrets` - `get` (特定命名空间)

### 2.2 ConfigMap 操作

#### 2.2.1 读取 ConfigMap (Get)

**位置和操作**:

```go
//  bkeagent/pkg/job/builtin/plugin/interface.go - 读取 BKECluster 的 ConfigMap
func GetBKECluster(bkeConfigNS string) (*bkev1beta1.BKECluster, error)
// bkeagent/pkg/job/builtin/kubeadm/kubelet/run.go - 读取 kubelet ConfigMap
func (kp *kubeletPlugin) joinWorkerPrepare(config map[string]string) error
// bkeagent/pkg/job/builtin/kubeadm/certs/certs.go
func (cp *CertPlugin) loadConfigMapFromParentCluster() error
// bkeagent/job/k8s/k8s.go
func (t *Task) Execute(execCommands []string) ([]string, error)
err = t.K8sClient.Get(context.Background(), client.ObjectKey{Namespace: namespace, Name: name}, obj)
```

**资源**:

- ConfigMap (特定命名空间)
- 命名空间: BKECluster 的 namespace 或 `kube-system`
- ConfigMap 名称: BKECluster 名称或指定的 kubelet configmap 名称

**权限需求**:

- `configmaps` - `get` (特定命名空间)

#### 2.2.2 创建 ConfigMap (Create)

**位置和操作**:

```go
// bkeagent/pkg/job/builtin/kubeadm/kubeadm.go
func (kp *kubeletPlugin) createOrUpdateKubeletConfigMap() error
err = kp.K8sClient.Create(context.Background(), cm)  // 创建 kubelet ConfigMap
// bkeagent/pkg/job/k8s/k8s.go
func (t *Task) Execute(execCommands []string) ([]string, error)
err = t.K8sClient.Create(context.Background(), obj)  // 执行 Kubernetes 类型命令时创建 ConfigMap (rw 操作，如果不存在)
```

**资源**:

- ConfigMap (特定命名空间)
- 命名空间: BKECluster 的 namespace 或 `kube-system`
- ConfigMap 名称: BKECluster 名称或指定的 kubelet configmap 名称

**权限需求**:

- `configmaps` - `create` (特定命名空间，可选，仅在特定场景需要)

#### 2.2.3 更新 ConfigMap (Update)

**位置和操作**:

```go
// bkeagent/pkg/job/builtin/kubeadm/kubeadm.go
func (kp *kubeletPlugin) createOrUpdateKubeletConfigMap() error
err = kp.K8sClient.Update(context.Background(), cm)  // 更新 kubelet ConfigMap
// bkeagent/pkg/job/k8s/k8s.go
func (t *Task) Execute(execCommands []string) ([]string, error)
err = t.K8sClient.Update(context.Background(), obj)  // 执行 Kubernetes 类型命令时更新 ConfigMap (rw 操作)
```

**资源**:

- ConfigMap (特定命名空间)
- 命名空间: BKECluster 的 namespace 或 `kube-system`
- ConfigMap 名称: BKECluster 名称或指定的 kubelet configmap 名称

**权限需求**:

- `configmaps` - `update` (特定命名空间，可选，仅在特定场景需要)

### 2.3 CRD 资源操作 (通过 DynamicClient)

#### 2.3.1 读取 BKECluster CRD (Get)

**位置和操作**:

```go
// bkeagent/pkg/job/builtin/plugin/interface.go
func GetBKECluster(bkeConfigNS string) (*bkev1beta1.BKECluster, error)
gvr := schema.GroupVersionResource{
    Group:    bkev1beta1.GVK.Group,      // "bke.bocloud.com"
    Version:  bkev1beta1.GVK.Version,   // "v1beta1"
    Resource: "bkeclusters",             // 复数形式
}
bkeCluster, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Get(context.Background(), name, v1.GetOptions{})
```

**资源**:

- `bkeclusters.bke.bocloud.com` (CRD)
- 命名空间: BKECluster 的 namespace

**权限需求**:

- `bkeclusters.bke.bocloud.com` - `get` (特定命名空间)

#### 2.3.2 读取 ContainerdConfig CRD (Get)

**位置和操作**:

```go

// bkeagent/pkg/job/builtin/plugin/interface.go
func GetContainerdConfig(containerdCconfigNS string) (*bkev1beta1.ContainerdConfigSpec, error)
gvr := schema.GroupVersionResource{
    Group:    "bke.bocloud.com",
    Version:  "v1beta1",
    Resource: "containerdconfigs",
}
containerdConfig, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Get(context.Background(), name, v1.GetOptions{})
```

**资源**:

- `containerdconfigs.bke.bocloud.com` (CRD)
- 命名空间: 由参数指定

**权限需求**:

- `containerdconfigs.bke.bocloud.com` - `get` (特定命名空间)

#### 2.3.3 读取 KubeletConfig CRD (Get)

**位置和操作**:

```go
// bkeagent/pkg/job/builtin/kubeadm/kubelet/run.go
func (kp *kubeletPlugin) readConfigFromKubeletConfigCR(config map[string]string) error
gvr := schema.GroupVersionResource{
    Group:    confv1beta1.GVK.Group,     // "bke.bocloud.com"
    Version:  confv1beta1.GVK.Version,  // "v1beta1"
    Resource: "kubeletconfigs",
}
unstructuredObj, err := c.DynamicClient.Resource(gvr).Namespace(kubeletConfigNamespace).Get(
    context.Background(), kubeletConfigName, metav1.GetOptions{})
```

**资源**:

- `kubeletconfigs.bke.bocloud.com` (CRD)
- 命名空间: 通常是 `bke-system`

**权限需求**:

- `kubeletconfigs.bke.bocloud.com` - `get` (特定命名空间)

### 2.4 Command CRD 操作 (通过 Controller Runtime Client)

#### 2.4.1 读取 Command CRD (Get)

**位置和操作**:

```go
// bkeagent/controllers/command_controller.go
func (r *CommandReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error)
obj := &agentv1beta1.Command{}
err := r.Client.Get(r.Ctx, client.ObjectKey{Namespace: namespace, Name: name}, obj)  // Reconcile 时获取 Command

func (r *CommandReconciler) ttlSecondAfterFinished()
obj := &agentv1beta1.Command{}
err := r.Client.Get(r.Ctx, client.ObjectKey{Namespace: namespace, Name: name}, obj)  // TTL 清理时检查 Command

func (r *CommandReconciler) syncStatusUntilComplete(cmd *agentv1beta1.Command) (err error)
err = r.APIReader.Get(r.Ctx, client.ObjectKey{Namespace: cmd.Namespace, Name: cmd.Name}, obj)  // 更新 status 前刷新 Command
```

**资源**:

- `commands.bke.bocloud.com` (CRD)
- 命名空间: Command 的 namespace

**权限需求**:

- `commands.bke.bocloud.com` - `get` (特定命名空间)

#### 2.4.2 列出 Command CRD (List)

**位置和操作**:

```go
// bkeagent/controllers/command_controller.go
func (r *CommandReconciler) ttlSecondAfterFinished()
var commandList agentv1beta1.CommandList
err := r.Client.List(r.Ctx, &commandList)  // TTL 清理时扫描所有 Command
```

**资源**:

- `commands.bke.bocloud.com` (CRD)
- 命名空间: Command 的 namespace

**权限需求**:

- `commands.bke.bocloud.com` - `list` (特定命名空间)

#### 2.4.3 监听 Command CRD (Watch)

**位置和操作**:

```go
// bkeagent/controllers/command_controller.go
func (r *CommandReconciler) SetupWithManager(mgr ctrl.Manager) error
return ctrl.NewControllerManagedBy(mgr).
    For(&agentv1beta1.Command{}).  // SetupWithManager 设置 watch
    Complete(r)
```

**资源**:

- `commands.bke.bocloud.com` (CRD)
- 命名空间: Command 的 namespace

**权限需求**:

- `commands.bke.bocloud.com` - `watch` (特定命名空间)

#### 2.4.4 更新 Command CRD (Update)

**位置和操作**:

```go
// bkeagent/controllers/command_controller.go
func (r *CommandReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error)
err := r.Client.Update(r.Ctx, obj)  // 添加 finalizer
err := r.Client.Update(r.Ctx, obj)  // 移除 finalizer
```

**资源**:

- `commands.bke.bocloud.com` (CRD)
- 命名空间: Command 的 namespace

**权限需求**:

- `commands.bke.bocloud.com` - `update` (特定命名空间)

#### 2.4.5 更新 Command CRD Finalizers (Update)

**位置和操作**:

```go
// bkeagent/controllers/command_controller.go
func (r *CommandReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error)
obj.Finalizers = append(obj.Finalizers, finalizerName)  // 添加 finalizer
obj.Finalizers = removeString(obj.Finalizers, finalizerName)  // 移除 finalizer
err := r.Client.Update(r.Ctx, obj)
```

**资源**:

- `commands.bke.bocloud.com` (CRD)
- 操作: 更新 finalizer 子资源

**权限需求**:

- `commands.bke.bocloud.com/finalizers` - `update` (特定命名空间)

#### 2.4.6 更新 Command CRD Status (Patch/Update)

**位置和操作**:

```go
// bkeagent/controllers/command_controller.go
func (r *CommandReconciler) syncStatusUntilComplete(cmd *agentv1beta1.Command) (err error)
err = r.Client.Status().Patch(r.Ctx, objCopy, client.MergeFrom(obj))  // 更新 Command status
err = r.APIReader.Get(r.Ctx, client.ObjectKey{Namespace: cmd.Namespace, Name: cmd.Name}, obj)  // 获取 status
```

**资源**:

- `commands.bke.bocloud.com` (CRD)
- 操作: 更新 status 子资源

**权限需求**:

- `commands.bke.bocloud.com` - `patch` (特定命名空间)
- `commands.bke.bocloud.com/status` - `get`, `patch`, `update` (特定命名空间)

#### 2.4.7 删除 Command CRD (Delete)

**位置和操作**:

```go
// bkeagent/controllers/command_controller.go
func (r *CommandReconciler) ttlSecondAfterFinished()
err = r.Delete(r.Ctx, obj)
```

**资源**:

- `commands.bke.bocloud.com` (CRD)

**权限需求**:

- `commands.bke.bocloud.com` - `delete` (特定命名空间)

## 3、权限需求总结

### 3.1 核心 API 资源权限

| 资源类型     | 操作                      | 命名空间                            | 说明                     |
| ------------ | ------------------------- | ----------------------------------- | ------------------------ |
| `secrets`    | `get` | BKECluster namespace                | 读写权限，用于读取、创建和更新证书 Secret |
| `configmaps` | `get`, `list`, `watch`    | `cluster-system` | 只读权限，用于读取 ConfigMap |
| `configmaps` | `get`, `list`, `watch`, `create`, `update` | BKECluster namespace                | 读写权限，用于读取、创建和更新 ConfigMap |

### 3.2 CRD 资源权限

| CRD 资源                            | 操作                     | 命名空间             | 说明               |
| ----------------------------------- | ------------------------ | -------------------- | ------------------ |
| `bkeclusters.bke.bocloud.com`       | `get`, `list`, `watch`   | 所有命名空间        | 读取集群配置（通过 ClusterRoleBinding） |
| `containerdconfigs.bke.bocloud.com` | `get`, `list`, `watch`   | 所有命名空间        | 读取容器运行时配置（通过 ClusterRoleBinding） |
| `kubeletconfigs.bke.bocloud.com`    | `get`, `list`, `watch`   | 所有命名空间        | 读取 kubelet 配置（通过 ClusterRoleBinding） |
| `commands.bkeagent.bocloud.com`     | `get`, `list`, `watch`, `update`, `patch`, `delete` | 所有命名空间        | Command CRD 操作（通过 ClusterRoleBinding） |
| `commands.bkeagent.bocloud.com/finalizers` | `get`, `list`, `watch`, `update` | 所有命名空间        | 更新 Command finalizer（通过 ClusterRoleBinding） |
| `commands.bkeagent.bocloud.com/status`   | `get`, `patch`, `update` | 所有命名空间        | 更新 Command 状态（通过 ClusterRoleBinding） |

### 3.3 命名空间范围和权限分配

**主要命名空间**:

1. **BKECluster 的 namespace** (动态命名空间):
   
   - RoleBinding: `bkeagent-readwrite`
   - 权限: secrets (get), configmaps (get, list, watch, create, update)
   - 用途: 读取 Secret (证书), 读取/创建 ConfigMap (集群配置)
2. **cluster-system** (固定命名空间):
   
   - RoleBinding: `bkeagent-configmap-only`
   - 权限: configmaps (get, list, watch)
   - 用途: 仅读取 ConfigMap，不提供 Secret 权限
3. **所有命名空间** (通过 ClusterRoleBinding):
   
   - ClusterRoleBinding: `bkeagent-cluster-access`
   - 权限: 所有 CRD 资源权限
   - 用途: 跨命名空间访问 CRD 资源

## 4、基于证书的最小权限 RBAC 配置方案

### 4.1 证书生成规范

#### 4.1.1 证书配置

生成客户端证书时，需要设置以下字段：

- **CN (Common Name)**: `bkeagent-cert-user` - 作为用户名，用于 RBAC 绑定
- **有效期**: 100年
- **CA**: 使用管理集群的 CA 证书签名（从 Secret `{bkeCluster.Name}-ca` 获取）

### 4.2 ClusterRole (定义权限)

**重要说明**: ClusterRole 本身是集群级别的，不能限制命名空间。但是，**RoleBinding 可以引用 ClusterRole**，当 RoleBinding 引用 ClusterRole 时，ClusterRole 的权限会被限制到 RoleBinding 所在的命名空间。这是 Kubernetes RBAC 的一个特性。

为了限制 Secret 和 ConfigMap 只能访问特定命名空间，需要将权限分为多个 ClusterRole：
**ClusterRole 1: bkeagent-readwrite（读写权限，用于 BKECluster 命名空间）**

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: bkeagent-readwrite
rules:
  # Core API 资源 - 通过 RoleBinding 限制命名空间
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch", "create", "update"]
```

**ClusterRole 2: bkeagent-configmap-only（仅 ConfigMap 权限，用于 cluster-system 命名空间）**

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: bkeagent-configmap-only
rules:
  # 仅提供 ConfigMap 权限，不包含 Secret 权限
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch"]
```

**ClusterRole 3: bkeagent-cluster-access（用于 CRD 资源，通过 ClusterRoleBinding 允许跨命名空间）**

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: bkeagent-cluster-access
rules:
  # BKE CRDs - 需要跨命名空间访问
  - apiGroups: ["bke.bocloud.com"]
    resources: ["bkeclusters"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["bke.bocloud.com"]
    resources: ["containerdconfigs"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["bke.bocloud.com"]
    resources: ["kubeletconfigs"]
    verbs: ["get", "list", "watch"]
  # BKEAgent CRDs - 读写权限（用于 Command 状态更新）
  - apiGroups: ["bkeagent.bocloud.com"]
    resources: ["commands"]
    verbs: ["get", "list", "watch", "update", "patch", "delete"]
  - apiGroups: ["bkeagent.bocloud.com"]
    resources: ["commands/finalizers"]
    verbs: ["get", "list", "watch", "update"]
  - apiGroups: ["bkeagent.bocloud.com"]
    resources: ["commands/status"]
    verbs: ["get", "patch", "update"]
```

**说明**:

- ClusterRole 定义权限规则，但不限制命名空间
- 当 RoleBinding 引用这些 ClusterRole 时，权限会被限制到 RoleBinding 所在的命名空间
- 例如：在 `bke-cluster` 命名空间创建 RoleBinding 引用 `bkeagent-readwrite` ClusterRole，则只能访问 `bke-cluster` 命名空间的 Secret/ConfigMap
- `bkeagent-cluster-access` 通过 ClusterRoleBinding 绑定，权限在所有命名空间生效，用于 CRD 资源的跨命名空间访问

### 4.3 RoleBinding 和 ClusterRoleBinding (绑定证书用户到权限)

**关键机制**:

- **RoleBinding 引用 ClusterRole**: 当 RoleBinding 引用 ClusterRole 时，ClusterRole 的权限会被限制到 RoleBinding 所在的命名空间
- **ClusterRoleBinding 引用 ClusterRole**: 当 ClusterRoleBinding 引用 ClusterRole 时，权限在所有命名空间生效

为了限制 Secret 和 ConfigMap 只能访问特定命名空间，需要使用 RoleBinding 来限制命名空间访问，同时使用 ClusterRoleBinding 允许 CRD 资源的跨命名空间访问。

**步骤 1**: 为每个需要的命名空间创建 RoleBinding（限制 Secret/ConfigMap 的命名空间）

**注意**: 不同的命名空间使用不同的 ClusterRole，以实现不同级别的权限控制。

```yaml
# 为 cluster-system 命名空间创建 RoleBinding（仅 ConfigMap 权限）
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: bkeagent
  namespace: cluster-system
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: bkeagent-configmap-only
subjects:
- kind: User
  name: bkeagent-cert-user
  apiGroup: rbac.authorization.k8s.io
```

```yaml
# 为 BKECluster 的命名空间创建 RoleBinding（读写权限，动态命名空间）
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: bkeagent
  namespace: <bkecluster-namespace>  # 从 BKECluster 资源动态获取
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: bkeagent-readwrite
subjects:
- kind: User
  name: bkeagent-cert-user
  apiGroup: rbac.authorization.k8s.io
```

**步骤 2**: 对于 CRD 资源（commands, bkeclusters 等），创建 ClusterRoleBinding

由于 CRD 资源需要跨命名空间访问，需要创建一个 ClusterRoleBinding 绑定到 `bkeagent-cluster-access` ClusterRole。

**注意**: ClusterRoleBinding 引用 ClusterRole 时，权限在所有命名空间生效，这正是 CRD 资源需要的。

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: bkeagent-cluster-access
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: bkeagent-cluster-access
subjects:
- kind: User
  name: bkeagent-cert-user
  apiGroup: rbac.authorization.k8s.io
```

**说明**:

- `kind: User` 表示这是一个证书用户（而非 ServiceAccount）
- `name: bkeagent-cert-user` 必须与证书的 CN 字段完全匹配
- 这样，使用该证书的客户端就会被识别为用户 `bkeagent-cert-user`，并获得相应的权限

**实现建议**:

1. **动态获取 BKECluster namespace**:
   
   ```go
   // 从 BKECluster 资源获取 namespace
   bkeCluster := &bkev1beta1.BKECluster{}
   err := client.Get(ctx, client.ObjectKey{Name: clusterName}, bkeCluster)
   clusterNamespace := bkeCluster.Namespace
   ```
2. **为每个命名空间创建 RoleBinding**:
   
   - `cluster-system`: 固定命名空间，绑定到 `bkeagent-configmap-only`，仅提供 ConfigMap 权限
   - `BKECluster.Namespace`: 动态命名空间，绑定到 `bkeagent-readwrite`，从 BKECluster 资源获取
3. **权限分离**:
   
   - 使用 RoleBinding 限制 Secret/ConfigMap 的命名空间访问
   - 使用 ClusterRoleBinding 允许 CRD 资源的跨命名空间访问

**优点**:

- 可以精确控制 Secret 和 ConfigMap 的命名空间访问范围
- 不同命名空间可以使用不同级别的权限（只读、读写、仅 ConfigMap）
- 提高安全性，防止访问未授权的命名空间资源
- 支持动态命名空间（BKECluster namespace）

**缺点**:

- 需要为每个命名空间创建 RoleBinding
- 需要维护多个 RBAC 资源

### 4.4 注意事项

1. **证书 CN 必须匹配**:
   
   - RBAC 中的 `name: bkeagent-cert-user` 必须与证书的 CN 字段完全一致
   - Kubernetes 在认证时会将证书的 CN 作为用户名
   - 证书 CN 在 `generateBKEAgentClientCert` 函数中设置为 `bkeagent-cert-user`
2. **证书有效期管理**:
   
   - 证书有效期设置为100年（`oneHundredYear * oneYearDay * oneDayHour * time.Hour`）
   - 需要在证书过期前重新生成并推送到所有节点
   - 暂不考虑实现证书自动轮换机制
3. **命名空间限制**:
   
   - 使用 RoleBinding 限制 Secret/ConfigMap 只能访问特定命名空间
   - 需要为每个命名空间创建 RoleBinding：
     - `cluster-system`: 绑定到 `bkeagent-configmap-only`
     - `BKECluster.Namespace`: 绑定到 `bkeagent-readwrite`（动态命名空间）
4. **动态命名空间处理**:
   
   - BKECluster 的 namespace 是动态的，需要从 BKECluster 资源动态获取
   - 实现时在 `CreateBKEAgentRBACWithLocalKubeConfig` 函数中：
     ```go
     if bkeCluster != nil {
         clusterNamespace := bkeCluster.Namespace
         if clusterNamespace != "" {
             createRoleBindingForNamespace(ctx, highPrivilegeClient, clusterNamespace, "bkeagent-readwrite")
         }
     }
     ```
5. **子资源权限**:
   
   - `commands/status` 和 `commands/finalizers` 需要单独声明，因为它们是子资源
   - 在 `bkeagent-cluster-access` ClusterRole 中单独定义
6. **CRD 资源的跨命名空间访问**:
   
   - CRD 资源（commands, bkeclusters 等）通过 `bkeagent-cluster-access` ClusterRoleBinding 绑定
   - 权限分为两部分：
     - 多个 ClusterRole（`bkeagent-readwrite`、`bkeagent-configmap-only`）：包含 Secret/ConfigMap 等核心资源权限（通过 RoleBinding 限制命名空间）
     - ClusterRole `bkeagent-cluster-access`：包含 CRD 资源权限（通过 ClusterRoleBinding 允许跨命名空间）
7. **命名空间自动创建**:
   
   - `createRoleBindingForNamespace` 函数会先调用 `ensureNamespaceExists` 确保命名空间存在
   - 如果命名空间不存在，会自动创建

## 5、实现步骤

### 5.1 证书生成和 KubeConfig 创建

1. **生成客户端证书**:
   
   - 使用管理集群的 CA 证书签名（从 Secret `{bkeCluster.Name}-ca` 获取）
   - 设置 CN 为 `bkeagent-cert-user`
   - 设置100年有效期（`oneHundredYear * oneYearDay * oneDayHour * time.Hour`）
   - 实现函数: `generateBKEAgentClientCert` (在 `localkubeconfig.go` 中)
2. **创建 KubeConfig**:
   
   - 使用生成的客户端证书和私钥
   - 从 remoteLocalKubeConfig 提取管理集群的 API Server 地址
   - 嵌入 CA 证书
   - 实现函数: `createKubeConfigWithClientCert` (在 `localkubeconfig.go` 中)
3. **验证证书**:
   
   - 使用新 kubeconfig 测试连接
   - 确认可以访问管理集群 API Server

### 5.2 RBAC 配置

1. **创建 ClusterRoles**:
   
   - `bkeagent-readwrite`: secrets (get), configmaps (get, list, watch, create, update)
   - `bkeagent-configmap-only`: configmaps (get, list, watch)
   - `bkeagent-cluster-access`: CRD 资源权限
   - 实现函数: `createBKEAgentClusterRoles` (在 `localkubeconfig.go` 中)
2. **创建固定命名空间的 RoleBinding**:
   
   ```go
   createRoleBindingForNamespace(ctx, client, "cluster-system", "bkeagent-configmap-only")
   ```
3. **动态获取 BKECluster namespace 并创建 RoleBinding**:
   
   ```go
   // 在 CreateBKEAgentRBACWithLocalKubeConfig 函数中
   if bkeCluster != nil {
       clusterNamespace := bkeCluster.Namespace
       if clusterNamespace != "" {
           // 为 BKECluster 的 namespace 创建 RoleBinding（读写权限）
           createRoleBindingForNamespace(ctx, client, clusterNamespace, "bkeagent-readwrite")
       }
   }
   ```
4. **创建 ClusterRoleBinding（用于 CRD 资源）**:
   
   ```go
   // 创建 ClusterRoleBinding，绑定 bkeagent-cluster-access ClusterRole
   createBKEAgentClusterAccessRoleBinding(ctx, client)
   ```
5. **实现细节**:
   
   - `createRoleBindingForNamespace`: 创建或更新 RoleBinding，自动确保命名空间存在
   - `getBKEAgentSubject`: 返回 Subject，用户名为 `bkeagent-cert-user`
   - `ensureNamespaceExists`: 确保命名空间存在，不存在则创建
   - `createOrUpdateRoleBinding`: 创建或更新 RoleBinding
   - `createOrUpdateClusterRoleBinding`: 创建或更新 ClusterRoleBinding
6. **验证权限**:
   
   - 使用新 kubeconfig 测试各个操作
   - 确认 Secret/ConfigMap 只能访问指定命名空间
   - 确认 CRD 资源可以跨命名空间访问

### 5.3 代码修改

1. **修改 ensureBKEAgent.go**:
   
   - `loadLocalKubeConfig()` 方法：
     - 如果 `!hasClusterAPI`，先尝试获取 `GetLeastPrivilegeKubeConfig`
     - 如果失败，回退到 `GetLocalKubeConfig`（不需要创建 RBAC）
     - 如果成功，使用 `GetLocalKubeConfig` 创建 RBAC 资源
     - 如果 `hasClusterAPI`，直接使用 `GetLocalKubeConfig`
2. **证书管理**:
   
   - 证书生成逻辑已在 `localkubeconfig.go` 中实现
   - 证书推送逻辑在 `ensureBKEAgent.go` 的 `pushAgent` 方法中
3. **RBAC 创建**:
   
   - RBAC 创建逻辑在 `CreateBKEAgentRBACWithLocalKubeConfig` 函数中实现
   - 仅在 `GetLeastPrivilegeKubeConfig` 成功时创建 RBAC
4. **测试验证**:
   
   - 确保所有 bkeagent 功能正常工作
   - 验证权限限制生效
   - 测试跨命名空间访问
   - 验证回退机制正常工作


