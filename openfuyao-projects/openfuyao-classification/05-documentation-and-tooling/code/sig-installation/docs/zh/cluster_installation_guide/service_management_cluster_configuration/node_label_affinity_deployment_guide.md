# 节点标签与亲和部署配置

## 背景信息

在使用BKE安装部署工具进行集群搭建时，用户可以实现：

1. **节点标签能力**：为集群中的节点添加自定义标签，用于描述节点特性（如硬件规格、业务用途等），支持多标签配置。
2. **Addon亲和部署**：在部署Addon组件时，根据节点标签进行亲和调度，确保特定Addon只部署在符合条件的节点上。

## 前提条件

- 已完成引导节点初始化，引导集群运行正常。
- 已准备好集群配置文件。

> ![](../figures/icon-note.gif)**说明**：
>
> 下文中bke-cluster.yaml和bke-node.yaml是引导节点上用于配置集群安装信息的文件，位于`/bke/cluster/`目录下。这些文件用于定义集群配置、Addon配置及节点信息等内容。

## 使用限制

- 节点标签的`key`和`value`需符合[Kubernetes标签命名规范](https://kubernetes.io/zh-cn/docs/concepts/overview/working-with-objects/labels/)。
---

## 操作步骤

1. 为节点添加标签。

   在`bke-node.yaml`配置文件的`spec`部分，为目标节点添加`labels`字段：

   ```yaml
   apiVersion: bke.bocloud.com/v1beta1
   kind: BKENode
   metadata:
     name: master-1
     namespace: bke-cluster
     labels:
       cluster.x-k8s.io/cluster-name: bke-cluster
   spec:
     hostname: master-1
     ip: 192.168.100.226
     password: '******'
     port: "22"
     role:
     - master/node
     - etcd
     username: root
     labels:
       - key: "machine-48"
         value: "true"
       - key: "gpu"
         value: "nvidia"
   ```

   > ![](../figures/icon-note.gif)**说明**：
   >
   > - 每个节点可配置多个标签。
   > - 标签将在集群创建时自动应用到对应节点。

2. 在Addon模板中添加nodeSelector支持。

   在`bke-manifests/kubernetes/<addon-name>/<version>/`目录下的Addon部署YAML文件中，为Pod规格添加`nodeSelector`模板变量：

   ```yaml
   # 示例：bke-manifests/redis/latest/deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: redis
   spec:
     template:
       spec:
         nodeSelector: {{ .nodeSelector }}  # 添加此行以支持节点选择
         containers:
         - name: redis
           image: {{ .imageRepo }}/redis:latest
   ```

   > ![](../figures/icon-note.gif)**说明**：
   >
   > - `{{ .nodeSelector }}`会在部署时被替换为`bke-cluster.yaml`中配置的JSON对象。
   > - 仅需对需要亲和调度的Addon模板进行修改。

3. 配置Addon节点亲和参数。

   在`bke-cluster.yaml`的`addons`部分，为需要亲和调度的Addon添加`nodeSelector`参数：

   ```yaml
   spec:
     clusterConfig:
       addons:
       - name: redis
         param:
           nodeSelector: '{"machine-48":"true"}'
         version: latest
   ```

   > ![](../figures/icon-note.gif)**说明**：
   >
   > - `nodeSelector`的值为JSON格式字符串，需使用单引号包裹。
   > - 确保目标节点已配置对应的标签，否则Addon将无法调度。

4. 创建集群。

   执行集群创建命令：

   ```bash
   bke cluster create -f /path/to/bke-cluster.yaml -n /path/to/bke-node.yaml
   ```

5. 验证亲和部署。


   集群创建完成后，验证Addon是否部署在指定节点：

   ```bash
   kubectl get pod -o wide -A | grep redis
   ```

   确认Pod所在节点是否包含配置的标签：

   ```bash
   kubectl get node <node-name> --show-labels
   ```

## 配置示例

以下为包含节点标签的完整配置片段：

**bke-cluster.yaml:**

```yaml
spec:
  clusterConfig:
    addons:
    - name: kubeproxy
      param:
        clusterNetworkMode: calico
      version: v1.33.1
    - name: calico
      param:
        calicoMode: vxlan
      version: v3.27.3
    - name: coredns
      version: v1.10.1
    - name: redis
      param:
        nodeSelector: '{"machine-48":"true"}'
      version: latest
```

**bke-node.yaml:**

```yaml
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: master-1
  namespace: bke-cluster
  labels:
    cluster.x-k8s.io/cluster-name: bke-cluster
spec:
  hostname: master-1
  ip: 192.168.100.***
  password: '******'
  port: "22"
  role:
  - master/node
  - etcd
  username: root
  labels:
    - key: "machine-48"
      value: "true"
```