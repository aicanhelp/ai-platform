# 使用addon安装NUMA亲和扩展组件使用指导

## 背景信息

在使用BKE安装部署工具进行集群搭建时，用户可以实现：

1. **NUMA亲和插件安装**：通过BKE的Addon机制在集群创建时直接安装NUMA亲和插件，无需依赖openFuyao应用市场。

## 前提条件

- 已完成引导节点初始化，引导集群运行正常。引导节点初始化完成后，会在`/bke/cluster/`目录下生成默认配置的yaml文件，其中`1master-cluster.yaml`用于创建1个master节点的集群，`1master-node.yaml`为对应的节点信息文件，参见[通过后台（命令行）安装业务集群](../cluster_installation_guide/service_cluster_installation/service_cluster_deployment_guide_(linux_backend).md)。
- 已准备好集群配置文件。

> ![](./figures/icon-note.gif)**说明**：
>
> 下文中`1master-cluster.yaml`和`1master-node.yaml`是引导节点`/bke/cluster/`目录下的集群配置文件，分别用于定义集群配置（含Addon配置）与节点信息等内容。

## 使用限制

本组件提供的各项功能依赖如下（参考[sig-numa-affinity组件依赖说明](https://gitcode.com/openFuyao/sig-numa-affinity/blob/master/docs/zh/user_guide/numa-aware_scheduling_user_guide.md#%E7%BB%84%E4%BB%B6%E4%BE%9D%E8%B5%96)）：

- 基于NUMA拓扑的集群级亲和调度：无硬件依赖（需多NUMA架构），无OS依赖。
- NUMA distance最优分配策略：无硬件依赖（需多NUMA架构），无OS依赖。
- POD间网络亲和感知优化：仅支持ARM架构，不支持x86架构，OS需为openEuler 22.03-sp3及以上版本，containerd >= 1.7。

## 操作步骤

1. 配置NUMA亲和插件。

   在`1master-cluster.yaml`的`addons`部分添加NUMA亲和插件配置：

   ```yaml
   spec:
     clusterConfig:
       addons:
       - name: numa-affinity-package
         param:
           openfuyaoRepo: oci://cr.openfuyao.cn/charts
         version: v0.0.2
   ```

   > ![](./figures/icon-note.gif)**说明**：
   >
   > - `openfuyaoRepo`：Chart仓库地址，请根据实际环境填写。
   > - `version`：插件版本号，请确保与集群环境兼容，如选择前置安装openFuyao组件，选择v0.0.1版本；不安装openFuyao独立安装numa亲和选择v0.0.2版本。

2. 创建集群。

   执行集群创建命令：

   ```bash
   bke cluster create -f /bke/cluster/1master-cluster.yaml -n /bke/cluster/1master-node.yaml
   ```

3. 验证安装。

   集群创建完成后，执行以下命令验证插件是否安装成功：

   ```bash
   kubectl get pod -A
   ```

## 配置示例

以下为包含NUMA亲和插件的完整配置片段：

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
    - name: numa-affinity-package
      param:
        openfuyaoRepo: oci://cr.openfuyao.cn/charts
      version: v0.0.2
```
