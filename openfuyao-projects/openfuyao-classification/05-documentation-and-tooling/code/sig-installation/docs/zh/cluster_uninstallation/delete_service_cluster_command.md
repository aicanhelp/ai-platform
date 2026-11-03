# 通过后台（命令行）删除业务集群

本文介绍在引导节点或管理集群控制节点上，通过编辑`BKECluster`（`bc`）卸载业务集群。通过编辑BKECluster（bc）卸载业务集群，须正确设置注解与spec.reset。

## 前提条件

- 使用具备`sudo`权限的账号操作，并具备目标集群的`kubectl`访问能力。
- 按业务集群的部署方式登录对应节点：
  - **由引导集群部署**：登录**引导节点**，kubeconfig指向引导集群。
  - **由管理集群部署**：登录**管理集群**控制节点，kubeconfig指向管理集群。
- 已阅读[集群卸载总览](./overview.md)。

## 操作步骤

1. 查询业务集群对应的`BKECluster`。

   ```bash
   kubectl get bc -A
   ```

2. 编辑目标`bc`（将占位符替换为实际命名空间与名称）。

   ```bash
   kubectl edit bc -n <命名空间> <名称>
   ```

3. 按如下方式修改（以下两项缺一不可，才能清理目标集群节点）：

   ```yaml
   metadata:
     annotations:
       # 必须为"false"才会 reset/清理目标业务集群节点；默认为"true"时只删API对象
       bke.bocloud.com/ignore-target-cluster-delete: "false"
   spec:
     reset: true   # 触发删除，等同于直接删除该BKECluster资源
   ```

   可选相关注解：

   | 注解 | 推荐（彻底卸载） | 说明 |
   |:---|:---|:---|
   | `bke.bocloud.com/deep-restore-node` | `"true"`（默认） | 深度清理节点，含容器运行时等。 |
   | `bke.bocloud.com/ignore-namespace-delete` | 按需 | `"false"`且namespace 下无其他`bc`时才会删namespace。 |

4. 保存退出，等待控制器完成卸载调谐。

   ```bash
   kubectl get bc -n <命名空间> <名称> -o yaml
   kubectl describe bc -n <命名空间> <名称>
   ```

## 删除内容说明

- 该业务集群的`BKECluster`/关联`BKENode`等对象。
- 当`ignore-target-cluster-delete: "false"`时：目标业务集群节点上的Kubernetes组件、addon，并按`deep-restore-node`决定是否深度清理运行时等。
- 当注解保持默认`"true"`时：**不会**清理目标节点，仅删除上述API对象（节点上集群仍可能残留）。

## 卸载成功示意

- `kubectl get bc -A`中已无该业务集群。

## 后续说明

- 更多注解说明见[cluster-api-provider-bke 配置参数](../cluster_installation_guide/appendix/cluster-api-provider-bke_configuration.md)。
- 其他卸载方式与删除顺序见[集群卸载总览](./overview.md)。
