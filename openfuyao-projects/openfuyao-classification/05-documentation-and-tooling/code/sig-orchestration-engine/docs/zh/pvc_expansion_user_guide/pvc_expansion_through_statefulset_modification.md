# 修改statefulset实现PVC扩容

## 特性介绍

当产品管理规模扩大时，需要调整业务实例、资源配额以及存储资源，但statefulset的PVC模板大小不支持修改，需要在扩容时手动调整PVC大小，需要手动操作，比较耗时；本特性旨在支持通过修改statefulset的PVC模板大小自动扩容PVC。

### 应用场景

产品管理规模扩大时，需要调整业务实例、资源配额以及存储资源时，通过修改statefulset模板实现扩容。

### 能力范围

集群中已存在配置有PVC的statefulset类型资源并开启sc特性开关。

### 亮点特征

实现通过修改statefulset的PVC模板大小自动扩容PVC。

### 实现原理

- 增加FeatureGate：HWStatefulSetVolumeExpansion，涉及kube-apiserver和kube-controller-manager，用于启用修改statefulset进行PVC扩容的能力。

- 现有特性已支持通过修改PVC扩容来存储资源，当前仅需要侦听statefulset controller中声明的PVC大小变更，动态patch修改PVC的storage大小，即可实现存储资源的动态扩容。

- statefulset controller增加侦听PVC的更新事件，来更新statefulset的状态。

- 在statefulset中增加扩展的字段来表示volumeClaimTemplates修改的状态。

  ```yaml
    status: 
      availableReplicas: 3 
      currentReplicas: 3 
      currentRevision: ex1-54c5bd476c 
      readyReplicas: 3 
      replicas: 3 
      updatedReplicas: 3 
      volumeClaimTemplates: 
      - finishedReconciliationGeneration: 2 
        readyReplicas: 1 
        templateName: vol1
  ```
  
### 与相关特性的关系

无。

## 安装

无需安装。

## 使用PVC扩容

### 前提条件

无。

### 背景信息

无。

### 使用限制

无。

### 操作步骤

使用本特性需要开启Local CSI的allowVolumeExpansion功能，通过修改PVC定义实现自动扩容。

1. 检查特性是否开启：

    ```shell
    kubectl get sc -A
    ```

    1.1 查看Local CSI的allowVolumeExpansion值是否为true（开启）。
  
    1.2 allowVolumeExpansion若为false，执行以下命令配置开启：

    ```shell
    kubectl edit sc <sc_name>
    ```

2. 查询所有statefulset：

    ```shell
    kubectl get sts -A
    ```

3. 修改目标statefulset，修改PVC大小配置：

    ```shell
    kubectl edit sts -n <namespace> <sts_name>
    ```

找到其中定义PVC的部分，修改storage大小即可，对应部分格式如下：

```yaml
  volumeClaimTemplates:
  - apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      # ...
    spec:
      # ...
      resource:
        requests:
          storage: 30Gi  # 修改该值即可
      # ...
    # ...
```
