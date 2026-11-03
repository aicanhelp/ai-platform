# 最佳实践
## 大规模集群场景下kubelet参数调优最佳实践

kubelet是Kubernetes集群中运行在每个节点上的核心组件，它负责以下几个主要功能。

- 管理Pod的生命周期：kubelet负责启动、停止和维护节点上的所有容器应用。它会定期通过与apiserver通信来获取并执行调度在该节点上的Pod对象。
- 健康检查和自愈：kubelet会定期检查容器的运行状况。如果发现某个容器处于不健康状态或崩溃，kubelet会重新启动该容器以确保应用的高可用性。
- 资源管理：kubelet负责节点上的资源管理，包括CPU、内存、网络和存储等资源的分配和监控。它会根据Pod的资源请求和限制来分配节点资源。
- 日志和监控：kubelet会收集并报告节点和容器的运行日志和性能指标，这些信息可以帮助管理员监控集群的健康状况和性能。
- 插件系统：kubelet支持多种插件系统，如CNI(Container Network Interface) 用于网络管理，CSI(Container Storage Interface) 用于存储管理等。这些插件扩展了kubelet的功能，使其能够与多种第三方工具和平台集成。

这些功能使kubelet成为Kubernetes集群中不可或缺的组成部分，保障了容器化应用的稳定运行。

### 目标
本最佳实旨在提升kubelet的性能，以支持大规模场景下的kubelet能力。主要包含如下部分。
- kubelet节点状态报告频率参数调优。
- kubelet与kube-apiserver交互消息频率参数调优。
- kubelet并行拉取镜像功能开启。

### 前提条件
确保集群满足以下要求。

- 节点代理：所有节点均已部署并运行kubelet。
- 核心职责：kubelet负责管理节点上Pod的容器生命周期。
- 通信链路：kubelet需能够与控制平面节点进行通信。

### 使用限制
本最佳实践基于Kubernetes 1.28、1.33、1.34版本进行分析。

### 背景信息
kubelet负责与kube-apiserver同步nodeState信息等交互任务，在大规模Kubernetes集群中，消息量倍数级增长，对集群尤其是kube-apiserver造成极大的处理压力。

- **节点状态上报**
  
  kubelet会定期将节点的状态信息上报给Kubernetes控制平面中的apiserver，以确保集群的整体健康状态和性能。节点信息包括节点资源使用情况、节点健康状况、节点条件、Pod状态等。这些状态信息是通过kubelet的心跳机制来上报的，kubelet会定期向apiserver发送心跳信号，并附带节点的最新状态数据。在大规模集群场景下，需要合理配置上报时间间隔。

- **kubelet与kube-apiserver交互消息频率参数调优**
  
  kubelet通过HTTP/HTTPS协议与kube-apiserver进行通信，主要交互内容包括，拉取Pod配置清单、上报节点及Pod状态、同步事件等。该通信机制确保了集群整体状态的协调一致，保障节点与容器能够按预期运行并响应变更。在大规模集群环境下，需合理调优相关上报速率与频率，以平衡通信负载与信息时效性。

- **kubelet并行拉取镜像功能开启**
  
  在大规模集群中，随着服务实例数量的增加，kubelet拉取镜像的负载也会增加。kubelet拉取镜像的性能直接影响集群的整体性能。在大规模集群场景下，需要合理配置拉取镜像的并发数和速率。

### 操作步骤
1. 执行以下命令，生成默认配置文件`KubeletConfiguration.yaml`。
    ```
    kubeadm config print init-defaults --component-configs KubeletConfiguration
    ```
2. 编辑配置文件
   
   打开生成的`KubeletConfiguration.yaml`，找到`KubeletConfiguration`配置块，将其内容复制到新的`kubelet-config.yaml`文件中，并根据集群规模与需求进行调整。

3. 在原有内容的基础上，调整部分参数，最终的`kubelet-config.yaml`内容示例如下。
    ```
    apiVersion: kubelet.config.k8s.io/v1beta1
    authentication:
      anonymous:
        enabled: false
      webhook:
        cacheTTL: 0s
        enabled: true
      x509:
        clientCAFile: /etc/kubernetes/pki/ca.crt
    authorization:
      mode: Webhook
      webhook:
        cacheAuthorizedTTL: 0s
        cacheUnauthorizedTTL: 0s
    cgroupDriver: systemd
    clusterDNS:
    - 10.96.0.10
    clusterDomain: cluster.local
    containerRuntimeEndpoint: ""
    cpuManagerReconcilePeriod: 0s
    evictionPressureTransitionPeriod: 0s
    fileCheckFrequency: 0s
    healthzBindAddress: 127.0.0.1
    healthzPort: 10248
    httpCheckFrequency: 0s
    imageMinimumGCAge: 0s
    kind: KubeletConfiguration
    logging:
      flushFrequency: 0
      options:
        json:
          infoBufferSize: "0"
      verbosity: 0
    memorySwap: {}
    nodeStatusReportFrequency: 5m0s
    nodeStatusUpdateFrequency: 10s
    rotateCertificates: true
    runtimeRequestTimeout: 0s
    shutdownGracePeriod: 0s
    shutdownGracePeriodCriticalPods: 0s
    staticPodPath: /etc/kubernetes/manifests
    streamingConnectionIdleTimeout: 0s
    syncFrequency: 0s
    volumeStatsAggPeriod: 0s
    eventBurst: 100
    eventRecordQPS: 50
    kubeAPIBurst: 100
    kubeAPIQPS: 50
    serializeImagePulls: false
    maxParallelImagePulls: 10
    nodeLeaseDurationSeconds: 240
    ```
4. 进入路径`/var/lib/kubelet`，编辑路径下文件config.yaml，将上一步修改完成的kubelet-config.yaml文件内容复制到config.yaml中。执行以下命令，重启kubelet，使新的配置文件生效。
    ```
    sudo systemctl daemon-reload          # 重新加载systemd配置
    sudo systemctl restart kubelet        # 重启kubelet服务
    ```

### 调优模型
1. KubeletConfiguration关键参数分析

    **表1** KubeletConfiguration关键参数

    | 参数 | 作用 | 建议值 | 备注 | 
    | - | - | - |  - | 
    | `nodeStatusUpdateFrequency` | kubelet计算节点状态的频率。 |  10s | - | 
    | `nodeStatusReportFrequency` | 节点状态未发生变化时，kubelet向控制面更新节点状态的频率。 |  5m | 如果节点状态发生变化，则kubelet会忽略这一频率设置， 立即更新节点状态。 | 
    | `nodeLeaseDurationSeconds` | kubelet会在其对应的Lease对象上设置的时长值。 |  240 | 如果租约过期，则节点可被视作不健康。根据KEP-0009约定，目前的租约每10秒钟续约一次，默认值为40s。 | 
    | `eventBurst` | 突发性事件创建的上限值，允许事件创建临时上升到所指定数量。 |  100 | - | 
    | `eventRecordQPS` | 设置每秒钟可创建的事件个数上限。 | 50 | - | 
    | `kubeAPIBurst` | 设置与Kubernetes API服务器通信时突发的流量级别。 | 100 | - | 
    | `kubeAPIQPS` | 设置与Kubernetes API服务器通信时突发的流量级别。 | 50 | - | 

2. KubeletConfiguration关键参数说明 

    **表2** KubeletConfiguration关键参数说明

    | 特性维度 | 早期机制  | 新机制(NodeLease特性，1.12+) |
    | :--- | :--- | :--- |
    | 核心机制 | 所有节点统一、完整上报。 | 将上报信息拆分为节点状态和轻量级Lease对象。 |
    | 上报频率 | 仅NodeStatusUpdateFrequency，固定10s周期。 | - 节点状态：变更时或NodeStatusReportFrequency(默认1分钟)。<br>- Lease对象：NodeStatusUpdateFrequency (默认10s)。 |
    | 上报内容 | 完整状态（状态信息 + 资源信息等），数据包大。 | - 节点状态：完整状态信息。<br>- Lease对象：仅包含最基本节点信息（轻量）。 |
    | 对控制平面压力 | 大：频繁的完整状态更新导致：<br>1. 节点控制器不断处理变更，开销大。<br>2. 极端场景下etcd可能迅速达到存储上限。 | 显著降低：<br>1. Lease对象变更轻量，对控制器和etcd压力小。<br>2. 完整状态更新频率降低（最长1分钟）。 |
    | 对云平台压力 | 大：每次上报都需调用云提供商API获取节点IP等信息。 | 降低：仅在更新节点状态时（频率降低）需要调用云API。 |
    | 健康判定机制 | 依赖完整的节点状态更新周期。 | 依赖NodeLeaseDurationSeconds(默认40s)：在此时长内若Lease对象未更新，则节点可被判定为不健康。 |
    | 设计目标 | 统一上报，简单直接。 | 解耦与优化：将高频的存活心跳（Lease）与低频的完整状态同步分离，以支持大规模集群。 |

### 结论
本最佳实践通过对kubelet节点状态报告频率、与kube-apiserver交互消息频率以及并行拉取镜像功能三个关键维度的针对性调优，系统性地提升了kubelet在大规模集群场景下的处理性能与稳定性，为高负载环境下的节点管理提供了可靠保障。

### 参考资料
[kubelet官方文档](https://kubernetes.io/docs/reference/command-line-tools-reference/kubelet/)