# 最佳实践
> 最佳实践是经过验证和广泛应用的最佳实践，旨在帮助用户在特定场景下实现最佳结果。

## 大规模集群场景下kubelet参数调优最佳实践
kubelet 功能概述： </br>
kubelet 是 Kubernetes 集群中运行在每个节点上的核心组件。它负责以下几个主要功能： </br>
- 管理 Pod 的生命周期：kubelet 负责启动、停止和维护节点上的所有容器应用。它会定期通过与 apiserver 通信来获取并执行调度在该节点上的 Pod 对象。 </br>
- 健康检查和自愈：kubelet 会定期检查容器的运行状况。如果发现某个容器处于不健康状态或崩溃，kubelet 会重新启动该容器以确保应用的高可用性。 </br>
- 资源管理：kubelet 负责节点上的资源管理，包括 CPU、内存、网络和存储等资源的分配和监控。它会根据 Pod 的资源请求和限制来分配节点资源。 </br>
- 日志和监控：kubelet 会收集并报告节点和容器的运行日志和性能指标，这些信息可以帮助管理员监控集群的健康状况和性能。 </br>
- 插件系统：kubelet 支持多种插件系统，如 CNI (Container Network Interface) 用于网络管理，CSI (Container Storage Interface) 用于存储管理等。这些插件扩展了 kubelet 的功能，使其能够与多种第三方工具和平台集成。 </br>

这些功能使 kubelet 成为 Kubernetes 集群中不可或缺的组成部分，保障了容器化应用的稳定运行。

### 目标
本最佳实践聚集在 kubelet 的性能提升，以支持大规模场景下的kubelet能力。主要包含如下部分：
- kubelet节点状态报告频率参数调优
- kubelet与API server交互消息频率参数调优
- kubelet并行拉取镜像功能开启

### 前提条件(可选)
集群使用 kubelet 管理节点容器和与master节点的通信。

### 使用限制(可选)
本最佳实践基于 kubernetes 1.28 版本进行分析。

### 背景信息(可选)
kubelet肩负着与API server同步nodeState信息等交互任务，在大规模 Kubernetes 集群中，消息量倍数级增长，对集群尤其是API server造成极大的处理压力。

1. 节点状态上报 </br>
kubelet 会定期将节点的状态信息上报给 Kubernetes 控制平面中的 apiserver，以确保集群的整体健康状态和性能。节点信息包括节点资源使用情况、节点健康状况、节点条件、Pod状态等。这些状态信息是通过 kubelet 的心跳机制来上报的，kubelet 会定期向 apiserver 发送心跳信号，并附带节点的最新状态数据。在大规模集群场景下，需要合理配置上报时间间隔。 </br>

2. kubelet与API server交互消息频率参数调优 </br>
kubelet 通过 HTTP/HTTPS 协议与 API server 进行通信，主要包括获取 Pod 信息、上报Pod信息、事件通知等。通过这些交互，kubelet 和 API server 可以保持集群的整体协调和一致，确保各个节点和容器能够正常运行和响应变化。在大规模集群场景下，需要合理配置这些消息上报速率。 </br>

3. kubelet并行拉取镜像功能开启 </br>
在大规模集群中，随着服务实例数量的增加，kubelet 拉取镜像的负载也会增加。kubelet 拉取镜像的性能直接影响集群的整体性能。在大规模集群场景下，需要合理配置拉取镜像的并发数和速率。 </br>

### 操作步骤(可选)
1. 生成默认KubeletConfiguration.yaml
```
kubeadm config print init-defaults --component-configs KubeletConfiguration
```
2. 定制 KubeletConfiguration 配置到 kubelet-config.yaml
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
### 调优模型
#### KubeletConfiguration关键参数分析
| 参数 | 作用 | 建议值 | 备注 | 
| - | - | - |  - | 
| nodeStatusUpdateFrequency | kubelet 计算节点状态的频率 |  10s | - | 
| nodeStatusReportFrequency | 节点状态未发生变化时，kubelet 向控制面更新节点状态的频率 |  5m | 如果节点状态发生变化，则 kubelet 会忽略这一频率设置， 立即更新节点状态。 | 
| nodeLeaseDurationSeconds | kubelet 会在其对应的 Lease 对象上设置的时长值 |  240 | 如果租约过期，则节点可被视作不健康。根据 KEP-0009 约定，目前的租约每 10 秒钟续约一次。 默认值为40s | 
| eventBurst | 突发性事件创建的上限值，允许事件创建临时上升到所指定数量 |  100 |  | 
| eventRecordQPS | 设置每秒钟可创建的事件个数上限 | 50 |  | 
| kubeAPIBurst | 设置与 Kubernetes API 服务器通信时突发的流量级别 | 100 |  | 
| kubeAPIQPS | 设置与 Kubernetes API 服务器通信时突发的流量级别 | 50 |  | 

#### KubeletConfiguration关键参数说明
早期版本只有 NodeStatusUpdateFrequency ，默认配置下所有节点每隔 10s 上报一次状态，而上报的信息包含状态信息和资源信息，因此需要传输的数据包较大。随着集群规模的增长，状态的频繁更新对控制平面组件造成较大压力：与节点相关的控制器会不断接收节点变更通知，从而增加控制器开销；极端场景中，它甚至会使etcd 迅速到达其存储上限；节点IP 地址等要上报的信息需要从云提供商的 API 获取，因此频繁的调用对底层云平台也造成较大压力。 </br>
自 1.12 版本起， Kubernetes 引入了 NodeLease 特性：将上报信息划分为更新表 2-6 中罗列的节点状态和Lease 对象。 Kubernetes 为每个节点创建一个轻量级的 Lease 对象，该对象只包含最基本的节点信息。它的频繁变更对系统造成的压力，会比直接更新节点对象小很多。
kubelet 在节点状态发生变更或者默认一分钟的 NodeStatusReportFrequency 时钟周期到达时，更新节点的状态信息，同时以默认10s 的 NodeStatusUpdateFrequency 周期更新 Lease 对象。在默认 40s 的 NodeLeaseDurationSeconds 周期内，若 Lease 对象没有被更新，则对应节点可以被判定为不健康。 </br>

### 后续步骤(可选)
### 示例/案例研究(可选)
> 展示这个最佳实践的应用和效果。
### 注意事项/常见问题(可选)
> 列出可能遇到的注意事项或常见问题，并提供相应的解决方案或建议。
### 结论
> 总结这个最佳实践的重要性和价值，并强调它的潜在影响或好处。
### 参考资料
> 列出参考或引用的任何相关资料、链接或资源。
# ref
