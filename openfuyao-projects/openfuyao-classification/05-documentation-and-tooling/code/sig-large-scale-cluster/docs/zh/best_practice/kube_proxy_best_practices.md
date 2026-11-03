# 最佳实践

## kube-proxy最佳实践

   - kube-proxy是集群中每个节点（node）上所运行的网络代理，实现Kubernetes服务（Service）概念的一部分。
   - kube-proxy维护节点上的一些网络规则，这些网络规则会允许从集群内部或外部的网络会话与Pod进行网络通信。
   - 如果操作系统提供了可用的数据包过滤层，则kube-proxy会通过它来实现网络规则。否则，kube-proxy仅做流量转发。
   - 如果你使用网络插件为Service实现本身的数据包转发，并提供与kube-proxy等效的行为，那么你不需要在集群中的节点上运行kube-proxy。


### 目标

本最佳实践聚集在kube-proxy本身的性能提升，显著提高kube-proxy在大规模场景下的性能和稳定性，以支持大规模场景下的流量转发，主要包含如下部分。

   - 选择合适的流量转发模式：IPVS。
   - 调优模型：分析kube-proxy与apiserver间的交互及内部实现，给出相关的调优因素分析。
   - 监控指标：定期监控kube-proxy的性能指标，及时进行调整和优化。

### 前提条件

使用kube-proxy插件做为维护节点上的网络规则以实现Service的功能。

#### 使用限制

本最佳实践基于Kubernetes v1.28、v1.33、v1.34版本进行分析。

### 背景信息

kube-proxy在大规模Kubernetes集群中有如下的关键作用。

- 服务发现和负载均衡：
   kube-proxy负责将网络流量分发到适当的后端服务，确保服务请求能够负载均衡地分配到多个容器实例上。在大规模集群中，这对于保持服务的高可用性和性能至关重要。

- 高效的网络通信：
   kube-proxy实现了Kubernetes服务的虚拟IP(VIP)和负载均衡。它使用iptables或IPVS规则来管理网络流量，从而使服务之间的通信更加高效。

- 扩展性和可靠性：
   在大规模集群中，随着服务实例数量的增加，kube-proxy的负载也会增加。kube-proxy能够通过优化网络规则和负载均衡机制，确保集群能够在高负载下保持稳定和高效。

- 灵活的网络配置：
   kube-proxy支持多种模式（如iptables、IPVS和userspace），使得管理员可以根据具体场景选择最合适的网络配置，以满足不同的性能和扩展性需求。

- 容错和故障恢复：
   kube-proxy能够检测并处理服务实例的故障，自动调整网络规则，确保服务请求能够迅速切换到健康的后端实例上，从而提高集群的容错能力。

在大规模Kubernetes集群中，kube-proxy的高效运行直接影响到整个集群的性能和可靠性。通过合理配置和优化kube-proxy，可以确保集群能够在高负载和复杂网络环境下保持高效和稳定。

### 操作步骤

1. 选择合适的流量转发模式

   在Kubernetes的kube-proxy实现中，userspace、iptables和IPVS模式各有其适用场景和特点。

   1.1 userspace模式

      适用场景：

      - 小型集群：适用于测试环境或非常小规模的集群。
      - 兼容性：可能需要在某些特殊环境中使用，以确保兼容性。

      特点：

      - 性能较差：由于需要在用户空间和内核空间之间进行数据拷贝，性能不如iptables和IPVS。
      - 延迟高：处理数据包的延迟较高，不适用于生产环境。

   1.2 iptables模式

      适用场景：

      - 小到中型集群：适用于有一定规模但不是非常庞大的集群。
      - 默认选择：大多数Kubernetes集群默认使用iptables模式。

      特点：

      - 性能较好：iptables使用内核级的网络转发，性能相对较高。
      - 简单：实现相对简单，不需要额外安装复杂的软件。
      - 延迟较高：在规则非常多的情况下，可能会增加数据包处理的延迟。

   1.3 IPVS模式

      适用场景：

      - 大型集群：适用于大规模集群，特别是有大量服务和端点的场景。
      - 高性能需求：需要低延迟和高吞吐量的场景。

      特点：

      - 高性能：使用哈希表进行查找，处理速度快，延迟低。
      - 更好的扩展性：能够处理更多的规则和端点，适用于大规模场景。
      - 复杂性：相比iptables，设置和维护稍复杂，需要额外的软件支持。

   1.4 总结

      - userspace模式：适用于测试和小型集群，性能较差，主要用于兼容性目的。
      - iptables模式：适用于中小型集群，默认选择，性能较好但在大规模场景下延迟可能较高。
      - IPVS模式：适用于大规模集群和高性能需求的场景，延迟低，扩展性好，但设置较复杂。

   综上，在大规模集群场景下，kube-proxy选用IPVS是最佳选择。

2. kube-proxy启用IPVS配置(基于kubeadm)。

   2.1 执行以下代码，生成默认`kubeadm-config.yaml`。

      ```
      kubeadm config print init-defaults > kubeadm-config.yaml
      ```
      
   2.2 执行以下代码，定制`KubeProxyConfiguration`配置到`kubeadm-config.yaml`

      ```
      cat >>kubeadm-config.yaml<<EOF
      ---
      apiVersion: kubeproxy.config.k8s.io/v1alpha1
      kind: KubeProxyConfiguration
      clientConnection:
         acceptContentTypes: "application/vnd.kubernetes.protobuf"  
         contentType: "" # 为空默认为"application/vnd.kubernetes.protobuf"
         kubeconfig: ""  # 为空默认为/var/lib/kube-proxy/kubeconfig.conf
         qps: 50         # 为0.0默认为5.0
         burst: 100      # 为0默认为10
      hostnameOverride: ""  
      bindAddress: ""   # 为空默认为0.0.0.0
      bindAddressHardFail: false
      healthzBindAddress: ""
      metricsBindAddress: ""
      enableProfiling: false
      showHiddenMetricsForVersion: ""
      mode: "ipvs"
      ipvs:
         excludeCIDRs: null
         minSyncPeriod: 0s
         scheduler: ""              # 设置为空将使用默认的调度算法rr
         strictARP: false           # 不回应ARP查询
         syncPeriod: 30s            # 为0默认为30s
         tcpFinTimeout: 0s
         tcpTimeout: 0s
         udpTimeout: 0s
      detectLocalMode: "ClusterCIDR" 
      clusterCIDR: ""              # 为空会从ClusterConfiguration中赋值Networking.PodSubnet
      detectLocal:
         bridgeInterface: ""
         interfaceNamePrefix: ""
      nodePortAddresses: null
      oomScoreAdj: -999            # 为nil默认为-999
      configSyncPeriod: 0s         # 为0默认为15min
      conntrack:
         maxPerCore: 32768          # 为nil默认为32*1024
         min: 131072                # 为nil默认为128*1024
         tcpEstablishedTimeout: 24h # 为nil默认为24h
         tcpCloseWaitTimeout: 1h    # 为nil默认为1h
         # tcpBeLiberal: true       # 以宽松模式运行，提升性能:1.28版本不支持
      portRange: ""
      logging:
         flushFrequency: 0
         options:
         json:
            infoBufferSize: "0"
         verbosity: 0
      EOF
      ```

   2.3 执行以下代码，进行kubeadm安装：

    ```
    kubeadm init --config kubeadm-config.yaml
    ```

3. 调优模型

   3.1 KubeProxyConfiguration关键参数分析

   **表1** KubeProxyConfiguration关键参数

   | 参数 | 作用 | 建议值 | 备注 | 
   | - | - | - |  - | 
   | mode | 设置将使用的代理模式。 |  ipvs | - | 
   | enableProfiling | 通过`/debug/pprof`处理程序在Web界面上启用性能分析。 |  false |  - | 
   | configSyncPeriod | 从API服务器刷新配置的频率(`SharedInformerFactory`中用于设置所有`SharedIndexInformer`的默认同步时间间隔)。 |  0 | 不进行缓存数据同步。 | 
   | clientConnection | 给出代理服务器与API服务器通信时要使用的kubeconfig文件和客户端链接设置。 |  -| ClientConnectionConfiguration专用配置结构体。 | 
   | ipvs | 包含与ipvs相关的配置选项。 | - | KubeProxyIPVSConfiguration专用配置结构体。 | 
   | conntrack | 包含与conntrack相关的配置选项。 | - | KubeProxyConntrackConfiguration专用配置结构体。 | 

   3.2 ClientConnectionConfiguration

   **表2** ClientConnectionConfiguration关键参数

   | 参数 | 作用 | 建议值 | 备注 | 
   | - | - | - | - | 
   | qps | 控制此连接上每秒钟可以发送的查询请求个数。 | 50 | - | 
   | burst | 允许客户端超出其速率限制时可以临时累积的额外查询个数。 | 100 | - | 

   3.3 KubeProxyIPVSConfiguration关键参数分析

   **表3** KubeProxyIPVSConfiguration关键参数


   | 参数 | 作用 | 建议值 | 备注 | 
   | - | - | - | - | 
   | syncPeriod | ipvs规则的刷新周期。 | 30s | 影响本机性能。 | 
   | minSyncPeriod | ipvs规则被刷新的最小周期。 | 0 | 影响本机性能，频繁更新时可设置为1s。 | 
   | strictARP | 配置arp_ignore和arp_announce，以避免（错误地）响应来自kube-ipvs0接口的ARP查询请求。 | false | - | 
   | tcpTimeout | 设置空闲IPVS TCP会话的超时值。默认值为0，意味着使用系统上当前的超时值设置。 | 0 | - | 
   | tcpFinTimeout | 设置IPVS TCP会话在收到FIN之后的超时值。默认值为0，意味着使用系统上当前的超时值设置。 | 0 | - | 
   | synudpTimeout Period | 设置IPVS UDP包的超时值。默认值为0，意味着使用系统上当前的超时值设置。 | 0 | - | 

   3.4 KubeProxyConntrackConfiguration关键参数分析

      **表4** KubeProxyConntrackConfiguration关键参数


   | 参数 | 作用 | 建议值 | 备注 | 
   | - | - | - | - | 
   | min | 要分配的链接追踪记录个数下限。 | nil | - | 
   | tcpEstablishedTimeout | 出空闲TCP连接的保留时间。 | nil | - | 
   | tcpCloseWaitTimeout| 设置空闲的、处于CLOSE_WAIT状态的conntrack条目。 | nil | - | 

   > ![](./figures/icon-note.gif)**说明：**
   >
   >  上面这些需要根据场景给出合理的设置，最好能够从监控指标里给出。

4. 与apiserver关键交互分析

   kube-proxy通过List/Watch机制，主要侦听并同步以下kube-apiserver中的资源（其数量与变化频率是评估kube-proxy CPU与内存使用情况的关键指标）。
   - node
   - service
   - endpointslice


5. 监控指标配置

   定期监控kube-proxy的性能指标，及时进行调整和优化。

   导入[kubernetes-proxy grafana dashboards](https://grafana.com/grafana/dashboards/12129-kubernetes-proxy/)对kube-proxy进行定期监控。

### 结论

通过对kube-proxy支持的主要模式、命令行参数及与kube-apiserver的资源交互，给出kube-proxy在大规模集群下可行的最佳实践建议，再结合后期的运维监控及时对系统进行调整，保持整个集群的持续高性能运行与稳定。

### 参考资料

- [Kubernetes架构](https://kubernetes.io/docs/concepts/architecture/)
- [虚拟IP地址和服务代理](https://kubernetes.io/docs/reference/networking/virtual-ips/)
- [KubeProxyConfiguration](https://kubernetes.io/docs/reference/config-api/kube-proxy-config.v1alpha1/)
- [kubernetes-proxy grafana dashboards](https://grafana.com/grafana/dashboards/12129-kubernetes-proxy/)