# Best Practices

## kube-proxy Best Practices

   - kube-proxy is a network proxy that runs on each node in the cluster, implementing part of the Kubernetes Service concept.
   - kube-proxy maintains network rules on nodes. These network rules allow network communication to Pods from network sessions inside or outside the cluster.
   - If the operating system provides an available packet filtering layer, kube-proxy uses it to implement network rules. Otherwise, kube-proxy only forwards traffic.
   - If you use a network plugin to implement packet forwarding for Services and provide behavior equivalent to kube-proxy, you don't need to run kube-proxy on cluster nodes.


### Objectives

This best practice focuses on improving kube-proxy's own performance, significantly enhancing kube-proxy's performance and stability in large-scale scenarios to support traffic forwarding in large-scale scenarios. It mainly includes the following sections:

   - Selecting the appropriate traffic forwarding mode: IPVS.
   - Tuning model: Analyze the interaction between kube-proxy and apiserver and the internal implementation, and provide analysis of related tuning factors.
   - Monitoring metrics: Regularly monitor kube-proxy's performance metrics for timely adjustments and optimization.

### Prerequisites

Using kube-proxy plugin to maintain network rules on nodes to implement Service functionality.

#### Usage Limitations

This best practice is based on analysis of Kubernetes v1.28, v1.33, and v1.34 versions.

### Background Information

kube-proxy plays the following key roles in large-scale Kubernetes clusters:

- Service Discovery and Load Balancing:
   kube-proxy is responsible for distributing network traffic to appropriate backend services, ensuring that service requests are load-balanced across multiple container instances. In large-scale clusters, this is critical for maintaining service high availability and performance.

- Efficient Network Communication:
   kube-proxy implements virtual IP (VIP) and load balancing for Kubernetes services. It uses iptables or IPVS rules to manage network traffic, making communication between services more efficient.

- Scalability and Reliability:
   In large-scale clusters, as the number of service instances increases, kube-proxy's load also increases. kube-proxy can optimize network rules and load balancing mechanisms to ensure the cluster remains stable and efficient under high loads.

- Flexible Network Configuration:
   kube-proxy supports multiple modes (such as iptables, IPVS, and userspace), allowing administrators to choose the most appropriate network configuration for specific scenarios to meet different performance and scalability requirements.

- Fault Tolerance and Failure Recovery:
   kube-proxy can detect and handle service instance failures, automatically adjust network rules, and ensure that service requests can quickly switch to healthy backend instances, thereby improving cluster fault tolerance.

In large-scale Kubernetes clusters, the efficient operation of kube-proxy directly affects the performance and reliability of the entire cluster. Through reasonable configuration and optimization of kube-proxy, you can ensure the cluster remains efficient and stable under high loads and complex network environments.

### Procedure

1. Select the Appropriate Traffic Forwarding Mode

   In the kube-proxy implementation of Kubernetes, userspace, iptables, and IPVS modes each have their applicable scenarios and characteristics.

   1.1 userspace Mode

      Applicable Scenarios:

      - Small clusters: Suitable for testing environments or very small-scale clusters.
      - Compatibility: May need to be used in certain special environments to ensure compatibility.

      Characteristics:

      - Poor performance: Due to the need for data copying between user space and kernel space, performance is inferior to iptables and IPVS.
      - High latency: Packet processing latency is high, not suitable for production environments.

   1.2 iptables Mode

      Applicable Scenarios:

      - Small to medium clusters: Suitable for clusters with a certain scale but not very large.
      - Default choice: Most Kubernetes clusters use iptables mode by default.

      Characteristics:

      - Good performance: iptables uses kernel-level network forwarding with relatively high performance.
      - Simple: Implementation is relatively simple and doesn't require complex software installation.
      - High latency: When there are many rules, packet processing latency may increase.

   1.3 IPVS Mode

      Applicable Scenarios:

      - Large clusters: Suitable for large-scale clusters, especially scenarios with many services and endpoints.
      - High performance requirements: Scenarios requiring low latency and high throughput.

      Characteristics:

      - High performance: Uses hash tables for lookups with fast processing speed and low latency.
      - Better scalability: Can handle more rules and endpoints, suitable for large-scale scenarios.
      - Complexity: Compared to iptables, setup and maintenance are slightly more complex, requiring additional software support.

   1.4 Summary

      - userspace mode: Suitable for testing and small clusters, poor performance, mainly used for compatibility purposes.
      - iptables mode: Suitable for small to medium clusters, default choice, good performance but may have high latency in large-scale scenarios.
      - IPVS mode: Suitable for large-scale clusters and high-performance requirement scenarios, low latency, good scalability, but more complex setup.

   In summary, in large-scale cluster scenarios, selecting IPVS for kube-proxy is the best choice.

2. Enabling IPVS Configuration for kube-proxy (based on kubeadm).

   2.1 Execute the following code to generate the default `kubeadm-config.yaml`.

      ```
      kubeadm config print init-defaults > kubeadm-config.yaml
      ```
      
   2.2 Execute the following code to customize `KubeProxyConfiguration` configuration in `kubeadm-config.yaml`

      ```
      cat >>kubeadm-config.yaml<<EOF
      ---
      apiVersion: kubeproxy.config.k8s.io/v1alpha1
      kind: KubeProxyConfiguration
      clientConnection:
         acceptContentTypes: "application/vnd.kubernetes.protobuf"  
         contentType: "" # Empty defaults to "application/vnd.kubernetes.protobuf"
         kubeconfig: ""  # Empty defaults to /var/lib/kube-proxy/kubeconfig.conf
         qps: 50         # 0.0 defaults to 5.0
         burst: 100      # 0 defaults to 10
      hostnameOverride: ""  
      bindAddress: ""   # Empty defaults to 0.0.0.0
      bindAddressHardFail: false
      healthzBindAddress: ""
      metricsBindAddress: ""
      enableProfiling: false
      showHiddenMetricsForVersion: ""
      mode: "ipvs"
      ipvs:
         excludeCIDRs: null
         minSyncPeriod: 0s
         scheduler: ""              # Empty will use default scheduling algorithm rr
         strictARP: false           # Do not respond to ARP queries
         syncPeriod: 30s            # 0 defaults to 30s
         tcpFinTimeout: 0s
         tcpTimeout: 0s
         udpTimeout: 0s
      detectLocalMode: "ClusterCIDR" 
      clusterCIDR: ""              # Empty will be assigned from ClusterConfiguration Networking.PodSubnet
      detectLocal:
         bridgeInterface: ""
         interfaceNamePrefix: ""
      nodePortAddresses: null
      oomScoreAdj: -999            # nil defaults to -999
      configSyncPeriod: 0s         # 0 defaults to 15min
      conntrack:
         maxPerCore: 32768          # nil defaults to 32*1024
         min: 131072                # nil defaults to 128*1024
         tcpEstablishedTimeout: 24h # nil defaults to 24h
         tcpCloseWaitTimeout: 1h    # nil defaults to 1h
         # tcpBeLiberal: true       # Run in liberal mode to improve performance: 1.28 version doesn't support
      portRange: ""
      logging:
         flushFrequency: 0
         options:
         json:
            infoBufferSize: "0"
         verbosity: 0
      EOF
      ```

   2.3 Execute the following code for kubeadm installation:

    ```
    kubeadm init --config kubeadm-config.yaml
    ```

3. Tuning Model

   3.1 KubeProxyConfiguration Key Parameter Analysis

   **Table 1** KubeProxyConfiguration Key Parameters

   | Parameter | Function | Recommended Value | Note | 
   | - | - | - |  - | 
   | mode | Sets the proxy mode to use. |  ipvs | - | 
   | enableProfiling | Enables profiling in the web interface through `/debug/pprof` handler. |  false |  - | 
   | configSyncPeriod | Frequency of configuration refresh from API server (used to set default sync interval for all `SharedIndexInformer` in `SharedInformerFactory`). |  0 | No cache data synchronization. | 
   | clientConnection | Provides kubeconfig file and client connection settings to use when proxy server communicates with API server. |  -| ClientConnectionConfiguration specific configuration structure. | 
   | ipvs | Contains configuration options related to ipvs. | - | KubeProxyIPVSConfiguration specific configuration structure. | 
   | conntrack | Contains configuration options related to conntrack. | - | KubeProxyConntrackConfiguration specific configuration structure. | 

   3.2 ClientConnectionConfiguration

   **Table 2** ClientConnectionConfiguration Key Parameters

   | Parameter | Function | Recommended Value | Note | 
   | - | - | - | - | 
   | qps | Controls the number of query requests per second that can be sent on this connection. | 50 | - | 
   | burst | Allows the client to temporarily accumulate additional queries when exceeding the rate limit. | 100 | - | 

   3.3 KubeProxyIPVSConfiguration Key Parameter Analysis

   **Table 3** KubeProxyIPVSConfiguration Key Parameters


   | Parameter | Function | Recommended Value | Note | 
   | - | - | - | - | 
   | syncPeriod | Refresh cycle for ipvs rules. | 30s | Affects local machine performance. | 
   | minSyncPeriod | Minimum refresh cycle for ipvs rules. | 0 | Affects local machine performance, can be set to 1s when updating frequently. | 
   | strictARP | Configures arp_ignore and arp_announce to avoid (incorrectly) responding to ARP query requests from kube-ipvs0 interface. | false | - | 
   | tcpTimeout | Sets timeout value for idle IPVS TCP sessions. Default value 0 means use current timeout value setting on the system. | 0 | - | 
   | tcpFinTimeout | Sets timeout value for IPVS TCP session after receiving FIN. Default value 0 means use current timeout value setting on the system. | 0 | - | 
   | synudpTimeout Period | Sets timeout value for IPVS UDP packets. Default value 0 means use current timeout value setting on the system. | 0 | - | 

   3.4 KubeProxyConntrackConfiguration Key Parameter Analysis

      **Table 4** KubeProxyConntrackConfiguration Key Parameters


   | Parameter | Function | Recommended Value | Note | 
   | - | - | - | - | 
   | min | Lower limit of connection tracking records to allocate. | nil | - | 
   | tcpEstablishedTimeout | Retention time for idle TCP connections. | nil | - | 
   | tcpCloseWaitTimeout| Sets idle conntrack entries in CLOSE_WAIT state. | nil | - | 

   > ![](./figures/icon-note.gif)**Note:**
   >
   >  These need to be set reasonably according to scenarios, preferably from monitoring metrics.

4. Key Interaction Analysis with apiserver

   kube-proxy uses List/Watch mechanism, mainly listening and synchronizing the following resources in kube-apiserver (their quantity and change frequency are key metrics for evaluating kube-proxy CPU and memory usage).
   - node
   - service
   - endpointslice


5. Monitoring Metrics Configuration

   Regularly monitor kube-proxy's performance metrics for timely adjustments and optimization.

   Import [kubernetes-proxy grafana dashboards](https://grafana.com/grafana/dashboards/12129-kubernetes-proxy/) for regular monitoring of kube-proxy.

### Conclusion

Through analysis of the main modes supported by kube-proxy, command-line parameters, and resource interaction with kube-apiserver, feasible best practice recommendations for kube-proxy in large-scale clusters are provided. Combined with later operation and maintenance monitoring for timely system adjustments, maintain continuous high-performance operation and stability of the entire cluster.

### References

- [Kubernetes Architecture](https://kubernetes.io/docs/concepts/architecture/)
- [Virtual IPs and Service Proxies](https://kubernetes.io/docs/reference/networking/virtual-ips/)
- [KubeProxyConfiguration](https://kubernetes.io/docs/reference/config-api/kube-proxy-config.v1alpha1/)
- [kubernetes-proxy grafana dashboards](https://grafana.com/grafana/dashboards/12129-kubernetes-proxy/)