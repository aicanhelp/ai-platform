# Best Practices
## Best Practices for Tuning kubelet in the Ultra-Large-Scale Cluster Scenario

Kubelet is a core component running on each node in a Kubernetes cluster, and it is responsible for the following main functions:

- Manage Pod lifecycle: kubelet is responsible for starting, stopping, and maintaining all container applications on the node. It periodically communicates with the apiserver to obtain and execute Pod objects scheduled on that node.
- Health checks and self-healing: kubelet periodically checks the running status of containers. If it finds a container in an unhealthy state or crashed, kubelet will restart the container to ensure application high availability.
- Resource management: kubelet is responsible for managing resources on the node, including allocation and monitoring of resources such as CPU, memory, network, and storage. It allocates node resources based on Pod resource requests and limits.
- Logging and monitoring: kubelet collects and reports node and container operation logs and performance metrics. This information can help administrators monitor cluster health and performance.
- Plugin system: kubelet supports multiple plugin systems, such as CNI (Container Network Interface) for network management and CSI (Container Storage Interface) for storage management. These plugins extend kubelet's functionality, enabling it to integrate with various third-party tools and platforms.

These functions make kubelet an indispensable component of a Kubernetes cluster, ensuring the stable operation of containerized applications.

### Objectives
This best practice aims to improve kubelet performance to support kubelet capabilities in large-scale scenarios. It mainly includes the following parts:
- Kubelet node status report frequency parameter tuning.
- Kubelet and kube-apiserver interaction message frequency parameter tuning.
- Enable kubelet parallel image pull functionality.

### Prerequisites
Ensure the cluster meets the following requirements:

- Node agent: kubelet is deployed and running on all nodes.
- Core responsibility: kubelet is responsible for managing the container lifecycle of Pods on nodes.
- Communication link: kubelet must be able to communicate with control plane nodes.

### Restrictions
This best practice is based on analysis of Kubernetes versions 1.28, 1.33, and 1.34.

### Background
Kubelet is responsible for synchronizing nodeState information and other interaction tasks with kube-apiserver. In large-scale Kubernetes clusters, the message volume grows exponentially, causing tremendous processing pressure on the cluster, especially on kube-apiserver.

- **Node Status Reporting**
  
  Kubelet periodically reports node status information to the apiserver in the Kubernetes control plane to ensure the overall health and performance of the cluster. Node information includes node resource usage, node health status, node conditions, Pod status, etc. This status information is reported through kubelet's heartbeat mechanism. Kubelet periodically sends heartbeat signals to the apiserver with the node's latest status data. In large-scale cluster scenarios, it is necessary to reasonably configure the reporting time interval.

- **Kubelet and kube-apiserver Interaction Message Frequency Parameter Tuning**
  
  Kubelet communicates with kube-apiserver through HTTP/HTTPS protocol. Main interaction content includes pulling Pod configuration manifests, reporting node and Pod status, synchronizing events, etc. This communication mechanism ensures coordination and consistency of the overall cluster state, ensuring nodes and containers can run as expected and respond to changes. In large-scale cluster environments, it is necessary to reasonably tune related reporting rates and frequencies to balance communication load and information timeliness.

- **Enable kubelet Parallel Image Pull Functionality**
  
  In large-scale clusters, as the number of service instances increases, the load on kubelet to pull images also increases. Kubelet's image pulling performance directly affects the overall performance of the cluster. In large-scale cluster scenarios, it is necessary to reasonably configure the concurrency and rate of image pulling.

### Procedure
1. Execute the following command to generate the default configuration file `KubeletConfiguration.yaml`.
    ```
    kubeadm config print init-defaults --component-configs KubeletConfiguration
    ```
2. Edit the configuration file
   
   Open the generated `KubeletConfiguration.yaml`, find the `KubeletConfiguration` configuration block, copy its content to a new `kubelet-config.yaml` file, and adjust according to cluster scale and requirements.

3. On the basis of the original content, adjust some parameters. The final `kubelet-config.yaml` content example is as follows:
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
4. Navigate to the path `/var/lib/kubelet`, edit the config.yaml file under the path, copy the content of the kubelet-config.yaml file modified in the previous step to config.yaml. Execute the following command to restart kubelet to make the new configuration file take effect:
    ```
    sudo systemctl daemon-reload          # Reload systemd configuration
    sudo systemctl restart kubelet        # Restart kubelet service
    ```

### Tuning Model
1. KubeletConfiguration Key Parameter Analysis

    **Table 1** KubeletConfiguration Key Parameters

    | Parameter | Function | Recommended Value | Note | 
    | - | - | - |  - | 
    | `nodeStatusUpdateFrequency` | Frequency at which kubelet computes node status. |  10s | - | 
    | `nodeStatusReportFrequency` | Frequency at which kubelet updates node status to the control plane when node status has not changed. |  5m | If the node status changes, kubelet ignores this frequency setting and immediately updates the node status. | 
    | `nodeLeaseDurationSeconds` | Duration value that kubelet sets on its corresponding Lease object. |  240 | If the lease expires, the node can be considered unhealthy. According to the KEP-0009 convention, the current lease is renewed every 10 seconds, with a default value of 40s. | 
    | `eventBurst` | Upper limit for burst event creation, allowing event creation to temporarily rise to the specified number. |  100 | - | 
    | `eventRecordQPS` | Sets the upper limit for the number of events that can be created per second. | 50 | - | 
    | `kubeAPIBurst` | Sets the burst traffic level when communicating with the Kubernetes API server. | 100 | - | 
    | `kubeAPIQPS` | Sets the QPS traffic level when communicating with the Kubernetes API server. | 50 | - | 

2. KubeletConfiguration Key Parameter Description 

    **Table 2** KubeletConfiguration Key Parameter Description

    | Feature Dimension | Early Mechanism  | New Mechanism (NodeLease feature, 1.12+) |
    | :--- | :--- | :--- |
    | Core Mechanism | All nodes report uniformly and completely. | Split reporting information into node status and lightweight Lease objects. |
    | Reporting Frequency | Only NodeStatusUpdateFrequency, fixed 10s cycle. | - Node status: on change or NodeStatusReportFrequency (default 1 minute).<br>- Lease object: NodeStatusUpdateFrequency (default 10s). |
    | Reporting Content | Complete status (status information + resource information, etc.), large data packets. | - Node status: complete status information.<br>- Lease object: only contains most basic node information (lightweight). |
    | Pressure on Control Plane | High: Frequent complete status updates lead to:<br>1. Node controller constantly processing changes, high overhead.<br>2. In extreme scenarios, etcd may quickly reach storage limit. | Significantly reduced:<br>1. Lease object changes are lightweight, low pressure on controller and etcd.<br>2. Complete status update frequency is reduced (up to 1 minute). |
    | Pressure on Cloud Platform | High: Each report requires calling cloud provider API to get node IP and other information. | Reduced: Only needs to call cloud API when updating node status (reduced frequency). |
    | Health Determination Mechanism | Depends on complete node status update cycle. | Depends on NodeLeaseDurationSeconds (default 40s): if Lease object is not updated within this duration, the node can be determined as unhealthy. |
    | Design Goal | Unified reporting, simple and direct. | Decoupling and optimization: separating high-frequency liveness heartbeat (Lease) from low-frequency complete status synchronization to support large-scale clusters. |

### Conclusion
This best practice systematically improves kubelet's processing performance and stability in large-scale cluster scenarios through targeted tuning of three key dimensions: kubelet node status report frequency, interaction message frequency with kube-apiserver, and parallel image pull functionality, providing reliable guarantees for node management in high-load environments.

### References
[kubelet Official Documentation](https://kubernetes.io/docs/reference/command-line-tools-reference/kubelet/)
