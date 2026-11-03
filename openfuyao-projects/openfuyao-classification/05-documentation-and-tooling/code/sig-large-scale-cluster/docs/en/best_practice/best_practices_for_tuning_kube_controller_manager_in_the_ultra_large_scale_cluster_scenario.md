# Best Practices

## Best Practices for Tuning kube-controller-manager in the Ultra-Large-Scale Cluster Scenario

In Kubernetes, each controller is a control loop that monitors the shared state of the cluster through the API server and attempts to make changes to move the current state toward the desired state.

### Objectives

This best practice focuses on improving kube-controller-manager performance to support kube-controller-manager capabilities in large-scale scenarios, mainly including the following parts:

- Adjust kube-controller-manager resource usage to support the normal operation of large-scale clusters.
- Adjust kube-controller-manager parameters to improve cluster availability and reduce memory usage.

### Prerequisites

The cluster uses kube-controller as the control manager for the Kubernetes cluster and deploys multiple instances.

### Restrictions

Applicable to large-scale cluster environments built with the installer deployment tool.

### Background

In large-scale scenarios, kube-controller-manager needs to reconcile the state of tens of thousands of Pods and other resources. Optimization is needed to prevent kube-controller-manager performance degradation or memory overflow.

### Procedure

Follow the steps below to complete optimization by modifying the deployment configuration corresponding to kube-controller-manager.

1. Resource Usage

   - CPU: Excessive CPU usage will affect availability. If conditions allow, it is recommended not to set an upper limit.

   - Memory: If kube-controller-manager memory exceeds the limit, it will cause kube-controller-manager to restart, causing kube-controller-manager to be unable to perform normal resource reconciliation for a certain period of time, affecting the availability of the entire cluster. If conditions allow, it is recommended not to set an upper limit. If set, it can be set to 90000Mi.

2. Startup Parameters

    **Table 1** Startup Parameter Description and Recommended Values

    | Parameter Name | Description and Recommended Value |
    | :--- | :--- |
    | `leader-elect` | **Description**: Whether to enable leader election functionality.<br>**Default value**: true.<br>**Recommended configuration**: `true`. |
    | `leader-elect-lease-duration` | **Description**: Leader lease duration, leader election will start after exceeding this duration. The kube-controller-manager primary instance will periodically renew the lease to maintain Leader status. If this value is set too low, kube-controller-manager will frequently access kube-apiserver to update data in etcd; if set too high, kube-controller-manager instance switchover time will be too long.<br>**Default value**: 15s.<br>**Recommended configuration**: `65s`, to improve availability. |
    | `leader-elect-renew-deadline` | **Description**: Period for renewing lease duration, will attempt to renew Leader status after exceeding this duration.<br>**Default value**: 10s.<br>**Recommended configuration**: `60s`, to improve availability. |
    | `kube-api-burst` | **Description**: Maximum number of burst peak requests when communicating with the Kubernetes API server.<br>**Default value**: 30 (times/s).<br>**Recommended configuration**: `15000` (times/s), to meet large-scale cluster needs. |
    | `kube-api-qps` | **Description**: Queries per second (QPS) limit when communicating with the API server.<br>**Default value**: 20 (times/s).<br>**Recommended configuration**: `10000` (times/s), to meet large-scale cluster needs. |
    | `node-eviction-rate` | **Description**: Node eviction rate. Default value is 0.1, meaning at most 1 node evicts Pods every 10s.<br>**Recommended configuration**: `0.05`. |
    | `node-monitor-period` | **Description**: Period for cloud-node-lifecycle-controller to synchronize node status.<br>**Default value**: 10s.<br>**Recommended configuration**: `5s`. |
    | `node-monitor-grace-period` | **Description**: Maximum duration allowed for a Node to be unresponsive before marking it as unhealthy. This value depends on the kubelet node status update frequency.<br>**Default value**: 40s.<br>**Recommended configuration**: Keep unchanged, configure as: `40s`. |
    | `feature-gates` | **Description**: ComponentSLIs: Enables the `/metrics/slis` endpoint on Kubernetes components, allowing health check metrics to be captured, but will consume additional resources.<br>**Recommended configuration in version 1.28**: `ComponentSLIs=false`. |


3. Environment Variables

    **Table 2** Environment Variable Description and Recommended Values

    | Parameter Name | Description and Recommended Value |
    | :--- | :--- |
    | `GOGC` | **Description**: Defines the ratio of newly allocated memory to live memory that triggers garbage collection. Since kube-scheduler has low memory consumption, it is recommended to configure as 70.<br>**Recommended configuration**: `70`. |
    | `GOMEMLIMIT` | **Description**: Overall memory limit that the Go runtime can use.<br>**Recommended configuration**: `67500MiB` (85% of total memory limit). |
    | `GOMAXPROCS` | **Description**: Sets the number of runtime cores.<br>**Recommended configuration**: In multi-node scenarios, it is recommended to configure to the number of CPU cores in the environment. |
    | `HTTP2_READ_IDLE_TIMEOUT_SECONDS` | **Description**: client-go HTTP/2 connection health check timeout.<br>**Recommended configuration**: `15` (s), to ensure availability. |
    | `HTTP2_PING_TIMEOUT_SECONDS` | **Description**: client-go HTTP/2 connection health check timeout using `ping`.<br>**Recommended configuration**: `10` (s), to ensure availability. |

### References

[kube-controller-manager Parameter Introduction](https://kubernetes.io/zh-cn/docs/reference/command-line-tools-reference/kube-controller-manager/)
