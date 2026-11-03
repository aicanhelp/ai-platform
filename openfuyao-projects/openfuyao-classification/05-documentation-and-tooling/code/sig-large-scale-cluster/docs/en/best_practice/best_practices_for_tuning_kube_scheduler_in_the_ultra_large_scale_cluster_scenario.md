# Best Practices

## Best Practices for Tuning kube-scheduler in the Ultra-Large-Scale Cluster Scenario

The Kubernetes scheduler is a core component of the control plane, responsible for assigning Pods to cluster nodes. It retrieves Pods from the scheduling queue, first filters all schedulable legal nodes based on resource requirements and constraint conditions, then ranks these nodes according to priority policies, and finally binds the Pod to the optimal node. Kubernetes supports running multiple different schedulers simultaneously.

### Objectives

This best practice aims to improve kube-scheduler performance to support kube-scheduler capabilities in large-scale scenarios, mainly including the following parts:

- Adjust kube-scheduler resource usage to support the normal operation of large-scale clusters.
- Adjust kube-scheduler parameters to improve cluster availability and reduce memory usage.

### Prerequisites

The cluster uses kube-scheduler as the scheduler for the Kubernetes cluster and deploys multiple instances.

### Restrictions

Applicable to large-scale cluster environments built with the installer deployment tool.

### Background

In large-scale scenarios, kube-scheduler needs to schedule tens of thousands of Pods on thousands of nodes. Optimization is needed to prevent kube-scheduler performance degradation or memory overflow.

### Procedure
Follow the steps below to complete optimization by modifying the deployment configuration corresponding to kube-scheduler.

1. Resource Usage

   - CPU: Excessive CPU usage will affect availability. If conditions allow, it is recommended not to set an upper limit.

   - Memory: If kube-scheduler memory exceeds the limit, it will cause kube-scheduler to restart, causing kube-scheduler to be unable to schedule normally for a certain period of time, affecting availability. If conditions allow, it is recommended not to set an upper limit. If set, it can be set to 90000Mi.

2. Startup Parameters

    **Table 1** Startup Parameter Description and Recommended Values

    | Parameter Name | Description and Recommended Value |
    | :--- | :--- |
    | `leader-elect` | **Description**: Whether to enable leader election functionality.<br>**Default value**: true.<br>**Recommended configuration**: `true`. |
    | `leader-elect-lease-duration` | **Description**: Leader lease duration, leader election will start after exceeding this duration. The kube-scheduler primary instance will periodically renew the lease to maintain Leader status. If this value is set too low, kube-scheduler will frequently access kube-apiserver to update data in etcd; if set too high, kube-scheduler instance switchover time will be too long.<br>**Default value**: 15s.<br>**Recommended configuration**: `65s`, to improve availability. |
    | `leader-elect-renew-deadline` | **Description**: Period for renewing lease duration, will attempt to renew Leader status after exceeding this duration. The kube-scheduler primary instance will periodically renew the lease to maintain Leader status. If this value is set too low, kube-scheduler will frequently access kube-apiserver to update data in etcd.<br>**Default value**: 10s.<br>**Recommended configuration**: `60s`, to improve availability. |
    | `kube-api-burst` | **Description**: Maximum number of burst peak requests when communicating with the Kubernetes API server.<br>**Default value**: 100 (times/s).<br>**Recommended configuration**: `10000` (times/s), to meet large-scale cluster needs. |
    | `kube-api-qps` | **Description**: Queries per second (QPS) limit when communicating with the API server.<br>**Default value**: 50 (times/s).<br>**Recommended configuration**: `5000` (times/s), to meet large-scale cluster needs. |
    | `feature-gates` | **Description**: ComponentSLIs: Enables the `/metrics/slis` endpoint on Kubernetes components, allowing health check metrics to be captured, but will consume additional resources.<br>**Recommended configuration in version 1.28**: `ComponentSLIs=false`. |



3. Environment Variables

    **Table 2** Environment Variable Description and Recommended Values

    | Parameter Name | Description and Recommended Value |
    | --- | --- |
    | `GOGC` | **Description**: Defines the ratio of newly allocated memory to live memory that triggers garbage collection. Since kube-scheduler has low memory consumption, it is recommended to configure as 70.<br>**Recommended configuration**: `70`. |
    | `GOMEMLIMIT` | **Description**: Overall memory limit that the Go runtime can use.<br>**Recommended configuration**: `67500MiB` (85% of total memory limit). |
    | `GOMAXPROCS` | **Description**: Sets the number of runtime cores.<br>**Recommended configuration**: In multi-node scenarios, it is recommended to configure to the number of CPU cores in the environment. |
    | `HTTP2_READ_IDLE_TIMEOUT_SECONDS` | **Description**: client-go HTTP/2 connection health check timeout.<br>**Recommended configuration**: `15` (s), to ensure availability. |
    | `HTTP2_PING_TIMEOUT_SECONDS` | **Description**: client-go HTTP/2 connection health check timeout using PING.<br>**Recommended configuration**: `10` (s), to ensure availability. |


### References

[kube-scheduler Parameter Description](https://kubernetes.io/zh-cn/docs/reference/command-line-tools-reference/kube-scheduler/)
