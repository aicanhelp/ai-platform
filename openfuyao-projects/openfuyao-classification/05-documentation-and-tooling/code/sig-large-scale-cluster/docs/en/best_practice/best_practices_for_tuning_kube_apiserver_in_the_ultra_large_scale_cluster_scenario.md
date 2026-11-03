# Best Practices

## Best Practices for Tuning kube-apiserver in the Ultra-Large-Scale Cluster Scenario

The Kubernetes API server validates and configures data for API objects, including pods, services, replicationcontrollers, and more. The API server provides services for REST operations and provides a front-end for the cluster's shared state, through which all other components interact.

### Objectives

This best practice focuses on improving kube-apiserver performance to support kube-apiserver capabilities in large-scale scenarios. It mainly includes the following parts:

- Adjust kube-apiserver resource usage to support the normal operation of large-scale clusters.
- Adjust kube-apiserver parameters to improve cluster availability and reduce memory usage.

### Prerequisites

The cluster uses kube-apiserver as the API server for the Kubernetes cluster and is deployed on six nodes.

### Restrictions

For use in large-scale cluster environments deployed with installer.

### Background

- In large-scale scenarios, the amount of data stored in ETCD may exceed 100GB. In such scenarios, kube-apiserver caching all data will consume more than 100GB of memory resources and may have performance bottlenecks.

- In large-scale scenarios, the number of read and write requests per second within the cluster is extremely large. Optimization is needed to prevent kube-apiserver and etcd performance degradation or memory overflow.

### Procedure

Modify the kube-apiserver deployment template.

1. Resource Usage

   - CPU: Excessive CPU usage will affect cluster availability. If conditions allow, it is recommended not to set an upper limit.
   - Memory: If kube-apiserver memory exceeds the limit, it will cause kube-apiserver to restart, affecting the availability of the entire cluster. If conditions allow, it is recommended not to set an upper limit.

2. Startup Parameters

    **Table 1** Startup Parameter Configuration

    | Parameter Name | Description and Recommended Value |
    | :--- | :--- |
    | `etcd-max-call-recv-msg-size` | **Description**: Maximum number of bytes the client can receive from etcd requests.<br>**Default value**: 0 (i.e., 2147483647 B).<br>**Recommended configuration**: `2147483647` (B). |
    | `etcd-max-call-send-msg-size` | **Description**: Maximum number of bytes the client can send in etcd requests.<br>**Default value**: 2MiB.<br>**Recommended configuration**: `110100480` (B). |
    | `hw-access-log-path` | **Description**: Access log storage path.<br>**Default value**: None.<br>**Recommendation**: Configure log storage path according to actual situation. |
    | `hw-access-log-check-log-deleted-period` | **Description**: Scan period for whether access logs need to be rebuilt.<br>**Default value**: 0.<br>**Recommended configuration**: `5`. |
    | `hw-access-log-permissions` | **Description**: Access log file permissions.<br>**Default value**: 0644.<br>**Recommended configuration**: `0600`. |
    | `audit-log-path` | **Description**: Audit log storage path.<br>**Default value**: None.<br>**Recommendation**: Configure log storage path according to actual situation. |
    | `audit-log-maxbackup` | **Description**: Maximum number of audit logs to retain.<br>**Recommended configuration**: `50`. |
    | `audit-log-maxage` | **Description**: Maximum number of days to retain audit logs.<br>**Recommended configuration**: `30` (days). |
    | `audit-log-maxsize` | **Description**: Maximum file size of a single audit log.<br>**Recommended configuration**: `10` (MB). |
    | `audit-log-mode` | **Description**: Audit log save mode.<br>**Recommended configuration**: `batch`. |
    | `audit-policy-file` | **Description**: Audit log recording policy configuration.<br>**Recommendation**: Configure policy configuration file path according to actual situation.<br>**Policy file content recommendation**: For detailed configuration, please refer to the `Audit Policy Configuration Example File` at the end of the table. |
    | `goaway-chance` | **Description**: For load balancing, the ratio of requests to randomly close a connection (GOAWAY). Should not be enabled for single kube-apiserver or clusters not using load balancers.<br>**Recommended configuration**: `0.005`. |
    | `etcd-servers-overrides` | **Description**: Overload settings for each resource in etcd. It is recommended to store event and lease data separately.<br>**Recommended configuration**: `/events#etcd-event cluster endpoints, coordination.k8s.io/leases#etcd-event cluster endpoints`. |
    | `delete-collection-workers` | **Description**: Number of worker threads for DeleteCollection calls, used to accelerate namespace cleanup.<br>**Default value**: 1.<br>**Recommended configuration**: `100`, to improve cleanup efficiency. |
    | `feature-gates` | **Description**: APIServerTracing allows enabling component tracing capabilities. When enabled, it consumes resources.<br>**Recommended configuration**: `APIServerTracing=false`. |
    | `max-requests-inflight` | **Description**: Limit the maximum number of in-progress non-mutating requests to reduce memory usage.<br>**Recommended configuration**: `10000` (times/s). |
    | `max-mutating-requests-inflight` | **Description**: Limit the maximum number of concurrent mutating requests to reduce memory usage.<br>**Recommended configuration**: `5000` (times/s). |
    | `default-not-ready-toleration-seconds` | **Description**: Default toleration duration for taint `NotReady:NoExecute`. Setting too high will prevent Pods from being evicted from unavailable nodes in a timely manner.<br>**Default value**: 300 (s).<br>**Recommended configuration**: `60` (s). |
    | `default-unreachable-toleration-seconds` | **Description**: Default toleration duration for taint `Unreachable:NoExecute`. Setting too high will prevent Pods from being evicted from unavailable nodes in a timely manner.<br>**Default value**: 300 (s).<br>**Recommended configuration**: `60` (s). |


    Example audit policy configuration file for the `audit-policy-file` field:

    ```yaml
    apiVersion: audit.k8s.io/v1 # This is required.
    kind: Policy
    # Don't generate audit events for all requests in RequestReceived stage.
    omitStages:
      - "RequestReceived"
    rules:
      - level: None
        verbs: ["get", "list", "watch"]
      # Don't log events requests.
      - level: None
        resources:
          - group: "" # core
            resources: ["events"]
      # Don't log status change
      - level: None
        verbs: ["update", "patch"]
        resources:
          - group: ""
            resources: ["*/status", "*/log"]
          - group: "apps"
            resources: ["*/status"]
          - group: "batch"
            resources: ["*/status"]
      # Don't log node update lease in heartbeat
      - level: None
        userGroups: ["system:nodes", "system:kube-controller-managers", "system:kube-schedulers"]
        verbs: ["update", "patch"]
        namespaces: ["kube-node-lease", "kube-system"]
        resources:
          - group: "coordination.k8s.io"
            resources: ["leases"]
      - level: Metadata
    ```

3. Environment Variables
   
    **Table 2** Environment Variable Configuration

    | Parameter Name | Description and Recommended Value |
    | :--- | :--- |
    | `GOGC` | **Description**: Defines the ratio of newly allocated memory to live memory that triggers garbage collection. Since kube-apiserver consumes a lot of memory, more aggressive memory management is needed.<br>**Recommended configuration**: `50`. |
    | `GOMAXPROCS` | **Description**: Sets the number of runtime cores.<br>**Recommended configuration**: In multi-node scenarios, it is recommended to configure to the number of CPU cores in the environment. |
    | `HTTP2_READ_IDLE_TIMEOUT_SECONDS` | **Description**: client-go HTTP/2 connection health check timeout.<br>**Recommended configuration**: `8` (s), to ensure availability. |
    | `HTTP2_PING_TIMEOUT_SECONDS` | **Description**: client-go HTTP/2 connection health check timeout using PING.<br>**Recommended configuration**: `4` (s), to ensure availability. |


### Follow-up Steps

#### Other Service Improvement Recommendations

- Use load balancing to distribute requests to multiple kube-apiserver instances.

    Load balancing for kube-apiserver can be configured using keepalived and haproxy.

- Client requests should carry resourceVersion=0 as much as possible.

    Kube-apiserver caches data queried from etcd to greatly reduce the frequency of access to etcd, save a lot of network calls, reduce read and write pressure on the etcd cluster, and improve access performance. However, when a client initiates a request and does not set resourceVersion in ListOption, kube-apiserver will directly pull the latest data from etcd and return it to the client. This behavior will consume access performance. Therefore, clients should avoid such operations as much as possible and should set resourceVersion=0 in ListOption to make kube-apiserver read data directly from the cache to improve efficiency.

### References

- [Kubernetes API Concepts](https://kubernetes.io/docs/reference/using-api/api-concepts/)
- [kube-apiserver Parameter Description](https://kubernetes.io/docs/reference/command-line-tools-reference/kube-apiserver/)
- [Performance Optimization of Kubernetes Clusters in Large-Scale Scenarios](https://cloud.tencent.com/developer/article/1553950)
