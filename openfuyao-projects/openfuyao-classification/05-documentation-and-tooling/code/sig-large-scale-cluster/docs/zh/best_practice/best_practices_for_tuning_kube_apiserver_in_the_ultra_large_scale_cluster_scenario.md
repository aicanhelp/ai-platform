# 最佳实践

## 大规模集群场景下kube-apiserver参数调优最佳实践

Kubernetes API服务器验证并配置API对象的数据，这些对象包括pods、services、replicationcontrollers等。API服务器为REST操作提供服务，并为集群的共享状态提供前端，所有其他组件都通过该前端进行交互。

### 目标

本最佳实践聚集在kube-apiserver的性能提升，以支持大规模场景下的kube-apiserver能力。主要包含如下部分。

- 调整kube-apiserver资源占用，以支持大规模集群正常工作。
- 调整kube-apiserver参数，提高集群可用性，减少内存占用。

### 前提条件

集群使用kube-apiserver作为Kubernetes集群的API服务器，并部署在六个节点上。

### 使用限制

基于installer部署的大规模集群环境下使用。

### 背景信息

- 在大规模的场景下，ETCD存储的数据量可能会达到100GB以上，在这种场景下，kube-apiserver缓存全量数据将消耗超过100GB的内存资源，且可能存在性能瓶颈。

- 在大规模的场景下，集群内每秒的读写请求量极大，需要对此进行优化，防止kube-apiserver及etcd性能劣化或内存溢出。

### 操作步骤

修改kube-apiserver的部署模板。

1. 资源占用

   -  CPU：CPU占用过高将影响集群可用性，若条件允许，建议不设置上限。
   - 内存：若kube-apiserver内存超过限制，将导致kube-apiserver重启，影响整个集群可用性，若条件允许，建议不设置上限。

2. 启动参数

    **表1** 启动参数配置信息

    | 参数名 | 说明与建议值 |
    | :--- | :--- |
    | `etcd-max-call-recv-msg-size` | **说明**：客户端请求etcd可接收的最大字节数。<br>**默认值**：0（即2147483647 B）。<br>**建议配置**：`2147483647` (B)。 |
    | `etcd-max-call-send-msg-size` | **说明**：客户端请求etcd可发送的最大字节数。<br>**默认值**：2MiB。<br>**建议配置**：`110100480` (B)。 |
    | `hw-access-log-path` | **说明**：访问日志存储路径。<br>**默认值**：无。<br>**建议**：根据实际情况配置日志存储路径。 |
    | `hw-access-log-check-log-deleted-period` | **说明**：访问日志是否需要重建扫描周期。<br>**默认值**：0。<br>**建议配置**：`5`。 |
    | `hw-access-log-permissions` | **说明**：访问日志文件权限。<br>**默认值**：0644。<br>**建议配置**：`0600`。 |
    | `audit-log-path` | **说明**：审计日志存储路径。<br>**默认值**：无。<br>**建议**：根据实际情况配置日志存储路径。 |
    | `audit-log-maxbackup` | **说明**：审计日志最大保留个数。<br>**建议配置**：`50` (个)。 |
    | `audit-log-maxage` | **说明**：审计日志最大保留天数。<br>**建议配置**：`30` (天)。 |
    | `audit-log-maxsize` | **说明**：单个审计日志最大文件大小。<br>**建议配置**：`10` (MB)。 |
    | `audit-log-mode` | **说明**：审计日志保存模式。<br>**建议配置**：`batch`。 |
    | `audit-policy-file` | **说明**：审计日志记录策略配置。<br>**建议**：根据实际情况配置策略配置文件路径。<br>**策略文件内容建议**：详细配置请参考表格末尾`审计策略配置示例文件`。 |
    | `goaway-chance` | **说明**：为均衡负载，随机关闭某连接（GOAWAY）的请求比例。单个kube-apiserver或不使用负载均衡器的集群不应启用。<br>**建议配置**：`0.005`。 |
    | `etcd-servers-overrides` | **说明**：etcd针对每个资源的重载设置。建议将event及lease数据单独存储。<br>**建议配置**：`/events#etcd-event集群endpoints，coordination.k8s.io/leases#etcd-event集群endpoints`。 |
    | `delete-collection-workers` | **说明**：为DeleteCollection调用而产生的工作线程数，用于加速名字空间清理。<br>**默认值**：1 (个)。<br>**建议配置**：`100` (个)，以提升清理效率。 |
    | `feature-gates` | **说明**：APIServerTracing允许开启组件追踪能力。开启后会占用资源。<br>**建议配置**：`APIServerTracing=false`。 |
    | `max-requests-inflight` | **说明**：限制进行中非变更类型请求的最大个数，以减少内存占用。<br>**建议配置**：`10000` (次/s)。 |
    | `max-mutating-requests-inflight` | **说明**：限制同时运行的变更类型的请求的个数上限，以减少内存占用。<br>**建议配置**：`5000` (次/s)。 |
    | `default-not-ready-toleration-seconds` | **说明**：对污点`NotReady:NoExecute`的默认容忍时长。配置过高将导致Pod无法及时从不可用节点驱逐。<br>**默认值**：300 (s)。<br>**建议配置**：`60` (s)。 |
    | `default-unreachable-toleration-seconds` | **说明**：对污点`Unreachable:NoExecute`的默认容忍时长。配置过高将导致Pod无法及时从不可用节点驱逐。<br>**默认值**：300 (s)。<br>**建议配置**：`60` (s)。 |


    `audit-policy-file`字段审计策略配置示例文件如下。

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

3. 环境变量
   
    **表2** 环境变量配置信息
    | 参数名 | 说明与建议值 |
    | :--- | :--- |
    | `GOGC` | **说明**：定义触发垃圾回收的新分配内存与存活内存之比。由于kube-apiserver内存消耗大，需要更积极的内存管理。<br>**建议配置**：`50`。 |
    | `GOMAXPROCS` | **说明**：设置运行时核心数量。<br>**建议配置**：多节点场景下建议配置为环境CPU核心数。 |
    | `HTTP2_READ_IDLE_TIMEOUT_SECONDS` | **说明**：client-go HTTP/2连接健康检查超时时间。<br>**建议配置**：`8` (s)，以确保可用性。 |
    | `HTTP2_PING_TIMEOUT_SECONDS` | **说明**：client-go使用PING执行HTTP/2连接健康检查超时时间。<br>**建议配置**：`4` (s)，以确保可用性。 |


### 后续步骤

#### 其它服务建议整改项

- 使用负载均衡，将请求分配到多个kube-apiserver实例。

    可通过keepalived、haproxy为kube-apiserver配置负载均衡。

- 客户端请求尽量携带resourceVersion=0。

    kube-apiserver会缓存从etcd查询的数据，以在极大程度上减少对etcd的访问频率，节省大量的网络调用，降低对etcd集群的读写压力，提高访问性能。但是当客户端发起请求时，在ListOption中未设置resourceVersion时，此时kube-apiserver将直接从etcd中拉取最新数据返回给客户端，这一行为将消耗访问性能。因此，客户端应尽量避免此类操作，应在ListOption中设置resourceVersion=0，使kube-apiserver直接从缓存中读取数据，提高效率。

### 参考资料

- [Kubernetes API概念](https://kubernetes.io/docs/reference/using-api/api-concepts/)
- [kube-apiserver参数说明](https:/kubernetes.io/docs/reference/command-line-tools-reference/kube-apiserver/)
- [大规模场景下Kubernetes集群的性能优化](https://cloud.tencent.com/developer/article/1553950)
