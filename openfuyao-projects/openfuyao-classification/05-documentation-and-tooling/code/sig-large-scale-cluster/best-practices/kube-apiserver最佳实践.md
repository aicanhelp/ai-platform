# 最佳实践

## 大规模集群场景下 kube-apiserver 参数调优最佳实践

kube-apiserver 功能概述：

Kubernetes API 服务器验证并配置 API 对象的数据， 这些对象包括 pods、services、replicationcontrollers 等。 API 服务器为 REST 操作提供服务，并为集群的共享状态提供前端， 所有其他组件都通过该前端进行交互。

### 目标

本最佳实践聚集在 kube-apiserver 的性能提升，以支持大规模场景下的 kube-apiserver 能力。主要包含如下部分：

- 调整 kube-apiserver 资源占用，以支持大规模集群正常工作
- 调整 kube-apiserver 参数，提高集群可用性，减少内存占用

### 前提条件(可选)

集群使用 kube-apiserver 作为 kubernetes 集群的 API 服务器，并部署在六个节点上。

### 使用限制(可选)

基于installer部署的大规模集群环境下使用。

### 背景信息(可选)

1.在大规模的场景下，ETCD存储的数据量可能会达到100GB以上，在这种场景下，kube-apiserver缓存全量数据将消耗超过100GB的内存资源，且可能存在性能瓶颈；

2.在大规模的场景下，集群内每秒的读写请求量极大，需要对此进行优化，防止 kube-apiserver 及 etcd 性能劣化或内存溢出。

### 操作步骤(可选)

修改 kube-apiserver 的部署模板。

#### 1.资源占用

CPU：CPU占用过高将影响集群可用性，若条件允许，建议不设置上限。

内存：若kube-apiserver内存超过限制，将导致kube-apiserver重启，影响整个集群可用性，若条件允许，建议不设置上限。

#### 2.启动参数

##### etcd-max-call-recv-msg-size（大规模集群新增定制参数）

客户端请求etcd可接收的最大字节数。默认值为：0，即2147483647。建议配置为：2147483647 (B)。

##### etcd-max-call-send-msg-size（大规模集群新增定制参数）

客户端请求etcd可发送的最大字节数。默认值为：2MiB。建议配置为：110100480 (B)。

##### hw-access-log-path（大规模集群新增定制参数）

访问日志存储路径。无默认值。建议根据实际情况配置日志存储路径。

##### hw-access-log-check-log-deleted-period

访问日志是否需要重建扫描周期。默认值0。建议配置为：5。

##### hw-access-log-permissions

访问日志文件权限。默认值0644。建议配置为：0600

##### audit-log-path

审计日志存储路径。无默认值。建议根据实际情况配置日志存储路径。

##### audit-log-maxbackup

审计日志最大保留个数。建议配置为：50 (个)。

##### audit-log-maxage

审计日志最大保留天数。建议配置为：30 (天)。

##### audit-log-maxsize

单个审计日志最大文件大小。建议配置为：10 (MB)。

##### audit-log-mode

审计日志保存模式。建议配置为：batch。

##### audit-policy-file

审计日志记录策略配置。建议根据实际情况配置策略配置文件路径。

策略配置文件配置如下：

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

##### goaway-chance

为了防止由于某 kube-apiserver 重启导致HTTP/2客户端卡在单个其它 kube-apiserver 上而使负载不均衡，随机关闭某连接（GOAWAY）。 客户端的其他运行中请求不会受到影响。被关闭的客户端将重新连接， 重新被负载均衡后可能会与其他 kube-apiserver 开始通信。 此参数设置将被发送 GOAWAY 指令的请求的比例。 只有一个 kube-apiserver 或不使用负载均衡器的集群不应启用此特性。建议配置为：0.005。

##### etcd-servers-overrides

etcd针对每个资源的重载设置，以逗号分隔。 单个替代格式：组/资源#服务器（group/resource#servers）， 其中服务器是 URL，以分号分隔。 注意，此选项仅适用于编译进此服务器二进制文件的资源。由于etcd中event事件和lease占用了大量的比重，因此建议将kubernetes集群中event及lease数据单独存储在一套 etcd 中，推荐配置为：/events#etcd-event集群endpoints,coordination.k8s.io/leases#etcd-event集群endpoints。

##### delete-collection-workers

为DeleteCollection调用而产生的工作线程数。 这些用于加速名字空间清理。默认值为1 (个)，建议配置为：100 (个)，提升清理效率。

##### feature-gates

APIServerTracing：允许开启组件tracing能力。由于1.27后此特性默认开启，开启后将增加trace信息，但是会占用资源，建议配置为：false。

##### max-requests-inflight

如果--enable-priority-and-fairness为true，那么此值和 --max-mutating-requests-inflight 的和将确定服务器的总并发限制。否则该值限制进行中非变更类型请求的最大个数，零表示无限制。需要对 kube-apiserver 的请求进行限制，以减少 kube-apiserver 内存占用，建议配置为：10000 (次/s)。

##### max-mutating-requests-inflight

如果 --enable-priority-and-fairness为true，那么此值和 --max-requests-inflight 的和将确定服务器的总并发限制。 否则该值限制同时运行的变更类型的请求的个数上限。0 表示无限制。需要对 kube-apiserver 的请求进行限制，以减少 kube-apiserver 内存占用，建议配置为：5000 (次/s)。

##### default-not-ready-toleration-seconds

对污点 NotReady:NoExecute 的默认容忍时长。默认值为：300 (s)，此值配置过高将导致Pod无法及时从不可用节点驱逐，影响业务可用性，建议配置为：60 (s)。

##### default-unreachable-toleration-seconds

对污点 Unreachable:NoExecute 的默认容忍时长。默认值为：300，此值配置过高将导致Pod无法及时从不可用节点驱逐，影响业务可用性，建议配置为：60 (s)。

#### 3.环境变量

##### GOGC

将触发垃圾收集的新未分配堆内存相对于活跃内存的百分比。由于 kube-apiserver 对内存的消耗极大，建议配置为：50。

##### GOMAXPROCS

设置运行时核心数量。多节点场景下建议配置为环境cpu核心数。

##### HTTP2_READ_IDLE_TIMEOUT_SECONDS

client-go HTTP/2连接健康检查超时时间。建议配置为：8 (s)，以确保可用性。

##### HTTP2_PING_TIMEOUT_SECONDS

client-go 使用PING执行HTTP/2连接健康检查超时时间。建议配置为：4 (s)，以确保可用性。

### 后续步骤(可选)

#### 其它服务建议整改项

##### 使用负载均衡，将请求分配到多个 kube-apiserver 实例

可通过 keepalived、haproxy 为 kube-apiserver 配置负载均衡。

##### 客户端请求尽量携带resourceVersion=0

kube-apiserver 会缓存从 etcd 查询的数据，以在极大程度上减少对 etcd 的访问频率，节省大量的网络调用，降低对 etcd 集群的读写压力，提高访问性能。但是当客户端发起请求时，在 ListOption 中未设置 resourceVersion 时，此时 kube-apiserver 将直接从 etcd 中拉取最新数据返回给客户端，这一行为将消耗访问性能。因此，客户端应尽量避免此类操作，应在 ListOption 中设置 resourceVersion=0，使 kube-apiserver 直接从缓存中读取数据，提高效率。

### 参考资料

[Kubernetes API概念](https://v1-28.docs.kubernetes.io/zh-cn/docs/reference/using-api/api-concepts/)

[kube-apiserver参数说明](https://v1-28.docs.kubernetes.io/zh-cn/docs/reference/command-line-tools-reference/kube-apiserver/)

[大规模场景下 kubernetes 集群的性能优化](https://cloud.tencent.com/developer/article/1553950)
