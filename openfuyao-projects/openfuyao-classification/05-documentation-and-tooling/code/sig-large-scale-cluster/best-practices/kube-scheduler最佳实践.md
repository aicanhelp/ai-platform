# 最佳实践

## 大规模集群场景下 kube-scheduler 参数调优最佳实践

Kubernetes 调度器是一个控制面进程，负责将 Pods 指派到节点上。 调度器基于约束和可用资源为调度队列中每个 Pod 确定其可合法放置的节点。 调度器之后对所有合法的节点进行排序，将 Pod 绑定到一个合适的节点。 在同一个集群中可以使用多个不同的调度器。

### 目标

本最佳实践聚集在 kube-scheduler 的性能提升，以支持大规模场景下的 kube-scheduler 能力。主要包含如下部分：

- 调整 kube-scheduler 资源占用，以支持大规模集群正常工作
- 调整 kube-scheduler 参数，提高集群可用性，减少内存占用

### 前提条件(可选)

集群使用 kube-scheduler 作为 kubernetes 集群的调度器，并部署多个实例。

### 使用限制(可选)

基于installer部署的大规模集群环境下使用。

### 背景信息(可选)

在大规模的场景下，kube-scheduler 需要对上万 Pod 进行上千节点的调度，需要对此进行优化，防止 kube-scheduler 性能劣化或内存溢出。

### 操作步骤(可选)

修改 kube-scheduler 的部署模板。

#### 1.资源占用

CPU：CPU占用过高将影响可用性，若条件允许，建议不设置上限。

内存：若kube-scheduler内存超过限制，将导致kube-scheduler重启，导致kube-scheduler在一定时间内无法正常调度，影响可用性，若条件允许，建议不设置上限。若要设置，则可设置为90000Mi。

#### 2.启动参数

##### leader-elect

是否开启选主功能。默认值为true，建议配置为：true。

##### leader-elect-lease-duration

选主租约时长，超过此时长将开始选主。默认值为：15s，kube-scheduler主实例将定期续租，以保证Leader身份，此值设置过低将导致kube-scheduler频繁访问kube-apiserver向etcd更新数据；设置过高将导致 kube-scheduler 实例倒换时间过长，建议配置为：65s，提高可用性。

##### leader-elect-renew-deadline

刷新租约时长的周期，超过此时长将尝试续约Leader身份。默认值为：10s，kube-scheduler 主实例将定期续租，以保证Leader身份，此值设置过低将导致 kube-scheduler 频繁访问 kube-apiserver 向 etcd 更新数据，建议配置为：60s，提高可用性。

##### kube-api-burst

与 Kubernetes API 服务器通信时突发峰值请求个数上限。默认值为：100 (次/s)，建议配置为：10000 (次/s)，以满足大规模集群需要。

##### kube-api-qps

与 API 服务器通信时每秒请求数（QPS）限制。默认值为：50 (次/s)，建议配置为：5000 (次/s)，以满足大规模集群需要。

##### feature-gates

ComponentSLIs：在Kubernetes组件上启用 `/metrics/slis` endpoint，从而允许抓取健康检查指标，但会消耗额外的资源。建议配置为：false。

#### 3.环境变量

##### GOGC

将触发垃圾收集的新未分配堆内存相对于活跃内存的百分比。由于kube-scheduler相对占用内存不多，因此建议配置为：70。

##### GOMEMLIMIT

Go运行时可以使用的总体内存限制。建议配置为：67500MiB（总内存限制的85%）。

##### GOMAXPROCS

设置运行时核心数量。多节点场景下建议配置为环境cpu核心数。

##### HTTP2_READ_IDLE_TIMEOUT_SECONDS

client-go HTTP/2连接健康检查超时时间。建议配置为：15 (s)，以确保可用性。

##### HTTP2_PING_TIMEOUT_SECONDS

client-go 使用PING执行HTTP/2连接健康检查超时时间。建议配置为：10 (s)，以确保可用性。

### 参考资料

[kube-scheduler参数说明](https://kubernetes.io/zh-cn/docs/reference/command-line-tools-reference/kube-scheduler/)
