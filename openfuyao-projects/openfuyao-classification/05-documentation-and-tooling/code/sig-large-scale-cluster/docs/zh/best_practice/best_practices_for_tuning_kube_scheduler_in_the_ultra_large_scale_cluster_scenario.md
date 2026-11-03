# 最佳实践

## 大规模集群场景下kube-scheduler参数调优最佳实践

Kubernetes调度器是控制平面的核心组件，负责将Pod指派到集群节点。它从调度队列中获取Pod，首先根据资源需求和约束条件筛选出所有可调度的合法节点，然后依据优先级策略对这些节点进行排序，最终将Pod绑定至最优节点，Kubernetes支持同时运行多个不同的调度器。

### 目标

本最佳实践旨在提升kube-scheduler的性能，以支持大规模场景下的kube-scheduler能力，主要包含如下部分。

- 调整kube-scheduler资源占用，以支持大规模集群正常工作。
- 调整kube-scheduler参数，提高集群可用性，减少内存占用。

### 前提条件

集群使用kube-scheduler作为Kubernetes集群的调度器，并部署多个实例。

### 使用限制

适用于通过installer部署工具构建的大规模集群环境。

### 背景信息

在大规模的场景下，kube-scheduler需要对上万Pod进行上千节点的调度，需要对此进行优化，防止kube-scheduler性能劣化或内存溢出。

### 操作步骤
请按以下步骤，通过修改kube-scheduler对应的部署配置以完成优化。

1. 资源占用

   - CPU：CPU占用过高将影响可用性，若条件允许，建议不设置上限。

   - 内存：若kube-scheduler内存超过限制，将导致kube-scheduler重启，导致kube-scheduler在一定时间内无法正常调度，影响可用性，若条件允许，建议不设置上限。若要设置，则可设置为90000Mi。

2. 启动参数

    **表1** 启动参数说明与建议值

    | 参数名 | 说明与建议值 |
    | :--- | :--- |
    | `leader-elect` | **说明**：是否开启选主功能。<br>**默认值**：true。<br>**建议配置**：`true`。 |
    | `leader-elect-lease-duration` | **说明**：选主租约时长，超过此时长将开始选主。kube-scheduler主实例将定期续租，以保证Leader身份，此值设置过低将导致kube-scheduler频繁访问kube-apiserver向etcd更新数据；设置过高将导致kube-scheduler实例倒换时间过长。<br>**默认值**：15s。<br>**建议配置**：`65s`，提高可用性。 |
    | `leader-elect-renew-deadline` | **说明**：刷新租约时长的周期，超过此时长将尝试续约Leader身份。kube-scheduler主实例将定期续租，以保证Leader身份，此值设置过低将导致kube-scheduler频繁访问kube-apiserver向etcd更新数据。<br>**默认值**：10s。<br>**建议配置**：`60s`，提高可用性。 |
    | `kube-api-burst` | **说明**：与Kubernetes API服务器通信时突发峰值请求个数上限。<br>**默认值**：100 (次/s)。<br>**建议配置**：`10000` (次/s)，以满足大规模集群需要。 |
    | `kube-api-qps` | **说明**：与API服务器通信时每秒请求数（QPS）限制。<br>**默认值**：50 (次/s)。<br>**建议配置**：`5000` (次/s)，以满足大规模集群需要。 |
    | `feature-gates` | **说明**：ComponentSLIs：在Kubernetes组件上启用 `/metrics/slis` endpoint，从而允许抓取健康检查指标，但会消耗额外的资源。<br>**在1.28版本中建议配置**：`ComponentSLIs=false`。 |



3. 环境变量

    **表2** 环境变量说明与建议值

    | 参数名 | 说明与建议值 |
    | --- | --- |
    | `GOGC` | **说明**：定义触发垃圾回收的新分配内存与存活内存之比。因kube-scheduler内存消耗较低，建议配置为70。<br>**建议配置**：`70`。 |
    | `GOMEMLIMIT` | **说明**：Go运行时可以使用的总体内存限制。<br>**建议配置**：`67500MiB`（总内存限制的85%）。 |
    | `GOMAXPROCS` | **说明**：设置运行时核心数量。<br>**建议配置**：多节点场景下建议配置为环境cpu核心数。 |
    | `HTTP2_READ_IDLE_TIMEOUT_SECONDS` | **说明**：client-go HTTP/2连接健康检查超时时间。<br>**建议配置**：`15` (s)，以确保可用性。 |
    | `HTTP2_PING_TIMEOUT_SECONDS` | **说明**：client-go使用PING执行HTTP/2连接健康检查超时时间。<br>**建议配置**：`10` (s)，以确保可用性。 |


### 参考资料

[kube-scheduler参数说明](https://kubernetes.io/zh-cn/docs/reference/command-line-tools-reference/kube-scheduler/)
