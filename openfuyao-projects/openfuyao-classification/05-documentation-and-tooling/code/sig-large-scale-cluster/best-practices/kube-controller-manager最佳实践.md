# 最佳实践

## 大规模集群场景下 kube-controller-manager 参数调优最佳实践

在 Kubernetes 中，每个控制器是一个控制回路，通过 API 服务器监视集群的共享状态， 并尝试进行更改以将当前状态转为期望状态。

### 目标

本最佳实践聚集在 kube-controller-manager 的性能提升，以支持大规模场景下的 kube-controller-manager 能力。主要包含如下部分：

- 调整 kube-controller-manager 资源占用，以支持大规模集群正常工作
- 调整 kube-controller-manager 参数，提高集群可用性，减少内存占用

### 前提条件(可选)

集群使用 kube-controller 作为 kubernetes 集群的控制管理器，并部署多个实例。

### 使用限制(可选)

基于installer部署的大规模集群环境下使用。

### 背景信息(可选)

在大规模的场景下，kube-controller-manager 需要对上万 Pod 等资源进行状态调谐，需要对此进行优化，防止 kube-controller-manager 性能劣化或内存溢出。

### 操作步骤(可选)

修改 kube-controller-manager 的部署模板。

#### 1.资源占用

CPU：CPU占用过高将影响可用性，若条件允许，建议不设置上限。

内存：若kube-controller-manager内存超过限制，将导致kube-controller-manager重启，导致kube-controller-manager在一定时间内无法正常进行资源调谐，影响整个集群可用性，若条件允许，建议不设置上限。若要设置，则可设置为90000Mi。

#### 2.启动参数

##### leader-elect

是否开启选主功能。默认值为true，建议配置为：true。

##### leader-elect-lease-duration

选主租约时长，超过此时长将开始选主。默认值为：15s，kube-controller-manager 主实例将定期续租，以保证Leader身份，此值设置过低将导致 kube-controller-manager 频繁访问 kube-apiserver 向 etcd 更新数据；设置过高将导致 kube-controller-manager 实例倒换时间过长，建议配置为：65s，提高可用性。

##### leader-elect-renew-deadline

刷新租约时长的周期，超过此时长将尝试续约Leader身份。默认值为：10s，建议配置为：60s，提高可用性。

##### kube-api-burst

与 Kubernetes API 服务器通信时突发峰值请求个数上限。默认值为：30 (次/s)，建议配置为：15000 (次/s)，以满足大规模集群需要。

##### kube-api-qps

与 API 服务器通信时每秒请求数（QPS）限制。默认值为：20 (次/s)，建议配置为：10000 (次/s)，以满足大规模集群需要。

##### node-eviction-rate

节点驱逐速率。 默认值为：0.1，表示每10s至多从1个节点驱逐Pod，建议配置为：0.05。

##### node-monitor-period

cloud-node-lifecycle-controller 对节点状态进行同步的周期。默认值为：10s，建议配置为：5s。

##### node-monitor-grace-period

在将一个 Node 标记为不健康之前允许其无响应的时长上限。默认值为：40s，此值依赖kubelet节点状态更新频率，建议保持不变，配置为：40s。

##### feature-gates

ComponentSLIs：在Kubernetes组件上启用 `/metrics/slis` endpoint，从而允许抓取健康检查指标，但会消耗额外的资源。建议配置为：false。

#### 3.环境变量

##### GOGC

将触发垃圾收集的新未分配堆内存相对于活跃内存的百分比。由于kube-controller-manager相对占用内存不多，因此建议配置为：70。

##### GOMEMLIMIT

Go运行时可以使用的总体内存限制。建议配置为：67500MiB（总内存限制的85%）。

##### GOMAXPROCS

设置运行时核心数量。多节点场景下建议配置为环境cpu核心数。

##### HTTP2_READ_IDLE_TIMEOUT_SECONDS

client-go HTTP/2连接健康检查超时时间。建议配置为：15 (s)，以确保可用性。

##### HTTP2_PING_TIMEOUT_SECONDS

client-go 使用PING执行HTTP/2连接健康检查超时时间。建议配置为：10 (s)，以确保可用性。

### 参考资料

> 列出参考或引用的任何相关资料、链接或资源。

[kube-controller-manager参数介绍](https://kubernetes.io/zh-cn/docs/reference/command-line-tools-reference/kube-controller-manager/)