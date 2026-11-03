# 开源组件列表

## prometheus

| 名称                        | 版本      | License             | 官网链接                                                                              | 描述                                    | 版本火车链接 | 状态           | 仓库                                                           | 备注                                         |
|---------------------------|---------|---------------------|-----------------------------------------------------------------------------------|---------------------------------------|--------|--------------|--------------------------------------------------------------|--------------------------------------------|
| Alertmanager              | v0.27.0 | Apache License v2.0 | <https://prometheus.io/>                                                          | 对告警进行处理                               | -      | 入版本火车 (不是GA) | <https://github.com/prometheus/alertmanager>                 | CleanSource V7.0                           |
| Blackbox Exporter         | v0.25.0 | Apache License v2.0 | <https://prometheus.io/>                                                          | 提供HTTP, HTTPS, DNS, TCP以及ICMP的监控数据采集  | -      | 未入版本火车       | <https://github.com/prometheus/blackbox_exporter>            | 未入版本火车                                     |
| Kube State Metrics        | v2.12.0 | Apache License v2.0 | <https://kubernetes.io/docs/concepts/cluster-administration/kube-state-metrics/>  | 监听Kubernetes API服务器并获取Kubernetes对象的指标 | -      | 未入版本火车       | <https://github.com/kubernetes/kube-state-metrics>           | v1.9.7: CBU-Cloud CleanSource V0.0         |
| Node Exporter             | v1.8.1  | Apache License v2.0 | <https://prometheus.io/>                                                          | 收集宿主机（如物理或虚拟服务器）上的硬件和操作系统级别指标的一个组件    | -      | 未入版本火车       | <https://github.com/prometheus/node_exporter>                | v0.18.1-0.2...: CBU-Cloud CleanSource V0.0 |
| Kubernetes Metrics Server | v0.7.1  | Apache License v2.0 | <https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-metrics-pipeline/> | 集群核心监控数据的聚合器                          | -      | 未入版本火车       | <https://github.com/kubernetes-sigs/metrics-server>          | 未入版本火车                                     |
| Prometheus                | v2.52.0 | Apache License v2.0 | <https://prometheus.io/>                                                          | 云原生系统监控                               | -      | 未入版本火车       | <https://github.com/prometheus/prometheus>                   | v2.50.1: CBG-Cloud CleanSource v5.1        |
| Prometheus Operator       | v0.74.0 | Apache License v2.0 | <https://prometheus-operator.dev/>                                                |                                       | -      | 未入版本火车       | <https://github.com/prometheus-operator/prometheus-operator> | v0.68.0: CBU-Cloud CleanSource V0.0        |

## k8s

| 名称         | 版本     | License             | 官网链接                     | 描述                    | 版本火车链接 | 状态     | 仓库                                             | 备注               |
|------------|--------|---------------------|--------------------------|-----------------------|--------|--------|------------------------------------------------|------------------|
| kubernetes | 1.28.8 | Apache License v2.0 | <https://kubernetes.io/> |                       | -      | 未入版本火车 | <https://github.com/kubernetes/kubernetes>     | 版本火车里最近版本是1.28.4 |
| kube-vip   | 0.7.2  | Apache License v2.0 | <https://kube-vip.io/>   | 部署k8s高可用组件            | -      | 未入版本火车 | <https://github.com/kube-vip/kube-vip>         | -                |
| cri-tools  | 1.28.0 | Apache License v2.0 | -                        | Kubernetes容器运行时接口交互工具 | -      | 未入版本火车 | <https://github.com/kubernetes-sigs/cri-tools> | 1.25.0 GA        |
| kubeadm    | 1.28.8 | Apache License v2.0 | <https://kubernetes.io/> |                       | -      | 未入版本火车 |                                                | -                |
| kubectl    | 1.28.8 | Apache License v2.0 | <https://kubernetes.io/> |                       | -      | 未入版本火车 |                                                | -                |
| nerdctl    | 1.7.5  | Apache License v2.0 | <https://kubernetes.io/> |                       | -      | 未入版本火车 |                                                | -                |

## Harbor

| 名称        | 版本     | License             | 官网链接                                       | 描述                               | 版本火车链接 | 状态     | 仓库                                         | 备注                                                                                           |
|-----------|--------|---------------------|--------------------------------------------|----------------------------------|--------|--------|--------------------------------------------|----------------------------------------------------------------------------------------------|
| nerdctl   | 1.7.5  | Apache License v2.0 | <https://containerd.io/>                   | containerd客户端工具，完全兼容docker cli命令 | -      | 未入版本火车 | <https://github.com/containerd/nerdctl>    | -                                                                                            |
| harbor    | 2.7.0  | Apache License v2.0 | <https://goharbor.io/>                     | 镜像仓                              | -      | 未入版本火车 | <https://github.com/goharbor/harbor>       | 官方不支持arm64架构，所以所有镜像均从三方获取（<https://github.com/orgs/octohelm/packages?tab=packages&q=harbor>） |
| helm-push | 0.10.4 | Apache License v2.0 | <https://github.com/chartmuseum/helm-push> | helm上传相关组件                       | -      | 未入版本火车 | <https://github.com/chartmuseum/helm-push> | console-service上传依赖                                                                          |

## CRI

| 名称         | 版本     | License             | 官网链接                     | 描述    | 版本火车链接 | 状态     | 仓库                                         | 备注         |
|------------|--------|---------------------|--------------------------|-------|--------|--------|--------------------------------------------|------------|
| containerd | 1.7.14 | Apache License v2.0 | <https://containerd.io/> | 容器运行时 | -      | 未入版本火车 | <https://github.com/containerd/containerd> | 1.7.7入 未GA |

## OCI

| 名称   | 版本     | License             | 官网链接 | 描述      | 版本火车链接 | 状态     | 仓库                                       | 备注                   |
|------|--------|---------------------|------|---------|--------|--------|------------------------------------------|----------------------|
| runc | 1.1.12 | Apache License v2.0 | -    | 低阶容器运行时 | -      | 未入版本火车 | <https://github.com/opencontainers/runc> | CleanSource V7.0 选型中 |

## CNI

| 名称          | 版本     | License             | 官网链接                                   | 描述             | 版本火车链接 | 状态     | 仓库                                               | 备注        |
|-------------|--------|---------------------|----------------------------------------|----------------|--------|--------|--------------------------------------------------|-----------|
| cni-plugins | 1.4.1  | Apache License v2.0 | -                                      | containerd网络插件 | -      | 未入版本火车 | <https://github.com/containernetworking/plugins> | -         |
| calico      | 3.27.3 | Apache License v2.0 | <https://www.tigera.io/project-calico> | k8s网络插件        | -      | 未如版本火车 | <https://github.com/projectcalico/calico>        | 3.23.2 GA |

## Ingress控制器

| 名称   | 版本     | License             | 官网链接 | 描述      | 版本火车链接 | 状态     | 仓库                                       | 备注                   |
|------|--------|---------------------|------|---------|--------|--------|------------------------------------------|----------------------|
| ingress-nginx | 1.9.4 | Apache License v2.0 | -    | 基于nginx的ingress控制器 | -      | 入版本火车（GA） | <https://kubernetes.github.io/ingress-nginx/> | CleanSource V7.0 |

## k8s包管理工具

| 名称   | 版本     | License             | 官网链接               | 描述    | 版本火车链接 | 状态     | 仓库                             | 备注        |
|------|--------|---------------------|--------------------|-------|--------|--------|--------------------------------|-----------|
| helm | 3.14.2 | Apache License v2.0 | <https://helm.sh/> | 包管理工具 | -      | 未入版本火车 | <https://github.com/helm/helm> | 3.10.3 GA | 

## 日志组件

| 名称       | 版本    | License | 官网链接                              | 描述 | 版本火车链接 | 状态 | 仓库               | 备注                     |
|----------|-------|---------|-----------------------------------|----|--------|----|------------------|------------------------|
| Loki     | 2.9.1 | AGPL v3 | <https://github.com/grafana/loki> |    |        |    | logging-operator | 不涉及源代码的修改和复用，只是原生接口的调用 |
| Promtail | 2.9.3 | AGPL v3 | <https://github.com/grafana/loki> |    |        |    | logging-operator | 不涉及源代码的修改和复用，只是原生接口的调用 |

## 基础工具

| 名称    | 版本     | License              | 官网链接                              | 描述         | 版本火车链接 | 状态     | 仓库                                    | 备注                                    |
|-------|--------|----------------------|-----------------------------------|------------|--------|--------|---------------------------------------|---------------------------------------|
| yq    | 4.43.1 | MIT license          | <https://mikefarah.gitbook.io/yq> | 编辑yaml文件工具 | -      | 未入版本火车 | <https://github.com/mikefarah/yq>     | 3.0.0 入CBU-Cloud CleanSource V0.0 未GA | 
| jq    | 1.7.1  | CC BY 3.0            | <https://jqlang.github.io/jq/>    | 编辑json文件工具 | -      | 未入版本火车 | <https://github.com/jqlang/jq>        | 1.6 入CBG-Cloud CleanSource V2.3 EOM   | 
| cfssl | 1.6.4  | BSD-2-Clause license | -                                 | 生成证书       | -      | 未入版本火车 | <https://github.com/cloudflare/cfssl> | 入CBU-Cloud CleanSource V0.0 未GA       | 
