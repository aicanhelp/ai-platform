# 最佳实践

## calico优化

Calico是一个高性能、可扩展的开源网络解决方案，专为云原生应用而设计。它提供了灵活的网络连接、网络安全和网络策略管理功能。Calico支持多种网络模型，包括纯三层网络（L3）、IP-in-IP隧道、VXLAN等，使其适用于各种不同的网络环境。

### 目标

通过配置BGP网络模型和部署Typha组件以实现集群网络性能的提升。

### 前提条件

Kubernetes集群已经部署，未部署网络插件。

### 使用限制

适用于Kubernetes 1.28

Calico版本为3.27.3

### 背景信息

#### Calico网络模型

Calico支持Overlay和UnderLay网络。Overlay网络是在现有网络基础设施上构建的虚拟网络，通过在物理网络上添加一层额外的协议来实现虚拟化和隔离。Underlay网络则是指底层的物理网络基础设施，它直接连接设备和网络节点。Overlay网络需要额外的协议和处理，这会增加数据传输的开销。Underlay网络则直接使用物理网络，减少了这些开销。Overlay网络的多层次结构会增加网络延迟。Underlay网络则直接连接设备，减少了传输路径上的延迟。在calico网络模型中IP-in-IP隧道、VXLAN网络模型为Overlay网络，BGP网络模型为UnderLay网络。

**IPIP模式**

把一个IP数据包又套在一个IP包里，即把IP层封装到IP层的一个 tunnel，它的作用其实基本上就相当于一个基于IP层的网桥，一般来说，普通的网桥是基于mac层的，根本不需要IP，而这个ipip则是通过两端的路由做一个tunnel，把两个本来不通的网络通过点对点连接起来；

calico以ipip模式部署完毕后，node上会有一个tunl0的网卡设备，这是ipip做隧道封装用的,也是一种overlay模式的网络。当我们把节点下线，calico容器都停止后，这个设备依然还在，执行 rmmodipip命令可以将它删除。

**BGP模式**

边界网关协议（Border Gateway Protocol, BGP）是互联网上一个核心的去中心化自治路由协议。它通过维护IP路由表或‘前缀’表来实现自治系统（AS）之间的可达性，属于矢量路由协议。BGP不使用传统的内部网关协议（IGP）的指标，而使用基于路径、网络策略或规则集来决定路由。因此，它更适合被称为矢量性协议，而不是路由协议。

虽然说calico有BGP模式和IPIP模式但是并不是说IPIP模式就不用建立BGP连接了，IPIP模式也是需要建立BGP连接的，只不过建立BGP链接的目标比较清晰，就是对端的tunl0对应的网卡。BGP模式对于IPIP模式的优点也并不是简单的描述为“可以跨节点通信”，IPIP模式也是可以跨节点通信的，只要两个节点能互相连通就可以。而BGP的优点是可以通过指定BGP的对端Peer，一般是交换机，那么只要能接入这个交换机的host或者PC都能通过路由的方式连通calico的其他节点上的容器。也就是说BGP的可扩展的网络拓扑更灵活。


**两种模式对比**

1. IPIP：

    - 流量：tunl0设备封装数据，形成隧道，承载流量。
    - 适用网络类型：适用于互相访问的Pod不在同一个网段中，跨网段访问的场景，外层封装的IP能够解决跨网段的路由问题。
    - 效率：流量需要tunl0设备封装，效率略低。

2. BGP:

    - 流量：使用路由信息导向流量。
    - 使用网络类型：适用于互相访问的Pod在同一个网段。
    - 效率：原生hostgw，效率高。   


**性能评测指标**

1.ping延迟: 用ping测试hosts之间和pods之间的延迟。

2.带宽测试: 用iperf测试hosts之间和pods之间的带。

| 带宽      | Node to Remote Node | Node to Remote Pod | Pod to Remote Pod |
|---------|---------------------|--------------------|-------------------|
| IPIP    | 5.07Gbits/sec       | 4.43Gbits/sec      | 4.76Gbits/sec     |
| BGP     | 5.08Gbits/sec       | 4.67Gbits/sec      | -                 |
| ping包延迟 |                     |                    |                   |
| IPIP    | 0.1891ms            | 0.2973ms           | 0.3120ms          |
| BGP     | 0.1888ms            | 0.2680ms           | -                 |

**BGP配置方式**

Calico的BGP网络模型有三种配置方式：

- **Full-mesh(全网状连接)**：启动BGP后，Calico默认会建立内部BGP连接的全网状连接，其中每个节点相互对等。全网状连接非常适合100个节点或更少节点的中小行集群，但在规模明显更大的情况下，全网状结构的效率会降低，Calico官方建议使用路由反射器模式。
- **Route reflectors(路由反射器)**：要构建大型内部BGP集群，可以使用BGP路由反射器来减少每个节点上使用的BGP对等体的数量。在此模型中，一些节点充当路由反射器，并被配置为在他们之间建立一个完整的网格，然后其他节点配置为与这些路由反射器的子集对等，与全网状相比减少了BGP对等连接的总数。
- **Top of Rack(ToR)**：在本地部署中，可以直接让calico和物理网络基础设施建立BGP连接，这需要把calico默认自带的Full-mesh先禁用掉，然后将Calico和本地的L3 ToR路由建立连接。此方式需要修改路由器或交换机配置，在不熟悉现网物理设备网络配置情况下，不建议使用此配置。

#### typha组件

calico在运行时会频繁访问Kubernetes apiserver或etcd数据库(简称数据存储区)以获取网络策略、同步配置信息等，在大规模集群中，Calico会导致数据存储区负载增加。typha组件位于数据存储区和Felix的许多实例之间。其主要目的是通过减少每个节点对数据存储区的访问来减轻数据存储区的压力。一个tyhpa实例可以支持数百个Felix实例，大大减少了数据存储区的压力。此外，typha可以过滤掉与Felix无关的更新，从而减少了Felix的CPU使用率。在大规模集群中typha发挥着至关重要的作用。

使用operator部署时会自动部署typha组件，同时operator会自动计算typha组件副本数量，其计算逻辑如下：

- 当节点数小于等于2个时，typha实例数量为1
- 当节点数小于等于4个时，typha实例数量为2
- 当节点数量大于4个，计算出typha实例数量公式为  (nodes_num/ 200) + 1 +1
- 当节点数量大于4个，计算出typha实例数量小于3个，typha实例数量为3

```
	// 源码
	maxNodesPerTypha := 200

	// This gives a count of how many 200s so we need 1+ this number to get at least
	// 1 typha for every 200 nodes.
	typhas := (nodes / maxNodesPerTypha) + 1

	// We add one more to ensure there is always 1 extra for high availability purposes.
	typhas += 1

	// We have a couple special cases for small clusters. We want to ensure that we run one fewer
	// Typha instances than there are nodes, so that there is room for rescheduling. We also want
	// to ensure we have at least two, where possible, so that we have redundancy.
	if nodes <= 2 {
		// For one and two node clusters, we only need a single typha.
		typhas = 1
	} else if nodes <= 4 {
		// For three and four node clusters, we can run an additional typha.
		typhas = 2
	} else if typhas < 3 {
		// For clusters with more than 4 nodes, make sure we have a minimum of three for redundancy.
		typhas = 3
	}
```



### 操作步骤

#### 部署Typha组件，同时配置BGP路由反射网络模型

- 配置networkManager，集群内所有节点执行

```shell
cat > /etc/NetworkManager/conf.d/calico.conf<<EOF
[keyfile]
unmanaged-devices=interface-name:cali*;interface-name:tunl*;interface-name:vxlan.calico;interface-name:vxlan-v6.calico;interface-name:wireguard.cali;interface-name:wg-v6.cali
EOF

systemctl restart NetworkManager
```

- 安装operator

```shell
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.3/manifests/tigera-operator.yaml
```

- 下载并修改自定义资源

```shell
# 下载
curl https://raw.githubusercontent.com/projectcalico/calico/v3.27.3/manifests/custom-resources.yaml -O
# 修改需要自定义的 podCIDR
```

- 如果使用kubeadm部署集群且在集群初始化时未设置podSubnet，可修改kubeadm-config CofigMap字段配置podSubnet

```shell
    networking:
      podSubnet: 172.20.64.0/18  # 添加项，pod的网络段，和calico custom-resources.yaml中podCIDR字段的值一致
      dnsDomain: cluster.local
      serviceSubnet: 10.96.0.0/12
```

- 应用自定义资源

```shell
kubectl apply -f custom-resources.yaml
```

- 安装calicoctl

```shell
ARCH=$(uname -m)
case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
esac

wget https://github.com/projectcalico/calico/releases/download/v3.27.3/calicoctl-linux-${ARCH}
chmod +x calicoctl-linux-${ARCH}
mv calicoctl-linux-${ARCH} /usr/local/bin/calicoctl
```

- 创建BGPConfiguration.yaml并应用

```shell
apiVersion: projectcalico.org/v3
kind: BGPConfiguration
metadata:
  name: default
spec:
  logSeverityScreen: Info
  nodeToNodeMeshEnabled: false
  asNumber: 64512
```

```shell
calicoctl apply -f BGPConfiguration.yaml
```

- 配置指定节点充当路由反射器，路由反射器建议指定至少2个系欸但，建议8~12个。节点配置建议(8u, 32G)

```shell
kubectl label node <node-name> route-reflector=true
```

- 配置节点的ClusterID(通常是一个未使用的IPv4地址)，配置路由反射器的节点都要配置ClusterID

```shell
kubectl annotate node <node-name> projectcalico.org/RouteReflectorClusterID=244.0.0.1
```

- 使用标签选择器将路由反射器节点与其他非路由反射器节点配置为对等，创建BGPPeer.yaml并应用

```shell
apiVersion: projectcalico.org/v3
kind: BGPPeer
metadata:
  name: peer-with-route-reflectors
spec:
  nodeSelector: all()
  peerSelector: route-reflector == 'true'
```

```shell
calicoctl apply -f BGPPeer.yaml
```

### 后续步骤

#### Calico监控配置

Felix是一个守护进程，运行在每台机器上实现网络策略等功能，Felix是calico的大脑——[监控指标](https://archive-os-3-27.netlify.app/calico/3.27/reference/felix/prometheus)

Typha是一组可选的pod，可扩展Felix以扩展Calico节点和数据存储之间的流量——[监控指标](https://archive-os-3-27.netlify.app/calico/3.27/reference/typha/prometheus)

kube-controllers pod运行一组控制器，这些控制器负责各种控制平面功能，例如资源垃圾收集和与Kubernetes API的同步——[监控指标](https://archive-os-3-27.netlify.app/calico/3.27/reference/kube-controllers/prometheus)

**重要指标**

| 序号 | 指标                            | 说明                     |
|----|-------------------------------|------------------------|
| 1  | felix_ipset_errors            | 执行ipset-restore失败次数    |
| 2  | felix_iptables_restore_calls  | 执行iptables-restore次数   |
| 3  | felix_iptables_restore_errors | 执行iptables-restore失败次数 |
| 4  | felix_iptables_save_calls     | 执行iptables-save次数      |
| 5  | felix_iptables_save_errors    | 执行iptables-save失败次数    |
| 6  | felix_log_errors              | 日志报告error的次数           |
| 7  | ipam_allocations_per_node     | 每个节点上IP分配的数量           |
| 8  | ipam_blocks_pre_node          | 每个节点上分配的Block数量        |

**说明**：大规模集群时，由于节点和Pod资源较多，Calico的网络数据也会随之增加，需要ETCD和APIService做好扩容等操作。

#### 监控配置

##### Felix 配置

Felix prometheus metrics 默认是禁用的，需要手动更改 Felix 的配置（**prometheusMetricsEnabled**）

```sh
# kubectl patch felixConfiguration default  --patch '{"spec":{"prometheusMetricsEnabled": true}}' --type=merge

felixconfiguration.crd.projectcalico.org/default patched
```

确认结果：

```sh
# kubectl get felixConfiguration default -o yaml
……
spec:
  bpfLogLevel: ""
  logSeverityScreen: Info
  prometheusMetricsEnabled: true         #prometheusMetricsEnabled 是 true
  reportingInterval: 0s
```

##### 创建 service 暴露 Felix metrics

Prometheus 使用 Kubernetes 服务动态发现 endpoint，创建一个名为 `felix-metrics-svc` 的服务，让 Prometheus 去发现所有的 Felix metrics endpoints。

Felix 默认使用 9091 TCP 上报他的 metrics

```sh
# kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  labels:
    k8s-app: calico-felix-metrics
  name: calico-felix-metrics
  namespace: calico-system
spec:
  selector:
    k8s-app: calico-node
  ports:
  - port: 9091
    targetPort: 9091
    name: felix-metrics
EOF
```

##### Typha 配置

确认在集群中使用了Typha

```sh
# kubectl get pods -A | grep typha
calico-system      calico-typha-6d8d9ffd6c-96z4z    1/1     Running   0      176m
calico-system      calico-typha-6d8d9ffd6c-k2zpm    1/1     Running   0      176m
calico-system      calico-typha-6d8d9ffd6c-s62nr    1/1     Running   0      176m
```

Calico 的 Operator 安装会根据集群规模自动部署一个或多个 Typha 实例。默认情况下，这些实例的指标处于禁用状态。

使用以下命令指示`tigera-operator`启用 Typha 指标。

```bash
kubectl patch installation default --type=merge -p '{"spec": {"typhaMetricsPort":9093}}'
```

你应该看到类似以下内容的结果：

```bash
installation.operator.tigera.io/default patched
```

##### 创建 service 暴露 Typha metrics

```sh
# kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  labels:
    k8s-app: calico-typha-metrics
  name: calico-typha-metrics
  namespace: calico-system
spec:
  selector:
    k8s-app: calico-typha
  ports:
  - port: 9093
    targetPort: 9093
    name: typha-metrics
EOF
```

##### kube-controllers 配置

kube-controllers prometheus metrics 默认是启动的，使用 9094 TCP 端口。可以通过修改 **KubeControllersConfiguration** 资源调整端口。

```sh
# 不修改默认端口无需执行此步骤

# kubectl patch kubecontrollersconfiguration default  --patch '{"spec":{"prometheusMetricsPort": 9095}}'
```

##### 创建 service 暴露 kube-controllers metrics

在calico 的安装中已经默认创建了 kube-controllers metrics 的service，以下步骤可以跳过。

```sh
# kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  labels:
    k8s-app: calico-kube-controllers
  name: calico-kube-controllers-metrics
  namespace: calico-system
spec:
  selector:
    k8s-app: calico-kube-controllers
  ports:
  - port: 9094
    targetPort: 9094
    name: kube-controllers-metrics
EOF
```

可以使用以下命令来验证是否已安装。

```sh
kubectl get svc -n calico-system
```

您应该看到类似以下内容的结果：

```sh
calico-kube-controllers-metrics   ClusterIP   10.43.77.57    <none>      9094/TCP   1d
```

#### prometheus 采集指标

##### 创建ServiceMonitor

kube-prometheus-stack在部署时会创建 Prometheus、PodMonitor、ServiceMonitor、AlertManager 和 PrometheusRule 这 5 个 CRD 资源对象，然后会一直监控并维持这 5 个资源对象的状态。

- Prometheus 这个资源对象就是对 Prometheus Server 的抽象。
- PodMonitor 和 ServiceMonitor 就是 exporter 的各种抽象，是用来提供专门提供指标数据接口的工具，
- Prometheus 就是通过 PodMonitor 和 ServiceMonitor 提供的指标数据接口去 pull 数据的。
- ServiceMonitor 要求被监控的服务必须有对应的 Service。
- PodMonitor 虽然不需要应用创建相应的 Service，但必须在 Pod 中指定指标的端口和名称。

我们已经创建了对应的 service，所有使用  ServiceMonitor 来采集指标。使用以下 yaml 创建 ServiceMonitor 。

```sh
# vim prometheus-ServiceMonitorCalico.yaml

apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  labels:
    release: prometheus # 必须添加此 label,因为 prometheus 的对象 serviceMonitorSelector 匹配 release: prometheus 的 ServiceMonitor
  name: prometheus-calico-felix-metrics
  namespace: monitoring
spec:
  endpoints:
    - interval: 15s
      path: /metrics
      port: felix-metrics
  namespaceSelector:
    matchNames:
    - calico-system
  selector:
    matchLabels:
      k8s-app: calico-felix-metrics
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  labels:
    release: prometheus
  name: prometheus-calico-typha-metrics
  namespace: monitoring
spec:
  endpoints:
    - interval: 15s
      path: /metrics
      port: typha-metrics
  namespaceSelector:
    matchNames:
    - calico-system
  selector:
    matchLabels:
      k8s-app: calico-typha-metrics
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  labels:
    release: prometheus
  name: prometheus-calico-kube-controllers-metrics
  namespace: monitoring
spec:
  endpoints:
    - interval: 15s
      path: /metrics
      port: kube-controllers-metrics
  namespaceSelector:
    matchNames:
    - calico-system
  selector:
    matchLabels:
      k8s-app: calico-kube-controllers
```

在这段YAML文件中，各代码段的含义如下：

-  metadata 下的 name 和 namespace 将指定 ServiceMonitor 所需的一些关键元信息。
-  spec 的 endpoints 为服务端点，代表 Prometheus 所需的采集 Metrics 的地址。endpoints 为一个数组，同时可以创建多个 endpoints。每个 endpoints 包含三个字段，每个字段的含义如下：
- interval：指定 Prometheus 对当前 endpoints 采集的周期。
- path：指定 Prometheus 的采集路径。
- port：指定采集数据需要通过的端口，设置的端口为创建 Service 时端口所设置的 name。
-  spec 的 namespaceSelector 为需要发现的 Service 的范围。namespaceSelector 包含两个互斥字段，字段的含义如下：
- any：有且仅有一个值 true，当该字段被设置时，将监听所有符合 Selector 过滤条件的 Service 的变动。
- matchNames：数组值，指定需要监听的 namespace 的范围
-  spec 的 selector 用于选择 Service。

应用 ServiceMonitor

```sh
# kubectl apply -f prometheus-ServiceMonitorCalico.yaml
servicemonitor.monitoring.coreos.com/felix-metrics created
servicemonitor.monitoring.coreos.com/typha-metrics created
servicemonitor.monitoring.coreos.com/kube-controllers-metrics create
```

##### 检查 target

登录Prometheus dashboard查看新建的target是否正常。

如Prometheus没有开放对外端口，可以修改service的端口类型为NodePort

```sh
kubectl edit svc prometheus-k8s -n monitoring
```

根据生成/指定的nodePort访问dashboard

![](imagesargets.PNG)

##### 安装配置 Grafana

##### helm 安装 Grafana

```sh
# 添加 Grafana 官方仓库
helm repo add grafana https://grafana.github.io/helm-charts

# 更新仓库
helm repo update grafana

# 安装 Grafana
helm install grafana grafana/grafana -n monitoring
```

##### Grafana 添加 Prometheus 源

访问 Granfana 界面，选择 Data sources -> Add new data source

填写 Prometheus server URL。如：http://prometheus-k8s:9090

![](imagesrafana_DataSourcec.PNG)

#### 导入 dashbaord 配置文件

在 Grafana 界面选择 Dashboards -> New -> import。导入导入 `typha-dashboard.json` 和  `felix-dashboard.json` 两个dashbaord。

Grafana面板配置文件：https://github.com/projectcalico/calico/blob/master/manifests/grafana-dashboards.yaml

felix：

![](imageselix_dashboard.PNG)

typha:

![](imagesypha_dashboard.PNG)

##### 配置文件

请根据实际情况修改文件中的 `datasource` 字段。


#### 已知问题

- Grafana 暂时没有找到监控 calico-kube-controllers 的 dashbaord 已经有 Felix 和 typha 的监控 dashboard。


### 注意事项/常见问题

- 将所有路由反射器节点从集群内移除前需选择新的路由反射器节点

### 结论

相比于Overlay网络模型，Ubderlay网络具有更高的数据面转发性能。同时在纯路由模式下，也有三种方案：Calico BGP的Full-mesh、Route reflectors与Top of Rack (ToR)。Full-mesh方案存在一些限制，适用于小规模集群，集群节点越多，BGP连接就越多，需要建立大量连接来保证网络的互通性，每增加一个节点就要成倍的增加连接保证网络的互通性，这样就会使用大量的网络消耗。Route reflectors模式，也称RR模式。RR模式中会指定一个或多个节点作为路由反射器，集群内其他节点只与路由反射结点建立连接就可以获得全部的路由信息，进而减少了网络开销。Top of Rack模式会将Calico直接与物理网络基础架构(路由器或交换机)建立对等，因此需要修改路由器或交换机配置，在不熟悉现网物理设备网络配置情况下，不建议使用此配置。

### 参考资料

- [calico部署](https://docs.tigera.io/calico/3.27/getting-started/kubernetes/self-managed-onprem/onpremises)
- [监控 Calico 组件指标](https://docs.tigera.io/calico/3.27/operations/monitor/monitor-component-metrics)
- [Felix metrics](https://docs.tigera.io/calico/3.27/reference/felix/prometheus)
- [Typha metrics](https://docs.tigera.io/calico/3.27/reference/typha/prometheus)
