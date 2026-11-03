# cluster-api-provider-bke配置参数

## 常见配置场景速查

**表1** 常见配置场景速查表

| 配置场景 | 主要配置项 | 参考章节 |
|---|---|---|
| 通过`bke init --file/-f`编写BKECluster YAML。 | `metadata.name`、`spec.clusterConfig.*`、`spec.controlPlaneEndpoint.*` | 本文`BKECluster CRD配置`；命令行flag见[bkeadm配置参数](bkeadm_configuration.md)。 |
| 配置集群版本、镜像仓库、系统包源和NTP。 | `spec.clusterConfig.cluster.*`、`spec.clusterConfig.customExtra{}` | 本文`BKECluster CRD配置`、`customExtra`。 |
| 配置addon及其参数。 | `spec.clusterConfig.addons[]`、`addons[].param` | 本文`addons[].param`。 |
| 声明目标节点和SSH连接信息。 | `metadata.name`、`spec.ip`、`spec.username`、`spec.password`、`spec.role[]` | 本文`BKENode CRD`。 |
| 自定义kubelet/containerd二进制组件配置（KCT/CCT）。 | `spec.KubeletConfigRef`、`spec.clusterConfig.cluster.containerdConfigRef` | 本文`KCT与CCT配置与引用`；字段详解与示例见[二进制组件自定义配置](../service_management_cluster_configuration/binary_file_custom_configuration_guide.md)。 |
| 覆盖containerd服务、registry或脚本配置。 | `ContainerdConfig.spec.service.*`、`spec.main.*`、`spec.registry.*`、`spec.script.*` | 本文`ContainerdConfig CRD`。 |
| 覆盖kubelet配置、systemd service、额外文件或命令。 | `KubeletConfig.spec.kubeletConfig{}`、`spec.kubeletService.*`、`spec.files[]`、`spec.commands[]` | 本文`KubeletConfig CRD`。 |
| 调整控制器行为或排查升级流程。 | BKECluster注解、Cluster API注解、升级相关注解 | 本文`注解`。 |

## cluster-api-provider-bke


### BKEControllerManager运行参数


**表2** BKEControllerManager command配置项表

| 配置项名称 | 缺省值                                      | 说明                                                                                               | 设置方式 |
| --- |------------------------------------------|--------------------------------------------------------------------------------------------------| --- |
| `--metrics-bind-address` | `0`                                      | metrics侦听地址；部署YAML覆盖为`:8080`，非必填。                                                                | flag/YAML |
| `--health-probe-bind-address` | `8081`                                   | 健康检查侦听地址，非必填。                                                                                    | flag |
| `--health-probe-scheme` | `http`                                   | 健康检查协议，非必填；此处为代码默认值，支持`https`（启用后启动独立HTTPS服务，监听`--health-probe-port`端口，使用证书`/etc/kubernetes/tls-server.crt`/`tls-server.key`）；部署YAML可覆盖，如v1.4.3版本清单即覆盖为`https`。                                                                | flag |
| `--health-probe-port` | `9444`                                   | HTTPS健康检查端口，非必填。                                                                                 | flag |
| `--leader-elect` | `false`                                  | leader election开关；controller_manager_config.yaml中为true，非必填。                                      | flag/YAML |
| `--webhook-cert-dir` | `/tmp/k8s-webhook-server/serving-certs/` | webhook TLS证书目录；需与Deployment中bke-webhook-secret的挂载路径/tmp/k8s-webhook-server/serving-certs一致，非必填。 | flag |
| `--webhook-port` | `9443`                                   | webhook端口，非必填。                                                                                   | flag |
| `--webhook-host` | 无                                        | webhook host，非必填。                                                                                | flag |
| `--bke-cluster-concurrency` | `10`                                     | BKECluster并发reconcile数，非必填。                                                                      | flag |
| `--bke-machine-concurrency` | `10`                                     | BKEMachine并发reconcile数，非必填。                                                                      | flag |
| `--enable-internal-update` | `false`                                  | 内部更新开关，非必填。                                                                                      | flag |
| `--oci-digest-check-interval` | `300`                                    | UpgradePath OCI digest检查周期，秒，非必填。                                                                | flag |
| `--oci-registry-username` | 无                                        | OCI registry用户名，非必填。                                                                             | flag |
| `--oci-registry-password` | 无                                        | OCI registry密码，非必填。                                                                              | flag |
| `--oci-registry-insecure-skip-verify` | `true`                                   | OCI拉取跳过TLS校验，非必填。                                                                                | flag |
| `--enable-oci-digest-monitor` | `true`                                   | UpgradePath digest monitor开关，非必填。                                                                | flag |
| `--release-cache-dir` | `/var/lib/bke/release-cache`             | release image bundle本地缓存目录，非必填。                                                                  | flag/hostPath |
| `--helm-component-support` | `false`                                  | 全局启用yaml/helm组件执行路径；集群注解`cvo.openfuyao.cn/helm-component`存在时以注解为准，非必填。                           | flag/YAML |
| `--client-qps` | `50`                                     | Kubernetes API 客户端每秒请求数上限。                                                                       | flag |
| `--client-burst` | `100`                                    | Kubernetes API 客户端瞬时请求峰值。                                                                        | flag |
| `--client-config-file` | `/etc/bke/client-config.yaml`                                        | 客户端配置限流文件路径, 可配置 qps、burst。                                                                      | flag |


**表3** BKEControllerManager env配置项表

| 配置项名称 | 缺省值 | 说明 | 设置方式 |
| --- | --- | --- | --- |
| `env.container` | `true` | 标识容器运行，非必填。 | YAML |
| `env.OLOGGER_CONFIG` | `/etc/openFuyao/ologger/ologger.yaml` | 日志配置路径，非必填。 | YAML/env |
| `env.OLOGGER_PATH` | `/var/log/openFuyao/$(POD_NAME).log` | 日志文件路径，非必填。 | YAML/env |


<a id="bkecluster-crd配置"></a>

### BKECluster CRD配置

通过`bke init --file/-f <BKECluster YAML>`使用这些配置时，请同时参考[bkeadm配置参数](bkeadm_configuration.md)中`bke init`的flag说明。

> ![说明](../figures/icon-note.gif) **说明：**<br />
> - `BKECluster`（kubectl 短名 `bc`）描述待创建集群的版本、仓库、网络、addon、kubelet/containerd 引用等。常用命令：`kubectl get bc -A`、`kubectl edit bc -n <命名空间> <名称>`、`kubectl describe bc -n <命名空间> <名称>`。
> - 下表「缺省值」列为 openFuyao 平台官方推荐配置（CRD 为空时由控制器/安装器填入的默认值）。自定义部署时，建议在确认业务约束后，优先沿用推荐值，仅按需覆盖仓库地址、节点网络、addon 参数等环境相关项。
> - 关联资源：`KubeletConfig`（`kct`）、`ContainerdConfig`（`cct`）的字段与引用方式见本文后续章节及[二进制组件自定义配置](../service_management_cluster_configuration/binary_file_custom_configuration_guide.md)。

**表4** BKECluster CRD配置项表

| 配置项名称                                                      | 缺省值                                                                  | 说明                                                                                                                                                                     | 设置方式 |
|------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| `metadata.creationTimestamp`                               | 无                                                                    | 创建时间；通常由Kubernetes写入，用户配置可不填，非必填。                                                                                                                                      | CRD YAML |
| `metadata.name`                                            | `bke-cluster`                                                        | BKECluster资源名称，非必填。                                                                                                                                                    | CRD YAML |
| `metadata.namespace`                                       | `bke-cluster`                                                        | BKECluster所在命名空间，非必填。                                                                                                                                                  | CRD YAML |
| `spec.KubeletConfigRef.name`                               | `bke-kubelet`                                                        | 引用的KubeletConfig名称，非必填。                                                                                                                                                | CRD YAML |
| `spec.KubeletConfigRef.namespace`                          | `bke-kubelet`                                                        | 引用的KubeletConfig命名空间，非必填。                                                                                                                                              | CRD YAML |
| `spec.clusterConfig.addons[].name`                         | 无                                                                    | addon名称，必填。                                                                                                                                                            | CRD YAML |
| `spec.clusterConfig.addons[].version`                      | 无                                                                    | addon版本，必填。                                                                                                                                                            | CRD YAML |
| `spec.clusterConfig.addons[].param{}`                      | 无                                                                    | addon扩展参数（[detail](#221-addons)），非必填。                                                                                                                                  | CRD YAML |
| `spec.clusterConfig.addons[].block`                        | 无                                                                    | 是否阻塞等待addon安装完成，非必填。                                                                                                                                                   | CRD YAML |
| `spec.clusterConfig.cluster.agentHealthPort`               | 无                                                                    | bkeagent健康端口：1-65535，必填。                                                                                                                                               | CRD YAML |
| `spec.clusterConfig.cluster.apiServer.extraArgs{}`         | `authorization-mode=Node,RBAC`                                       | apiserver额外命令行参数，非必填。                                                                                                                                                  | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.certificatesDir`               | `/etc/kubernetes/pki`                                                | 证书目录，非必填。                                                                                                                                                              | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.chartRepo.domain`              | `cr.openfuyao.cn`                                                    | chart repo域名，非必填。                                                                                                                                                      | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.chartRepo.port`                | `443/40443`                                                          | chart repo访问端口；默认公网chart为443，非默认域名为40443，非必填；需要与实际chart repo服务端口一致，并避免与同一节点上的其他仓库服务端口冲突。                                                                               | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.chartRepo.prefix`              | `chart`                                                              | chart repo前缀，非必填。                                                                                                                                                      | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.containerRuntime.cri`          | `containerd`                                                         | CRI，支持`containerd`，非必填。                                                                                                                                                | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.containerRuntime.param{}`      | `cgroupDriver=systemd`<br>`containerd data-root=/var/lib/containerd` | 运行时扩展参数，如`data-root`、`cgroupDriver`，非必填。                                                                                                                               | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.containerRuntime.runtime`      | `runc`                                                               | OCI runtime，支持`runc`、`richrunc`、`kata`，非必填。                                                                                                                            | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.containerdConfigRef.name`      | `bke-containerd`                                                     | 引用的ContainerdConfig名称，必填。                                                                                                                                              | CRD YAML |
| `spec.clusterConfig.cluster.containerdConfigRef.namespace` | `bke-containerd`                                                     | 引用的ContainerdConfig命名空间，必填。                                                                                                                                            | CRD YAML |
| `spec.clusterConfig.cluster.containerdVersion`             | `v2.1.1`                                                             | containerd默认版本，非必填。                                                                                                                                                    | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.controllerManager.extraArgs{}` | 无                                                                    | controller-manager额外命令行参数，非必填。                                                                                                                                         | CRD YAML |
| `spec.clusterConfig.cluster.etcd.dataDir`                  | `/var/lib/openFuyao/etcd`                                            | etcd数据目录，非必填。                                                                                                                                                          | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.etcdVersion`                   | `v3.6.7-of.1`                                                        | etcd默认版本，非必填。                                                                                                                                                          | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.httpRepo.domain`               | `http.bocloud.k8s`                                                   | 系统包源域名，非必填。                                                                                                                                                            | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.httpRepo.ip`                   | 无                                                                    | 系统包源IP，必填。                                                                                                                                                             | CRD YAML |
| `spec.clusterConfig.cluster.httpRepo.port`                 | `40080`                                                              | 系统包源端口，非必填；与镜像仓库、chart repo是不同服务端口，自定义时需要彼此避开；离线部署时与`bke init`的`--yumRepoPort`为同一服务端口。                                                                                                                         | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.httpRepo.prefix`               | 无                                                                    | 系统包源前缀，必填。                                                                                                                                                             | CRD YAML |
| `spec.clusterConfig.cluster.imageRepo.domain`              | `deploy.bocloud.k8s`                                                 | 镜像仓库域名，非必填。                                                                                                                                                            | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.imageRepo.ip`                  | 无                                                                    | 镜像仓库IP，必填。                                                                                                                                                             | CRD YAML |
| `spec.clusterConfig.cluster.imageRepo.port`                | `40443`                                                              | 镜像仓库端口，非必填；与系统包源、chart repo是不同服务端口，自定义时需要彼此避开。                                                                                                                         | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.imageRepo.prefix`              | `kubernetes`                                                         | 镜像仓库前缀，非必填。                                                                                                                                                            | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.kubelet.extraArgs{}`           | 无                                                                    | kubelet额外命令行参数，非必填。                                                                                                                                                    | CRD YAML |
| `spec.clusterConfig.cluster.kubelet.extraVolumes.hostPath` | `/var/lib/kubelet`                                                   | kubelet额外挂载hostpath，非必填。                                                                                                                                               | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.kubelet.extraVolumes.name`     | `kubelet-root-dir`                                                   | kubelet额外挂载name，非必填。                                                                                                                                                   | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.kubernetesVersion`             | `v1.34.3-of.1`                                                       | Kubernetes默认版本，非必填。                                                                                                                                                    | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.networking.dnsDomain`          | `cluster.local`                                                      | Service DNS域名后缀，决定集群内服务的完整DNS名称格式。例如某个Service是：  kube-system/kube-dns默认完整域名就是：kube-dns.kube-system.svc.cluster.local，非必填。                                              | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.networking.podSubnet`          | `10.250.0.0/16`                                                      | Pod CIDR，非必填。                                                                                                                                                          | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.networking.serviceSubnet`      | `10.96.0.0/16`                                                       | Service CIDR，非必填。                                                                                                                                                      | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.ntpServer`                     | 无                                                                    | NTP服务器，必填。                                                                                                                                                             | CRD YAML |
| `spec.clusterConfig.cluster.openFuyaoVersion`              | `latest`                                                             | openFuyao默认版本，非必填。                                                                                                                                                     | CRD YAML，CRD为空时默认 |
| `spec.clusterConfig.cluster.scheduler.extraArgs{}`         | 无                                                                    | scheduler额外命令行参数，非必填。                                                                                                                                                  | CRD YAML |
| `spec.clusterConfig.customExtra{}`                         | 无                                                                    | 扩展参数，供模板/插件读取（[更详细的说明](#222-customextra)），非必填。                                                                                                                         | CRD YAML |
| `spec.controlPlaneEndpoint.host`                           | 无                                                                    | 控制面访问入口host，必填。                                                                                                                                                        | CRD YAML |
| `spec.controlPlaneEndpoint.port`                           | 无                                                                    | 控制面访问入口port，必填。                                                                                                                                                        | CRD YAML |
| `spec.pause`                                               | 无                                                                    | 暂停reconcile，非必填。                                                                                                                                                       | CRD YAML |
| `spec.reset`                                               | `false`                                                              | 设为 `true` 时触发该 `BKECluster` 的删除调谐流程。彻底清理目标节点时须同时将注解 `bke.bocloud.com/ignore-target-cluster-delete` 设为 `"false"`。操作步骤见[集群卸载](../../cluster_uninstallation/overview.md)。 | CRD YAML |

<a id="221-addons"></a>

#### addons[].param

bkeadm默认生成的addon：

**表5** bkeadm默认addon参数表

| addon name | 默认版本 | bkeadm默认param | 备注 |
|---|---|---|---|
| `kubeproxy` | Kubernetes版本对应目录，例如`v1.34.3-of.1` | `clusterNetworkMode` | 缺省值：`calico`。 |
| `calico` | `v3.31.3` | `calicoMode`<br>`ipAutoDetectionMethod`<br>`allowTypha`<br>`typhaReplicas` | 缺省值：<br>`calicoMode=vxlan`<br>`ipAutoDetectionMethod=skip-interface=nerdctl*`<br>`allowTypha=false`<br>`typhaReplicas=1`。 |
| `coredns` | `v1.12.2-of.1` | `EnableAntiAffinity` | `true/false` |
| `cluster-api` | `v1.4.3` | `manage`<br>`offline`<br>`sandbox`<br>`replicas`<br>`containerdVersion`<br>`openFuyaoVersion`<br>`manifestsVersion`<br>`providerVersion`<br>`ntpServer`<br>`healthPort` |  |
| `openfuyao-system-controller` | `latest` | `helmRepo`<br>`tagVersion` |  |

常用的addon名称和参数：

**表6** 常用addon参数表

| addon name | 采用版本 | 组件专属/可覆盖param key | 备注 |
|---|---|---|---|
| `nodelocaldns` | `v1.26.4` | `clusterDNS`<br>`DNSserver`<br>`domain`<br>`localdns` | 注意是`domain`，不是`dnsDomain`。 |
| `numa-affinity-package` | `v0.0.2` | `openfuyaoRepo` | NUMA affinity。 |
| `redis` | `6.1.12` | `nodeselector` | key名按模板小写。 |
| `victoriametrics-controller` | `latest` | `grafanaNodePort`<br>`useVMSingle`<br>`kubeStateMetricsAutoSharding`<br>`kubeStateMetricsCpuCount`<br>`kubeStateMetricsMemorySize`<br>`kubeStateMetricsReplicaCount`<br>`vmAgentAllowStatefulSet`<br>`vmAgentCpuCount`<br>`vmAgentMemorySize`<br>`vmAgentReplicaCount`<br>`vmAgentScrapeInterval`<br>`vmAgentShareCount`<br>`vmAgentStorageSize`<br>`vmAlertCpuCount`<br>`vmAlertMemorySize`<br>`vmAlertReplicaCount`<br>`vmAlertManagerCpuCount`<br>`vmAlertManagerMemorySize`<br>`vmAlertManagerReplicaCount`<br>`vmClusterReplicationFactor`<br>`vmClusterRetentionPeriod`<br>`vmInsertCpuCount`<br>`vmInsertMemorySize`<br>`vmInsertReplicaCount`<br>`vmSelectCpuCount`<br>`vmSelectMemorySize`<br>`vmSelectReplicaCount`<br>`vmSelectStorageSize`<br>`vmSingleStorageSize`<br>`vmStorageCPUCount`<br>`vmStorageMemorySize`<br>`vmStorageReplicaCount`<br>`vmStorageStorageSize` | webhook会为缺失项补缺省值，并校验部分replica/share关系。 |

<a id="222-customextra"></a>

#### `spec.clusterConfig.customExtra{}`

**表7** customExtra配置项表

| 配置项名称 | 缺省值                                      | 说明 | 设置方式 |
| --- |------------------------------------------| --- | --- |
| `customExtra.chartRepoPort` | `38080`                                  | 本地chart repo端口，非必填；与`customExtra.imageRepoPort`、`customExtra.yumRepoPort`分别对应不同本地服务，自定义时需要彼此避开。 | CLI派生/YAML |
| `customExtra.clusterapi` | `latest`                                 | cluster-api-provider-bke版本，非必填。 | CLI派生/YAML |
| `customExtra.containerd` | `containerd-v2.1.1-linux-{.arch}.tar.gz` | containerd安装包模板名，非必填。 | 版本物料/YAML |
| `customExtra.domain` | `deploy.bocloud.k8s`                     | 镜像仓库域名，非必填。 | CLI派生/YAML |
| `customExtra.host` | 无                                        | 启动节点IP，非必填。 | 生成环境/YAML |
| `customExtra.imageRepoPort` | `40443`                                  | 本地镜像仓库端口，非必填；与chart repo、yum/http repo端口相互独立，自定义时需要彼此避开。 | CLI派生/YAML |
| `customExtra.nfsserverpath` | `/`                                      | NFS服务路径，非必填。 | 代码默认/YAML |
| `customExtra.onlineImage` | 无                                        | 在线安装镜像，非必填。 | CLI派生/YAML |
| `customExtra.otherChart` | 无                                        | 私有chart repo地址，非必填。 | CLI派生/YAML |
| `customExtra.otherChartIp` | 无                                        | 私有chart repo解析IP，非必填。 | 生成逻辑/YAML |
| `customExtra.otherRepo` | 无                                        | 私有镜像仓库，非必填。 | CLI派生/YAML |
| `customExtra.otherRepoIp` | 无                                        | 私有镜像仓库解析IP，非必填。 | 生成逻辑/YAML |
| `customExtra.otherSource` | 无                                        | 私有系统包源，非必填。 | CLI派生/YAML |
| `customExtra.yumRepoPort` | `40080`                                  | 本地系统包源（HTTP 仓）服务端口，非必填；与镜像仓库、chart repo端口相互独立，自定义时需要彼此避开；与Cluster CRD的`spec.clusterConfig.cluster.httpRepo.port`为同一服务端口。 | CLI派生/YAML |

端口关系与冲突检查：`customExtra.imageRepoPort`、`customExtra.chartRepoPort`、`customExtra.yumRepoPort`通常由`bke init`的`--imageRepoPort`、`--chartRepoPort`、`--yumRepoPort`派生，分别对应本地镜像仓库、chart repo、yum/http repo三个独立服务。自定义时需要配置为不同端口，并避开Kubernetes API、console端口和宿主机已占用端口。`bke init`预检会检查端口是否已被宿主机占用，并过滤已存在的BKE管理容器端口；但不会显式检查这些参数之间是否重复。如果端口重复，可能在服务启动或端口发布阶段失败，需要根据日志中的端口占用信息调整后重试。


### BKENode CRD


**表8** BKENode CRD配置项表

| 配置项名称 | 缺省值           | 说明 | 设置方式 |
|---|---------------|---|---|
| `metadata.creationTimestamp` | 无 | 创建时间；通常由Kubernetes写入，用户配置可不填，非必填。 | CRD YAML |
| `metadata.labels.cluster.x-k8s.io/cluster-name` | `bke-cluster`  | 关联的Cluster API集群名称，非必填。 | CRD YAML |
| `metadata.name` | 无 | BKENode资源名称，必填。 | CRD YAML |
| `metadata.namespace` | `bke-cluster` | BKENode所在命名空间，非必填。 | CRD YAML |
| `spec.hostname` | 无 | 节点主机名，必填。 | CRD YAML |
| `spec.ip` | 无 | 节点IP，必填。 | CRD YAML |
| `spec.password` | 无 | SSH密码，必填。 | CRD YAML |
| `spec.port` | `22`          | SSH端口，示例为`"22"`，非必填。 | CRD YAML |
| `spec.role[]` | 无 | 节点角色，例如`master/node`、`etcd`、`node`，必填。 | CRD YAML |
| `spec.username` | `root`        | SSH用户名，非必填。 | CRD YAML |
| `spec.controlPlane` | 无 | 节点级控制面组件覆盖配置，非必填。 | CRD YAML |
| `spec.kubelet` | 无 | 节点级kubelet覆盖配置，非必填。 | CRD YAML |
| `spec.labels[]` | 无 | 节点级标签覆盖配置，非必填。 | CRD YAML |

### ContainerdConfig CRD

**表9** ContainerdConfig CRD配置项表

| 配置项名称 | 缺省值 | 说明 | 设置方式 |
|---|---|---|---|
| `spec.configType` | `combined` | 配置类型，支持`service/main/registry/combined`，非必填。 | CRD YAML |
| `spec.description` | 无 | 人类可读说明，必填。 | CRD YAML |
| `spec.service.execStart` | 无 | systemd ExecStart，必填。 | CRD YAML |
| `spec.service.slice` | `system.slice` | systemd slice，非必填。 | CRD YAML |
| `spec.service.killMode` | `process` | KillMode，支持`control-group/process/mixed/none`，非必填。 | CRD YAML |
| `spec.service.restart` | `always` | Restart策略，非必填。 | CRD YAML |
| `spec.service.restartSec` | `5s` | 重启等待，非必填。 | CRD YAML |
| `spec.service.startLimitInterval` | `10s` | 启动限流窗口，非必填。 | CRD YAML |
| `spec.service.startLimitBurst` | `5` | 启动限流次数，非必填。 | CRD YAML |
| `spec.service.timeoutStopSec` | `90s` | 停止超时，非必填。 | CRD YAML |
| `spec.service.logging.standardOutput` | `journal` | stdout日志目标，非必填。 | CRD YAML |
| `spec.service.logging.standardError` | `journal` | stderr日志目标，非必填。 | CRD YAML |
| `spec.service.logging.syslogIdentifier` | 无 | syslog标识，必填。 | CRD YAML |
| `spec.service.logging.logLevelMax` | 无 | 最大日志级别，非必填。 | CRD YAML |
| `spec.service.customExtra{}` | 无 | service模板扩展变量，非必填。 | CRD YAML |
| `spec.main.metricsAddress` | 无 | containerd metrics地址，必填。 | CRD YAML |
| `spec.main.root` | `/var/lib/containerd` | containerd root，非必填。 | CRD YAML |
| `spec.main.state` | `/run/containerd` | containerd state，非必填。 | CRD YAML |
| `spec.main.sandboxImage` | `registry.k8s.io/pause:3.9` | pause镜像，非必填。 | CRD YAML |
| `spec.main.configPath` | `/etc/containerd/certs.d` | registry配置目录，非必填。 | CRD YAML |
| `spec.main.rawTOML` | 无 | 原始TOML，提供后优先使用，非必填。 | CRD YAML |
| `spec.registry.configPath` | `/etc/containerd/certs.d` | registry hosts配置目录，非必填。 | CRD YAML |
| `spec.registry.configs{host}` | 无 | registry host配置映射，非必填。 | CRD YAML |
| `registryHost.host` | 无 | 单个registry host地址，必填。 | CRD YAML |
| `registryHost.capabilities` | 无 | 单个registry host能力，非必填。 | CRD YAML |
| `registryHost.skipVerify` | 无 | 单个registry host跳过校验，非必填。 | CRD YAML |
| `registryHost.plainHTTP` | 无 | 单个registry host使用plain HTTP，非必填。 | CRD YAML |
| `registryHost.insecure` | 无 | 单个registry host insecure设置，非必填。 | CRD YAML |
| `registryHost.tls.caFile` | 无 | TLS CA文件，必填。 | CRD YAML |
| `registryHost.tls.certFile` | 无 | TLS cert文件，必填。 | CRD YAML |
| `registryHost.tls.keyFile` | 无 | TLS key文件，必填。 | CRD YAML |
| `registryHost.tls.insecureSkipVerify` | 无 | TLS跳过证书校验，非必填。生产环境禁止设置为`true`，仅允许在临时排障或隔离测试环境短时使用。 | CRD YAML |
| `registryHost.auth.username` | 无 | registry认证用户名，必填。 | CRD YAML |
| `registryHost.auth.password` | 无 | registry认证密码，必填。 | CRD YAML |
| `registryHost.auth.auth` | 无 | registry auth字段，非必填。 | CRD YAML |
| `registryHost.auth.identityToken` | 无 | registry identity token，非必填。 | CRD YAML |
| `registryHost.auth.registryToken` | 无 | registry token，非必填。 | CRD YAML |
| `registryHost.header` | 无 | 附加header，非必填。 | CRD YAML |
| `registryHost.overridePath` | 无 | path override，必填。 | CRD YAML |
| `spec.script.content` | 无 | shell脚本内容，非必填。 | CRD YAML |
| `spec.script.path` | 无 | shell脚本路径，非必填。 | CRD YAML |
| `spec.script.args` | 无 | shell脚本参数，非必填。 | CRD YAML |
| `spec.script.interpreter` | `/bin/bash` | shell脚本解释器，非必填。 | CRD YAML |

### KubeletConfig CRD

**表10** KubeletConfig CRD配置项表

| 配置项名称 | 缺省值 | 说明 | 设置方式 |
|---|---|---|---|
| `spec.kubeletConfig{}` | 无 | kubelet原生配置map，RawExtension，非必填。 | CRD YAML |
| `spec.kubeletService.enabled` | 无 | 是否创建kubelet service，非必填。 | CRD YAML |
| `spec.kubeletService.serviceName` | 无 | systemd service名称，必填。 | CRD YAML |
| `spec.kubeletService.unit.description` | 无 | Unit描述，非必填。 | CRD YAML |
| `spec.kubeletService.unit.documentation` | 无 | Unit文档，非必填。 | CRD YAML |
| `spec.kubeletService.unit.after` | 无 | Unit After依赖，非必填。 | CRD YAML |
| `spec.kubeletService.unit.wants` | 无 | Unit Wants依赖，非必填。 | CRD YAML |
| `spec.kubeletService.unit.requires[]` | 无 | Unit Requires依赖，非必填。 | CRD YAML |
| `spec.kubeletService.service.execStart` | 无 | kubelet启动命令，必填。 | CRD YAML |
| `spec.kubeletService.service.restart` | 无 | service重启策略，必填。 | CRD YAML |
| `spec.kubeletService.service.startLimitInterval` | 无 | service启动限流窗口，必填。 | CRD YAML |
| `spec.kubeletService.service.restartSec` | 无 | service重启等待时间，必填。 | CRD YAML |
| `spec.kubeletService.service.environment` | 无 | service环境变量，非必填。 | CRD YAML |
| `spec.kubeletService.service.environmentFile` | 无 | service环境变量文件，非必填。 | CRD YAML |
| `spec.kubeletService.service.execStartPre` | 无 | service前置启动命令，非必填。 | CRD YAML |
| `spec.kubeletService.service.startLimitBurst` | 无 | service启动限流次数，必填。 | CRD YAML |
| `spec.kubeletService.service.killMode` | 无 | service KillMode，必填。 | CRD YAML |
| `spec.kubeletService.service.standardOutput` | 无 | stdout日志输出设置，非必填。 | CRD YAML |
| `spec.kubeletService.service.standardError` | 无 | stderr日志输出设置，非必填。 | CRD YAML |
| `spec.kubeletService.service.syslogIdentifier` | 无 | syslog标识，必填。 | CRD YAML |
| `spec.kubeletService.service.workingDirectory` | 无 | service工作目录，非必填。 | CRD YAML |
| `spec.kubeletService.service.user` | 无 | service运行用户，非必填。 | CRD YAML |
| `spec.kubeletService.service.group` | 无 | service运行用户组，非必填。 | CRD YAML |
| `spec.kubeletService.service.customExtra` | 无 | service模板扩展变量，非必填。 | CRD YAML |
| `spec.kubeletService.install.wantedBy` | 无 | Install WantedBy配置，非必填。 | CRD YAML |
| `spec.kubeletService.install.requiredBy[]` | 无 | Install RequiredBy配置，非必填。 | CRD YAML |
| `spec.files[].path` | 无 | 额外文件路径，非必填。 | CRD YAML |
| `spec.files[].content` | 无 | 额外文件内容，非必填。 | CRD YAML |
| `spec.files[].permissions` | 无 | 额外文件权限，非必填。 | CRD YAML |
| `spec.files[].owner` | 无 | 额外文件属主，非必填。 | CRD YAML |
| `spec.commands[].command` | 无 | 额外命令，非必填。 | CRD YAML |
| `spec.commands[].args` | 无 | 额外命令参数，非必填。 | CRD YAML |
| `spec.commands[].workingDir` | 无 | 额外命令工作目录，非必填。 | CRD YAML |

### KCT与CCT配置与引用

KCT（KubeletConfig）和CCT（ContainerdConfig）是管理集群上的自定义资源，用于在节点引导阶段下发kubelet与containerd配置，无需登录节点手工编辑文件。

**表11** KCT与CCT对照表

| 缩写 | CR类型 | API资源名 | kubectl短名 | BKECluster引用字段 |
| --- | --- | --- | --- | --- |
| KCT | `KubeletConfig` | `kubeletconfigs` | `kct` | `spec.KubeletConfigRef.name` / `namespace` |
| CCT | `ContainerdConfig` | `containerdconfigs` | `cct` | `spec.clusterConfig.cluster.containerdConfigRef.name` / `namespace` |

**使用流程**

1. 在管理集群（与BKECluster同集群）创建KCT/CCT资源，填写`spec`配置。
2. 在BKECluster YAML中通过引用字段指向对应资源；`bke init`生成的默认集群YAML已包含示例引用。
3. 执行`bke create cluster`后，bkeagent在节点引导阶段读取并应用配置：
   - **KCT**：当`KubeletConfigRef`非空时，bkeagent从管理集群拉取KubeletConfig，依次处理`spec.files[]`、`spec.kubeletConfig`（写入`/var/lib/kubelet/config.yaml`）、`spec.kubeletService`（生成`kubelet.service`）和`spec.commands[]`；同时自动启用变量替换（`${EXPR|command|END}`）。
   - **CCT**：当`containerdConfigRef`非空时，bkeagent以`namespace:name`形式传入InstallContainerd插件，按`spec.script` → `spec.service` → `spec.main` → `spec.registry`顺序渲染containerd配置并重启服务。

**KCT/CCT配置示例（BKECluster片段）**

```yaml
spec:
  KubeletConfigRef:
    name: bke-kubelet
    namespace: bke-kubelet
  clusterConfig:
    cluster:
      containerdConfigRef:
        name: bke-containerd
        namespace: bke-containerd
```

完整字段说明、变量替换示例和YAML样例见[二进制组件自定义配置](../service_management_cluster_configuration/binary_file_custom_configuration_guide.md)。

## 注解

### 可配置的注解
这些注解作用于BKECluster行为

**表12** 可配置注解表

| key | 语义与取值                     | 缺省值和优先级                            |
|---|---------------------------|------------------------------------|
| **bke.bocloud.com/retry** | 空值重试全部节点；非空值为逗号分隔的节点IP。   | key存在即触发；消费完成后删除。                  |
| **bke.bocloud.com/deep-restore-node** | true开启节点初始化和reset的深度恢复。   | 默认true；注解优先，缺失时也按true，并默认写入true。 |
| **bke.bocloud.com/ignore-namespace-delete** | false才允许删除namespace；其他值均跳过。 | 默认true；只有精确false才进入删除分支。         |
| **bke.bocloud.com/ignore-target-cluster-delete** | true表示删除管理面对象时不reset目标集群。 | 默认true；只有精确false才继续清理目标节点。彻底卸载步骤见[集群卸载总览](../../cluster_uninstallation/overview.md)。 |
| **bke.bocloud.com/node-boot-wait-timeout** | Go duration，例如30s、10m、1h。 | 默认10m；合法注解优先，缺失或解析失败回退10m。        |

关于retry注解使用的场景示例：

1、在EnsureBKEAgent阶段推送Agent失败。

2、在EnsureBKEAgent阶段推送Agent成功，但是对应节点上的bkeagent启动失败。

3、对应节点上安装kubelet失败。

4、对应节点上的域名解析服务失败导致镜像拉取失败。

遇到上述场景，用户可以在手动修复后执行如下命令添加retry注解，使失败节点重新执行后续阶段。
```bash
# 注意：当retry注解的值为空值时，会重试所有节点。
kubectl annotate bc -n <集群命名空间> <集群名称>  bke.bocloud.com/retry="<失败节点ip>"
```

### 不支持配置的注解
#### BKE内部状态、流程和运维注解（13个）

**表13** BKE内部状态、流程和运维注解表

| key | 语义与取值 | 优先级和生命周期 |
| --- | --- | --- |
| **bke.bocloud.com/collectd** | 逗号分隔的base、agent，表示基础信息和agent信息已采集。 | 控制器保留已有token，只追加缺失token。 |
| **bke.bocloud.com/last-update-configuration** | 序列化后的对象JSON，作为上次配置快照。 | 内部三方合并基线，不参与普通用户覆盖链。 |
| **bke.bocloud.com/status-record** | 请求StatusManager记录一次当前状态，通常为空值。 | key存在即触发，记录后删除。 |
| **bke.bocloud.com/cluster-tracker-healthy-check-failed** | 标记ClusterTracker健康检查失败并触发重新调谐。 | key存在即生效；恢复Ready后删除。 |
| **bke.bocloud.com/addon-boot-wait-timeout** | 设计上表示addon启动等待超时。 | 尚无生产读取或写入点，当前设置不会生效。 |
| **bke.bocloud.com/event** | 标记这是BKE普通事件。 | 每次AnnotatedEventf创建Event时写入空值。 |
| **bke.bocloud.com/complete** | 与event一起标记某次流程完成事件。 | 完成事件创建时写入空值。 |
| **bke.bocloud.com/bkeagent-listener** | current表示agent侦听当前集群；bkecluster表示待切换到BKECluster。 | 缺失默认写current；addon阶段写bkecluster，切换完成后流程恢复current。 |
| **bke.bocloud.com/cluster-from** | bke、bocloud、other表示集群来源。 | 缺失或空值按bke；不同来源影响配置校验、完全管理和证书分发分支。 |
| **bke.bocloud.com/cluster-api-manager-applied** | true表示004-manage.yaml已应用。 | 只有目标版本存在、postprocess完成且该值不是true时才执行；成功后写true。 |
| **etcd-cert-dir** | 暂存采集到的etcd证书目录。 | 优先于Kubernetes certificatesDir；agent信息收集完成后删除。 |
| **deployAction** | 固定写k8s_upgrade，供BOC兼容流程识别。 | 缺失或值不是k8s_upgrade时覆盖为k8s_upgrade；代码中未见自动清理。 |
| **bke.openfuyao.cn/restartedAt** | RFC3339时间戳，用于改变PodTemplate并触发Deployment滚动重启。 | 每次PatchDeploymentImage更新为当前时间。 |


#### ClusterVersion、升级路径和ReleaseImage注解（6个）
这些key是升级控制器之间的协议。参数说明与协作关系见[后端声明式升级指导](../../upgrade_guide/backend_declarative_upgrade_guide.md)#控制器参数与升级注解。

**表14** ClusterVersion、升级路径和ReleaseImage注解表

| key | 语义与取值 | 优先级和生命周期 |
| --- | --- | --- |
| **cvo.openfuyao.cn/upgrade-ready** | 当前升级hop的目标版本，也是声明式升级DAG的执行门闩。 | trim后非空即允许DAG执行；其值是当前hop，不一定是最终desiredVersion。 |
| **cvo.openfuyao.cn/cluster-version** | 关联的ClusterVersion对象名称。 | 与upgrade-ready同时写入和清理，不单独作为门闩。 |
| **cvo.openfuyao.cn/upgrade-path** | 已选升级路径，例如v1->v2,v2->v3。 | 与upgrade-ready同时写入和清理。 |
| **cvo.openfuyao.cn/helm-component** | trim后忽略大小写等于true时，为该集群启用yaml/helm组件执行器。 | 注解存在时以注解值为准；缺失时回退到`--helm-component-support`全局flag。 |
| **config.openfuyao.com/oci-digest** | 保存UpgradePath OCI镜像digest，并同步到status.lastDigest。 | 每次从OCI刷新路径时覆盖为最新digest。 |
| **cvo.openfuyao.cn/bkecluster-name** | 关联ReleaseImage与目标BKECluster名称。 | ReleaseImage控制器读取，用于解析和同步目标集群版本信息。 |

升级完成、取消或清理时，控制器会一起移除upgrade-ready、cluster-version和upgrade-path。

#### Cluster API标准注解（8个）
这些key的语义主要由上游Cluster API定义。本仓库负责设置、删除或消费它们，因此也属于实际注解面。

**表15** Cluster API标准注解表

| key | 语义与取值 | 优先级和生命周期 |
| --- | --- | --- |
| **cluster.x-k8s.io/cluster-name** | 指明节点所属Cluster名称。 | 与cluster-namespace同时存在才会映射到对应BKECluster。 |
| **cluster.x-k8s.io/cluster-namespace** | 指明节点所属Cluster namespace。 | 缺失时不进行对象映射。 |
| **cluster.x-k8s.io/paused** | key存在表示暂停Cluster API对象调谐。 | Cluster.spec.paused=true或对象存在此key均可暂停；**value不参与判断**。 |
| **cluster.x-k8s.io/delete-machine** | 将带注解的机器设为优先删除候选。 | key存在即生效，通常写空值。 |
| **cluster.x-k8s.io/managed-by** | 表示基础设施对象由外部系统管理，本控制器跳过映射。 | key存在即视为externally managed。 |
| **machine.cluster.x-k8s.io/certificates-expiry** | RFC3339证书到期时间；当前写为当前时间加100年。 | 控制器生成，用于避免上游因短证书周期触发不必要行为。 |
| **controlplane.cluster.x-k8s.io/skip-kube-proxy** | 要求KCP不负责升级kube-proxy。 | 上游按key是否存在判断；BKE自己管理相关组件。 |
| **controlplane.cluster.x-k8s.io/skip-coredns** | 要求KCP不负责升级CoreDNS。 | 上游按key是否存在判断；BKE自己管理相关组件。 |

#### kubeadm、Kubernetes和静态Pod注解（7个）

**表16** kubeadm、Kubernetes和静态Pod注解表

| key | 语义与取值 | 优先级和生命周期 |
| --- | --- | --- |
| **kubeadm.kubernetes.io/etcd.advertise-client-urls** | 通常为节点IP:2379，描述etcd client advertise URL。 | 纳管阶段key已存在就不覆盖；master升级阶段缺失或空值时补齐。 |
| **bkeagent.bocloud.com/etcd.advertise-client-urls** | 设计上也是etcd advertise client URL。 | 当前生产代码实际使用上面的kubeadm key。 |
| **kubeadm.alpha.kubernetes.io/cri-socket** | 保存init/join时使用的CRI socket。 | 本仓库只有常量定义，没有生产读写点。 |
| **kubeadm.kubernetes.io/kube-apiserver.advertise-address.endpoint** | 描述API Server advertise address和端口。 | 本仓库只有常量定义，没有生产读写点。 |
| **kubeadm.kubernetes.io/component-config.hash** | 保存组件配置SHA256，用于检测用户修改。 | 本仓库只有常量定义，没有生产读写点。 |
| **kubernetes.io/service-account.name** | 指明该Secret属于哪个ServiceAccount，供Kubernetes token controller填充token。 | 创建Secret时固定为目标ServiceAccount名称。 |
| **kubernetes.io/config.hash** | 静态Pod manifest内容哈希，用于判断组件Pod是否已更新。 | BKE只读取并等待hash变化，不写入。 |


#### 部署清单和生成工具注解（5个）

**表17** 部署清单和生成工具注解表

| key | 作用 | 配置和校验 |
| --- | --- | --- |
| **controller-gen.kubebuilder.io/version** | 记录生成CRD所用controller-gen版本。 | 由controller-gen生成，仅用于溯源，不控制运行时业务逻辑。 |
| **cert-manager.io/inject-ca-from** | 告诉cert-manager从namespace/name指定的Certificate/Secret来源注入caBundle。 | config中含Kustomize变量，构建后变成实际namespace/name；cert-manager未部署或注入器未运行时不会生效。 |
| **kubectl.kubernetes.io/default-container** | kubectl logs/exec未指定容器时默认选择manager。 | 固定字符串manager；仅影响kubectl使用体验。 |
| **prometheus.io/port** | 告诉基于Prometheus注解发现的采集器使用8080端口。 | 值是字符串8080；采集系统不启用注解发现时不会生效。 |
| **prometheus.io/scrape** | true表示允许基于注解发现的Prometheus抓取指标。 | 由采集系统解释，Kubernetes本身不校验。 |

