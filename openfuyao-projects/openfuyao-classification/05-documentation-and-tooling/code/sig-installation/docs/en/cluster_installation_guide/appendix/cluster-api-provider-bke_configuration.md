# cluster-api-provider-bke Configuration Parameters

## Common Configuration Scenarios Quick Reference

**Table 1** Common Configuration Scenarios Quick Reference Table

| Configuration Scenario | Main Configuration Items | Reference Section |
|---|---|---|
| Write BKECluster YAML via `bke init --file/-f`. | `metadata.name`, `spec.clusterConfig.*`, `spec.controlPlaneEndpoint.*` | This article's `BKECluster CRD Configuration`; for command-line flags see [bkeadm Configuration Parameters](bkeadm_configuration.md). |
| Configure cluster version, image repository, system package source, and NTP. | `spec.clusterConfig.cluster.*`, `spec.clusterConfig.customExtra{}` | This article's `BKECluster CRD Configuration`, `customExtra`. |
| Configure addons and their parameters. | `spec.clusterConfig.addons[]`, `addons[].param` | This article's `addons[].param`. |
| Declare target nodes and SSH connection information. | `metadata.name`, `spec.ip`, `spec.username`, `spec.password`, `spec.role[]` | This article's `BKENode CRD`. |
| Customize kubelet/containerd binary component configuration (KCT/CCT). | `spec.KubeletConfigRef`, `spec.clusterConfig.cluster.containerdConfigRef` | This article's `KCT and CCT Configuration and References`; for field details and examples see [Binary Component Custom Configuration](../service_management_cluster_configuration/binary_file_custom_configuration_guide.md). |
| Override containerd service, registry, or script configuration. | `ContainerdConfig.spec.service.*`, `spec.main.*`, `spec.registry.*`, `spec.script.*` | This article's `ContainerdConfig CRD`. |
| Override kubelet configuration, systemd service, extra files, or commands. | `KubeletConfig.spec.kubeletConfig{}`, `spec.kubeletService.*`, `spec.files[]`, `spec.commands[]` | This article's `KubeletConfig CRD`. |
| Adjust controller behavior or troubleshoot the upgrade process. | BKECluster annotations, Cluster API annotations, upgrade-related annotations | This article's `Annotations`. |

## cluster-api-provider-bke


### BKEControllerManager Runtime Parameters


**Table 2** BKEControllerManager command Configuration Items Table

| Configuration Item Name | Default Value                                      | Description                                                                                               | Setting Method |
| --- |------------------------------------------|--------------------------------------------------------------------------------------------------| --- |
| `--metrics-bind-address` | `0`                                      | metrics listen address; overridden to `:8080` in deployment YAML, optional.                                                                | flag/YAML |
| `--health-probe-bind-address` | `8081`                                   | Health check listen address, optional.                                                                                    | flag |
| `--health-probe-scheme` | `http`                                   | Health check protocol, optional; this is the code default value, supports `https` (when enabled, starts a separate HTTPS service listening on `--health-probe-port` port, using certificates `/etc/kubernetes/tls-server.crt`/`tls-server.key`); deployment YAML can override, e.g., v1.4.3 manifest overrides to `https`.                                                                                      | flag |
| `--health-probe-port` | `9444`                                   | HTTPS health check port, optional.                                                                                 | flag |
| `--leader-elect` | `false`                                  | Leader election switch; set to true in controller_manager_config.yaml, optional.                                      | flag/YAML |
| `--webhook-cert-dir` | `/tmp/k8s-webhook-server/serving-certs/` | Webhook TLS certificate directory; must match the mount path /tmp/k8s-webhook-server/serving-certs of bke-webhook-secret in the Deployment, optional. | flag |
| `--webhook-port` | `9443`                                   | Webhook port, optional.                                                                                   | flag |
| `--webhook-host` | None                                        | Webhook host, optional.                                                                                | flag |
| `--bke-cluster-concurrency` | `10`                                     | Number of concurrent BKECluster reconciles, optional.                                                                      | flag |
| `--bke-machine-concurrency` | `10`                                     | Number of concurrent BKEMachine reconciles, optional.                                                                      | flag |
| `--enable-internal-update` | `false`                                  | Internal update switch, optional.                                                                                      | flag |
| `--oci-digest-check-interval` | `300`                                    | UpgradePath OCI digest check interval, in seconds, optional.                                                                | flag |
| `--oci-registry-username` | None                                        | OCI registry username, optional.                                                                             | flag |
| `--oci-registry-password` | None                                        | OCI registry password, optional.                                                                              | flag |
| `--oci-registry-insecure-skip-verify` | `true`                                   | Skip TLS verification when pulling from OCI, optional.                                                                                | flag |
| `--enable-oci-digest-monitor` | `true`                                   | UpgradePath digest monitor switch, optional.                                                                | flag |
| `--release-cache-dir` | `/var/lib/bke/release-cache`             | Local cache directory for release image bundle, optional.                                                                  | flag/hostPath |
| `--helm-component-support` | `false`                                  | Globally enable yaml/helm component execution path; when cluster annotation `cvo.openfuyao.cn/helm-component` exists, the annotation takes precedence, optional.                           | flag/YAML |
| `--client-qps` | `50`                                     | Maximum requests per second for the Kubernetes API client.                                                                       | flag |
| `--client-burst` | `100`                                    | Burst request peak for the Kubernetes API client.                                                                        | flag |
| `--client-config-file` | `/etc/bke/client-config.yaml`                                        | Client configuration rate-limiting file path; can configure qps and burst.                                                                      | flag |


**Table 3** BKEControllerManager env Configuration Items Table

| Configuration Item Name | Default Value | Description | Setting Method |
| --- | --- | --- | --- |
| `env.container` | `true` | Indicates running in a container, optional. | YAML |
| `env.OLOGGER_CONFIG` | `/etc/openFuyao/ologger/ologger.yaml` | Logging configuration path, optional. | YAML/env |
| `env.OLOGGER_PATH` | `/var/log/openFuyao/$(POD_NAME).log` | Log file path, optional. | YAML/env |


<a id="bkecluster-crd-configuration"></a>

### BKECluster CRD Configuration

When using these configurations via `bke init --file/-f <BKECluster YAML>`, also refer to the `bke init` flag descriptions in [bkeadm Configuration Parameters](bkeadm_configuration.md).

> ![icon](../figures/icon-note.gif) **Note:**<br />
> - `BKECluster` (kubectl short name `bc`) describes the version, repository, network, addons, kubelet/containerd references, etc. of the cluster to be created. Common commands: `kubectl get bc -A`, `kubectl edit bc -n <namespace> <name>`, `kubectl describe bc -n <namespace> <name>`.
> - The "Default Value" column in the table below lists the officially recommended configurations for the openFuyao platform (default values filled in by the controller/installer when the CRD is empty). For custom deployments, it is recommended to first follow the recommended values after confirming business constraints, and only override environment-related items such as repository addresses, node networks, and addon parameters as needed.
> - Related resources: For the fields and reference methods of `KubeletConfig` (`kct`) and `ContainerdConfig` (`cct`), see the subsequent sections of this article and [Binary Component Custom Configuration](../service_management_cluster_configuration/binary_file_custom_configuration_guide.md).

**Table 4** BKECluster CRD Configuration Items Table

| Configuration Item Name                                                      | Default Value                                                                  | Description                                                                                                                                                                     | Setting Method |
|------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| `metadata.creationTimestamp`                               | None                                                                    | Creation time; usually written by Kubernetes, can be left empty in user configuration, optional.                                                                                                                                      | CRD YAML |
| `metadata.name`                                            | `bke-cluster`                                                        | BKECluster resource name, optional.                                                                                                                                                    | CRD YAML |
| `metadata.namespace`                                       | `bke-cluster`                                                        | Namespace where BKECluster resides, optional.                                                                                                                                                  | CRD YAML |
| `spec.KubeletConfigRef.name`                               | `bke-kubelet`                                                        | Name of the referenced KubeletConfig, optional.                                                                                                                                                | CRD YAML |
| `spec.KubeletConfigRef.namespace`                          | `bke-kubelet`                                                        | Namespace of the referenced KubeletConfig, optional.                                                                                                                                              | CRD YAML |
| `spec.clusterConfig.addons[].name`                         | None                                                                    | Addon name, required.                                                                                                                                                            | CRD YAML |
| `spec.clusterConfig.addons[].version`                      | None                                                                    | Addon version, required.                                                                                                                                                            | CRD YAML |
| `spec.clusterConfig.addons[].param{}`                      | None                                                                    | Addon extended parameters ([detail](#221-addons)), optional.                                                                                                                                  | CRD YAML |
| `spec.clusterConfig.addons[].block`                        | None                                                                    | Whether to block and wait for addon installation to complete, optional.                                                                                                                                                   | CRD YAML |
| `spec.clusterConfig.cluster.agentHealthPort`               | None                                                                    | bkeagent health port: 1-65535, required.                                                                                                                                               | CRD YAML |
| `spec.clusterConfig.cluster.apiServer.extraArgs{}`         | `authorization-mode=Node,RBAC`                                       | apiserver extra command-line arguments, optional.                                                                                                                                                  | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.certificatesDir`               | `/etc/kubernetes/pki`                                                | Certificate directory, optional.                                                                                                                                                              | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.chartRepo.domain`              | `cr.openfuyao.cn`                                                    | Chart repo domain, optional.                                                                                                                                                      | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.chartRepo.port`                | `443/40443`                                                          | Chart repo access port; default public chart is 443, non-default domain is 40443, optional; must be consistent with the actual chart repo service port and avoid conflicts with other repository service ports on the same node.                                                                               | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.chartRepo.prefix`              | `chart`                                                              | Chart repo prefix, optional.                                                                                                                                                      | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.containerRuntime.cri`          | `containerd`                                                         | CRI, supports `containerd`, optional.                                                                                                                                                | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.containerRuntime.param{}`      | `cgroupDriver=systemd`<br>`containerd data-root=/var/lib/containerd` | Runtime extended parameters, such as `data-root`, `cgroupDriver`, optional.                                                                                                                               | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.containerRuntime.runtime`      | `runc`                                                               | OCI runtime, supports `runc`, `richrunc`, `kata`, optional.                                                                                                                            | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.containerdConfigRef.name`      | `bke-containerd`                                                     | Name of the referenced ContainerdConfig, required.                                                                                                                                              | CRD YAML |
| `spec.clusterConfig.cluster.containerdConfigRef.namespace` | `bke-containerd`                                                     | Namespace of the referenced ContainerdConfig, required.                                                                                                                                            | CRD YAML |
| `spec.clusterConfig.cluster.containerdVersion`             | `v2.1.1`                                                             | containerd default version, optional.                                                                                                                                                    | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.controllerManager.extraArgs{}` | None                                                                    | controller-manager extra command-line arguments, optional.                                                                                                                                         | CRD YAML |
| `spec.clusterConfig.cluster.etcd.dataDir`                  | `/var/lib/openFuyao/etcd`                                            | etcd data directory, optional.                                                                                                                                                          | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.etcdVersion`                   | `v3.6.7-of.1`                                                        | etcd default version, optional.                                                                                                                                                          | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.httpRepo.domain`               | `http.bocloud.k8s`                                                   | System package source domain, optional.                                                                                                                                                            | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.httpRepo.ip`                   | None                                                                    | System package source IP, required.                                                                                                                                                             | CRD YAML |
| `spec.clusterConfig.cluster.httpRepo.port`                 | `40080`                                                              | System package source port, optional; independent from image repository and chart repo service ports, must avoid conflicts when customized; in offline deployment, this is the same service port as `bke init`'s `--yumRepoPort`.                                                                                                                         | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.httpRepo.prefix`               | None                                                                    | System package source prefix, required.                                                                                                                                                             | CRD YAML |
| `spec.clusterConfig.cluster.imageRepo.domain`              | `deploy.bocloud.k8s`                                                 | Image repository domain, optional.                                                                                                                                                            | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.imageRepo.ip`                  | None                                                                    | Image repository IP, required.                                                                                                                                                             | CRD YAML |
| `spec.clusterConfig.cluster.imageRepo.port`                | `40443`                                                              | Image repository port, optional; it is a different service port from the system package source and chart repo, and must be kept separate when customizing.                                                                                                                         | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.imageRepo.prefix`              | `kubernetes`                                                         | Image repository prefix, optional.                                                                                                                                                            | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.kubelet.extraArgs{}`           | None                                                                    | kubelet extra command-line arguments, optional.                                                                                                                                                    | CRD YAML |
| `spec.clusterConfig.cluster.kubelet.extraVolumes.hostPath` | `/var/lib/kubelet`                                                   | kubelet extra mounted hostpath, optional.                                                                                                                                               | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.kubelet.extraVolumes.name`     | `kubelet-root-dir`                                                   | kubelet extra mounted volume name, optional.                                                                                                                                                   | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.kubernetesVersion`             | `v1.34.3-of.1`                                                       | Kubernetes default version, optional.                                                                                                                                                    | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.networking.dnsDomain`          | `cluster.local`                                                      | Service DNS domain suffix, determines the full DNS name format of in-cluster services. For example, for a Service kube-system/kube-dns, the default full domain name is: kube-dns.kube-system.svc.cluster.local, optional.                                              | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.networking.podSubnet`          | `10.250.0.0/16`                                                      | Pod CIDR, optional.                                                                                                                                                          | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.networking.serviceSubnet`      | `10.96.0.0/16`                                                       | Service CIDR, optional.                                                                                                                                                      | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.ntpServer`                     | None                                                                    | NTP server, required.                                                                                                                                                             | CRD YAML |
| `spec.clusterConfig.cluster.openFuyaoVersion`              | `latest`                                                             | openFuyao default version, optional.                                                                                                                                                     | CRD YAML, default when CRD is empty |
| `spec.clusterConfig.cluster.scheduler.extraArgs{}`         | None                                                                    | scheduler extra command-line arguments, optional.                                                                                                                                                  | CRD YAML |
| `spec.clusterConfig.customExtra{}`                         | None                                                                    | Extended parameters, read by templates/plugins ([more details](#222-customextra)), optional.                                                                                                                         | CRD YAML |
| `spec.controlPlaneEndpoint.host`                           | None                                                                    | Control plane access entry host, required.                                                                                                                                                        | CRD YAML |
| `spec.controlPlaneEndpoint.port`                           | None                                                                    | Control plane access entry port, required.                                                                                                                                                        | CRD YAML |
| `spec.pause`                                               | None                                                                    | Pause reconcile, optional.                                                                                                                                                       | CRD YAML |
| `spec.reset`                                               | `false`                                                              | When set to `true`, triggers the deletion reconciliation process for this `BKECluster`. To thoroughly clean up target nodes, the annotation `bke.bocloud.com/ignore-target-cluster-delete` must also be set to `"false"`. For operating steps see [Cluster Uninstallation](../../cluster_uninstallation/overview.md). | CRD YAML |

<a id="221-addons"></a>

#### addons[].param

Addons generated by bkeadm by default:

**Table 5** bkeadm Default Addon Parameters Table

| addon name | Default Version | bkeadm default param | Remarks |
|---|---|---|---|
| `kubeproxy` | Kubernetes version corresponding directory, e.g. `v1.34.3-of.1` | `clusterNetworkMode` | Default value: `calico`. |
| `calico` | `v3.31.3` | `calicoMode`<br>`ipAutoDetectionMethod`<br>`allowTypha`<br>`typhaReplicas` | Default values:<br>`calicoMode=vxlan`<br>`ipAutoDetectionMethod=skip-interface=nerdctl*`<br>`allowTypha=false`<br>`typhaReplicas=1`. |
| `coredns` | `v1.12.2-of.1` | `EnableAntiAffinity` | `true/false` |
| `cluster-api` | `v1.4.3` | `manage`<br>`offline`<br>`sandbox`<br>`replicas`<br>`containerdVersion`<br>`openFuyaoVersion`<br>`manifestsVersion`<br>`providerVersion`<br>`ntpServer`<br>`healthPort` |  |
| `openfuyao-system-controller` | `latest` | `helmRepo`<br>`tagVersion` |  |

Commonly used addon names and parameters:

**Table 6** Common Addon Parameters Table

| addon name | Adopted Version | Component-specific/overridable param key | Remarks |
|---|---|---|---|
| `nodelocaldns` | `v1.26.4` | `clusterDNS`<br>`DNSserver`<br>`domain`<br>`localdns` | Note that it is `domain`, not `dnsDomain`. |
| `numa-affinity-package` | `v0.0.2` | `openfuyaoRepo` | NUMA affinity. |
| `redis` | `6.1.12` | `nodeselector` | The key name is lowercase per template. |
| `victoriametrics-controller` | `latest` | `grafanaNodePort`<br>`useVMSingle`<br>`kubeStateMetricsAutoSharding`<br>`kubeStateMetricsCpuCount`<br>`kubeStateMetricsMemorySize`<br>`kubeStateMetricsReplicaCount`<br>`vmAgentAllowStatefulSet`<br>`vmAgentCpuCount`<br>`vmAgentMemorySize`<br>`vmAgentReplicaCount`<br>`vmAgentScrapeInterval`<br>`vmAgentShareCount`<br>`vmAgentStorageSize`<br>`vmAlertCpuCount`<br>`vmAlertMemorySize`<br>`vmAlertReplicaCount`<br>`vmAlertManagerCpuCount`<br>`vmAlertManagerMemorySize`<br>`vmAlertManagerReplicaCount`<br>`vmClusterReplicationFactor`<br>`vmClusterRetentionPeriod`<br>`vmInsertCpuCount`<br>`vmInsertMemorySize`<br>`vmInsertReplicaCount`<br>`vmSelectCpuCount`<br>`vmSelectMemorySize`<br>`vmSelectReplicaCount`<br>`vmSelectStorageSize`<br>`vmSingleStorageSize`<br>`vmStorageCPUCount`<br>`vmStorageMemorySize`<br>`vmStorageReplicaCount`<br>`vmStorageStorageSize` | The webhook fills in default values for missing items and validates some replica/share relationships. |

<a id="222-customextra"></a>

#### `spec.clusterConfig.customExtra{}`

**Table 7** customExtra Configuration Items Table

| Configuration Item Name | Default Value                                      | Description | Setting Method |
| --- |------------------------------------------| --- | --- |
| `customExtra.chartRepoPort` | `38080`                                  | Local chart repo port, optional; corresponds to different local services from `customExtra.imageRepoPort` and `customExtra.yumRepoPort` respectively, and must be kept separate when customizing. | CLI-derived/YAML |
| `customExtra.clusterapi` | `latest`                                 | cluster-api-provider-bke version, optional. | CLI-derived/YAML |
| `customExtra.containerd` | `containerd-v2.1.1-linux-{.arch}.tar.gz` | containerd installation package template name, optional. | Version artifacts/YAML |
| `customExtra.domain` | `deploy.bocloud.k8s`                     | Image repository domain, optional. | CLI-derived/YAML |
| `customExtra.host` | None                                        | Bootstrap node IP, optional. | Generated environment/YAML |
| `customExtra.imageRepoPort` | `40443`                                  | Local image repository port, optional; independent from the chart repo and yum/http repo ports, and must be kept separate when customizing. | CLI-derived/YAML |
| `customExtra.nfsserverpath` | `/`                                      | NFS service path, optional. | Code default/YAML |
| `customExtra.onlineImage` | None                                        | Online installation image, optional. | CLI-derived/YAML |
| `customExtra.otherChart` | None                                        | Private chart repo address, optional. | CLI-derived/YAML |
| `customExtra.otherChartIp` | None                                        | Private chart repo resolved IP, optional. | Generation logic/YAML |
| `customExtra.otherRepo` | None                                        | Private image repository, optional. | CLI-derived/YAML |
| `customExtra.otherRepoIp` | None                                        | Private image repository resolved IP, optional. | Generation logic/YAML |
| `customExtra.otherSource` | None                                        | Private system package source, optional. | CLI-derived/YAML |
| `customExtra.yumRepoPort` | `40080`                                  | Local system package source (HTTP repo) service port, optional; independent from image repository and chart repo ports, must avoid conflicts when customized; same service port as Cluster CRD's `spec.clusterConfig.cluster.httpRepo.port`. | CLI-derived/YAML |

Port relationships and conflict check: `customExtra.imageRepoPort`, `customExtra.chartRepoPort`, and `customExtra.yumRepoPort` are usually derived from the `--imageRepoPort`, `--chartRepoPort`, and `--yumRepoPort` flags of `bke init`, corresponding to three independent services: the local image repository, chart repo, and yum/http repo. When customizing, they must be configured as different ports, and must avoid the Kubernetes API, console ports, and ports already occupied on the host. The `bke init` preflight checks whether ports are already occupied on the host and filters out existing BKE management container ports; however, it does not explicitly check for duplicates among these parameters. If ports are duplicated, failures may occur during service startup or port publishing, and you need to adjust based on the port occupancy information in the logs and retry.


### BKENode CRD


**Table 8** BKENode CRD Configuration Items Table

| Configuration Item Name | Default Value           | Description | Setting Method |
|---|---------------|---|---|
| `metadata.creationTimestamp` | None | Creation time; usually written by Kubernetes, can be left empty in user configuration, optional. | CRD YAML |
| `metadata.labels.cluster.x-k8s.io/cluster-name` | `bke-cluster`  | Associated Cluster API cluster name, optional. | CRD YAML |
| `metadata.name` | None | BKENode resource name, required. | CRD YAML |
| `metadata.namespace` | `bke-cluster` | Namespace where BKENode resides, optional. | CRD YAML |
| `spec.hostname` | None | Node hostname, required. | CRD YAML |
| `spec.ip` | None | Node IP, required. | CRD YAML |
| `spec.password` | None | SSH password, required. | CRD YAML |
| `spec.port` | `22`          | SSH port, example is `"22"`, optional. | CRD YAML |
| `spec.role[]` | None | Node role, e.g. `master/node`, `etcd`, `node`, required. | CRD YAML |
| `spec.username` | `root`        | SSH username, optional. | CRD YAML |
| `spec.controlPlane` | None | Node-level control plane component override configuration, optional. | CRD YAML |
| `spec.kubelet` | None | Node-level kubelet override configuration, optional. | CRD YAML |
| `spec.labels[]` | None | Node-level label override configuration, optional. | CRD YAML |

### ContainerdConfig CRD

**Table 9** ContainerdConfig CRD Configuration Items Table

| Configuration Item Name | Default Value | Description | Setting Method |
|---|---|---|---|
| `spec.configType` | `combined` | Configuration type, supports `service/main/registry/combined`, optional. | CRD YAML |
| `spec.description` | None | Human-readable description, required. | CRD YAML |
| `spec.service.execStart` | None | systemd ExecStart, required. | CRD YAML |
| `spec.service.slice` | `system.slice` | systemd slice, optional. | CRD YAML |
| `spec.service.killMode` | `process` | KillMode, supports `control-group/process/mixed/none`, optional. | CRD YAML |
| `spec.service.restart` | `always` | Restart policy, optional. | CRD YAML |
| `spec.service.restartSec` | `5s` | Restart wait time, optional. | CRD YAML |
| `spec.service.startLimitInterval` | `10s` | Start rate-limit window, optional. | CRD YAML |
| `spec.service.startLimitBurst` | `5` | Start rate-limit burst count, optional. | CRD YAML |
| `spec.service.timeoutStopSec` | `90s` | Stop timeout, optional. | CRD YAML |
| `spec.service.logging.standardOutput` | `journal` | stdout log destination, optional. | CRD YAML |
| `spec.service.logging.standardError` | `journal` | stderr log destination, optional. | CRD YAML |
| `spec.service.logging.syslogIdentifier` | None | syslog identifier, required. | CRD YAML |
| `spec.service.logging.logLevelMax` | None | Maximum log level, optional. | CRD YAML |
| `spec.service.customExtra{}` | None | Service template extended variables, optional. | CRD YAML |
| `spec.main.metricsAddress` | None | containerd metrics address, required. | CRD YAML |
| `spec.main.root` | `/var/lib/containerd` | containerd root, optional. | CRD YAML |
| `spec.main.state` | `/run/containerd` | containerd state, optional. | CRD YAML |
| `spec.main.sandboxImage` | `registry.k8s.io/pause:3.9` | Pause image, optional. | CRD YAML |
| `spec.main.configPath` | `/etc/containerd/certs.d` | Registry configuration directory, optional. | CRD YAML |
| `spec.main.rawTOML` | None | Raw TOML, takes precedence when provided, optional. | CRD YAML |
| `spec.registry.configPath` | `/etc/containerd/certs.d` | Registry hosts configuration directory, optional. | CRD YAML |
| `spec.registry.configs{host}` | None | Registry host configuration mapping, optional. | CRD YAML |
| `registryHost.host` | None | Single registry host address, required. | CRD YAML |
| `registryHost.capabilities` | None | Single registry host capabilities, optional. | CRD YAML |
| `registryHost.skipVerify` | None | Single registry host skip verification, optional. | CRD YAML |
| `registryHost.plainHTTP` | None | Single registry host use plain HTTP, optional. | CRD YAML |
| `registryHost.insecure` | None | Single registry host insecure setting, optional. | CRD YAML |
| `registryHost.tls.caFile` | None | TLS CA file, required. | CRD YAML |
| `registryHost.tls.certFile` | None | TLS cert file, required. | CRD YAML |
| `registryHost.tls.keyFile` | None | TLS key file, required. | CRD YAML |
| `registryHost.tls.insecureSkipVerify` | None | TLS skip certificate verification, optional. Must not be set to `true` in production environments; only allowed for temporary troubleshooting or isolated test environments for short periods. | CRD YAML |
| `registryHost.auth.username` | None | Registry authentication username, required. | CRD YAML |
| `registryHost.auth.password` | None | Registry authentication password, required. | CRD YAML |
| `registryHost.auth.auth` | None | Registry auth field, optional. | CRD YAML |
| `registryHost.auth.identityToken` | None | Registry identity token, optional. | CRD YAML |
| `registryHost.auth.registryToken` | None | Registry token, optional. | CRD YAML |
| `registryHost.header` | None | Additional header, optional. | CRD YAML |
| `registryHost.overridePath` | None | Path override, required. | CRD YAML |
| `spec.script.content` | None | Shell script content, optional. | CRD YAML |
| `spec.script.path` | None | Shell script path, optional. | CRD YAML |
| `spec.script.args` | None | Shell script arguments, optional. | CRD YAML |
| `spec.script.interpreter` | `/bin/bash` | Shell script interpreter, optional. | CRD YAML |

### KubeletConfig CRD

**Table 10** KubeletConfig CRD Configuration Items Table

| Configuration Item Name | Default Value | Description | Setting Method |
|---|---|---|---|
| `spec.kubeletConfig{}` | None | kubelet native configuration map, RawExtension, optional. | CRD YAML |
| `spec.kubeletService.enabled` | None | Whether to create the kubelet service, optional. | CRD YAML |
| `spec.kubeletService.serviceName` | None | systemd service name, required. | CRD YAML |
| `spec.kubeletService.unit.description` | None | Unit description, optional. | CRD YAML |
| `spec.kubeletService.unit.documentation` | None | Unit documentation, optional. | CRD YAML |
| `spec.kubeletService.unit.after` | None | Unit After dependency, optional. | CRD YAML |
| `spec.kubeletService.unit.wants` | None | Unit Wants dependency, optional. | CRD YAML |
| `spec.kubeletService.unit.requires[]` | None | Unit Requires dependency, optional. | CRD YAML |
| `spec.kubeletService.service.execStart` | None | kubelet startup command, required. | CRD YAML |
| `spec.kubeletService.service.restart` | None | Service restart policy, required. | CRD YAML |
| `spec.kubeletService.service.startLimitInterval` | None | Service start rate-limit window, required. | CRD YAML |
| `spec.kubeletService.service.restartSec` | None | Service restart wait time, required. | CRD YAML |
| `spec.kubeletService.service.environment` | None | Service environment variables, optional. | CRD YAML |
| `spec.kubeletService.service.environmentFile` | None | Service environment variables file, optional. | CRD YAML |
| `spec.kubeletService.service.execStartPre` | None | Service pre-start command, optional. | CRD YAML |
| `spec.kubeletService.service.startLimitBurst` | None | Service start rate-limit burst count, required. | CRD YAML |
| `spec.kubeletService.service.killMode` | None | Service KillMode, required. | CRD YAML |
| `spec.kubeletService.service.standardOutput` | None | stdout log output setting, optional. | CRD YAML |
| `spec.kubeletService.service.standardError` | None | stderr log output setting, optional. | CRD YAML |
| `spec.kubeletService.service.syslogIdentifier` | None | syslog identifier, required. | CRD YAML |
| `spec.kubeletService.service.workingDirectory` | None | Service working directory, optional. | CRD YAML |
| `spec.kubeletService.service.user` | None | Service running user, optional. | CRD YAML |
| `spec.kubeletService.service.group` | None | Service running user group, optional. | CRD YAML |
| `spec.kubeletService.service.customExtra` | None | Service template extended variables, optional. | CRD YAML |
| `spec.kubeletService.install.wantedBy` | None | Install WantedBy configuration, optional. | CRD YAML |
| `spec.kubeletService.install.requiredBy[]` | None | Install RequiredBy configuration, optional. | CRD YAML |
| `spec.files[].path` | None | Extra file path, optional. | CRD YAML |
| `spec.files[].content` | None | Extra file content, optional. | CRD YAML |
| `spec.files[].permissions` | None | Extra file permissions, optional. | CRD YAML |
| `spec.files[].owner` | None | Extra file owner, optional. | CRD YAML |
| `spec.commands[].command` | None | Extra command, optional. | CRD YAML |
| `spec.commands[].args` | None | Extra command arguments, optional. | CRD YAML |
| `spec.commands[].workingDir` | None | Extra command working directory, optional. | CRD YAML |

### KCT and CCT Configuration and References

KCT (KubeletConfig) and CCT (ContainerdConfig) are custom resources on the management cluster, used to deliver kubelet and containerd configurations during the node bootstrap phase, without the need to log into nodes and manually edit files.

**Table 11** KCT and CCT Comparison Table

| Abbreviation | CR Type | API Resource Name | kubectl Short Name | BKECluster Reference Field |
| --- | --- | --- | --- | --- |
| KCT | `KubeletConfig` | `kubeletconfigs` | `kct` | `spec.KubeletConfigRef.name` / `namespace` |
| CCT | `ContainerdConfig` | `containerdconfigs` | `cct` | `spec.clusterConfig.cluster.containerdConfigRef.name` / `namespace` |

**Usage Workflow**

1. Create KCT/CCT resources on the management cluster (the same cluster as BKECluster) and fill in the `spec` configuration.
2. Point to the corresponding resources via the reference fields in the BKECluster YAML; the default cluster YAML generated by `bke init` already contains example references.
3. After running `bke create cluster`, bkeagent reads and applies the configuration during the node bootstrap phase:
   - **KCT**: When `KubeletConfigRef` is not empty, bkeagent pulls the KubeletConfig from the management cluster and processes `spec.files[]`, `spec.kubeletConfig` (writes to `/var/lib/kubelet/config.yaml`), `spec.kubeletService` (generates `kubelet.service`), and `spec.commands[]` in order; variable substitution (`${EXPR|command|END}`) is automatically enabled.
   - **CCT**: When `containerdConfigRef` is not empty, bkeagent passes it to the InstallContainerd plugin in `namespace:name` form, and renders the containerd configuration and restarts the service in the order `spec.script` → `spec.service` → `spec.main` → `spec.registry`.

**KCT/CCT Configuration Example (BKECluster snippet)**

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

For complete field descriptions, variable substitution examples, and YAML samples, see [Binary Component Custom Configuration](../service_management_cluster_configuration/binary_file_custom_configuration_guide.md).

## Annotations

### Configurable Annotations
These annotations affect BKECluster behavior

**Table 12** Configurable Annotations Table

| key | Semantics and Values                     | Default Value and Priority                            |
|---|---------------------------|------------------------------------|
| **bke.bocloud.com/retry** | An empty value retries all nodes; a non-empty value is a comma-separated list of node IPs.   | Triggered when the key exists; deleted after consumption.                  |
| **bke.bocloud.com/deep-restore-node** | true enables deep recovery for node initialization and reset.   | Defaults to true; annotation takes precedence, and when missing it is also treated as true, with true written by default. |
| **bke.bocloud.com/ignore-namespace-delete** | Only false allows namespace deletion; any other value skips it. | Defaults to true; only an exact false enters the deletion branch.         |
| **bke.bocloud.com/ignore-target-cluster-delete** | true means the target cluster is not reset when deleting management plane objects. | Defaults to true; only an exact false continues cleaning up target nodes. For complete uninstallation steps see [Cluster Uninstallation Overview](../../cluster_uninstallation/overview.md). |
| **bke.bocloud.com/node-boot-wait-timeout** | Go duration, e.g. 30s, 10m, 1h. | Defaults to 10m; valid annotation takes precedence, falls back to 10m when missing or parsing fails.        |

Example scenarios for using the retry annotation:

1. Pushing the Agent fails during the EnsureBKEAgent phase.

2. The Agent is pushed successfully during the EnsureBKEAgent phase, but bkeagent fails to start on the corresponding node.

3. kubelet installation fails on the corresponding node.

4. The DNS resolution service on the corresponding node fails, causing image pull failures.

When encountering the above scenarios, users can manually fix the issue and then run the following command to add the retry annotation, so that the failed node re-executes the subsequent phases.
```bash
# Note: When the value of the retry annotation is empty, all nodes will be retried.
kubectl annotate bc -n <cluster-namespace> <cluster-name>  bke.bocloud.com/retry="<failed-node-ip>"
```

### Non-configurable Annotations
#### BKE Internal State, Process, and Operations Annotations (13)

**Table 13** BKE Internal State, Process, and Operations Annotations Table

| key | Semantics and Values | Priority and Lifecycle |
| --- | --- | --- |
| **bke.bocloud.com/collectd** | Comma-separated base, agent, indicating that base information and agent information have been collected. | The controller retains existing tokens and only appends missing tokens. |
| **bke.bocloud.com/last-update-configuration** | Serialized object JSON, used as the last configuration snapshot. | Internal three-way merge baseline, not part of the normal user override chain. |
| **bke.bocloud.com/status-record** | Requests StatusManager to record the current state once, usually an empty value. | Triggered when the key exists; deleted after recording. |
| **bke.bocloud.com/cluster-tracker-healthy-check-failed** | Marks ClusterTracker health check failure and triggers re-reconciliation. | Takes effect when the key exists; deleted after recovering to Ready. |
| **bke.bocloud.com/addon-boot-wait-timeout** | Designed to indicate the addon boot wait timeout. | No production read or write points yet; current settings do not take effect. |
| **bke.bocloud.com/event** | Marks this as a regular BKE event. | An empty value is written each time AnnotatedEventf creates an Event. |
| **bke.bocloud.com/complete** | Together with event, marks a process completion event. | An empty value is written when a completion event is created. |
| **bke.bocloud.com/bkeagent-listener** | current means the agent listens to the current cluster; bkecluster means it is to be switched to BKECluster. | Defaults to current when missing; bkecluster is written during the addon phase, and the process reverts to current after the switch is complete. |
| **bke.bocloud.com/cluster-from** | bke, bocloud, other indicate the cluster source. | Defaults to bke when missing or empty; different sources affect the configuration validation, full management, and certificate distribution branches. |
| **bke.bocloud.com/cluster-api-manager-applied** | true means 004-manage.yaml has been applied. | Only executed when the target version exists, postprocess is complete, and the value is not true; writes true upon success. |
| **etcd-cert-dir** | Temporarily stores the collected etcd certificate directory. | Takes precedence over Kubernetes certificatesDir; deleted after agent information collection is complete. |
| **deployAction** | Fixed value k8s_upgrade, used for BOC compatibility process identification. | Overridden to k8s_upgrade when missing or the value is not k8s_upgrade; no automatic cleanup seen in the code. |
| **bke.openfuyao.cn/restartedAt** | RFC3339 timestamp, used to change the PodTemplate and trigger a Deployment rolling restart. | Updated to the current time on each PatchDeploymentImage. |


#### ClusterVersion, Upgrade Path, and ReleaseImage Annotations (6)
These keys are protocols between upgrade controllers. For parameter descriptions and collaboration relationships, see [Backend Declarative Upgrade Guide](../../upgrade_guide/backend_declarative_upgrade_guide.md)#controller-parameters-and-upgrade-annotations.

**Table 14** ClusterVersion, Upgrade Path, and ReleaseImage Annotations Table

| key | Semantics and Values | Priority and Lifecycle |
| --- | --- | --- |
| **cvo.openfuyao.cn/upgrade-ready** | The target version of the current upgrade hop, and also the execution gate for the declarative upgrade DAG. | After trimming, a non-empty value allows DAG execution; its value is the current hop, not necessarily the final desiredVersion. |
| **cvo.openfuyao.cn/cluster-version** | The name of the associated ClusterVersion object. | Written and cleaned up together with upgrade-ready, not used as a gate on its own. |
| **cvo.openfuyao.cn/upgrade-path** | The selected upgrade path, e.g. v1->v2,v2->v3. | Written and cleaned up together with upgrade-ready. |
| **cvo.openfuyao.cn/helm-component** | When trimmed and case-insensitively equal to true, enables the yaml/helm component executor for this cluster. | When the annotation exists, the annotation value takes precedence; when missing, falls back to the `--helm-component-support` global flag. |
| **config.openfuyao.com/oci-digest** | Saves the UpgradePath OCI image digest and syncs it to status.lastDigest. | Overwritten with the latest digest each time the path is refreshed from OCI. |
| **cvo.openfuyao.cn/bkecluster-name** | Associates the ReleaseImage with the target BKECluster name. | Read by the ReleaseImage controller, used to resolve and sync target cluster version information. |

When the upgrade is completed, cancelled, or cleaned up, the controller removes upgrade-ready, cluster-version, and upgrade-path together.

#### Cluster API Standard Annotations (8)
The semantics of these keys are mainly defined by upstream Cluster API. This repository is responsible for setting, deleting, or consuming them, so they are also part of the actual annotation surface.

**Table 15** Cluster API Standard Annotations Table

| key | Semantics and Values | Priority and Lifecycle |
| --- | --- | --- |
| **cluster.x-k8s.io/cluster-name** | Indicates the name of the Cluster to which the node belongs. | Only maps to the corresponding BKECluster when present together with cluster-namespace. |
| **cluster.x-k8s.io/cluster-namespace** | Indicates the namespace of the Cluster to which the node belongs. | No object mapping is performed when missing. |
| **cluster.x-k8s.io/paused** | The presence of the key indicates pausing Cluster API object reconciliation. | Either Cluster.spec.paused=true or the presence of this key on the object can pause; **the value does not participate in the judgment**. |
| **cluster.x-k8s.io/delete-machine** | Sets the annotated machine as a priority deletion candidate. | Takes effect when the key exists, usually written as an empty value. |
| **cluster.x-k8s.io/managed-by** | Indicates that the infrastructure object is managed by an external system, and this controller skips mapping. | When the key exists, it is treated as externally managed. |
| **machine.cluster.x-k8s.io/certificates-expiry** | RFC3339 certificate expiration time; currently written as the current time plus 100 years. | Generated by the controller, used to avoid unnecessary behavior triggered by upstream due to short certificate periods. |
| **controlplane.cluster.x-k8s.io/skip-kube-proxy** | Requires KCP to not be responsible for upgrading kube-proxy. | Upstream judges based on whether the key exists; BKE manages the related components itself. |
| **controlplane.cluster.x-k8s.io/skip-coredns** | Requires KCP to not be responsible for upgrading CoreDNS. | Upstream judges based on whether the key exists; BKE manages the related components itself. |

#### kubeadm, Kubernetes, and Static Pod Annotations (7)

**Table 16** kubeadm, Kubernetes, and Static Pod Annotations Table

| key | Semantics and Values | Priority and Lifecycle |
| --- | --- | --- |
| **kubeadm.kubernetes.io/etcd.advertise-client-urls** | Usually node IP:2379, describes the etcd client advertise URL. | During the onboarding phase, if the key already exists it is not overwritten; during the master upgrade phase, it is filled in when missing or empty. |
| **bkeagent.bocloud.com/etcd.advertise-client-urls** | Designed to also be the etcd advertise client URL. | The current production code actually uses the kubeadm key above. |
| **kubeadm.alpha.kubernetes.io/cri-socket** | Saves the CRI socket used during init/join. | This repository only has a constant definition, with no production read or write points. |
| **kubeadm.kubernetes.io/kube-apiserver.advertise-address.endpoint** | Describes the API Server advertise address and port. | This repository only has a constant definition, with no production read or write points. |
| **kubeadm.kubernetes.io/component-config.hash** | Saves the component configuration SHA256, used to detect user modifications. | This repository only has a constant definition, with no production read or write points. |
| **kubernetes.io/service-account.name** | Indicates which ServiceAccount this Secret belongs to, for the Kubernetes token controller to fill in the token. | Fixed to the target ServiceAccount name when creating the Secret. |
| **kubernetes.io/config.hash** | Static Pod manifest content hash, used to determine whether the component Pod has been updated. | BKE only reads and waits for the hash to change, and does not write. |


#### Deployment Manifest and Generation Tool Annotations (5)

**Table 17** Deployment Manifest and Generation Tool Annotations Table

| key | Purpose | Configuration and Validation |
| --- | --- | --- |
| **controller-gen.kubebuilder.io/version** | Records the controller-gen version used to generate the CRD. | Generated by controller-gen, used only for traceability, and does not control runtime business logic. |
| **cert-manager.io/inject-ca-from** | Tells cert-manager to inject the caBundle from the Certificate/Secret source specified by namespace/name. | The config contains Kustomize variables, which become the actual namespace/name after build; does not take effect when cert-manager is not deployed or the injector is not running. |
| **kubectl.kubernetes.io/default-container** | When kubectl logs/exec does not specify a container, manager is selected by default. | Fixed string manager; only affects the kubectl usage experience. |
| **prometheus.io/port** | Tells the collector based on Prometheus annotation discovery to use port 8080. | The value is the string 8080; does not take effect when the collection system does not enable annotation discovery. |
| **prometheus.io/scrape** | true means scraping metrics by Prometheus based on annotation discovery is allowed. | Interpreted by the collection system; Kubernetes itself does not validate it. |
