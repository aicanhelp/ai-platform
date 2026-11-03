# KubeOS Custom Configuration

## Background

Traditional node OS upgrades rely on package managers to update packages one by one, which easily leads to environment drift. KubeOS is an immutable OS designed by openEuler for Kubernetes worker node scenarios. It adopts a read-only rootfs, A/B root partitions, and a persistent PERSIST partition to perform atomic upgrades and rollbacks via whole-image updates, ensuring consistent and reproducible node OS states.

openFuyao supports a hybrid OS cluster topology where regular openEuler master nodes and KubeOS worker VMs coexist. When users need worker nodes with immutable OS capabilities to implement OS atomic upgrades and rollbacks, they can follow this guide to complete the KubeOS adaptation deployment. After the cluster installation is complete, the KubeOS addon takes over the OS lifecycle management (upgrade/rollback/configuration) of worker nodes.

**Table 1** Comparison between regular openEuler and KubeOS

| Item | Regular openEuler | KubeOS |
| ---- | ------------- | ------ |
| rootfs | Read-write | Read-only |
| OS upgrade method | Package manager updates packages one by one | Whole-image atomic upgrade/rollback |
| Partition model | Conventional partitions | A/B root + PERSIST persistent partition |
| Binary installation | Installable at runtime | Built-in during image building |
| Applicable roles | master/worker | worker only |

The current deployment topology is regular openEuler master + KubeOS VM worker. Running KubeOS on master is not supported, and physical machine KubeOS workers are not supported. This deployment topology supports small, medium, and large-scale clusters. The number of workers is limited by the virtualization resource pool and the cluster size.

## Prerequisites

- The bootstrap node has been initialized, and the bootstrap cluster is running normally. For the bootstrap node initialization method, see [Install Bootstrap Cluster](../bootstrap_cluster_configuration).
- Regular openEuler master nodes have been prepared.
- The KubeOS VM system disk image `system.qcow2` or `system.img` has been built. The image has containerd, kubelet, bke-agent, os-agent, and other runtime dependencies built in. For the build method, see [KubeOS Image Build Manual](https://gitcode.com/openFuyao/kubeos-support/blob/master/docs/KubeOS镜像构建手册.md).
- KubeOS addon artifacts (os-operator image, os-proxy image, Helm chart) have been prepared. For the artifact source, see [kubeos-support repository](https://gitcode.com/openFuyao/kubeos-support).
- The virtualization resource pool supports creating VMs from qcow2/raw system disks.

> ![](../figures/icon-note.gif)**Note:**
>
> In the following, bke-cluster.yaml and bke-node.yaml are files on the bootstrap node used to configure cluster installation information. They are located in the `/bke/cluster/` directory by default.

## Usage Restrictions

- The master node does not support running KubeOS and does not support container OS upgrades.
- Worker nodes must be created from KubeOS VM images. Regular openEuler nodes are not allowed to join the KubeOS worker pool.
- Physical machine KubeOS workers are not supported at the current stage. PXE (Preboot eXecution Environment) physical machine installation is designed as a separate future extension.
- Online conversion of regular openEuler workers to KubeOS workers is not supported.
- KubeOS worker VMs must use independent system disks. Multiple VMs cannot share the same `system.qcow2` system disk.
- The master node does not install os-agent and does not mount `/var/run/os-agent`.

## Operation Steps

### Create KubeOS Worker VMs

1. Import the KubeOS `system.qcow2` or `system.img` into the virtualization platform.

2. Create multiple worker VMs through FC (FusionCompute) or QEMU. Allocate a unique system disk, MAC, hostname, and network configuration to each VM.

   > ![](../figures/icon-notice.gif)**Notice:**
   >
   > `system.qcow2` should be used as a template image. IP, hostname, and machine-id must be configured dynamically and must not be set to fixed values. Each worker must clone an independent system disk.

3. Confirm that after the first boot of the worker VM, the network and hostname configuration is complete, and the following mounts are in effect:

   ```bash
   # Confirm the persistent partition is mounted
   mount | grep persist
   # Confirm /etc is overlay writable
   mount | grep '/etc'
   # Confirm /var is a bind mount
   mount | grep '/var'
   # Confirm /opt/cni is overlay writable
   mount | grep '/opt/cni'
   ```

### Configure Cluster Installation Parameters

1. Configure cluster information in bke-cluster.yaml.

   The cluster configuration is the same as a regular cluster. KubeOS adaptation does not require additional parameters in bke-cluster.yaml. It is triggered via the `--immutable-os` parameter of the installation command.

2. Configure node information in bke-node.yaml.

   The master node role is configured as `master/node` and `etcd`, and the worker node role is configured as `node`. The IP and hostname must match the actual KubeOS VM. The following only shows the KubeOS worker node configuration. The master node configuration is the same as a regular cluster.

   ```yaml
   apiVersion: bke.bocloud.com/v1beta1
   kind: BKENode
   metadata:
     labels:
       cluster.x-k8s.io/cluster-name: bke-cluster
     name: bke-cluster-n1
     namespace: bke-cluster
   spec:
     hostname: n1
     ip: 192.168.201.51
     port: "22"
     role:
     - node
     username: root
     password: 'your-password'
   ```

### Start Cluster Installation

On the bootstrap node or management cluster node, execute the cluster creation command. Declare that the worker nodes of this cluster use the KubeOS image via the `--immutable-os=kubeos` parameter:

```bash
bke cluster create -f /bke/cluster/bke-cluster.yaml -n /bke/cluster/bke-node.yaml --immutable-os=kubeos
```

The installer injects the `openfuyao.io/immutable-os=kubeos` annotation on the BKECluster CR. The subsequent installation logic follows the KubeOS adaptation logic based on this flag. The state changes and verification methods during the cluster installation process are the same as a regular cluster.

> ![](../figures/icon-note.gif)**Note:**
>
> The `--immutable-os` parameter only takes effect on worker nodes. Master nodes always follow the regular openEuler installation process.

### Installation Adaptation Description

When the installation process detects KubeOS workers, the installation logic for worker nodes is adapted as follows:

| Component | Regular OS Process | KubeOS Adaptation Process |
| ---- | ---------- | -------------- |
| containerd | Download and extract binary, write service file | Skip download (built into image), only render `/etc/containerd/config.toml` and other configurations. |
| kubelet | Download binary, write service file | Skip download (built into image), only generate bootstrap configuration, kubeconfig, and other variable configurations. |
| bke-agent | SSH push binary to `/usr/local/bin` | Skip push (built into image), only deliver configuration, certificates, and kubeconfig. |
| kubectl | Download and distribute to worker | Skip installation. |
| CNI Calico | Write to `/opt/cni/bin` and `/etc/cni/net.d` | Remains unchanged (`/opt/cni` and `/etc` are writable on KubeOS). |

After the worker joins the cluster, the following labels are automatically applied:

```yaml
node.openfuyao.io/os: kubeos
node.openfuyao.io/role: worker
upgrade.openeuler.org/node-selector: default-worker-pool
```

### Install KubeOS addon

The KubeOS addon must be declared in advance in bke-cluster.yaml and is automatically installed when the cluster is created. The addon includes OS/OSInstance CRD, RBAC, os-operator (Deployment, scheduled to control-plane), and os-proxy (DaemonSet, scheduled only to KubeOS workers), used to take over the OS lifecycle management of worker nodes.

1. Configure the chart repo.

   Declare the chart repository address in `spec.clusterConfig.cluster.chartRepo` of bke-cluster.yaml:

   ```yaml
   spec:
     clusterConfig:
       cluster:
         chartRepo:
           domain: cr.openfuyao.cn
           port: "443"
           prefix: charts
   ```

   > ![](../figures/icon-note.gif)**Note:**
   >
   > For chart repository authentication and TLS configuration, see [addon usage guide](./using_add-ons.md#chart-form-addon).

2. Add the KubeOS addon configuration.

   Add the following to `spec.clusterConfig.addons`:

   ```yaml
   spec:
     clusterConfig:
       addons:
       - name: kubeos-addon
         type: chart
         version: 0.0.0-latest
         releaseName: kubeos-addon
         namespace: upgrade-system
         block: true
   ```

   > ![](../figures/icon-note.gif)**Note:**
   >
   > `block: true` means to block and wait for this addon installation to complete before continuing subsequent processes. The KubeOS addon is recommended to be set to `true` to ensure the OS management components are ready before subsequent operations.

3. Configure a custom values.yaml (optional).

   If valuesConfigMapRef is not configured, the addon uses the chart package's default values.yaml, and the image is pulled from the repository specified by `global.imageRegistry` by default. To customize the image repository address or version, specify it via valuesConfigMapRef. First create a ConfigMap in the management cluster:

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: kubeos-addon-values
     namespace: bke-cluster
   data:
     values.yaml: |
       global:
         imageRegistry: "cr.openfuyao.cn/openfuyao"
         imageNamespace: "kubeos"
       operator:
         image:
           repository: kubeos-operator
           tag: "v1.0.8"
       proxy:
         image:
           repository: kubeos-proxy
           tag: "v1.0.8"
   ```

   Then reference it in the addon configuration:

   ```yaml
           valuesConfigMapRef:
             name: kubeos-addon-values
             namespace: bke-cluster
             valuesKey: values.yaml
   ```

   > ![](../figures/icon-note.gif)**Note:**
   >
   > `global.imageRegistry` must be filled with an actually accessible image repository address. The os-operator and os-proxy images must be pushed to this repository in advance.

4. Verify the addon installation.

   After the cluster is created, execute the following commands to verify the KubeOS addon component status:

   ```bash
   # Confirm os-operator running status, expected 1/1 Running
   kubectl get pods -n upgrade-system -l control-plane=upgrade-operator

   # Confirm os-proxy covers all KubeOS workers, expected pod count equals worker node count
   kubectl get pods -n upgrade-system -l control-plane=upgrade-proxy -o wide

   # Confirm CRDs are registered, expected status Established
   kubectl get crd os.upgrade.openeuler.org osinstances.upgrade.openeuler.org
   ```

## Configuration Example

The following is a complete bke-cluster.yaml configuration snippet that includes the KubeOS addon, where `valuesConfigMapRef` is an optional field:

```yaml
spec:
  clusterConfig:
    cluster:
      chartRepo:
        domain: cr.openfuyao.cn
        port: "443"
        prefix: charts
      containerRuntime:
        cri: containerd
        runtime: runc
        param:
          cgroup-driver: systemd
          data-root: /var/lib/containerd
      containerdVersion: v2.1.1
      kubernetesVersion: v1.34.3-of.1
      imageRepo:
        domain: default
        ip: 192.168.200.71
        port: "40443"
    addons:
    - name: kubeproxy
      version: v1.34.3-of.1
      block: true
      param:
        clusterNetworkMode: calico
    - name: calico
      version: v3.27.3
      param:
        calicoMode: vxlan
    - name: coredns
      version: v1.10.1
    - name: kubeos-addon
      type: chart
      version: 0.0.0-latest
      releaseName: kubeos-addon
      namespace: upgrade-system
      # valuesConfigMapRef is an optional field; when not configured, the chart default values.yaml is used
      valuesConfigMapRef:
        name: kubeos-addon-values
        namespace: bke-cluster
        valuesKey: values.yaml
```

## KubeOS Immutability Constraints

KubeOS adopts a read-only rootfs. The following directories are read-only at runtime. Do not write files or install software packages:

| Directory | Description |
| ---- | ---- |
| `/`, `/usr`, `/bin`, `/sbin`, `/lib`, `/lib64` | System root filesystem and base system directories. |
| `/usr/local/bin` | Traditional bke-agent push path; changed to built-in under KubeOS. |
| `/usr/lib/systemd/system` | systemd unit directory; service files are not written at runtime. |
| `/boot` | Kernel, initramfs, and grub related content. |

The following directories provide persistent writable capability via overlay and bind mount. Configuration rendering and certificate writing of the installation process take place under these directories:

| Directory | Backend Storage | Typical Content |
| ---- | -------- | -------- |
| `/etc` | overlay to `/persist/etc` | Hostname, network, containerd configuration, kubelet configuration, CNI configuration. |
| `/etc/containerd` | `/etc` overlay | `config.toml`, runtime configuration, private repository certificates. |
| `/etc/kubernetes` | `/etc` overlay | kubelet kubeconfig, CA, bootstrap configuration. |
| `/var` | bind to `/persist/var` | Runtime data, logs, container and kubelet data. |
| `/var/lib/kubelet` | `/var` bind | kubelet state, Pod volume, config.yaml. |
| `/var/lib/containerd` | `/var` bind | Image cache, container snapshot, runtime state. |
| `/opt/cni` | overlay to `/persist/opt` | CNI plugin binaries. |

> ![](../figures/icon-note.gif)**Note:**
>
> `/opt/cni` is overlaid to the PERSIST partition via `opt-cni.mount`. Calico can write CNI plugins and configurations in the conventional way, which persist across reboots and A/B upgrades.

## Troubleshooting

### Worker Node Cannot Join the Cluster

1. On the worker VM, execute `ping <management node IP>` to confirm network reachability, and check whether the DHCP or static IP configuration is correct.
2. On the worker VM, execute `date` to compare the time with the management node. Time deviation causes TLS certificate verification failure and repeated bkeagent restarts. On the worker VM, execute `systemctl status chronyd` or `systemctl status ntpd` to confirm that the NTP service is running normally.
3. On the worker node, execute `systemctl status bkeagent` to check the service status. If it shows `failed` or restarts frequently, execute `journalctl -u bkeagent -n 100` to view the logs and confirm whether it can connect to the management cluster API server.

### containerd Configuration Failure

1. On the worker node, execute `cat /etc/containerd/config.toml` to check whether the configuration file exists and is correct.
2. Execute `ls -la /etc/containerd/` to confirm the directory is writable. If the directory does not exist, check whether the KubeOS image correctly built the `/etc` overlay mount.
3. Execute `journalctl -u bkeagent | grep containerd` to view containerd-related logs. If `containerd config version 3 is not supported` is reported, this error does not affect bkeagent operation and can be ignored.

### KubeOS addon Installation Failure

1. On the management cluster node, execute `curl -k https://cr.openfuyao.cn` to confirm the chart repository network is reachable.
2. Execute `kubectl get configmap -n bke-cluster` to confirm that the values.yaml ConfigMap has been created correctly.
3. Execute `kubectl describe pod -n upgrade-system` to view pod events and confirm whether the image can be pulled. If `ImagePullBackOff` is reported, check whether the image repository address in values.yaml and the image have been pushed.
4. Execute `kubectl get nodes --show-labels | grep kubeos` to confirm the worker node label `node.openfuyao.io/os=kubeos` exists.
5. Execute `kubectl describe bkecluster -n bke-cluster` to view the BKECluster status and confirm whether the chartRepo address, version, and valuesConfigMapRef in the addon configuration are correct.

### os-proxy Not Scheduled to KubeOS Worker

1. Execute `kubectl get nodes --show-labels | grep kubeos` to confirm the worker node has been labeled `node.openfuyao.io/os=kubeos`.
2. Execute `kubectl get ds -n upgrade-system -o yaml` to check whether the nodeSelector of the os-proxy DaemonSet is `node.openfuyao.io/os: kubeos`.
3. Execute `kubectl get nodes` to confirm the KubeOS worker node status is Ready. If the node is NotReady, execute `kubectl describe node <node-name>` to view the cause.
