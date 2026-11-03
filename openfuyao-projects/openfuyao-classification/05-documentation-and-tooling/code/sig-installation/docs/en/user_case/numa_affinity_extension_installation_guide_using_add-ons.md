# NUMA Affinity Extension Installation Guide Using Add-ons

## Background

When using the BKE installation and deployment tool for cluster construction, users can implement:

1. **NUMA Affinity Plugin Installation**: Directly install NUMA affinity plugin through BKE's Addon mechanism during cluster creation, without relying on the openFuyao application marketplace.

## Prerequisites

- Bootstrap node initialization is complete, and the bootstrap cluster is running normally. After the bootstrap node initialization is complete, yaml files with default configurations are generated in the `/bke/cluster/` directory. Among them, `1master-cluster.yaml` is used to create a cluster with 1 master node, and `1master-node.yaml` is the corresponding node information file. For details, see [Service Cluster Deployment Guide (Linux Backend)](../cluster_installation_guide/service_cluster_installation/service_cluster_deployment_guide_(linux_backend).md).
- Cluster configuration files are prepared.

> ![](./figures/icon-note.gif)**Note**:
>
> The `1master-cluster.yaml` and `1master-node.yaml` referenced below are cluster configuration files in the `/bke/cluster/` directory on the bootstrap node, used to define cluster configuration (including Addon configuration) and node information respectively.

## Restrictions

The functions provided by this component depend on the following (see the [component dependencies description of sig-numa-affinity](https://gitcode.com/openFuyao/sig-numa-affinity/blob/master/docs/en/user_guide/numa-aware_scheduling_user_guide.md#component-dependencies)):

- Cluster-level affinity scheduling based on NUMA topology: No hardware dependencies (requires multi-NUMA architecture), no OS dependencies.
- Optimal NUMA distance allocation strategy: No hardware dependencies (requires multi-NUMA architecture), no OS dependencies.
- Network affinity-aware optimization between PODs: Supports the ARM architecture only, the x86 architecture is not supported; the OS must be openEuler 22.03-sp3 or higher, containerd >= 1.7.

## Procedure

1. Configure NUMA affinity plugin.

   Add NUMA affinity plugin configuration in the `addons` section of `1master-cluster.yaml`:

   ```yaml
   spec:
     clusterConfig:
       addons:
       - name: numa-affinity-package
         param:
           openfuyaoRepo: oci://cr.openfuyao.cn/charts
         version: v0.0.2
   ```

   > ![](./figures/icon-note.gif)**Note**:
   >
   > - `openfuyaoRepo`: Chart repository address, please fill in according to actual environment.
   > - `version`: Plugin version number, please ensure compatibility with cluster environment. If you choose to pre-install openFuyao components, select version v0.0.1; if you independently install numa affinity without installing openFuyao, select version v0.0.2.

2. Create cluster.

   Execute cluster creation command:

   ```bash
   bke cluster create -f /bke/cluster/1master-cluster.yaml -n /bke/cluster/1master-node.yaml
   ```

3. Verify installation.

   After cluster creation is complete, execute the following command to verify whether the plugin is successfully installed:

   ```bash
   kubectl get pod -A
   ```

## Configuration Example

The following is a complete configuration snippet including NUMA affinity plugin:

```yaml
spec:
  clusterConfig:
    addons:
    - name: kubeproxy
      param:
        clusterNetworkMode: calico
      version: v1.33.1
    - name: calico
      param:
        calicoMode: vxlan
      version: v3.27.3
    - name: coredns
      version: v1.10.1
    - name: numa-affinity-package
      param:
        openfuyaoRepo: oci://cr.openfuyao.cn/charts
      version: v0.0.2
```
