# volcano-config-service

## Feature Introduction

In modern high-performance computing and large-scale distributed systems, Non-Uniform Memory Access (NUMA) architecture is becoming increasingly common. NUMA architecture divides memory into different nodes (NUMA nodes), each node has its own local memory and CPU, aiming to reduce memory access latency and improve system performance. However, the complexity of NUMA architecture increases the difficulty of system resource management, especially in multi-task, multi-threaded environments. To fully utilize the advantages of NUMA architecture, fine-grained management and monitoring of system resources are needed. NUMA resource monitoring visualization aims to display the allocation and usage of NUMA resources in the system in real-time through an intuitive graphical interface, helping users better understand and manage NUMA resources, thereby improving system performance and resource utilization. In container clusters, diverse and rich schedulers provide resource scheduling capabilities. This feature is divided into three perspectives, providing different types of optimization methods for tasks with three different priority levels: high, medium, and low. At the levels of cluster scheduling and intra-node NUMA scheduling, it ensures that Pods achieve optimal allocation effects, achieving significant results in performance improvement. For details, please refer to [NUMA Affinity Scheduling](https://gitcode.com/openFuyao/sig-numa-affinity/blob/master/docs/en/user_guide/numa_aware_scheduling.md#feature-overview).

### Capability Scope

This feature is divided into two code repositories: volcano-config-service and volcano-config-website, containing backend and frontend functions respectively.

- NUMA topology diagram: Intuitively displays the topology structure of each NUMA node in the system, including the CPU and memory distribution of each node. Displays the connection relationships and access latency between nodes.
- Real-time resource monitoring: Real-time displays the CPU usage of each NUMA node, including utilization and idle status. Real-time displays the memory usage of each NUMA node, including total memory, used memory, and free memory.
- Detailed resource information: Displays detailed resource information for each NUMA node, such as CPU list and memory size. Supports viewing the CPU and memory allocation of each container on NUMA nodes.

### Constraints and Limitations

- When using this module, you need to modify the Kubelet's Config file. The specific configuration steps are as follows. This step will be automatically completed when enabling NUMA affinity policy or optimal NUMA Distance policy.

  1. Open the Kubelet configuration file.

      ```shell
      vi /var/lib/kubelet/config.yaml
      ```

  2. Add or modify configuration items.

      ```shell
      cpuManagerPolicy:static
      systemReserved:
          cpu: "0.5"
      kubeReserved:
          cpu: "0.5"
      topologyManagerPolicy:xxx
      ```

  3. Modify the application.

      ```shell
      rm -rf /var/lib/kubelet/cpu_manager_state
      systemctl daemon-reload
      systemctl restart kubelet
      ```

  4. Check the Kubelet running status.

      ```shell
      systemctl status kubelet
      ```

  If the Kubelet running status is running, it indicates success. Modifying the node's topology policy will restart Kubelet, which may cause some Pods to be rescheduled. Please use with caution in production environments.

- When using runtime Pod affinity optimization function, you need to modify the system configuration in advance. The specific steps are as follows.
  1. Modify the operating system cmdline parameters.

     ```shell
     vim /etc/grub2-efi.cfg
     ```

     Add the following to the linux parameters corresponding to the current operating system image:

     ```shell
     mem_sampling_on numa_icon=enable
     ```

     For example:

     ```shell
     linux   /vmlinuz-5.10.0-216.0.0.115.oe2203sp4.aarch64 root=/dev/mapper/openeuler-root ro rd.lvm.lv=openeuler/root rd.lvm.lv=openeuler/swap video=VGA-1:640x480-32@60me cgroup_disable=files apparmor=0 crashkernel=1024M,high smmu.bypassdev=0x1000:0x17 smmu.bypassdev=0x1000:0x15 arm64.nopauth console=tty0 kpti=off mem_sampling_on numa_icon=enable
     ```

  2. Modify the containerd configuration file.

     ```shell
     vi /etc/containerd/config.toml
     ```

     Modify the content in it:

     ```shell
     [plugins."io.containerd.nri.v1.nri"]
     disable = false
     ```

     Restart containerd to apply the update:

     ```shell
     systemctl restart containerd
     ```

  Note: The runtime Pod affinity optimization function currently only supports ARM environment + openEuler 22.03 LTS SP4 environment. And the priority is lower than core binding. When there are Pods with core binding in the cluster, the feature is automatically disabled.

- When using the cluster NUMA monitoring function, you need to configure Prometheus in advance.
  1. Configure Prometheus Operator to access NUMA Exporter, refer to task scenario 2.
  2. Configure Prometheus to access NUMA Exporter, refer to task scenario 3.

## Security Capabilities

For openFuyao v26.06 and earlier versions, only basic security capabilities are provided. Inter-service mutual TLS (mTLS) and component-level user authentication and authorization (RBAC) require users or integration partners to adapt and harden based on their specific deployment environments (e.g., introducing cert-manager, implementing unified certificate management, deploying authentication middleware, and configuring NetworkPolicies to restrict network reachability). The current release focuses on core functionality delivery; security hardening is the responsibility of users. Security capabilities will be gradually planned and introduced in future releases.

## Local Build

### Image Build

#### Build Parameters

- `GOPRIVATE`: Configure Go language private repository, equivalent to `GOPRIVATE` environment variable.
- `COMMIT`: Hash value is the current git commit.
- `VERSION`: Component version.
- `SOURCE_DATE_EPOCH`: Timestamp of the image rootfs.

#### Build Commands

- Build and push to specified OCI repository.

  <details open>
  <summary>Using <code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>
  <details>
  <summary>Using <code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  Where `<path/to/dockerfile>` is the Dockerfile path (build/Dockerfile), `<oci/repository>` is the image address, and `<tag>` is the image tag.

- Build and export OCI Layout to local tarball.

  <details open>
  <summary>Using <code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>
  <details>
  <summary>Using <code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  Where `<path/to/dockerfile>` is the Dockerfile path (build/Dockerfile), `<oci/repository>` is the image address, `<tag>` is the image tag, and `path/to/oci-layout.tar` is the tar package path.

- Build and export image rootfs to local directory.

  <details open>
  <summary>Using <code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>
  <details>
  <summary>Using <code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>

  Where `<path/to/dockerfile>` is the Dockerfile path (build/Dockerfile) and `path/to/output` is the local directory path.

### Helm Chart Build

- Package Helm Chart.

  ```bash
  helm package <path/to/chart> -u \
      --version=0.0.0-latest \
      --app-version=openFuyao-v25.09
  ```

  Where `<path/to/chart>` is the Chart folder path.

- Push Chart package to specified OCI repository.

  ```bash
  helm push <path/to/chart.tgz> oci://<oci/repository>:<tag>
  ```

  Where `<path/to/chart.tgz>` is the Chart package path, `<oci/repository>` is the Chart package push address, and `<tag>` is the Chart package tag.

## Installation

### Prerequisites

- Kubernetes v1.21 and above has been deployed.
- Prometheus has been deployed.
- Containerd has been deployed and NRI function is enabled.

### Start Installation

#### openFuyao Platform

1. In the left navigation bar on the openFuyao platform, select "Application Market > Application List" to enter the "Application List" interface.
2. Select the type "Extension Components" on the left to view all extension components. Or enter "volcano" in the search box.
3. Click the "volcano-config-service" card to enter the scheduling extension component "Details" interface.
4. Click "Deploy" to enter the "Deployment" interface.
5. Enter the application name, select the installation version and namespace.
6. In the "Values.yaml" of parameter configuration, enter the values information to be deployed.
7. Click "Deploy" to complete the deployment.
8. Click "Extension Component Management" in the left navigation bar to manage the scheduling component.

#### Standalone Deployment

Compared to application market installation and deployment, this component provides a standalone deployment function. The steps are as follows:

1. Pull the image.

   ```shell
   helm pull oci://cr.openfuyao.cn/charts/volcano-config-service --version xxx
   ```

   Replace xxx with the helm image version to be pulled, for example: 0.0.0-latest.

2. Extract the installation package.

   ```shell
   tar -zxvf volcano-config-service-xxx.tgz
   ```

3. Disable openFuyao and OAuth switches.

   ```shell
   vim volcano-config-service/charts/volcano-config-website/values.yaml
   ```

   Change the enableOAuth and openFuyao options to false.

4. Standalone installation.

   ```shell
   helm install volcano-config-service ./
   ```

5. Connect to Prometheus.

   Refer to NUMA affinity scheduling development guide [task scenario 2](https://gitcode.com/openFuyao/sig-numa-affinity/blob/master/docs/en/developer_guide/numa-aware_scheduling_development_guide.md#task-scenario-2-connecting-prometheus-operator-to-numa-exporter) and [task scenario 3](https://gitcode.com/openFuyao/sig-numa-affinity/blob/master/docs/en/developer_guide/numa-aware_scheduling_development_guide.md#task-scenario-3-connecting-prometheus-to-numa-exporter).

6. Access the standalone frontend.

   You can access the standalone frontend through http://*management-plane-client-login-IP-address*:30881.
