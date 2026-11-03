# vNPU

## Feature Introduction

vNPU is an Ascend NPU computing power partitioning and dynamic scheduling capability component built on top of the container platform. It supports multiple containers sharing the same NPU device while providing isolation capabilities for memory and AI Core resources, thereby significantly improving NPU resource utilization. By providing diverse scheduling strategies, diverse computing power partitioning modes, observability, and other capabilities, it creates an integrated, simple, and easy-to-use Ascend NPU computing power access and management solution.

When users run AI training/inference workloads in a Kubernetes cluster, wish to share expensive Ascend NPU resources at a finer granularity, and require per-container isolation of memory and computing power quotas, the vNPU feature can be used.

### Application Scenarios

- **Multi-tenant Inference Services**: Multiple inference service instances run concurrently on the same Ascend NPU card, with each instance requesting memory and AI Core quotas on demand, reducing the cost per instance.
- **AI Development Environment Sharing**: Multiple algorithm engineers each occupy a vNPU share on the same physical card as a development and debugging environment, without affecting each other's memory usage.
- **Training and Inference Colocation**: Training and inference tasks run simultaneously on the same node, with scheduling policies (binpack/spread) controlling Pod distribution across nodes and devices, balancing resource utilization and fault isolation.
- **Fine-grained Resource Operations**: xpu-exporter integrates with Prometheus to collect vNPU utilization, Pod count, and other metrics, supporting capacity planning and cost accounting.

### Capability Scope

- **Computing Power Partitioning**: Supports soft partitioning (user-space virtualization via CANN Runtime API hijacking) and hard partitioning (virtualization based on Ascend HDK) modes, configurable at the node level.
  - Soft partitioning: A single physical NPU can be partitioned into up to 20 vNPU instances, with AI Core partitioning granularity as fine as 1% and memory partitioning granularity as fine as 1Gi; AI Core and memory are isolated (memory is strictly isolated, AI Core is time-shared with fluctuation less than 10% of the full card); supports limiting memory only without limiting AI Core; supports three AI Core scheduling strategies: fixed quota, elastic, and best-effort; performance overhead less than 5%.
  - Hard partitioning: Based on Ascend HDK, partitions the NPU into multiple vNPU instances mounted into containers for use.
- **Cluster Scheduling**: Extends Volcano with NPU resource scheduling plugins, supporting resource sharing scheduling, binpack/spread scheduling, whole-card scheduling, and [DRA-based scheduling](https://gitcode.com/openFuyao/sig-orchestration-engine/blob/master/docs/en/npu_dra_plugin_user_guide/npu_dra_plugin.md). Allows some containers on the same node to use whole cards while others share a single card.
- **Observability**: xpu-exporter collects vNPU partitioning mode, scheduling strategy, memory utilization, vNPU count, Pod count, and other metrics, integrating with Prometheus.

### Highlight Features

- **Fine-grained Soft Partitioning**: AI Core partitioning granularity as fine as 1%, memory partitioning granularity as fine as 1Gi, offering finer partitioning granularity compared to similar solutions in the industry.
- **Low Performance Overhead in Soft Partitioning**: Performance overhead less than 5%, close to bare-metal computing power.
- **Rich AI Core Scheduling Strategies**: Three strategies — fixed quota, elastic, and best-effort. The elastic strategy improves full-card utilization through a time-slice borrowing mechanism.
- **Flexible Scheduling Policies**: Node-level and device-level binpack/spread support, adapting to different workload characteristics.
- **Whole-card and Partitioned Colocation**: The same node simultaneously supports whole-card containers and shared containers, without requiring a separate cluster.

### Basic Concepts

- **Soft Partitioning (Soft)**: A user-space virtualization scheme based on CANN Runtime API hijacking. AI Core is time-shared, not strictly isolated; memory is strictly isolated.
- **Hard Partitioning (Hard)**: A virtualization scheme based on Ascend HDK, partitioning a physical NPU into multiple vNPU instances mounted into containers for use.
- **vNPU**: Virtual NPU, a logical unit partitioned from a physical NPU. Business Pods request usage through resource declarations such as `huawei.com/vnpu-number`.
- **dieID**: The unique identifier of an NPU device, used to precisely specify which physical card a Pod should be placed on.
- **AI Core Scheduling Strategy**: The allocation method for AI Core time slices in soft partitioning scenarios, including three types: fixed quota (`fixed-share`), elastic (`elastic`), and best-effort (`best-effort`).
- **Node-level Scheduling Policy** (`huawei.com/vnpu-pod-node-scheduler-policy`): Controls Pod distribution across nodes. `binpack` for compact scheduling, `spread` for dispersed scheduling.
- **Device-level Scheduling Policy** (`huawei.com/vnpu-pod-device-scheduler-policy`): Controls vNPU distribution across NPU devices on a single node. `binpack` for compact scheduling, `spread` for dispersed scheduling.

### Implementation Principle

vNPU collaborates through four core components to complete NPU resource discovery, scheduling, mounting, and monitoring. The overall workflow is as follows:

1. **xpu-device-plugin** reports NPU devices and partitionable resources on the node to kubelet through the Kubernetes Device Plugin mechanism (registered as `huawei.com/vnpu-*` resources), and writes device details (dieID, chip type, and other fields) to the node annotation `huawei.com/node-vnpu-register`.
2. **volcano-xpu-plugin** extends the Volcano scheduler. During the Pod scheduling phase, it filters and scores candidate nodes and devices based on the Pod's declared `huawei.com/vnpu-*` resource quantities, partitioning mode, and scheduling strategy, selecting from available vNPUs on candidate nodes.
3. **client_update** deploys the soft partitioning hijack library on the node side. When a business Pod starts, the device plugin mounts the hijack library into the container, implementing AI Core and memory isolation and limiting.
4. **xpu-exporter** continuously collects vNPU runtime metrics (partitioning mode, memory utilization, vNPU count, etc.) and exposes them via Prometheus for monitoring.

Key Pod scheduling moments:

- After a Pod declares `spec.schedulerName: volcano`, scheduling is taken over by Volcano.
- Volcano calls the vxpu plugin to perform NPU resource filtering and scoring, selecting the node and device.
- After the Pod is bound to a node, the device plugin mounts device files and the hijack library into the container through the `Allocate` callback based on the scheduling result.
- When the Pod starts its business process, the hijack library takes over CANN Runtime calls, executing AI Core time-slice scheduling and memory isolation according to the declared quotas.

### Relationship with Related Features

- **Volcano**: vNPU scheduling depends on Volcano 1.15.0 as the scheduler foundation, integrating NPU resource scheduling through Volcano's extension plugin mechanism. Business Pods must explicitly declare `spec.schedulerName: volcano`.
- **Ascend Docker Runtime**: Used for mounting NPU devices into containers. vNPU requires Ascend Docker Runtime to be deployed on the node (version 7.2.RC1+ recommended).
- **Ascend HDK Driver**: NPU hardware driver, required by both soft and hard partitioning. Hard partitioning depends on the device partitioning capability provided by HDK; for supported chip models and HDK version compatibility, see [Appendix - Supported NPU Models](#supported-npu-models).
- **Ascend MindCluster NPU Exporter**: Used for collecting whole-card NPU metrics. vNPU's xpu-exporter only collects vNPU partitioning-dimension metrics; the two are complementary.
- **npu-dra-plugin**: vNPU provides [DRA-based scheduling](https://gitcode.com/openFuyao/sig-orchestration-engine/blob/master/docs/en/npu_dra_plugin_user_guide/npu_dra_plugin.md) as an optional alternative to the Volcano scheduling path.

### Related Examples

For examples of using business Pods, see the [Using vNPU - Configuration Examples](#configuration-examples) section of this document. For more examples and demo projects, refer to the [vNPU repository](https://gitcode.com/openFuyao/vNPU).

## Installation

### Prerequisites

- A Kubernetes cluster is prepared.
- Ascend HDK driver has been installed on nodes as the root user (for supported chip models and HDK version compatibility, see [Appendix - Supported NPU Models](#supported-npu-models)).
- Ascend Docker Runtime has been deployed; version 7.2.RC1+ is recommended. Taking version 7.2.RC1 as an example, for detailed deployment instructions, refer to the [Ascend official documentation](https://www.hiascend.com/document/detail/zh/mindcluster/72rc1/clustersched/dlug/dlug_installation_017.html).
- The build machine has the following tools installed:
  - git
  - docker > 20.10
  - helm

### Starting Installation

#### Build and Compilation

1. Download the source code and initialize submodules.

   ```shell
   git clone https://gitcode.com/openFuyao/vNPU.git
   cd vNPU
   git submodule update --init --recursive
   ```

2. Enter the `ci` directory and execute the build script.

   ```shell
   cd vNPU/ci
   # Build with default parameters, image version defaults to 1.0.0
   sh build.sh
   # Specify image version as 1.1.1
   sh build.sh 1.1.1
   ```

   Build artifacts are output in the `vNPU/output` directory.

#### Online Deployment

1. Add a label to nodes that need vNPU components deployed.

   ```bash
   kubectl label node {nodename} huawei.com/vnpu=ready
   ```

2. Enable NPU device sharing mode. Refer to the [Ascend official documentation](https://support.huawei.com/enterprise/zh/doc/EDOC1100540371/d18b0230?idPath=23710424|251366513|254884019|261408772|261457531). For 910B chips, HDK driver 25.5.0 or above is required to support sharing mode. For example, set the container sharing mode to enabled for all chips on device 0:

   ```bash
   npu-smi set -t device-share -i 0 -d 1
   ```

   >![image](figures/icon-notice.gif) **Notice:**
   >
   >Device sharing mode is a prerequisite for subsequent Volcano scheduling and device plugin resource reporting. It must be configured before deploying vNPU components; otherwise, Pods may fail to schedule due to the node not having sharing enabled. In the `npu-smi` command, `-i` specifies the device ID and `-d` specifies the sharing mode (1 for enabled). Please replace with actual device values. For supported chip models and HDK version compatibility, see [Appendix - Supported NPU Models](#supported-npu-models).

3. Create the xpu Namespace.

   ```bash
   kubectl apply -f charts/yaml/namespace.yaml
   ```

4. Deploy npu-client-update, npu-device-plugin, and xpu-exporter via Helm Chart.

   ```bash
   helm install vxpu oci://cr.openfuyao.cn/charts/vxpu --version 1.0.0
   ```

   >![image](figures/icon-note.gif) **Note:**
   >
   >- `oci://cr.openfuyao.cn/charts/vxpu` pulls the Chart directly from the remote OCI repository; for offline environments, please use the local tgz package instead, see [Offline Deployment](#offline-deployment).
   >- `--version` must match the actual released Chart version number. This document uses `1.0.0` as an example; please refer to the [vNPU repository](https://gitcode.com/openFuyao/vNPU) releases for the actual version.

5. Create Volcano log directories on the host OS and set permissions. Volcano containers run as UID 1000 and require write permission on the directories.

   ```bash
   mkdir -p /var/log/volcano-{admission,controller,scheduler}
   chown -R 1000:1000 /var/log/volcano-{admission,controller,scheduler}
   ```

6. Deploy Volcano via `charts/yaml/volcano-deployment.yaml`.

   ```bash
   kubectl apply -f charts/yaml/volcano-deployment.yaml
   ```

#### Offline Deployment

1. Add a label to nodes that need vNPU components deployed.

   ```bash
   kubectl label node {nodename} huawei.com/vnpu=ready
   ```

2. Enable NPU device sharing mode. Refer to the [Ascend official documentation](https://support.huawei.com/enterprise/zh/doc/EDOC1100540371/d18b0230?idPath=23710424|251366513|254884019|261408772|261457531). For 910B chips, HDK driver 25.5.0 or above is required to support sharing mode. For example, set the container sharing mode to enabled for all chips on device 0:

   ```bash
   npu-smi set -t device-share -i 0 -d 1
   ```

   >![image](figures/icon-notice.gif) **Notice:**
   >
   >Device sharing mode is a prerequisite for subsequent Volcano scheduling and device plugin resource reporting. It must be configured before deploying vNPU components; otherwise, Pods may fail to schedule due to the node not having sharing enabled. In the `npu-smi` command, `-i` specifies the device ID and `-d` specifies the sharing mode (1 for enabled). Please replace with actual device values. For supported chip models and HDK version compatibility, see [Appendix - Supported NPU Models](#supported-npu-models).

3. Create the xpu Namespace.

   ```bash
   kubectl apply -f charts/yaml/namespace.yaml
   ```

4. Import offline image packages. Taking containerd local images as an example, image package names are subject to actual conditions.

   ```shell
   ctr -n k8s.io images import acl_client_update-1.0.0-aarch64.tar
   ctr -n k8s.io images import npu_device_plugin-1.0.0-aarch64.tar
   ctr -n k8s.io images import xpu_exporter-1.0.0-aarch64.tar
   ctr -n k8s.io images import vc_controller_manager-1.15.0-aarch64.tar
   ctr -n k8s.io images import vc_scheduler-1.15.0-aarch64.tar
   ctr -n k8s.io images import vc_webhook_manager-1.15.0-aarch64.tar
   ```

5. Deploy npu-client-update, npu-device-plugin, and xpu-exporter via Helm Chart.

   ```bash
   helm upgrade --install vxpu vxpu-1.0.0.tgz -n xpu --wait
   ```

6. Create Volcano log directories on the host OS and set permissions. Volcano containers run as UID 1000 and require write permission on the directories.

   ```bash
   mkdir -p /var/log/volcano-{admission,controller,scheduler}
   chown -R 1000:1000 /var/log/volcano-{admission,controller,scheduler}
   ```

7. Deploy Volcano via `charts/yaml/volcano-deployment.yaml`.

   ```bash
   kubectl apply -f charts/yaml/volcano-deployment.yaml
   ```

## Using vNPU

### Prerequisites

- Kubernetes cluster is ready.
- Ascend HDK driver has been installed on nodes (for supported chip models and HDK version compatibility, see [Appendix - Supported NPU Models](#supported-npu-models)).
- Ascend Docker Runtime has been deployed (version 7.2.RC1+ recommended).
- vNPU components have been deployed according to the [Installation](#installation) section, and nodes have been labeled with `huawei.com/vnpu=ready`.
- Volcano is available in the cluster where the business Pod resides, and `spec.schedulerName` is set to `volcano`. You can confirm Volcano component running status with the following commands:

  ```bash
  kubectl get deployment -n volcano-system
  kubectl get pod -n volcano-system
  ```
  
  Expected output: All Pods in the volcano-system namespace are in Running state.

### Background Information

Business Pods only need to declare the required vNPU resources without modifying business code. vNPU's scheduler (volcano-xpu-plugin) and device plugin (xpu-device-plugin) automatically schedule Pods to appropriate nodes and NPU devices and mount devices. Pods can configure partitioning mode (soft/hard), AI CPU allocation strategy, soft partitioning computing power scheduling strategy, node-level and device-level scheduling policies, as well as precisely specify dieID and NPU device type through annotations.

The soft partitioning AI Core scheduling strategies are compared as follows.

**Table 1** Soft Partitioning AI Core Scheduling Strategy Comparison

| Mode Name | Feature Description |
| :--- | :--- |
| Fixed Quota Mode (`fixed-share`) | Each vNPU strictly executes time slices according to the configured ratio. a. If there are unallocated vNPU resources, when the last vNPU's time slice is exhausted, it sleeps for a period, during which the NPU idles. b. If no task is running on a vNPU, it still consumes the vNPU's time slice according to the given ratio. |
| Elastic Mode (`elastic`) | Optimizes scheduling logic on top of per-ratio time-slice execution to improve full-card utilization. a. When the last vNPU's time slice is exhausted, the NPU usage right is immediately released to the next vNPU, skipping the sleep logic. b. If no task is running on a vNPU, the unconsumed time slice is directly skipped, switching to another vNPU. c. Time-slice borrowing mechanism: when a running vNPU identifies idle card resources, it allows other vNPUs to borrow the wasted resources within that time slice, preventing card resources from being wasted. |
| Best-effort Mode (`best-effort`) | vNPUs compete for NPU resources on their own. This mode has the highest resource utilization but cannot guarantee vNPU QoS. |

The scheduling strategies are compared as follows.

**Table 2** Scheduling Strategy Comparison

| Strategy Name | Feature Description |
| :--- | :--- |
| Compact Scheduling (`binpack`) | Prioritizes concentrating Pods/vNPUs onto fewer nodes/devices to improve resource utilization per node/device and reduce fragmentation. |
| Dispersed Scheduling (`spread`) | Prioritizes distributing Pods/vNPUs across more nodes/devices to improve load balancing and fault isolation. |

### Usage Restrictions

- **Soft Partitioning**: A container can request at most one vNPU; only one process per container can use vNPU; only AI Core and memory capacity partitioning and limiting are supported, not AI CPU, VPC, VDEC, JPEGD, or other core partitioning control; business containers must not be privileged containers.
- **Supported NPU Chip Models**: 310P3, 910B4 (both soft and hard partitioning supported); 910C is not currently supported (see [Appendix - Supported NPU Models](#supported-npu-models)).
- `huawei.com/vnpu-number` value range is 1 to the total number of physical devices on a single node; values greater than 1 indicate whole-card allocation; for soft/hard partitioning, it can only be 1.
- `huawei.com/vnpu-cores` unit is 1%; for soft partitioning `fixed-share`/`elastic` strategies the value range is 1-100, for `best-effort` strategy the value range is 0-100; for hard partitioning the value range is 1-100; if not configured, defaults to 100.
- `huawei.com/vnpu-memory.1Gi` unit is 1Gi; for soft partitioning it cannot be set to 0, if not configured it defaults to the total memory of a single card; for hard partitioning it does not need to be configured.
- 910B chips require HDK driver 25.5.0 or above to support sharing mode for soft/hard partitioning.
- The business Pod's `spec.schedulerName` must be `volcano`.

>![image](figures/icon-notice.gif) **Notice:**
>
>- Business containers must not be privileged containers; otherwise, the soft partitioning hijack library cannot take effect and resource isolation fails.
>- In soft partitioning scenarios, only one process per container is allowed to use vNPU. Concurrent use of the same vNPU by multiple processes may cause AI Core time-slice scheduling anomalies.
>- `huawei.com/vnpu-memory.1Gi` cannot be set to 0 in soft partitioning mode; otherwise, Pod scheduling will fail.

### Configuration Description

All annotations and resource fields supported by Pods are as follows.

**Table 3** Pod vNPU Configuration Field Description

| Configuration Item | Function | Value Description |
| :--- | :--- | :--- |
| `huawei.com/vnpu-mode` | Configure partitioning mode | `soft`: soft partitioning; `hard`: hard partitioning. Optional, defaults to `soft`. |
| `huawei.com/vnpu-hard-aicpu-level` | Configure hard partitioning AI CPU allocation strategy | `low`: low configuration; `high`: high configuration. Optional, defaults to `low`. |
| `huawei.com/vnpu-soft-scheduler-policy` | Soft partitioning computing power scheduling mode | `fixed-share`: fixed quota; `elastic`: elastic mode; `best-effort`: best-effort mode. Optional, defaults to `fixed-share`. |
| `huawei.com/vnpu-pod-node-scheduler-policy` | Node-level scheduling policy | `binpack`: compact scheduling, prioritizes concentrating Pods onto fewer nodes; `spread`: dispersed scheduling, prioritizes distributing Pods across more nodes. Optional, defaults to `binpack`. |
| `huawei.com/vnpu-pod-device-scheduler-policy` | Device-level scheduling policy | `binpack`: compact scheduling, prioritizes concentrating vNPUs onto fewer NPU devices; `spread`: dispersed scheduling, prioritizes distributing vNPUs across more NPU devices. Optional, defaults to `binpack`. |
| `huawei.com/vnpu-use-npu-dieID` | Specify the dieID of the NPU card | The value is the dieID of the NPU card, e.g., `"37E96E64-20C10A5A-8C747124-6E78030A-BF003019"`. The dieID can be found in the 2nd field of the node annotation `huawei.com/node-vnpu-register`; see the command below. Optional, defaults to empty, matching any dieID. |
| `huawei.com/vnpu-use-npu-type` | Specify the NPU device type | The value is `"NPU-ASCEND-{chip-type}"`, e.g., `"NPU-ASCEND-310P3"`. The device type can be found in the 5th field of the node annotation `huawei.com/node-vnpu-register`; see the command below. Optional, defaults to empty, matching any device type. |
| `spec.schedulerName` | Specify the scheduler name | Must be `volcano`. |
| `huawei.com/vnpu-number` | vNPU count | Required. Value range: 1 to the total number of physical devices on a single node. If greater than 1, indicates whole-card allocation; for soft/hard partitioning, it can only be 1. |
| `huawei.com/vnpu-cores` | AI Core usage rate | Optional. Unit is 1%. Defaults to 100 if not configured. For soft partitioning `fixed-share`/`elastic` strategies the value range is 1-100; for `best-effort` strategy the value range is 0-100; for hard partitioning the value range is 1-100. |
| `huawei.com/vnpu-memory.1Gi` | Memory size | Optional. Unit is 1Gi. For soft partitioning, defaults to the total memory of a single card if not configured, and cannot be set to 0; for hard partitioning, no configuration is needed. |

Example command to view NPU device details (dieID, chip type, and other fields) in the node annotation `huawei.com/node-vnpu-register`:

```bash
# Output the full content of the node vnpu-register annotation
kubectl get node {nodename} -o jsonpath='{.metadata.annotations.huawei\.com/node-vnpu-register}'
# Fields are separated by semicolons; the 2nd field is dieID, the 5th field is chip type
```

>![image](figures/icon-note.gif) **Note:**
>
>- The values of the same field in `resources.limits` and `resources.requests` must be identical; otherwise, kubelet validation fails.
>- Requesting a whole card in soft partitioning mode (`huawei.com/vnpu-number` > 1) is not supported. Please use hard partitioning or explicitly declare `huawei.com/vnpu-mode: "hard"`.
>- `huawei.com/vnpu-use-npu-dieID` and `huawei.com/vnpu-use-npu-type` can be used together to precisely specify the target NPU; when used individually, the matching dimension is limited to that field only.

### Configuration Examples

**Example 1**: Soft partitioning fixed quota mode, declaring 1 vNPU, 50% AI Core, 8Gi memory.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-soft-scheduler-policy: "fixed-share"
spec:
  schedulerName: volcano
  containers:
  - name: test-container
    image: {business-image}
    resources:
      limits:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 50
        huawei.com/vnpu-memory.1Gi: 8
      requests:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 50
        huawei.com/vnpu-memory.1Gi: 8
```

**Example 2**: Soft partitioning best-effort mode, limiting only memory to 8Gi, not limiting AI Core.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-soft-scheduler-policy: "best-effort"
spec:
  schedulerName: volcano
  containers:
  - name: test-container
    image: {business-image}
    resources:
      limits:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 0
        huawei.com/vnpu-memory.1Gi: 8
      requests:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 0
        huawei.com/vnpu-memory.1Gi: 8
```

**Example 3**: Hard partitioning low configuration mode, requesting 50% AI Core.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-mode: "hard"
    huawei.com/vnpu-hard-aicpu-level: "low"
spec:
  schedulerName: volcano
  containers:
  - name: test-container
    image: {business-image}
    resources:
      limits:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 50  # AI Core percentage
      requests:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 50
```

**Example 4**: Declaring use of two whole cards.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
spec:
  schedulerName: volcano
  containers:
  - name: test-container
    image: {business-image}
    resources:
      limits:
        huawei.com/vnpu-number: 2
      requests:
        huawei.com/vnpu-number: 2
```

**Example 5**: Complete configuration example (setting partitioning mode, scheduling policies, dieID, and device type simultaneously).

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-mode: "hard"                                                          # Partitioning mode
    huawei.com/vnpu-hard-aicpu-level: "low"                                               # Hard partitioning AI CPU allocation strategy
    huawei.com/vnpu-soft-scheduler-policy: "elastic"                                     # Soft partitioning computing power scheduling mode
    huawei.com/vnpu-pod-node-scheduler-policy: "spread"                                  # Node-level scheduling policy
    huawei.com/vnpu-pod-device-scheduler-policy: "binpack"                              # Device-level scheduling policy
    huawei.com/vnpu-use-npu-dieID: "37E96E64-20C10A5A-8C747124-6E78030A-BF003019"        # Specify the dieID of the NPU card
    huawei.com/vnpu-use-npu-type: "NPU-ASCEND-310P3"                                      # Specify the NPU device type
spec:
  schedulerName: volcano  # Specify the scheduler name as volcano
  containers:
  - name: test-container
    image: {business-image}
    resources:
      limits:
        huawei.com/vnpu-number: 1        # Required, vNPU count
        huawei.com/vnpu-cores: 5        # Optional, AI Core usage rate available to the container
        huawei.com/vnpu-memory.1Gi: 8   # Optional, memory available to the container
      requests:
        huawei.com/vnpu-number: 1        # Same as limits
        huawei.com/vnpu-cores: 5        # Same as limits
        huawei.com/vnpu-memory.1Gi: 8   # Same as limits
```

### Operation Steps

1. Set the node partitioning mode.

   vNPU supports both soft and hard partitioning modes, configurable at the node level. The node defaults to soft partitioning mode. When deploying the npu-device-plugin service via Helm, the node partitioning mode can be set through `values.yaml` or Helm command-line parameters.

   - Method 1: Set the node partitioning mode in `values.yaml`, then deploy via Helm.

     ```yaml
     vnpuNodeMode:
       worker1:        # Node name
         mode: "hard"  # Node partitioning mode, hard|soft, defaults to soft
       worker2:
         mode: "soft"
     ```

   - Method 2: Set via Helm command-line parameters.

     ```bash
     helm upgrade --install vxpu ./vxpu-1.0.0.tgz -n xpu --set vnpuNodeMode.worker1.mode="hard"
     ```

     >![image](figures/icon-note.gif) **Note:**
     >
     >- The node name (`worker1`) must match the node name output by `kubectl get nodes`.
     >- Nodes not configured in `vnpuNodeMode` default to soft partitioning (`soft`) mode.
     >- The partitioning mode of a node must not conflict with the Pod annotation `huawei.com/vnpu-mode`; otherwise, Pod scheduling will fail.

2. Create a Pod that uses vNPU.

   After the business Pod specifies `spec.schedulerName` as `volcano`, it declares the required vNPU count, AI Core usage rate, and memory size through `resources.limits`/`resources.requests`, and configures the partitioning mode and scheduling policies through `metadata.annotations`. For complete field descriptions, see [Configuration Description](#configuration-description); for available examples, see [Configuration Examples](#configuration-examples).

### Subsequent Operations

- After deployment, use `kubectl get pod -n xpu` to check the running status of npu-client-update, npu-device-plugin, and xpu-exporter components, confirming DaemonSet readiness.
- Check whether the node annotation `huawei.com/node-vnpu=ready` exists, and whether NPU device details (dieID, chip type, and other fields) have been reported in the node annotation `huawei.com/node-vnpu-register`. For the viewing command, see the [Configuration Description](#configuration-description) section.
- If you need to integrate with a monitoring system, see the [vNPU Metric Monitoring](#related-operations) section to configure Prometheus to scrape xpu-exporter.
- If you need to adjust the partitioning mode of a node, set `vnpuNodeMode.<nodename>.mode` during Helm redeployment; no node rebuild is required. Partitioning mode changes only take effect for newly scheduled Pods on that node; already running Pods are not affected. To apply the new partitioning mode to already running Pods, restart the corresponding Pods.

### Related Operations

xpu-exporter supports collecting vNPU-related monitoring metrics, integrating with Prometheus to achieve fine-grained monitoring of NPU resources. Whole-card NPU metrics can be collected by installing the Ascend MindCluster NPU Exporter. The metrics supported by vNPU are as follows.

**Table 4** vNPU Monitoring Metrics

| Metric Name | Description | Labels |
| :--- | :--- | :--- |
| `xpu_vnpu_mode` | vNPU partitioning mode | nodeName, nodeIp, npuUUid |
| `xpu_vnpu_soft_scheduler_policy` | vNPU soft partitioning scheduling mode; not applicable to hard partitioning | nodeName, nodeIp, npuUUid |
| `xpu_vnpu_mem_util` | Memory utilization per vNPU | npuUUid, nodeName, nodeIp, podUid, cntrName, vnpuId, vnpuCoreLimit, vnpuMemLimit |
| `xpu_vnpu_num` | Number of vNPUs allocated per NPU card | nodeName, nodeIp, npuUUid |
| `xpu_vnpu_pod_num` | Number of Pods with allocated vNPUs per NPU | nodeName, nodeIp, npuUUid |

>![image](figures/icon-note.gif) **Note:**
>
>- The above metrics are exposed by xpu-exporter in Prometheus exposition format at the `/metrics` endpoint. The default listening port is subject to the deployment Chart's `values.yaml`.
>- Whole-card NPU metrics (temperature, power, whole-card memory, etc.) require separate installation of the Ascend MindCluster NPU Exporter, complementary to vNPU metrics.
>- `xpu_vnpu_soft_scheduler_policy` only has a value in soft partitioning mode; hard partitioning nodes do not expose this metric.

Use `kubectl describe node {nodename}` to view the Capacity/Allocatable of `huawei.com/vnpu-*` resources on the node, confirming that NPU partitioning resources have been correctly registered. The node annotation `huawei.com/node-vnpu-register` contains details such as dieID and chip type for each NPU card. Pods can precisely specify placement through `huawei.com/vnpu-use-npu-dieID` and `huawei.com/vnpu-use-npu-type`. For the command to view node annotations, see the [Configuration Description](#configuration-description) section.

## Appendix

### Code Structure

- **volcano-xpu-plugin**: Volcano plugin, implements NPU virtualization resource scheduling.
- **xpu-device-plugin**: Device plugin, implements NPU resource reporting and allocation.
- **client_update**: Responsible for deploying the soft partitioning hijack library.
- **xpu-exporter**: Responsible for reporting vNPU metrics.
- **ci**: Build and compilation scripts.
- **charts**: Files related to vNPU component deployment.

### Supported NPU Models

The NPU chip models and software version compatibility supported by vNPU are as follows.

**Table 5** Supported NPU Models and Software Version Compatibility

| Chip Model | Soft Partitioning | Hard Partitioning | HDK Version Compatibility | CANN Version Compatibility (Soft Partitioning) |
| :--- | :--- | :--- | :--- | :--- |
| 310P3 | Supported | Supported | 25.5.0 and above | 8.5.0, 9.1.0 |
| 910B4 | Supported | Supported | 25.5.0 and above | 8.5.0, 9.1.0 |
| 910C | Not supported | Not supported | NA | NA |

>![image](figures/icon-note.gif) **Note:**
>
>The above chip models and software version compatibility have been fully validated. Software and hardware outside this range are not validated and may have compatibility issues. Both 310P3 and 910B4 chips require HDK driver 25.5.0 or above to support device sharing mode for soft/hard partitioning.

### Supported Volcano Versions

Currently Volcano version 1.15.0 is supported. The latest version is to be planned.
