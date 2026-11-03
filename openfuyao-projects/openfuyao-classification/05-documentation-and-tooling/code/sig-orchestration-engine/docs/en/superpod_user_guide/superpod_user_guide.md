# SuperPod

## Feature Introduction

SuperPod is a Kubernetes cluster topology abstraction entity built on the UBS (Unified Bus System) topology discovery mechanism. It aggregates physical nodes by `superPodId` into logical topology units and manages them in the cluster as custom resources (`SuperPod` CR). The controller periodically collects node topology and memory information, automatically maintains the lifecycle of `SuperPod` CRs, and optionally generates Volcano `HyperNode` CRs to enable the network-topology-aware scheduling capability of the Volcano scheduler. It also provides a standalone `superpod-exporter` component that exposes SuperPod membership, memory borrowing, shared memory, URMA device, and other metrics in Prometheus format, supporting the monitoring platform's visualization and operations of SuperPod resources.

The current version only supports the **FM (Full-Mesh) network topology**: physical nodes within a single SuperPod are fully interconnected (pairwise direct connections, all with hop count 1), and SuperPods are not interconnected with each other (only connected via Eth).

When users need to logically group nodes by physical topology boundaries (UB interconnect domains) in large-scale clusters and want the Volcano scheduler to perform topology-aware scheduling accordingly (e.g., scheduling communication-intensive workloads within the same SuperPod), the SuperPod feature can be used.

### Application Scenarios

- **Large-scale Cluster Tiered Scheduling**: When cluster nodes are numerous and distributed across multiple SuperPods, the SuperPod topology abstraction supports Volcano's network-topology-aware scheduling, placing communication-intensive workloads (such as distributed training) within the same SuperPod to improve communication locality.
- **UB Node Topology Awareness**: In UB interconnect scenarios, nodes with the same `superPodId` are merged. Volcano selects the communication-optimal node combination based on `HyperNode` for deploying distributed training or inference services, avoiding communication performance degradation caused by cross-SuperPod scheduling.
- **Memory Pooling Topology Management**: Works with the container memory borrowing feature to provide topology boundaries for remote memory borrowing, with borrowing preferentially occurring within the same SuperPod.
- **Operations-side Topology Override**: Operations personnel can override a node's SuperPod membership by applying the `unifiedbus.com/superpod` label to the node, without relying on underlying metric reporting.
- **SuperPod Resource Monitoring and Operations**: After cluster administrators deploy `superpod-exporter`, they can view information about each SuperPod, NUMA memory borrowing, shared memory distribution, and URMA device health on the monitoring platform, without logging into nodes to execute CLI commands, facilitating timely detection of resource skew and device failures.

### Capability Scope

- **Architecture Support**: Supports openEuler 24.03 LTS SP3 and above, Kubernetes v1.31.1 and above, ARM64 architecture.
- **Network Topology**: The current version only supports FM networking (full node interconnection, top Tier=1, single Tier-1 subgroup).
- **Topology Abstraction**: Supports aggregating nodes by `superPodId`, automatically generating `SuperPod` CR (cluster scope, group `matrix.openfuyao.cn`, version `v1`).
- **Dual-Source superPodId Resolution**:
  - Priority source 1: Node label `unifiedbus.com/superpod` (operations-side override).
  - Fallback source 2: The `superPodId` field reported in MatrixMetric topology metrics.
- **Memory Management**: Collects node `total`/`used` memory based on NUMA information, writing it to the `spec.groups[].nodes[].memory` field of `SuperPod`.
- **Optional HyperNode Linkage**: Through the `HYPERNODE_ENABLED` environment variable switch, optionally generates Volcano `HyperNode` CR (`topology.volcano.sh/v1alpha1`) for the Volcano scheduler to perform network-topology-aware scheduling.
- **Reconciliation Mechanism**: MatrixMetric CR changes trigger debounced reconciliation (5s debounce), with periodic full resync (default 5min); topology data refreshed every 12h, NUMA data refreshed every 30s.
- **Metric Collection and Reporting**: The standalone `superpod-exporter` DaemonSet component exposes SuperPod membership, NUMA memory borrowing, shared memory provisioning, URMA device information and health metrics in Prometheus format, exposed via the `:9102/metrics` endpoint by default, and scraped by the monitoring platform for aggregation.
- **Specification Limits**:
  - All nodes under the same `superPodId` are placed into the same Tier-1 group (`group-0`).
  - `HyperNode` linkage requires Volcano to be installed in the cluster and the `hypernodes` CRD to be registered.

> ![icon](./figures/icon-note.gif) **Note:**<br />
> A single kube-matrix-agent instance can manage up to 150 Pods, 300 containers, and 300 processes. This is a general specification limit of the matrix-agent component and is not a constraint unique to the SuperPod feature.

### Highlight Features

- **Dual-Source superPodId**: Node labels take priority over metrics, ensuring that the operations side can override topology membership by applying labels without relying on underlying metric reporting or restarting components.
- **Declarative Reconciliation**: MatrixMetric CR changes automatically trigger reconciliation with eventual consistency, without manual intervention.
- **Pluggable HyperNode**: Does not produce `HyperNode` CR by default; only links when `HYPERNODE_ENABLED=true` is explicitly set and the `HyperNode` CRD is registered, avoiding overhead for clusters without Volcano installed.
- **CRD Readiness Awareness**: Automatically retries with backoff when `SuperPod`/`HyperNode` CRDs are not ready, without crashing or affecting existing controllers.
- **Standalone Metric Exporter**: `superpod-exporter` is deployed as an independent DaemonSet; failure in one metric category does not block others, and `/metrics` remains accessible; the SuperPod dimension `superpod_name` label is derived from node labels and can immediately support SuperPod-level aggregation.

### Basic Concepts

- **SuperPod**: A super node, a logical topology unit formed by aggregating physical nodes with the same `superPodId`, corresponding to the `SuperPod` CR (`matrix.openfuyao.cn/v1`, cluster-scoped).
- **superPodId**: SuperPod identifier, directly determining the SuperPod name (`superpod-<superPodId>`); can be overridden via the node label `unifiedbus.com/superpod`.
- **FM Networking**: A network topology where physical nodes within a SuperPod are fully interconnected (pairwise direct connections, all with hop count 1) and SuperPods are not interconnected, with the top Tier=1.
- **HyperNode**: A topology-level CR defined by Volcano (`topology.volcano.sh/v1alpha1`), produced by this feature's controller when `HYPERNODE_ENABLED=true`, for the Volcano scheduler to perform network-topology-aware scheduling.
- **Tier-1 Subgroup**: A topology subgroup within a SuperPod divided by 1-hop connected components. Under FM networking, each SuperPod contains only one Tier-1 subgroup (`group-0`), which includes all nodes under that SuperPod.
- **superpod-exporter**: A standalone SuperPod metric export component, deployed as a DaemonSet on each physical node of the SuperPod, exposing metrics in Prometheus text format at the `:9102/metrics` endpoint, scraped and aggregated by the monitoring platform.

### Implementation Principle

#### Overall Architecture

Overall approach: matrixagent collects single physical node topology (including `superPodId`) and reports it via MatrixMetric CR → matrixcontroller aggregates and assembles `SuperPod` resources by `superPodId` → Volcano topology-aware scheduling.

- Collection and reporting reuse the existing matrixagent DaemonSet framework and MatrixMetric CR, adding the `node_network_topology_info` metric item.
- The assembly controller is embedded within the existing matrixcontroller process, running in parallel with the existing container escape alert controller without interference.
- `SuperPod` is a new CRD added in this repository (`matrix.openfuyao.cn/v1`, cluster-scoped).
- Whether to assemble Volcano `HyperNode` resources is controlled by the environment variable `HYPERNODE_ENABLED`, **default `false` (disabled)**. When disabled, the Controller only produces `SuperPod` (the `hyperNodeRef` field is left empty) and does not depend on Volcano or the HyperNode CRD; when enabled, it additionally produces `HyperNode` and populates the `hyperNodeRef` reference in `SuperPod`.

#### Workflow

SuperPod is assembled and maintained by the controller goroutine within matrixcontroller. The overall workflow is as follows:

1. **Topology Reporting**: matrixagent collects the local node's `superPodId` and neighbor link information, writing topology information and NUMA memory information into the MatrixMetric CR.
2. **Event Triggering**: Add, modify, and delete events on MatrixMetric CRs trigger debounced reconciliation (5s debounce window); a periodic full resync is also executed every 5min.
3. **Reconciliation Flow**: The controller executes a complete reconcile:
   1. Verifies that the `SuperPod` CRD is registered; if not ready, retries with backoff.
   2. If `HYPERNODE_ENABLED=true` is set, verifies that the `HyperNode` CRD is registered; if not ready, skips HyperNode assembly and only produces `SuperPod`.
   3. Reads all MatrixMetric CRs and Node labels, parsing each node's `superPodId` and memory information.
   4. **superPodId Resolution**: Node label takes priority, metric field is the fallback; if both are empty, the node is skipped.
   5. Groups all nodes by `superPodId`. Under FM networking, all nodes with the same `superPodId` are placed into a single Tier-1 group (`group-0`).
   6. If HyperNode is enabled, an additional Tier-1 `HyperNode` CR is generated for each `superPodId`.
   7. Creates or updates each SuperPod/HyperNode CR, and cleans up stale CRs that no longer exist.

#### Metric Collection Mechanism

`superpod-exporter` is deployed as an independent DaemonSet within the SuperPod (selecting nodes with the `unifiedbus.com/superpod` label via node affinity), exposing Prometheus metrics via HTTP `/metrics` (default `:9102/metrics`), scraped by the cluster monitoring platform. Metrics are **node-level**: all metrics carry `node`/`slot_id` labels; although URMA is SuperPod-granularity data, each node reports in full and is aggregated to the SuperPod-level view via PromQL `min by`. The collection cycle is driven by the Prometheus `scrape_interval` (recommended 30s); the exporter caches low-frequency data (topology) for 12h and collects high-frequency data (borrowing/devices) in real time.

> ![icon](./figures/icon-note.gif) **Note:**<br />
> - The `superpod_name` label is taken from the local node's K8s Node label `unifiedbus.com/superpod`, cached for 12h. When the label is missing, it is `"unknown"`.
> - URMA device data is SuperPod-granularity data, using a per-node full reporting + PromQL `min by (superpod_name, device_name)` aggregation deduplication pattern (fault-first: if any node observes a fault, it is judged as faulty).

### Relationship with Related Features

- **Depends on UBS Engine**: The topology information reported by matrixagent depends on the underlying ubs-engine and its topology discovery component, which must be pre-installed. The UBS Engine SDK socket (`/run/ubse`) must be available.
- **Shares Components with Container Memory Borrowing Feature**: SuperPod and the container memory borrowing feature share the matrixagent and matrixcontroller components, with an identical deployment process (see the [Installation](#installation) section).
- **Optional Volcano HyperNode Linkage**: Requires Volcano to be installed in the cluster and the `hypernodes` CRD (`topology.volcano.sh/v1alpha1`) to be registered. The Volcano Scheduler must enable the network-topology feature to consume `HyperNode` for network-topology-aware scheduling. When Volcano is not installed, keep `HYPERNODE_ENABLED=false` (default), which only produces `SuperPod` CR.
- **Relationship with Prometheus/Grafana Monitoring Platform**: `superpod-exporter` exposes `ubs_*` metrics in Prometheus text format. The cluster must have Prometheus (scraper) and Grafana (visualization, optional) deployed to aggregate and view them. Without a monitoring platform, the exporter still runs but the metrics are not consumed. `superpod-exporter` does not create any CRs and does not affect scheduling or existing resources.

### Related Examples

For examples of using business Pods, see the [Configuration Examples](#configuration-examples) section of this document, including complete examples of SuperPod CR, HyperNode CR, and deploying business Pods to the same SuperPod using Volcano gang scheduling. `superpod-exporter` metric output examples and Grafana PromQL aggregation examples are also in the [Configuration Examples](#configuration-examples) section.

## Installation

### Prerequisites

* **Operating System:** openEuler 24.03 LTS SP3 or higher
* **CPU Architecture:** ARM64
* **Memory:** 64GB or more
* **Disk:** SSD, IOPS 500MB/s
* **Chip Interconnect:** UB
* **User Permissions:** root permissions required for installation and management
* **Software Requirements:**
    1. Kubernetes v1.31.1 or higher.
    2. Refer to [ubs-engine](https://gitcode.com/openeuler/ubs-engine) to install ubs-engine and its dependency components, ensuring the UBS Engine SDK socket (`/run/ubse`) is available. The node topology reporting feature requires ubs-engine version v0.1.7 or higher.
    3. Refer to the [Helm installation documentation](https://helm.sh/docs/intro/install/) to install Helm.
    4. (Optional) If you need to use the `superpod-exporter` metric collection and reporting capability, the cluster must have Prometheus (scraper) and Grafana (visualization, optional) deployed, and the target nodes must have the `unifiedbus.com/superpod` label configured.

### Starting Installation

1. Build Instructions.

    1.1 Pull the source code.

    ```shell
    git clone -b master https://gitcode.com/openFuyao/ubs-k8s-enable.git
    ```

    1.2 Install dependencies.

    Before building, ensure the following tools are installed on the host (version requirements as follows):

    ```shell
    docker  # version requirement > 20.10
    helm    # version requirement v3 and above
    ```
    The Dockerfile uses BuildKit features. Please ensure BuildKit is enabled before executing `docker build`.

    1.3 Build images.

    ```shell
    # Version number example, can be adjusted according to actual release version
    export VERSION=1.0.0
    export DOCKER_BUILDKIT=1

    # Build matrixagent image
    # To use a custom image repository, replace cr.openfuyao.cn with the actual image repository address
    docker build -f build/matrixagent.dockerfile -t cr.openfuyao.cn/openfuyao/matrixagent:${VERSION} .

    # Build matrixcontroller image
    docker build -f build/matrixcontroller.dockerfile -t cr.openfuyao.cn/openfuyao/matrixcontroller:${VERSION} .
    ```

    1.4 Export image packages.

    ```shell
    mkdir -p output

    docker save cr.openfuyao.cn/openfuyao/matrixagent:${VERSION} | gzip -c > output/ubs-k8s.matrixagent.image.${VERSION}.aarch64.tgz
    docker save cr.openfuyao.cn/openfuyao/matrixcontroller:${VERSION} | gzip -c > output/ubs-k8s.matrixcontroller.image.${VERSION}.aarch64.tgz
    ```

    1.5 Package Helm Chart.

    ```shell
    helm package charts/matrixagent --destination output
    helm package charts/matrixcontroller --destination output
    mv output/matrixagent-*.tgz output/ubs-k8s.matrixagent.chart.${VERSION}.aarch64.tgz
    mv output/matrixcontroller-*.tgz output/ubs-k8s.matrixcontroller.chart.${VERSION}.aarch64.tgz
    ```
    Build artifacts are as follows:
    ```
        └── output
        ├── ubs-k8s.matrixagent.image.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixagent.chart.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixcontroller.image.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixcontroller.chart.${VERSION}.aarch64.tgz
    ```

    1.6 (Optional) Build superpod-exporter image and Chart.
    If you need to use the metric collection and reporting capability, you must additionally build the `superpod-exporter` image and Chart.

    ```shell
    export VERSION=1.0.0
    export DOCKER_BUILDKIT=1

    # Build superpod-exporter image
    # To use a custom image repository, replace cr.openfuyao.cn with the actual image repository address
    docker build -f build/superpodexporter.dockerfile -t cr.openfuyao.cn/openfuyao/superpod-exporter:${VERSION} .

    # Export image package
    docker save cr.openfuyao.cn/openfuyao/superpod-exporter:${VERSION} | gzip -c > output/ubs-k8s.superpodexporter.image.${VERSION}.aarch64.tgz

    # Package Helm Chart
    helm package charts/superpodexporter --destination output
    mv output/superpod-exporter-*.tgz output/ubs-k8s.superpodexporter.chart.${VERSION}.aarch64.tgz
    ```

    Build artifacts updated as follows:
    ```
        └── output
        ├── ubs-k8s.matrixagent.image.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixagent.chart.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixcontroller.image.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixcontroller.chart.${VERSION}.aarch64.tgz
        ├── ubs-k8s.superpodexporter.image.${VERSION}.aarch64.tgz
        ├── ubs-k8s.superpodexporter.chart.${VERSION}.aarch64.tgz
    ```

2. Deployment Steps.
    Execute the following command to set version variables:

    ```bash
    export VERSION=1.0.0
    export OCI_VERSION=0.0.0-latest
    ```

    > ![icon](./figures/icon-note.gif) **Note:**<br />
    > `VERSION` is used for the offline method (Method 1) to match the local build artifact version number; `OCI_VERSION` is used for the online method (Method 2) to pull the Chart version from the OCI repository. The two are independent; set one as needed for your scenario.

    2.1 Obtain deployment files.
    Choose one of the following methods to obtain the images and Helm Chart needed for deployment, depending on your scenario.

    - Method 1: Use offline release artifacts.

    Prepare the following files:

    * `ubs-k8s.matrixagent.image.${VERSION}.aarch64.tgz`
    * `ubs-k8s.matrixagent.chart.${VERSION}.aarch64.tgz`
    * `ubs-k8s.matrixcontroller.image.${VERSION}.aarch64.tgz`
    * `ubs-k8s.matrixcontroller.chart.${VERSION}.aarch64.tgz`

    - Method 2: Obtain from image repository and OCI repository.

    Pull images:

    ```bash
    docker pull cr.openfuyao.cn/openfuyao/matrixcontroller:latest
    docker pull cr.openfuyao.cn/openfuyao/matrixagent:latest
    ```

    Pull Helm Chart:

    ```bash
    helm pull oci://cr.openfuyao.cn/charts/matrixagent --version ${OCI_VERSION}
    helm pull oci://cr.openfuyao.cn/charts/matrixcontroller --version ${OCI_VERSION}
    ```

   2.2 Import offline images (offline method only).

    ```bash
    gunzip -c ubs-k8s.matrixagent.image.${VERSION}.aarch64.tgz | ctr -n k8s.io images import -
    gunzip -c ubs-k8s.matrixcontroller.image.${VERSION}.aarch64.tgz | ctr -n k8s.io images import -
    ```
    > ![icon](./figures/icon-note.gif) **Note:**<br />
    > The image packages exported in step 1.4 using `docker save` are in docker tar format. `ctr images import` is compatible with this format and can import directly without additional conversion.
    > If using "Method 2" to pull images directly from the image repository, this step can be skipped.

    2.3 Deploy services.
    Choose one of the following methods to deploy services depending on your scenario.
    - Deploy using offline Chart.

    ```bash
    helm install matrixagent ubs-k8s.matrixagent.chart.${VERSION}.aarch64.tgz -n kube-system \
      --set images.matrixagent.tag=${VERSION}
    helm install matrixcontroller ubs-k8s.matrixcontroller.chart.${VERSION}.aarch64.tgz -n kube-system \
      --set images.matrixcontroller.tag=${VERSION}
    ```
    - Deploy using OCI Chart.

    ```bash
    helm install matrixagent oci://cr.openfuyao.cn/charts/matrixagent --version ${OCI_VERSION} -n kube-system \
      --set images.matrixagent.tag=latest
    helm install matrixcontroller oci://cr.openfuyao.cn/charts/matrixcontroller --version ${OCI_VERSION} -n kube-system \
      --set images.matrixcontroller.tag=latest
    ```
    2.4 Verify results.
    Execute the following command to check Pod status.

    ```bash
    kubectl get pods -A
    ```

    Expected results:
    * Each node should have a corresponding `matrixagent` related Pod with status `Running`.
    * The cluster should have `matrixcontroller` related Pods with status `Running`.

    2.5 (Optional) Deploy superpod-exporter.
    If you need to use the metric collection and reporting capability, deploy the `superpod-exporter` DaemonSet.

    - Method 1: Deploy using offline Chart.

    ```bash
    helm install superpod-exporter ubs-k8s.superpodexporter.chart.${VERSION}.aarch64.tgz -n kube-system \
      --set image.tag=${VERSION}
    ```

    - Method 2: Deploy using OCI Chart.

    ```bash
    helm install superpod-exporter oci://cr.openfuyao.cn/charts/superpod-exporter --version ${OCI_VERSION} -n kube-system \
      --set image.tag=latest
    ```

    > ![icon](./figures/icon-note.gif) **Note:**<br />
    > The superpod-exporter DaemonSet is configured by default with node affinity to select nodes with the `unifiedbus.com/superpod` label, exposes port 9102, and is associated with a ServiceAccount. To integrate with Prometheus Operator auto-discovery, set `serviceMonitor.enabled=true` during deployment. Before deployment, confirm that the target nodes have the `unifiedbus.com/superpod` label configured and that the UBS socket is accessible.

    Verify superpod-exporter deployment results:

    ```bash
    kubectl get pods -n kube-system -l app.kubernetes.io/name=superpod-exporter -o wide
    ```

    Expected result: Each node configured with the `unifiedbus.com/superpod` label should have a corresponding `superpod-exporter` Pod with status `Running`.

## Using SuperPod

### Prerequisites

- The matrixagent and matrixcontroller components have been deployed according to the [Installation](#installation) section, and all component Pods are in `Running` status.
- The UBS Engine SDK socket (`/run/ubse`) is available.
- (Optional) If you need to enable Volcano HyperNode linkage, Volcano must be installed and the `hypernodes` CRD (`topology.volcano.sh/v1alpha1`) registered, and the Volcano Scheduler must have the network-topology feature enabled.
- (Optional) If you need to view SuperPod metrics, `superpod-exporter` must be deployed as described in step 2.5 of the [Installation](#installation) section, and Prometheus scraping and Grafana visualization must be deployed.

### Background Information

In large-scale clusters or UB interconnect scenarios, scheduling from a single-node perspective cannot perceive physical topology boundaries, easily leading to workload dispersion across SuperPods and decreased communication efficiency. By deploying the SuperPod controller within the UBS K8S Enable components, physical nodes can be aggregated by `superPodId` into logical topology units and exposed to the upper-layer scheduler as `SuperPod` CRs. When `HYPERNODE_ENABLED=true` is enabled, the controller additionally produces Volcano `HyperNode` CRs, enabling the Volcano scheduler to perform network-topology-aware scheduling based on topology tiers, placing communication-intensive workloads (such as distributed training) within the same SuperPod to improve communication locality and fault isolation. The SuperPod controller automatically maintains the lifecycle of `SuperPod`/`HyperNode` CRs in a declarative manner, without manual intervention.

The current version does not support Clos networking.

The `superpod-exporter` component exposes SuperPod membership, NUMA memory borrowing, shared memory, URMA device, and other metrics in Prometheus format. Administrators can view the resource usage and health of each SuperPod on the monitoring platform without logging into nodes to execute CLI commands, facilitating timely detection of resource skew and device failures (for metric definitions, see [Configuration Description - Metric Definition](#metric-definition); for usage examples, see [Configuration Examples](#configuration-examples)).

### Usage Restrictions

- **Network Topology Limitation**: Clos networking is not supported.
- **Architecture Limitation**: Only ARM64 architecture is supported.
- **superPodId Source Requirement**: Nodes must have at least one of the following superPodId sources; otherwise, the node will not be included in any `SuperPod`:
  - Node label `unifiedbus.com/superpod`;
  - The `node_network_topology_info` reported by matrixagent contains a non-empty `superPodId` field.
- **Tier Limitation**: Under FM networking, the top Tier is always 1, and all nodes with the same `superPodId` are placed into the same Tier-1 group (`group-0`). The Volcano topology constraint `highest-tier` can only be `"1"`.
- **HyperNode Linkage Prerequisite**: When `HYPERNODE_ENABLED=true` is enabled, the cluster must have Volcano installed and the `hypernodes` CRD registered; otherwise, the controller will skip HyperNode assembly and only produce `SuperPod` CR.
- **Override Semantics**: Node label is the priority source for `superPodId`. If a node has both a label and a metric `superPodId`, the label takes precedence; the metric value is only used as fallback when the label is empty.
- **Metric Exporter Limitations**:
  - When the node label `unifiedbus.com/superpod` is missing or RBAC permissions are insufficient, `superpod_name` falls back to `"unknown"`, SuperPod-level aggregation is unavailable, but node-level metrics are normal.
  - When URMA service is not supported, `ubs_urma_*` metrics are missing, but other metric categories are normal.

> ![image](./figures/icon-notice.gif) **Notice:**
>
> - `SuperPod` CR is a cluster-scoped resource. `metadata.name` is automatically generated by the controller following the `superpod-<superPodId>` rule. Do not manually create or rename it; otherwise, it will be treated as a stale resource and deleted by the controller.
> - After modifying a node label, the controller will take effect on the next debounce or periodic resync (up to 5min), without restarting matrixcontroller.

### Configuration Description

The `SuperPod` CRD is registered under group `matrix.openfuyao.cn`, version `v1`, cluster scope, with resource name `superpods`. Its field descriptions are as follows.

**Table 3** SuperPod CR Field Description

| Field Path | Type | Description |
| :--- | :--- | :--- |
| `spec.superPodId` | string | Required. SuperPod identifier, directly determining the SuperPod name (`superpod-<superPodId>`). |
| `spec.tier` | integer | Required. Top tier of the SuperPod. Fixed at 1 under FM networking. |
| `spec.hyperNodeRef` | string | Optional. References the top-level Volcano HyperNode name, populated only when `HYPERNODE_ENABLED=true`, in the format `hn-t1-<superPodId>`. |
| `spec.groups[]` | array | Required. List of Tier-1 topology subgroups. Under FM networking, each SuperPod contains only one group (`group-0`), including all nodes under that SuperPod. |
| `spec.groups[].name` | string | Group name, in the format `group-<ordinal>` (`group-0` under FM). |
| `spec.groups[].tier` | integer | Group tier, fixed at 1 under FM networking. |
| `spec.groups[].hyperNodeRef` | string | Optional. References the corresponding Tier-1 HyperNode for this group, populated only when `HYPERNODE_ENABLED=true`. |
| `spec.groups[].nodes[]` | array | List of node resource information under the group. |
| `spec.groups[].nodes[].name` | string | Node name. |
| `spec.groups[].nodes[].ip` | string | Node internal IP address. |
| `spec.groups[].nodes[].memory.total` | string | Total physical memory of the node (BinarySI, e.g., `256Gi`). |
| `spec.groups[].nodes[].memory.used` | string | Used physical memory of the node (BinarySI). |
| `status.nodeCount` | integer | Number of member nodes in the SuperPod. |
| `status.conditions[]` | array | List of status conditions, following the Kubernetes Condition specification. |
| `metadata.annotations["superpod.matrix.huawei.com/node-hash"]` | string | The first 8 characters of SHA256 of sorted member node names, used as a topology fingerprint for membership change verification. |

#### Metric Definition

The Prometheus metrics produced by `superpod-exporter` share the `ubs_` prefix, with units in bytes. Metrics are node-level, all carrying `node`/`slot_id` labels, with the SuperPod dimension carried by the `superpod_name` label.

**Table 5** superpod-exporter Common Label Semantics

| Label | Description |
| :--- | :--- |
| `superpod_name` | SuperPod name, taken from node label `unifiedbus.com/superpod` |
| `node` | K8s node name |
| `slot_id` | UBS physical node unique identifier |
| `export_node` | K8s node name of the lending/providing node |
| `export_slot_id` | UBS slot_id of the lending/providing node |
| `name` | Name of the borrowed/shared resource |
| `numa_id` | Remote NUMA id formed by borrowing |
| `device_name` | URMA device name |
| `hw_res_id` | URMA hardware resource ID |

**Table 6** superpod-exporter Metric Definition

| Metric Name | Description | Data Type | Metric Value | Metric Labels |
| :--- | :--- | :--- | :--- | :--- |
| `ubs_exporter_up` | superpod-exporter readiness status | Gauge | 1=ready, 0=unavailable | none |
| `ubs_superpod_info` | Node and SuperPod membership information | Gauge | Always 1 | `superpod_name`, `node`, `slot_id` |
| `ubs_mem_numa_borrow_bytes` | Size of NUMA remote memory borrowed by this node | Gauge | NUMA borrow size (bytes) | `superpod_name`, `node`, `slot_id`, `export_node`, `export_slot_id`, `numa_id`, `name` |
| `ubs_mem_numa_borrow_count` | Number of NUMA borrow relationships on this node | Gauge | Total NUMA borrow relationships | `superpod_name`, `node`, `slot_id` |
| `ubs_mem_shm_provide_bytes` | Size of shared memory provided by this node | Gauge | Shared memory size (bytes) | `superpod_name`, `node`, `slot_id`, `name` |
| `ubs_mem_shm_provide_count` | Number of shared memory regions provided by this node | Gauge | Number of shared memory regions | `superpod_name`, `node`, `slot_id` |
| `ubs_urma_device_info` | URMA device information (used for discovery/association) | Gauge | Always 1 | `superpod_name`, `node`, `slot_id`, `device_name`, `hw_res_id` |
| `ubs_urma_device_healthy` | URMA device health status | Gauge | 1=healthy, 0=fault | `superpod_name`, `node`, `slot_id`, `device_name` |

### Configuration Examples

**Table 2** FM Networking Assembly Result

Taking a cluster with 2 SuperPods (`superPodId=0`, `superPodId=1`, 16 nodes total) as an example, the assembly output (grouped into 2 SuperPods by `superPodId`, with a single Tier-1 subgroup per SuperPod under FM full interconnection, topTier=1):

| Tier | HyperNode | Member Type | Members | Description |
| :--- | :--- | :--- | :--- | :--- |
| Tier 1 | `hn-t1-0` | Node | node1~node8 | 8-node 1-hop connected component within SuperPod A. |
| Tier 1 | `hn-t1-1` | Node | node9~node16 | 8-node 1-hop connected component within SuperPod B. |

**Example 1**: SuperPod CR (FM networking, 8-node full interconnection, `HYPERNODE_ENABLED=true`).

```yaml
apiVersion: matrix.openfuyao.cn/v1
kind: SuperPod
metadata:
  name: superpod-0               # Name directly derived from superPodId
  annotations:
    superpod.matrix.huawei.com/node-hash: a1b2c3d4   # First 8 characters of sorted member node name hash
spec:
  superPodId: "0"                 # SuperPod identifier
  tier: 1                         # FM full interconnection, Tier-1 only
  hyperNodeRef: hn-t1-0           # References top-level HyperNode (populated when HYPERNODE_ENABLED=true)
  groups:                         # Single group (all 8 nodes 1-hop connected)
    - name: group-0
      tier: 1
      hyperNodeRef: hn-t1-0       # References the same-tier Tier-1 HyperNode
      nodes:
        - name: node1
          ip: "10.8.0.1"
          memory:
            total: "256Gi"
            used: "128Gi"
        # ... node2 ~ node7
        - name: node8
          ip: "10.8.0.8"
          memory:
            total: "256Gi"
            used: "120Gi"
status:
  nodeCount: 8
```

> ![icon](./figures/icon-note.gif) **Note:**<br />
> When `HYPERNODE_ENABLED=false` (default), the `hyperNodeRef` field is left empty. The SuperPod still fully carries `superPodId`, topology tier, and node resources, just without referencing HyperNode. The above example shows the enabled form; when disabled, remove the `hyperNodeRef` lines.

**Example 2**: HyperNode CR (FM networking, Tier-1, 8-node full interconnection, automatically produced by the controller when `HYPERNODE_ENABLED=true`).

```yaml
apiVersion: topology.volcano.sh/v1alpha1
kind: HyperNode
metadata:
  name: hn-t1-0
spec:
  tier: 1
  tierName: "superpod"
  members:
    - type: Node
      selector:
        exactMatch:
          name: node1
    # ... node2 ~ node7
    - type: Node
      selector:
        exactMatch:
          name: node8
status:
  nodeCount: 8
```

**Example 3**: Three-Pod Gang Deployment to the Same SuperPod (FM networking, `highest-tier=1`).

Scenario: 3 business Pods (e.g., distributed training Workers) need to be deployed within the same SuperPod to use high-speed interconnect and pooled memory communication, with gang scheduling required (all 3 must be scheduled successfully, otherwise all wait).

1. PodGroup: Declare gang policy + topology hard constraint.

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: gang-in-superpod
  annotations:
    volcano.sh/network-topology-mode: "hard"       # Hard constraint: Pods in the same group must land on the same HyperNode
    volcano.sh/network-topology-highest-tier: "1" # FM networking has only Tier-1, constraining to the same SuperPod
spec:
  minMember: 3                    # Gang policy: all 3 Pods must be scheduled successfully, otherwise all pending
  queue: default
  priorityClassName: high
```

2. Business Pods: Associate with PodGroup, scheduled by Volcano.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: worker-0
  annotations:
    scheduling.k8s.io/group-name: gang-in-superpod   # Associate with PodGroup
spec:
  schedulerName: volcano                               # Use Volcano scheduler
  containers:
    - name: worker
      image: registry.example.com/app/worker:1.0
      resources:
        requests: { cpu: "8", memory: "16Gi" }
        limits:   { cpu: "8", memory: "16Gi" }
---
apiVersion: v1
kind: Pod
metadata:
  name: worker-1
  annotations:
    scheduling.k8s.io/group-name: gang-in-superpod
spec:
  schedulerName: volcano
  containers:
    - name: worker
      image: registry.example.com/app/worker:1.0
      resources:
        requests: { cpu: "8", memory: "16Gi" }
        limits:   { cpu: "8", memory: "16Gi" }
---
apiVersion: v1
kind: Pod
metadata:
  name: worker-2
  annotations:
    scheduling.k8s.io/group-name: gang-in-superpod
spec:
  schedulerName: volcano
  containers:
    - name: worker
      image: registry.example.com/app/worker:1.0
      resources:
        requests: { cpu: "8", memory: "16Gi" }
        limits:   { cpu: "8", memory: "16Gi" }
```

> ![icon](./figures/icon-note.gif) **Note:**<br />
> - **gang + hard topology combination**: Volcano first performs gang check (`minMember`), then topology constraint validation; in `hard` mode, if no Tier-1 HyperNode can accommodate all 3 Pods, all will be pending, with no partial scheduling.
> - **soft mode** (optional): Change `mode` to `soft`, then topology becomes a scoring preference rather than a hard constraint, preferring scheduling to the same SuperPod but allowing fallback to other SuperPods.
> - **FM networking `highest-tier`**: Under FM, SuperPods are not interconnected, so `highest-tier` can only be `"1"`.

**Example 4**: superpod-exporter metric output example (`curl <node>:9102/metrics`, excerpt).

```
# HELP ubs_exporter_up superpod-exporter is up and UBSE SDK is initialized (1=up, 0=SDK unavailable).
# TYPE ubs_exporter_up gauge
ubs_exporter_up 1
# HELP ubs_superpod_info SuperPod membership info: which SuperPods exist and which physical nodes belong to each.
# TYPE ubs_superpod_info gauge
ubs_superpod_info{superpod_name="0",node="node1",slot_id="1"} 1
ubs_superpod_info{superpod_name="0",node="node2",slot_id="2"} 1
# HELP ubs_mem_numa_borrow_bytes Bytes of numa-form remote memory borrowed by this node from export_node.
# TYPE ubs_mem_numa_borrow_bytes gauge
ubs_mem_numa_borrow_bytes{superpod_name="0",node="node1",slot_id="1",export_node="node2",export_slot_id="2",numa_id="4",name="numa-remote-0"} 2.147483648e+09
# HELP ubs_mem_numa_borrow_count Number of numa-form memory borrow relationships on this node.
# TYPE ubs_mem_numa_borrow_count gauge
ubs_mem_numa_borrow_count{superpod_name="0",node="node1",slot_id="1"} 1
# HELP ubs_mem_shm_provide_bytes Bytes of shared memory provided by this node (export_node == local).
# TYPE ubs_mem_shm_provide_bytes gauge
ubs_mem_shm_provide_bytes{superpod_name="0",node="node1",slot_id="1",name="shm-provide-0"} 1.073741824e+09
# HELP ubs_mem_shm_provide_count Number of shared memory regions provided by this node.
# TYPE ubs_mem_shm_provide_count gauge
ubs_mem_shm_provide_count{superpod_name="0",node="node1",slot_id="1"} 1
# HELP ubs_urma_device_info URMA device info (always 1, used for discovery/association).
# TYPE ubs_urma_device_info gauge
ubs_urma_device_info{superpod_name="0",node="node1",slot_id="1",device_name="urma-0",hw_res_id="100"} 1
# HELP ubs_urma_device_healthy URMA device health status: 1=healthy, 0=fault.
# TYPE ubs_urma_device_healthy gauge
ubs_urma_device_healthy{superpod_name="0",node="node1",slot_id="1",device_name="urma-0"} 1
```

**Example 5**: Grafana PromQL Aggregation Examples.

| View | PromQL |
| :--- | :--- |
| Number of SuperPods | `count(count by (superpod_name)(ubs_superpod_info))` |
| Number of physical nodes per SuperPod | `count by (superpod_name)(ubs_superpod_info)` |
| Member node list of a specified SuperPod | `ubs_superpod_info{superpod_name="$superpod"}` |
| SuperPod NUMA total borrowed | `sum by (superpod_name)(ubs_mem_numa_borrow_bytes)` |
| Node NUMA total borrowed | `sum by (node)(ubs_mem_numa_borrow_bytes)` |
| Node total lent | `sum by (export_node)(ubs_mem_numa_borrow_bytes)` |
| SuperPod total shared memory provided | `sum by (superpod_name)(ubs_mem_shm_provide_bytes)` |
| Number of healthy URMA devices per SuperPod | `sum by (superpod_name)(min by (superpod_name, device_name)(ubs_urma_device_healthy))` |
| URMA faulty devices | `min by (superpod_name, device_name)(ubs_urma_device_healthy) == 0` |

> ![icon](./figures/icon-note.gif) **Note:**<br />
> - URMA metrics are SuperPod-granularity data. Per-node full reporting produces N duplicate series. SuperPod-level queries must use the `min by (superpod_name, device_name)` prefix for aggregation and deduplication (fault-first: if any node observes a fault, it is judged as faulty).
> - Node memory borrowing/lending has only one runtime state (a borrowing or lending relationship instance). The total borrowing/lending can be directly aggregated from the borrowing relationship series reported by the node: total node borrowed = `sum by (node)(ubs_mem_numa_borrow_bytes)`, total node lent = `sum by (export_node)(ubs_mem_numa_borrow_bytes)`.

### Operation Steps

1. Enable SuperPod Topology Management.
    **Prerequisites**
    Complete the installation of matrixagent and matrixcontroller. The UBS Engine SDK socket (`/run/ubse`) is available.
    1.1 Configure node superPodId labels.
    Configure worker node labels via command line on the K8s master node to identify the SuperPod to which each node belongs. Node label is the priority source for superPodId; nodes without labels will fall back to the `superPodId` field reported by matrixagent; if both are empty, the node will not be included in any SuperPod.

    ```shell
    kubectl label nodes <node-name> unifiedbus.com/superpod=<superPodId>
    # Replace <superPodId> with the SuperPod identifier to which this node belongs (e.g., 0, 1, 2)
    # Replace <node-name> with the name of the node to be managed
    # Example: Assign node1~node8 to SuperPod 0 (FM networking)
    #   kubectl label nodes node1 unifiedbus.com/superpod=0
    #   kubectl label nodes node2 unifiedbus.com/superpod=0
    #   ... through node8
    ```

    > ![icon](./figures/icon-note.gif) **Note:**<br />
    > The value of `unifiedbus.com/superpod` is a string form of superPodId. The controller uses it to assign nodes to `superpod-<superPodId>`. Under FM networking, all physical nodes within the same UB interconnect domain (same SuperPod) are recommended to use the same superPodId.

    1.2 (Optional) Enable Volcano HyperNode Linkage.
    If you need to link SuperPod topology to Volcano's `HyperNode` CR (for the Volcano scheduler to perform network-topology-aware scheduling), set the `HYPERNODE_ENABLED` environment variable of matrixcontroller to `true`. The default value is `false`, which only produces `SuperPod` CR without HyperNode linkage.

    Prerequisites:
    - Volcano is installed in the cluster and the `hypernodes` CRD (`topology.volcano.sh/v1alpha1`) is registered. This can be confirmed with the following command:

    ```bash
    kubectl get crd hypernodes.topology.volcano.sh
    ```

    - Volcano Scheduler has the network-topology feature enabled.

    - Method 1: Modify after deployment via `kubectl set env` (no need to redeploy the Chart).

    ```bash
    kubectl set env deployment/kube-matrix-controller -n kube-system HYPERNODE_ENABLED=true
    kubectl rollout restart deployment/kube-matrix-controller -n kube-system
    ```

    - Method 2: Before deployment, modify `charts/matrixcontroller/templates/deploy.yaml`, change the `value` of `HYPERNODE_ENABLED` to `"true"`, then deploy matrixcontroller following the service deployment steps in [Starting Installation](#starting-installation).

    > ![image](./figures/icon-notice.gif) **Notice:**<br />
    > The controller only assembles HyperNode when the `HyperNode` CRD is registered; if the CRD is not ready, the controller will skip HyperNode assembly and print a warning log, but `SuperPod` CR production is not affected.

    1.3 Trigger Topology Reconciliation.
    After completing node label configuration, matrixcontroller will automatically detect MatrixMetric CR changes and trigger reconciliation:
    - MatrixMetric CR add, modify, and delete events trigger debounced reconciliation (5s debounce window).
    - A periodic full resync is executed every 5min.
    - Topology data is refreshed every 12h.

    If you need to trigger reconciliation immediately, wait for the next matrixagent report, or manually trigger a MatrixMetric CR change (e.g., `kubectl annotate` any MatrixMetric CR to trigger an UpdateEvent).

    1.4 Verify SuperPod CR.
    Execute the following command to view the SuperPod CRs in the cluster.

    ```bash
    kubectl get superpods.matrix.openfuyao.cn
    ```

    Expected result: Each `superPodId` with node membership corresponds to a `superpod-<id>` resource, for example:

    ```
    NAME          SUPERPODID   TIER   NODECOUNT
    superpod-0    0            1      8
    superpod-1    1            1      8
    ```

    View SuperPod details (including groups, nodes, memory):

    ```bash
    kubectl get superpod superpod-0 -o yaml
    ```

    Expected output (FM networking, `HYPERNODE_ENABLED=true`, excerpt):

    ```yaml
    apiVersion: matrix.openfuyao.cn/v1
    kind: SuperPod
    metadata:
      name: superpod-0
      annotations:
        superpod.matrix.huawei.com/node-hash: a1b2c3d4
    spec:
      superPodId: "0"
      tier: 1
      hyperNodeRef: hn-t1-0
      groups:
        - name: group-0
          tier: 1
          hyperNodeRef: hn-t1-0
          nodes:
            - name: node1
              ip: 10.8.0.1
              memory:
                total: 256Gi
                used: 128Gi
            # ... node2 ~ node7
            - name: node8
              ip: 10.8.0.8
              memory:
                total: 256Gi
                used: 120Gi
    status:
      nodeCount: 8
    ```

    1.5 (Optional) Verify HyperNode CR.
    If `HYPERNODE_ENABLED=true` is enabled, execute the following command to view the Volcano HyperNode CRs produced by linkage.

    ```bash
    kubectl get hypernodes.topology.volcano.sh
    ```

    Expected result: Each `superPodId` corresponds to a `hn-t1-<id>` resource, whose `spec.members` includes all nodes under that SuperPod.

    1.6 View Node Hash Fingerprint.
    The SuperPod annotation `superpod.matrix.huawei.com/node-hash` is the first 8 characters of SHA256 of sorted member node names, which can be used to quickly determine whether the SuperPod member topology has changed.

    ```bash
    kubectl get superpod <superpod-name> \
      -o jsonpath='{.metadata.annotations.superpod\.matrix\.huawei\.com/node-hash}'
    ```

    - Observe topology change results.
    When node label changes or MatrixMetric CR updates cause SuperPod membership changes, the controller will update `spec.groups[].nodes[]` and `status.nodeCount` of the `SuperPod` CR in the next reconciliation, and refresh `node-hash`. You can repeat the commands in the Verify SuperPod CR section to observe changes.
2. (Optional) Schedule Business Pods to the Same SuperPod.
    **Prerequisites**
    Complete HyperNode linkage enablement (the "Enable SuperPod Topology Management" section of [Operation Steps](#operation-steps)), and Volcano Scheduler has the network-topology feature enabled.
    2.1 Create a PodGroup.
    Refer to [Configuration Examples - Example 3](#configuration-examples) to create a PodGroup that declares gang policy and topology hard constraints. Under FM networking, `highest-tier` can only be `"1"`.

    ```bash
    kubectl apply -f podgroup-gang-in-superpod.yaml
    ```

    2.2 Create business Pods.
    Create business Pods associated with the PodGroup, scheduled by the Volcano scheduler.

    ```bash
    kubectl apply -f workers.yaml
    ```

    2.3 Verify scheduling results.
    Execute the following command to check Pod scheduling status.

    ```bash
    kubectl get pod -o wide
    ```

    Expected result: All 3 worker Pods are successfully scheduled and located within the same SuperPod (on nodes with the same `superPodId`). If no Tier-1 HyperNode can accommodate all 3 Pods, all will be pending.
3. (Optional) Verify superpod-exporter Metrics.
    **Prerequisites**
    `superpod-exporter` has been deployed as described in step 2.5 of the [Installation](#installation) section, and the Pod status is `Running`.
    3.1 Verify exporter readiness.
    Execute the following command to check the superpod-exporter Pod status and readiness metric.

    ```bash
    kubectl get pods -n kube-system -l app.kubernetes.io/name=superpod-exporter -o wide
    ```

    Expected result: Each node configured with the `unifiedbus.com/superpod` label should have a corresponding `superpod-exporter` Pod with status `Running`.

    3.2 Query the metrics endpoint.
    Enter any superpod-exporter Pod via `kubectl exec`, or directly `curl` the node IP to query the metrics endpoint.

    ```bash
    # Method 1: kubectl exec query
    POD=$(kubectl -n kube-system get pods -l app.kubernetes.io/name=superpod-exporter -o jsonpath='{.items[0].metadata.name}')
    kubectl -n kube-system exec $POD -- curl -s localhost:9102/metrics | grep ubs_exporter_up

    # Method 2: Direct curl to node IP (requires node port reachability)
    curl -s <node-ip>:9102/metrics | grep ubs_superpod_info
    ```

    Expected result: `ubs_exporter_up` value is 1; `ubs_superpod_info` contains the local node's `node`/`slot_id`/`superpod_name`, and the `superpod_name` value matches the node label `unifiedbus.com/superpod`.

    3.3 Verify SuperPod Dimension Aggregation.
    Execute the PromQL from [Configuration Examples - Example 5](#configuration-examples) in Grafana or Prometheus to verify the SuperPod-level aggregation view.

    ```bash
    # Number of SuperPods
    count(count by (superpod_name)(ubs_superpod_info))
    # Number of physical nodes per SuperPod
    count by (superpod_name)(ubs_superpod_info)
    # SuperPod NUMA total borrowed
    sum by (superpod_name)(ubs_mem_numa_borrow_bytes)
    # URMA faulty devices
    min by (superpod_name, device_name)(ubs_urma_device_healthy) == 0
    ```

    Expected result: Queries for SuperPod count, member count, total borrowed, URMA health status, etc. return correct results.

### Subsequent Operations

- **Verify Component Status**: Use `kubectl get pods -A` to check the running status of matrixagent, matrixcontroller, and `superpod-exporter`, confirming all are `Running`.
- **Check CRD Registration**: Use `kubectl get crd superpods.matrix.openfuyao.cn` to confirm the SuperPod CRD is registered; if HyperNode linkage is enabled, use `kubectl get crd hypernodes.topology.volcano.sh` to confirm the HyperNode CRD is registered.
- **Verify Metric Exporter**: Use `curl <node-ip>:9102/metrics` to confirm `ubs_exporter_up=1` and that the number of `ubs_*` metric series matches the actual borrowing relationships on the node; `superpod_name` should match the node label, not `unknown`.
- **Adjust Topology Membership**: To adjust a node's SuperPod membership, re-execute the steps for configuring node superPodId labels in [Operation Steps](#operation-steps) without restarting matrixcontroller; the controller will take effect on the next debounce or periodic resync (up to 5min). The `superpod_name` label of `superpod-exporter` is derived from the node label; after a label change, the exporter will refresh after the 12h cache expires, or immediately upon exporter restart.
- **Toggle HyperNode Linkage**: To enable/disable HyperNode linkage, use `kubectl set env deployment/kube-matrix-controller -n kube-system HYPERNODE_ENABLED=<true|false>` and `kubectl rollout restart`; the full resync will idempotently overwrite existing resources.
- **Troubleshooting**:
  1. `kubectl get superpod` to check count and members; `nodeCount` should match the actual number of nodes in the SuperPod.
  2. Check matrixcontroller logs: `kubectl logs -n kube-system -l app=kube-matrix-controller`.
  3. Check matrixagent logs to confirm normal collection: `kubectl logs -n kube-system -l app=kube-matrix-agent`.
  4. Check UBS Engine SDK socket: whether `/run/ubse` exists on the node.
  5. Missing metrics troubleshooting: `kubectl logs -n kube-system -l app.kubernetes.io/name=superpod-exporter` to check exporter logs; when `superpod_name` is `unknown`, check the node label `unifiedbus.com/superpod` and RBAC permissions.

### Related Operations

- **View SuperPod**:

  ```bash
  kubectl get superpods.matrix.openfuyao.cn
  kubectl get superpod <superpod-name> -o yaml
  kubectl get superpod <superpod-name> \
    -o jsonpath='{.metadata.annotations.superpod\.matrix\.huawei\.com/node-hash}'
  ```

- **View HyperNode** (only exists when `HYPERNODE_ENABLED=true`):

  ```bash
  kubectl get hypernodes.topology.volcano.sh
  kubectl get hypernode <hn-name> -o yaml
  ```

- **Delete SuperPod / HyperNode**:

  > ![image](./figures/icon-notice.gif) **Notice:**
  >
  > The controller automatically maintains the lifecycle of `SuperPod`/`HyperNode` CRs. Under normal circumstances, manual deletion is not required. Only manually delete when decommissioning the feature, cleaning up residual resources, or troubleshooting anomalies. It is recommended to delete specific resources by name first, and use batch deletion only for decommission or reset scenarios.

  ```bash
  kubectl delete superpod <superpod-name>
  kubectl delete hypernode <hn-name>
  # Batch cleanup of all resources (use only for decommission/reset scenarios)
  kubectl delete superpods.matrix.openfuyao.cn --all
  kubectl delete hypernodes.topology.volcano.sh --all
  ```

- **View superpod-exporter Metrics**:

  ```bash
  kubectl get pods -n kube-system -l app.kubernetes.io/name=superpod-exporter -o wide
  curl -s <node-ip>:9102/metrics | grep ubs_
  curl -s <node-ip>:9102/healthz
  ```

- **Decommission superpod-exporter**: Uninstalling the `superpod-exporter` DaemonSet disables metric collection, leaving no residual K8s resources, and does not affect existing matrixagent, matrixcontroller, or `SuperPod`/`HyperNode` resources.

  ```bash
  helm uninstall superpod-exporter -n kube-system
  ```
