# Many-Core Orchestrator（MCO）

## Feature Introduction

Many-Core Orchestrator (MCO, many-core scheduling orchestration system) is designed for many-core mixed-deployment clusters, providing host interference metric collection, node interference analysis, Volcano interference-aware scheduling, and optional Kata Containers VM-level isolation capabilities. Its goal is to reduce tail latency jitter of online services in I/O, LLC cache, and memory pressure mixed-deployment scenarios without modifying application code, while improving node safe mixed-deployment density.

### Application Scenarios

MCO is suitable for scenarios where online latency-sensitive services and offline throughput-oriented services share the same set of many-core nodes, such as Redis, real-time inference, search services mixed with batch processing, full queries, and I/O stress testing tasks. The system focuses on sensing the following interference signals:

- LLC cache miss rate and LLC occupancy, for identifying cache pollution.
- block I/O latency P95/P99, for identifying block device queue congestion.
- PSI I/O and PSI memory, for identifying resource waiting pressure.
- When PSI is unavailable, use MemAvailable ratio and page scan rate for degraded evaluation.

MCO does not automatically determine service types, nor does it force switching certain Pods to Kata. Whether to use Kata is determined by the user setting `runtimeClassName` in the Pod.

### Capability Scope

MCO consists of four types of core components:

**Table 1** Core Components

| Component | Deployment Form | Purpose |
|---|---|---|
| Collector | DaemonSet | Collects LLC, I/O, PSI, memory reclaim and other host metrics on each node, and reports them to Analyzer via gRPC. |
| Analyzer | Deployment | Performs window smoothing, rule detection, and interference level calculation on Collector samples, and writes them into `NodeInterferenceReport`. |
| MCO Volcano Plugin | Volcano scheduler plugin | Excludes high-pressure or Kata-unsupported nodes during the Filter phase, and prefers low-interference nodes during the Score phase. |
| Kata Deploy | DaemonSet sub-chart | Checks node virtualization capabilities, installs/registers Kata runtime handler, and creates RuntimeClass. |

MCO uses two CRDs:

**Table 2** CRD Description

| CRD | Scope | Abbreviation | Description |
|---|---|---|---|
| `NodeInterferenceReport` | Cluster | `nir` | Stores the latest interference levels, metric snapshots, detection reasons, and sampling status for all nodes. Default object name is `cluster`. |
| `DetectionRuleConfig` | Namespaced | `drc` | Stores interference detection thresholds, weights, penalty coefficients, and PSI degradation rules. Default is in the `mco-system` namespace with name `cluster`. |

### Highlight Features

- **Low-modification integration**: Users only need to set `runtimeClassName` in the Pod to access Kata VM isolation, without modifying application code or introducing Sidecars.
- **Microarchitecture-level interference awareness**: Breaks through the limitation of traditional schedulers that only rely on CPU/memory availability, directly collecting LLC miss rate, block I/O P99 latency, PSI and other underlying microarchitecture metrics, identifying hardware-level resource competition that Cgroups cannot isolate.
- **Scheduling + Isolation dual-layer protection**: Interference-aware scheduling avoids high-interference nodes during node selection; Kata VM-level isolation physically cuts off crosstalk from offline services to online services. Both can be used independently or in combination.
- **Fail-safe degradation**: When Analyzer is unavailable or interference reports expire, the scheduling plugin automatically falls back to basic scoring, without blocking normal service deployment.
- **Strong observability**: Collector and Analyzer each expose Prometheus `/metrics` endpoints, interference reports are presented through CRs, and the scheduling plugin provides filtering reasons and score distribution metrics, facilitating continuous tuning.

### Basic Concepts

**Data Pipeline:**

```text
Collector -> Analyzer -> NodeInterferenceReport CR -> Volcano mco-plugin -> Pod scheduling placement
```

- **Collector** runs as a DaemonSet on each node, collecting host microarchitecture metrics through eBPF/perf, and streaming them to Analyzer via gRPC.
- **Analyzer** receives metric snapshots from all nodes, performs window smoothing, then conducts interference detection and level calculation based on thresholds and weights in `DetectionRuleConfig`, and finally aggregates interference reports for all nodes into a single cluster-level `NodeInterferenceReport` CR (name fixed as `cluster`).
- **MCO Plugin** acts as a Volcano scheduler plugin, excluding high-pressure or Kata-unsupported nodes during the Filter phase, and scoring with `score = 100 × (1 - interferenceLevel)` during the Score phase, where low-interference nodes score higher.
- **Kata Deploy** automatically checks node virtualization conditions, installs Kata runtime and registers RuntimeClass, and after completion labels the node with `mco.many-core.io/kata-ready=true`.

**Interference Level Calculation:**

Analyzer categorizes interference into three dimensions — cache pollution (LLC miss rate × LLC occupancy), memory pressure (PSI memory some/full), and I/O queue congestion (bio latency P99 + PSI I/O some), normalizes each to the [0, 1] interval, then performs weighted summation with weights, and applies dynamic penalty coefficients (amplifying weights when compound interference occurs), ultimately outputting a node interference level $L \in [0, 1]$. When PSI is unavailable, it automatically switches to the degradation path, using MemAvailable ratio and page scan rate as substitutes for PSI metrics.

### Implementation Principle

1. **Metric Collection**: Collector collects host metrics (LLC miss/occupancy, bio latency P95/P99, PSI I/O/memory) once per second, streaming them to Analyzer via gRPC.
2. **Interference Analysis**: Analyzer performs mean smoothing on samples within a window (default 10s), determines whether cache pollution, I/O queue congestion, or memory pressure detection is triggered based on thresholds in the `DetectionRuleConfig` CR, calculates normalized interference indices, and weighted summation yields the interference level.
3. **Report Writing**: Analyzer periodically (default 1s) writes each node's interference level, raw metric snapshots, and detection details into `NodeInterferenceReport/cluster` CR's `status.nodes`.
4. **Scheduling Decision**: MCO Plugin listens to `NodeInterferenceReport/cluster` through a Kubernetes informer, excluding nodes where PSI memory full > 0 or PSI I/O some exceeds hard constraint thresholds during the Filter phase; scoring with `score = 100 × (1 - interferenceLevel)` during the Score phase.
5. **Kata Isolation**: After the user sets `runtimeClassName: kata-*` in the Pod, the scheduling plugin ensures the Pod is only placed on Kata-ready nodes, and the container runtime runs with an independent Guest OS, physically cutting off LLC cache pollution and I/O queue crosstalk.

### Relationship with Related Features

**Table 3** Relationship with Related Features

| Related Feature | Relationship |
|---|---|
| **Volcano** | MCO Plugin runs as a Volcano scheduler plugin, relying on Volcano's Session/Plugin mechanism. MCO Chart can install Volcano alongside. |
| **Kata Containers** | MCO automatically deploys and registers the Kata runtime through kata-deploy, and users select it as needed in Pods. Kata isolation is an optional enhancement capability. |
| **Kubernetes Scheduler** | MCO does not modify or replace kube-scheduler, only enhancing Filter/Score logic in the Volcano scheduling chain. Pods not managed by Volcano are unaffected. |
| **Prometheus** | Collector and Analyzer each expose `/metrics` endpoints, integrated with Prometheus through ServiceMonitor or manual scrape job configuration. |

## Installation

### Prerequisites

- Kubernetes cluster, recommended v1.33 or above.
- Helm 3.x.
- If interference-aware scheduling is needed, Volcano 1.9.0 is required. The Chart can install Volcano alongside MCO.
- If Kata isolation is needed, nodes must support KVM virtualization and allow kata-deploy to complete runtime registration.
- Collector depends on host capabilities to collect metrics, and by default mounts `/sys`, `/lib/modules`, `/usr/src`, and `/var/log/mco-collector`, using `perf` and `biolatency` commands.
- ServiceMonitor depends on the cluster having Prometheus Operator or compatible ServiceMonitor CRD installed. If the cluster does not have Prometheus configured, disable `collector.serviceMonitor.enabled` and `analyzer.serviceMonitor.enabled`.

### Start Installation

Pull the Chart (shared across all scenarios):

```bash
helm pull oci://cr.openfuyao.cn/charts/many-core-orchestrator --version 0.0.0-latest
```

Select the installation command based on your scenario below. The Chart enables all components by default; `--set` only lists key parameters that need to **override default values**.

#### Scenario 1: Full Feature Deployment (Recommended)

Deploy the complete set of collection, analysis, scheduling plugin, and Kata isolation components:

```bash
helm install mco many-core-orchestrator-0.0.0-latest.tgz
```

>![image](figures/icon-note.gif)**Note:**
> - By default, Volcano (including CRD, controller, admission) is installed simultaneously, and its built-in scheduler is disabled, with MCO scheduler taking over.
> - If the cluster does not have Prometheus Operator, add the following to disable ServiceMonitor:
```bash
helm install mco many-core-orchestrator-0.0.0-latest.tgz \
  --set collector.serviceMonitor.enabled=false \
  --set analyzer.serviceMonitor.enabled=false
```

#### Scenario 2: Deploy Only Kata Isolation

```bash
helm install mco many-core-orchestrator-0.0.0-latest.tgz \
  --set collector.enabled=false \
  --set analyzer.enabled=false \
  --set mcoPlugin.enabled=false
```

#### Scenario 3: Deploy Only Collection and Analysis (No Scheduling Intervention)

```bash
helm install mco many-core-orchestrator-0.0.0-latest.tgz \
  --set kata-deploy.enabled=false \
  --set mcoPlugin.enabled=false
```

#### Scenario 4: Integrating with Existing Volcano Cluster

If the cluster already has Volcano installed, only deploy MCO components and register the scheduling plugin.

>![image](figures/icon-notice.gif)**Notice:**<br />
> - In this scenario, you must simultaneously set `volcano.enabled=false` (do not reinstall Volcano sub-chart) and `mcoPlugin.manageSchedulerConfig=false` (do not render ConfigMap, preserve existing configuration). **If `volcano.enabled=false` is omitted (default `true`), the Chart will reinstall the Volcano sub-chart and overwrite the existing `volcano-scheduler-configmap`, potentially causing cluster conflicts and loss of original scheduling configuration.**
>
> - `Collector`, `Analyzer`, `Kata Deploy`, and `MCO Plugin` are all enabled by default and do not require additional `--set`.

##### Impact on Existing Volcano Components

Integrating MCO disables the cluster's existing core scheduler `volcano-scheduler`, replacing it with the MCO-deployed `mco-volcano-scheduler` to take over all Volcano scheduling. The changes and impacts on each component are as follows.

**Table 4** Impact of Integrating MCO on Existing Volcano Components

| Component | Change | Impact |
|---|---|---|
| `volcano-scheduler` (Deployment) | Scaled down to `replicas=0`, object retained but not deleted | Scheduling capability suspended. Between the scale-down and `mco-volcano-scheduler` becoming ready, newly submitted Volcano Jobs/PodGroups cannot be scheduled and remain Pending. |
| `mco-volcano-scheduler` (Deployment) | Created by MCO, using an independent ServiceAccount `mco-scheduler` | Takes over all Volcano scheduling. Until it is ready, the cluster has no available scheduler; business scheduling resumes only after it reaches Running. |
| `volcano-scheduler-configmap` | Manually edited by the user to register `mco-plugin` | If the original scheduler is still running when edited, it may read a config without the registered plugin. Always disable the original scheduler before editing. |
| Volcano CRDs, controller, admission | Unchanged | `volcano.enabled=false` only skips installing the Volcano sub-chart; existing CRDs, controller, and admission remain unchanged and keep running. |
| `volcano-scheduler` related Service and RBAC | Unchanged | The original `volcano-scheduler-service`, ClusterRole/ClusterRoleBinding objects are retained, but no Pod matches anymore, leaving the Service Endpoints empty. |

>![image](figures/icon-notice.gif)**Notice:**<br />
> - The original `volcano-scheduler` is only scaled down, and the Deployment object is not deleted. To roll back, restore its `replicas` to the original value and uninstall MCO. However, you must also remove the `mco-plugin` configuration section from the ConfigMap to prevent the original scheduler from failing to start due to an unloadable plugin.

##### Operation Steps

1. **Disable the original Volcano scheduler** (avoid conflicts from two schedulers running simultaneously).
   ```bash
   kubectl scale deploy -n volcano-system volcano-scheduler --replicas=0
   ```
   If the cluster has multiple Volcano scheduler instances (such as multi-replica or HA deployments), scale all replicas down to 0 to ensure only `mco-volcano-scheduler` participates in scheduling; otherwise, duplicate scheduling or scheduling conflicts may occur.

2. **Manually register mco-plugin in the existing volcano-scheduler-configmap**.
   Refer to the ConfigMap format in [Adjusting Scheduling Plugin Parameters](#adjusting-scheduling-plugin-parameters), and insert the `mco-plugin` segment in `tiers[0].plugins` (after `conformance`).
   ```bash
   kubectl edit configmap volcano-scheduler-configmap -n volcano-system
   ```

3. **Install MCO** (do not reinstall Volcano, do not manage scheduler config).
   ```bash
   helm install mco many-core-orchestrator-0.0.0-latest.tgz \
     --set volcano.enabled=false \
     --set mcoPlugin.manageSchedulerConfig=false
   ```
   MCO will deploy `mco-volcano-scheduler` (replacing the original `volcano-scheduler`), which reads the edited ConfigMap at startup without requiring an additional restart.

If MCO is installed before editing the ConfigMap, execute the following command to restart `mco-volcano-scheduler` for the configuration to take effect:
```bash
kubectl rollout restart deploy mco-volcano-scheduler -n volcano-system
```
The original `volcano-scheduler` remains scaled down (`replicas=0`), with the MCO-deployed `mco-volcano-scheduler` taking over all Volcano scheduling.

#### Deployment Scenario Quick Reference

**Table 5** Deployment Scenario Quick Reference

| Scenario | Collector | Analyzer | Kata | Scheduling Plugin | Volcano |
|---|---|---|---|---|---|
| Full Feature (Default) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Kata Isolation Only | ❌ | ❌ | ✅ | ❌ | ✅ |
| Collection and Analysis Only | ✅ | ✅ | ❌ | ❌ | ✅ |
| Integrating with Existing Volcano | ✅ | ✅ | ✅ | ✅ | ❌ |

>![image](figures/icon-notice.gif)**Notice:**<br /> 
> - ✅: Indicates the MCO Chart deploys this component by default.
> - ❌: Indicates the MCO Chart does not deploy this component by default.
> - In Scenario 4, Volcano=❌ means the MCO Chart does not install the Volcano sub-chart, but a ** usable Volcano must already exist in the cluster** (including CRD, controller, admission), otherwise the scheduling plugin cannot function.

### Post-Deployment Verification

Check component status:

```bash
kubectl get pods -n mco-system
kubectl get pods -n volcano-system
```
> If normal, Collector and Analyzer Pods should both be in Running state, and the Volcano scheduler Pod should be in Running state.

Check CRDs:

```bash
kubectl get crd | grep mco.many-core.io
```
> If normal, it should return `nodeinterferencereports.mco.many-core.io` and `detectionruleconfigs.mco.many-core.io` CRDs.

Check node interference reports:

```bash
kubectl get nir cluster -o yaml
```
> If normal, `status.nodes` should contain `interferenceLevel`, `metrics`, and `detections` fields for each node.

Check detection rules:

```bash
kubectl get drc cluster -n mco-system -o yaml
```
> If normal, it should return the default detection thresholds, weights, and degradation parameter configuration.

Check Kata node capability labels:

```bash
kubectl get nodes -L katacontainers.io/kata-runtime,mco.many-core.io/kata-ready
```
> If normal, physical node labels should show true.

## Using Interference-Aware Scheduling and Kata Isolation

### Prerequisites

- MCO full feature deployment (Scenario 1) has been completed, or integrated with an existing Volcano cluster (Scenario 4), ensuring Collector, Analyzer, and MCO Plugin are all running normally.
- If Kata isolation is needed, target nodes must have KVM virtualization capabilities and `mco.many-core.io/kata-ready=true`.
- Pods to be scheduled must go through Volcano scheduling (i.e., managed by Volcano scheduler), otherwise MCO Plugin will not participate in scheduling decisions.

### Background Information

After deploying MCO, the system's core workflow is as follows:

1. **Collector→Analyzer→CR** pipeline runs continuously, with `NodeInterferenceReport/cluster` updating each node's interference level (0.0~1.0) every 1s.
2. When a Pod scheduled through Volcano enters the scheduling process, MCO Plugin automatically intervenes in the Filter and Score phases.
3. Users can view each node's interference status at any time through `kubectl get nir cluster`, and adjust detection rules through `kubectl edit drc cluster -n mco-system`.

Key objects:
- `NodeInterferenceReport/cluster`: Cluster-level interference report, `status.nodes` keyed by node name.
- `DetectionRuleConfig/mco-system/cluster`: Detection rules, hot-loaded by Analyzer, changes take effect immediately.

### Usage Limitations

MCO's degradation strategy follows the principle of "not blocking service scheduling"; the following limitations should be understood before use:

- When Analyzer is unavailable or `NodeInterferenceReport` has not been refreshed beyond TTL, the scheduling plugin returns a neutral score of `50`; nodes are not excluded, but interference awareness is temporarily disabled.
- When a node has no report, it is not filtered due to missing interference data.
- When PSI is unavailable, the scheduling plugin skips PSI hard constraints (`psiMemoryFull` and `psiIoSome` filtering); Analyzer uses memory availability ratio, page scan rate, and bio latency to continue calculating interference levels.
- When Kata deployment fails, the node lacks `mco.many-core.io/kata-ready=true`, only affecting Pods requesting `runtimeClassName: kata-*`, not affecting regular runc Pods.
- Detection thresholds and weights are initial values based on experiments or experience; differences in LLC, memory bandwidth, and I/O stack capabilities across machine types will affect the meaning of the same threshold, so calibration per machine type is recommended.

### Operation Steps

#### Enable Interference-Aware Scheduling

After deploying the MCO scheduling plugin, Pods scheduled through Volcano automatically participate in interference-aware scheduling. Plugin behavior consists of two steps:

1. **Filter**: If a node's PSI memory full or PSI I/O some exceeds hard constraint thresholds, the node is excluded; if the Pod requests Kata runtime, the node must simultaneously have `katacontainers.io/kata-runtime=true` and `mco.many-core.io/kata-ready=true`.
2. **Score**: Reads the node's `interferenceLevel` from `NodeInterferenceReport/cluster`, scoring with `score = 100 * (1 - interferenceLevel)`. Lower interference levels yield higher scores.

When CR data is missing, expired, or the node report status is not `ok`, the plugin returns a neutral score of `50`, without blocking normal scheduling.

#### Using Kata VM Isolation

Users select the Kata runtime through the Pod's `runtimeClassName`. It is generally recommended to place offline high-throughput, high-I/O, or cache-scanning workloads in Kata to reduce their crosstalk to online services on the same node.

Example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: etl-batch
  labels:
    mco.many-core.io/workload-type: offline
spec:
  runtimeClassName: kata-qemu
  containers:
  - name: job
    image: example/etl:latest
```

`mco.many-core.io/workload-type` is a reserved label that currently does not affect scheduling decisions, and can be used for service classification and future policy extension.

Available RuntimeClasses are registered by kata-deploy; common names include `kata-qemu`, `kata-clh`, etc., specific to the cluster's actual objects:

```bash
kubectl get runtimeclass
```

#### View Interference Reports

`NodeInterferenceReport/cluster`'s `status.nodes` is keyed by node name, storing each node's latest status. Core fields are as follows:

**Table 6** Interference Report Core Field Description

| Field | Description |
|---|---|
| `interferenceLevel` | Interference level, range 0.0 to 1.0; higher values indicate greater node congestion. |
| `psiAvailable` | Whether PSI files are available on the current node. |
| `status` | Report status; normal is `ok`. |
| `metrics.llcMissRate` | LLC miss rate. |
| `metrics.llcOccupancy` | LLC occupancy. |
| `metrics.bioLatencyP95Ms` | block I/O P95 latency, in ms. |
| `metrics.bioLatencyP99Ms` | block I/O P99 latency, in ms. |
| `metrics.psiIoSome` | PSI I/O some avg10. |
| `metrics.psiIoFull` | PSI I/O full avg10. |
| `metrics.psiMemorySome` | PSI memory some avg10. |
| `metrics.psiMemoryFull` | PSI memory full avg10. |
| `metrics.memAvailableRatio` | Available memory ratio, used for degraded evaluation when PSI is unavailable. |
| `metrics.pageScanRate` | Page reclaim rate, in pages/s. |
| `detections` | Detection types and explanations triggered by Analyzer. |
| `availability` | Availability, status, and failure reasons of each collector/metric. |
| `sequence` | Node sample sequence number, used to determine whether the report has been refreshed. |
| `sampleCount` | Number of samples in the current smoothing window. |
| `windowSeconds` | Smoothing window length. |

Execute the following command to quickly view node interference levels:

```bash
kubectl get nir cluster -o jsonpath='{.status.nodes}'
```

View a specific node's detailed report:

```bash
kubectl get nir cluster -o yaml
```

#### Adjust Detection Rules

Default rules are rendered by Helm as `DetectionRuleConfig/mco-system/cluster`. Analyzer listens to this object through an informer and hot-loads changes automatically; when parsing fails or the object is deleted, the last valid version of rules is preserved.

The following is the default configuration (also the complete structure seen when executing `kubectl edit drc cluster -n mco-system`):

```yaml
spec:
  # —— Three-dimension weights (sum should approximate 1.0) ——
  weights:
    cache: 0.30        # LLC cache pollution weight
    memory: 0.30       # Memory pressure weight
    io: 0.40           # I/O congestion weight (usually has the largest impact)

  # —— Per-dimension saturation limits (normalization denominator) ——
  saturation:
    cacheProduct: 0.175
    psiMemory: 50
    bioLatencyP99Ms: 100
    psiIo: 20

  # —— Detection trigger thresholds (exceeding triggers corresponding interference type) ——
  thresholds:
    cacheMissRate: 0.20
    llcOccupancy: 0.70
    bioLatencyP99Ms: 50      # Commonly adjusted: I/O-sensitive services can lower
    psiIo: 20                # Commonly adjusted: paired with bioLatencyP99Ms
    psiMemorySome: 10
    psiMemoryFull: 0
    ioPenaltyP99Ms: 50       # Dynamic weight trigger condition
    ioPenaltyPsiIo: 20
    memIoPenaltyMemory: 50
    memIoPenaltyPsiIo: 20

  # —— Dynamic penalty coefficients (amplify weights during compound interference) ——
  penalties:
    ioMultiplier: 1.2
    memIoMultiplier: 1.1

  # —— Degraded memory evaluation when PSI is unavailable ——
  degradedMemory:
    memAvailableRatioThreshold: 0.30
    pageScanRateThreshold: 10000
```

**Table 7** Common Tuning Scenarios

| Scenario | Adjusted Field | Recommended Direction |
|---|---|---|
| Online services are extremely sensitive to I/O latency | `thresholds.bioLatencyP99Ms`, `thresholds.psiIo` | Lower thresholds (e.g., P99→30ms) to make Analyzer mark I/O congestion earlier |
| Node memory is abundant, PSI memory frequently false-positives | `thresholds.psiMemorySome` | Appropriately increase (e.g., 10→20) |
| Certain machine type has small LLC, high cache miss baseline | `thresholds.cacheMissRate` | Calibrate per machine type baseline (e.g., 0.20→0.30) |
| I/O and memory compound interference needs more aggressive avoidance | `penalties.ioMultiplier`, `penalties.memIoMultiplier` | Moderately increase penalty coefficients |

**Modification Methods:**

```bash
# Method 1: Helm upgrade (recommended, maintaining GitOps traceability)
helm upgrade mco many-core-orchestrator-0.0.0-latest.tgz \
  --set analyzer.detectionRules.thresholds.bioLatencyP99Ms=30 \
  --set analyzer.detectionRules.thresholds.psiIo=15

# Method 2: Directly edit CR (takes effect immediately, suitable for debugging)
kubectl edit drc cluster -n mco-system
```

#### Adjusting Scheduling Plugin Parameters

Plugin behavior is controlled by the `mco-plugin` configuration section in Volcano's `volcano-scheduler-configmap`. When the MCO Chart manages configuration, it renders through `.Values.mcoPlugin.config`. When integrating with an existing Volcano, directly edit the ConfigMap to register the plugin.

**ConfigMap Format (actual structure seen when executing `kubectl edit configmap volcano-scheduler-configmap -n volcano-system`)**

```yaml
data:
  volcano-scheduler.conf: |
    actions: "enqueue, allocate, backfill"
    tiers:
    - plugins:
      - name: priority
      - name: gang
      - name: conformance
      - name: mco-plugin
        arguments:
          mco-plugin:
            crName: "cluster"               # Fixed, not recommended to modify
            crStatusTTLSeconds: 30           # CR cache TTL
            hardConstraints:
              psiMemoryFullAvg10: 0          # Memory full hard filter threshold
              psiIoSomeAvg10Threshold: 20    # I/O some hard filter threshold
            fallbackWhenCROutdated: true     # Degradation when CR expires (currently always effective)
    - plugins:
      - name: overcommit
      - name: drf
      - name: predicates
      - name: proportion
      - name: nodeorder
      - name: binpack
```

**Values.yaml Corresponding Path (used with Helm `--set`)**

```yaml
mcoPlugin:
  config:
    crName: "cluster"
    crStatusTTLSeconds: 30
    hardConstraints:
      psiMemoryFullAvg10: 0.0
      psiIoSomeAvg10Threshold: 20.0
    fallbackWhenCROutdated: true
```

>![image](figures/icon-notice.gif)**Notice:**<br />
> - Although `crName` is configurable, the plugin informer currently hardcodes processing only the `NodeInterferenceReport` named `cluster`; modification is not recommended.
> - When integrating with an existing Volcano (`manageSchedulerConfig=false`), you must manually register `mco-plugin` in tiers according to the **ConfigMap Format**, rather than through values.yaml.

**Table 8** Common Scheduling Plugin Parameter Tuning

| Parameter | Default Value | Helm `--set` Path | ConfigMap Path | Description |
|---|---:|---|---|---|
| `crStatusTTLSeconds` | `30` | `mcoPlugin.config.crStatusTTLSeconds` | `arguments.mco-plugin.crStatusTTLSeconds` | CR cache expiration seconds. Can be increased when Collector/Analyzer is jittering. |
| `psiMemoryFullAvg10` | `0.0` | `mcoPlugin.config.hardConstraints.psiMemoryFullAvg10` | `arguments.mco-plugin.hardConstraints.psiMemoryFullAvg10` | Nodes exceeding this value are directly excluded during the Filter phase. |
| `psiIoSomeAvg10Threshold` | `20.0` | `mcoPlugin.config.hardConstraints.psiIoSomeAvg10Threshold` | `arguments.mco-plugin.hardConstraints.psiIoSomeAvg10Threshold` | PSI I/O hard filter threshold. Can be lowered for I/O-sensitive services (e.g., 10.0). |

Modification examples:

```bash
# Method 1: Helm upgrade (recommended, when Chart manages ConfigMap)
helm upgrade mco many-core-orchestrator-0.0.0-latest.tgz \
  --reuse-values \
  --set mcoPlugin.config.crStatusTTLSeconds=60 \
  --set mcoPlugin.config.hardConstraints.psiIoSomeAvg10Threshold=15.0

# Method 2: Directly edit ConfigMap (requires restarting scheduler Pod to take effect)
kubectl edit configmap volcano-scheduler-configmap -n volcano-system
kubectl rollout restart deploy mco-volcano-scheduler -n volcano-system
```

### Follow-up Operations

After deploying and enabling MCO, it is recommended to continuously observe system status through the following methods:

- Monitor the `mco_node_interference_level{node}` metric to observe the correlation between interference levels and service P99 latency.
- Use `mco_scheduler_filter_rejections{node,reason}` and `mco_scheduler_score_distribution{node,dimension}` metrics to understand scheduling decisions.
- Periodically check the `detections` field in `NodeInterferenceReport/cluster` to understand triggered interference types.

### Related Operations

- **Uninstall MCO**: `helm uninstall mco`.
- **Upgrade MCO**: `helm upgrade mco many-core-orchestrator-0.0.0-latest.tgz --reuse-values`.
- **Disable specific components**: Through Helm `--set` parameters to turn off Collector, Analyzer, Kata, or scheduling plugin, see scenario selection in [Start Installation](#start-installation).
- **Temporarily disable interference awareness**: Set `psiIoSomeAvg10Threshold` in the scheduling plugin configuration to a very large value (e.g., 100.0) to essentially prevent I/O hard filtering from triggering; PSI memory full hard filter default threshold is 0.0 (i.e., only triggers when PSI memory full > 0), which can keep its default value.

## Appendix

### Key Helm Parameters

**Table 9** Key Helm Parameters

| Parameter | Default Value | Description |
|---|---|---|
| `global.imageRegistry` | `cr.openfuyao.cn/openfuyao` | Global image registry. |
| `global.systemNamespace` | `mco-system` | MCO component namespace. |
| `volcano.enabled` | `true` | Whether to install Volcano sub-chart. |
| `kata-deploy.enabled` | `true` | Whether to install Kata Deploy sub-chart. |
| `analyzer.enabled` | `true` | Whether to deploy Analyzer. |
| `analyzer.grpc.port` | `50051` | Analyzer gRPC service port. |
| `analyzer.metrics.port` | `9878` | Analyzer metrics port. |
| `analyzer.smoothWindow` | `10s` | Analyzer sample smoothing window. |
| `analyzer.crFlushInterval` | `1s` | Analyzer write interval for `NodeInterferenceReport`. |
| `collector.enabled` | `true` | Whether to deploy Collector. |
| `collector.collectInterval` | `1s` | Collector collection interval. |
| `collector.sampleDuration` | `1s` | Single sampling duration. |
| `collector.metricsPort` | `9877` | Collector metrics port. |
| `collector.collectors.llcMissRate` | `true` | Whether to collect LLC miss rate. |
| `collector.collectors.llcOccupancy` | `true` | Whether to collect LLC occupancy. |
| `collector.collectors.ioLatency` | `true` | Whether to collect block I/O latency. |
| `collector.collectors.psi` | `true` | Whether to collect PSI. |
| `collector.commands.perf` | `/usr/bin/perf` | perf command path injected during Helm deployment; when not injected through Chart, the program defaults to using `perf`. |
| `collector.commands.biolatency` | `/usr/share/bcc/tools/biolatency` | biolatency command path injected during Helm deployment; when not injected through Chart, the program defaults to using `biolatency`. |
| `mcoPlugin.enabled` | `true` | Whether to enable MCO scheduling plugin. |
| `mcoPlugin.manageSchedulerConfig` | `true` | Whether the Chart renders `volcano-scheduler-configmap`. Must be set to `false` when integrating with an existing Volcano, otherwise existing configuration will be overwritten. |

### Environment Variables

**Table 10** Collector Environment Variables

Collector main environment variables:

| Environment Variable | Default Value | Description |
|---|---|---|
| `NODE_NAME` | hostname | Node name, from `spec.nodeName` in Chart. |
| `ANALYZER_ADDR` | `mco-analyzer.mco-system.svc.cluster.local:50051` | Analyzer gRPC address. |
| `COLLECT_INTERVAL` | `1s` | Collection period. |
| `SAMPLE_DURATION` | Same as `COLLECT_INTERVAL` | Single sampling duration. |
| `CACHE_PATH` | `/var/log/mco-collector/metrics.jsonl` | Local JSONL cache path. |
| `PERF_CMD` | `perf` | perf command path. |
| `BIOLATENCY_CMD` | `biolatency` | biolatency command path. |
| `COLLECTOR_METRICS_PORT` | `9877` | metrics port. |
| `METRICS_ADDR` | empty | Collector's own `/metrics` HTTP service listening address; when set, takes priority over `COLLECTOR_METRICS_PORT`. |
| `GRPC_TIMEOUT` | `3s` | Collector's gRPC timeout for reporting to Analyzer. |
| `COLLECTOR_LLC_MISS_RATE_ENABLED` | `true` | Whether to enable LLC miss rate collection. |
| `COLLECTOR_LLC_OCCUPANCY_ENABLED` | `true` | Whether to enable LLC occupancy collection. |
| `COLLECTOR_IO_LATENCY_ENABLED` | `true` | Whether to enable block I/O latency collection. |
| `COLLECTOR_PSI_ENABLED` | `true` | Whether to enable PSI collection. |

**Table 11** Analyzer Environment Variables

Analyzer main environment variables:

| Environment Variable | Default Value | Description |
|---|---|---|
| `GRPC_ADDR` | `:50051` | gRPC listening address. |
| `METRICS_ADDR` | `:9878` | Analyzer's own `/metrics` HTTP service listening address. |
| `SMOOTH_WINDOW` | `10s` | Smoothing window. |
| `REPORT_TTL` | `60s` | Analyzer report TTL. |
| `CR_FLUSH_INTERVAL` | `1s` | CR write interval. |
| `DETECTION_RULES_NAMESPACE` | `mco-system` | Namespace where detection rules reside. |
| `DETECTION_RULES_NAME` | `cluster` | Detection rules object name. |
| `DETECTION_RULES_RESYNC_INTERVAL` | `30s` | `DetectionRuleConfig` informer resync interval. |
| `DETECTION_RULES_GROUP` | `mco.many-core.io` | Detection rules CRD API group. |
| `DETECTION_RULES_VERSION` | `v1alpha1` | Detection rules CRD API version. |
| `DETECTION_RULES_RESOURCE` | `detectionruleconfigs` | Detection rules CRD resource name. |
| `ENABLE_CR_UPDATE` | `true` | Whether to write `NodeInterferenceReport`. |
| `CR_NAME` | `cluster` | Interference report object name. |
| `CR_GROUP` | `mco.many-core.io` | Interference report CRD API group. |
| `CR_VERSION` | `v1alpha1` | Interference report CRD API version. |
| `CR_RESOURCE` | `nodeinterferencereports` | Interference report CRD resource name. |

### Observability Metrics

Collector exposes `/metrics` on port `9877` by default:

**Table 12** Collector Observability Metrics

| Metric | Description |
|---|---|
| `mco_llc_miss_rate_raw{node}` | LLC miss rate. |
| `mco_llc_occupancy_raw{node}` | LLC occupancy. |
| `mco_bio_latency_p95_ms_raw{node}` | block I/O P95 latency. |
| `mco_bio_latency_p99_ms_raw{node}` | block I/O P99 latency. |
| `mco_psi_io_some_avg10_raw{node}` | PSI I/O some avg10. |
| `mco_psi_memory_some_avg10_raw{node}` | PSI memory some avg10. |
| `mco_mem_available_ratio_raw{node}` | Available memory ratio. |
| `mco_pgscan_rate_pages_per_second_raw{node}` | Page reclaim rate. |
| `mco_collector_psi_available{node}` | Whether PSI is available; 1 indicates available. |
| `mco_snapshot_status{node}` | Sampling snapshot status; 1 indicates normal. |
| `mco_cache_references_raw{node}` | Raw cache references value collected by perf. |
| `mco_cache_misses_raw{node}` | Raw cache misses value collected by perf. |
| `mco_mem_total_kb_raw{node}` | MemTotal from `/proc/meminfo`. |
| `mco_mem_available_kb_raw{node}` | MemAvailable from `/proc/meminfo`. |
| `mco_bio_samples_raw{node}` | biolatency histogram sample count. |
| `mco_psi_io_full_avg10_raw{node}` | PSI I/O full avg10. |
| `mco_psi_memory_full_avg10_raw{node}` | PSI memory full avg10. |
| `mco_collector_status{node,collector}` | Individual collector status; 1 indicates normal. |
| `mco_snapshot_timestamp_ms{node}` | Latest sampling snapshot Unix millisecond timestamp. |

Analyzer exposes `/metrics` on port `9878` by default:

**Table 13** Analyzer Observability Metrics

| Metric | Description |
|---|---|
| `mco_node_interference_level{node}` | Node interference level calculated by Analyzer. |
| `mco_analyzer_report_generation_duration_seconds` | Report generation duration. |
| `mco_analyzer_cr_updates_total{status}` | CR update count. |
| `mco_analyzer_collector_samples_total{node}` | Number of node samples received. |
| `mco_analyzer_grpc_requests_total{method,status}` | gRPC request statistics. |
| `mco_analyzer_rule_reload_total{status}` | Rule reload count. |
| `mco_analyzer_cr_last_success_timestamp_seconds` | Timestamp of the last successful CR write. |
| `mco_analyzer_invalid_metric_values_total{node,metric,reason}` | Count of invalid metric values cleaned by Analyzer (reason: nan/inf). |
| `mco_analyzer_rule_config_info{hash,path}` | Information about the currently effective detection rule configuration (path is the CR resource path). |
| `mco_analyzer_rule_last_reload_timestamp_seconds` | Timestamp of the last successful rule load. |

The Chart supports ServiceMonitor by default, which can be automatically discovered by Prometheus Operator.

If the cluster does not have Prometheus Operator or ServiceMonitor CRD installed, you need to disable the relevant configuration:

```bash
helm upgrade mco many-core-orchestrator-0.0.0-latest.tgz \
  --set collector.serviceMonitor.enabled=false \
  --set analyzer.serviceMonitor.enabled=false
```

### Building Images

```bash
docker build -f build/analyzer.Dockerfile -t mco-analyzer:latest .
docker build -f build/collector.Dockerfile -t mco-collector:latest .
docker build -f build/scheduler.Dockerfile -t mco-volcano-scheduler:latest .
```

You can also use the Makefile:

```bash
make build-analyzer
make build-collector
make build-plugin-so
make build-scheduler
```

Development verification:

```bash
go test ./...
```

### Best Practices

- Keep online services on the default runc or existing runtime, and explicitly set high-I/O, high-memory-bandwidth, and cache-scanning offline tasks to `runtimeClassName: kata-*`.
- Conservatively adjust `psiIoSomeAvg10Threshold` and `bioLatencyP99Ms` during initial deployment, first observing the correlation between `mco_node_interference_level` and service P99.
- Calibrate thresholds separately for different machine types. Differences in LLC, memory bandwidth, and I/O stack capabilities affect the meaning of the same threshold.
- Keep `fallbackWhenCROutdated=true` to avoid monitoring pipeline anomalies affecting service creation.
- Use the `detections` field to distinguish cache pollution, I/O queue contention, and memory pressure, then decide whether to avoid through scheduling, adjust thresholds, or migrate offline workloads into Kata.

## FAQ

1. Pod is not scheduled using the MCO plugin.

   ```bash
   kubectl get configmap volcano-scheduler-configmap -n volcano-system -o yaml
   kubectl logs -n volcano-system deploy/mco-volcano-scheduler
   ```

   Confirm that the configuration contains `mco-plugin`, and that the scheduler image with the plugin is being used.

2. No interference report is generated.

   ```bash
   kubectl logs -n mco-system deploy/mco-analyzer
   kubectl logs -n mco-system ds/mco-collector
   kubectl get nir cluster -o yaml
   ```

   Focus on checking whether Collector can connect to `ANALYZER_ADDR`, and whether Analyzer has permission to write `nodeinterferencereports/status`.

3. PSI metrics are unavailable.

   ```bash
   kubectl get nir cluster -o yaml
   ```

   Check the target node's `psiAvailable` and `availability` fields. When PSI is unavailable, the system will still perform degraded calculations, but PSI-related Filters will not take effect.

4. Kata Pod cannot be scheduled.

   ```bash
   kubectl get runtimeclass
   kubectl get nodes -L katacontainers.io/kata-runtime,mco.many-core.io/kata-ready
   kubectl describe pod <pod-name> -n <namespace>
   ```

   Pods requesting `runtimeClassName: kata-*` will only be scheduled to nodes that simultaneously have `katacontainers.io/kata-runtime=true` and `mco.many-core.io/kata-ready=true`.

5. Collector metrics are abnormal.

   ```bash
   kubectl logs -n mco-system ds/mco-collector
   kubectl get ds mco-collector -n mco-system -o yaml
   ```

   Focus on checking whether the host has `perf`, `biolatency`, kernel module directories, and required collection permissions.

   I/O latency metrics are collected using eBPF, typically through the `biolatency` tool. If I/O metric collection fails, focus on checking whether the node has installed kernel headers matching the currently running kernel version, for example confirming whether `/lib/modules/$(uname -r)/build` exists. When matching kernel headers are missing, the eBPF program may be unable to compile or load.

   **Configuration Method:**

   ```bash
   yum install kernel-devel-$(uname -r)
   ```

   After installation, verify that the following two paths appear:
   ```bash
   ls -l /lib/modules/$(uname -r)/build
   ls -ld /usr/src/kernels/$(uname -r) 
   ```

   And the following three commands return matching versions:
   ```bash
   uname -r
   rpm -qa | grep kernel-devel
   ls -ld /lib/modules/$(uname -r)/build
   ```

   Finally, restart the collector.

6. PSI metrics are unavailable but you wish to enable them.

   ```bash
   kubectl get nir cluster -o yaml
   ```

   If the node report shows `psiAvailable: false`, check whether the node kernel supports PSI, and confirm that the boot parameter has enabled PSI. A common approach is to add `psi=1` to the node kernel boot parameters, restart the node, and then confirm whether `/proc/pressure/cpu`, `/proc/pressure/memory`, and `/proc/pressure/io` exist.
