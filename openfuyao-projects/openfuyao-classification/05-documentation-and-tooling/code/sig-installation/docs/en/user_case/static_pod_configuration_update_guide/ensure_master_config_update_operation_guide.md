# Update Control Plane Static Pod Configuration

> Scope: **updating the static Pod configuration of the control plane (`kube-apiserver` / `kube-controller-manager` / `kube-scheduler`)** (`extraArgs`).
> To update etcd parameters, see [Update etcd Static Pod Configuration](./ensure_etcd_config_update_operation_guide.md).
> For design details, see [Declarative Cluster Static Pod Configuration Update Design](../../../../design/支持静态pod配置热更新/声明式集群静态pod配置更新方案.md) (Chinese only).

---

## 1. Overview

### 1.1 What It Is, Who Uses It, and When to Use It

The three control-plane components (`kube-apiserver`, `kube-controller-manager`, and `kube-scheduler`) are the most critical components of a Kubernetes cluster. During day-to-day operation, operators often need to tune their runtime parameters, for example: increasing the retention period of apiserver audit logs, adjusting the scheduling behavior of the scheduler, or tightening the authorization parameters of the controller manager.

This feature is intended for **cluster operators and SREs** and applies to the following scenario: **without changing the cluster version and without running a complete upgrade procedure, modify only the runtime parameters (`extraArgs`) of the three control-plane components so that the change takes effect safely and with minimal impact.**

Capabilities it provides:

- **Declarative**: modify the `extraArgs` of the control-plane components directly in the `BKECluster` resource (`BKECluster` / bc) using any of `kubectl edit`, `kubectl patch`, or `kubectl apply`. The controller distributes the change automatically; there is no need to log in to nodes and edit files by hand.
- **Minimal impact**: only the manifests of the components that actually changed are regenerated. Components that did not change, etcd, and worker nodes are not affected.
- **Controllable**: the update runs serially across Master nodes; the next node is updated only after the previous one succeeds and becomes Ready, which reduces risk.

### 1.2 Background: What Is a Static Pod and Why Updates Differ

- **What a static Pod is**: the manifest of a static Pod is an ordinary YAML file in the `/etc/kubernetes/manifests/` directory on the node (for example `/etc/kubernetes/manifests/kube-apiserver.yaml`). The file is **read and managed directly by the kubelet on the node and does not go through the API Server**: unlike a normal Pod, it is not submitted to the cluster with `kubectl` and then scheduled by the scheduler. Instead, the kubelet continuously watches the directory and, as soon as the file content changes, automatically recreates the corresponding Pod (deleting the old Pod first and then creating a new one).
- **Why it differs from a normal Pod**: creating, updating, and deleting a normal Pod are all handled by the API Server and the corresponding controller, and changing its configuration is usually done with `kubectl edit` or by updating a Deployment. The "configuration" of a static Pod, however, is the manifest file on disk, so changing a parameter essentially means **rewriting that file and letting the kubelet restart the component**. Static Pods are not stored in etcd and are not managed by any Workload controller, so they do not have the "update the Deployment and get a rolling update" capability; any change must be applied directly to the node file system.
- **The three control-plane components are static Pods**: `kube-apiserver`, `kube-controller-manager`, and `kube-scheduler` all run as static Pods (etcd is handled separately by [Update etcd Static Pod Configuration](./ensure_etcd_config_update_operation_guide.md)). This is exactly **why a dedicated feature is needed to change their parameters**: the essence of the feature is "rewrite the manifest of the corresponding component under `/etc/kubernetes/manifests/` on every control-plane node and wait for the kubelet to restart it", not a conventional Pod update flow.

### 1.3 Characteristics

- **Component-level minimal change**: changing an apiserver parameter does not cause the controller-manager or scheduler manifests to be rewritten.
- **etcd is not involved**: etcd configuration updates are handled by a dedicated flow. This phase does not touch the etcd manifest and performs **no etcd backup**.
- **Node-by-node serial execution**: the next Master node is processed only after the previous one succeeds and becomes Ready.
- **Retryable on failure**: `status.controlPlaneConfigHash` is written back only after **all** Master nodes succeed; on failure the old baseline is kept and the next reconcile round retries automatically.

### 1.4 Execution Flow (Internal Implementation, for Troubleshooting)

1. The bc is modified (by `kubectl apply`, `kubectl edit`, or `kubectl patch`) -> the `validateStaticPodConfigUpdate` webhook validates that the three control-plane components are all present and that no etcd immutable field has been changed (if validation fails, the write operation fails immediately).
2. The capbke controller `Reconcile` -> `PhaseFlow` calls `NeedExecute` on each phase in order (`EnsureEtcdConfigUpdate` runs before `EnsureMasterConfigUpdate`).
3. `EnsureMasterConfigUpdate.NeedExecute` decides: the control plane is initialized, the Kubernetes and etcd versions in spec and status are aligned, Master nodes exist, the desired hash is not empty, and the baseline is not empty and **differs from the desired hash** -> trigger.
4. `Execute`: triggers `reconcileMasterConfigUpdate()`.
5. The `UpgradeControlPlaneConfig` command is issued to each Master node serially, carrying `controlPlaneConfigOldHash` / `controlPlaneConfigNewHash` (**these two parameters are used only by this phase and do not affect other phases**), and the controller waits for completion.
6. Node bkeagent: `updateControlPlaneConfig(old, new)` -> records the current Pod hash of the three components -> compares them component by component to obtain the changed set -> in the order `kube-apiserver -> kube-controller-manager -> kube-scheduler`: skip unchanged components (log `config unchanged, skip`), and for changed components re-render `/etc/kubernetes/manifests/<component>.yaml` -> `waitComponentReady` (wait for the Pod hash to change and the Pod to be Running).
7. `status.controlPlaneConfigHash = newHash` is written back only after **all** Master nodes succeed; if any node fails, the old baseline is kept and the next reconcile round retries automatically.

---

## 2. Prerequisites

| No. | Condition | How to check |
| --- | --- | --- |
| 1 | The cluster is deployed and healthy (the control plane is initialized) | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.clusterStatus}'` |
| 2 | **The baseline hash is established**: `status.controlPlaneConfigHash` is not empty | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.controlPlaneConfigHash}{"\n"}'` |
| 3 | The cluster has nodes with the Master role | `kubectl get bkenode` |
| 4 | **The three control-plane component configurations all exist** (`apiServer` / `controllerManager` / `scheduler`), and only `extraArgs` is changed this time | See [4.2 Modify Control-Plane Component Parameters](#42-modify-control-plane-component-parameters) |
| 5 | `kubernetesVersion` and `etcdVersion` **remain unchanged** | See [4.2 Modify Control-Plane Component Parameters](#42-modify-control-plane-component-parameters) |
| 6 | No other phase is currently running (upgrade, scaling, etc.) | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.phase}'` |

> **About condition 2**: a newly created cluster establishes the baseline automatically after one health check (`EnsureCluster`); an **existing cluster** that is upgraded to a version containing this feature also needs one health-check round before the baseline is written. If `status.controlPlaneConfigHash` is empty, changing parameters **does not** trigger an update (this prevents static Pods from being rewritten by mistake during deployment).

> **About the baseline hash (`status.controlPlaneConfigHash`)**: it records the hash of the **configuration that was last delivered successfully to the static Pods on the nodes**, that is, the "effective configuration baseline". The hash is computed from **all fields that participate in hash computation**, which includes the `extraArgs` of each component as well as `extraVolumes` / `certSANs` / `apiEndpoint`, which participate in the hash but are currently not rendered into the manifest. **Only `extraArgs` is actually rendered into the manifest**; the other fields participate in the hash (changing them changes the hash value and therefore triggers this phase) but are not written into the manifest. After every `apply`, the controller computes the **desired hash** from the current `spec`; only when the **desired hash differs from the baseline hash** does this phase re-deliver the configuration. The baseline is written only after the first health check (`EnsureCluster`) of a new or existing cluster completes, which prevents static Pods from being rewritten by mistake during deployment or early stages. After all nodes are updated successfully, the baseline is written back with the latest desired hash; once they are aligned, this phase is not triggered repeatedly.

---

## 3. Usage Constraints

### 3.1 Only `extraArgs` Really Takes Effect

The manifest templates of the three control-plane components currently **render only `extraArgs`**:

```yaml
# utils/bkeagent/mfutil/tmpl/k8s/kube-apiserver.yaml.tmpl (excerpt)
- command:
  - kube-apiserver
  ...
  {{- range . | extraArgs }}
  - --{{ . }}
  {{- end }}
```

Each entry in `extraArgs` is appended to the end of the component startup command as `--<key>=<value>`. When it conflicts with a built-in template parameter, the **user parameter wins (last-wins, overriding the built-in value)**.

| Component | Field | In hash | Rendered into manifest |
| --- | --- | :---: | :---: |
| `apiServer` | `extraArgs` | Yes | Yes — **truly takes effect** |
| `apiServer` | `extraVolumes` | Yes | No — not yet implemented upstream |
| `apiServer` | `certSANs` | Yes | No — not yet implemented upstream |
| `apiServer` | `apiEndpoint` (host/port) | Yes | No — not yet implemented upstream |
| `controllerManager` | `extraArgs` | Yes | Yes — **truly takes effect** |
| `controllerManager` | `extraVolumes` | Yes | No — not yet implemented upstream |
| `scheduler` | `extraArgs` | Yes | Yes — **truly takes effect** |
| `scheduler` | `extraVolumes` | Yes | No — not yet implemented upstream |
| `etcd` | all fields | No | Handled independently by the etcd phase |

**Conclusion: adjust only `extraArgs` during operations.**

> <span style="color:red;font-weight:bold;">Note: for fields that are not rendered into the manifest (such as `extraVolumes` / `certSANs` / `apiEndpoint`), the update flow still runs `waitComponentReady` to completion and writes the new hash value back to the status, so the flow is reported as "successful"; however, the target configuration does not actually take effect (for example, the volume is not mounted). After that the hashes are aligned and this phase no longer retries, which ends up as "the configuration was modified, the flow reports success, but nothing actually took effect".</span>

Changing `kubernetesVersion`, `etcdVersion`, `kubelet.*`, `networking.*`, or any Repo field **does not** trigger this phase.

### 3.2 Parameter Validity Cannot Be Guaranteed by the Program (Manual Verification Required)

This feature only does **"detect a configuration change -> re-issue the manifest"**; it **does not validate parameter semantics**. Operators must confirm all of the following manually:

1. **Wrong parameter names or values are not intercepted**: a misspelled `--max-requests-inflight`, an out-of-range value, or a boolean written with a non-standard literal (for example, a Chinese literal) produces no error from the webhook or the hash computation. The component fails to start with the illegal parameter (CrashLoopBackOff), and after `waitComponentReady` times out the command fails — **during this period the control plane is unavailable and the cluster cannot be managed**.
2. **Overriding built-in parameters is extremely risky**: `extraArgs` overrides built-in template parameters with the same name (for example the apiserver parameters `--secure-port`, `--advertise-address`, `--etcd-servers`, and `--service-cluster-ip-range`). A wrong override can directly make the apiserver unavailable.
3. **apiserver parameters have the widest impact**: once rate limiting, auditing, FeatureGates, or certificate-related parameters are misconfigured, the API of the entire cluster can become unavailable.
4. **No pre-check and no dry-run**: there is currently no parameter allowlist, no syntax validation, and no change-rehearsal capability.

**Recommended practices**:

- Validate the parameters on a test cluster first: confirm that the component becomes Ready and cluster functions work normally before applying them to production.
- **Change only one category of parameters for one component at a time** so problems can be located.
- Before `apply`, check every parameter name and value against the official parameter documentation of the current Kubernetes version.
- Run the change during an **off-peak window** and reserve a rollback window.
- In a multi-Master scenario, note that this phase runs **node by node serially**: the control-plane capacity drops during the update, so confirm that the remaining nodes can carry the traffic.
- After `apply`, verify item by item as described in [4.4 Verification](#44-verification).

---

## 4. Procedure

> The management cluster is used as the example below. Where the `BKECluster` CR lives:
> - **Management cluster**: modify it in the management cluster itself (after the listener switch completes, a management cluster manages its own `BKECluster` / bc).
> - **Service cluster**: modify it in the management cluster.
>
> Replace `<cluster>` and `<ns>` with the actual cluster name and namespace.

### 4.1 Record the Current Configuration (for Rollback)

```bash
kubectl get bkecluster <cluster> -n <ns> -o yaml > bkecluster-backup-$(date +%Y%m%d%H%M).yaml

# Record the current baseline hash
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.controlPlaneConfigHash}{"\n"}'

# Example:
#root@test:~/bkeadmTest# kubectl get bc -A
#NAMESPACE     NAME                  PHASE          STATE     CLUSTER STATUS   ENDPOINT          ENDPOINT PORT   VERSION        AGENT STATUS   AGE
#bke-cluster   bke-cluster-manager   ClusterReady   Healthy   Ready            192.168.122.190   6443            v1.34.3-of.1   1/1            5m25s
#root@test:~# kubectl get bkecluster bke-cluster-manager -n bke-cluster -o jsonpath='{.status.controlPlaneConfigHash}{"\n"}'
#{"apiServer":"a9960a43cd92903e44af103123bfc11912d95beaae553175796e160788c9964d","controllerManager":"23e5bb4df921abc205b24146ab796f09c329dfafdf07eb84254bcd63cb6598d6","scheduler":"c4587d74be70eb5b6f634df4d421fabf0156cc79043568db59f0a340b326dfd5"}
```

### 4.2 Modify Control-Plane Component Parameters

The method used to modify the bc does not affect how this feature is triggered: `kubectl edit`, `kubectl patch`, and `kubectl apply -f <new configuration YAML file>` are **equivalent** — they all change the `spec` of the `BKECluster`. As long as the `extraArgs` of a component under `spec.clusterConfig.cluster` changes and the conditions in [2. Prerequisites](#2-prerequisites) are met, the controller reconciles again and triggers this phase.

**Method 1: edit directly (recommended, easier to review manually)**

```bash
kubectl edit bkecluster <cluster> -n <ns>
```

Locate `apiServer` / `controllerManager` / `scheduler` under `spec.clusterConfig.cluster`:

```yaml
spec:
  clusterConfig:
    cluster:
      apiServer:
        extraArgs:
          max-requests-inflight: "800"        # Example: rate limiting
          max-mutating-requests-inflight: "400"
          feature-gates: "RemoveSelfLink=false"
      controllerManager:
        extraArgs:
          concurrent-deployment-syncs: "10"   # Example: concurrent sync count
          kube-api-qps: "50"
      scheduler:
        extraArgs:
          kube-api-qps: "50"                  # Example: QPS limit
```

> **Important**: the `apiServer` / `controllerManager` / `scheduler` **sections must all be present**. If any section is missing or empty, the webhook rejects the write directly (`APIServer/ControllerManager/Scheduler must all be present`). If a component temporarily needs no extra parameters, keep an empty section, for example `scheduler: {}` or `scheduler: {extraArgs: {}}`.

**Method 2: patch (suitable for automation)**

```bash
kubectl patch bkecluster <cluster> -n <ns> --type=merge -p '
{
  "spec": {
    "clusterConfig": {
      "cluster": {
        "apiServer": {
          "extraArgs": {
            "max-requests-inflight": "800",
            "max-mutating-requests-inflight": "400"
          }
        }
      }
    }
  }
}'
```

**Method 3: apply a new YAML file (suitable for managing configuration as code)**

```bash
kubectl apply -f new-bc.yaml
```

> Note: `new-bc.yaml` is the `BKECluster` manifest with the modified `extraArgs`; it can be produced by editing the backup file created in [4.1 Record the Current Configuration (for Rollback)](#41-record-the-current-configuration-for-rollback).

> **Notes**
> - `extraArgs` uses **full-overwrite** semantics. A merge patch only merges keys with the same name; to delete a parameter you must explicitly remove that key (delete the line with `edit`, or pass `null` with a JSON merge patch).
> - Do not modify `kubernetesVersion` or `etcdVersion` in the same `apply`; otherwise this phase is not executed and the version-upgrade phase handles the change instead.
> - Do not delete or empty any of the three control-plane component configuration sections.

### 4.3 Monitor Execution Progress

```bash
# 1) Cluster phase
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.phase}{"\n"}'

# 2) Issued commands (the phase should be UpgradeControlPlaneConfig)
kubectl get command -l bke.bocloud.com/cluster-command

# 3) Controller logs (reason keyword: EnsureMasterConfigUpdate)
cat /var/log/openFuyao/bke-controller-manager/<bke-controller-manager-pod-name> | grep -i "EnsureMasterConfigUpdate"

# 4) Node bkeagent logs
# Keywords: update control plane static pod config / component <name> config changed, regenerate manifest
#           component <name> config unchanged, skip / component <name> config update success
cat /var/log/openFuyao/bkeagent.log | grep -i "update control plane static pod config"
```

### 4.4 Verification

```bash
# 1) The manifests on the node have been updated
# Log in to the target Master node as root and check the update time of each component YAML file
cd /etc/kubernetes/manifests/
ls -l
# Example output (the YAML file of a changed component should have a later update time than the others)
# total 24
# drwxr-xr-x 2 root root 4096 Sep 14 18:27 ./
# drwxr-xr-x 4 root root 4096 Sep 14 18:27 ../
# -rw-r--r-- 1 root root 1941 Sep 14 18:27 etcd.yaml
# -rw-r--r-- 1 root root 3426 Sep 14 18:55 kube-apiserver.yaml
# -rw-r--r-- 1 root root 2519 Sep 14 18:27 kube-controller-manager.yaml
# -rw-r--r-- 1 root root 1590 Sep 14 18:27 kube-scheduler.yaml

# 2) The actual startup parameters of the static Pods have taken effect
kubectl -n kube-system get pod kube-apiserver-<node> -o yaml | grep -A 60 'command:'
kubectl -n kube-system get pod kube-controller-manager-<node> -o yaml | grep -A 40 'command:'
kubectl -n kube-system get pod kube-scheduler-<node> -o yaml | grep -A 30 'command:'

# 3) All three control-plane components are Running (on all Master nodes)
kubectl -n kube-system get pod -l tier=control-plane -o wide

# 4) Unchanged components were not restarted (their AGE should be much larger than that of the just-updated component)
kubectl -n kube-system get pod -l tier=control-plane -o wide

# 5) The cluster is healthy
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.clusterStatus}{"\n"}'

# 6) The baseline hash has been refreshed
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.controlPlaneConfigHash}{"\n"}'
```

> **About item 6**: the hash returned here must be **different** from the one recorded in [4.1 Record the Current Configuration (for Rollback)](#41-record-the-current-configuration-for-rollback); if it stays identical to that record, this phase was not triggered or the configuration was not actually delivered.

**Pass criteria**: the baseline hash has changed and is stable (not triggering repeatedly); the three control-plane components are Running on all Master nodes; unmodified components were not restarted; the cluster `clusterStatus` is Ready; `kubectl get nodes` works normally.

---

## 5. Troubleshooting and Handling

| Symptom | Possible cause | Action |
| --- | --- | --- |
| `kubectl apply` fails with `APIServer/ControllerManager/Scheduler must all be present` | One of the three component sections was deleted or emptied | Restore that component section (an empty section is acceptable) and apply again |
| `kubectl apply` fails with an etcd field `Forbidden` | The etcd fields `dataDir` / `serverCertSANs` / `peerCertSANs` were modified at the same time | Revert the changes to the etcd immutable fields |
| `apply` succeeds but no command appears for a long time and the hash does not change | (1) `status.controlPlaneConfigHash` is empty (baseline not established); (2) the changed field is not part of the hash; (3) a version field was changed at the same time | Wait for one health-check round to establish the baseline; confirm only `extraArgs` was changed; confirm no version field was changed |
| The command fails and the logs show `MasterConfigUpdateFailed` | (1) An illegal parameter prevents the component from starting; (2) waiting for the Pod hash to change timed out | Check the node bkeagent logs and the container logs; roll back as described in [6. Rollback](#6-rollback) |
| `kube-apiserver` is in CrashLoopBackOff | The `extraArgs` parameter name or value is wrong, or a critical built-in parameter was overridden | Roll back the configuration; verify the parameters and retry |
| Only some Master nodes were updated | A node is unhealthy (disk, port, kubelet status, etc.) | Fix the node and wait for the automatic retry (the old baseline is kept, so retries continue) |
| `extraVolumes` / `certSANs` / `apiEndpoint` were modified; the flow reports success but the configuration does not take effect (the volume is not mounted) | These fields participate in the hash (so they trigger the update flow) but are **not rendered into the manifest**; the flow writes the new hash back as usual, after which the hashes are aligned and no retry occurs | Such fields are not supported by hot update; do not rely on them. Revert the change and keep only the `extraArgs` change; if you really need to mount such configuration, rebuild the control plane |

Troubleshooting commands:

```bash
# Command objects and failed nodes
kubectl get command -l bke.bocloud.com/cluster-command -o yaml

# Failure reason on the controller side
kubectl -n <capbke-ns> logs deploy/capbke-controller-manager | grep -i MasterConfigUpdateFailed

# Node bkeagent logs
ssh <master-node> 'journalctl -u bkeagent -f'

# Component container logs from the container runtime
ssh <master-node> 'crictl ps -a | grep -E "kube-apiserver|kube-controller-manager|kube-scheduler"; crictl logs <container-id>'
```

### 5.1 Recovery After a Failed `EnsureMasterConfigUpdate` Reconcile (Manual Retry)

> <span style="color:red;font-weight:bold;">Important prerequisite: you must correct the bc configuration before triggering reconciliation again; otherwise reconciliation only fails once more and exhausts the retries again.</span>

When `EnsureMasterConfigUpdate` fails (for example, a component enters CrashLoopBackOff because of an illegal parameter, or the command times out because the component does not become Ready), recover in the following order:

1. <span style="color:red;font-weight:bold;">Wait until the phase retries are exhausted</span>: wait for the current retry round to end. At that point the cluster `CLUSTER STATUS` becomes `UpgradeFailed` (reconciliation has stopped and no longer keeps consuming retries).
2. <span style="color:red;font-weight:bold;">Correct the bc configuration</span>: after locating the cause using the table above, restore the original value or set a correct value for `extraArgs` of the corresponding component under `spec.clusterConfig.cluster` (see [6. Rollback](#6-rollback)). This step cannot be skipped.
3. **Trigger reconciliation again**: add the retry annotation to the bc to manually trigger a new reconciliation round:

```bash
kubectl annotate bkecluster <cluster> -n <ns> bke.bocloud.com/retry="" --overwrite
```

After that, monitor the phase and command issuance as described in [4.3 Monitor Execution Progress](#43-monitor-execution-progress), and verify as described in [4.4 Verification](#44-verification).

> <span style="color:red;font-weight:bold;">Notes</span>
> - <span style="color:red;font-weight:bold;">Correcting the configuration is a prerequisite</span>: if the configuration items are still invalid, triggering reconciliation again only fails once more and consumes retries repeatedly.
> - <span style="color:red;font-weight:bold;">Exception for self-managed management clusters (no recovery method available)</span>: after the listener switch, a management cluster is reconciled by its own controller managing its own bc. If its control plane is already unavailable (the `kubectl` operations above cannot be executed inside the cluster), <span style="color:red;font-weight:bold;">no recovery method is available inside this cluster, and the upper-layer cluster must be healthy</span> before anyone can intervene.

---

## 6. Rollback

This feature provides **no automatic rollback**. To roll back, **restore `extraArgs` to the original values and apply again** to trigger a new configuration update.

### 6.1 Rollback Example (Explained with Hash Values)

> The hash values below are abbreviated for readability; the actual values are longer hexadecimal digests. The key point for judging whether a rollback succeeded is whether the hash is consistent with the original baseline.

Assume that the **initial value** of the control-plane baseline hash `status.controlPlaneConfigHash` of a management cluster is `9f3a2b7c` (corresponding to the `extraArgs` delivered successfully the first time: apiserver `--audit-log-maxage=7`, with the controller-manager and scheduler at their default values).

| Stage | apiserver.extraArgs value | Desired hash | Description |
| --- | --- | --- | --- |
| Initial baseline | `--audit-log-maxage=7` | `9f3a2b7c` | Written to `status.controlPlaneConfigHash` after the first delivery |
| After the change | `--audit-log-maxage=30` | `4d8e1a6c` | The desired hash differs from the baseline, triggering the configuration update of this phase; after the rolling update, `status.controlPlaneConfigHash` is written back as `4d8e1a6c` |
| After the rollback | `--audit-log-maxage=7` (original value restored) | `9f3a2b7c` | The desired hash `9f3a2b7c` differs from the current baseline `4d8e1a6c`, so a rolling update is triggered again; after it completes, `status.controlPlaneConfigHash` is written back as `9f3a2b7c`, consistent with the initial baseline |

**Criteria for a completed rollback**: **after restoring `extraArgs` to the original value and applying again, and after the rolling update finishes on all Master nodes, `status.controlPlaneConfigHash` should again equal the initial baseline `9f3a2b7c`.** In other words, the hash after the rollback should be consistent with the original baseline hash, which indicates that the static Pod configuration has been restored to the state before the change.

```bash
# 1) Restore from the backup
kubectl apply -f bkecluster-backup-<timestamp>.yaml

# Or: use edit to restore the original parameters manually
# (for example, restore apiserver.extraArgs to --audit-log-maxage=7)
kubectl edit bkecluster <cluster> -n <ns>

# 2) Check whether the rollback is complete: the hash should return to the initial baseline 9f3a2b7c
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.controlPlaneConfigHash}{"\n"}'
kubectl -n kube-system get pod -l tier=control-plane -o wide
```

A rollback also regenerates the manifests of the corresponding components and waits for Ready; the verification method is the same as in [4.4 Verification](#44-verification).

---

## 7. Notes

1. <span style="color:red;font-weight:bold;">Only `extraArgs` takes effect</span>: fields such as `extraVolumes` / `certSANs` / `apiEndpoint` participate in the hash (changing them triggers the update flow) but are not rendered into the manifest, which ends up as <span style="color:red;font-weight:bold;">"the configuration was modified, the flow reports success, but nothing actually took effect"</span>. Do not rely on such hot updates (see [3. Usage Constraints](#3-usage-constraints)).
2. **Parameter validity must be verified manually**: the program does not validate parameter semantics; wrong parameters directly prevent the control-plane components from starting.
3. **The three control-plane components must all be present**: if any configuration section is missing, the webhook rejects the request.
4. **No etcd backup and no etcd involvement**: etcd configuration changes are handled by the dedicated etcd configuration update flow.
5. **No interruption guarantee**: the corresponding component is briefly unavailable while its static Pod is being recreated. In addition, this phase runs **node by node serially** across Master nodes, so the control-plane capacity drops during the update — evaluate the capacity in multi-Master scenarios.
6. **Do not change versions at the same time**: when `kubernetesVersion` or `etcdVersion` changes, this phase is not executed and the version-upgrade phase handles the change instead.
7. **Component-level minimal change applies only to "components that changed"**: the decision is based on a component-by-component comparison of the old and new hash structures; components that did not change are skipped.
8. **No automatic rollback**: rollback means applying the original configuration again.
9. **Only Master nodes are affected**: worker nodes are not affected by this phase; kubelet parameters are also out of scope of this feature.
10. **Failures can be retried manually**: after a failed reconcile, wait until the retries are exhausted (`CLUSTER STATUS` is `UpgradeFailed`), correct the configuration, and re-trigger reconciliation with the `bke.bocloud.com/retry` annotation. However, <span style="color:red;font-weight:bold;">when a management cluster manages its own `BKECluster` (bc) and its control plane is already unavailable, no recovery method is available inside this cluster and the upper-layer cluster must be healthy</span> (see [5.1 Recovery After a Failed EnsureMasterConfigUpdate Reconcile (Manual Retry)](#51-recovery-after-a-failed-ensuremasterconfigupdate-reconcile-manual-retry)).
