# Update etcd Static Pod Configuration

> Scope: **only updating the etcd static Pod configuration** (`etcd.extraArgs`).
> To update the parameters of `kube-apiserver` / `kube-controller-manager` / `kube-scheduler`, see
> [Update Control Plane Static Pod Configuration](./ensure_master_config_update_operation_guide.md).
> For design details, see [Declarative Cluster Static Pod Configuration Update Design](../../../../design/支持静态pod配置热更新/声明式集群静态pod配置更新方案.md) (Chinese only).

---

## 1. Overview

### 1.1 What It Is, Who Uses It, and When to Use It

etcd is the underlying key-value store of a Kubernetes cluster, and almost all cluster state is stored in it. During day-to-day operation, operators sometimes need to tune the runtime parameters of etcd, for example: increasing `quota-backend-bytes` (backend database quota), adjusting heartbeat and election timeouts, or changing the snapshot frequency.

This feature is intended for **cluster operators and SREs** and applies to the following scenario: **without changing the etcd version and without running a complete upgrade procedure, modify only the runtime parameters (`extraArgs`) of etcd so that the change takes effect safely and with minimal impact.**

Capabilities it provides:

- **Declarative**: modify `etcd.extraArgs` in the `BKECluster` resource (`BKECluster` / bc) using any of `kubectl edit`, `kubectl patch`, or `kubectl apply`. The controller distributes the change automatically; there is no need to log in to nodes and edit files by hand.
- **Minimal impact**: only the etcd manifest is regenerated. The three control-plane components and worker nodes are not affected.
- **Controllable**: the update runs serially across etcd nodes; the next node is updated only after the previous one succeeds and becomes Ready, which reduces risk.

### 1.2 Background: What Is a Static Pod and Why Updates Differ

- **What a static Pod is**: the manifest of a static Pod is an ordinary YAML file in the `/etc/kubernetes/manifests/` directory on the node (for example `/etc/kubernetes/manifests/etcd.yaml`). The file is **read and managed directly by the kubelet on the node and does not go through the API Server**: unlike a normal Pod, it is not submitted to the cluster with `kubectl` and then scheduled by the scheduler. Instead, the kubelet continuously watches the directory and, as soon as the file content changes, automatically recreates the corresponding Pod (deleting the old Pod first and then creating a new one).
- **Why it differs from a normal Pod**: creating, updating, and deleting a normal Pod are all handled by the API Server and the corresponding controller, and changing its configuration is usually done with `kubectl edit` or by updating a Deployment. The "configuration" of a static Pod, however, is the manifest file on disk, so changing a parameter essentially means **rewriting that file and letting the kubelet restart the component**. Static Pods are not stored in etcd and are not managed by any Workload controller, so they do not have the "update the Deployment and get a rolling update" capability; any change must be applied directly to the node file system.
- **etcd is also a static Pod**: the four control-plane components `kube-apiserver`, `kube-controller-manager`, `kube-scheduler`, and `etcd` all run as static Pods (the first three are handled by [Update Control Plane Static Pod Configuration](./ensure_master_config_update_operation_guide.md)). This is exactly **why a dedicated feature is needed to change etcd parameters**: the essence of the feature is "rewrite `/etc/kubernetes/manifests/etcd.yaml` on every etcd node and wait for the kubelet to restart etcd", not a conventional Pod update flow.

### 1.3 Characteristics

- **Triggered independently**: it neither depends on nor affects the three control-plane components, and supports scenarios where etcd is deployed independently or mounted separately.
- **No etcd backup**: unlike a version upgrade, a pure configuration update does not trigger any backup action.
- **Node-by-node serial execution**: the next etcd node is processed only after the previous one succeeds and becomes Ready.
- **Retryable on failure**: `status.etcdConfigHash` is written back only after **all** etcd nodes succeed; on failure the old baseline is kept and the next reconcile round retries automatically.

### 1.4 Execution Flow (Internal Implementation, for Troubleshooting)

1. The bc is modified (by `kubectl apply`, `kubectl edit`, or `kubectl patch`) -> the `validateStaticPodConfigUpdate` webhook validates that the three control-plane components are all present and that no etcd immutable field has been changed (if validation fails, the write operation fails immediately).
2. The capbke controller `Reconcile` -> `PhaseFlow` calls `NeedExecute` on each phase in order.
3. `EnsureEtcdConfigUpdate.NeedExecute` decides: the control plane is initialized, the etcd and Kubernetes versions are aligned, etcd nodes exist, the desired hash is not empty, and the baseline is not empty and **differs from the desired hash** -> trigger.
4. `Execute`: issues the `UpgradeEtcdConfig` command to each etcd node serially and waits for completion.
5. Node bkeagent: `updateEtcdConfig()` -> records the current etcd Pod hash -> re-renders `/etc/kubernetes/manifests/etcd.yaml` -> `waitComponentReady` (waits for the Pod hash to change and the Pod to be Running).
6. `status.etcdConfigHash` is written back only after **all** etcd nodes succeed; if any node fails, the old baseline is kept and the next reconcile round retries automatically.

---

## 2. Prerequisites

| No. | Condition | How to check |
| --- | --- | --- |
| 1 | The cluster is deployed and healthy (the control plane is initialized) | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.clusterStatus}'` |
| 2 | **The baseline hash is established**: `status.etcdConfigHash` is not empty | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.etcdConfigHash}{"\n"}'` |
| 3 | The cluster has BKENodes with the etcd role | `kubectl get bkenode -l ... ` (an external etcd cluster has no etcd BKENode, so this feature does not take effect) |
| 4 | **Only** `etcd.extraArgs` is to be changed; `etcdVersion` and `kubernetesVersion` **remain unchanged** | See [4.2 Modify etcd Parameters](#42-modify-etcd-parameters) |
| 5 | No other phase is currently running (upgrade, scaling, etc.) | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.phase}'` |

> **About condition 2**: a newly created cluster establishes the baseline automatically after one health check (`EnsureCluster`); an **existing cluster** that is upgraded to a version containing this feature also needs one health-check round before the baseline is written. If `status.etcdConfigHash` is empty, changing parameters **does not** trigger an update (this prevents static Pods from being rewritten by mistake during deployment).

> **About the baseline hash (`status.etcdConfigHash`)**: it records the hash of the **configuration that was last delivered successfully to the static Pods on the nodes**, that is, the "effective configuration baseline". The hash is computed from **all fields that participate in hash computation**: `etcd.extraArgs` is actually rendered into the manifest, whereas `etcd.extraVolumes` **participates in the hash** (changing it changes the hash value and therefore triggers this phase) but is **not rendered** into the manifest. On the etcd side the hash is **still verified**: this phase is triggered only when the desired hash differs from the baseline hash. After every `apply`, the controller computes the **desired hash** from the current `spec`; only when the **desired hash differs from the baseline hash** does it re-deliver the configuration. After all etcd nodes are updated successfully, the baseline is written back with the latest desired hash; once they are aligned, this phase is not triggered repeatedly.

---

## 3. Usage Constraints

### 3.1 Only `extraArgs` Really Takes Effect

The etcd static Pod manifest template currently **renders only `extraArgs`**:

```yaml
# utils/bkeagent/mfutil/tmpl/k8s/etcd.yaml.tmpl (excerpt)
- command:
  - etcd
  ...
  {{- range . | extraArgs }}
  - --{{ . }}
  {{- end }}
```

Each entry in `extraArgs` is appended to the end of the etcd startup command as `--<key>=<value>`. When it conflicts with a built-in template parameter, the **user parameter wins (last-wins, overriding the built-in value)**.

The following fields **participate in hash computation** (changing them triggers the update flow) but are **not rendered into the manifest**. Do **not** rely on them during operations:

> <span style="color:red;font-weight:bold;">Note: for fields that are not rendered into the manifest (such as `etcd.extraVolumes`), the update flow still runs `waitComponentReady` to completion and writes the new hash value back to the status, so the flow is reported as "successful"; however, the target configuration does not actually take effect (for example, the volume is not mounted). After that the hashes are aligned and this phase no longer retries, which ends up as "the configuration was modified, the flow reports success, but nothing actually took effect".</span>

| Field | In hash | Rendered into manifest |
| --- | --- | :---: | :---: |
| `etcd.extraArgs` | Yes | Yes — **the only one that truly takes effect** |
| `etcd.extraVolumes` | Yes | No — not yet implemented upstream |
| `etcd.dataDir` | No | No — immutable at bootstrap; the webhook rejects changes |
| `etcd.serverCertSANs` | No | No — immutable at bootstrap; the webhook rejects changes |
| `etcd.peerCertSANs` | No | No — immutable at bootstrap; the webhook rejects changes |

In addition, on the etcd side the hash is **still verified**: this phase is triggered only when the desired hash differs from the `status.etcdConfigHash` baseline; once the hashes are consistent, it is not executed repeatedly.

Changing `kubernetesVersion`, `etcdVersion`, `kubelet.*`, `networking.*`, or any Repo field **does not** trigger this phase.

### 3.2 Parameter Validity Cannot Be Guaranteed by the Program (Manual Verification Required)

This feature only does **"detect a configuration change -> re-issue the manifest"**; it **does not validate parameter semantics**. Operators must confirm all of the following manually:

1. **Wrong parameter names or values are not intercepted**: a misspelled `--heartbeat-interval`, an out-of-range value, or a wrong unit produces no error from the webhook or the hash computation. However, the etcd process fails to start with the illegal parameter, and after `waitComponentReady` times out the command fails — **during this period etcd is unavailable and the whole cluster is not writable**.
2. **Overriding built-in parameters is extremely risky**: `extraArgs` overrides built-in template parameters with the same name (for example `--data-dir`, `--listen-client-urls`, `--name`, and `--initial-cluster`). A wrong override can prevent etcd from joining the cluster or corrupt the data directory.
3. **etcd parameters have a wide impact**: heartbeat and election timeouts, quotas, and compaction policies must be evaluated against cluster size, disk performance, and network latency. Wrong values may cause frequent leader switches or even data unavailability.
4. **No pre-check and no dry-run**: there is currently no parameter allowlist, no syntax validation, and no change-rehearsal capability.

**Recommended practices**:

- Validate the parameters on a test cluster first: confirm that etcd becomes Ready and that the cluster is readable and writable before applying them to production.
- Change only **one category of parameters** at a time so problems can be located.
- Before `apply`, check every parameter name and value against the official documentation of the current etcd version.
- Run the change during an **off-peak window** and reserve a rollback window.
- After `apply`, verify item by item as described in [4.4 Verification](#44-verification).

---

## 4. Procedure

> The management cluster is used as the example below. Where the `BKECluster` CR lives:
> - **Management cluster**: modify it in the management cluster.
> - **Service cluster**: modify it in the management cluster.
>
> Replace `<cluster>` and `<ns>` with the actual cluster name and namespace.

### 4.1 Record the Current Configuration (for Rollback)

```bash
kubectl get bkecluster <cluster> -n <ns> -o yaml > bkecluster-backup-$(date +%Y%m%d%H%M).yaml

# Record the current baseline hash
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.etcdConfigHash}{"\n"}'

# Example:
#root@test:~/bkeadmTest# kubectl get bc -A
#NAMESPACE     NAME                  PHASE          STATE     CLUSTER STATUS   ENDPOINT          ENDPOINT PORT   VERSION        AGENT STATUS   AGE
#bke-cluster   bke-cluster-manager   ClusterReady   Healthy   Ready            192.168.122.190   6443            v1.34.3-of.1   1/1            5m25s
#root@test:~/bkeadmTest# kubectl get bkecluster bke-cluster-manager -n bke-cluster -o jsonpath='{.status.etcdConfigHash}{"\n"}'
#a31b319c40d9c92292aa0e6e4f4addf2b50458333e853be5bca071ada2004440
```

### 4.2 Modify etcd Parameters

The method used to modify the bc does not affect how this feature is triggered: `kubectl edit`, `kubectl patch`, and `kubectl apply -f <new configuration YAML file>` are **equivalent** — they all change the `spec` of the `BKECluster`. As long as `spec.clusterConfig.cluster.etcd.extraArgs` changes and the conditions in [2. Prerequisites](#2-prerequisites) are met, the controller reconciles again and triggers this phase.

**Method 1: edit directly (recommended, easier to review manually)**

```bash
kubectl edit bkecluster <cluster> -n <ns>
```

Locate `spec.clusterConfig.cluster.etcd.extraArgs` and modify the target parameters:

```yaml
spec:
  clusterConfig:
    cluster:
      etcd:
        extraArgs:
          heartbeat-interval: "500"      # Example: heartbeat interval (ms)
          election-timeout: "2500"       # Example: election timeout (ms)
          quota-backend-bytes: "2147483648"   # Example: backend storage quota (bytes)
```

**Method 2: patch (suitable for automation)**

```bash
kubectl patch bkecluster <cluster> -n <ns> --type=merge -p '
{
  "spec": {
    "clusterConfig": {
      "cluster": {
        "etcd": {
          "extraArgs": {
            "heartbeat-interval": "500",
            "election-timeout": "2500"
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

> Note: `new-bc.yaml` is the `BKECluster` manifest with the modified `etcd.extraArgs`; it can be produced by editing the backup file created in [4.1 Record the Current Configuration (for Rollback)](#41-record-the-current-configuration-for-rollback).

> **Notes**
> - `extraArgs` uses **full-overwrite** semantics. A merge patch only merges keys with the same name; to delete a parameter you must explicitly set that key to `null` (delete the line with `edit`, or pass `null` with a JSON merge patch).
> - Do not modify `etcdVersion` or `kubernetesVersion` in the same `apply`; otherwise this phase is not executed and the version-upgrade phase handles the change instead.
> - Do not modify `dataDir` / `serverCertSANs` / `peerCertSANs`; the webhook rejects such changes directly.

### 4.3 Monitor Execution Progress

```bash
# 1) Cluster phase (the result should be EnsureEtcdConfigUpdate)
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.phase}{"\n"}'

# 2) Issued commands
kubectl get command -l bke.bocloud.com/cluster-command

# 3) Controller logs (reason keyword: EnsureEtcdConfigUpdate)
cat /var/log/openFuyao/bke-controller-manager/<bke-controller-manager-pod-name> | grep -i "EnsureEtcdConfigUpdate"

# 4) Node bkeagent logs
# Keywords: update etcd static pod config / re-generate etcd manifest / etcd config update success
cat /var/log/openFuyao/bkeagent.log | grep -i "update etcd static pod config"
```

### 4.4 Verification

```bash
# 1) The etcd manifest on the node has been updated
# Log in to the target etcd node as root and check the update time of each component YAML file
cd /etc/kubernetes/manifests/
ls -l
# Example output (the update time of etcd.yaml should be later than the record before the change)
# total 24
# drwxr-xr-x 2 root root 4096 Sep 14 18:27 ./
# drwxr-xr-x 4 root root 4096 Sep 14 18:27 ../
# -rw-r--r-- 1 root root 1941 Sep 14 18:55 etcd.yaml
# -rw-r--r-- 1 root root 3426 Sep 14 18:27 kube-apiserver.yaml
# -rw-r--r-- 1 root root 2519 Sep 14 18:27 kube-controller-manager.yaml
# -rw-r--r-- 1 root root 1590 Sep 14 18:27 kube-scheduler.yaml

# 2) The actual startup parameters of the etcd static Pod have taken effect
kubectl -n kube-system get pod etcd-<node> -o yaml | grep -A 40 'command:'

# 3) The etcd Pod is Running
kubectl -n kube-system get pod -l component=etcd -o wide

# 4) The cluster is healthy
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.clusterStatus}{"\n"}'

# 5) The baseline hash has been refreshed
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.etcdConfigHash}{"\n"}'
```

> **About item 5**: the hash returned here must be **different** from the one recorded in [4.1 Record the Current Configuration (for Rollback)](#41-record-the-current-configuration-for-rollback); if it stays identical to that record, this phase was not triggered or the configuration was not actually delivered.

**Pass criteria**: the baseline hash has changed and is stable (not triggering repeatedly); all etcd Pods are Running; the cluster `clusterStatus` is Ready.

---

## 5. Troubleshooting and Handling

| Symptom | Possible cause | Action |
| --- | --- | --- |
| `kubectl apply` fails with `spec.clusterConfig.cluster.etcd.dataDir: Forbidden` | A bootstrap-immutable field was modified | Revert the change to that field; such a change requires rebuilding the control plane and cannot be done through a configuration update |
| `kubectl apply` fails with `APIServer/ControllerManager/Scheduler must all be present` | The configuration of one of the three control-plane components was deleted or emptied | Restore the three component configurations and apply again |
| `apply` succeeds but no command appears for a long time and the hash does not change | (1) `status.etcdConfigHash` is empty (baseline not established); (2) the changed field is not part of the hash; (3) a version field was changed at the same time | Check in order: wait for one health-check round to establish the baseline; confirm only `extraArgs` was changed; confirm no version field was changed |
| The command fails after being issued and the logs show `EtcdConfigUpdateFailed` | (1) An illegal parameter prevents etcd from starting; (2) waiting for the Pod hash to change timed out | Check the node bkeagent logs and `crictl logs`; roll back as described in [6. Rollback](#6-rollback) |
| The etcd Pod is in CrashLoopBackOff | The `extraArgs` parameter name or value is wrong, or a critical built-in parameter was overridden | Roll back the configuration; verify the parameters and retry |
| Some nodes succeed while others fail | A specific node is unhealthy (insufficient disk space, port occupied, etc.) | Fix the node and wait for the automatic retry (the old baseline is kept, so retries continue) |
| `extraVolumes` was modified; the flow reports success but the volume is not mounted | This field participates in the hash (so it triggers the update flow) but is **not rendered into the manifest**; the flow writes the new hash back as usual, after which the hashes are aligned and no retry occurs | This field is not supported by this update method; do not rely on it. Revert the change and keep only the `extraArgs` change |

Troubleshooting commands:

```bash
# Command objects and failed nodes
kubectl get command -l bke.bocloud.com/cluster-command -o yaml

# Failure reason on the controller side
kubectl -n <capbke-ns> logs deploy/capbke-controller-manager | grep -i EtcdConfigUpdateFailed

# Node bkeagent logs
ssh <etcd-node> 'journalctl -u bkeagent -f'

# etcd container logs from the container runtime
ssh <etcd-node> 'crictl ps -a | grep etcd; crictl logs <container-id>'
```

### 5.1 Recovery After a Failed `EnsureEtcdConfigUpdate` Reconcile

> <span style="color:red;font-weight:bold;">Important: if an etcd configuration update fails, no recovery method is currently available. There is neither an automatic rollback nor a manual way to re-trigger reconciliation, so prevention is the only option.</span>

<span style="color:red;font-weight:bold;">If an etcd configuration update fails, no recovery method is currently available.</span>

Unlike the control plane, `EnsureEtcdConfigUpdate` currently <span style="color:red;font-weight:bold;">cannot be re-triggered manually</span> (for example, adding the retry annotation to the bc does not apply to this scenario). Once etcd fails to start because of an illegal parameter, the cluster becomes unwritable and the control plane breaks down, so <span style="color:red;font-weight:bold;">the cluster cannot heal itself from inside</span>. Prevention is therefore essential:

- Strictly verify parameter names and values manually as described in [3.2 Parameter Validity Cannot Be Guaranteed by the Program (Manual Verification Required)](#32-parameter-validity-cannot-be-guaranteed-by-the-program-manual-verification-required), and validate them on a test cluster before applying them to production.
- Back up the etcd data yourself before important changes.
- Reserve a sufficient rollback window and run the change during off-peak hours.

> <span style="color:red;font-weight:bold;">Note</span>
> - <span style="color:red;font-weight:bold;">Self-managed scenarios have no recovery method available either</span>: if the failure occurs in a scenario where <span style="color:red;font-weight:bold;">the management cluster manages its own bc after the listener switch</span>, the cluster cannot recover from inside; <span style="color:red;font-weight:bold;">the upper-layer cluster must be healthy</span> before anyone can intervene.

---

## 6. Rollback

This feature provides **no automatic rollback**. To roll back, **restore `extraArgs` to the original values and apply again** to trigger a new configuration update.

### 6.1 Rollback Example (Explained with Hash Values)

> The hash values below are abbreviated for readability; the actual values are longer hexadecimal digests. The key point for judging whether a rollback succeeded is whether the hash is consistent with the original baseline.

Assume that the **initial value** of the etcd baseline hash `status.etcdConfigHash` of a cluster is `2b5f9c1a` (corresponding to the `extraArgs` delivered successfully the first time: `--quota-backend-bytes=2147483648`).

| Stage | etcd.extraArgs value | Desired hash | Description |
| --- | --- | --- | --- |
| Initial baseline | `--quota-backend-bytes=2147483648` | `2b5f9c1a` | Written to `status.etcdConfigHash` after the first delivery |
| After the change | `--quota-backend-bytes=8589934592` | `c7e3a0d4` | The desired hash differs from the baseline, triggering the configuration update of this phase; after all etcd nodes are updated, `status.etcdConfigHash` is written back as `c7e3a0d4` |
| After the rollback | `--quota-backend-bytes=2147483648` (original value restored) | `2b5f9c1a` | The desired hash `2b5f9c1a` differs from the current baseline `c7e3a0d4`, so the update is triggered again; after it completes, `status.etcdConfigHash` is written back as `2b5f9c1a`, consistent with the initial baseline |

**Criteria for a completed rollback**: **after restoring `extraArgs` to the original value and applying again, and after the update finishes on all etcd nodes, `status.etcdConfigHash` should again equal the initial baseline `2b5f9c1a`.** In other words, the hash after the rollback should be consistent with the original baseline hash, which indicates that the etcd static Pod configuration has been restored to the state before the change.

```bash
# 1) Restore from the backup
kubectl apply -f bkecluster-backup-<timestamp>.yaml

# Or: use edit to restore the original parameters manually
# (for example, restore etcd.extraArgs to --quota-backend-bytes=2147483648)
kubectl edit bkecluster <cluster> -n <ns>

# 2) Check whether the rollback is complete: the hash should return to the initial baseline 2b5f9c1a
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.etcdConfigHash}{"\n"}'
kubectl -n kube-system get pod -l component=etcd -o wide
```

A rollback also regenerates the etcd manifest and waits for Ready; the verification method is the same as in [4.4 Verification](#44-verification).

---

## 7. Notes

1. <span style="color:red;font-weight:bold;">Only `extraArgs` takes effect</span>: fields such as `extraVolumes` participate in the hash (changing them triggers the update flow) but are not rendered into the manifest, which ends up as <span style="color:red;font-weight:bold;">"the configuration was modified, the flow reports success, but nothing actually took effect"</span>. Do not rely on such hot updates (see [3. Usage Constraints](#3-usage-constraints)).
2. **Parameter validity must be verified manually**: the program does not validate parameter semantics; wrong parameters directly prevent etcd from starting.
3. **No etcd backup**: the configuration-update path does not trigger a backup; back up etcd data yourself before important changes.
4. **No interruption guarantee**: the cluster is not writable while the etcd static Pod is being recreated; run the change during off-peak hours.
5. **Immutable fields must not be changed**: changes to `dataDir` / `serverCertSANs` / `peerCertSANs` are rejected by the webhook and require rebuilding the control plane.
6. **Do not change versions at the same time**: when `etcdVersion` or `kubernetesVersion` changes, this phase is not executed and the version-upgrade phase handles the change instead.
7. **External etcd is not supported**: the phase is not triggered when there is no BKENode with the etcd role.
8. **No automatic rollback**: rollback means applying the original configuration again.
9. **Only etcd nodes are affected**: control-plane nodes are not affected by this phase.
10. <span style="color:red;font-weight:bold;">No recovery method for failures</span>: an etcd configuration update failure has neither an automatic rollback nor an available manual retry method. When a management cluster manages its own `BKECluster` (bc), the cluster cannot heal itself from inside either, and <span style="color:red;font-weight:bold;">the upper-layer cluster must be healthy — prevention is essential</span> (see [5.1 Recovery After a Failed EnsureEtcdConfigUpdate Reconcile](#51-recovery-after-a-failed-ensureetcdconfigupdate-reconcile)).
