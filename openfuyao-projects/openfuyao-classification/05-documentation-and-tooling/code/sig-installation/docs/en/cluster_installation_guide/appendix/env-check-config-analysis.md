# env-check Configuration File (config.json) Analysis of Check Item Meanings and Impacts

env-check is distributed to each node via the `run` command to execute `run-local`, performing the following 8 checks on each node (`run.go:167`):

```
kernel, port, disk, clock, fileQuery, programCheck, route, iptables
```

---

## log_file

| Attribute | Value |
|------|-----|
| Current configuration | `"./envCheck.log"` |
| Purpose | Log file path for the env-check tool itself |

**Meaning**: All log output from the check process is written to this file, including the start, pass, warning, and failure information for each check item.

**Impact**: Only affects the env-check tool's own logging, unrelated to BKE initialization.

---

## output_format

| Attribute | Value |
|------|-----|
| Current configuration | `"text"` |
| Available values | `text`, `json` |
| Purpose | Output format of the check result report |

**Meaning**: Controls the output format of the check results. `text` is suitable for command-line viewing, `json` is suitable for programmatic parsing.

**Impact**: Only affects the report display format, unrelated to BKE initialization.

---

## paths (File Conflict Check — fileQuery)

### Check Mechanism

Implemented via `FileQuery.Execute()` in `pkg/query/query.go`: for each path, it expands environment variables (such as `$HOME`) and wildcards (such as `kube*`), then checks whether the file/directory exists. If it exists, it is marked as a "conflict".

Check result determination (`run.go:360-367`):
- `TotalExists == 0` → **pass** (all residual paths do not exist)
- `TotalExists > 0` → **fail** (residual files exist)

### 14 Paths in Current Configuration

| No. | Path | Meaning | Impact of Residual on Bootstrap Node Initialization | Impact of Residual on Cluster Creation | Whether Initialization Handles Automatically | Severity |
|---|------|------|------------------------|---------------------|-------------------|----------|
| 1 | `$HOME/.kube` | Kubernetes kubeconfig directory | bkeadm overwrites `~/.kube/config` after starting k3s. If k3s skips startup (residual k3s data causes `isKubernetesAvailable` to return true), the residual kubeconfig points to the old cluster, and subsequent kubectl/CRD deployment operations hit the wrong cluster | capbke overwrites `~/.kube/config` on Master nodes; Worker nodes do not generate `~/.kube/config`, residual is not overwritten, only affects manual kubectl | Bootstrap node: conditional overwrite; Master: overwrite; Worker: not touched | 🟡 Medium |
| 2 | `/etc/kubernetes` | Kubernetes core configuration directory (pki/, manifests/, *.conf) | bkeadm does not directly operate on this directory (managed inside the k3s container). If k3s skips startup, residual configuration may cause k3s in-container processes to read old configuration | capbke overwrites most files (pki certificates loaded from Secrets and overwritten, kubeconfig regenerated, manifests overwritten with `O_TRUNC`), but there may be residual files in the directory that are not overwritten | Partially overwritten, not fully cleaned | 🟡 Medium |
| 3 | `/usr/bin/kube*` | Wildcard matching kubectl, kubelet, kubeadm and other binaries | bkeadm copies and overwrites `/usr/bin/kubectl` from the k3s container. The bootstrap node does not install the kubelet binary | capbke first `rm -rf /usr/bin/kubelet` then downloads (`run.go:576-596`), kubectl download overwrites | Automatic overwrite | 🟢 Low |
| 4 | `/usr/local/bin/kube*` | Wildcard matching k8s binaries under `/usr/local/bin/` | BKE does not install any files in this path, residual will not be overwritten. If `/usr/local/bin` has higher priority in PATH, the old version may be executed | Same as left | Not touched | 🟢 Low |
| 5 | `/usr/local/bin/crictl` | crictl CLI tool (CRI management tool) | **BKE installs crictl to `/usr/bin/crictl` (containerd tar extraction), not to `/usr/local/bin/crictl`**. Residual in this path will not be overwritten by BKE. If there is a PATH priority issue, the old version of crictl may be executed | Same as left | Not touched (path does not match) | 🟡 Medium |
| 6 | `/etc/sysctl.d/k8s.conf` | Kubernetes kernel parameter configuration (ip_forward, bridge-nf-call-iptables, etc.) | bkeadm **does not write** this file on the bootstrap node (bkeadm's `SetSysctl` only operates on `/etc/sysctl.conf`). Residual kernel parameters remain in effect, and incorrect values may cause network anomalies | capbke **completely overwrites** this file with `O_TRUNC` on cluster nodes (`init.go:342`) | Bootstrap node: not touched; Cluster node: overwrite | 🟡 Medium |
| 7 | `/etc/systemd/system/kubelet.service` | kubelet systemd unit file | bkeadm does not install kubelet (runs inside the k3s container). If the residual file is loaded by systemd, it may cause the old kubelet process to start, conflicting with the kubelet inside the k3s container | capbke **completely overwrites** with `O_TRUNC` (`run.go:435`) | Bootstrap node: not touched; Cluster node: overwrite | 🟡 Medium |
| 8 | `/etc/systemd/system/kubelet.service.d` | kubelet systemd drop-in directory | Neither bkeadm nor capbke **cleans this directory**. Residual old drop-in `.conf` files will be merged and loaded by systemd, overriding the kubelet.service parameters written by capbke, causing kubelet to start with incorrect configuration | Same as left | **Not cleaned** | 🔴 High |
| 9 | `/var/lib/etcd` | Standard kubeadm etcd data directory | BKE uses `/var/lib/openFuyao/etcd` (`defaults.go:49`), **does not use** `/var/lib/etcd`. Residual data will not be read, but indicates the machine previously had standard k8s installed | Same as left | Not touched (BKE does not use this path) | 🟢 Low |
| 10 | `/var/lib/kubelet` | kubelet working root directory (config.yaml, plugin data, volume mount points) | bkeadm does not directly operate on it (kubelet inside the k3s container uses the in-container directory) | capbke overwrites `config.yaml`, but old mount points, Pod data, and the `pki/` subdirectory in the directory **are not cleaned**. Residual old mount points may cause kubelet to attempt mounting non-existent volumes at startup | Partially overwritten (only config.yaml) | 🔴 High |
| 11 | `/run/containerd/containerd.sock` | containerd CRI Unix socket | bkeadm installs containerd and starts it, containerd creates this socket. If the old containerd is still running, the socket already exists, and the new containerd may fail to start | Same as left | Created/overwritten when containerd starts | 🟡 Medium |
| 12 | `/usr/lib/systemd/system/kubelet.service.d` | kubelet systemd drop-in directory installed by package manager | Neither bkeadm nor capbke **cleans this directory**. Residual old drop-ins will override kubelet.service parameters | Same as left | **Not cleaned** | 🟡 Medium |
| 13 | `/var/run/containerd/containerd.sock` | Same as `/run/containerd/containerd.sock` (`/var/run` is usually a symlink to `/run`) | Same as item 11 | Same as item 11 | Same as above | 🟡 Medium |
| 14 | `/var/run/docker.sock` | Docker daemon Unix socket | If the bootstrap node uses containerd mode (default), residual docker.sock indicates the old docker is still running. bkeadm does not check this socket | capbke pre-check detects the container runtime type (`check.go:415-430`), if docker.sock exists but containerd is configured, **pre-check fails** | Not touched | 🔴 High |

## clean_force

| Attribute | Value |
|------|-----|
| Current configuration | `false` |
| Purpose | Controls whether to skip user confirmation and directly delete residual files in `fileClean` mode |

**Meaning**:
- `false`: Interactively asks the user `[y/n]` before deleting each residual file.
- `true`: Skips confirmation and directly deletes all residual files.

**Impact**: Only affects the behavior in `fileClean` mode. The current configuration is `false`, deletion operations require the user to confirm one by one, preventing accidental deletion.

---

## program_list (Program Check — programCheck)

### Check Mechanism

Implemented via `ApplicationChecker.Execute()` in `pkg/program/check.go`: for each program, it uses `exec.LookPath(name)` to detect whether it can be found in PATH, then compares with the `should_exist` expected value.

Check result determination (`run.go:396-403`):
- `TotalFailed == 0` → **pass** (all programs match the expected state)
- `TotalFailed > 0` → **fail** (some programs do not match the expected state)

### 4 Programs in Current Configuration

| No. | Program name | should_exist | Meaning | Impact of Residual on Initialization | Initialization Handling Behavior | Severity |
|---|--------|-------------|------|------------------|--------------|----------|
| 1 | `docker` | `false` (should not exist) | Docker container runtime | In containerd mode, capbke pre-check detects runtime type mismatch (`check.go:415-430`), **node initialization fails**. bkeadm does not automatically uninstall docker | **Does not automatically uninstall**. bkeadm uses the existing docker if it detects docker is installed; capbke fails the pre-check if it detects a mismatch | 🔴 High |
| 2 | `kubelet` | `false` (should not exist) | Kubernetes node agent (system service) | If the old kubelet is running and not managed by systemd, capbke's `systemctl stop kubelet` fails (only Warning), the old process continues to occupy ports 10250/10248, and the new kubelet fails to start (`run.go:172-176`). The old kubelet on the bootstrap node may interfere with iptables and `/var/lib/kubelet` | capbke first `systemctl stop` then `rm -rf /usr/bin/kubelet` then downloads (`run.go:576-596`), but **does not kill running processes not managed by systemd** | 🔴 High |
| 3 | `containerd` | `false` (should not exist) | containerd container runtime | The binary is overwritten by tar extraction, but residual configuration (`/etc/containerd/config.toml`) and data (`/var/lib/containerd`) are not fully cleaned | bkeadm extracts the containerd tar to **overwrite** the binary (`containerd.go:172`) | 🟡 Medium |
| 4 | `tar` | `true` (should exist) | tar archive tool | tar is a basic dependency during the BKE installation process: bkeadm extracts the containerd tar (`containerd.go:172`), extracts CNI plugins (`containerd.go:387`), and extracts images/source data (`repository.go`) all depend on tar. If tar does not exist, **initialization fails directly** | Not installed (assumed to be provided by the system) | 🔴 High |

### Program Check Summary

| Program | Check direction | Severity | Description |
|------|---------|----------|------|
| docker | Should not exist | 🔴 High | Residual in containerd mode causes capbke pre-check failure. |
| kubelet | Should not exist | 🔴 High | Residual running process causes new kubelet port conflict and startup failure. |
| containerd | Should not exist | 🟡 Medium | Binary overwritten, configuration/data residual. |
| tar | Should exist | 🔴 High | Missing causes all extraction operations to fail. |

---

## hosts (Host List)

### Current Configuration

| IP | Role | SSH port |
|----|------|---------|
| 192.168.2.135 | bootstrap | 22 |
| 192.168.2.221 | master | 22 |
| 192.168.2.229 | worker | 22 |

**Meaning**: Defines the list of target hosts that env-check needs to check. The `role` of each host determines which port checks are performed (the port list corresponding to the role in `port_check.ports`).

**Impact**:
- In `dispatch` mode, env-check distributes its own binary to each host and executes checks remotely via SSH.
- `role` is used for role matching in port checks (`port/check.go:130-155`): the `bootstrap` role performs "bootstrap node port checks", `master` performs "Master node port checks", `worker` performs "Worker node port checks".
- The `clock_threshold` clock check also requires the hosts list (obtains the time of each host via SSH and compares with the local time).

---

## clock_threshold (Clock Synchronization Check — clock)

| Attribute | Value |
|------|-----|
| Current configuration | `10` (seconds) |
| Purpose | Maximum allowed time difference between hosts |

### Check Mechanism

Implemented via `CheckerClock.Execute()` in `pkg/clock/clock.go`: connects to each host via SSH to obtain the remote time, compares it with the local time, and calculates the time difference. If the time difference between any two hosts exceeds `clock_threshold` seconds, the check fails.

### Impact

| Scenario | Impact | Severity |
|------|------|----------|
| Time difference between hosts > 10 seconds | The etcd cluster relies on consistent time for leader election and log replication. Excessive time difference will cause etcd election failure or data inconsistency. Kubernetes certificates also have time validity checks, and time deviation may cause certificate validation anomalies | 🔴 High |
| bkeadm initialization | bkeadm's `setTimezone()` sets the timezone and NTP server, but only sets the bootstrap node. Time synchronization of cluster nodes is the responsibility of capbke env init | 🟡 Medium |

**Note**: The `runClockCheck()` in the current `run.go:327-333` only returns the local time and marks it as pass. The actual multi-host clock comparison is performed by `CheckerClock.Execute()` in `dispatch` mode.

---

## kernel_check (Kernel Version Check — kernel)

| Attribute | Value |
|------|-----|
| Current configuration | `{"min_version": "4.19", "operator": ">="}` |
| Purpose | Checks whether the kernel version meets the minimum requirement |

### Check Mechanism

Implemented via `Checker.Execute()` in `pkg/kernel/check.go`: uses `gopsutil/host.Info()` to obtain the kernel version, and compares it with `min_version` according to `operator`.

Special handling (`check.go:97-115`): special judgment for the `3.10.0-xxx` kernel of CentOS 7 (with old-style build number) — even if the version number >= 4.19, if the `3.10.0` prefix is detected with an old-style build number, it is also judged as not meeting the `>=` condition.

### Impact

| Scenario | Impact | Severity |
|------|------|----------|
| Kernel version < 4.19 | BKE depends on relatively new kernel features such as cgroup v2, eBPF, and overlayfs. Low-version kernels may cause container runtime, CNI plugin, and kubelet function anomalies. The `native` snapshotter of containerd and the eBPF datapath of Calico both require a newer kernel | 🔴 High |

---

## port_check (Port Occupancy Check — port)

### Check Mechanism

Implemented via `Checker.Execute()` in `pkg/port/check.go`: selects the corresponding port list based on the node role, and performs a TCP dial to `127.0.0.1:<port>` for each port. If the connection succeeds, the port is occupied. It can also identify the occupying process via gopsutil.

Check result determination (`run.go:274-280`):
- `Used == 0` → **pass** (all ports are free)
- `Used > 0` → **fail** (occupied ports exist)

### Current Configuration

#### Bootstrap Ports (5)

| Port | Service | Source code | Consequence of occupancy | Severity |
|------|------|---------|----------|----------|
| 36443 | Local k3s API Server | `bkeadm/constants.go:17` `DefaultKubernetesPort` | 🔴 bkeadm `validatePorts()` hard failure, initialization aborted | 🔴 High |
| 40080 | Yum repository | `capbke/defaults.go:63` `DefaultYumRepoPort` | 🔴 bkeadm `validatePorts()` hard failure | 🔴 High |
| 40443 | Image repository | `capbke/defaults.go:60` `DefaultImageRepoPort` | 🔴 bkeadm `validatePorts()` hard failure | 🔴 High |
| 38080 | Chart repository | `bkeadm/constants.go:38` `DefaultChartRegistryPort` | 🔴 bkeadm `validatePorts()` hard failure | 🔴 High |
| 30010 | k3s NodePort mapping (ingress-nginx), i.e. the Web access port of the openFuyao management plane (`https://<bootstrap-node-IP>:30010`) | `bkeadm/k3s.go:322` `-p 30010:30010` | 🟡 k3s container creation failure (`validatePorts()` does not check this port, the error message may be unclear) | 🟡 Medium |

Among them, ports 36443, 40080, 40443, and 38080 are configurable. If specific ports are used in `bke init`, they need to be replaced with the corresponding ports to check.

#### Master Ports (13)

| Port | Service | Source code | Consequence of occupancy | Severity |
|------|------|---------|----------|----------|
| 6443 | kube-apiserver | `capbke/defaults.go:38` `DefaultAPIBindPort` | 🔴 apiserver cannot bind, startup fails | 🔴 High |
| 2379 | etcd client | `capbke/consts_sca.go:90` `EtcdListenClientPort` | 🔴 etcd cannot bind, startup fails | 🔴 High |
| 2380 | etcd peer | `capbke/consts_sca.go:106` `EtcdListenPeerPort` | 🔴 etcd cannot form a cluster | 🔴 High |
| 2381 | etcd metrics | `capbke/consts_sca.go:92` `EtcdMetricsPort` | 🟡 etcd metrics cannot bind (non-fatal) | 🟡 Medium |
| 10248 | kubelet healthz | `capbke/consts_sca.go:295` `KubeletHealthzPort` | 🔴 kubelet cannot start | 🔴 High |
| 10249 | kube-proxy metrics | Kubernetes default port | 🟡 kube-proxy metrics cannot bind | 🟡 Medium |
| 10250 | kubelet secure port | `capbke/consts_sca.go:350` `KubeletPort` | 🔴 kubelet cannot start | 🔴 High |
| 10256 | kube-proxy healthz | Kubernetes default port | 🟡 kube-proxy healthz cannot bind | 🟡 Medium |
| 10257 | kube-controller-manager | `capbke/consts_sca.go:356` `KubeControllerManagerPort` | 🔴 KCM cannot start | 🔴 High |
| 10259 | kube-scheduler | `capbke/consts_sca.go:353` `KubeSchedulerPort` | 🔴 scheduler cannot start | 🔴 High |
| 3377 | bkeagent-launcher /readyz | `capbke/cmd/bkeagent-launcher/main.go:279` | 🟡 launcher readiness probe fails | 🟡 Medium |
| 58080 | bkeagent health | `bkeadm/constants.go:49` `DefaultAgentHealthPort` | 🟡 agent health check fails | 🟡 Medium |
| 1338 | containerd metrics | `bkeadm/.../containerd_default.yaml:27` `metricsAddress` | 🔴 containerd startup fails, metrics port binding failure causes process exit | 🔴 High |

> Port 1338 is bound to `127.0.0.1` (loopback address), configured by the ContainerdConfig CR (`metricsAddress: "127.0.0.1:1338"`), and capbke reads this CR and renders it into the containerd `config.toml` on cluster nodes. The containerd on the bootstrap node uses bkeadm's own template (`[metrics] address = ''`), does not enable metrics, so the bootstrap port list does not include 1338.

#### Worker Ports (7)

| Port | Service | Source code | Consequence of occupancy | Severity |
|------|------|---------|----------|----------|
| 3377 | bkeagent-launcher /readyz | `capbke/cmd/bkeagent-launcher/main.go:279` | 🟡 launcher readiness probe fails | 🟡 Medium |
| 58080 | bkeagent health | `bkeadm/constants.go:49` `DefaultAgentHealthPort` | 🟡 agent health check fails | 🟡 Medium |
| 10248 | kubelet healthz | `capbke/consts_sca.go:295` `KubeletHealthzPort` | 🔴 kubelet cannot start | 🔴 High |
| 10249 | kube-proxy metrics | Kubernetes default port | 🟡 kube-proxy metrics cannot bind | 🟡 Medium |
| 10250 | kubelet secure port | `capbke/consts_sca.go:350` `KubeletPort` | 🔴 kubelet cannot start | 🔴 High |
| 10256 | kube-proxy healthz | Kubernetes default port | 🟡 kube-proxy healthz cannot bind | 🟡 Medium |
| 1338 | containerd metrics | `bkeadm/.../containerd_default.yaml:27` `metricsAddress` | 🔴 containerd startup fails, metrics port binding failure causes process exit | 🔴 High |

### Port Check Summary

| Role | Number of configured ports | Communication matrix coverage |
|------|-----------|----------------|
| Bootstrap | 5 | Covers communication matrix bootstrap TCP ports (36443/40080/40443/38080/30010)|
| Master | 13 | Covers communication matrix master core ports + containerd metrics (6443/2379/2380/2381/10248/10249/10250/10256/10257/10259/3377/58080/1338)|
| Worker | 7 | Covers communication matrix worker core ports + containerd metrics (3377/58080/10248/10249/10250/10256/1338)|

---

## disk_check (Disk Space Check — disk)

| Attribute | Value |
|------|-----|
| Current configuration | `{"check_items": [{"path": "/", "min_free_gb": 50}]}` |
| Purpose | Checks whether the available disk space of the specified path meets the minimum requirement |

### Check Mechanism

Implemented via `Checker.Execute()` in `pkg/disk/check.go`: for each `check_items` entry, uses `gopsutil/disk.Usage()` to obtain the disk usage of the specified path, and compares the `Free` space with `MinFreeGB * 1GB`.

If the path does not exist, it automatically searches upward for the mount point of the parent directory. If the `check_items` entry has a `roles` field configured, only nodes matching the role are checked (the current configuration does not set `roles`, so all nodes are checked).

Check result determination (`run.go:315-322`):
- `InsufficientPath == 0` → **pass** (all paths have sufficient space)
- `InsufficientPath > 0` → **fail** (paths with insufficient space exist)

### Current Configuration Analysis

| Path | Minimum free space | Meaning |
|------|------------|------|
| `/` | 50 GB | The root partition requires at least 50 GB of free space |

### Impact

| Scenario | Impact | Severity |
|------|------|----------|
| Root partition free < 50 GB | bkeadm `validateDiskSpace()` checks the disk space of the working directory: if there is existing image data, it requires ≥3 GB, otherwise requires ≥20 GB (`initialize.go:217-231`). env-check's 50 GB requirement is stricter, reserving sufficient space for BKE installation (image repository, source repository, k3s data, containerd images, etc.) | 🔴 High |

---

## dispatch (Distribution Configuration)

| Attribute | Value |
|------|-----|
| timeout | `600` (seconds, i.e. 10 minutes) |
| poll_interval | `15` (seconds) |
| work_dir | `/tmp/envcheck` |
| concurrent_limit | `10` |

### Meaning of Each Field

| Field | Meaning |
|------|------|
| `timeout` | Timeout for distribution execution. env-check distributes its own binary to each remote node via SCP, executes checks on each node, and collects results. The 600-second timeout covers the check time of all nodes. |
| `poll_interval` | Interval for polling remote node check results. Checks whether each node has completed every 15 seconds. |
| `work_dir` | Working directory on the remote node. The env-check binary and check result file (`result.json`) are stored in this directory. |
| `concurrent_limit` | Upper limit on the number of nodes checked concurrently. At most 10 nodes are checked simultaneously. |

### Impact

| Scenario | Impact |
|------|------|
| `timeout` too short | Node check not completed but timed out, result marked as failed. Actual checks (8 items) usually complete within 1-2 minutes, 600 seconds is usually sufficient. |
| `work_dir` not writable | The `/tmp/envcheck` directory on the remote node cannot be created (permission issue or `/tmp` not writable), check results cannot be saved. |
| `concurrent_limit` too small | When there are many nodes, the check takes a long time. Currently 3 nodes, 10 concurrent is completely sufficient. |

**Note**: The dispatch configuration only affects the distribution behavior of the env-check tool, and is unrelated to BKE initialization.

---

## route and iptables Checks

These two checks are executed by default in `run-local` (`run.go:167`), but **are not explicitly configured in config.json** — they use fixed logic and do not depend on configuration parameters.

### route (Default Route Check)

**Check Mechanism** (`pkg/route/route.go`): executes the `route -n` command to parse the routing table and checks whether a default route exists. If there is no default route, the node's network configuration is abnormal, which may cause communication failure between BKE components.

**Impact**: Nodes without a default route cannot access the external network (pulling images, downloading binaries, etc.), and BKE initialization fails.

### iptables (FORWARD Chain Policy Check)

**Check Mechanism** (`pkg/iptables/iptables.go`): executes the `iptables -L FORWARD -n` command to check whether the default policy of the FORWARD chain is `ACCEPT`.

**Impact**: If the FORWARD chain policy is `DROP`, cross-node communication between Kubernetes Pods will be dropped by iptables, causing network unreachability. Calico CNI relies on the FORWARD chain to forward Pod traffic. bkeadm's `prepareEnvironment()` installs iptables and switches to legacy mode, but does not set the FORWARD policy — this is the responsibility of capbke env init (sets iptables rules in `init.go`).
