# Environment Pre-check

This chapter introduces the environment pre-check tools available before openFuyao installation and deployment:

- **bke built-in unified preflight check (preflight)**: Built into the bke installation tool since v26.09, it uses the same input as the installation command to perform read-only checks and output a repairable report. It is the recommended entry point for pre-installation checks.
- **envCheck**: An independent community environment check tool that provides generalized check capabilities such as residual file query/cleanup and remote batch distribution checks.

According to the version plan, the bke built-in unified preflight check will gradually replace envCheck in subsequent versions: from v26.09, the pre-installation check entry switches to `bke preflight`. bke preflight is currently an alpha-stage feature and may be unstable. The user documentation for both tools is in the corresponding sections below.

## bke Built-in Unified Preflight Check

### Background Information

`bke init` and `bke cluster create` depend on multiple types of conditions from business input, the management cluster, the local machine, and target nodes before the actual installation. Existing errors are typically exposed only during the installation process, and network, authentication, permission, resource, and configuration issues are easily mixed together, increasing the cost of diagnosis and recovery.

To address this, bke includes a built-in unified preflight check since v26.09. Before executing installation commands, it uses **exactly the same input** as the installation command to generate a check plan, collect facts in read-only mode, calculate results based on versioned release rules, and output a repairable report:

```bash
# Before executing bke init
bke preflight init [same business parameters as bke init]

# Before executing bke cluster create
bke preflight cluster-create \
  --file ./bkecluster.yaml \
  --nodes ./bkenodes.yaml
```

The preflight check has the following characteristics:

- **Read-only**: Does not modify hosts, nodes, or Kubernetes resources; does not create directories or test files; does not install, repair, or clean up components. On the Kubernetes side, it only uses Discovery, GET, and LIST; on the SSH side, it only executes whitelisted read-only commands.
- **Shares input with installation commands**: `preflight init` and `bke init` reuse the same parameter definitions, default values, and validation logic; `preflight cluster-create` and `bke cluster create` use the same cluster/nodes files and management cluster kubeconfig, checking exactly the values that this installation will actually use.
- **Traceable rules**: Fixed rules are embedded in the release Bundle within the bke binary. Each rule has a clear source of evidence (release configuration, Provider validation, manifest, or installation documentation) and is protected by digest verification.
- **Repairable results**: Each FAIL/WARN explains the check target, expected value, actual value, reason, and repair suggestion; input errors, unmet environment conditions, and framework failures are distinguished by different conclusions and exit codes.
- **No state caching**: Input and environment may change between two commands. PASS only indicates that the applicable checks in the report were satisfied at the time of checking, and does not guarantee that the subsequent installation will succeed.

### Prerequisites

- Use a bke binary tool that includes the `preflight` subcommand (v26.09 and above). The runtime environment is Linux amd64 or Linux arm64.
- `preflight cluster-create` requires:
  - Successfully deployed a bootstrap node by referring to [Online Bootstrap Cluster Installation](../bootstrap_cluster_installation/online_bootstrap_cluster_installation.md) or [Offline Bootstrap Cluster Installation](../bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md), or successfully deployed a management cluster by referring to [Management Cluster Deployment Guide](../management_cluster_installation/management_cluster_deployment_guide.md).
  - Prepared the cluster configuration file (`--file`) and node configuration file (`--nodes`) required for creating a cluster. Both have the same format as used by `bke cluster create`. For writing examples, see [Service Cluster Deployment via Backend (Command Line)](../service_cluster_installation/service_cluster_deployment_guide_(linux_backend).md).
  - Target nodes allow SSH login (default port 22, configurable per node).
- To use the `--preflight-config` external policy, prepare a complete `preflight.yaml` matching the current bke version (see [Configure External Policy](#configure-external-policy)).

### Usage Limitations

- Only covers the `init` and `cluster-create` installation phases; does not cover build, Web, upgrade, or scaling.
- Only performs checks with clear BKE/Provider/installation requirement sources. Some generalized checks from the legacy envCheck (residual files, clock sync, default route, iptables, SELinux, Swap, etc.) are not within the scope of the built-in checks. For these capabilities, see [envCheck](#envcheck).
- Does not automatically execute installation commands; after handling FAILs and confirming WARNs, the user must explicitly execute the actual business command.
- Source connectivity in both the init and cluster-create phases only verifies reachability, not authentication or artifact download success.
- The current version's SSH check does not validate host keys; it can only prove session establishment and should be executed in a trusted network.
- Does not provide `--request`, Bundle override, or Provider defaults file override parameters.

### Usage Tutorial

#### Command Overview

```bash
# View help
bke preflight -h
bke preflight init -h
bke preflight cluster-create -h
```

`preflight` is an independent, read-only top-level subcommand that contains two phase subcommands:

| Command | Description |
| ---- | ---- |
| `bke preflight init` | Checks the bootstrap node before `bke init` (identity, platform, resources, tools, working directory, ports, sources, offline artifacts, custom repositories, and CA) |
| `bke preflight cluster-create` | Checks input, target nodes (SSH, resources, platform, role ports), and sources before `bke cluster create` |

#### Runtime Parameter Description

Common runtime parameters for the `preflight` command:

| Parameter | Description |
| ---- | ---- |
| `--output` | Output format, supports `text` and `json`, defaults to `text`, optional. |
| `--output-file` | Writes the complete report to the specified file, optional; when not specified, text mode automatically generates a detailed report file (`bke-preflight-<phase>-*.txt`) in the system temporary directory and displays the path in the terminal. |
| `--preflight-config` | Path to a complete external `preflight.yaml` policy file, optional; when not specified, uses the default policy embedded in the binary. |
| `--timeout` | Total timeout for the complete preflight check, optional; defaults to **10min**, covering input parsing, management cluster reading, collection, calculation, and other phases; must be greater than 0. |
| `--fact-timeout` | Timeout for a single fact probe, optional; defaults to `30s`; must be greater than 0. |
| `--max-concurrency` | Maximum concurrency for fact probes, optional; defaults to `0` (bounded adaptive), valid range is 0~64. |

The business parameters for `preflight init` are identical to those of `bke init` (common parameters are listed below; for the complete list, see `bke preflight init -h`):

| Parameter | Description |
| ---- | ---- |
| `-f, --file` | init input file (bkecluster.yaml), optional. |
| `--hostIP` | Bootstrap node's own IP, optional; when not provided, automatically detects the local outbound IP. |
| `--onlineImage` | Online installation image address, optional. |
| `--imageFilePath` | Offline installation package path, optional. |
| `--installConsole` | Whether to install Console, optional; defaults to `true`; Console port is only checked when set to `true`. |
| `--enableNTP` / `--ntpServer` | NTP switch and NTP server, optional; enabled by default; local UDP 123 port is only checked when the local machine provides NTP (`local`). |
| `--otherRepo` / `--otherSource` / `--otherChart` | Custom image, package, and Chart repository addresses, optional. |
| `--imageRepoCAFile` | Image repository CA certificate file path, optional; when configured, checks that the file exists, is readable, and can be parsed as a PEM X.509 certificate. |

Business parameters for `preflight cluster-create`:

| Parameter | Description |
| ---- | ---- |
| `-f, --file` | BKECluster configuration file, required. |
| `-n, --nodes` | BKENodeList node configuration file, required. The IP, SSH port, role, credentials, and other information in the node file use the same format as `bke cluster create`. |

#### Check Item Description

The built-in checks are jointly determined by the release policy (`configs/preflight/preflight.yaml`) and the embedded Bundle, and are dynamically instantiated based on the actual plan of this input: inapplicable checks do not produce results; only applicable checks that are depended on, timed out, or cancellation-blocked will show as SKIP. Multiple downstream checks triggered by the same root cause will show as SKIP and reference the root cause, preventing one issue from spreading into multiple FAILs.

> ![](../figures/icon-note.gif)**Note:**
> In **Table 1** and **Table 2**, "Blocking" refers to the **conclusion level** when a check is not satisfied, not the command behavior:
> - **Yes**: When the check fails or facts cannot be collected, the conclusion is FAIL (exit code 3). This condition is a hard prerequisite for installation and must be fixed before retrying; directly executing the installation command is expected to fail.
> - **No**: The conclusion is WARN (exit code 2). This condition is a risk warning and can be continued after manual confirmation.
> - preflight and installation commands are independent of each other; FAIL does not prevent the execution of `bke init` / `bke cluster create`; except for dependency-blocked checks, all applicable checks will complete. Downstream checks showing SKIP are part of the dependency denoising mechanism (triggered by either root dependency FAIL or WARN) and are unrelated to the meaning of this column.

#### init Phase Preflight Check

- Execute the following command to check the bootstrap node using the same business parameters as `bke init`:

  ```bash
  sudo bke preflight init \
    --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest \
    --installConsole=false
  ```

- When using an offline installation package:

  ```bash
  sudo bke preflight init \
    --imageFilePath /root/bke-offline.tar.gz \
    --installConsole=false
  ```

- When using custom repositories and CA certificates, append the actual addresses to the command:

  ```bash
  sudo bke preflight init \
    --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest \
    --otherRepo https://registry.example.com:40443/openfuyao \
    --otherSource http://repo.example.com:40080 \
    --otherChart https://chart.example.com:443/chart \
    --imageRepoCAFile /etc/openFuyao/ca.crt
  ```

**Table 1** Main check items for the init phase (all targets are the bootstrap node):

| Category | Check Content | Default Requirement | Blocking |
| ---- | ---- | ---- | ---- |
| Execution identity | Local execution user | UID is 0 (root) | Yes |
| Platform | OS, distribution, kernel, architecture | Linux; openEuler 20.03/22.03/24.03 or Ubuntu 22.04 or UOS 20 1070e; kernel matches distribution requirements (e.g., openEuler 22.03 requires 5.10 kernel); amd64 or arm64 | Yes |
| Resources | Logical CPU, available memory, system disk space | No less than 2 cores, 4 GB, 100 GB | Yes |
| Tools | `tar`, `systemctl` | Present in PATH and executable | Yes |
| Working directory | Workspace (default `/bke`, can be overridden via the `BKE_WORKSPACE` environment variable) | Already exists or parent path allows creation; the file system is not a read-only mount | Yes |
| Ports | Kubernetes 36443, image repository 40443, Chart repository 38080, package repository 40080; Console 30010 (when Console is enabled); local NTP 123/UDP (when the local machine provides NTP) | Actual ports are free (when ports are explicitly passed, they are checked by the passed values) | Yes |
| Sources | The installation sources actually used this time (online image, custom repository, etc.) | Can establish a TCP connection to the masked host:port | Yes |
| Offline artifacts | Offline installation package exists → readable → format | Exists; readable; is tar, tar.gz, or tgz | Yes |
| Custom repositories | Custom repository address, CA certificate | URL/domain/IP/port can be resolved; CA file exists, is readable, and can be parsed as a PEM X.509 certificate | Yes |

- After the check passes and issues in the report are handled, execute the following command to deploy the bootstrap node:

  ```bash
  sudo bke init \
    --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest \
    --installConsole=false
  ```

#### cluster-create Phase Preflight Check

- Execute the following command using the same cluster, node files, and management cluster kubeconfig as `bke cluster create`:

  ```bash
  sudo bke preflight cluster-create \
    --file /root/openfuyao/bkecluster.yaml \
    --nodes /root/openfuyao/bkenodes.yaml
  ```

- After the check passes and issues in the report are handled, execute the following command to deploy the cluster:

  ```bash
  sudo bke cluster create \
    --file /root/openfuyao/bkecluster.yaml \
    --nodes /root/openfuyao/bkenodes.yaml
  ```

- When there are many nodes, you can adjust the concurrency to speed up the check:

  ```bash
  sudo bke preflight cluster-create \
    --file /root/openfuyao/bkecluster.yaml \
    --nodes /root/openfuyao/bkenodes.yaml \
    --max-concurrency 16  # 16 is an example value, indicating the maximum concurrency; valid range 0~64, default 0 (adaptive)
  ```

**Table 2** Main check items for the cluster-create phase:

| Category | Check Content | Default Requirement | Blocking |
| ---- | ---- | ---- | ---- |
| Input uniqueness | Node IP, hostname | Non-empty IPs are not duplicated; non-empty hostnames are not duplicated | Yes |
| Topology | Number of masters, number of etcd nodes, number of worker capacities | Masters no less than 1 and odd; etcd no less than 1; workers (including master+worker) no less than 1 | Yes |
| Management cluster | API reachable → Discovery readable → CRD registered; `kube-system/localkubeconfig` Secret can be GET → data non-empty → kubeconfig parseable → server endpoint syntactically valid | BKECluster/BKENode CRDs registered; Secret chain complete | Yes |
| Local input | Schema and business validation of cluster file and node file | Passes (errors are located to specific field paths or file line numbers) | Yes |
| Local path | Cluster configuration output path | Already exists or parent path allows creation; accessible; not a read-only mount | Yes |
| Network | Pod subnet, Service subnet | CIDR syntax valid (only syntax is validated; subnet overlap is not inferred) | Yes |
| HA and addons | `kubernetesVersion` version reference; addons (extension components declared as yaml/chart in BKECluster `clusterConfig.addons`) configuration validity; source configuration for chart-type addons; bootstrap node and HA master co-deployment warning (only instantiated when bootstrap node identity can be proven and the number of masters is greater than 1) | Version is a valid semver and no lower than v1.21.0 (no upper limit); addon name+version combination is unique, type only allows yaml/chart (defaults to yaml when empty); chart-type addons must have chartRepo configured (only configuration completeness is verified; repository reachability is handled by the "Sources" row); co-deployment warning alerts on capacity and fault domain risks when the bootstrap node also serves as an HA master | Version reference and addons: Yes; HA co-deployment warning: No (WARN only, requires manual confirmation, not instantiated by default in the current version) |
| SSH (per node) | transport (pure TCP) → authentication → permissions → systemd | TCP connectable; credentials can establish a session; root or passwordless sudo available; systemd available | Yes |
| Node resources | Logical CPU, available memory, system disk space | No less than 8 cores, 16 GB, 100 GB per node | Yes |
| Node platform | OS, distribution, kernel | Consistent with the init phase platform matrix | Yes |
| Role ports | master: apiserver 6443, controller-manager 10257, scheduler 10259, local HA 36443 (when multiple masters); master/etcd: etcd client 2379, peer 2380, metrics 2381; master/worker: kubelet 10250, kubelet health 10248; all nodes: BKE Agent 58080 | Actual ports are free (explicitly configured ports in the business file take priority) | Yes |
| Sources | Each node to the actual installation source | Performs a curl HEAD probe via that node's SSH session, using the node's own DNS and `/etc/hosts` for resolution; when the repository has both domain and IP configured, connects to the IP and retains the domain as Host/SNI | Yes |

> ![](../figures/icon-note.gif)**Note:**
> - Remote source connectivity is probed via curl HEAD from the target node, which requires curl to be pre-installed on the target node; the probe does not validate certificates (self-signed certificates do not affect the result), and HTTP 401/403/404 and other error statuses are still considered reachable; private domains must be resolvable on every node.
> - When reading the management cluster Provider default configuration fails, the check continues using static default values while reporting the `CLUSTER-MGMT-PROVIDER-DEFAULTS` issue.
> - The meaning of "Blocking" in **Table 2** is consistent with the init phase check item description: FAIL (exit code 3) is a hard prerequisite for installation and must be fixed before retrying; WARN (exit code 2) is a risk warning and can be continued after manual confirmation. This column does not affect the execution of subsequent check items.

#### Configure External Policy

The default policy is embedded in the bke binary along with the matching Bundle. When you need to adjust thresholds, timeouts, or display information, you can copy the complete default policy, modify it, and explicitly load it via `--preflight-config` without rebuilding bke:

```bash
# Copy the complete default file of the same revision as the current bke from the bkeadm repository
# (First record the version and gitCommitID via bke version, then fetch configs/preflight/preflight.yaml of the corresponding revision)
cp preflight.yaml /etc/openFuyao/preflight.yaml

sudo bke preflight --preflight-config /etc/openFuyao/preflight.yaml \
  init --hostIP 192.0.2.10  # 192.0.2.10 is an example IP, please replace with the actual bootstrap node IP

sudo bke --kubeconfig /path/to/kubeconfig \
  preflight --preflight-config /etc/openFuyao/preflight.yaml \
  cluster-create -f bkecluster.yaml -n bkenodes.yaml
```

**Default Policy File Content**

Taking v26.09 as an example, the complete content of the default policy file `configs/preflight/preflight.yaml` is as follows (the values of `metadata.release` and other fields are subject to the actual release file matching the current bke revision):

```yaml
apiVersion: preflight.openfuyao.io/v1
kind: PreflightConfig
metadata:
  release: v26.09
  description: Human-readable openFuyao init and cluster-create preflight policy
operations:
  init:
    platform:
      distributions:
        - name: openeuler
          version: "20.03"
          architectures: [amd64, arm64]
          kernelPrefixes: ["4.19."]
        - name: openeuler
          version: "22.03"
          architectures: [amd64, arm64]
          kernelPrefixes: ["5.10."]
        - name: openeuler
          version: "24.03"
          architectures: [amd64, arm64]
          kernelPrefixes: ["6.6."]
        - name: ubuntu
          version: "22.04"
          architectures: [amd64, arm64]
          kernelPrefixes: ["5.15."]
        - name: uos
          version: "20 1070e"
          architectures: [amd64, arm64]
          kernelPrefixes: ["4.19.", "5.10."]
    resources:
      - key: cpu
        description: Logical CPU count
        minimum: 2
      - key: memory
        description: Available memory
        minimum: 4294967296
      - key: systemDisk
        description: Available system disk space
        minimum: 107374182400
    tools:
      - key: tar
        description: tar executable
        names: [tar]
      - key: systemctl
        description: systemctl executable
        names: [systemctl]
    paths:
      - key: workspace
        description: Workspace is creatable and not read-only
        require: [create, notReadOnly]
    ports:
      - key: kubernetes
        description: Kubernetes API port
        defaultPort: 36443
        portFrom: port.kubernetes
      - key: imageRepo
        description: Image repository port
        defaultPort: 40443
        portFrom: port.imageRepo
      - key: chart
        description: Chart repository port
        defaultPort: 38080
        portFrom: port.chart
      - key: yum
        description: Package repository port
        defaultPort: 40080
        portFrom: port.yum
      - key: console
        description: BKE Console port
        defaultPort: 30010
        portFrom: port.console
        when: feature.installConsole
      - key: ntp
        description: Local NTP port
        defaultPort: 123
        protocol: udp
        when: feature.localNTP
    customChecks: []
  clusterCreate:
    platform:
      distributions:
        - name: openeuler
          version: "20.03"
          architectures: [amd64, arm64]
          kernelPrefixes: ["4.19."]
        - name: openeuler
          version: "22.03"
          architectures: [amd64, arm64]
          kernelPrefixes: ["5.10."]
        - name: openeuler
          version: "24.03"
          architectures: [amd64, arm64]
          kernelPrefixes: ["6.6."]
        - name: ubuntu
          version: "22.04"
          architectures: [amd64, arm64]
          kernelPrefixes: ["5.15."]
        - name: uos
          version: "20 1070e"
          architectures: [amd64, arm64]
          kernelPrefixes: ["4.19.", "5.10."]
    topology:
      masters: {minimum: 1, odd: true}
      etcd: {minimum: 1}
      workers: {minimum: 1}
      unique: [hostname, ip]
    managementCluster:
      api: true
      discovery: true
      crds: [bkeclusters, bkenodes]
      requiredKubeconfig: kube-system/localkubeconfig
    resources:
      - key: cpu
        description: Node logical CPU count
        minimum: 8
      - key: memory
        description: Node available memory
        minimum: 17179869184
      - key: systemDisk
        description: Node available system disk space
        minimum: 107374182400
    ports:
      - key: apiServer
        description: Node Kubernetes API port
        defaultPort: 6443
        portFrom: cluster.apiPort
        roles: [master]
      - key: etcdClient
        description: Etcd client port
        defaultPort: 2379
        roles: [etcd]
      - key: etcdPeer
        description: Etcd peer port
        defaultPort: 2380
        roles: [etcd]
      - key: etcdMetrics
        description: Etcd metrics port
        defaultPort: 2381
        roles: [etcd]
      - key: kubelet
        description: Kubelet secure port
        defaultPort: 10250
        roles: [master, worker]
      - key: kubeletHealth
        description: Kubelet health port
        defaultPort: 10248
        roles: [master, worker]
      - key: controllerManager
        description: Controller manager port
        defaultPort: 10257
        roles: [master]
      - key: scheduler
        description: Scheduler port
        defaultPort: 10259
        roles: [master]
      - key: localHA
        description: Multi-master local HA port
        defaultPort: 36443
        roles: [master]
        when: cluster.topology.multiMaster
      - key: bkeAgent
        description: BKE Agent port
        defaultPort: 58080
        roles: [all]
    customChecks: []
```

**Configuration Item Description**

Top-level fields:

| Field | Description |
| ---- | ---- |
| `apiVersion` | Policy API version, fixed as `preflight.openfuyao.io/v1`. |
| `kind` | Document type, fixed as `PreflightConfig`. |
| `metadata.release` | The release version corresponding to the policy; must exactly match the release of the current bke embedded Bundle, otherwise the policy cannot be loaded. |
| `metadata.description` | Human-readable description of the policy, for documentation purposes only; does not affect check behavior. |
| `operations` | Check policies for each installation phase; both `init` and `clusterCreate` phases must be retained. |

Common fields available for each check item entry:

| Field | Applicable Scope | Description |
| ---- | ---- | ---- |
| `description` | All check items | Check item description, required; displayed as the check item name in the report. |
| `when` | Port checks | Instantiation condition for the check; the value is a boolean key from the installation plan (e.g., `feature.installConsole` means Console is enabled this time, `feature.localNTP` means the local machine provides NTP, `cluster.topology.multiMaster` means multiple masters); the check only applies to this input when true. |
| `roles` | Port checks for cluster-create | Node roles to which the check applies; supports `master`, `worker`, `etcd`, `all` (`all` cannot be combined with other roles). |

`operations.init` check families:

| Check Family/Field | Description |
| ---- | ---- |
| `platform.distributions` | Supported OS matrix; each entry consists of `name` (distribution), `version` (version), `architectures` (architecture), `kernelPrefixes` (allowed kernel version prefixes); must be exactly consistent with the release policy and cannot expand or shrink the support scope. |
| `resources[].key` | Resource type; only supports `cpu`, `memory`, `systemDisk`. |
| `resources[].minimum` | Minimum resource threshold: `cpu` is the number of cores; `memory` and `systemDisk` are byte integers or strings with `B`/`K`/`M`/`G`/`T` suffixes (base 1024). |
| `tools[].key` / `tools[].names` | Executables required to be in PATH (`tar`, `systemctl`); `names` can only be a single executable name and cannot include a path. Only supported in the init phase. |
| `paths[].key` | `workspace` is the built-in workspace check, used with `require`; other keys need to specify a clean absolute path via `path`, only checking path existence and not supporting `require`. Only supported in the init phase. |
| `paths[].require` | Workspace requirements; supports `create` (self or parent path allows creation) and `notReadOnly` (the file system is not a read-only mount). |
| `ports[].key` | Port check identifier; only supports fixed keys: `kubernetes`, `imageRepo`, `chart`, `yum`, `console`, `ntp`. |
| `ports[].defaultPort` | Default port (1~65535); the actual checked port first takes the business input value pointed to by `portFrom`, and uses the default value only when not explicitly provided. |
| `ports[].portFrom` | Installation plan key for the actual port value source (e.g., `port.kubernetes`); the value source for each port is fixed and cannot be modified. |
| `ports[].protocol` | Port protocol; only supports `tcp`, `udp`; `ntp` must be `udp`, all other ports must be `tcp`. |

`operations.clusterCreate` check families:

| Check Family/Field | Description |
| ---- | ---- |
| `platform.distributions` | Platform matrix consistent with the init phase; must be exactly consistent with the release policy. |
| `topology.masters` | Master count requirements: `minimum` is the minimum count, `odd` requires an odd number and must remain enabled. |
| `topology.etcd` / `topology.workers` | Minimum count for etcd and workers (`minimum`). |
| `topology.unique` | Node uniqueness validation fields; only supports `hostname`, `ip`. |
| `managementCluster.api` / `managementCluster.discovery` | Management cluster API reachability and Discovery readability checks; must remain enabled. |
| `managementCluster.crds` | List of CRDs required to be registered; only supports `bkeclusters`, `bkenodes`. |
| `managementCluster.requiredKubeconfig` | Secret reference for the management cluster kubeconfig; only supports `kube-system/localkubeconfig`. |
| `resources` | Node resource thresholds; field meanings are consistent with the init phase (`cpu`, `memory`, `systemDisk`). |
| `ports[].key` | Role port check identifier; only supports fixed keys: `apiServer`, `etcdClient`, `etcdPeer`, `etcdMetrics`, `kubelet`, `kubeletHealth`, `controllerManager`, `scheduler`, `localHA`, `bkeAgent`. |
| `ports[].roles` | Node roles to which the port check applies; the actual port for `apiServer` is taken from the business file via `portFrom: cluster.apiPort`, while other ports use `defaultPort`. |

Rules for external policies:

- Must be **a complete policy file**; no field-level merge with the built-in YAML is performed.
- Top-level `apiVersion: preflight.openfuyao.io/v1`, `kind: PreflightConfig`, `metadata.release`, and `operations.init`, `operations.clusterCreate` must be retained.
- Fixed configuration keys already in the default policy (platform, ports, resources, paths, tools, topology, management cluster) must be retained; release checks cannot be disabled by deleting list entries.
- Resource `minimum` supports byte integers or strings with `B`/`K`/`M`/`G`/`T` suffixes (base 1024, e.g., `4294967296` or `4G`); after adjusting the threshold, the repair suggestion in the report will display the actual threshold.
- The platform matrix must be exactly consistent with the release policy; the support scope cannot be expanded or shrunk via an external file.
- Port `defaultPort`, `protocol`, `portFrom`, `when`, and `roles` are fixed; actual business ports should be passed via the installation command or cluster/node configuration (explicit CLI/business file values take priority over the policy's `defaultPort`).
- External policies cannot replace fixed rule IDs, Facts, operators, dependencies, SourceLocks, release facts, Collector contracts, or Provider defaults.
- External policies only affect this check and do not write back to subsequent installation command parameters.

Violating any of the above rules (empty file, multiple documents, unknown fields, release mismatch, deleted fixed keys, invalid field types, etc.) will prevent the preflight check from starting and will report the specific configuration path (`PREFLIGHT-CONFIG-INVALID`) in the report.

### Output Description

#### Terminal Output and Detailed Report

- During execution, progress information is written to the standard error stream stderr (e.g., `Preflight progress: collecting facts 3/12`), with a 5-second heartbeat prompt; slow nodes or stuck probes will not appear as command hangs.
- In the default `text` mode, the terminal displays the run conclusion, run issues, and FAIL/WARNs that need to be handled, using a six-column table of "Status, Check Item, Target, Expected, Actual, Reason and Suggestion"; terminals narrower than 72 columns automatically degrade to a vertical layout.
- text mode automatically generates a UTF-8 detailed report (`bke-preflight-<phase>-*.txt`) in the system temporary directory and displays the path at the end of the terminal; the detailed report includes PASS, FAIL, WARN, and all SKIPs (including dependency blocking reasons), without terminal colors, for easy archiving. You can specify a fixed archive path via `--output-file`.
- In `json` mode, stdout outputs a single JSON document (containing `overall`, `summary`, `issues`, `results`); progress is only written to stderr, which can be directly parsed by scripts and pipelines.
- The report contains `planHash`, `configDigest`, and `bundleDigest`; when auditing is required, they should be saved along with the complete report. The report may contain node addresses, paths, and source location information; before sharing, check for desensitization according to organizational requirements.

#### Check Status and Overall Conclusion

Individual check items have only four public statuses:

| Status | Description |
| ---- | ---- |
| PASS | Expectation met. |
| FAIL | Expectation not met; needs to be fixed before retrying. |
| WARN | Expectation not met but is only a warning; requires manual confirmation. |
| SKIP | Skipped due to dependency failure, timeout, cancellation, or policy reasons; references the earliest root cause. |

#### Exit Codes

| Exit Code | Overall Conclusion | Description |
| ---- | ---- | ---- |
| 0 | PASS | All applicable checks passed. |
| 2 | WARN or SKIP | WARNs exist, or no applicable items/all skipped; requires manual confirmation. |
| 3 | FAIL | Environment check failures exist. |
| 4 | INPUT_ERROR or BUNDLE_ERROR | Input, external policy, or embedded Bundle error; preflight check did not start. |
| 5 | INTERNAL_ERROR | SDK, Adapter, or Collector internal error. |
| 124 | TIMED_OUT | Overall timeout; some checks did not complete. |
| 130 | CANCELED | User cancellation. |

### Common Issues

#### Prompt PREFLIGHT-OPTIONS

Invalid runtime parameters. Please confirm: `--output` is `text` or `json`; `--timeout` and `--fact-timeout` are greater than 0; `--max-concurrency` is within the range of 0~64.

#### Prompt PREFLIGHT-CONFIG-INVALID

The external policy file failed validation. Common causes: file is empty, contains multiple YAML documents, has unknown fields, `metadata.release` does not match the current bke, fixed configuration keys were deleted, or the `customChecks` field is invalid. Please start over from the complete default file of the same revision as the current bke, modify only the supported fields, and retry.

#### Prompt INIT-INPUT-* or CLUSTER-INPUT-* (Exit Code 4)

Business input is invalid. Please fix the input according to the field path (e.g., `cluster.spec.clusterConfig.cluster.networking.podSubnet`, `nodes[0].spec.ip`) or file line number in the report, and re-run with parameters consistent with the installation command. Missing `--hostIP`, `--file`, or `--nodes` also falls into this category.

#### Prompt CLUSTER-MGMT-PROVIDER-DEFAULTS

Unable to read the management cluster Provider default configuration (e.g., kubeconfig is invalid, API is unreachable, or no permission to read ConfigMap `cluster-system/bke-config`). The check will continue using static default values, but it is recommended to restore the management cluster connection and grant read-only permissions before retrying.

#### SSH Check Failure or Mass SKIP

The SSH chain is layered as transport → authentication → permissions; after a root dependency fails, all downstream checks SKIP and reference the root cause. Please first resolve the first SSH failure in the report (network connectivity, credentials, root/passwordless sudo), and do not treat multiple SKIPs as multiple independent issues. Note that credential errors will not be misreported as network failures.

#### Source Check Passed but Pull Failed During Installation

The init phase source check only proves TCP connectivity; the HEAD probe in the cluster-create phase does not validate certificates and treats HTTP error statuses as reachable—neither represents authentication or artifact download success. Please continue troubleshooting the repository service, certificates, and credentials according to the installation error.

#### Command Execution Timeout (Exit Code 124)

A total timeout may be triggered when nodes are unreachable or the environment is too slow. Retry after restoring the network, or appropriately increase `--timeout`; when a single probe is stuck, adjust `--fact-timeout`.

#### Exit Code 2 but No FAIL

Exit code 2 indicates that WARNs exist or the overall conclusion is SKIP (no applicable checks or all skipped). Please review the cause and suggestions of WARNs in the detailed report and manually confirm; SKIP items will note the blocking root cause.

#### No Detailed Report Generated After Check

The detailed report is only automatically generated in `text` mode (written to the system temporary directory); in `json` mode, please specify the output file via `--output-file`.

## envCheck

> ![icon](../figures/icon-notice.gif)**Notice:**
> Since v26.09 provides the bke built-in unified preflight check (preflight), which shares input with installation commands and has more comprehensive check capabilities, it will gradually replace envCheck. During the transition period, if you need envCheck's unique capabilities such as residual file query/cleanup and remote batch distribution, you can continue to refer to this chapter; for new environments, please prioritize using the [bke Built-in Unified Preflight Check](#bke-built-in-unified-preflight-check).

### Background Information

Users have found during the use of the installation and deployment tool that residual configuration files from other software in the environment, as well as some applications that conflict with the specifications, can cause the installation and deployment process to be abnormal or fail. To ensure the successful execution of installation and deployment, a pre-check environment tool needs to be built to ensure that the environment is usable before installation and deployment. The tool currently implements the following functions.

- Given a list of file paths, query whether residual files exist in the current environment and output a check report.
- Given a list of file paths, supports safe deletion (asks the user whether to confirm deletion) and forced deletion (directly deletes without asking) of files, and outputs a check report.
- Checks whether specified programs are installed in the environment and outputs a check report.
- Supports time sync validation across multiple machines, detecting whether the clocks of other nodes are synchronized with the bootstrap node.
- Checks whether the OS kernel version meets configuration requirements.
- Checks local port usage, supporting port list configuration by role.
- Checks whether the remaining disk space meets configuration requirements, supporting filtering by role.
- Checks whether the current node has a default route.
- Checks whether the default policy of the iptables FORWARD chain is ACCEPT.
- Supports remote batch distribution checks, distributing the tool to multiple remote nodes to execute checks in parallel, collecting results centrally and generating a summary report.

The remote batch distribution check is recommended for the above functions; the other functions can be used when specific checks are needed.

### Prerequisites

This tool currently only supports execution in `Linux amd64` and `Linux arm64` environments. Please download the corresponding version of the tool according to your environment. There are three ways to obtain the tool and configuration file (for a configuration file example, see [Full Configuration File Example](#full-configuration-file-example)); all three methods have the same effect, and users can choose any of them:

- Obtain the tool and default configuration file directly via download links. The configuration file can be created by yourself or downloaded as the default configuration file for modification.

  |Download Item|amd64|arm64|
  |------|-----|-----|
  |envCheck tool|[Download](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/envCheck)|[Download](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/envCheck)|
  |Default configuration file|[Download](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/config.json)|[Download](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/config.json)|

- Execute the following commands in a Linux environment to directly download the tool and configuration file:

  ```bash
  # amd64 architecture
  wget -O envCheck https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/envCheck
  wget -O config.json https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/config.json
  chmod +x envCheck
  ```

  ```bash
  # arm64 architecture
  wget -O envCheck https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/envCheck
  wget -O config.json https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/config.json
  chmod +x envCheck
  ```

- Build from source; the source code is located in the [openfuyao/env-check](https://gitcode.com/openfuyao/env-check) repository. After cloning the source code, execute the following commands to build:

  ```bash
  git clone https://gitcode.com/openfuyao/env-check.git
  cd env-check

  # Build for the current architecture
  go build -o envCheck ./cmd

  # Cross-architecture build (required for remote distribution scenarios)
  GOOS=linux GOARCH=amd64 go build -o build/envCheck_amd64 ./cmd
  GOOS=linux GOARCH=arm64 go build -o build/envCheck_arm64 ./cmd
  ```

### Usage Limitations

None.

### Usage Tutorial

> ![](../figures/icon-note.gif)**Note:**
> - After download, the tool is named `envCheck` and is a binary executable file.
> - You can use different functions of the tool by modifying the configuration file and executing different commands. The corresponding configuration file needs to be in the same folder as the tool (if the folder is not in the PATH, you need to use ./envCheck when executing commands in the subsequent tutorial) and named `config.json` (the location of the configuration file can be placed freely, but when placed elsewhere, you need to explicitly specify it using the command-line parameter `--config` or `-c`). Here it is assumed to be in the same folder with the content shown in the example below.

#### Full Configuration File Example

```json
{
  "log_file": "./envCheck.log",
  "output_format": "text",
  "paths": [
    "$HOME/.kube",
    "/etc/kubernetes",
    "/usr/bin/kube*",
    "/usr/local/bin/kube*",
    "/usr/local/bin/crictl",
    "/etc/sysctl.d/k8s.conf",
    "/etc/systemd/system/kubelet.service",
    "/etc/systemd/system/kubelet.service.d",
    "/var/lib/etcd",
    "/var/lib/kubelet",
    "/run/containerd/containerd.sock",
    "/usr/lib/systemd/system/kubelet.service.d",
    "/var/run/containerd/containerd.sock",
    "/var/run/docker.sock"
  ],
  "clean_force": false,
  "program_list": [
    {
      "name": "docker",
      "should_exist": false
    },
    {
      "name": "kubelet",
      "should_exist": false
    },
    {
      "name": "containerd",
      "should_exist": false
    },
    {
      "name": "tar",
      "should_exist": true
    }
  ],
  "hosts": [
    {
      "ip": "192.168.2.10",
      "username": "root",
      "password": "******",
      "port": "22",
      "role": ["bootstrap"]
    },
    {
      "ip": "192.168.2.11",
      "username": "root",
      "password": "******",
      "port": "22",
      "role": ["master"]
    },
    {
      "ip": "192.168.2.12",
      "username": "root",
      "password": "******",
      "port": "22",
      "role": ["worker"]
    }
  ],
  "clock_threshold": 10,
  "kernel_check": {
    "min_version": "4.19",
    "operator": ">="
  },
  "port_check": {
    "ports": {
      "bootstrap": ["36443", "40080", "40443", "38080", "30010"],
      "master": ["6443", "2379", "2380", "2381", "10248", "10249", "10250", "10256", "10257", "10259", "3377", "58080", "1338"],
      "worker": ["3377", "58080", "10248", "10249", "10250", "10256", "1338"]
    },
    "timeout": 10
  },
  "disk_check": {
    "check_items": [
      {"path": "/", "min_free_gb": 50}
    ]
  },
  "dispatch": {
    "timeout": 600,
    "poll_interval": 15,
    "work_dir": "/tmp/envcheck",
    "concurrent_limit": 10
  }
}
```

> ![icon](../figures/icon-notice.gif)**Notice:**
> Regarding the configuration file, the tool performs the following validations.
> - When using the file query or cleanup function, `paths` cannot be empty.
> - When using the program check function, `program_list` cannot be empty.
> - When using the clock sync detection function, `hosts` cannot be empty.
> - Route check and iptables check do not require additional configuration.

For detailed configuration descriptions of each check item, the specific impact of residual files on bootstrap node initialization, and the analysis of automatic handling behavior during initialization, see [Meaning and Impact Analysis of Each Check Item in the Configuration File](../appendix/env-check-config-analysis.md).

Configuration file field descriptions:

| Field Name | Description |
| ------ | ---- |
| log_file | Path for storing the log file (including file name)|
| output_format | Output format for the tool execution result report; currently supports text, json |
| paths | List of file paths for file query and cleanup; supports environment variables and wildcards |
| clean_force | Whether to perform forced deletion (without user confirmation). The current implementation uses the command-line parameter `--force` as the authority; please explicitly add `--force` when you need to skip confirmation |
| program_list | List of applications to check for installation; each item contains `name` (program name) and `should_exist` (whether it is expected to exist)|
| hosts | List of machines for clock sync detection and remote distribution checks |
| ip | IP address for SSH connection |
| username | Username for SSH connection |
| password | Password for SSH connection |
| port | Port for SSH connection |
| role | Role of the machine (e.g., bootstrap, master, worker)|
| arch | Target architecture (amd64/arm64); automatically detected when empty |
| clock_threshold | Allowed threshold for machine clock error (unit: seconds)|
| kernel_check | Kernel version check configuration |
| kernel_check.min_version | Expected kernel version (string comparison), default 4.19 |
| kernel_check.operator | Comparison operator; supports `>=`, `>`, `==`, `<=`, `<`; default `>=` |
| port_check | Port usage check configuration; configures port lists by role |
| port_check.ports | Port lists mapped by role; port lists for each role are independent |
| port_check.timeout | TCP connection probe timeout (unit: seconds), default 3 seconds |
| disk_check | Disk space check configuration |
| disk_check.check_items | List of disk check items |
| disk_check.check_items[].path | Directory path to check; when the path does not exist, automatically searches upward for the nearest parent directory |
| disk_check.check_items[].min_free_gb | Minimum available space required for this path (GB)|
| disk_check.check_items[].roles | (Optional) Only check this item on nodes with the specified role; empty means all nodes are checked |
| dispatch | Remote distribution execution configuration |
| dispatch.timeout | Total timeout for result collection (unit: seconds), default 600 |
| dispatch.poll_interval | Time interval for polling results (unit: seconds), default 15 |
| dispatch.work_dir | Working directory on remote nodes, default `/tmp/envcheck` |
| dispatch.concurrent_limit | Maximum concurrency for SSH connections/operations, default 10 |
| dispatch.checks | List of check items for remote execution |
| dispatch.skip_checks | List of check items to skip during remote execution |

> ![](../figures/icon-note.gif)**Note:**
> When running, the tool by default outputs the result to the terminal and generates the corresponding HTML file in the current folder. During execution, logs are recorded at the path specified by `log_file` in the configuration file (default is the `envCheck.log` file in the current folder). When `output_format` is `text`, the result is output to the terminal; when `output_format` is `json`, the corresponding JSON result file is additionally saved. The HTML report is always generated when the command is executed. Users can configure this by modifying the corresponding fields in the configuration file. The following sections assume `output_format` is `text` in the default configuration file.

Users can execute the following command to get the tool help documentation.

```bash
envCheck -h
```

#### Remote Batch Check

The `run` command connects to the remote nodes in the configuration, distributes `envCheck` and the configuration file to the remote working directory, executes checks on the remote side, and then summarizes the results.

**Prerequisites**

- When using the remote distribution `run` command, the target nodes in the configuration need to allow SSH login and support SFTP file transfer.
- Remote distribution uses password login by default; the configuration file needs to provide `hosts[].username`, `hosts[].password`, `hosts[].port`.
- When you need to check system directories, ports, routes, iptables, or clean up system files, it is recommended to execute with a user that has the corresponding permissions.

**Getting Started**

- Execute the following command to perform all check items on all configured nodes.

  ```bash
  envCheck run
  ```

- Execute the following command to specify the check items to execute.

  ```bash
  envCheck run --checks kernel,port,disk
  ```

- Execute the following command to specify the check items to skip.

  ```bash
  envCheck run --skip clock,fileQuery
  ```

- Execute the following command to perform remote checks only on nodes with the specified roles.

  ```bash
  envCheck run --roles master,worker
  ```

- Execute the following command to set the timeout and specify the configuration file.

  ```bash
  envCheck run -c ./clusterConfig.json --checks kernel,port,disk --roles master --timeout 900
  ```

  After the tool finishes running, it outputs the summary result to the terminal and generates a `dispatch.html` file in the current folder.

**Parameter Description**

| Parameter | Description |
| ---- | ---- |
| --checks | Specifies the check items to execute, comma-separated |
| --skip | Specifies the check items to skip, comma-separated |
| --roles | Performs remote checks only on nodes with the specified roles |
| --timeout | Timeout for waiting for remote results, in seconds, default 600 |

Available check items:

```text
kernel, port, disk, clock, fileQuery, programCheck, route, iptables
```

> ![](../figures/icon-note.gif)**Note:**
> The remote execution flow is as follows:
> 1. Connect to the target nodes in `hosts` (concurrent, limited by `concurrent_limit`).
> 2. Automatically detect the remote architecture, identifying `amd64` or `arm64`.
> 3. Clean up and create the remote working directory, default `/tmp/envcheck`.
> 4. Distribute the architecture-matched `envCheck` binary (gzip compressed transfer; skipped if the remote already has it with the same size) and node configuration (port configuration trimmed by role).
> 5. Execute local checks in the remote background.
> 6. Poll and pull the remote `result.json` or `error.log` and generate a summary report.
>
> If the remote architecture has been identified, the tool will first look for `./build/envCheck_<arch>` or `./envCheck_<arch>` (e.g., `./build/envCheck_amd64`). If no architecture-specific file is found, it uses the currently running `envCheck` executable.

#### File Query Function

- Execute the following command to perform a default query.

  ```bash
  envCheck query
  ```

  The tool queries whether files exist at the paths specified by `paths` in `config.json` in the current folder. After the tool finishes running, it outputs the query result to the terminal and generates a `query.html` file in the current folder. `paths` supports environment variable and wildcard expansion, e.g., `$HOME/.kube`, `/usr/bin/kube*`.

- Execute the following command to specify the configuration file path during the query.

  ```bash
  envCheck query --config=./queryConfig.json
  ```

  The tool queries whether files exist at the paths specified by `paths` in `queryConfig.json`. After the tool finishes running, it outputs the query result to the terminal and generates a `query.html` file in the current folder.

#### File Cleanup Function

> ![](../figures/icon-note.gif)**Note:**
> Before executing the file cleanup function, you need to confirm the value of the `clean_force` field in the configuration file, which defaults to `false`. `false` means you will be asked for confirmation before deleting files; `true` means no confirmation will be asked. The current implementation uses the command-line parameter as the authority; if you add the `--force` flag to the command, the tool will directly delete the corresponding files regardless of the value of the `clean_force` field. It is recommended to first execute `query` to confirm the target path, then execute `clean`.

- Execute the following command to perform a default cleanup (the user will be asked before each file is deleted).

  ```bash
  envCheck clean
  ```

  The tool asks whether to delete the files at the paths specified by `paths` in `config.json` in the current folder. After the tool finishes running, it outputs the cleanup result to the terminal and generates a `clean.html` file in the current folder.

- Execute the following command to perform a forced deletion; files will be deleted directly without asking the user.

  ```bash
  envCheck clean --force
  ```

  The tool directly deletes the files at the paths specified by `paths` in `config.json` in the current folder. After the tool finishes running, it outputs the cleanup result to the terminal and generates a `clean.html` file in the current folder.

- Execute the following command to specify the configuration file path during deletion.

  ```bash
  envCheck clean --config=./cleanConfig.json
  ```

  The tool performs the file cleanup function according to the settings in the `cleanConfig.json` file. After the tool finishes running, it outputs the cleanup result to the terminal and generates a `clean.html` file in the current folder.

#### Program Existence Detection

> ![](../figures/icon-note.gif)**Note:**
> The `program_list` field has changed from a string array format to an object array format. Old configuration files need to be updated to the latest format, otherwise the tool will report a JSON parsing error.

The program check function supports configuring the expected installation status for each program. `should_exist` being `false` means the program should not exist; if it is detected as installed, the check fails. `should_exist` being `true` means the program should exist; if it is detected as not installed, the check fails.

Configuration example:

```json
"program_list": [
  {
    "name": "docker",
    "should_exist": false
  },
  {
    "name": "kubelet",
    "should_exist": false
  },
  {
    "name": "containerd",
    "should_exist": false
  },
  {
    "name": "tar",
    "should_exist": true
  }
]
```

- Execute the following command to perform a default program query.

  ```bash
  envCheck program
  ```

  The tool queries whether the applications specified by `program_list` in `config.json` in the current folder meet the expected installation status. After the tool finishes running, it outputs the query result to the terminal and generates a `check.html` file in the current folder.

- Execute the following command to specify the configuration file path during the query.

  ```bash
  envCheck program --config=./checkConfig.json
  ```

  The tool queries whether the applications specified by `program_list` in `checkConfig.json` meet the expected installation status. After the tool finishes running, it outputs the query result to the terminal and generates a `check.html` file in the current folder.

#### Clock Sync Detection Function

**Prerequisites**

Using the clock sync detection function requires the user to first modify the relevant `hosts` information in the default configuration file, and at least one node with the `bootstrap` role must be configured.

**Getting Started**

- Execute the following command to query the clock error between other hosts and the bootstrap node host.

  ```bash
  envCheck clock
  ```

  The tool uses `hosts` and `clock_threshold` in `config.json` in the current folder to determine whether the clock error between non-bootstrap nodes and the bootstrap node is no more than the specified threshold. After the tool finishes running, it outputs the detection result to the terminal and generates a `clock.html` file in the current folder.

- Execute the following command to specify the configuration file during the query.

  ```bash
  envCheck clock --config=./clockConfig.json
  ```

  The tool uses `hosts` and `clock_threshold` in `clockConfig.json` to determine whether the clock error between non-bootstrap nodes and the bootstrap node is no more than the specified threshold. After the tool finishes running, it outputs the detection result to the terminal and generates a `clock.html` file in the current folder.

#### Kernel Version Check

**Prerequisites**

The kernel version check function depends on the `kernel_check` field in the configuration file. If `kernel_check` is not configured, the command uses the default requirement: `>= 4.19`.

**Getting Started**

- Execute the following command to check whether the current node's kernel version meets the requirements.

  ```bash
  envCheck kernel
  ```

  The tool checks the kernel version according to the `kernel_check` configuration in `config.json` in the current folder. After the tool finishes running, it outputs the check result to the terminal and generates a `kernel.html` file in the current folder.

- Execute the following command to specify the configuration file path during the check.

  ```bash
  envCheck kernel --config=./kernelConfig.json
  ```

  The tool checks the kernel version according to the `kernel_check` configuration in `kernelConfig.json`. After the tool finishes running, it outputs the check result to the terminal and generates a `kernel.html` file in the current folder.

> ![](../figures/icon-note.gif)**Note:**
> - Supported comparison operators: `>=`, `>`, `==`, `<=`, `<`.
> - Special compatibility is provided for the `3.10.0-xxx` kernel version of CentOS 7: if the version string is in the old build number format, it is treated as not satisfied in `>=`/`>` comparisons, preventing old build numbers from being misjudged as passing.

#### Port Usage Check

**Prerequisites**

The port usage check function depends on the `port_check` field in the configuration file. If `port_check` is not configured, the command uses the default port configuration.

**Getting Started**

- Execute the following command to check the port usage of all roles on the current node.

  ```bash
  envCheck port
  ```

  The tool checks whether the ports corresponding to the current node's roles are occupied according to `port_check.ports` in `config.json` in the current folder. After the tool finishes running, it outputs the check result to the terminal and generates a `port.html` file in the current folder.

- Execute the following command to check only the ports of the specified roles.

  ```bash
  envCheck port --roles master,worker
  ```

  The tool only checks the ports corresponding to the `master` and `worker` roles. After the tool finishes running, it outputs the check result to the terminal and generates a `port.html` file in the current folder.

- Execute the following command to specify the configuration file path during the check.

  ```bash
  envCheck port --config=./portConfig.json --roles bootstrap
  ```

  The tool checks the ports corresponding to the `bootstrap` role according to `port_check.ports` in `portConfig.json`. After the tool finishes running, it outputs the check result to the terminal and generates a `port.html` file in the current folder.

> ![](../figures/icon-note.gif)**Note:**
> - When `--roles` is empty, all configured role ports are checked.
> - `port_check.timeout` is the port probe timeout, in seconds.
> - During distributed execution, the Runner filters the configuration based on the node role, retaining only the ports relevant to that node's role.

#### Disk Space Check

**Prerequisites**

The disk space check function depends on the `disk_check` field in the configuration file. If `disk_check` is not configured, the command uses the default check items.

Configuration example:

```json
"disk_check": {
  "check_items": [
    {
      "path": "/",
      "min_free_gb": 50
    },
    {
      "path": "/var/lib",
      "min_free_gb": 80,
      "roles": ["worker"]
    }
  ]
}
```

**Getting Started**

- Execute the following command to check whether the remaining disk space meets the requirements.

  ```bash
  envCheck disk
  ```

  The tool checks whether the remaining space of the file system where the specified path is located meets the threshold according to `disk_check.check_items` in `config.json` in the current folder. After the tool finishes running, it outputs the check result to the terminal and generates a `disk.html` file in the current folder.

- Execute the following command to specify the configuration file path during the check.

  ```bash
  envCheck disk --config=./diskConfig.json
  ```

  The tool checks the disk space according to `disk_check.check_items` in `diskConfig.json`. After the tool finishes running, it outputs the check result to the terminal and generates a `disk.html` file in the current folder.

> ![](../figures/icon-note.gif)**Note:**
> - If the path does not exist, automatically searches upward for the nearest parent directory for space statistics.
> - If the path is a file, uses its directory for statistics.
> - If a `check_items` entry has the `roles` field configured, this item is only checked on nodes with matching roles.

#### Default Route Check

- Execute the following command to check whether the current node has a default route.

  ```bash
  envCheck route
  ```

  The tool checks whether the current node has a default route and outputs gateway, network interface, metric, and other information. This check does not require additional configuration. After the tool finishes running, it outputs the check result to the terminal and generates a `route.html` file in the current folder.

- Execute the following command to specify the configuration file path during the check.

  ```bash
  envCheck route --config=./routeConfig.json
  ```

  After the tool finishes running, it outputs the check result to the terminal and generates a `route.html` file in the current folder.

#### iptables Policy Check

- Execute the following command to check whether the default policy of the iptables FORWARD chain is ACCEPT.

  ```bash
  envCheck iptables
  ```

  The tool checks whether the default policy of the iptables `FORWARD` chain is `ACCEPT`. This check does not require additional configuration. After the tool finishes running, it outputs the check result to the terminal and generates an `iptables.html` file in the current folder.

- Execute the following command to specify the configuration file path during the check.

  ```bash
  envCheck iptables --config=./iptablesConfig.json
  ```

  After the tool finishes running, it outputs the check result to the terminal and generates an `iptables.html` file in the current folder.

### Output Description

Each command execution writes to the log file and generates an HTML report based on the command. When `output_format` is `json`, the corresponding JSON result file is also saved.

| Command | HTML Report | JSON Result File |
| ---- | --------- | ------------- |
| query | query.html | query-result.json |
| clean | clean.html | clean-result.json |
| program | check.html | program-check-result.json |
| clock | clock.html | clock-check-result.json |
| kernel | kernel.html | kernel-check-result.json |
| port | port.html | port-check-result.json |
| disk | disk.html | disk-check-result.json |
| route | route.html | route-check-result.json |
| iptables | iptables.html | iptables-check-result.json |
| run | dispatch.html | dispatch-result.json |

> ![](../figures/icon-note.gif)**Note:**
> - When `output_format` is `text`, the result is output to the terminal and no JSON result file is saved.
> - When `output_format` is `json`, the corresponding JSON result file is saved.
> - The HTML report is always generated when the command is executed.

### Common Issues

#### Prompt: at least one path is required

The `query` or `clean` command requires `paths` to be configured. Please check whether the configuration file path is correct and whether `paths` is empty.

#### Prompt: program list is required

The `program` command requires `program_list` to be configured. Please add at least one program check item.

#### Prompt: hosts field is required

The `clock` command requires `hosts` to be configured. Please add at least one node and ensure that a `bootstrap` role node is included.

#### Remote Check Connection Failure

Please check:

- Whether the IP, port, username, and password are correct.
- Whether the remote SSH service is available.
- Whether the remote allows password login.
- Whether the network from the current machine to the remote is connected.
- Whether the remote user has permission to create `dispatch.work_dir`.

#### No JSON File Generated

The JSON result file is only saved when `output_format` is set to `json`. The HTML report is generated when the command is executed.
