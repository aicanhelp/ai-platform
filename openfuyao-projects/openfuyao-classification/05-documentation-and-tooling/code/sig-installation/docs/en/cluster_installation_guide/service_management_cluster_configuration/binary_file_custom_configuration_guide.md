# Binary Component Custom Configuration (Containerd and kubelet)

## Background Information

This document describes how to use Custom Resources (CRs) to dynamically configure the Kubelet and Containerd components. In a BKECluster (abbreviated as bc), the corresponding configuration CR is associated through a reference field:

| Abbreviation | CR Type | kubectl Short Name | BKECluster Reference Field |
| --- | --- | --- | --- |
| KCT | `KubeletConfig` | `kct` | `spec.KubeletConfigRef` |
| CCT | `ContainerdConfig` | `cct` | `spec.clusterConfig.cluster.containerdConfigRef` |

After creating a KCT/CCT, fill in the reference in the BKECluster YAML to automatically deliver the configuration during the node bootstrap phase, without logging into the node to manually edit files. For the complete parameter list, see [cluster-api-provider-bke Configuration Parameters](../appendix/cluster-api-provider-bke_configuration.md).

## Prerequisites

- The management cluster (the same cluster as the BKECluster) has deployed cluster-api-provider-bke and the corresponding CRDs.
- Before creating the cluster, prepare the KCT/CCT resources and complete the references in the BKECluster.

## Usage Restrictions

- The KCT/CCT resources must be located in the same management cluster as the BKECluster.
- The KCT `spec.files[]` supports absolute path writes; ensure the CR content is trusted to avoid writing to sensitive system paths.
- The CCT `spec.configType` supports `service`, `main`, `registry`, and `combined` (default). Only the declared `spec` subsections are rendered.

## Tutorial

### Referencing KCT/CCT in BKECluster

Before deploying the cluster, first create the KCT/CCT resources and complete the references in the BKECluster, then run `bke create cluster`. Common editing commands are as follows:

```bash
# View BKECluster / KubeletConfig / ContainerdConfig
kubectl get bc -A
kubectl get kct -A
kubectl get cct -A

# Edit (replace the placeholders with the actual namespace and name)
kubectl edit bc -n <namespace> <name>
kubectl edit kct -n <namespace> <name>
kubectl edit cct -n <namespace> <name>

# Or create/update via YAML and then apply
kubectl apply -f my-kubelet.yaml
kubectl apply -f my-containerd.yaml
```

The example of referencing in the BKECluster YAML is as follows:

```yaml
apiVersion: bke.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: bke-cluster
  namespace: bke-cluster
spec:
  KubeletConfigRef:          # Reference KCT
    name: my-kubelet
    namespace: my-kubelet
  clusterConfig:
    cluster:
      containerdConfigRef:   # Reference CCT
        name: my-containerd
        namespace: my-containerd
```

During node bootstrap, bkeagent reads the KCT/CCT from the management cluster and applies it to the target node.

### KubeletConfig CR Configuration

#### Overview

The `KubeletConfig` CR allows you to dynamically configure Kubelet's configuration file and systemd service file through Kubernetes resources, without manually editing files on the node.

#### CR Structure

```bash
apiVersion: bke.bocloud.com/v1beta1
kind: KubeletConfig
metadata:
  name: <config-name>
  namespace: <namespace>  # Defaults to bke-system
spec:
  kubeletConfig:
    kubelet.conf:
      raw: |
        # YAML content of the Kubelet configuration file
  kubeletService:
    service:
      # Kubelet systemd service configuration
```

**spec Field Description**

**`spec.kubeletConfig`**

Used to configure Kubelet's configuration file (`/var/lib/kubelet/config.yaml`).

| Parameter           | Type | Required | Description                          |
|--------------------| --- |------|-------------------------|
| `kubelet.conf`     | `map[string]RawExtension` | Yes    | Configuration object containing the `raw` field        |
| `kubelet.conf.raw` | `string` | Yes    | Complete YAML content of the Kubelet configuration file |

**`raw` field format:**

The `raw` field should contain the complete YAML content of the Kubelet configuration file, for example:

```yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: systemd
clusterDNS:
  - 10.96.0.10
clusterDomain: cluster.local
...
```

The kubelet.config implements passthrough of the filled-in kubelet.conf.raw; the user is responsible for ensuring the correctness of the passed-in configuration items.

**`spec.kubeletService`**

Used to configure Kubelet's systemd service file (`/etc/systemd/system/kubelet.service`).

| Parameter        | Type | Required | Description                     |
|-----------| --- |------|------------------------|
| `service` | `KubeletService` | No    | Kubelet systemd service configuration object |

**kubeletService field description:**

| Parameter            | Type | Required | Description                               |
|----------------------| --- |------|----------------------------------|
| `ExecStart`          | `string` | No    | Service startup command                           |
| `Restart`            | `string` | No    | Restart policy (e.g., always, on-failure)     |
| `RestartSec`         | `string` | No    | Restart interval (e.g., 5s, 10s)              |
| `StartLimitInterval` | `string` | No    | Startup limit time window                         |
| `StartLimitBurst`    | `int` | No    | Startup limit count                           |
| `KillMode`           | `string` | No    | Kill mode (e.g., control-group, process) |
| `StandardOutput`     | `string` | No    | Standard output redirection (e.g., journal, null)       |
| `StandardError`      | `string` | No    | Standard error redirection                          |
| `SyslogIdentifier`   | `string` | No    | Syslog identifier                          |
| `WorkingDirectory`   | `string` | No    | Working directory                             |
| `User`               | `string` | No    | Running user                             |
| `Group`              | `string` | No    | Running group                              |
| `ExecStartPre`       | `[]string` | No    | List of commands to execute before startup                       |
| `Environment`        | `[]string` | No    | Environment variable list (format: KEY=VALUE)           |
| `EnvironmentFile`    | `[]string` | No    | List of environment variable file paths                       |
| `CustomExtra`        | `map[string]string` | No    | Custom extra configuration items (key-value pairs)                   |

kubelet.service contains the configuration items supported by the service during kubelet startup.

**`spec.files[]`**

Before the kubelet configuration is applied, write additional files to the node (relative paths are based on `/etc/kubernetes`; absolute paths are also supported).

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `path` | `string` | Yes | Target file path |
| `content` | `string` | Yes | File content, supports variable substitution |
| `permissions` | `string` | No | Octal permissions, e.g., `0644` |
| `owner` | `string` | No | File owner, e.g., `root:root` |

**`spec.commands[]`**

Execute additional commands after the kubelet configuration is applied.

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `command` | `string` | Yes | Command or executable script; when `args` is empty, executed via `sh -c` |
| `args` | `[]string` | No | Command argument list |
| `workingDir` | `string` | No | Working directory |

**kubelet Dynamic Value Passing Feature**

The dynamic value passing mechanism allows variables to be used in configuration files. These variables are replaced with actual values during deployment. This provides great flexibility, allowing the same configuration to be applied to different nodes and environments.

**Enabling Variable Substitution**

During the deployment of `BKECluster`, the system automatically detects `KubeletConfigRef` and enables the variable substitution feature. Variable substitution takes effect in the following locations:

1. **kubelet.conf** configuration
2. **kubelet.service** systemd service file
3. File contents in **spec.files[]**

Expression variables: `${EXPR|command|END}`

Executes a shell command and uses the output as the variable value.

**Syntax**:

```
${EXPR|shell_command|END}
```

**Example**:

```yaml
spec:  
  kubeletConfig:    
    kubelet.conf:      
      raw: |        
          apiVersion: kubelet.config.k8s.io/v1beta1
          kind: KubeletConfiguration
          # Get the number of CPU cores
          cpuManagerPolicy: "static"
          reservedSystemCPUs: "${EXPR|nproc|END}"
          # Get memory size (MB)
          evictionHard:
            memory.available: "${EXPR|free -m | grep Mem | awk '{print $2}'|END}Mi"
          # Get the hostname
          hostnameOverride: "${EXPR|hostname|END}"
```

> ![icon](../figures/icon-note.gif) **Note:**<br />
>- Commands are executed on the target node
>- When command execution fails, the variable remains as-is (not replaced)
>- Command output has leading and trailing whitespace removed

**kubelet configuration file kct reference**

**my-kubelet.yaml**

```yaml
apiVersion: bke.bocloud.com/v1beta1
kind: KubeletConfig
metadata:
  name: my-kubelet
  namespace: my-kubelet
  labels:
    test: "true"
    component: "kubelet"
spec:

  kubeletConfig:
    kubelet.conf:
      raw: |
        apiVersion: kubelet.config.k8s.io/v1beta1
        kind: KubeletConfiguration
        authentication:
          anonymous:
            enabled: false
          webhook:
            cacheTTL: 2m0s
            enabled: true
          x509:
            clientCAFile: /etc/kubernetes/pki/ca.crt
        authorization:
          mode: Webhook
          webhook:
            cacheAuthorizedTTL: 5m0s
            cacheUnauthorizedTTL: 30s
        address: "${EXPR|ip -4 addr show | awk '/inet / {split($2,a,"/"); print a[1]}' | grep -vE '^127\.0\.0\.1|^169\.254\.' | head -n1|END}"
        port: 10250
        readOnlyPort: 0
        cgroupDriver: systemd
        hairpinMode: promiscuous-bridge
        serializeImagePulls: false
        clusterDNS:
          - "10.96.0.10"
        clusterDomain: "cluster.local"
        healthzBindAddress: 127.0.0.1
        healthzPort: 10248
        rotateCertificates: true
        staticPodPath: /etc/kubernetes/manifests
        streamingConnectionIdleTimeout: 4h0m0s
        syncFrequency: 1m0s
        volumeStatsAggPeriod: 1m0s

  kubeletService:
    enabled: true
    serviceName: "kubelet"
    unit:
      description: "Kubernetes Kubelet (my-Test)"
      documentation: "https://kubernetes.io/docs/concepts/overview/components/#kubelet"
      after:
        - "containerd.service"
        - "network-online.target"
      wants:
        - "network-online.target"
      requires:
        - "containerd.service"
    service:
      workingDirectory: "/var/lib/kubelet"
      execStartPre:
        - "/usr/bin/mkdir -p /sys/fs/cgroup/cpuacct/system.slice/kubelet.service"
        - "/usr/bin/mkdir -p /sys/fs/cgroup/hugetlb/system.slice/kubelet.service"
      execStart: "/usr/bin/kubelet \
      --bootstrap-kubeconfig=/etc/kubernetes/bootstrap-kubelet.conf \
      --kubeconfig=/etc/kubernetes/admin.conf \
      --config=/var/lib/kubelet/config.yaml \
      --container-runtime-endpoint=unix:///run/containerd/containerd.sock \
      --pod-infra-container-image=registry.k8s.io/pause:3.9 \
      --cgroup-driver=systemd \
      --register-node=true \
      --v=2 \
      --node-ip=${EXPR|ip -4 addr show scope global | awk '/inet /{print $2}' | cut -d'/' -f1|END} \
      --hostname-override=${EXPR|hostname -i | awk '{print $1}'|END}
      "
      restart: "always"
      restartSec: 10
      startLimitInterval: 30
      startLimitBurst: 3
      killMode: "process"
      standardOutput: "syslog"
      standardError: "syslog"
      syslogIdentifier: "kubelet"
      user: "root"
      group: "root"

    install:
      wantedBy:
        - "multi-user.target"
      requiredBy: []

    variables:
      kubeletBinaryPath: "/usr/bin/kubelet"
      kubeletConfigPath: "/etc/kubernetes/kubelet.conf"
      nodeName: "test-node-01"
      nodeIP: "xx"
      containerdEndpoint: "unix:///run/containerd/containerd.sock"
      pauseImage: "registry.aliyuncs.com/google_containers/pause:3.9"

  files:
    - path: "extra/kubelet-custom.conf"
      content: |
        clusterDomain: "cluster.local"
        nodeHostname: "${EXPR|hostname|END}"
        nodePrimaryIP: "${EXPR|ip -4 addr show scope global | awk '/inet /{print $2}' | cut -d'/' -f1|END}"
      permissions: "0644"
      owner: "root:root"
    - path: "/var/lib/kubelet/custom-flag.txt"
      content: |
        createdAt: "${EXPR|date +%Y-%m-%d_%H:%M:%S|END}"
        configSource: my-kubelet-config
      permissions: "0600"
      owner: "root:root"

  commands:
    - command: "/bin/bash"
      args:
        - "-c"
        - |
          echo "=== KubeletConfig post-command ===" >> /var/log/kubelet-custom.log
          echo "Execution time: $(date +%Y-%m-%d_%H:%M:%S)" >> /var/log/kubelet-custom.log
          if [ -f /var/lib/kubelet/config.yaml ]; then
            echo "kubelet config generated, size: $(du -sh /var/lib/kubelet/config.yaml | awk '{print $1}')" >> /var/log/kubelet-custom.log
          else
            echo "Warning: kubelet config does not exist" >> /var/log/kubelet-custom.log
          fi
      workingDir: "/var/log"
    - command: "/usr/bin/systemctl"
      args:
        - "is-active"
        - "containerd.service"
      workingDir: "/"

```

> ![icon](../figures/icon-note.gif) **Note:**<br />
> - The application order is `spec.files[]` → `spec.kubeletConfig` → `spec.kubeletService` → `spec.commands[]`.
> - When `files[].path` is a relative path, it is based on `/etc/kubernetes`; absolute paths are also supported (e.g., `/var/lib/kubelet/custom-flag.txt`).
> - `files[].content` supports `${EXPR|command|END}` and plain variable `${VAR_NAME}` substitution.
> - `commands[]` is executed after the kubelet configuration is applied; when `args` is empty, `command` is executed via `sh -c`. For multi-line scripts, it is recommended to use `command` + `args: ["-c", "..."]`.

**files reference (can be extracted separately)**

```yaml
  files:
    - path: "extra/kubelet-custom.conf"       # Relative path → /etc/kubernetes/extra/kubelet-custom.conf
      content: |
        clusterDomain: "cluster.local"
        nodeHostname: "${EXPR|hostname|END}"
      permissions: "0644"
      owner: "root:root"
    - path: "/var/lib/kubelet/custom-flag.txt" # Absolute path
      content: |
        createdAt: "${EXPR|date +%Y-%m-%d_%H:%M:%S|END}"
      permissions: "0600"
      owner: "root:root"
```

**commands reference (can be extracted separately)**

```yaml
  commands:
    # Method 1: args is empty, command executed via sh -c (suitable for single-line commands)
    - command: "echo kubelet post-hook >> /var/log/kubelet-custom.log"

    # Method 2: command + args (suitable for multi-line scripts, recommended)
    - command: "/bin/bash"
      args:
        - "-c"
        - |
          echo "=== KubeletConfig post-command ===" >> /var/log/kubelet-custom.log
          systemctl is-active containerd.service >> /var/log/kubelet-custom.log
      workingDir: "/var/log"

    # Method 3: directly call an executable file
    - command: "/usr/bin/systemctl"
      args:
        - "is-active"
        - "containerd.service"
      workingDir: "/"
```

### ContainerdConfig CR Configuration

#### Overview

The `ContainerdConfig` CR allows you to dynamically configure Containerd's main configuration file, service file, and image registry configuration through Kubernetes resources.

#### CR Structure

```yaml
apiVersion: bke.bocloud.com/v1beta1
kind: ContainerdConfig
metadata:
  name: <config-name>
  namespace: <namespace>
spec:
  configType: combined   # service / main / registry / combined (default)
  description: <description>
  script:
    # Script configuration
  service:
    # Containerd systemd service configuration
  main:
    # Containerd main configuration file (config.toml)
  registry:
    # Image registry configuration
```

**`spec.configType`**

| Value | Description |
| --- | --- |
| `combined` | Default value; renders the declared subsections in `spec` in sequence |
| `service` | Only renders the systemd drop-in |
| `main` | Only renders `config.toml` |
| `registry` | Only renders `hosts.toml` |

**spec Field Description**

**`spec.script`**

Used to execute a custom script before configuring Containerd.

| Parameter       | Type | Required | Description                                        |
|---------------| --- |------|-------------------------------------------|
| `Content`     | `string` | No    | Script content (inline script)                               |
| `Path`        | `string` | No    | Script file path (local file)                             |
| `Interpreter` | `string` | No    | Interpreter (e.g., /bin/bash, /bin/sh), defaults to /bin/bash |
| `Args`        | `[]string` | No    | Argument list passed to the script                                |

> ![icon](../figures/icon-note.gif) **Note:**<br />`Content` and `Path` are mutually exclusive; if both are provided, `Content` takes precedence.

**`spec.service`**

Used to configure Containerd's systemd service drop-in file (`/etc/systemd/system/containerd.service.d/10-override.conf`).

| Parameter            | Type | Required | Description               |
|----------------------| --- |------|------------------|
| `ExecStart`          | `string` | No    | Service startup command           |
| `Slice`              | `string` | No    | systemd slice unit |
| `KillMode`           | `string` | No    | Kill mode             |
| `Restart`            | `string` | No    | Restart policy             |
| `RestartSec`         | `string` | No    | Restart interval           |
| `TimeoutStopSec`     | `string` | No    | Stop timeout           |
| `StartLimitInterval` | `string` | No    | Startup limit time window         |
| `StartLimitBurst`    | `int` | No    | Startup limit count           |
| `Logging`            | `LoggingConfig` | No    | Logging configuration             |
| `CustomExtra`        | `map[string]string` | No    | Custom extra configuration items         |

**LoggingConfig field:**

| Parameter           | Type | Required | Description      |
|--------------------| --- |------|---------|
| `StandardOutput`   | `string` | No    | Standard output redirection |
| `StandardError`    | `string` | No    | Standard error redirection |
| `SyslogIdentifier` | `string` | No    | Syslog identifier |
| `LogLevelMax`      | `string` | No    | Maximum log level  |

**`spec.main`**

Used to configure Containerd's main configuration file (`/etc/containerd/config.toml`).

| Parameter          | Type | Required | Description                                        |
|------------------| --- |------|-------------------------------------------|
| `SandboxImage`   | `string` | No    | Pod sandbox image (e.g., pause image)                     |
| `Root`           | `string` | No    | Containerd data root directory, defaults to `/var/lib/containerd` |
| `State`          | `string` | No    | Containerd state directory, defaults to `/run/containerd`     |
| `ConfigPath`     | `string` | No    | Configuration file path, defaults to `/etc/containerd/certs.d`      |
| `MetricsAddress` | `string` | No    | Metrics monitoring address                                  |

> ![icon](../figures/icon-note.gif) **Note:**<br />
>If the `main` field is specified, the system generates the `config.toml` file based on the template. If not specified, the default configuration is used.

**`spec.registry`**

Used to configure the `hosts.toml` file for the image registry.

| Parameter     | Type | Required | Description                                            |
|--------------| --- |------|-----------------------------------------------|
| `ConfigPath` | `string` | No    | hosts.toml configuration directory, defaults to `/etc/containerd/certs.d` |
| `Configs`    | `map[string]RegistryHostConfig` | Yes    | Registry configuration mapping; key is the registry name                                 |

**RegistryHostConfig field:**

| Parameter       | Type | Required | Description                             |
|----------------| --- |------|--------------------------------|
| `Host`         | `string` | Yes    | Registry host address                         |
| `Capabilities` | `[]string` | No    | Supported capabilities (e.g., pull, push, resolve) |
| `SkipVerify`   | `bool` | No    | Skip TLS verification                      |
| `PlainHTTP`    | `bool` | No    | Use HTTP instead of HTTPS               |
| `Insecure`     | `bool` | No    | Insecure connection                          |
| `OverridePath` | `bool` | No    | Override path                           |
| `TLS`          | `TLSConfig` | No    | TLS configuration                         |
| `Auth`         | `RegistryAuthConfig` | No    | Authentication configuration                           |
| `Header`       | `map[string][]string` | No    | Custom HTTP headers                     |

**TLSConfig field:**

| Parameter           | Type | Required | Description        |
|----------------------| --- |------|-----------|
| `CAFile`             | `string` | No    | CA certificate file path |
| `CertFile`           | `string` | No    | Client certificate file path |
| `KeyFile`            | `string` | No    | Client private key file path |
| `InsecureSkipVerify` | `bool` | No    | Skip TLS verification |

**RegistryAuthConfig field:**

| Parameter       | Type | Required | Description              |
|-----------------| --- |------|-----------------|
| `Auth`          | `string` | No    | Base64-encoded authentication string |
| `Username`      | `string` | No    | Username             |
| `Password`      | `string` | No    | Password              |
| `IdentityToken` | `string` | No    | Identity token            |
| `RegistryToken` | `string` | No    | Registry token            |

**containerd configuration file cct reference**

**my-containerd.yaml**

```yaml
apiVersion: bke.bocloud.com/v1beta1
kind: ContainerdConfig
metadata:
  name: my-containerd
  namespace: my-containerd
  labels:
    test: "true"
    component: "containerd"
spec:
  configType: combined
  description: "Custom containerd configuration example"

  script:
    content: |
      #!/bin/bash
      mkdir -p /tmp/log
      echo "Executing containerd setup script" > /tmp/log/test_containerd.log
    interpreter: "/bin/bash"

  service:
    execStart: "/usr/bin/containerd --config /etc/containerd/config.toml"
    slice: "system.slice"
    killMode: "process"
    restart: "always"
    restartSec: "5s"
    startLimitInterval: "10s"
    startLimitBurst: 5
    timeoutStopSec: "90s"
    logging:
      standardOutput: "journal"
      standardError: "journal"
      syslogIdentifier: "containerd"

  main:
    root: "/var/lib/containerd"
    state: "/run/containerd"
    sandboxImage: "registry.k8s.io/pause:3.9"
    metricsAddress: "127.0.0.1:1338"
    configPath: "/etc/containerd/certs.d"

  registry:
    configPath: "/etc/containerd/certs.d"
    configs:
      "my.harbor.com":
        host: "my.harbor.com"
        capabilities: ["pull", "resolve"]
        skipVerify: false
        plainHTTP: false
        tls:
          caFile: /etc/openFuyao/ca.crt
          insecureSkipVerify: false
        auth:
          username: "<username>"
          password: "<password>"

      "registry.k8s.io":
        host: "registry.aliyuncs.com/google_containers"
        capabilities: ["pull", "resolve"]
        skipVerify: true
```

> ![icon](../figures/icon-note.gif) **Note:**<br />
> - The application order is `spec.script` → `spec.service` → `spec.main` → `spec.registry`; containerd is automatically restarted upon completion.
> - `script.content` and `script.path` are mutually exclusive; if both are provided, `content` takes precedence.
> - If you need to fully customize `config.toml`, you can provide raw TOML in `spec.main.rawTOML` (when provided, it takes precedence over other `main` fields).
