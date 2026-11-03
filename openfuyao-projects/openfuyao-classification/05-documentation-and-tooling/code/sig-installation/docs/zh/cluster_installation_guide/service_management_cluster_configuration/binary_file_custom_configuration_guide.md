# 二进制组件自定义配置（Containerd和kubelet）

## 背景信息

本文档介绍如何使用 Custom Resource（CR）动态配置 Kubelet 和 Containerd 组件。在 BKECluster（简称 bc）中通过引用字段关联对应配置 CR：

| 缩写 | CR 类型 | kubectl 短名 | BKECluster 引用字段 |
| --- | --- | --- | --- |
| KCT | `KubeletConfig` | `kct` | `spec.KubeletConfigRef` |
| CCT | `ContainerdConfig` | `cct` | `spec.clusterConfig.cluster.containerdConfigRef` |

创建 KCT/CCT 后，在 BKECluster YAML 中填写引用即可在节点引导阶段自动下发配置，无需登录节点手工编辑文件。完整参数表见 [cluster-api-provider-bke 配置参数](../appendix/cluster-api-provider-bke_configuration.md)。

## 前提条件

- 管理集群（与 BKECluster 同集群）已部署 cluster-api-provider-bke 及对应 CRD。
- 在创建集群前准备好 KCT/CCT 资源，并在 BKECluster 中完成引用。

## 使用限制

- KCT/CCT 资源须与 BKECluster 位于同一管理集群。
- KCT 的 `spec.files[]` 支持绝对路径写入；请确保 CR 内容可信，避免写入敏感系统路径。
- CCT 的 `spec.configType` 支持 `service`、`main`、`registry`、`combined`（默认），仅渲染已声明的 `spec` 子段。

## 使用教程

### 在 BKECluster 中引用 KCT/CCT

在部署集群前，先创建 KCT/CCT 资源并在 BKECluster 中完成引用，再执行 `bke create cluster`。常用编辑命令如下：

```bash
# 查看 BKECluster / KubeletConfig / ContainerdConfig
kubectl get bc -A
kubectl get kct -A
kubectl get cct -A

# 编辑（将占位符替换为实际命名空间与名称）
kubectl edit bc -n <命名空间> <名称>
kubectl edit kct -n <命名空间> <名称>
kubectl edit cct -n <命名空间> <名称>

# 或通过 YAML 创建/更新后 apply
kubectl apply -f my-kubelet.yaml
kubectl apply -f my-containerd.yaml
```

在 BKECluster YAML 中引用示例如下：

```yaml
apiVersion: bke.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: bke-cluster
  namespace: bke-cluster
spec:
  KubeletConfigRef:          # 引用 KCT
    name: my-kubelet
    namespace: my-kubelet
  clusterConfig:
    cluster:
      containerdConfigRef:   # 引用 CCT
        name: my-containerd
        namespace: my-containerd
```

节点引导时，bkeagent 从管理集群读取 KCT/CCT 并应用到目标节点。

### KubeletConfig CR配置

#### 概述

`KubeletConfig`CR允许通过Kubernetes资源来动态配置Kubelet的配置文件和systemd服务文件，而无需手动编辑节点上的文件。

#### CR结构

```bash
apiVersion: bke.bocloud.com/v1beta1
kind: KubeletConfig
metadata:
  name: <config-name>
  namespace: <namespace>  # 默认为 bke-system
spec:
  kubeletConfig:
    kubelet.conf:
      raw: |
        # Kubelet 配置文件的 YAML 内容
  kubeletService:
    service:
      # Kubelet systemd 服务配置
```

**spec字段说明**

**`spec.kubeletConfig`**

用于配置Kubelet的配置文件(`/var/lib/kubelet/config.yaml`)。

| 参数                 | 类型 | 是否必选 | 描述                      |
|--------------------| --- |------|-------------------------|
| `kubelet.conf`     | `map[string]RawExtension` | 是    | 包含`raw`字段的配置对象        |
| `kubelet.conf.raw` | `string` | 是    | Kubelet配置文件的完整YAML内容 |

**`raw`字段格式：**

`raw`字段应包含完整的Kubelet配置文件YAML内容，例如：

```yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: systemd
clusterDNS:
  - 10.96.0.10
clusterDomain: cluster.local
...
```

kubelet.config实现对填写的kubelet.conf.raw的透传，需要由用户保证传入的配置项正确。

**`spec.kubeletService`**

用于配置Kubelet的systemd服务文件(`/etc/systemd/system/kubelet.service`)。

| 参数        | 类型 | 是否必选 | 描述                     |
|-----------| --- |------|------------------------|
| `service` | `KubeletService` | 否    | Kubelet systemd服务配置对象 |

**kubeletService字段说明：**

| 参数                   | 类型 | 是否必选 | 描述                               |
|----------------------| --- |------|----------------------------------|
| `ExecStart`          | `string` | 否    | 服务启动命令                           |
| `Restart`            | `string` | 否    | 重启策略(如：always，on-failure)     |
| `RestartSec`         | `string` | 否    | 重启间隔时间(如：5s，10s)              |
| `StartLimitInterval` | `string` | 否    | 启动限制时间窗口                         |
| `StartLimitBurst`    | `int` | 否    | 启动限制次数                           |
| `KillMode`           | `string` | 否    | 终止模式(如：control-group，process) |
| `StandardOutput`     | `string` | 否    | 标准输出重定向 (如：journal，null)       |
| `StandardError`      | `string` | 否    | 标准错误重定向                          |
| `SyslogIdentifier`   | `string` | 否    | 系统日志标识符                          |
| `WorkingDirectory`   | `string` | 否    | 工作目录                             |
| `User`               | `string` | 否    | 运行用户                             |
| `Group`              | `string` | 否    | 运行组                              |
| `ExecStartPre`       | `[]string` | 否    | 启动前执行的命令列表                       |
| `Environment`        | `[]string` | 否    | 环境变量列表 (格式：KEY=VALUE)           |
| `EnvironmentFile`    | `[]string` | 否    | 环境变量文件路径列表                       |
| `CustomExtra`        | `map[string]string` | 否    | 自定义额外配置项(键值对)                   |

kubelet.service是对kubelet启动时service支持的配置项。

**`spec.files[]`**

在 kubelet 配置落地前，向节点写入额外文件（相对路径基于 `/etc/kubernetes`，也支持绝对路径）。

| 参数 | 类型 | 是否必选 | 描述 |
| --- | --- | --- | --- |
| `path` | `string` | 是 | 目标文件路径 |
| `content` | `string` | 是 | 文件内容，支持变量替换 |
| `permissions` | `string` | 否 | 八进制权限，如 `0644` |
| `owner` | `string` | 否 | 文件属主，如 `root:root` |

**`spec.commands[]`**

在 kubelet 配置落地后执行额外命令。

| 参数 | 类型 | 是否必选 | 描述 |
| --- | --- | --- | --- |
| `command` | `string` | 是 | 命令或可执行脚本；`args` 为空时按 `sh -c` 执行 |
| `args` | `[]string` | 否 | 命令参数列表 |
| `workingDir` | `string` | 否 | 工作目录 |

**kubelet动态传值功能**

动态传值机制允许在配置文件中使用变量，这些变量会在部署时被实际值替换。这提供了极大的灵活性，使得同一份配置可以适用于不同的节点和环境。

**启用变量替换**

在`BKECluster`的部署过程中，系统会自动检测`KubeletConfigRef`，并启用变量替换功能。变量替换会在以下位置生效：

1. **kubelet.conf**配置
2. **kubelet.service**systemd服务文件
3. **spec.files[]**中的文件内容

表达式变量：`${EXPR|command|END}`

执行shell命令并将输出作为变量值。

**语法**：

```
${EXPR|shell_command|END}
```

**示例**：

```yaml
spec:  
  kubeletConfig:    
    kubelet.conf:      
      raw: |        
          apiVersion: kubelet.config.k8s.io/v1beta1
          kind: KubeletConfiguration
          # 获取 CPU 核心数
          cpuManagerPolicy: "static"
          reservedSystemCPUs: "${EXPR|nproc|END}"
          # 获取内存大小（MB）
          evictionHard:
            memory.available: "${EXPR|free -m | grep Mem | awk '{print $2}'|END}Mi"
          # 获取主机名
          hostnameOverride: "${EXPR|hostname|END}"
```

> ![说明](../figures/icon-note.gif)**说明：**<br />
>- 命令会在目标节点上执行
>- 命令执行失败时，变量会保持原样（不替换）
>- 命令输出会去除首尾空白字符

**kubelet配置文件kct参考**

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
          echo "=== KubeletConfig 后置命令 ===" >> /var/log/kubelet-custom.log
          echo "执行时间: $(date +%Y-%m-%d_%H:%M:%S)" >> /var/log/kubelet-custom.log
          if [ -f /var/lib/kubelet/config.yaml ]; then
            echo "kubelet config 已生成，大小: $(du -sh /var/lib/kubelet/config.yaml | awk '{print $1}')" >> /var/log/kubelet-custom.log
          else
            echo "警告：kubelet config 不存在" >> /var/log/kubelet-custom.log
          fi
      workingDir: "/var/log"
    - command: "/usr/bin/systemctl"
      args:
        - "is-active"
        - "containerd.service"
      workingDir: "/"

```

> ![说明](../figures/icon-note.gif)**说明：**<br />
> - 应用顺序为 `spec.files[]` → `spec.kubeletConfig` → `spec.kubeletService` → `spec.commands[]`。
> - `files[].path` 为相对路径时基于 `/etc/kubernetes`；也支持绝对路径（如 `/var/lib/kubelet/custom-flag.txt`）。
> - `files[].content` 支持 `${EXPR|command|END}` 与普通变量 `${VAR_NAME}` 替换。
> - `commands[]` 在 kubelet 配置落地后执行；`args` 为空时 `command` 按 `sh -c` 执行，多行脚本建议配合 `command` + `args: ["-c", "..."]` 使用。

**files 参考（可单独摘取）**

```yaml
  files:
    - path: "extra/kubelet-custom.conf"       # 相对路径 → /etc/kubernetes/extra/kubelet-custom.conf
      content: |
        clusterDomain: "cluster.local"
        nodeHostname: "${EXPR|hostname|END}"
      permissions: "0644"
      owner: "root:root"
    - path: "/var/lib/kubelet/custom-flag.txt" # 绝对路径
      content: |
        createdAt: "${EXPR|date +%Y-%m-%d_%H:%M:%S|END}"
      permissions: "0600"
      owner: "root:root"
```

**commands 参考（可单独摘取）**

```yaml
  commands:
    # 方式一：args 为空，command 按 sh -c 执行（适合单行命令）
    - command: "echo kubelet post-hook >> /var/log/kubelet-custom.log"

    # 方式二：command + args（适合多行脚本，推荐）
    - command: "/bin/bash"
      args:
        - "-c"
        - |
          echo "=== KubeletConfig 后置命令 ===" >> /var/log/kubelet-custom.log
          systemctl is-active containerd.service >> /var/log/kubelet-custom.log
      workingDir: "/var/log"

    # 方式三：直接调用可执行文件
    - command: "/usr/bin/systemctl"
      args:
        - "is-active"
        - "containerd.service"
      workingDir: "/"
```

### ContainerdConfig CR配置

#### 概述

`ContainerdConfig`CR允许通过Kubernetes资源来动态配置Containerd的主配置文件、服务文件和镜像仓库配置。

#### CR结构

```yaml
apiVersion: bke.bocloud.com/v1beta1
kind: ContainerdConfig
metadata:
  name: <config-name>
  namespace: <namespace>
spec:
  configType: combined   # service / main / registry / combined（默认）
  description: <说明>
  script:
    # 脚本配置
  service:
    # Containerd systemd 服务配置
  main:
    # Containerd 主配置文件 (config.toml)
  registry:
    # 镜像仓库配置
```

**`spec.configType`**

| 取值 | 说明 |
| --- | --- |
| `combined` | 默认值，按 `spec` 中已声明的子段依次渲染 |
| `service` | 仅渲染 systemd drop-in |
| `main` | 仅渲染 `config.toml` |
| `registry` | 仅渲染 `hosts.toml` |

**spec字段说明**

**`spec.script`**

用于在配置Containerd前执行自定义脚本。

| 参数            | 类型 | 是否必选 | 描述                                        |
|---------------| --- |------|-------------------------------------------|
| `Content`     | `string` | 否    | 脚本内容(内联脚本)                               |
| `Path`        | `string` | 否    | 脚本文件路径(本地文件)                             |
| `Interpreter` | `string` | 否    | 解释器(如：/bin/bash, /bin/sh)，默认为/bin/bash |
| `Args`        | `[]string` | 否    | 传递给脚本的参数列表                                |

> ![说明](../figures/icon-note.gif)**说明：**<br />`Content`和`Path`二选一，如果都提供则优先使用`Content`。

**`spec.service`**

用于配置Containerd的systemd服务drop-in文件(`/etc/systemd/system/containerd.service.d/10-override.conf`)。

| 参数                   | 类型 | 是否必选 | 描述               |
|----------------------| --- |------|------------------|
| `ExecStart`          | `string` | 否    | 服务启动命令           |
| `Slice`              | `string` | 否    | systemd slice单元 |
| `KillMode`           | `string` | 否    | 终止模式             |
| `Restart`            | `string` | 否    | 重启策略             |
| `RestartSec`         | `string` | 否    | 重启间隔时间           |
| `TimeoutStopSec`     | `string` | 否    | 停止超时时间           |
| `StartLimitInterval` | `string` | 否    | 启动限制时间窗口         |
| `StartLimitBurst`    | `int` | 否    | 启动限制次数           |
| `Logging`            | `LoggingConfig` | 否    | 日志配置             |
| `CustomExtra`        | `map[string]string` | 否    | 自定义额外配置项         |

**LoggingConfig字段：**

| 参数                 | 类型 | 是否必选 | 描述      |
|--------------------| --- |------|---------|
| `StandardOutput`   | `string` | 否    | 标准输出重定向 |
| `StandardError`    | `string` | 否    | 标准错误重定向 |
| `SyslogIdentifier` | `string` | 否    | 系统日志标识符 |
| `LogLevelMax`      | `string` | 否    | 最大日志级别  |

**`spec.main`**

用于配置Containerd的主配置文件 (`/etc/containerd/config.toml`)。

| 参数               | 类型 | 是否必选 | 描述                                        |
|------------------| --- |------|-------------------------------------------|
| `SandboxImage`   | `string` | 否    | Pod沙箱镜像 (如：pause镜像)                     |
| `Root`           | `string` | 否    | Containerd数据根目录，默认为`/var/lib/containerd` |
| `State`          | `string` | 否    | Containerd状态目录，默认为`/run/containerd`     |
| `ConfigPath`     | `string` | 否    | 配置文件路径，默认为`/etc/containerd/certs.d`      |
| `MetricsAddress` | `string` | 否    | 指标监控地址                                  |

> ![说明](../figures/icon-note.gif)**说明：**<br />
>如果指定了`main`字段，系统会根据模板生成`config.toml`文件。如果未指定，则使用默认配置。

**`spec.registry`**

用于配置镜像仓库的`hosts.toml`文件。

| 参数           | 类型 | 是否必选 | 描述                                            |
|--------------| --- |------|-----------------------------------------------|
| `ConfigPath` | `string` | 否    | hosts.toml配置目录，默认为`/etc/containerd/certs.d` |
| `Configs`    | `map[string]RegistryHostConfig` | 是    | 仓库配置映射，键为仓库名称                                 |

**RegistryHostConfig字段：**

| 参数             | 类型 | 是否必选 | 描述                             |
|----------------| --- |------|--------------------------------|
| `Host`         | `string` | 是    | 仓库主机地址                         |
| `Capabilities` | `[]string` | 否    | 支持的能力(如：pull，push，resolve) |
| `SkipVerify`   | `bool` | 否    | 跳过TLS验证                      |
| `PlainHTTP`    | `bool` | 否    | 使用HTTP而非HTTPS               |
| `Insecure`     | `bool` | 否    | 不安全连接                          |
| `OverridePath` | `bool` | 否    | 覆盖路径                           |
| `TLS`          | `TLSConfig` | 否    | TLS配置                         |
| `Auth`         | `RegistryAuthConfig` | 否    | 认证配置                           |
| `Header`       | `map[string][]string` | 否    | 自定义HTTP头                     |

**TLSConfig字段：**

| 参数                   | 类型 | 是否必选 | 描述        |
|----------------------| --- |------|-----------|
| `CAFile`             | `string` | 否    | CA证书文件路径 |
| `CertFile`           | `string` | 否    | 客户端证书文件路径 |
| `KeyFile`            | `string` | 否    | 客户端私钥文件路径 |
| `InsecureSkipVerify` | `bool` | 否    | 跳过TLS验证 |

**RegistryAuthConfig字段：**

| 参数              | 类型 | 是否必选 | 描述              |
|-----------------| --- |------|-----------------|
| `Auth`          | `string` | 否    | Base64编码的认证字符串 |
| `Username`      | `string` | 否    | 用户名             |
| `Password`      | `string` | 否    | 密码              |
| `IdentityToken` | `string` | 否    | 身份令牌            |
| `RegistryToken` | `string` | 否    | 仓库令牌            |

**containerd配置文件cct参考**

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
  description: "自定义 containerd 配置示例"

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
          username: "<用户名>"
          password: "<密码>"

      "registry.k8s.io":
        host: "registry.aliyuncs.com/google_containers"
        capabilities: ["pull", "resolve"]
        skipVerify: true
```

> ![说明](../figures/icon-note.gif)**说明：**<br />
> - 应用顺序为 `spec.script` → `spec.service` → `spec.main` → `spec.registry`，完成后自动重启 containerd。
> - `script.content` 与 `script.path` 二选一；均提供时优先使用 `content`。
> - 若需完全自定义 `config.toml`，可在 `spec.main.rawTOML` 中提供原始 TOML（提供后优先于其他 `main` 字段）。

