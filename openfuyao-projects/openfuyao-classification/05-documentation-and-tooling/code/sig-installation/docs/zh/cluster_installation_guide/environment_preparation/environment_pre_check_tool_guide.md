# 环境前置检查

本章节介绍 openFuyao 安装部署前可用的环境前置检查工具：

- **bke 内置统一前置检查（preflight）**：v26.09 起内置于 bke 安装工具，使用与安装命令相同的输入执行只读检查并输出可修复报告，是推荐的安装前检查入口。
- **envCheck**：独立的社区环境检查工具，提供残留文件查询/清理、远程批量分发检查等泛化检查能力。

按照版本计划，后续版本 bke 内置统一前置检查将逐步替代 envCheck：v26.09 起安装前检查入口切换为 `bke preflight`，bke preflight当前版本是 alpha 阶段特性，可能存在不稳定情况。两个工具的用户文档分别见下文对应章节。

## bke 内置统一前置检查

### 背景信息

`bke init` 和 `bke cluster create` 在真正安装前依赖业务输入、管理集群、本机和目标节点的多类条件。现有错误通常在安装过程中才暴露，且网络、认证、权限、资源和配置问题容易被混在一起，增加定位和恢复成本。

为此，bke 从 v26.09 起内置统一前置检查（preflight）。它在执行安装命令前，使用与安装命令**完全相同的输入**生成检查计划、只读采集事实、按版本化发布规则计算结果，并输出可修复报告：

```bash
# 在执行 bke init 之前
bke preflight init [与 bke init 相同的业务参数]

# 在执行 bke cluster create 之前
bke preflight cluster-create \
  --file ./bkecluster.yaml \
  --nodes ./bkenodes.yaml
```

前置检查具有以下特点：

- **只读**：不修改主机、节点或 Kubernetes 资源，不创建目录或测试文件，不安装、修复、清理组件。Kubernetes 侧只使用 Discovery、GET、LIST；SSH 侧只执行白名单只读命令。
- **与安装命令共享输入**：`preflight init` 与 `bke init` 复用同一份参数定义、默认值和校验逻辑；`preflight cluster-create` 与 `bke cluster create` 使用相同的 cluster/nodes 文件和管理集群 kubeconfig，检查的就是本次安装实际会用到的值。
- **规则可追溯**：固定规则内嵌于 bke 二进制的发布 Bundle 中，每条规则都有明确的来源证据（发布配置、Provider 校验、manifest 或安装文档），并受摘要校验保护。
- **结果可修复**：每条 FAIL/WARN 都说明检查对象、期望值、实际值、原因和修复建议；输入错误、环境不满足和框架故障使用不同结论和退出码区分。
- **不缓存状态**：两次命令之间输入和环境可能变化，PASS 只表示报告中适用检查在检查时满足，不保证后续安装必然成功。

### 前提条件

- 使用包含 `preflight` 子命令的 bke 二进制工具（v26.09 及以上版本），运行环境为 Linux amd64 或 Linux arm64。
- `preflight cluster-create` 需要：
  - 已参考[在线安装引导集群](../bootstrap_cluster_installation/online_bootstrap_cluster_installation.md)或[离线安装引导集群](../bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md)成功部署引导节点，或参考[安装管理集群](../management_cluster_installation/management_cluster_deployment_guide.md)成功部署管理集群。
  - 已准备好创建集群所需的集群配置文件（`--file`）和节点配置文件（`--nodes`），两者格式与 `bke cluster create` 使用的完全相同，编写示例参见[通过后台（命令行）安装业务集群](../service_cluster_installation/service_cluster_deployment_guide_(linux_backend).md)。
  - 目标节点允许 SSH 登录（默认 22 端口，可按节点配置）。
- 如需使用 `--preflight-config` 外部策略，应准备与当前 bke 版本匹配的完整 `preflight.yaml`（参见[配置外部策略](#配置外部策略)）。

### 使用限制

- 仅覆盖 `init` 和 `cluster-create` 两个安装阶段，不覆盖 build、Web、升级和扩缩容。
- 只做有明确 BKE/Provider/安装要求来源的检查。历史 envCheck 的部分泛化检查（残留文件、时钟同步、默认路由、iptables、SELinux、Swap 等）不在内置检查范围内，如需相关能力请参见 [envCheck](#envcheck)。
- 不自动执行安装命令；用户处理 FAIL、确认 WARN 后，需显式执行真正的业务命令。
- init 和 cluster-create 阶段的来源连通性均只验证可达性，不代表认证或制品下载成功。
- 当前版本 SSH 检查不校验 host key，只能证明会话建立，应在可信网络中执行。
- 不提供 `--request`、Bundle 覆盖或 Provider defaults 文件覆盖参数。

### 使用教程

#### 命令总览

```bash
# 查看帮助
bke preflight -h
bke preflight init -h
bke preflight cluster-create -h
```

`preflight` 是独立、只读的顶层子命令，包含两个阶段子命令：

| 命令 | 说明 |
| ---- | ---- |
| `bke preflight init` | 在 `bke init` 前检查引导节点（身份、平台、资源、工具、工作目录、端口、来源、离线制品、自定义仓库和 CA） |
| `bke preflight cluster-create` | 在 `bke cluster create` 前检查输入、目标节点（SSH、资源、平台、角色端口）和来源 |

#### 运行参数说明

`preflight` 命令的公共运行参数：

| 参数名称 | 描述 |
| ---- | ---- |
| `--output` | 输出格式，支持 `text` 和 `json`，默认为 `text`，可选。 |
| `--output-file` | 将完整报告写入指定文件，可选；不指定时，text 模式会自动在系统临时目录生成详细报告文件（`bke-preflight-<阶段>-*.txt`）并在终端提示路径。 |
| `--preflight-config` | 完整外部 `preflight.yaml` 策略文件路径，可选；不指定时使用二进制内嵌的默认策略。 |
| `--timeout` | 完整前置检查总超时时间，可选；默认 **10min**，覆盖输入解析、管理集群读取、采集、计算等阶段，必须大于 0。 |
| `--fact-timeout` | 单个事实探测超时时间，可选；默认 `30s`，必须大于 0。 |
| `--max-concurrency` | 事实探测最大并发数，可选；默认 `0`（有界自适应），合法范围为 0~64。 |

`preflight init` 的业务参数与 `bke init` 完全一致（常用参数如下，完整列表请参见 `bke preflight init -h`）：

| 参数名称 | 描述 |
| ---- | ---- |
| `-f, --file` | init 输入文件（bkecluster.yaml），可选。 |
| `--hostIP` | 引导节点自身 IP，可选；未提供时自动探测本机出方向 IP。 |
| `--onlineImage` | 在线安装镜像地址，可选。 |
| `--imageFilePath` | 离线安装包路径，可选。 |
| `--installConsole` | 是否安装 Console，可选；默认 `true`，为 `true` 时才检查 Console 端口。 |
| `--enableNTP` / `--ntpServer` | NTP 开关与 NTP 服务器，可选；默认启用，本机提供 NTP（`local`）时才检查本地 UDP 123 端口。 |
| `--otherRepo` / `--otherSource` / `--otherChart` | 自定义镜像、软件包、Chart 仓库地址，可选。 |
| `--imageRepoCAFile` | 镜像仓库 CA 证书文件路径，可选；配置后检查文件存在、可读且可解析为 PEM X.509 证书。 |

`preflight cluster-create` 的业务参数：

| 参数名称 | 描述 |
| ---- | ---- |
| `-f, --file` | BKECluster 集群配置文件，必填。 |
| `-n, --nodes` | BKENodeList 节点配置文件，必填。节点文件中的 IP、SSH 端口、角色、凭据等信息与 `bke cluster create` 使用同一格式。 |

#### 检查项说明

内置检查由发布策略（`configs/preflight/preflight.yaml`）和内嵌 Bundle 共同决定，按本次输入的实际计划动态实例化：不适用的检查不会产生结果，只有适用且被依赖、超时或取消阻塞的检查才会显示为 SKIP。同一根因触发的多条下游检查会显示为 SKIP 并引用根因，避免一个问题扩散成多条 FAIL。

> ![](../figures/icon-note.gif)**说明：**
> **表1**和**表2**中"是否阻塞"指该检查不满足时的**结论级别**，不是命令行为：
> - **是**：检查失败或事实无法采集时，结论为 FAIL（退出码 3）。该条件是安装的硬性前提，必须修复后重试；直接执行安装命令预期会失败。
> - **否**：结论为 WARN（退出码 2）。该条件是风险提示，人工确认后可以继续。
> - preflight 与安装命令相互独立，FAIL 并不会阻止执行 `bke init` / `bke cluster create`；除被依赖阻塞的检查外，所有适用检查都会执行完。下游检查显示 SKIP 属于依赖去噪机制（根依赖 FAIL 或 WARN 均会触发），与本列含义无关。

#### init 阶段前置检查

- 执行如下命令，使用与 `bke init` 相同的业务参数检查引导节点：

  ```bash
  sudo bke preflight init \
    --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest \
    --installConsole=false
  ```

- 使用离线安装包时：

  ```bash
  sudo bke preflight init \
    --imageFilePath /root/bke-offline.tar.gz \
    --installConsole=false
  ```

- 使用自定义仓库和 CA 证书时，将实际地址追加到命令后面：

  ```bash
  sudo bke preflight init \
    --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest \
    --otherRepo https://registry.example.com:40443/openfuyao \
    --otherSource http://repo.example.com:40080 \
    --otherChart https://chart.example.com:443/chart \
    --imageRepoCAFile /etc/openFuyao/ca.crt
  ```

**表1** init 阶段主要检查项表（对象均为引导节点 bootstrap）：

| 类别 | 检查内容 | 默认要求 | 是否阻塞 |
| ---- | ---- | ---- | ---- |
| 执行身份 | 本地执行用户 | UID 为 0（root） | 是 |
| 平台 | 操作系统、发行版、内核、架构 | Linux；openEuler 20.03/22.03/24.03 或 Ubuntu 22.04 或 UOS 20 1070e；内核匹配发行版要求（如 openEuler 22.03 需 5.10 内核）；amd64 或 arm64 | 是 |
| 资源 | 逻辑 CPU、可用内存、系统盘空间 | 不少于 2 核、4 GB、100 GB | 是 |
| 工具 | `tar`、`systemctl` | 在 PATH 中存在且可执行 | 是 |
| 工作目录 | 工作空间（默认 `/bke`，可通过 `BKE_WORKSPACE` 环境变量覆盖） | 已存在或父路径允许创建；所在文件系统非只读挂载 | 是 |
| 端口 | Kubernetes 36443、镜像仓库 40443、Chart 仓库 38080、软件包仓库 40080；Console 30010（启用 Console 时）；本地 NTP 123/UDP（本机提供 NTP 时） | 实际端口空闲（显式传入端口时按传入值检查） | 是 |
| 来源 | 本次实际使用的安装来源（在线镜像、自定义仓库等） | 可建立到脱敏 host:port 的 TCP 连接 | 是 |
| 离线制品 | 离线安装包存在 → 可读 → 格式 | 存在；可读；为 tar、tar.gz 或 tgz | 是 |
| 自定义仓库 | 自定义仓库地址、CA 证书 | URL/域名/IP/端口可解析；CA 文件存在、可读、可解析为 PEM X.509 证书 | 是 |

- 检查通过、处理报告中的问题后，再执行如下命令部署引导节点：

  ```bash
  sudo bke init \
    --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest \
    --installConsole=false
  ```

#### cluster-create 阶段前置检查

- 执行如下命令，使用与 `bke cluster create` 相同的集群、节点文件和管理集群 kubeconfig：

  ```bash
  sudo bke preflight cluster-create \
    --file /root/openfuyao/bkecluster.yaml \
    --nodes /root/openfuyao/bkenodes.yaml
  ```

- 检查通过、处理报告中的问题后，再执行如下命令部署集群：

  ```bash
  sudo bke cluster create \
    --file /root/openfuyao/bkecluster.yaml \
    --nodes /root/openfuyao/bkenodes.yaml
  ```

- 节点较多时，可调整并发数加速检查：

  ```bash
  sudo bke preflight cluster-create \
    --file /root/openfuyao/bkecluster.yaml \
    --nodes /root/openfuyao/bkenodes.yaml \
    --max-concurrency 16  # 16 为示例值，表示最大并发数；取值范围 0~64，默认 0（自适应）
  ```

**表2** cluster-create 阶段主要检查项表：

| 类别 | 检查内容 | 默认要求 | 是否阻塞 |
| ---- | ---- | ---- | ---- |
| 输入唯一性 | 节点 IP、hostname | 非空 IP 不重复；非空 hostname 不重复 | 是 |
| 拓扑 | master 数量、etcd 数量、worker 能力数量 | master 不少于 1 且为奇数；etcd 不少于 1；worker（含 master+worker）不少于 1 | 是 |
| 管理集群 | API 可连接 → Discovery 可读 → CRD 已注册；`kube-system/localkubeconfig` Secret 可 GET → 数据非空 → kubeconfig 可解析 → server endpoint 语法有效 | BKECluster/BKENode CRD 已注册；Secret 链路完整 | 是 |
| 本地输入 | 集群文件、节点文件的 schema 和业务校验 | 通过（错误定位到具体字段路径或文件行号） | 是 |
| 本地路径 | 集群配置输出路径 | 已存在或父路径允许创建；可访问；非只读挂载 | 是 |
| 网络 | Pod subnet、Service subnet | CIDR 语法有效（仅校验语法，不推测网段重叠） | 是 |
| HA 与 addon | `kubernetesVersion` 版本引用；addons（BKECluster `clusterConfig.addons` 声明的 yaml/chart 形态扩展组件）配置合法性；chart 类型 addon 的来源配置；引导节点与 HA master 共部署告警（需可证明引导节点身份且 master 数量大于 1 才实例化） | 版本为合法 semver 且不低于 v1.21.0（不设上限）；addon 的 name+version 组合唯一、type 仅允许 yaml/chart（留空默认 yaml）；chart 类型 addon 必须已配置 chartRepo（仅验证配置完整，仓库可达性由"来源"行负责）；共部署告警提示引导节点同时充当 HA master 的容量与故障域风险 | 版本引用、addon 为是；HA 共部署提示为否（仅 WARN，需人工确认，当前版本默认不实例化） |
| SSH（每个节点） | transport（纯 TCP）→ 认证 → 权限 → systemd | TCP 可连接；凭据可建立会话；root 或免密 sudo 可用；systemd 可用 | 是 |
| 节点资源 | 逻辑 CPU、可用内存、系统盘空间 | 每节点不少于 8 核、16 GB、100 GB | 是 |
| 节点平台 | OS、发行版、内核 | 与 init 阶段平台矩阵一致 | 是 |
| 角色端口 | master：apiserver 6443、controller-manager 10257、scheduler 10259、本地 HA 36443（多 master 时）；master/etcd：etcd client 2379、peer 2380、metrics 2381；master/worker：kubelet 10250、kubelet health 10248；所有节点：BKE Agent 58080 | 实际端口空闲（业务文件中显式配置的端口优先） | 是 |
| 来源 | 每个节点到实际安装来源 | 通过该节点的 SSH 会话执行 curl HEAD 探测，使用节点自己的 DNS 和 `/etc/hosts` 解析；仓库同时配置域名和 IP 时，连接 IP 并保留域名作为 Host/SNI | 是 |

> ![](../figures/icon-note.gif)**说明：**
> - 远端来源连通性由目标节点通过 curl 发起 HEAD 探测，要求目标节点预装 curl；探测不校验证书（自签名证书不影响结果），HTTP 401/403/404 等错误状态仍视为可达；私有域名必须在每个节点上可解析。
> - 管理集群 Provider 默认配置读取失败时，检查会继续使用静态默认值，同时报告 `CLUSTER-MGMT-PROVIDER-DEFAULTS` 问题。
> - **表2**中"是否阻塞"的含义与 init 阶段检查项说明一致：FAIL（退出码 3）为安装硬性前提，须修复后重试；WARN（退出码 2）为风险提示，人工确认后可继续。该列不影响后续检查项的执行。

#### 配置外部策略

默认策略与匹配 Bundle 一起内嵌在 bke 二进制中。需要调整阈值、超时或展示信息时，可以复制完整默认策略、修改后通过 `--preflight-config` 显式加载，无需重新构建 bke：

```bash
# 从 bkeadm 仓库复制与当前 bke 相同修订的完整默认文件
# （先通过 bke version 记录 version 和 gitCommitID，再取对应修订的 configs/preflight/preflight.yaml）
cp preflight.yaml /etc/openFuyao/preflight.yaml

sudo bke preflight --preflight-config /etc/openFuyao/preflight.yaml \
  init --hostIP 192.0.2.10  # 192.0.2.10 为示例 IP，请替换为实际引导节点 IP

sudo bke --kubeconfig /path/to/kubeconfig \
  preflight --preflight-config /etc/openFuyao/preflight.yaml \
  cluster-create -f bkecluster.yaml -n bkenodes.yaml
```

**默认策略文件内容**

以 v26.09 版本为例，默认策略文件 `configs/preflight/preflight.yaml` 的完整内容如下（其中 `metadata.release` 等取值以与当前 bke 修订匹配的实际发布文件为准）：

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

**配置项说明**

顶层字段：

| 字段 | 说明 |
| ---- | ---- |
| `apiVersion` | 策略 API 版本，固定为 `preflight.openfuyao.io/v1`。 |
| `kind` | 文档类型，固定为 `PreflightConfig`。 |
| `metadata.release` | 策略对应的发布版本，必须与当前 bke 内嵌 Bundle 的 release 完全一致，否则策略无法加载。 |
| `metadata.description` | 策略的可读描述，仅用于说明，不影响检查行为。 |
| `operations` | 各安装阶段的检查策略，必须保留 `init` 和 `clusterCreate` 两个阶段。 |

各检查项条目可用的公共字段：

| 字段 | 适用范围 | 说明 |
| ---- | ---- | ---- |
| `description` | 所有检查项 | 检查项描述，必填，作为报告中的检查项名称展示。 |
| `when` | 端口检查 | 检查的实例化条件，取值为安装计划中的布尔键（如 `feature.installConsole` 表示本次启用 Console、`feature.localNTP` 表示本机提供 NTP、`cluster.topology.multiMaster` 表示多 master），为真时该检查才适用于本次输入。 |
| `roles` | cluster-create 的端口检查 | 检查适用的节点角色，支持 `master`、`worker`、`etcd`、`all`（`all` 不能与其他角色组合）。 |

`operations.init` 检查族：

| 检查族/字段 | 说明 |
| ---- | ---- |
| `platform.distributions` | 支持的操作系统矩阵，每项由 `name`（发行版）、`version`（版本）、`architectures`（架构）、`kernelPrefixes`（允许的内核版本前缀）组成；必须与发布策略完全一致，不能扩缩支持范围。 |
| `resources[].key` | 资源类型，仅支持 `cpu`、`memory`、`systemDisk`。 |
| `resources[].minimum` | 资源最低阈值：`cpu` 为核数；`memory`、`systemDisk` 为字节数整数或带 `B`/`K`/`M`/`G`/`T` 后缀的字符串（1024 进制）。 |
| `tools[].key` / `tools[].names` | 要求存在于 PATH 中的可执行文件（`tar`、`systemctl`）；`names` 只能为单个可执行文件名，不能包含路径。仅 init 阶段支持。 |
| `paths[].key` | `workspace` 为内置工作空间检查，配合 `require` 使用；其余 key 需通过 `path` 指定干净的绝对路径，仅检查路径存在性，不支持 `require`。仅 init 阶段支持。 |
| `paths[].require` | 工作空间要求，支持 `create`（自身或父路径允许创建）、`notReadOnly`（所在文件系统非只读挂载）。 |
| `ports[].key` | 端口检查标识，仅支持固定 key：`kubernetes`、`imageRepo`、`chart`、`yum`、`console`、`ntp`。 |
| `ports[].defaultPort` | 默认端口（1~65535）；实际检查端口优先取 `portFrom` 指向的业务输入值，未显式传入时才使用默认值。 |
| `ports[].portFrom` | 实际端口取值来源的安装计划键（如 `port.kubernetes`），各端口的取值固定，不能修改。 |
| `ports[].protocol` | 端口协议，仅支持 `tcp`、`udp`；`ntp` 必须为 `udp`，其余端口必须为 `tcp`。 |

`operations.clusterCreate` 检查族：

| 检查族/字段 | 说明 |
| ---- | ---- |
| `platform.distributions` | 与 init 阶段一致的平台矩阵，必须与发布策略完全一致。 |
| `topology.masters` | master 数量要求：`minimum` 为最小数量，`odd` 要求为奇数且必须保持启用。 |
| `topology.etcd` / `topology.workers` | etcd、worker 的最小数量（`minimum`）。 |
| `topology.unique` | 节点唯一性校验字段，仅支持 `hostname`、`ip`。 |
| `managementCluster.api` / `managementCluster.discovery` | 管理集群 API 可达与 Discovery 可读检查，必须保持启用。 |
| `managementCluster.crds` | 要求已注册的 CRD 列表，仅支持 `bkeclusters`、`bkenodes`。 |
| `managementCluster.requiredKubeconfig` | 管理集群 kubeconfig 的 Secret 引用，仅支持 `kube-system/localkubeconfig`。 |
| `resources` | 节点资源阈值，字段含义与 init 阶段一致（`cpu`、`memory`、`systemDisk`）。 |
| `ports[].key` | 角色端口检查标识，仅支持固定 key：`apiServer`、`etcdClient`、`etcdPeer`、`etcdMetrics`、`kubelet`、`kubeletHealth`、`controllerManager`、`scheduler`、`localHA`、`bkeAgent`。 |
| `ports[].roles` | 端口检查适用的节点角色；`apiServer` 的实际端口通过 `portFrom: cluster.apiPort` 取自业务文件，其余端口使用 `defaultPort`。 |

外部策略的规则：

- 必须是**一份完整的策略文件**，不与内置 YAML 做字段级合并。
- 顶层 `apiVersion: preflight.openfuyao.io/v1`、`kind: PreflightConfig`、`metadata.release`和 `operations.init`、`operations.clusterCreate` 必须保留。
- 默认策略已有的固定配置键（平台、端口、资源、路径、工具、拓扑、管理集群）必须保留，不能通过删除列表关闭发布检查。
- 资源 `minimum` 支持字节数整数或带 `B`/`K`/`M`/`G`/`T` 后缀的字符串（1024 进制，如 `4294967296` 或 `4G`）；调整阈值后，报告中的修复建议会按实际阈值显示。
- 平台矩阵必须与发布策略完全一致，不能通过外部文件扩缩支持范围。
- 端口的 `defaultPort`、`protocol`、`portFrom`、`when`、`roles` 固定；实际业务端口应通过安装命令或集群、节点配置传入（显式 CLI/业务文件值优先于 policy 的 `defaultPort`）。
- 外部策略不能替换固定规则 ID、Fact、operator、依赖、SourceLock、发布事实、Collector 契约或 Provider defaults。
- 外部策略只影响本次检查，不回写后续安装命令的参数。

违反上述任一规则（空文件、多文档、未知字段、release 不匹配、删除固定键、字段类型不合法等）时，前置检查不会启动，并在报告中给出具体的配置路径（`PREFLIGHT-CONFIG-INVALID`）。

### 输出说明

#### 终端输出与详细报告

- 运行过程中，进度信息写入标准错误流 stderr（如 `前置检查进度：采集事实 3/12`），并有 5 秒心跳提示，慢节点或卡住的探测不会表现为命令挂起。
- 默认 `text` 模式下，终端展示运行结论、运行问题和需要处理的 FAIL/WARN，使用"状态、检查项、目标、期望、实际、原因及建议"六列表格；窄于 72 列的终端自动降级为纵向布局。
- text 模式会自动在系统临时目录生成 UTF-8 详细报告（`bke-preflight-<阶段>-*.txt`），并在终端末尾提示路径；详细报告包含 PASS、FAIL、WARN 和全部 SKIP（含依赖阻塞原因），不含终端颜色，便于归档。可通过 `--output-file` 指定固定归档路径。
- `json` 模式下，stdout 输出单一 JSON 文档（含 `overall`、`summary`、`issues`、`results`），进度只写 stderr，可直接供脚本和流水线解析。
- 报告包含 `planHash`、`configDigest`、`bundleDigest`，需要审计时应与完整报告一起保存。报告可能包含节点地址、路径和来源定位信息，分享前应按组织要求检查脱敏。

#### 检查状态与总体结论

单个检查项的公开状态只有四种：

| 状态 | 含义 |
| ---- | ---- |
| PASS | 满足期望。 |
| FAIL | 不满足期望，需要修复后重试。 |
| WARN | 不满足期望但仅为提示，需要人工确认。 |
| SKIP | 因依赖失败、超时、取消或策略原因跳过，会引用最早根因。 |

#### 退出码

| 退出码 | 总体结论 | 含义 |
| ---- | ---- | ---- |
| 0 | PASS | 所有适用检查通过。 |
| 2 | WARN 或 SKIP | 存在 WARN，或无适用项/全部跳过；需要人工确认。 |
| 3 | FAIL | 存在环境检查失败项。 |
| 4 | INPUT_ERROR 或 BUNDLE_ERROR | 输入、外部策略或内嵌 Bundle 错误，前置检查未启动。 |
| 5 | INTERNAL_ERROR | SDK、Adapter 或 Collector 内部错误。 |
| 124 | TIMED_OUT | 总体超时，部分检查未完成。 |
| 130 | CANCELED | 用户取消。 |

### 常见问题

#### 提示 PREFLIGHT-OPTIONS

运行参数非法。请确认：`--output` 为 `text` 或 `json`；`--timeout` 和 `--fact-timeout` 大于 0；`--max-concurrency` 在 0~64 范围内。

#### 提示 PREFLIGHT-CONFIG-INVALID

外部策略文件无法通过校验。常见原因：文件为空、包含多个 YAML 文档、存在未知字段、`metadata.release` 与当前 bke 不匹配、删除了固定配置键，或 `customChecks` 字段不合法。请从与当前 bke 相同修订的完整默认文件重新开始，只修改受支持的字段后重试。

#### 提示 INIT-INPUT-* 或 CLUSTER-INPUT-*（退出码 4）

业务输入不合法。请按报告中的字段路径（如 `cluster.spec.clusterConfig.cluster.networking.podSubnet`、`nodes[0].spec.ip`）或文件行号修复输入，并以与安装命令一致的参数重新运行。缺 `--hostIP`、`--file` 或 `--nodes` 也属于此类。

#### 提示 CLUSTER-MGMT-PROVIDER-DEFAULTS

无法读取管理集群 Provider 默认配置（如 kubeconfig 无效、API 不可达或无权读取 ConfigMap `cluster-system/bke-config`）。检查会继续使用静态默认值，但建议恢复管理集群连接并授予只读权限后重试。

#### SSH 检查失败或大量 SKIP

SSH 链路按 transport → 认证 → 权限分层，根依赖失败后下游全部 SKIP 并引用根因。请先解决报告中第一条 SSH 失败（网络连通、凭据、root/免密 sudo），不要把多条 SKIP 当作多个独立问题处理。注意凭据错误不会被误报为网络失败。

#### 来源检查通过但安装时拉取失败

init 阶段来源检查只证明 TCP 连通；cluster-create 阶段的 HEAD 探测不校验证书、且把 HTTP 错误状态视为可达，均不代表认证或制品下载成功。请按安装报错继续排查仓库服务、证书和凭据。

#### 命令运行超时（退出码 124）

节点不可达或环境过慢时可能触发总超时。恢复网络后重试，或适当增大 `--timeout`；单个探测卡住时可调整 `--fact-timeout`。

#### 退出码为 2 但没有 FAIL

退出码 2 表示存在 WARN 或总体结论为 SKIP（无适用检查或全部跳过）。请查看详细报告中 WARN 的原因及建议并人工确认；SKIP 项会注明被阻塞的根因。

#### 检查结束后没有生成详细报告

详细报告仅在 `text` 模式下自动生成（写入系统临时目录）；`json` 模式请通过 `--output-file` 指定输出文件。

## envCheck

> ![img](../figures/icon-notice.gif)**注意：**
> 因 v26.09 版本提供了 bke 内置统一前置检查（preflight），该工具与安装命令共享输入、检查能力更全面，将逐步替代 envCheck。过渡期内如需残留文件查询/清理、远程批量分发等 envCheck 独有能力，可继续参考本章，新环境请优先使用 [bke 内置统一前置检查](#bke-内置统一前置检查)。

### 背景信息

用户在使用安装部署工具的过程中发现，由于环境中可能存在其他软件残留的配置文件，以及部分和规格中冲突的应用程序，导致安装部署过程异常或失败。为了保障安装部署的成功运行，需要构建一个前置环境检查工具，确保安装部署之前，环境是可用的。目前工具已实现以下功能。

- 给定文件路径列表，查询当前环境中是否残留文件，并输出检查报告。
- 给定文件路径列表，支持安全删除（询问用户是否确认删除）和强制删除（直接删除不询问）文件，并输出检查报告。
- 检查环境中是否安装指定程序，并输出检查报告。
- 支持多机器之间的时间同步校验，检测其余节点和引导节点之间时钟是否同步。
- 检查操作系统内核版本是否符合配置要求。
- 检查本地端口占用情况，支持按角色配置端口列表。
- 检查磁盘剩余空间是否满足配置要求，支持按角色过滤。
- 检查当前节点是否存在默认路由。
- 检查 iptables FORWARD 链默认策略是否为 ACCEPT。
- 支持远程批量分发检查，将工具分发到多台远程节点并行执行检查，统一收集结果并生成汇总报告。

以上功能推荐使用远程批量分发检查，其余功能在需要进行特定检查时使用即可。

### 前提条件

本工具目前只支持在`Linux amd64`和`Linux arm64`环境下执行，请根据环境自行下载对应版本的工具。获取工具和配置文件（配置文件示例请参见[全量配置文件示例](#全量配置文件示例)）有以下三种方式，三种方式效果相同，用户可自行选择：

- 通过下载链接直接获取工具和默认配置文件，配置文件可以自行新建或者下载默认配置文件进行修改。

  |下载项|amd64|arm64|
  |------|-----|-----|
  |envCheck工具|[下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/envCheck)|[下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/envCheck)|
  |默认配置文件|[下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/config.json)|[下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/config.json)|

- 在 Linux 环境下执行以下命令直接下载工具和配置文件：

  ```bash
  # amd64 架构
  wget -O envCheck https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/envCheck
  wget -O config.json https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/config.json
  chmod +x envCheck
  ```

  ```bash
  # arm64 架构
  wget -O envCheck https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/envCheck
  wget -O config.json https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/config.json
  chmod +x envCheck
  ```

- 从源码构建，源码位于 [openfuyao/env-check](https://gitcode.com/openfuyao/env-check) 代码仓。克隆源码后执行以下命令构建：

  ```bash
  git clone https://gitcode.com/openfuyao/env-check.git
  cd env-check

  # 构建当前架构
  go build -o envCheck ./cmd

  # 跨架构构建（远程分发场景需要）
  GOOS=linux GOARCH=amd64 go build -o build/envCheck_amd64 ./cmd
  GOOS=linux GOARCH=arm64 go build -o build/envCheck_arm64 ./cmd
  ```

### 使用限制

无。

### 使用教程

> ![](../figures/icon-note.gif)**说明：**
> - 下载后工具名为`envCheck`，且是二进制可执行文件。
> - 可以通过修改配置文件和执行不同的命令使用工具的不同功能。相应的配置文件需要和工具在同一文件夹下（如果文件夹未在PATH中，执行后续教程中的命令时需要使用./envCheck）且名为`config.json`（配置文件的位置可以自行放置，但是放置在其余位置时，需要使用命令行参数`--config`或`-c`显式指定），这里默认在同一文件夹下且内容为下面示例中展示的内容。

#### 全量配置文件示例

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

> ![img](../figures/icon-notice.gif)**注意：**
> 关于配置文件，工具会进行以下校验。
> - 使用文件查询或清理功能时`paths`不能为空。
> - 使用程序检查功能时`program_list`不能为空。
> - 使用时钟同步检测功能时`hosts`不能为空。
> - 路由检查和 iptables 检查不需要额外配置。

各检查项的详细配置说明、残留文件对引导节点初始化的具体影响以及初始化自动处理行为分析，请参阅[配置文件各检查项含义与影响分析](../appendix/env-check-config-analysis.md)。

配置文件字段说明：

| 字段名 | 说明 |
| ------ | ---- |
| log_file | 存放日志文件的路径（包含文件名）|
| output_format | 工具执行结果报告输出格式，目前支持 text、json |
| paths | 文件查询和清理需要指定的文件路径列表，支持环境变量和通配符 |
| clean_force | 是否执行强制删除（无需用户确认）。当前实现以命令行参数`--force`为准，需跳过确认时请显式添加`--force` |
| program_list | 需要检查是否安装的应用列表，每项包含`name`（程序名）和`should_exist`（预期是否应存在）|
| hosts | 需要进行时钟同步检测和远程分发检查的机器列表 |
| ip | 用于进行ssh连接的IP地址 |
| username | 用于进行ssh连接的用户名 |
| password | 用于进行ssh连接的密码 |
| port | 用于进行ssh连接的端口 |
| role | 机器所属角色（如 bootstrap、master、worker）|
| arch | 目标架构（amd64/arm64），为空时自动探测 |
| clock_threshold | 机器时钟误差允许阈值（单位：秒）|
| kernel_check | 内核版本检查配置 |
| kernel_check.min_version | 期望的内核版本（字符串比较），默认 4.19 |
| kernel_check.operator | 比较操作符，支持`>=`、`>`、`==`、`<=`、`<`，默认`>=` |
| port_check | 端口占用检查配置，按角色配置端口列表 |
| port_check.ports | 按角色映射的端口列表，各角色端口列表相互独立 |
| port_check.timeout | TCP 连接探测超时时间（单位：秒），默认 3 秒 |
| disk_check | 磁盘空间检查配置 |
| disk_check.check_items | 磁盘检查项列表 |
| disk_check.check_items[].path | 需要检查的目录路径，路径不存在时自动向上查找最近父目录 |
| disk_check.check_items[].min_free_gb | 该路径要求的最小可用空间（GB）|
| disk_check.check_items[].roles | （可选）只在指定角色的节点上检查此项，为空表示所有节点都检查 |
| dispatch | 远程分发执行配置 |
| dispatch.timeout | 结果收集总超时时间（单位：秒），默认 600 |
| dispatch.poll_interval | 轮询结果的时间间隔（单位：秒），默认 15 |
| dispatch.work_dir | 远程节点上的工作目录，默认`/tmp/envcheck` |
| dispatch.concurrent_limit | SSH 连接/操作的最大并发数，默认 10 |
| dispatch.checks | 远程执行的检查项列表 |
| dispatch.skip_checks | 远程执行时跳过的检查项列表 |

> ![](../figures/icon-note.gif)**说明**：
> 工具在运行时，默认会在终端输出运行结果并在当前文件夹下生成对应的html文件，并且运行过程中会将日志记录在配置文件中`log_file`指定的路径下（默认为当前文件夹下的`envCheck.log`文件）。当`output_format`为`text`时，结果输出到终端；当`output_format`为`json`时，会额外保存对应的 JSON 结果文件。HTML 报告会在命令执行时始终生成。用户可以通过修改配置文件中相应字段进行设置，以下章节默认配置文件中`output_format`为`text`。

用户可以执行如下命令获取工具帮助文档。

```bash
envCheck -h
```

#### 远程批量检查

`run`命令会连接配置中的远程节点，将`envCheck`和配置文件分发到远端工作目录，在远端执行检查，然后汇总结果。

**前提条件**

- 使用远程分发`run`命令时，配置中的目标节点需要允许 SSH 登录，并支持 SFTP 文件传输。
- 远程分发默认使用密码登录，配置文件中需要提供`hosts[].username`、`hosts[].password`、`hosts[].port`。
- 需要检查系统目录、端口、路由、iptables 或清理系统文件时，建议使用具备相应权限的用户执行。

**开始使用**

- 执行如下命令，对所有配置的节点执行全部检查项。

  ```bash
  envCheck run
  ```

- 执行如下命令，指定要执行的检查项。

  ```bash
  envCheck run --checks kernel,port,disk
  ```

- 执行如下命令，指定要跳过的检查项。

  ```bash
  envCheck run --skip clock,fileQuery
  ```

- 执行如下命令，只在指定角色节点上执行远程检查。

  ```bash
  envCheck run --roles master,worker
  ```

- 执行如下命令，设置超时时间并指定配置文件。

  ```bash
  envCheck run -c ./clusterConfig.json --checks kernel,port,disk --roles master --timeout 900
  ```

  工具运行结束后会在终端输出汇总结果并且在当前文件夹下生成`dispatch.html`文件。

**参数说明**

| 参数 | 说明 |
| ---- | ---- |
| --checks | 指定要执行的检查项，逗号分隔 |
| --skip | 指定要跳过的检查项，逗号分隔 |
| --roles | 只在指定角色节点上执行远程检查 |
| --timeout | 等待远端结果的超时时间，单位为秒，默认 600 |

可用检查项：

```text
kernel, port, disk, clock, fileQuery, programCheck, route, iptables
```

> ![](../figures/icon-note.gif)**说明：**
> 远程执行流程如下：
> 1. 连接`hosts`中的目标节点（并发，受`concurrent_limit`限制）。
> 2. 自动探测远端架构，识别`amd64`或`arm64`。
> 3. 清理并创建远端工作目录，默认`/tmp/envcheck`。
> 4. 分发适配架构的`envCheck`二进制文件（gzip 压缩传输，远端已存在且大小一致则跳过）和节点配置（按角色裁剪端口配置）。
> 5. 在远端后台执行本地检查。
> 6. 轮询拉取远端`result.json`或`error.log`并生成汇总报告。
>
> 如果远端架构已识别，工具会优先查找`./build/envCheck_<arch>`或`./envCheck_<arch>`（例如`./build/envCheck_amd64`）。如果没有找到架构专用文件，则使用当前正在运行的`envCheck`可执行文件。

#### 文件查询功能

- 执行如下命令，进行默认查询。

  ```bash
  envCheck query
  ```

  工具会查询当前文件夹下`config.json`中`paths`指定的路径下是否存在文件，工具运行结束后会在终端输出查询结果并且在当前文件夹下生成`query.html`文件。`paths`支持环境变量和通配符展开，例如`$HOME/.kube`、`/usr/bin/kube*`。

- 执行如下命令，在查询时指定配置文件路径。

  ```bash
  envCheck query --config=./queryConfig.json
  ```

  工具会查询`queryConfig.json`中`paths`指定的路径下是否存在文件，工具运行结束后会在终端输出查询结果并且在当前文件夹下生成`query.html`文件。

#### 文件清理功能

> ![](../figures/icon-note.gif)**说明：**
> 在执行文件清理功能之前，您需要确认配置文件中的`clean_force`字段值，默认为`false`。`false`表示在删除文件之前会向您进行确认，`true`表示不会进行确认。当前实现以命令行参数为准；如果您添加了`--force`执行命令，则表示无论`clean_force`字段为何值，工具都会直接删除对应文件。建议先执行`query`确认目标路径，再执行`clean`。

- 执行如下命令，进行默认清理（在删除每一文件之前会询问用户）。

  ```bash
  envCheck clean
  ```

  工具会询问是否删除当前文件夹下`config.json`中`paths`指定的路径下的文件，工具运行结束后会在终端输出清理结果并且在当前文件夹下生成`clean.html`文件。

- 执行如下命令，进行强制删除，删除文件不会询问用户，直接进行删除。

  ```bash
  envCheck clean --force
  ```

  工具会直接删除当前文件夹下`config.json`中`paths`指定的路径下的文件，工具运行结束后会在终端输出清理结果并且在当前文件夹下生成`clean.html`文件。

- 执行如下命令，在删除时指定配置文件路径。

  ```bash
  envCheck clean --config=./cleanConfig.json
  ```

  工具会按照`cleanConfig.json`文件中的设置执行文件清理功能，工具运行结束后会在终端输出清理结果并且在当前文件夹下生成`clean.html`文件。

#### 程序存在性检测

> ![](../figures/icon-note.gif)**说明：**
> `program_list` 字段已从字符串数组格式变更为对象数组格式。旧版配置文件需更新为最新格式，否则工具将报 JSON 解析错误。

程序检查功能支持配置每个程序的预期安装状态。`should_exist`为`false`表示程序不应存在，若检测到已安装则该项检查失败；`should_exist`为`true`表示程序应存在，若检测到未安装则该项检查失败。

配置示例：

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

- 执行如下命令，进行默认查询应用。

  ```bash
  envCheck program
  ```

  工具会查询当前文件夹下`config.json`中`program_list`指定的应用是否符合预期安装状态，工具运行结束后会在终端输出查询结果并且在当前文件夹下生成`check.html`文件。

- 执行如下命令，在查询时指定配置文件路径。

  ```bash
  envCheck program --config=./checkConfig.json
  ```

  工具会查询`checkConfig.json`中`program_list`指定的应用是否符合预期安装状态，工具运行结束后会在终端输出查询结果并且在当前文件夹下生成`check.html`文件。

#### 时钟同步检测功能

**前提条件**

使用时钟同步检测功能需要用户先修改默认配置文件中的相关`hosts`信息，且至少应配置一个`bootstrap`角色的节点。

**开始使用**

- 执行如下命令，查询其余host与引导节点host的时钟误差。

  ```bash
  envCheck clock
  ```

  工具会根据当前文件夹下`config.json`中的`hosts`和`clock_threshold`判断非引导节点和引导节点（bootstrap）之间的时钟误差不多于指定阈值，工具运行结束后会在终端输出检测结果并且在当前文件夹下生成`clock.html`文件。

- 执行如下命令，查询时指定配置文件。

  ```bash
  envCheck clock --config=./clockConfig.json
  ```

  工具会根据`clockConfig.json`中的`hosts`和`clock_threshold`判断非引导节点和引导节点（bootstrap）之间的时钟误差不多于指定阈值，工具运行结束后会在终端输出检测结果并且在当前文件夹下生成`clock.html`文件。

#### 内核版本检查

**前提条件**

内核版本检查功能依赖配置文件中的`kernel_check`字段。如果未配置`kernel_check`，命令会使用默认要求：`>= 4.19`。

**开始使用**

- 执行如下命令，检查当前节点内核版本是否满足要求。

  ```bash
  envCheck kernel
  ```

  工具会根据当前文件夹下`config.json`中的`kernel_check`配置检查内核版本，工具运行结束后会在终端输出检查结果并且在当前文件夹下生成`kernel.html`文件。

- 执行如下命令，在检查时指定配置文件路径。

  ```bash
  envCheck kernel --config=./kernelConfig.json
  ```

  工具会根据`kernelConfig.json`中的`kernel_check`配置检查内核版本，工具运行结束后会在终端输出检查结果并且在当前文件夹下生成`kernel.html`文件。

> ![](../figures/icon-note.gif)**说明：**
> - 支持的比较操作符：`>=`、`>`、`==`、`<=`、`<`。
> - 针对 CentOS 7 的`3.10.0-xxx`内核版本做了特殊兼容：若版本字符串为旧版构建号格式，则在`>=`/`>`比较时视为不满足，避免旧构建号被误判为通过。

#### 端口占用检查

**前提条件**

端口占用检查功能依赖配置文件中的`port_check`字段。如果未配置`port_check`，命令会使用默认端口配置。

**开始使用**

- 执行如下命令，检查当前节点所有角色的端口占用情况。

  ```bash
  envCheck port
  ```

  工具会根据当前文件夹下`config.json`中的`port_check.ports`检查当前节点对应角色的端口是否被占用，工具运行结束后会在终端输出检查结果并且在当前文件夹下生成`port.html`文件。

- 执行如下命令，只检查指定角色的端口。

  ```bash
  envCheck port --roles master,worker
  ```

  工具只会检查`master`和`worker`角色对应的端口，工具运行结束后会在终端输出检查结果并且在当前文件夹下生成`port.html`文件。

- 执行如下命令，在检查时指定配置文件路径。

  ```bash
  envCheck port --config=./portConfig.json --roles bootstrap
  ```

  工具会根据`portConfig.json`中的`port_check.ports`检查`bootstrap`角色对应的端口，工具运行结束后会在终端输出检查结果并且在当前文件夹下生成`port.html`文件。

> ![](../figures/icon-note.gif)**说明：**
> - `--roles`为空时检查配置中的所有角色端口。
> - `port_check.timeout`是端口探测超时时间，单位为秒。
> - 分发执行时，Runner 会根据节点角色过滤配置，只保留该节点角色相关的端口。

#### 磁盘空间检查

**前提条件**

磁盘空间检查功能依赖配置文件中的`disk_check`字段。如果未配置`disk_check`，命令会使用默认检查项。

配置示例：

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

**开始使用**

- 执行如下命令，检查磁盘剩余空间是否满足要求。

  ```bash
  envCheck disk
  ```

  工具会根据当前文件夹下`config.json`中的`disk_check.check_items`检查指定路径所在文件系统的剩余空间是否满足阈值，工具运行结束后会在终端输出检查结果并且在当前文件夹下生成`disk.html`文件。

- 执行如下命令，在检查时指定配置文件路径。

  ```bash
  envCheck disk --config=./diskConfig.json
  ```

  工具会根据`diskConfig.json`中的`disk_check.check_items`检查磁盘空间，工具运行结束后会在终端输出检查结果并且在当前文件夹下生成`disk.html`文件。

> ![](../figures/icon-note.gif)**说明：**
> - 若路径不存在，自动向上查找最近的父目录进行空间统计。
> - 若路径是文件，则取其所在目录进行统计。
> - 若`check_items`条目配置了`roles`字段，则仅对匹配角色的节点检查此项。

#### 默认路由检查

- 执行如下命令，检查当前节点是否存在默认路由。

  ```bash
  envCheck route
  ```

  工具会检查当前节点是否存在默认路由，并输出网关、网卡、metric 等信息。该检查不需要额外配置。工具运行结束后会在终端输出检查结果并且在当前文件夹下生成`route.html`文件。

- 执行如下命令，在检查时指定配置文件路径。

  ```bash
  envCheck route --config=./routeConfig.json
  ```

  工具运行结束后会在终端输出检查结果并且在当前文件夹下生成`route.html`文件。

#### iptables 策略检查

- 执行如下命令，检查 iptables FORWARD 链默认策略是否为 ACCEPT。

  ```bash
  envCheck iptables
  ```

  工具会检查 iptables `FORWARD`链默认策略是否为`ACCEPT`。该检查不需要额外配置。工具运行结束后会在终端输出检查结果并且在当前文件夹下生成`iptables.html`文件。

- 执行如下命令，在检查时指定配置文件路径。

  ```bash
  envCheck iptables --config=./iptablesConfig.json
  ```

  工具运行结束后会在终端输出检查结果并且在当前文件夹下生成`iptables.html`文件。

### 输出说明

每次命令运行会写入日志文件，并根据命令生成 HTML 报告。当`output_format`为`json`时，还会保存对应的 JSON 结果文件。

| 命令 | HTML 报告 | JSON 结果文件 |
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

> ![](../figures/icon-note.gif)**说明：**
> - 当`output_format`为`text`时，结果会输出到终端，不保存 JSON 结果文件。
> - 当`output_format`为`json`时，会保存对应的 JSON 结果文件。
> - HTML 报告会在命令执行时始终生成。

### 常见问题

#### 提示 at least one path is required

`query`或`clean`命令要求配置`paths`。请检查配置文件路径是否正确，以及`paths`是否为空。

#### 提示 program list is required

`program`命令要求配置`program_list`。请添加至少一个程序检查项。

#### 提示 hosts field is required

`clock`命令要求配置`hosts`。请添加至少一个节点，并确保其中存在`bootstrap`角色节点。

#### 远程检查连接失败

请检查：

- IP、端口、用户名、密码是否正确。
- 远端 SSH 服务是否可用。
- 远端是否允许密码登录。
- 当前机器到远端的网络是否连通。
- 远端用户是否有创建`dispatch.work_dir`的权限。

#### 没有生成 JSON 文件

只有当`output_format`设置为`json`时，才会保存 JSON 结果文件。HTML 报告会在命令执行时生成。
