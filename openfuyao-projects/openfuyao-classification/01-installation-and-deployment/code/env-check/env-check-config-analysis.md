# env-check 配置文件（config.json）各检查项含义与影响分析

env-check 通过 `run` 命令分发到各节点执行 `run-local`，在每台节点上执行以下 8 项检查（`run.go:167`）：

```
kernel, port, disk, clock, fileQuery, programCheck, route, iptables
```

---

## 一、log_file

| 属性 | 值 |
|------|-----|
| 当前配置 | `"./envCheck.log"` |
| 作用 | env-check 工具自身的日志文件路径 |

**含义**：所有检查过程的日志输出到此文件，包括每项检查的开始、通过、警告、失败信息。

**影响**：仅影响 env-check 工具自身的日志记录，与 BKE 初始化无关。

---

## 二、output_format

| 属性 | 值 |
|------|-----|
| 当前配置 | `"text"` |
| 可选值 | `text`、`json` |
| 作用 | 检查结果报告的输出格式 |

**含义**：控制检查结果以文本、JSON 。`text` 适合命令行查看，`json` 适合程序解析。

**影响**：仅影响报告展示格式，与 BKE 初始化无关。

---

## 三、paths（文件冲突检查 — fileQuery）

### 检查机制

通过 `pkg/query/query.go` 的 `FileQuery.Execute()` 实现：对每条路径展开环境变量（如 `$HOME`）和通配符（如 `kube*`），然后检测文件/目录是否存在。若存在则标记为"冲突"。

检查结果判定（`run.go:360-367`）：
- `TotalExists == 0` → **pass**（所有残留路径均不存在）
- `TotalExists > 0` → **fail**（存在残留文件）

### 当前配置的 14 条路径

| # | 路径 | 含义 | 残留对引导节点初始化的影响 | 残留对集群创建的影响 | 初始化是否自动处理 | 严重程度 |
|---|------|------|------------------------|---------------------|-------------------|----------|
| 1 | `$HOME/.kube` | Kubernetes kubeconfig 目录 | bkeadm 启动 k3s 后覆盖 `~/.kube/config`。若 k3s 跳过启动（残留 k3s 数据导致 `isKubernetesAvailable` 返回 true），残留 kubeconfig 指向旧集群，后续 kubectl/CRD 部署操作打到错误集群 | capbke 在 Master 节点覆盖 `~/.kube/config`；Worker 节点不生成 `~/.kube/config`，残留不被覆盖，仅影响手动 kubectl | 引导节点：条件覆盖；Master：覆盖；Worker：不触碰 | 🟡 中 |
| 2 | `/etc/kubernetes` | Kubernetes 核心配置目录（pki/、manifests/、*.conf） | bkeadm 不直接操作此目录（k3s 容器内管理）。若 k3s 跳过启动，残留配置可能导致 k3s 容器内进程读取旧配置 | capbke 覆盖大部分文件（pki 证书从 Secret 加载覆盖、kubeconfig 重新生成、manifests `O_TRUNC` 覆盖），但目录中可能有不被覆盖的残留文件 | 部分覆盖，不完全清理 | 🟡 中 |
| 3 | `/usr/bin/kube*` | 通配符匹配 kubectl、kubelet、kubeadm 等二进制 | bkeadm 从 k3s 容器复制覆盖 `/usr/bin/kubectl`。引导节点不安装 kubelet 二进制 | capbke 先 `rm -rf /usr/bin/kubelet` 再下载（`run.go:576-596`），kubectl 下载覆盖 | 自动覆盖 | 🟢 低 |
| 4 | `/usr/local/bin/kube*` | 通配符匹配 `/usr/local/bin/` 下的 k8s 二进制 | BKE 不在此路径安装任何文件，残留不会被覆盖。若 PATH 中 `/usr/local/bin` 优先级更高，可能执行到旧版本 | 同左 | 不触碰 | 🟢 低 |
| 5 | `/usr/local/bin/crictl` | crictl CLI 工具（CRI 管理工具） | **BKE 将 crictl 安装到 `/usr/bin/crictl`（containerd tar 解压），不安装到 `/usr/local/bin/crictl`**。此路径的残留不会被 BKE 覆盖。若 PATH 优先级问题，可能执行到旧版本 crictl | 同左 | 不触碰（路径不匹配） | 🟡 中 |
| 6 | `/etc/sysctl.d/k8s.conf` | Kubernetes 内核参数配置（ip_forward、bridge-nf-call-iptables 等） | bkeadm 在引导节点上**不写入**此文件（bkeadm 的 `SetSysctl` 只操作 `/etc/sysctl.conf`）。残留的内核参数继续生效，若值不正确可能导致网络异常 | capbke 在集群节点上以 `O_TRUNC` **完全覆盖**此文件（`init.go:342`） | 引导节点：不触碰；集群节点：覆盖 | 🟡 中 |
| 7 | `/etc/systemd/system/kubelet.service` | kubelet 的 systemd unit 文件 | bkeadm 不安装 kubelet（k3s 容器内运行）。残留文件若被 systemd 加载，可能导致旧 kubelet 进程启动，与 k3s 容器内 kubelet 冲突 | capbke 以 `O_TRUNC` **完全覆盖**（`run.go:435`） | 引导节点：不触碰；集群节点：覆盖 | 🟡 中 |
| 8 | `/etc/systemd/system/kubelet.service.d` | kubelet systemd drop-in 目录 | bkeadm 和 capbke **均不清理此目录**。残留的旧 drop-in `.conf` 文件会被 systemd 合并加载，覆盖 capbke 写入的 kubelet.service 参数，导致 kubelet 使用错误配置启动 | 同左 | **不清理** | 🔴 高 |
| 9 | `/var/lib/etcd` | 标准 kubeadm etcd 数据目录 | BKE 使用 `/var/lib/openFuyao/etcd`（`defaults.go:49`），**不使用** `/var/lib/etcd`。残留数据不会被读取，但表明机器曾安装过标准 k8s | 同左 | 不触碰（BKE 不使用此路径） | 🟢 低 |
| 10 | `/var/lib/kubelet` | kubelet 工作根目录（config.yaml、插件数据、卷挂载点） | bkeadm 不直接操作（k3s 容器内 kubelet 使用容器内目录） | capbke 覆盖 `config.yaml`，但目录中的旧挂载点、Pod 数据、`pki/` 子目录**不被清理**。残留旧挂载点可能导致 kubelet 启动时尝试挂载已不存在的卷 | 部分覆盖（仅 config.yaml） | 🔴 高 |
| 11 | `/run/containerd/containerd.sock` | containerd CRI Unix socket | bkeadm 安装 containerd 并启动，containerd 创建此 socket。若旧 containerd 仍在运行，socket 已存在，新 containerd 启动可能失败 | 同左 | containerd 启动时创建/覆盖 | 🟡 中 |
| 12 | `/usr/lib/systemd/system/kubelet.service.d` | 包管理器安装的 kubelet systemd drop-in 目录 | bkeadm 和 capbke **均不清理此目录**。残留的旧 drop-in 会覆盖 kubelet.service 参数 | 同左 | **不清理** | 🟡 中 |
| 13 | `/var/run/containerd/containerd.sock` | 同 `/run/containerd/containerd.sock`（`/var/run` 通常为 `/run` 的符号链接） | 同第 11 项 | 同第 11 项 | 同上 | 🟡 中 |
| 14 | `/var/run/docker.sock` | Docker daemon Unix socket | 若引导节点使用 containerd 模式（默认），残留 docker.sock 表明旧 docker 仍在运行。bkeadm 不检查此 socket | capbke 预检检测容器运行时类型（`check.go:415-430`），若发现 docker.sock 存在但配置为 containerd，**预检失败** | 不触碰 | 🔴 高 |

## 四、clean_force

| 属性 | 值 |
|------|-----|
| 当前配置 | `false` |
| 作用 | 控制 `fileClean` 模式下是否跳过用户确认，直接删除残留文件 |

**含义**：
- `false`：删除每个残留文件前交互式询问用户 `[y/n]`
- `true`：跳过确认，直接删除所有残留文件

**影响**：仅影响 `fileClean` 模式的行为。当前配置为 `false`，删除操作需要用户逐个确认，防止误删。

---

## 五、program_list（程序检查 — programCheck）

### 检查机制

通过 `pkg/program/check.go` 的 `ApplicationChecker.Execute()` 实现：对每个程序使用 `exec.LookPath(name)` 检测是否在 PATH 中可找到，然后与 `should_exist` 期望值比较。

检查结果判定（`run.go:396-403`）：
- `TotalFailed == 0` → **pass**（所有程序符合预期状态）
- `TotalFailed > 0` → **fail**（有程序不符合预期）

### 当前配置的 4 个程序

| # | 程序名 | should_exist | 含义 | 残留对初始化的影响 | 初始化处理行为 | 严重程度 |
|---|--------|-------------|------|------------------|--------------|----------|
| 1 | `docker` | `false`（不应存在） | Docker 容器运行时 | containerd 模式下，capbke 预检检测运行时类型不匹配（`check.go:415-430`），**节点初始化失败**。bkeadm 不自动卸载 docker | **不自动卸载**。bkeadm 检测到 docker 已安装则使用现有 docker；capbke 检测到不匹配则预检失败 | 🔴 高 |
| 2 | `kubelet` | `false`（不应存在） | Kubernetes 节点代理（系统服务） | 若旧 kubelet 正在运行且非 systemd 管理，capbke 的 `systemctl stop kubelet` 失败（仅 Warning），旧进程持续占用端口 10250/10248，新 kubelet 启动失败（`run.go:172-176`）。引导节点上旧 kubelet 可能干扰 iptables 和 `/var/lib/kubelet` | capbke 先 `systemctl stop` 再 `rm -rf /usr/bin/kubelet` 再下载（`run.go:576-596`），但**不杀死非 systemd 管理的运行中进程** | 🔴 高 |
| 3 | `containerd` | `false`（不应存在） | containerd 容器运行时 | 二进制被 tar 解压覆盖，但残留的配置（`/etc/containerd/config.toml`）和数据（`/var/lib/containerd`）不被完全清理 | bkeadm 解压 containerd tar **覆盖**二进制（`containerd.go:172`） | 🟡 中 |
| 4 | `tar` | `true`（应存在） | tar 归档工具 | tar 是 BKE 安装过程中的基础依赖：bkeadm 解压 containerd tar（`containerd.go:172`）、解压 CNI 插件（`containerd.go:387`）、解压镜像/源数据（`repository.go`）均依赖 tar。若 tar 不存在，**初始化直接失败** | 不安装（假设系统自带） | 🔴 高 |

### 程序检查总结

| 程序 | 检查方向 | 严重程度 | 说明 |
|------|---------|----------|------|
| docker | 不应存在 | 🔴 高 | containerd 模式下残留导致 capbke 预检失败 |
| kubelet | 不应存在 | 🔴 高 | 残留运行进程导致新 kubelet 端口冲突启动失败 |
| containerd | 不应存在 | 🟡 中 | 二进制被覆盖，配置/数据残留 |
| tar | 应存在 | 🔴 高 | 缺失导致解压操作全部失败 |

---

## 六、hosts（主机列表）

### 当前配置

| IP | 角色 | SSH 端口 |
|----|------|---------|
| 192.168.2.135 | bootstrap | 22 |
| 192.168.2.221 | master | 22 |
| 192.168.2.229 | worker | 22 |

**含义**：定义 env-check 需要检查的目标主机列表。每台主机的 `role` 决定执行哪些端口检查（`port_check.ports` 中对应角色的端口列表）。

**影响**：
- `dispatch` 模式下，env-check 将自身二进制分发到每台主机，通过 SSH 远程执行检查
- `role` 用于端口检查的角色匹配（`port/check.go:130-155`）：`bootstrap` 角色执行"引导节点端口检查"，`master` 执行"Master 节点端口检查"，`worker` 执行"Worker 节点端口检查"
- `clock_threshold` 时钟检查也需要 hosts 列表（通过 SSH 获取各主机时间并与本地比较）

---

## 七、clock_threshold（时钟同步检查 — clock）

| 属性 | 值 |
|------|-----|
| 当前配置 | `10`（秒） |
| 作用 | 各主机间允许的最大时间差 |

### 检查机制

通过 `pkg/clock/clock.go` 的 `CheckerClock.Execute()` 实现：通过 SSH 连接每台主机获取远程时间，与本地时间比较，计算时间差。若任意两台主机间时间差超过 `clock_threshold` 秒，则检查失败。

### 影响

| 场景 | 影响 | 严重程度 |
|------|------|----------|
| 主机间时间差 > 10 秒 | etcd 集群依赖一致的时间进行 leader 选举和日志复制，时间差过大会导致 etcd 选举失败或数据不一致。Kubernetes 证书也有时间有效性校验，时间偏差可能导致证书校验异常 | 🔴 高 |
| bkeadm 初始化 | bkeadm 的 `setTimezone()` 会设置时区和 NTP 服务器，但仅设置引导节点。集群节点的时间同步由 capbke env init 负责 | 🟡 中 |

**说明**：当前 `run.go:327-333` 中的 `runClockCheck()` 仅返回本地时间并标记为 pass，实际的多主机时钟比较在 `dispatch` 模式中由 `CheckerClock.Execute()` 完成。

---

## 八、kernel_check（内核版本检查 — kernel）

| 属性 | 值 |
|------|-----|
| 当前配置 | `{"min_version": "4.19", "operator": ">="}` |
| 作用 | 检查内核版本是否满足最低要求 |

### 检查机制

通过 `pkg/kernel/check.go` 的 `Checker.Execute()` 实现：使用 `gopsutil/host.Info()` 获取内核版本，与 `min_version` 按 `operator` 比较。

特殊处理（`check.go:97-115`）：对 CentOS 7 的 `3.10.0-xxx` 内核（带有旧式 build number）特殊判断——即使版本号 >= 4.19，若检测到 `3.10.0` 前缀且带旧式 build number，也判定为不满足 `>=` 条件。

### 影响

| 场景 | 影响 | 严重程度 |
|------|------|----------|
| 内核版本 < 4.19 | BKE 依赖 cgroup v2、eBPF、overlayfs 等较新内核特性。低版本内核可能导致容器运行时、CNI 插件、kubelet 功能异常。containerd 的 `native` snapshotter 和 Calico 的 eBPF datapath 都需要较新内核 | 🔴 高 |

---

## 九、port_check（端口占用检查 — port）

### 检查机制

通过 `pkg/port/check.go` 的 `Checker.Execute()` 实现：根据节点角色选择对应端口列表，对每个端口 TCP 拨号 `127.0.0.1:<port>`，若连接成功则端口被占用。还能通过 gopsutil 识别占用进程。

检查结果判定（`run.go:274-280`）：
- `Used == 0` → **pass**（所有端口空闲）
- `Used > 0` → **fail**（存在被占用端口）

### 当前配置

#### Bootstrap 端口（5 个）

| 端口 | 服务 | 来源代码 | 占用后果 | 严重程度 |
|------|------|---------|----------|----------|
| 36443 | 本地 k3s API Server | `bkeadm/constants.go:17` `DefaultKubernetesPort` | 🔴 bkeadm `validatePorts()` 硬失败，初始化中止 | 🔴 高 |
| 40080 | Yum 仓库 | `capbke/defaults.go:63` `DefaultYumRepoPort` | 🔴 bkeadm `validatePorts()` 硬失败 | 🔴 高 |
| 40443 | 镜像仓库 | `capbke/defaults.go:60` `DefaultImageRepoPort` | 🔴 bkeadm `validatePorts()` 硬失败 | 🔴 高 |
| 38080 | Chart 仓库 | `bkeadm/constants.go:38` `DefaultChartRegistryPort` | 🔴 bkeadm `validatePorts()` 硬失败 | 🔴 高 |
| 30010 | k3s NodePort 映射（ingress-nginx） | `bkeadm/k3s.go:322` `-p 30010:30010` | 🟡 k3s 容器创建失败（`validatePorts()` 不检查此端口，报错可能不明确） | 🟡 中 |

其中36443、40080、40443、38080这些端口是支持可配置的，如果在bke init使用了指定的端口时，需要替换为检查对应的端口。

#### Master 端口（13 个）

| 端口 | 服务 | 来源代码 | 占用后果 | 严重程度 |
|------|------|---------|----------|----------|
| 6443 | kube-apiserver | `capbke/defaults.go:38` `DefaultAPIBindPort` | 🔴 apiserver 无法绑定，启动失败 | 🔴 高 |
| 2379 | etcd 客户端 | `capbke/consts_sca.go:90` `EtcdListenClientPort` | 🔴 etcd 无法绑定，启动失败 | 🔴 高 |
| 2380 | etcd peer | `capbke/consts_sca.go:106` `EtcdListenPeerPort` | 🔴 etcd 无法组建集群 | 🔴 高 |
| 2381 | etcd metrics | `capbke/consts_sca.go:92` `EtcdMetricsPort` | 🟡 etcd metrics 无法绑定（非致命） | 🟡 中 |
| 10248 | kubelet healthz | `capbke/consts_sca.go:295` `KubeletHealthzPort` | 🔴 kubelet 无法启动 | 🔴 高 |
| 10249 | kube-proxy metrics | Kubernetes 默认端口 | 🟡 kube-proxy metrics 无法绑定 | 🟡 中 |
| 10250 | kubelet 安全端口 | `capbke/consts_sca.go:350` `KubeletPort` | 🔴 kubelet 无法启动 | 🔴 高 |
| 10256 | kube-proxy healthz | Kubernetes 默认端口 | 🟡 kube-proxy healthz 无法绑定 | 🟡 中 |
| 10257 | kube-controller-manager | `capbke/consts_sca.go:356` `KubeControllerManagerPort` | 🔴 KCM 无法启动 | 🔴 高 |
| 10259 | kube-scheduler | `capbke/consts_sca.go:353` `KubeSchedulerPort` | 🔴 scheduler 无法启动 | 🔴 高 |
| 3377 | bkeagent-launcher /readyz | `capbke/cmd/bkeagent-launcher/main.go:279` | 🟡 launcher 就绪探针失败 | 🟡 中 |
| 58080 | bkeagent health | `bkeadm/constants.go:49` `DefaultAgentHealthPort` | 🟡 agent 健康检查失败 | 🟡 中 |
| 1338 | containerd metrics | `bkeadm/.../containerd_default.yaml:27` `metricsAddress` | 🔴 containerd 启动失败，metrics 端口绑定失败导致进程退出 | 🔴 高 |

> 1338 端口绑定在 `127.0.0.1`（环回地址），由 ContainerdConfig CR 配置（`metricsAddress: "127.0.0.1:1338"`），capbke 读取此 CR 渲染到集群节点的 containerd `config.toml`。引导节点的 containerd 使用 bkeadm 自有模板（`[metrics] address = ''`），不启用 metrics，因此 bootstrap 端口列表不含 1338。

#### Worker 端口（7 个）

| 端口 | 服务 | 来源代码 | 占用后果 | 严重程度 |
|------|------|---------|----------|----------|
| 3377 | bkeagent-launcher /readyz | `capbke/cmd/bkeagent-launcher/main.go:279` | 🟡 launcher 就绪探针失败 | 🟡 中 |
| 58080 | bkeagent health | `bkeadm/constants.go:49` `DefaultAgentHealthPort` | 🟡 agent 健康检查失败 | 🟡 中 |
| 10248 | kubelet healthz | `capbke/consts_sca.go:295` `KubeletHealthzPort` | 🔴 kubelet 无法启动 | 🔴 高 |
| 10249 | kube-proxy metrics | Kubernetes 默认端口 | 🟡 kube-proxy metrics 无法绑定 | 🟡 中 |
| 10250 | kubelet 安全端口 | `capbke/consts_sca.go:350` `KubeletPort` | 🔴 kubelet 无法启动 | 🔴 高 |
| 10256 | kube-proxy healthz | Kubernetes 默认端口 | 🟡 kube-proxy healthz 无法绑定 | 🟡 中 |
| 1338 | containerd metrics | `bkeadm/.../containerd_default.yaml:27` `metricsAddress` | 🔴 containerd 启动失败，metrics 端口绑定失败导致进程退出 | 🔴 高 |

### 端口检查总结

| 角色 | 配置端口数 | 覆盖通信矩阵情况 |
|------|-----------|----------------|
| Bootstrap | 5 | 覆盖通信矩阵 bootstrap TCP 端口（36443/40080/40443/38080/30010）|
| Master | 13 | 覆盖通信矩阵 master 核心端口 + containerd metrics（6443/2379/2380/2381/10248/10249/10250/10256/10257/10259/3377/58080/1338）|
| Worker | 7 | 覆盖通信矩阵 worker 核心端口 + containerd metrics（3377/58080/10248/10249/10250/10256/1338）|



---

## 十、disk_check（磁盘空间检查 — disk）

| 属性 | 值 |
|------|-----|
| 当前配置 | `{"check_items": [{"path": "/", "min_free_gb": 50}]}` |
| 作用 | 检查指定路径的可用磁盘空间是否满足最低要求 |

### 检查机制

通过 `pkg/disk/check.go` 的 `Checker.Execute()` 实现：对每个 `check_items` 条目，使用 `gopsutil/disk.Usage()` 获取指定路径的磁盘使用情况，比较 `Free` 空间与 `MinFreeGB * 1GB`。

若路径不存在，自动向上查找父目录的挂载点。若 `check_items` 条目配置了 `roles` 字段，则仅对匹配角色的节点检查（当前配置未设置 `roles`，对所有节点检查）。

检查结果判定（`run.go:315-322`）：
- `InsufficientPath == 0` → **pass**（所有路径空间充足）
- `InsufficientPath > 0` → **fail**（存在空间不足的路径）

### 当前配置分析

| 路径 | 最低空闲空间 | 含义 |
|------|------------|------|
| `/` | 50 GB | 根分区至少需要 50 GB 空闲空间 |

### 影响

| 场景 | 影响 | 严重程度 |
|------|------|----------|
| 根分区空闲 < 50 GB | bkeadm `validateDiskSpace()` 检查工作目录磁盘空间：若已有镜像数据要求 ≥3 GB，否则要求 ≥20 GB（`initialize.go:217-231`）。env-check 的 50 GB 要求更严格，为 BKE 安装预留充足空间（镜像仓库、源仓库、k3s 数据、containerd 镜像等） | 🔴 高 |

---

## 十一、dispatch（分发配置）

| 属性 | 值 |
|------|-----|
| timeout | `600`（秒，即 10 分钟） |
| poll_interval | `15`（秒） |
| work_dir | `/tmp/envcheck` |
| concurrent_limit | `10` |

### 各字段含义

| 字段 | 含义 |
|------|------|
| `timeout` | 分发执行的超时时间。env-check 将自身二进制通过 SCP 分发到各远程节点，在每台节点上执行检查并收集结果。600 秒超时覆盖所有节点的检查时间 |
| `poll_interval` | 轮询远程节点检查结果的间隔时间。每 15 秒检查一次各节点是否完成 |
| `work_dir` | 远程节点上的工作目录。env-check 二进制和检查结果文件（`result.json`）存放在此目录下 |
| `concurrent_limit` | 并发检查的节点数上限。最多同时检查 10 台节点 |

### 影响

| 场景 | 影响 |
|------|------|
| `timeout` 过短 | 节点检查未完成但超时，结果标记为失败。实际检查（8 项）通常在 1-2 分钟内完成，600 秒通常充足 |
| `work_dir` 不可写 | 远程节点上 `/tmp/envcheck` 目录无法创建（权限问题或 `/tmp` 不可写），检查结果无法保存 |
| `concurrent_limit` 过小 | 节点数多时检查耗时长。当前 3 台节点，10 并发完全足够 |

**说明**：dispatch 配置仅影响 env-check 工具的分发行为，与 BKE 初始化无关。

---

## 十二、route 和 iptables 检查

这两项检查在 `run-local` 中默认执行（`run.go:167`），但**不在 config.json 中显式配置**——它们使用固定逻辑，不依赖配置参数。

### route（默认路由检查）

**检查机制**（`pkg/route/route.go`）：执行 `route -n` 命令解析路由表，检查是否存在默认路由。若无默认路由，节点网络配置异常，可能导致 BKE 组件间通信失败。

**影响**：无默认路由的节点无法访问外部网络（拉取镜像、下载二进制等），BKE 初始化失败。

### iptables（FORWARD 链策略检查）

**检查机制**（`pkg/iptables/iptables.go`）：执行 `iptables -L FORWARD -n` 命令，检查 FORWARD 链的默认策略是否为 `ACCEPT`。

**影响**：若 FORWARD 链策略为 `DROP`，Kubernetes Pod 间跨节点通信会被 iptables 丢弃，导致网络不通。Calico CNI 依赖 FORWARD 链转发 Pod 流量。bkeadm 的 `prepareEnvironment()` 会安装 iptables 并切换到 legacy 模式，但不设置 FORWARD 策略——这由 capbke env init 负责（`init.go` 中设置 iptables 规则）。
