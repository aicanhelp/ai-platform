# envCheck 工具

`envCheck` 是一个用于部署前环境检查和残留清理的命令行工具，支持本机检查、远程批量分发检查、结果文本/JSON 输出，以及 HTML 报告生成。

## 功能概览

- **文件查询（query）**：查询节点上是否存在指定残留文件或目录，输出存在性、权限、属主等信息。
- **文件清理（clean）**：清理指定残留文件或目录，支持交互确认和强制删除。
- **程序检查（program）**：检查冲突程序是否符合预期安装状态。
- **时钟同步检查（clock）**：检查多节点时钟是否与 bootstrap 节点同步。
- **内核版本检查（kernel）**：检查操作系统内核版本是否符合配置要求。
- **端口占用检查（port）**：检查本地端口占用情况，按角色配置端口列表。
- **磁盘空间检查（disk）**：检查磁盘剩余空间是否满足配置要求，支持按角色过滤。
- **默认路由检查（route）**：检查当前节点是否存在默认路由。
- **iptables 策略检查（iptables）**：检查 iptables FORWARD 链默认策略是否为 ACCEPT。
- **远程批量检查（run）**：将工具分发到多台远程节点并行执行检查，统一收集结果并生成汇总报告。

## 前提条件

本工具目前只支持在 `Linux amd64` 和 `Linux arm64` 环境下执行，请根据环境自行下载对应版本的工具。

| 下载项 | amd64 | arm64 |
| --- | --- | --- |
| envCheck 工具 | [下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/envCheck) | [下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/envCheck) |
| 默认配置文件 | [下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/config.json) | [下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/config.json) |

配置文件可以自行新建或下载默认配置文件进行修改。

## 快速开始

### 从源码构建

```bash
go build -o envCheck ./cmd
```

远程分发检查支持按远端架构选择二进制文件。需要跨架构分发时，可提前构建：

```bash
GOOS=linux GOARCH=amd64 go build -o build/envCheck_amd64 ./cmd
GOOS=linux GOARCH=arm64 go build -o build/envCheck_arm64 ./cmd
```

### 基本要求

- 建议在 Linux 节点上运行。
- 需要检查系统目录、端口、路由、iptables 或清理系统文件时，建议使用具备相应权限的用户执行。
- 使用远程分发 `run` 命令时，配置中的目标节点需要允许 SSH 登录，并支持 SFTP 文件传输。
- 远程分发默认使用密码登录，配置文件中需要提供 `hosts[].username`、`hosts[].password`、`hosts[].port`。

### 运行

将 `envCheck` 和 `config.json` 放在同一目录，授予执行权限后运行：

```bash
chmod +x ./envCheck
./envCheck --help
./envCheck query
```

默认配置文件路径为 `./config.json`。如需指定其他配置文件：

```bash
./envCheck query --config ./my-config.json
```

`--config` 也可以简写为 `-c`。

## 配置文件

配置文件使用 JSON 格式。下面是一个脱敏后的完整示例：

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

### 字段说明

| 字段 | 说明 |
| --- | --- |
| `log_file` | 日志文件路径，默认可使用 `./envCheck.log`。 |
| `output_format` | 命令行结果格式，支持 `text` 和 `json`。默认 `text`。 |
| `paths` | 文件查询和文件清理使用的路径列表，支持环境变量和通配符。 |
| `clean_force` | 清理强制模式配置项。当前实现以命令行 `clean --force` 为准；需要跳过确认时请显式添加 `--force`。 |
| `program_list` | 程序检查列表。`name` 是程序名，`should_exist` 表示预期是否应存在。 |
| `hosts` | 多节点时钟检查和远程分发使用的主机列表。 |
| `hosts[].ip` | 用于 SSH 连接的 IP 地址。 |
| `hosts[].username` | 用于 SSH 连接的用户名。 |
| `hosts[].password` | 用于 SSH 连接的密码。 |
| `hosts[].port` | 用于 SSH 连接的端口。 |
| `hosts[].role` | 节点角色，如 `bootstrap`、`master`、`worker`。 |
| `hosts[].arch` | 目标架构（`amd64`/`arm64`），为空时自动探测。 |
| `clock_threshold` | 时钟误差阈值，单位为秒。 |
| `kernel_check` | 内核版本检查配置。 |
| `kernel_check.min_version` | 期望的内核版本（字符串比较），默认 `4.19`。 |
| `kernel_check.operator` | 比较操作符，支持 `>=`、`>`、`==`、`<=`、`<`，默认 `>=`。 |
| `port_check` | 端口占用检查配置，按角色配置端口列表。 |
| `port_check.ports` | 按角色映射的端口列表，各角色端口列表相互独立。 |
| `port_check.timeout` | TCP 连接探测超时时间（秒），默认 3 秒。 |
| `disk_check` | 磁盘空间检查配置。 |
| `disk_check.check_items` | 磁盘检查项列表。 |
| `disk_check.check_items[].path` | 需要检查的目录路径。路径不存在时自动向上查找最近父目录。 |
| `disk_check.check_items[].min_free_gb` | 该路径要求的最小可用空间（GB）。 |
| `disk_check.check_items[].roles` | （可选）只在指定角色的节点上检查此项；为空表示所有节点都检查。 |
| `dispatch` | 远程分发执行配置。 |
| `dispatch.timeout` | 结果收集总超时时间（秒），默认 600。 |
| `dispatch.poll_interval` | 轮询结果的时间间隔（秒），默认 15。 |
| `dispatch.work_dir` | 远程节点上的工作目录，默认 `/tmp/envcheck`。 |
| `dispatch.concurrent_limit` | SSH 连接/操作的最大并发数，默认 10。 |
| `dispatch.checks` | 远程执行的检查项列表。 |
| `dispatch.skip_checks` | 远程执行时跳过的检查项列表。 |

### 配置校验

工具会进行以下校验：

- 使用文件查询或清理功能时 `paths` 不能为空。
- 使用程序检查功能时 `program_list` 不能为空。
- 使用时钟同步检测功能时 `hosts` 不能为空。
- 路由检查和 iptables 检查不需要额外配置。

> 各检查项的详细配置说明、残留文件对引导节点初始化的具体影响以及初始化自动处理行为分析，请参阅 [配置文件各检查项含义与影响分析](./env-check-config-analysis.md)。

## 命令说明

### 查询残留文件

```bash
./envCheck query
./envCheck query -c ./query-config.json
```

该命令读取 `paths`，检查每个路径是否存在，并输出文件类型、权限、属主等信息。

注意：

- `paths` 不能为空。
- `query` 会展开环境变量和通配符，例如 `$HOME/.kube`、`/usr/bin/kube*`。
- 工具运行结束后会在终端输出查询结果并生成 `query.html` 文件。

### 清理残留文件

```bash
./envCheck clean
./envCheck clean --force
./envCheck clean -c ./clean-config.json
```

该命令读取 `paths` 并清理匹配到的文件或目录。

注意：

- `paths` 不能为空。
- 当前实现以命令行参数为准；需要跳过交互确认时，请使用 `clean --force`。
- `clean --force` 会直接删除匹配项，请谨慎使用。
- 建议先执行 `query` 确认目标路径，再执行 `clean`。
- 工具运行结束后会在终端输出清理结果并生成 `clean.html` 文件。

### 检查冲突程序

```bash
./envCheck program
./envCheck program -c ./program-config.json
```

该命令读取 `program_list`，检查程序安装状态是否符合预期。

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
    "name": "tar",
    "should_exist": true
  }
]
```

`should_exist: false` 表示程序不应存在；如果检测到已安装，则该项检查失败。`should_exist: true` 表示程序应存在；如果检测到未安装，则该项检查失败。

工具运行结束后会在终端输出查询结果并生成 `check.html` 文件。

### 检查时钟同步

```bash
./envCheck clock
./envCheck clock -c ./clock-config.json
```

该命令读取 `hosts` 和 `clock_threshold`，以角色包含 `bootstrap` 的节点作为参考节点，检查其他节点与参考节点的时间差是否超过阈值。

注意：

- `hosts` 不能为空。
- 至少应配置一个 `bootstrap` 节点。
- 需要能够通过 SSH 连接配置中的节点。
- 工具运行结束后会在终端输出检测结果并生成 `clock.html` 文件。

### 检查内核版本

```bash
./envCheck kernel
./envCheck kernel -c ./kernel-config.json
```

该命令读取 `kernel_check`，检查当前节点内核版本是否满足要求。

配置示例：

```json
"kernel_check": {
  "min_version": "4.19",
  "operator": ">="
}
```

- 支持比较操作符：`>=`、`>`、`==`、`<=`、`<`。
- 如果未配置 `kernel_check`，命令会使用默认要求：`>= 4.19`。
- 针对 CentOS 7 的 `3.10.0-xxx` 内核版本做了特殊兼容：若版本字符串为旧版构建号格式，则在 `>=` / `>` 比较时视为不满足，避免旧构建号被误判为通过。
- 工具运行结束后会在终端输出检查结果并生成 `kernel.html` 文件。

### 检查端口占用

```bash
./envCheck port
./envCheck port --roles master,worker
./envCheck port -c ./port-config.json --roles bootstrap
```

该命令读取 `port_check.ports`，检查当前节点指定角色对应的端口是否被占用。

注意：

- `--roles` 为空时检查配置中的所有角色端口。
- `--roles master,worker` 只检查 `master` 和 `worker` 对应端口。
- `port_check.timeout` 是端口探测超时时间，单位为秒。
- 分发执行时，Runner 会根据节点角色过滤配置，只保留该节点角色相关的端口。
- 工具运行结束后会在终端输出检查结果并生成 `port.html` 文件。

### 检查磁盘空间

```bash
./envCheck disk
./envCheck disk -c ./disk-config.json
```

该命令读取 `disk_check.check_items`，检查指定路径所在文件系统的剩余空间是否满足阈值。

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

- 若路径不存在，自动向上查找最近的父目录进行空间统计。
- 若路径是文件，则取其所在目录进行统计。
- 如果未配置 `disk_check`，命令会使用默认检查项。
- 工具运行结束后会在终端输出检查结果并生成 `disk.html` 文件。

### 检查默认路由

```bash
./envCheck route
./envCheck route -c ./route-config.json
```

该命令检查当前节点是否存在默认路由，并输出网关、网卡、metric 等信息。该检查不需要额外配置。

工具运行结束后会在终端输出检查结果并生成 `route.html` 文件。

### 检查 iptables FORWARD 策略

```bash
./envCheck iptables
./envCheck iptables -c ./iptables-config.json
```

该命令检查 iptables `FORWARD` 链默认策略是否为 `ACCEPT`。该检查不需要额外配置。

工具运行结束后会在终端输出检查结果并生成 `iptables.html` 文件。

### 远程批量检查

`run` 命令会连接配置中的远程节点，将 `envCheck` 和配置文件分发到远端工作目录，在远端执行检查，然后汇总结果。

```bash
./envCheck run
./envCheck run --checks kernel,port,disk
./envCheck run --checks route,iptables
./envCheck run --skip clock,fileQuery
./envCheck run --roles master,worker
./envCheck run --timeout 900
./envCheck run -c ./cluster-config.json --checks kernel,port,disk --roles master
```

#### 参数说明

| 参数 | 说明 |
| --- | --- |
| `--checks` | 指定要执行的检查项，逗号分隔。 |
| `--skip` | 指定要跳过的检查项，逗号分隔。 |
| `--roles` | 只在指定角色节点上执行远程检查。 |
| `--timeout` | 等待远端结果的超时时间，单位为秒，默认 `600`。 |

可用检查项：

```text
kernel, port, disk, clock, fileQuery, programCheck, route, iptables
```

#### 远程执行流程

1. 连接 `hosts` 中的目标节点（并发，受 `concurrent_limit` 限制）。
2. 自动探测远端架构，识别 `amd64` 或 `arm64`。
3. 清理并创建远端工作目录，默认 `/tmp/envcheck`。
4. 分发适配架构的 `envCheck` 二进制文件（gzip 压缩传输，远端已存在且大小一致则跳过）和节点配置（按角色裁剪端口配置）。
5. 在远端后台执行本地检查（通过隐藏命令 `run-local`）。
6. 轮询拉取远端 `result.json` 或 `error.log` 并生成汇总报告。

#### 二进制选择规则

如果远端架构已识别，工具会优先查找：

```text
./build/envCheck_<arch>
./envCheck_<arch>
```

例如：

```text
./build/envCheck_amd64
./build/envCheck_arm64
```

如果没有找到架构专用文件，则使用当前正在运行的 `envCheck` 可执行文件。

工具运行结束后会在终端输出汇总结果并生成 `dispatch.html` 文件。

## 输出说明

每次命令运行会写入日志文件，并根据命令生成 HTML 报告。

| 命令 | HTML 报告 | JSON 结果文件 |
| --- | --- | --- |
| `query` | `query.html` | `query-result.json` |
| `clean` | `clean.html` | `clean-result.json` |
| `program` | `check.html` | `program-check-result.json` |
| `clock` | `clock.html` | `clock-check-result.json` |
| `kernel` | `kernel.html` | `kernel-check-result.json` |
| `port` | `port.html` | `port-check-result.json` |
| `disk` | `disk.html` | `disk-check-result.json` |
| `route` | `route.html` | `route-check-result.json` |
| `iptables` | `iptables.html` | `iptables-check-result.json` |
| `run` | `dispatch.html` | `dispatch-result.json` |

- 当 `output_format` 为 `text` 时，结果会输出到终端，不保存 JSON 结果文件。
- 当 `output_format` 为 `json` 时，会保存对应的 JSON 结果文件。
- HTML 报告会在命令执行时始终生成。

## 架构设计

### 模块列表

| 模块名 | 功能 | 对应子命令 |
| --- | --- | --- |
| query | 残留文件查询（存在性、权限、属主） | `envCheck query` |
| clean | 残留文件清理 | `envCheck clean` |
| program | 冲突程序检查（安装状态、预期是否应存在） | `envCheck program` |
| clock | 节点间时钟同步检查 | `envCheck clock` |
| kernel | 内核版本检查（版本号可配置） | `envCheck kernel` |
| port | 端口占用检查（按角色配置端口） | `envCheck port` |
| disk | 磁盘空间检查（路径和大小可配置） | `envCheck disk` |
| route | 默认路由检查 | `envCheck route` |
| iptables | iptables FORWARD 策略检查 | `envCheck iptables` |
| dispatch | 节点分发、远程执行和结果收集 | `envCheck run` |

### 包结构

| 包路径 | 职责 |
| --- | --- |
| `cmd/` | Cobra 子命令定义与路由，包含 `run-local` 隐藏命令 |
| `pkg/config/` | 配置加载、校验、类型定义 |
| `pkg/query/` | 文件查询 |
| `pkg/clean/` | 文件清理 |
| `pkg/program/` | 程序安装状态检查 |
| `pkg/clock/` | 时钟同步检查 |
| `pkg/kernel/` | 内核版本读取与比较 |
| `pkg/port/` | TCP 端口占用探测，按角色分配检查项 |
| `pkg/disk/` | 磁盘空间统计，支持路径不存在时回退到父目录 |
| `pkg/route/` | 默认路由检查 |
| `pkg/iptables/` | iptables FORWARD 链策略检查 |
| `pkg/dispatch/` | SSH 连接管理、架构探测、文件分发、远程执行、结果轮询收集 |
| `pkg/sshutil/` | SSH 连接工具 |
| `pkg/output/` | Text/JSON 格式化输出、HTML 模板渲染（embed + html/template） |
| `pkg/logger/` | 日志记录 |
| `pkg/utils/` | 路径处理工具 |

### 命令执行框架

所有子命令通过 `cmd/utils.go` 中的统一框架执行，负责完成：加载配置 → 初始化日志 → 执行业务逻辑 → 生成 HTML 报告 → 格式化输出 → 控制台打印（text 模式）/ 保存 JSON 文件（json 模式）。

## 兼容性

- 支持 Linux 主流发行版。
- 支持 x86_64（映射为 `amd64`）和 aarch64（映射为 `arm64`）架构。
- 自动探测远程节点架构并选择对应二进制文件。

## 常见问题

### 提示 `at least one path is required`

`query` 或 `clean` 命令要求配置 `paths`。请检查配置文件路径是否正确，以及 `paths` 是否为空。

### 提示 `program list is required`

`program` 命令要求配置 `program_list`。请添加至少一个程序检查项。

### 提示 `hosts field is required`

`clock` 命令要求配置 `hosts`。请添加至少一个节点，并确保其中存在 `bootstrap` 角色节点。

### 远程检查连接失败

请检查：

- IP、端口、用户名、密码是否正确。
- 远端 SSH 服务是否可用。
- 远端是否允许密码登录。
- 当前机器到远端的网络是否连通。
- 远端用户是否有创建 `dispatch.work_dir` 的权限。

### 没有生成 JSON 文件

只有当 `output_format` 设置为 `json` 时，才会保存 JSON 结果文件。HTML 报告会在命令执行时生成。
