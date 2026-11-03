---
# 提案标题
title: envCheck v2.0 K8s集群部署前置检查能力增强
# 提案编号
ofep-number: 0055
# 提案作者
authors:
  - "@cuileilei"
  - "@luolong817"
# 提案主导SIG
owning-sig: sig-Installation
# 提案协作SIG
participating-sigs:

# 提案状态(初步草案|准备开始实现|已在SIG中实现并合并|提案被暂缓|提案被否决|作者主动撤回提案|被另一个oFEP取代)
status: provisional
# 创建日期
creation-date: 2026-05-08
# 评审人
reviewers:

# 批准人
approvers:

# 相关联的其他oFEP
see-also:

# 当前oFEP替代了哪些已有提案
replaces:

# 当前oFEP被哪些提案替代，当status为replaced时需要填写此字段
replaced-by:

# 此oFEP当前开发阶段
stage: alpha|beta|stable
# 最近一次推进的版本
latest-milestone: "v2.0"
# 各阶段目标版本
milestone:
  alpha: "v2.0"
  beta: "v2.0"
  stable: "v2.0"
# 功能开关及影响组件
feature-gates:

# 是否支持关闭该功能
disable-supported: false
# 该功能引入的监控指标
metrics:

---
# oFEP-0055：envCheck v2.0 K8s集群部署前置检查能力增强

<!-- toc -->

[发布签核清单](#发布签核清单)

- [摘要](#摘要)
- [动机](#动机)
  - [目标](#目标)
  - [非目标](#非目标)
- [提案](#提案)
  - [用户故事](#用户故事)
    - [故事 1：默认路由验证](#故事-1默认路由验证)
    - [故事 2：容器网络验证](#故事-2容器网络验证)
    - [故事 3：必要工具验证](#故事-3必要工具验证)
    - [故事 4：统一日志管理](#故事-4统一日志管理)
    - [故事 5：内核版本合规检查](#故事-5内核版本合规检查)
    - [故事 6：端口占用预检](#故事-6端口占用预检)
    - [故事 7：磁盘空间预检](#故事-7磁盘空间预检)
    - [故事 8：多节点并行分发检查](#故事-8多节点并行分发检查)
  - [注释约束警告](#注释约束警告)
  - [风险与缓解措施](#风险与缓解措施)
- [设计细节](#设计细节)
  - [整体架构](#整体架构)
    - [命令行接口](#命令行接口)
    - [配置文件格式](#配置文件格式)
    - [输出格式约定](#输出格式约定)
  - [默认路由检查](#默认路由检查)
    - [实现方式](#实现方式)
    - [核心接口](#核心接口)
  - [iptables检查](#iptables检查)
    - [实现方式](#实现方式-1)
    - [核心接口](#核心接口-1)
  - [程序存在性检查](#程序存在性检查)
    - [实现要点](#实现要点)
  - [日志规范整改](#日志规范整改)
    - [日志接口](#日志接口)
  - [内核检查模块 (kernel)](#内核检查模块-kernel)
  - [端口检查模块 (port)](#端口检查模块-port)
  - [磁盘检查模块 (disk)](#磁盘检查模块-disk)
  - [节点分发模块 (dispatch)](#节点分发模块-dispatch)
  - [输出与报告](#输出与报告)
  - [多架构构建](#多架构构建)
  - [测试计划](#测试计划)
    - [单元测试](#单元测试)
    - [集成测试](#集成测试)
    - [e2e 测试](#e2e-测试)
  - [毕业标准](#毕业标准)
  - [升级降级策略](#升级降级策略)
  - [版本倾斜策略](#版本倾斜策略)
- [生产可用性审查问卷](#生产可用性审查问卷)
  - [功能启用和回滚](#功能启用和回滚)
  - [推出、升级和回滚规划](#推出升级和回滚规划)
  - [监控要求](#监控要求)
  - [依赖项](#依赖项)
  - [可扩展性](#可扩展性)
  - [故障排除](#故障排除)
- [实施历史](#实施历史)
- [缺点](#缺点)
- [替代方案](#替代方案)
- [所需基础设施](#所需基础设施)

<!-- /toc -->

## 发布签核清单

- [ ] (R) 发布里程碑中的增强问题
- [ ] (R) oFEP 审批者已批准 oFEP 状态为可实施
- [ ] (R) 设计细节已适当记录
  - [ ] (R) 测试计划已到位
    - [ ] 单元测试覆盖核心检查逻辑
    - [ ] 集成测试验证多节点分发流程
- [ ] (R) 毕业标准已设定
- [ ] (R) 生产准备情况审查完成
- [ ] (R) 生产准备情况审查已获批准
- [ ] 实施历史部分已更新里程碑
- [ ] 面向用户的文档已创建
- [ ] 支持文档 - 设计文档、相关 PR问题、发行说明

## 摘要

本文档描述 envCheck 工具的功能增强设计方案，包括新增默认路由检查、iptables 规则检查、程序存在性检查增强、日志规范整改、OS 内核检查、端口占用检查、磁盘空间检查以及节点分发执行八个功能模块。envCheck 是用于检查部署环境的预检查验证工具，本次增强旨在满足更多环境检查场景的需求，提供更全面的环境验证能力。

新增功能包括：

- **默认路由检查**：检查系统默认路由是否存在，输出路由详细信息
- **iptables 规则检查**：检查 FORWARD 链默认策略是否为 ACCEPT，确保容器网络转发正常
- **程序存在性检查增强**：支持配置期望状态（应存在或不应存在），验证冲突应用或必要工具
- **日志规范整改**：将自定义日志替换为 ologger 统一日志框架
- **OS 内核检查**：检查内核版本是否满足最低版本要求，支持多种比较运算符
- **端口占用检查**：检查指定端口是否被占用，支持按节点角色（bootstrap/master/worker）检查不同端口集合
- **磁盘空间检查**：检查指定路径的磁盘空间是否充足，支持按节点角色配置不同的检查项
- **节点分发执行**：支持将检查工具分发到多个节点并行执行，统一收集结果并生成汇总报告

通过这些增强，envCheck 将能够覆盖 K8s 集群部署前常见的环境合规性检查场景，减少因环境配置不当导致的部署失败。

## 动机

在实际的 K8s 集群部署过程中，部署前环境检查是保障安装成功率的关键步骤。常见的问题包括：

- 节点没有默认路由，导致集群内外流量在三层路由器上转发失败
- iptables规则不符合要求，导致流量丢失
- 环境中缺少必要的工具，导致相应操作失败
- 日志规格未统一，定位获取信息不足
- 操作系统内核版本过低，导致容器运行时或网络组件不兼容
- 目标端口被占用，导致 kube-apiserver、etcd 等服务启动失败
- 磁盘空间不足，导致镜像拉取或日志写入失败
- 多节点环境下手动逐一检查效率低下，且容易遗漏

现有的 envCheck 工具仅支持文件残留检查、程序冲突检查和时钟同步检查，缺乏对默认路由、iptables规则、必要工具、内核版本、端口占用、磁盘空间等关键环境指标的自动化检查能力，也无法高效地在多节点场景下并行执行检查任务。

因此，我们需要对 envCheck 进行能力扩展，使其成为一站式的 K8s 集群部署前置环境检查工具。

### 目标

- 提供默认路由检查功能，验证网络配置正确性
- 提供 iptables 规则检查功能，确保容器网络转发正常
- 扩展程序存在性检查，支持配置期望状态，支持识别冲突应用（如 docker、kubectl）和必要工具（如 tar）
- 将自定义日志实现替换为 ologger 统一日志框架
- 新增内核版本检查功能，支持可配置的版本号、比较操作符，并对 CentOS 7 旧构建号做特殊兼容
- 新增端口占用检查功能，支持按节点角色（bootstrapmasterworker）配置检查端口列表
- 新增磁盘空间检查功能，支持按路径、最小可用空间、角色进行可配置检查
- 新增节点分发执行功能，支持通过 SSH 将二进制分发到远程节点并行执行检查，统一收集结果并生成 HTMLJSON 报告
- 支持多架构构建（amd64arm64），分发执行时自动探测远程节点架构并匹配对应二进制

### 非目标

- 不涉及网络修复功能，仅提供检查和诊断
- 不修改 iptables 规则，仅检查策略状态
- 不自动安装或卸载程序，仅验证存在性
- 不替代现有的 AnsiblePuppet 等配置管理工具进行环境修复，envCheck 只负责检查不负责修复
- 不支持非 Linux 操作系统（Windows、macOS）的远程节点分发检查
- 不涉及对 K8s 集群运行时的动态监控，仅作为部署前的静态环境检查工具
- 不引入 Web UI 或服务端，保持命令行工具的单体形态

## 提案

### 用户故事

#### 故事 1：默认路由验证

作为一名集群部署工程师，我希望在部署集群之前检查默认路由是否存在，以便确保网络通信正常。

#### 故事 2：容器网络验证

作为一名集群部署工程师，我希望检查 iptables FORWARD 链策略，以便确保容器网络转发不会被阻断。

#### 故事 3：必要工具验证

作为一名集群部署工程师，我希望在部署 K8s 之前自动检查所有节点的内核版本是否满足要求（例如 = 4.18），以便提前发现并升级不符合要求的节点，避免容器运行时启动失败。

#### 故事 4：统一日志管理

作为一名集群部署工程师，我希望检查工具的所有日志输出符合统一规范，以便集中收集和分析日志。

#### 故事 5：内核版本合规检查

作为一名集群部署工程师，我希望在部署 K8s 之前自动检查所有节点的内核版本是否满足要求（例如 = 4.18），以便提前发现并升级不符合要求的节点，避免容器运行时启动失败。

#### 故事 6：端口占用预检

作为一名集群部署工程师，我希望根据节点角色检查特定端口是否被占用。例如，master 节点需要检查 6443、30029 等端口，worker 节点需要检查 10250、10256 等端口，bootstrap 节点需要检查 36443、40080 等端口。各角色的端口列表相互独立、可配置不同端口号。这样可以在部署前发现并解决端口冲突。

#### 故事 7：磁盘空间预检

作为一名集群部署工程师，我希望检查关键路径（如 varlibdocker、varlibkubelet、varlibetcd）的磁盘空间是否满足最小要求。对于 etcd 数据目录，我只想在 master 节点上检查。这样可以在部署前确保磁盘资源充足。

#### 故事 8：多节点并行分发检查

作为一名集群部署工程师，我有 100 台节点需要检查。我希望将 envCheck 工具自动分发到所有节点上并行执行，然后统一收集结果生成一个汇总报告。分发过程应该支持 gzip 压缩、自动跳过已上传的文件、自动探测节点架构（x86_64aarch64）并选择对应二进制，整个流程应该在 10 分钟内完成。

### 注释约束警告

- CentOS 7 内核兼容：针对 CentOS 7 的 `3.10.0-xxx` 内核版本，若版本字符串长度超过 12 位且第 7~10 位包含 `.`（旧版构建号格式），则在 `=`  `` 比较时视为不满足，避免旧构建号被误判为通过
- 角色分配逻辑：
  - `bootstrap` → 引导节点端口检查
  - `master` → Master 节点端口检查
  - `worker` → Worker 节点端口检查
- 超时时间可配置：端口检查 TCP 探测的超时时间可通过配置文件中的 `port_check.timeout` 设置，默认 3 秒
- 磁盘路径回退：若配置的磁盘路径不存在，自动向上查找最近的父目录进行空间统计；若路径是文件，则取其所在目录进行统计
- 分发配置裁剪：Dispatcher 在上传配置文件时，会根据节点角色对配置进行裁剪。对于 `port_check`，只保留该节点角色对应的端口列表

### 风险与缓解措施

 SSH 连接失败导致部分节点无法检查  中等  单节点连接失败不影响其他节点，最终结果中会标记为 failed
 大文件传输导致分发耗时过长  中等  采用 gzip 压缩传输；若远程已存在同名文件且大小一致则跳过上传
 端口探测存在误判（某些服务短暂监听）  低  使用 TCP 连接探测，超时时间可配置（默认 3 秒），结果仅作为参考
 内核版本字符串比较规则与预期不符  低  针对 CentOS 7 做了特殊兼容处理，并在文档中明确说明比较规则

## 设计细节

### 整体架构

envCheck 工具采用命令行工具架构，基于 Cobra 框架构建子命令体系，每个检查功能作为独立的子命令实现。设计前提包括：

1. **运行环境**：Linux 操作系统（网络检查功能暂不支持 Windows）
2. **权限要求**：部分功能（如 iptables）需要 root 或 CAP_NET_ADMIN 权限
3. **依赖工具**：系统需预装 `ip` 或 `route` 命令（默认路由检查）、`iptables` 命令（规则检查）
4. **配置格式**：JSON 格式配置文件

所有子命令通过 `cmdutils.go` 中的统一框架 `actionHandler` + `runAction` 执行，负责配置加载、日志初始化、业务执行、HTML 生成、格式化输出、文件保存等通用流程。

#### 命令行接口

```bash
# 完整命令列表
envCheck query      # 查询残留文件
envCheck clean      # 清理残留文件
envCheck program    # 程序存在性检查
envCheck clock      # 时钟同步检测
envCheck route      # 默认路由检查（新增）
envCheck iptables   # iptables 规则检查（新增）
envCheck kernel     # 内核版本检查（新增）
envCheck port       # 端口占用检查（新增）
envCheck disk       # 磁盘空间检查（新增）
envCheck run        # 多节点分发执行（新增）

# 全局参数
--config string      配置文件路径（默认 "config.json"）
--output string      输出格式：text、json（默认 "text"）

# run 命令特有参数
--roles string       目标节点角色（逗号分隔，如 "master,worker"）
--checks string      要执行的检查项（逗号分隔）
--skip string        要跳过的检查项（逗号分隔）
--timeout int        执行超时时间（秒）
--run-local          仅在本地执行（内部参数）
```

#### 配置文件格式

```json
{
  "mode": "check",
  "log_file": "envcheck.log",
  "output_format": "text",
  "program_list": [
    {
      "name": "docker",
      "should_exist": false
    },
    {
      "name": "kubectl",
      "should_exist": false
    },
    {
      "name": "tar",
      "should_exist": true
    }
  ],
  "clock_threshold": 5,
  "kernel_check": {
    "min_version": "4.18",
    "operator": ">="
  },
  "port_check": {
    "ports": {
      "bootstrap": ["36443", "40080", "40443", "38080"],
      "master": ["6443", "30029", "30909", "30903", "3030", "30019", "9012"],
      "worker": ["10250", "10256", "30000", "30001", "30022", "30033", "30024"]
    },
    "timeout": 3
  },
  "disk_check": {
    "check_items": [
      {
        "path": "/",
        "min_free_gb": 50,
        "roles": []
      },
      {
        "path": "/var/lib/docker",
        "min_free_gb": 80,
        "roles": ["master", "worker"]
      },
      {
        "path": "/var/lib/openFuyao/etcd",
        "min_free_gb": 20,
        "roles": ["master"]
      }
    ]
  },
  "hosts": [
    {
      "ip": "192.168.1.1",
      "username": "root",
      "password": "******",
      "port": "22",
      "role": ["bootstrap"]
    },
    {
      "ip": "192.168.1.2",
      "username": "root",
      "password": "******",
      "port": "22",
      "role": ["master"]
    }
  ],
  "dispatch": {
    "timeout": 600,
    "poll_interval": 15,
    "work_dir": "/tmp/envcheck",
    "concurrent_limit": 10,
    "checks": [],
    "skip_checks": []
  }
}
```

#### 输出格式约定

**JSON 格式输出结构**：

```json
{
  "timestamp": "2025-01-01T00:00:00Z",
  "check_type": "route|iptables|program|kernel|port|disk|dispatch",
  "results": { ... },
  "summary": {
    "check_passed": true|false,
    "message": "...",
    "total_checked": 10,
    "passed": 8,
    "failed": 2
  }
}
```

**节点分发输出结构**：

```json
{
  "timestamp": "2025-01-01T00:00:00Z",
  "check_type": "dispatch",
  "nodes": [
    {
      "host": "192.168.1.1",
      "success": true,
      "results": { ... },
      "error": ""
    }
  ],
  "summary": {
    "total_nodes": 5,
    "success_nodes": 4,
    "failed_nodes": 1,
    "message": "..."
  }
}
```

### 默认路由检查

#### 实现方式

默认路由检查通过以下方式实现：

1. 执行 `ip route show default` 或 `route -n` 命令获取所有默认路由
2. 解析命令输出，提取每条路由信息（网关、接口、metric等）
3. 如果存在默认路由，返回所有默认路由详细信息列表；否则返回空列表

#### 核心接口

```go
type RouteChecker struct {
    config *config.AppConfig
    logger logger.Logger
}

func NewRouteChecker(cfg *config.AppConfig, log logger.Logger) *RouteChecker
func (r *RouteChecker) Execute() (*RouteCheckResult, error)
```

### iptables检查

#### 实现方式

iptables规则检查通过以下方式实现：

1. 执行 `iptables -L FORWARD` 命令获取FORWARD链信息
2. 解析命令输出，提取默认策略（policy）
3. 判断策略是否为ACCEPT

#### 核心接口

```go
type IptablesChecker struct {
    config *config.AppConfig
    logger logger.Logger
}

func NewIptablesChecker(cfg *config.AppConfig, log logger.Logger) *IptablesChecker
func (i *IptablesChecker) Execute() (*IptablesCheckResult, error)
```

### 程序存在性检查

#### 实现要点

程序存在性检查的实现要点：

1. 遍历配置中的程序列表，检查每个程序的存在性
2. 使用 `exec.LookPath()` 检查程序是否存在
3. 执行版本命令获取版本信息，使用正则表达式提取并保留纯版本号
4. 对比实际存在性与配置中的 `should_exist`，判断是否通过
5. 当 `installed=true` 且 `should_exist=true`，或 `installed=false` 且 `should_exist=false` 时，`passed=true`
6. 汇总统计中包含已安装、未安装、通过、失败的计数

### 日志规范整改

将当前的自定义日志实现替换为 ologger 统一日志框架，以符合项目的日志规范要求。将日志实现替换为 ologger，位于 `pkg/logger/logger.go`。该实现通过适配器模式，将 ologger 封装为项目所需的 Logger 接口。

#### 日志接口

```go
type Logger interface {
    Info(message string)
    Success(message string)
    Warning(message string)
    Error(message string)
}
```

### 内核检查模块 (kernel)

- 实现包：`pkgkernel`
- 核心接口：
  ```go
  func NewChecker(cfg config.KernelCheckConfig, log logger.Logger) Checker
  func (c Checker) Execute() (config.KernelCheckResult, error)
  ```
- 实现细节：
  - 使用 `github.comshirougopsutilv3host` 读取当前系统内核版本
  - 与配置文件中指定的期望版本按字符串进行比较
  - 支持比较操作符：`=`、``、`==`、`=`、``
  - 针对 CentOS 7 旧构建号做特殊兼容处理

### 端口检查模块 (port)

- 实现包：`pkgport`
- 核心接口：
  ```go
  func NewChecker(cfg config.PortCheckConfig, log logger.Logger) Checker
  func (c Checker) Execute(roles []string) config.PortCheckResult
  ```
- 实现细节：
  - 使用 `net.DialTimeout` 连接 `127.0.0.1端口`，超时时间可配置（默认 3 秒）
  - 根据节点角色过滤配置中对应的端口列表，支持 `bootstrap`  `master`  `worker` 三个角色
  - 各角色的端口列表相互独立，可配置不同端口号
  - 输出每个端口的占用状态和关联进程（若可获取）

### 磁盘检查模块 (disk)

- 实现包：`pkgdisk`
- 核心接口：
  ```go
  func NewChecker(cfg config.DiskCheckConfig, log logger.Logger, roles []string) Checker
  func (c Checker) Execute() (config.DiskCheckResult, error)
  ```
- 实现细节：
  - 使用 `gopsutildisk` 获取路径所在分区的磁盘使用情况
  - 路径不存在时向上查找父目录；路径是文件时取其目录
  - 根据 `roles` 过滤 `check_items` 中带有 `roles` 限制的检查项
  - 输出总空间、可用空间、已用百分比、文件系统类型等信息

### 节点分发模块 (dispatch)

- 实现包：`pkgdispatch`
- 核心接口：
  ```go
  func NewDispatcher(cfg config.DispatchConfig, appCfg config.AppConfig, log logger.Logger) Dispatcher
  func (d Dispatcher) Execute() (config.DispatchResult, error)
  func SaveResults(results []config.CheckResult, resultPath string) error
  ```
- 执行流程：
  1. 建立到所有目标节点的 SSH 连接（并发，受 `concurrent_limit` 限制）
  2. 检测远程节点架构（`x86_64` → `amd64`，`aarch64` → `arm64`）
  3. 清理远程工作目录
  4. 分发 envCheck 二进制和配置文件（gzip 压缩，大小一致则跳过）
  5. 在目标节点后台启动 `envCheck run-local` 执行检查
  6. 轮询收集各节点的 `result.json` 或 `error.log`
  7. 汇总生成统一的 HTMLJSON 报告

格式  说明
----------

 `text`  控制台表格输出（默认），使用 `gotable` 格式化
 `json`  保存为 `{baseName}.json` 文件
 `html`  生成对应的 `.html` 报告文件（通过 embed + template）

HTML 模板文件位于 `pkgoutput` 目录下。

### 测试计划

[ x ] 我我们理解，相关组件的所有者可能会要求更新已有的测试，以便在提交实现该增强功能所需的更改之前，使代码达到足够稳固的质量标准。

#### 单元测试

- `pkgkernel` 内核版本读取与比较逻辑单元测试
- `pkgport` TCP 端口探测与角色过滤逻辑单元测试
- `pkgdisk` 磁盘空间统计与路径回退逻辑单元测试
- `pkgdispatch` SSH 连接管理、文件分发、结果收集逻辑单元测试
- `pkgconfig` 配置加载与类型校验单元测试
- `pkgoutput` 格式化输出与 HTML 生成单元测试

#### 集成测试

- 验证 `envCheck run` 在本地模拟多节点场景下的完整分发-执行-收集流程
- 验证不同角色过滤条件下配置裁剪的正确性
- 验证内核检查对 CentOS 7 特殊版本字符串的兼容行为

#### e2e 测试

- 在真实的 3 节点环境（1 bootstrap + 1 master + 1 worker）中执行完整检查流程
- 验证 HTML 报告和 JSON 结果文件的生成内容与格式

### 毕业标准

#### Alpha 阶段

- [ ] 功能已实现并可通过命令行调用
- [ ] 初步的端到端测试已完成

#### Beta 阶段

- [ ] 收集到内部用户反馈并进行了优化
- [ ] 所有核心功能均已实现
- [ ] 所有安全相关机制均已完备（SSH 密码认证、超时控制、错误隔离）
- [ ] 所有测试要求均已满足

#### GA 阶段

- [ ] 已在多个真实环境中完成部署验证
- [ ] 所有在 Beta 阶段反馈的问题与缺陷都已解决

### 升级降级策略

- 本增强功能以向后兼容的方式添加到现有 envCheck 工具中
- 原有配置项（`paths`、`program_list`、`hosts`、`clock_threshold`）保持不变
- 新增配置项（`kernel_check`、`port_check`、`disk_check`、`dispatch`）为可选配置，未配置时对应子命令会给出明确提示或跳过
- 升级时无需修改现有调用方式，仅当需要使用新功能时才在配置文件中添加新字段

### 版本倾斜策略

- envCheck 作为独立的命令行工具，不涉及与控制面组件的版本协同
- 分发执行时，控制端和远程节点使用同一版本的二进制文件，因此不存在版本倾斜问题
- 配置文件中的字段增加不影响旧版本工具的解析（忽略未知字段）

## 生产可用性审查问卷

### 功能启用和回滚

###### 如何在实时集群中启用禁用此功能？

- 其他机制：envCheck 是一个独立的命令行工具，不需要部署到集群中常驻运行。用户需要使用时直接执行即可，不使用时不产生任何影响。

###### 启用该功能会改变任何默认行为吗？

不会。本增强功能完全向后兼容，原有子命令（query、clean、check、clock）的行为没有任何改变。

###### 该功能一旦启用，是否可以禁用？

可以。由于 envCheck 是独立工具，用户可以选择不使用新增子命令（kernel、port、disk、run）。不影响现有工作负载。

###### 如果该功能之前已回滚，现在我们重新启用它会发生什么情况？

不会有任何副作用。重新使用新增子命令即可恢复功能。

###### 是否有任何针对功能启用禁用的测试？

已在单元测试中覆盖配置项缺失时的默认行为（如 `kernel_check` 未配置时子命令会提示跳过）。

### 推出、升级和回滚规划

###### 部署或回滚为何会失败？这会影响正在运行的工作负载吗？

- 部署失败可能由于远程节点 SSH 连接不通、密码错误、架构不匹配导致二进制无法执行
- 这些失败仅影响 envCheck 检查流程本身，不会影响节点上正在运行的工作负载
- 单节点失败会被隔离，不会级联影响其他节点

###### 哪些具体指标应该通知回滚？

- 分发执行时成功率低于预期阈值
- 大量节点出现 SSH 连接超时或架构探测失败
- 结果收集阶段大量节点超时未返回

###### 升级和回滚测试了吗？

已手动验证：从旧版本配置升级到新版本配置无需任何修改，新增字段为可选。回滚到旧版本时，只需不使用新增子命令即可。

###### 推出时是否伴随任何功能、API、API 类型的字段、标志等的弃用和或删除？

否。本次增强没有弃用或删除任何现有功能、字段或标志。

### 监控要求

###### 操作员如何确定该功能是否正在被工作负载使用？

envCheck 作为命令行工具，没有常驻工作负载。操作员可以通过检查执行日志（`envCheck.log`）和生成的 HTMLJSON 报告来确认工具的使用情况。

###### 使用此功能的人如何知道它适用于他们的实例？

- 命令行输出会直接显示检查结果（text 模式）
- JSON 文件保存了完整的结构化结果
- HTML 报告提供了可视化的汇总和明细

###### 增强的合理 SLO 是什么？

- 单节点检查耗时  10s
- 100 节点分发耗时  10min
- 文件分发阶段 gzip 压缩后传输，大小一致时跳过上传

###### 运维人员可以使用哪些服务级别指标（SLIs）来判断服务的健康状况？

- 分发汇总报告中的 `success_nodes  total_nodes` 比例
- 各单项检查中的 `passed  total_checked` 比例
- 日志中记录的各阶段耗时和错误数量

###### 是否存在任何尚未覆盖的指标，可以用来进一步提升该功能的可观测性？

目前尚未引入 Prometheus 等结构化指标暴露机制。考虑到 envCheck 是临时执行工具而非常驻服务，日志和报告已能满足基本可观测性需求。未来如需增强，可考虑在报告中输出各阶段的耗时分布统计。

### 依赖项

- SSH 服务

  - 使用说明：节点分发模块依赖目标节点上的 SSH 服务（默认 22 端口）可用
  - 若该服务发生中断，对该功能的影响：无法建立连接，该节点会被标记为 failed
  - 若该服务性能下降，对该功能的影响：连接建立和文件传输变慢，整体分发耗时增加
- gopsutil 库

  - 使用说明：内核检查和磁盘检查依赖 `github.comshirougopsutilv3`
  - 若该库在特定操作系统上存在兼容性问题，可能导致检查结果获取失败

### 可扩展性

###### 启用使用此功能会导致任何新的 API 调用吗？

不会。envCheck 是离线命令行工具，不涉及任何 API Server 调用。

###### 启用使用此功能是否会导致引入新的 API 类型？

不会。

###### 启用使用此功能是否会导致对云提供商的任何新的 API 调用？

不会。

###### 启用使用此功能是否会导致现有 API 对象的大小或数量增加？

不会。

###### 启用使用此功能是否会导致现有 SLISLO 所涵盖操作的耗时增加？

不会。

###### 启用使用此功能是否会导致任何组件的资源使用率不可忽略的增加？

- envCheck 作为临时执行工具，执行期间资源占用有限
- 分发执行时并发 SSH 连接数可配置（默认 10），不会对控制节点造成显著网络或内存压力
- 远程节点上 `run-local` 为一次性执行，执行完毕后立即退出，不会留下常驻进程

###### 启用使用此功能是否会导致某些节点资源（PID、套接字、inode 等）耗尽？

不会。envCheck 检查完成后会清理远程工作目录，且进程立即退出。极端情况下若节点本身资源紧张，超时机制会确保不会长时间占用资源。

### 故障排除

###### 如果 API 服务器和或 etcd 不可用，此功能如何反应？

envCheck 是独立工具，不依赖 K8s API Server 或 etcd。即使集群尚未部署，也可以正常运行。

###### 其他已知故障模式有哪些？

- SSH 连接失败

  - 检测方式：查看 `dispatch.html` 或 `dispatch-result.json` 中节点的 `status` 字段是否为 `failed`，以及 `error` 字段内容
  - 缓解手段：检查目标节点网络连通性、SSH 端口、用户名密码配置
  - 诊断信息：日志级别 `ERROR`，内容为 `failed to connect to ip error`
  - 测试情况：已在集成测试中覆盖 SSH 连接超时场景
- 架构探测失败导致二进制分发失败

  - 检测方式：查看节点错误信息中是否包含 `architecture` 或 `binary not found`
  - 缓解手段：确保 `build` 目录下存在对应架构的二进制文件（`envCheck_amd64` 或 `envCheck_arm64`）
  - 诊断信息：日志级别 `ERROR`，内容为 `unsupported architecture arch`
  - 测试情况：已覆盖 amd64arm64 两种架构的映射逻辑
- 结果收集超时

  - 检测方式：节点 `status` 长时间为 `running`，最终变为 `failed`，`error` 包含 `timeout`
  - 缓解手段：增加 `--timeout` 参数值；检查远程节点上 `envCheck run-local` 是否正常启动
  - 诊断信息：日志级别 `WARNERROR`，记录轮询次数和超时信息
  - 测试情况：已在单元测试中覆盖超时处理逻辑

###### 如果未满足 SLO，应采取哪些步骤来确定问题？

1. 检查 `envCheck.log` 中各阶段的耗时记录
2. 检查网络带宽和延迟（尤其是大量节点场景）
3. 检查远程节点的 CPUIO 负载（影响 `run-local` 执行速度）
4. 逐步减少 `concurrent_limit` 观察是否因并发过高导致资源竞争

## 实施历史

- 2025-04-08：实现内核检查、端口检查、磁盘检查三个基础模块
- 2025-04-09：实现节点分发模块（dispatch），支持 SSH 分发、架构探测、结果收集
- 2025-04-10：集成所有模块，完成 `run` 和 `run-local` 命令
- 2025-04-14：优化多节点分发性能（并发控制、gzip 压缩、跳过已存在文件）
- 2025-04-15：完成全部功能开发，通过内部验证，标记为 implemented

## 缺点

- 功能边界有限：envCheck 仅做检查不做修复，用户发现问题后仍需手动处理
- SSH 密码认证的安全性问题：当前使用明文密码进行 SSH 认证，对于生产环境安全性要求较高的场景可能不适用。未来可考虑支持 SSH Key 认证
- 无 Web UI：对于不熟悉命令行的用户，纯 CLI 工具的友好度不如 Web UI

## 替代方案

- 使用 Ansible Playbook 替代：Ansible 也可以实现多节点并行检查和报告生成。但 Ansible 依赖 Python 环境和额外的 Playbook 维护成本，而 envCheck 是单一二进制文件，部署和使用更简单。因此保留了 envCheck 的独立工具定位。
- 将功能拆分为多个独立工具：例如分别做一个 `kernel-check`、一个 `port-check`。但这会增加用户的学习成本和工具管理成本，与一站式前置检查的目标相悖，因此选择将所有功能集成在一个工具中。

## 所需基础设施

无额外基础设施需求。现有 git 仓库和 CI 流程已满足构建和测试需要。
