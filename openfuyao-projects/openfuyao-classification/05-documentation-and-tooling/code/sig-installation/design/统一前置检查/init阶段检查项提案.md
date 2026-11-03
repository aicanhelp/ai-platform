# init 阶段检查项提案

## 1. 阶段描述

`bke preflight init` 在执行 `bke init` 前，以相同业务参数检查引导节点是否满足初始化条件。该命令
只生成计划、采集只读事实并输出报告，不执行 K3s、仓库、Console 或其他组件的安装、修复和清理。

### 1.1 目标

- 与 `bke init` 共享 FlagSpec、纯 Resolver 和 Validate，保持参数、默认值和互斥关系一致。
- 使用最终解析值检查平台、资源、路径、端口、来源、离线制品和自定义仓库。
- 仅实例化当前输入适用的规则，不用“不适用”或“无法确认”制造告警。
- 对文件、来源和证书建立真实依赖，一个根因只显示一次。
- 报告明确引导节点、端口、路径、来源、期望、实际和修复动作。

### 1.2 输入与优先级

```bash
bke preflight init --hostIP 192.0.2.10 --installConsole=false

bke preflight init \
  --preflight-config ./preflight.yaml \
  --hostIP 192.0.2.10 \
  --output json --output-file report.json
```

v1 不提供 `--request`。init 业务值优先级为：显式 CLI 或业务配置值 > 当前生效 policy 的 fallback。
指定`--preflight-config`时当前policy就是该外部完整文件；未指定时才是内嵌默认文件。外部policy只改变
本次检查，不回写后续 `bke init` 的安装参数。

## 2. 计划与实例化设计

Host Adapter 将解析后的 init 输入转换为通用 OperationPlan：

```text
init CLI flags
  → 共享 Resolver / Validate
  → AdapterInput
  → OperationPlan（bootstrap、values、sources）
  → RuleBinding
  → SDK Collector / CEL / Report
```

Plan 只保存完成检查所需的非敏感最终值。用户名、密码、Token、私钥和完整 kubeconfig 不得进入
PlanHash、Fact、Evidence 或 Report。

规则适用条件：

- 平台、资源、工具、Workspace 和基础端口绑定到 bootstrap。
- Console 端口只在 `installConsole=true` 时实例化。
- 本地 NTP 端口只在最终计划确认本机提供 NTP 时实例化；外部 NTP 不检查 UDP 123。
- 来源规则按本次 init 实际使用的来源逐项实例化。
- 离线制品规则只在配置离线包时实例化。
- 自定义仓库和 CA 规则只在对应参数存在时实例化。
- 当前 init 输入不能可靠证明 bootstrap 与目标集群共部署，因此 `INIT-COLOCATE` 不实例化。
- `Applicable=false` 不产生 Result；只有适用项被依赖、超时或取消阻塞时才产生 SKIP。

## 3. 固定检查项

`来源`为 Bundle Source ID，并由 SourceLock 映射到固定 repo、revision、path、locator 和
sourceURL。固定目录包含27个模板，但每次运行的实际实例数由输入和适用条件决定。
除 `INIT-COLOCATE` 使用 `WARNING/WARN` 外，其余固定模板使用 `ERROR/BLOCK`。

### 3.1 执行环境与平台（8项）

| # | Rule ID | 检查对象 | 期望值/判断 | 条件 | 来源 |
|---:|---|---|---|---|---|
| 1 | `INIT-EXEC-001` | 本地执行身份 | UID 为0 | 始终 | `RELEASE-PROFILE` |
| 2 | `INIT-PLAT-OS` | 引导节点 OS | Linux | 始终 | `RELEASE-PROFILE` |
| 3 | `INIT-PLAT-DISTRO` | 发行版 | 命中当前策略平台矩阵 | 始终 | `RELEASE-PROFILE` |
| 4 | `INIT-PLAT-KERNEL` | 内核 | 命中发行版对应内核前缀 | 始终 | `RELEASE-PROFILE` |
| 5 | `INIT-PLAT-ARCH` | 架构 | amd64 或 arm64 | 始终 | `RELEASE-PROFILE` |
| 6 | `INIT-RES-CPU` | 逻辑 CPU | 默认不少于2核 | 始终 | `RELEASE-PROFILE` |
| 7 | `INIT-RES-MEMORY` | 可用内存 | 默认不少于4 GiB | 始终 | `RELEASE-PROFILE` |
| 8 | `INIT-RES-SYSTEM-DISK` | 系统盘容量 | 默认不少于100 GiB | 始终 | `RELEASE-PROFILE` |

### 3.2 工具、路径和端口（10项）

| # | Rule ID | 检查对象 | 期望值/判断 | 条件 | 来源 |
|---:|---|---|---|---|---|
| 9 | `INIT-TOOL-TAR` | `tar` | PATH 中存在且可执行 | 始终 | `RELEASE-PROFILE` |
| 10 | `INIT-TOOL-SYSTEMCTL` | `systemctl` | PATH 中存在且可执行 | 始终 | `RELEASE-PROFILE` |
| 11 | `INIT-PATH-CREATE` | 最终 Workspace | 已存在或父路径允许创建 | 始终 | `RELEASE-PROFILE` |
| 12 | `INIT-PATH-MOUNT` | Workspace 挂载点 | 非只读挂载 | 始终 | `RELEASE-PROFILE` |
| 13 | `INIT-PORT-K8S` | Kubernetes 服务端口 | 实际端口空闲，默认36443/TCP | 始终 | `BKE-INIT-PORTS` |
| 14 | `INIT-PORT-REGISTRY` | 镜像服务端口 | 实际端口空闲，默认40443/TCP | 始终 | `BKE-INIT-PORTS` |
| 15 | `INIT-PORT-CHART` | Chart 服务端口 | 实际端口空闲，默认38080/TCP | 始终 | `BKE-INIT-PORTS` |
| 16 | `INIT-PORT-HTTP` | 软件包服务端口 | 实际端口空闲，默认40080/TCP | 始终 | `BKE-INIT-PORTS` |
| 17 | `INIT-PORT-CONSOLE` | Console 端口 | 实际端口空闲，默认30010/TCP | Console 启用 | `BKE-INIT-PORTS` |
| 18 | `INIT-PORT-NTP` | 本地 NTP 端口 | 实际端口空闲，默认123/UDP | 本机提供 NTP | `RELEASE-PROFILE` |

路径检查只读取路径和挂载信息，不创建目录或测试文件；端口检查使用最终业务值。

### 3.3 来源、离线制品和自定义仓库（8项）

| # | Rule ID | 检查对象 | 期望值/判断 | 条件 | 来源 |
|---:|---|---|---|---|---|
| 19 | `INIT-SRC-REACHABLE` | 本次实际安装来源 | 可建立到脱敏 host:port 的 TCP 连接 | 按来源实例化 | `BKE-INIT-EXEC` |
| 20 | `INIT-OFFLINE-EXISTS` | 离线制品路径 | 文件存在 | 离线模式 | `DOC-INTEGRITY` |
| 21 | `INIT-OFFLINE-READ` | 离线制品路径 | 文件可读 | 依赖存在 | `DOC-INTEGRITY` |
| 22 | `INIT-OFFLINE-FORMAT` | 离线制品格式 | tar、tar.gz 或 tgz | 依赖可读 | `DOC-INTEGRITY` |
| 23 | `INIT-REPO-URL` | 自定义仓库地址 | URL、域名/IP 和端口可解析 | 已配置仓库 | `RELEASE-PROFILE` |
| 24 | `INIT-REPO-CA-EXISTS` | 自定义 CA 文件 | 文件存在 | 已配置 CA | `RELEASE-PROFILE` |
| 25 | `INIT-REPO-CA-READ` | 自定义 CA 文件 | 文件可读 | 依赖存在 | `RELEASE-PROFILE` |
| 26 | `INIT-REPO-CA-PARSE` | 自定义 CA 内容 | 可解析为 PEM X.509 证书 | 依赖可读 | `RELEASE-PROFILE` |

来源检查只证明 TCP 连通，不执行 HTTP、TLS、认证或对象下载。失败建议不得编造密码、证书或镜像
tag 等未被探测的原因。

### 3.4 共部署模板（1项）

| # | Rule ID | 检查对象 | 期望值/判断 | 级别与条件 | 来源 |
|---:|---|---|---|---|---|
| 27 | `INIT-COLOCATE` | bootstrap 与集群角色关系 | 可证明共部署时提示容量和故障域风险 | WARNING；当前普通 init 不实例化 | `DOC-QUICKSTART` |

### 3.5 自定义检查（0..64项）

init 自定义检查位于 `operations.init.customChecks`，仅绑定 bootstrap，不接受 roles：

| Rule ID | Collector | 判断 | 来源 |
|---|---|---|---|
| `CONFIG-INIT-CUSTOM-<KEY>` | `fileContent` 或受控 `exec` | `expression(actual)` 必须为 true | `RUNTIME-PREFLIGHT-CONFIG` |

文件原文和脚本私有字段不得进入报告。完整配置、类型和安全边界见 SDK 提案。

## 4. 依赖、错误与报告

最小依赖：

```text
离线文件存在 → 可读 → 格式
CA 文件存在 → 可读 → X.509 解析
```

根依赖失败后下游为 SKIP，默认控制台只展示根因。输入缺失、类型错误、互斥参数和外部 policy
错误通过 RunIssue 显示“前置检查未启动”，不伪装成环境 FAIL。

目标列必须包含 bootstrap 和具体对象，例如：

- `bootstrap (192.0.2.10) / TCP 36443`
- `bootstrap / /bke`
- `bootstrap / Image repository example.com:443`
- `bootstrap / /etc/openFuyao/ca.crt`

## 5. 只读、安全与质量属性

- 不创建 Workspace、测试文件或目录，不下载、解压、安装或清理制品。
- 不修改服务、端口、网络、sysctl 或系统配置。
- 凭据只存在于 Host 短生命周期执行器中，不进入计划和报告。
- 总超时覆盖解析、计划、Bundle、采集、计算和渲染前阶段；单 Fact 受独立超时约束。
- 同一 `(bootstrap,factKey)` 只采集一次，不同独立事实可有界并发。
- 输入顺序变化不应改变 PlanHash；业务端口、来源或条件变化应改变 PlanHash。

## 6. 测试与验收

必须覆盖：

- preflight 与真实 init 使用等价参数时解析结果一致，且没有安装副作用。
- 非法参数、缺文件、未知字段和 policy 错误可定位到字段或文件。
- Console、本地 NTP、离线、仓库和 CA 的适用/不适用分支。
- CPU、内存、磁盘、平台、TCP/UDP 端口和来源的边界值。
- 文件与 CA 根依赖失败时下游 SKIP。
- custom fileContent/exec 的成功、失败、类型、权限、超限和脱敏。
- text/JSON/output-file 的结论和计数一致，JSON stdout 为单一文档。
- Linux AMD64/ARM64 环境中的真实 CLI、端口占用、离线制品、仓库/CA 和只读审计。

## 7. 参考资料

- [openFuyao 统一前置检查提案](./openFuyao统一前置检查提案.md)
- [校验框架 SDK 提案](./校验框架SDK提案.md)
