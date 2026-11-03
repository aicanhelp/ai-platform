# openFuyao 统一前置检查提案

## 1. 特性描述

### 1.1 背景

`bke init` 和 `bke cluster create` 在真正安装前依赖业务输入、管理集群、本机和目标节点的多类
条件。现有错误通常在安装过程中才暴露，且网络、认证、权限、资源和配置问题容易被混在一起，
增加定位和恢复成本。

本特性在 bke CLI 中提供独立、只读、可追溯的统一前置检查：

```bash
bke preflight init [与 bke init 对等的业务参数]

bke --kubeconfig /path/to/management.kubeconfig \
  preflight cluster-create \
  --file ./cluster.yaml \
  --nodes ./nodes.yaml
```

前置检查负责生成最终检查计划、采集事实、使用发布规则计算结果并输出可修复报告；它不自动执行
安装命令，不安装、修复、清理或修改 Kubernetes 资源。用户处理 FAIL 和 WARN 后，再显式执行真正
的业务命令。

### 1.2 目标

- preflight 与业务命令复用输入解析、默认化和本地校验逻辑。
- 固定规则来自版本化 PreflightBundle，每条规则有真实、不可变的来源证据。
- Collector 只读采集，CEL 统一完成类型化计算和比较。
- 使用依赖图抑制单根因多 FAIL，并保留详细 SKIP 审计链路。
- 文本和 JSON 都能说明目标、期望、实际、原因和修复建议。
- 输入、Bundle、内部错误与真实环境不满足分别表达。
- 支持少量、受控、可审计的现场自定义检查。

### 1.3 非目标

- 不覆盖 build、Web、installer-service、website、升级和扩缩容。
- 不提前执行安装流程，不做 Kubernetes 写入或 server-side dry-run。
- 不自动修复、清理、启停服务、修改 sysctl 或创建测试文件。
- 不把历史 envCheck、node-checker 等工具的规则并集作为长期来源。
- 不增加无明确当前来源的残留路径、时钟、默认路由、iptables、SELinux、Swap、runtime socket、
  50 GiB 等泛化检查。
- 不提供 `--request`、Bundle 覆盖或 Provider defaults 文件覆盖参数。
- 不提供通用表达式 DSL、CEL I/O 函数、内联 shell、脚本上传下载和隐式 sudo。

### 1.4 使用者与职责

| 角色 | 职责 |
|---|---|
| 安装操作员 | 使用与业务命令相同的输入运行检查，修复 FAIL、确认 WARN、保存报告 |
| BKE 开发者 | 维护共享 Resolver/Validate、Host Adapter、Collector 路由和 CLI |
| Provider 开发者 | 提供可复用的纯默认化与校验事实，不让 preflight 调用写路径 |
| 发布工程师 | 审核默认 preflight.yaml、静态规则来源和发布门禁结果 |
| 文档维护者 | 同步阶段规则、命令、配置、报告、限制和修复说明 |

### 1.5 依赖与 License

本特性依赖 bkeadm、cluster-api-provider-bke、Kubernetes 只读 API、CEL-Go、go-pretty 和 SSH
客户端能力。代码随 bkeadm 使用木兰宽松许可证第2版；第三方依赖必须经过仓库 License、漏洞和
供应链审查。

## 2. 需求与场景

### 2.1 用户流程

init：

```bash
bke preflight init \
  --hostIP 192.0.2.10 \
  --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest \
  --installConsole=false

# 用户处理报告后显式执行
bke init \
  --hostIP 192.0.2.10 \
  --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest \
  --installConsole=false
```

cluster-create：

```bash
bke --kubeconfig /path/to/management.kubeconfig \
  preflight cluster-create -f cluster.yaml -n nodes.yaml

# 用户处理报告后显式执行
bke --kubeconfig /path/to/management.kubeconfig \
  cluster create --file cluster.yaml --nodes nodes.yaml
```

前置检查不缓存“已检查”状态。两次命令之间输入和环境可能变化，PASS 只表示报告中适用条件在检查
时满足，不保证后续安装必然成功。

### 2.2 需求分解

| 编号 | 需求 |
|---|---|
| R01 | 提供独立的 `preflight init` 和 `preflight cluster-create` 命令 |
| R02 | 与对应业务命令共享输入解析、默认化和校验语义 |
| R03 | 使用产品无关的 OperationPlan、RuleBinding、Fact 和 Report SDK |
| R04 | 使用固定发布 Bundle、SourceLock 和摘要保证规则可追溯 |
| R05 | Collector 与 CEL 分离，采集只读，计算唯一 |
| R06 | 支持依赖去噪、超时、取消和有界并发 |
| R07 | 支持完整外部 preflight.yaml 和受限 customChecks |
| R08 | 输出可读文本、详细报告和单一 JSON 文档 |
| R09 | 所有可见问题可定位、可解释并提供修复动作 |
| R10 | 建立生成、静态校验、单元测试和真实环境发布门禁 |

### 2.3 威胁建模

| 风险 | 影响 | 控制 |
|---|---|---|
| 安装凭据或自定义敏感值进入报告 | 密码、Token、私钥泄露 | Host 凭据只存于 executor/client；custom 使用 Display/publicFields 投影，scalar/list 由提供者按公开值审核 |
| 自定义脚本执行任意命令 | 环境被修改 | 禁止 raw shell/inline/`bash -c`/upload/sudo/env；只允许预装、安全权限的绝对路径脚本 |
| 恶意配置或输出 | 内存、CPU 或日志耗尽 | 严格 schema、大小限制、CEL 长度/成本、Fact 数量与类型契约 |
| TCP 结果被过度解释 | 产生错误修复建议 | transport、auth、privilege 分层；TCP 不解释为 HTTP/TLS/auth |
| 根依赖失败继续探测 | 告警风暴和额外风险 | 显式 DAG，下游 SKIP，不回退错误 Collector |
| 规则或来源漂移 | 不同二进制得出不可审计结果 | 内嵌 policy，进程内 Bundle 携带 PolicyDigest、BundleDigest 和 SourceLock |

## 3. 总体设计

### 3.1 架构

```text
CLI flags / cluster.yaml / nodes.yaml / kubeconfig
                │
                ▼
        BKE Host Adapter
  - 共享 Resolver / Validate
  - 只读 Provider defaults / management facts
  - 生成 OperationPlan / RuleBinding / RunIssue
                │
                ▼
         通用 preflight SDK  ◀── 内嵌 policy + PreflightBundle
  - Plan/Bundle 契约校验       - 固定 Rule catalog
  - DAG / Collector            - SourceLock
  - CEL evaluator              - policy/bundle digest
                │
                ▼
     Report → 终端表格 / 详细文本 / JSON
```

### 3.2 组件职责

| 组件 | 输入 | 输出 | 责任边界 |
|---|---|---|---|
| CLI | flags、kubeconfig、运行参数 | 已解析请求和总体 context | 不执行安装；预期输入错误进入 Report |
| Host Resolver | init flags 或 cluster/nodes | 已默认化 AdapterInput | 复用业务纯逻辑；单次解析；无持久副作用 |
| Management/Provider | kubeconfig、ConfigMap 和 Provider | defaults、facts、ResolvedCluster | 同一 context；只读；错误准确归因 |
| Host Adapter | AdapterInput、policy | Plan、Bindings、Facts、Issues | 稳定 PlanHash；无敏感信息；不适用项不实例化 |
| Bundle loader | policy、Bundle、release | 已验证规则目录 | 校验摘要、DAG、SourceLock、类型和配对 |
| Collector Router | Subject、fact key | 一个 Fact | 只读；identity、type/value、Evidence 严格 |
| Engine | Rule、Binding、Fact | Result、Overall | 依赖后采集；类型错误不降级为环境 FAIL |
| Renderer | Report | text、JSON | 不重新计算状态、阈值、计数和退出码 |

### 3.3 执行流程

```text
建立总体 context
  → 加载内嵌 policy，并在进程内构造和验证 Bundle
  → 可选严格加载外部 preflight.yaml
  → 解析和校验业务输入
  → 读取 Provider defaults / management facts
  → 生成并校验 PlanHash、Bindings 和 RunIssues
  → 加载规则并构建实例 DAG
  → 按依赖和 (subject,key) 去重采集
  → 校验 Fact identity/type/value
  → CEL 计算四态 Result
  → 汇总 Report、渲染和返回退出码
```

输入、策略、Bundle 或计划问题产生 `BLOCK_START` 时不启动 Collector。独立环境分支可继续执行；
Collector 或 Adapter 契约错误升级为 INTERNAL_ERROR。

### 3.4 信任边界

- 用户输入决定检查对象；外部 policy 只能调整 schema 允许的策略值和增加受控 customChecks。
- 内嵌 Bundle 固定规则身份、Fact 形状、operator、依赖、SourceLock、发布事实和 Collector 能力。
- Collector 只陈述事实，不直接产生 PASS/FAIL/WARN。
- CEL 只计算 typed Fact，不执行 I/O。
- Renderer 只展示 Report，不猜测规则业务语义。

## 4. CLI、输入与运行控制

### 4.1 公共参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output` | `text` | `text` 或 `json` |
| `--output-file` | 空 | 保存 UTF-8 完整报告；默认文本运行可生成详细报告 |
| `--preflight-config` | 空 | 加载完整外部 preflight.yaml；空值使用内嵌策略 |
| `--timeout` | `10m` | 覆盖输入、management、Adapter、SDK 和渲染前阶段 |
| `--fact-timeout` | `30s` | 单个 `(subject,key)` 采集超时 |
| `--max-concurrency` | `0` | 0 为有界自适应；合法范围0..64 |

自动化直接使用稳定 CLI 参数，不额外提供 `PreflightRequest`。cluster/nodes 必须是普通文件并严格
解析；机器调用应读取 JSON 的 overall、issues 和 results，而不是解析彩色文本。

### 4.2 输入优先级

```text
显式 init CLI / cluster.yaml / nodes.yaml / management Provider 值
  > 当前生效 policy 的 fallback

当前生效 policy = 指定的完整外部文件；未指定 --preflight-config 时才是内嵌默认文件
```

policy 只改变本次检查计划，不修改业务命令输入。例如显式业务端口优先于 policy defaultPort；
资源 minimum 和平台矩阵等无业务参数的检查策略直接来自本次 policy。

### 4.3 RunIssue 与退出码

输入、计划和 setup 问题使用 RunIssue：

```text
severity: INFO | WARN | ERROR
stage: INPUT | PLAN | SETUP | EXECUTION
effect: BLOCK_START | CONTINUE | INTERRUPT
code / field / cause / action / sourceRefs
```

YAML 未知字段和类型错误应提供路径式 field；无法恢复路径的语法错误保留文件与行号。

| Overall | 退出码 |
|---|---:|
| PASS | 0 |
| WARN 或全部 SKIP | 2 |
| FAIL | 3 |
| INPUT_ERROR / BUNDLE_ERROR | 4 |
| INTERNAL_ERROR | 5 |
| TIMED_OUT | 124 |
| CANCELED | 130 |

## 5. 规则与配置设计

### 5.1 固定规则范围

| 阶段 | 固定模板 | 主要范围 |
|---|---:|---|
| init | 27 | 身份、平台、资源、工具、Workspace、端口、来源、离线制品、仓库和 CA |
| cluster-create | 46 | 输入、拓扑、管理集群、路径、CIDR、HA/addon、SSH、资源、平台、角色端口和来源 |
| customChecks | 每个 operation 0..64 | fileContent 或受控 exec 的 typed Fact |

完整 Rule ID、条件和 Source ID 以两个阶段提案为准。

规则进入固定目录必须同时满足：

1. 有当前安装文档、BKE、Provider、manifest 或 release profile 的固定来源。
2. 能在业务动作前通过只读方式取得足够事实。
3. 失败时可给出明确且不误导的修复动作。
4. expected、operator、ValueType、level、unknown 和适用条件可静态表达。
5. SourceLock 完整且使用不可变 revision/digest。
6. 有正向、失败、未知、依赖、不适用和脱敏测试。

“旧工具曾检查”“业界通常检查”或“可能有风险”不能单独成为规则来源。

### 5.2 默认 preflight.yaml

默认策略与匹配 Bundle 一起内嵌到 bke 单二进制。配置结构以 bkeadm
`configs/preflight/preflight.yaml` 为准。以下仅展示结构和代表性检查，发布文件中的固定项不能省略：

```yaml
apiVersion: preflight.openfuyao.io/v1
kind: PreflightConfig
metadata:
  release: v26.06
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
      - {key: cpu, description: Logical CPU count, minimum: 2}
      - {key: memory, description: Total memory capacity, minimum: 4G}
      - {key: systemDisk, description: System disk capacity, minimum: 100G}
    ports:
      - {key: kubernetes, description: Kubernetes API port, defaultPort: 36443, portFrom: port.kubernetes}
      - {key: imageRepo, description: Image repository port, defaultPort: 40443, portFrom: port.imageRepo}
      - {key: chart, description: Chart repository port, defaultPort: 38080, portFrom: port.chart}
      - {key: yum, description: Package repository port, defaultPort: 40080, portFrom: port.yum}
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
    checks:
      - key: workspace-access
        description: 工作目录可创建且所在文件系统可写
        pathAccess:
          pathFrom: workspace
          targetType: directory
          require: [creatable, notReadOnly]
      - key: tar
        description: tar 可执行文件
        when: feature.consoleHelmInstallRequired
        program: {name: tar, state: exists}
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
    resources:
      - {key: cpu, description: Node physical CPU core count, roles: [master], minimum: 2}
      - {key: memory, description: Node total memory capacity, roles: [master], minimum: 2G}
    ports:
      - {key: apiServer, description: Node Kubernetes API port, defaultPort: 6443, portFrom: provider.node.apiServer, roles: [master]}
      - {key: etcdClient, description: Etcd client port, defaultPort: 2379, portFrom: provider.node.etcdClient, roles: [master, etcd]}
      - {key: etcdPeer, description: Etcd peer port, defaultPort: 2380, portFrom: provider.node.etcdPeer, roles: [master, etcd]}
      - {key: etcdMetrics, description: Etcd metrics port, defaultPort: 2381, portFrom: provider.node.etcdMetrics, roles: [master, etcd]}
      - {key: kubelet, description: Kubelet secure port, defaultPort: 10250, portFrom: provider.node.kubelet, roles: [master, worker]}
      - {key: kubeletHealth, description: Kubelet health port, defaultPort: 10248, portFrom: provider.node.kubeletHealth, roles: [master, worker]}
      - {key: kubeletReadOnly, description: Kubelet read-only port when enabled, defaultPort: 10255, portFrom: provider.node.kubeletReadOnly, roles: [master, worker]}
      - {key: controllerManager, description: Controller manager port, defaultPort: 10257, portFrom: provider.node.controllerManager, roles: [master]}
      - {key: scheduler, description: Scheduler port, defaultPort: 10259, portFrom: provider.node.scheduler, roles: [master]}
      - key: localHA
        description: Provider-managed control-plane load-balancer port
        defaultPort: 36443
        portFrom: cluster.controlPlaneEndpointPort
        roles: [master]
        when: cluster.loadBalancer.required
      - {key: bkeAgent, description: BKE Agent port, defaultPort: 58080, portFrom: cluster.agentHealthPort, roles: [all]}
    checks:
      - key: system-root-free
        description: 节点根文件系统实际可用空间
        roles: [all]
        diskFree: {path: /, minimumBytes: 50G, metric: totalFree}
    customChecks: []
```

### 5.3 外部策略

`--preflight-config` 接受完整、严格解码的 preflight.yaml，不与内嵌 YAML 做字段级 merge。用户先
通过 `bke version` 获取 `gitCommitID`，再从 bkeadm 仓库同一修订的
`configs/preflight/preflight.yaml` 复制完整默认文件。外部策略：

- 平台矩阵必须与发布策略完全一致；资源阈值、内置检查允许的阈值或超时以及展示信息可在固定契约内调整。
- 端口的`defaultPort`、`protocol`、`portFrom`、`when`和`roles`固定；实际业务端口通过安装命令或
  cluster/nodes 配置传入。资源和内置检查的`when`、`subjectWhen`、`roles`、`nodes`、检查类型与
  采集目标固定；`topology.minimum`只能提高，`topology.unique`和`masters.odd`的发布要求必须保留。
- 必须保留默认 policy 已有的固定配置键，不能通过删除列表关闭发布检查。
- `failure`和`unknown`只能保持或加强，不能调弱发布检查。
- 必须与当前 release 匹配，未知字段或无法投影的值使检查不启动。
- 不能覆盖固定 Rule ID、fact、operator、dependsOn、SourceRef、发布事实和 Provider defaults。
- 合法 customChecks 按固定公式生成 Rule ID，并进入本次内存 Bundle。
- 只改变本次检查，不改变后续安装行为。

### 5.4 自定义检查

自定义项直接放入对应 operation：

```yaml
operations:
  init:
    customChecks:
      - key: hosts-loopback
        description: /etc/hosts 回环地址配置
        expected: 包含 127.0.0.1 或 ::1 回环地址映射
        collector:
          type: fileContent
          path: /etc/hosts
        expression: >-
          actual.matches('(?m)^\\s*(127\\.0\\.0\\.1|::1)(\\s|$)')
        failure: fail
        unknown: fail
        remediation: 补充回环地址映射后重试
```

`fileContent` 最多读取64 KiB，原文只进入 CEL；报告仅记录路径、Collector 和观测摘要。

`exec` 只执行节点上预装、安全权限的绝对路径脚本，使用结构化 args。resultType 支持 bool、
integer、string、stringList、integerList 和 object；object 必须通过 publicFields 声明报告可见字段。
脚本 stdout 必须是单一 typed Fact JSON，不能直接返回 PASS/FAIL。用户不填写 digest，由 Host 计算
观测 SHA-256。fileContent 原文和 object 非公开字段不进入报告；scalar/list value 会作为实际值展示，
必须只返回可公开内容。完整契约见 SDK 提案。

## 6. 报告与可修复性

### 6.1 Report

Report 包含 schemaVersion、operation、planHash、configDigest、bundleDigest、overall、exitCode、
summary、issues 和 results。Result 至少包含规则/实例 ID、名称、状态、Subject、typed
Expected/Actual、Evidence、SourceRefs、原因、依赖和修复建议。

公开检查状态只有 PASS、FAIL、WARN、SKIP。不适用项不产生 Result；Summary 满足
`total = pass + fail + warn + skip`。

### 6.2 文本和 JSON

宽终端使用六列表格：

| 状态 | 检查项 | 目标 | 期望 | 实际 | 原因及建议 |
|---|---|---|---|---|---|

- 目标同时说明节点/管理集群/本机和具体字段、端口、路径或来源。
- 期望与实际分列，原因与建议合并且保持简洁。
- 按终端显示宽度动态分配列宽，中文、英文和 Unicode 正确对齐。
- 窄于72列时降级为纵向布局，不超过真实窗口宽度。
- 控制台只展开 RunIssue 和 FAIL/WARN；完整 text/JSON 保留四态。
- JSON stdout 为单一文档，进度只写 stderr。

每个可见问题至少回答：目标是什么、检查什么、期望与实际是什么、根因属于哪一层、用户应修改
哪个 flag/YAML 字段/节点状态并如何重试。证据不足时不得编造更细的原因。

## 7. 只读、安全与质量属性

| 边界 | 允许 | 禁止 |
|---|---|---|
| 本机 | 默认允许stat、挂载、端口、文件存在/可读/格式和TCP；显式授权后允许临时磁盘写入与TCP监听 | 改服务、安装和清理 |
| Kubernetes | Discovery、GET、LIST | CREATE、UPDATE、PATCH、DELETE、server-side dry-run |
| SSH | transport/auth/privilege、固定探测命令；显式授权后允许暂存并清理官方探测程序 | 用户提供的raw shell/inline、隐式sudo、安装和清理 |
| 来源 | host:port TCP | HTTP HEAD、TLS、认证、对象下载 |
| 报告 | 脱敏 endpoint、主体、路径、摘要和公开字段 | 密码、Token、私钥、完整 kubeconfig/Secret |

质量属性：

- 可靠性：根因去噪、明确错误域、总体/单 Fact 超时和有界并发。
- 安全性：默认只读、最小权限、敏感信息结构性隔离、自定义脚本受控。
- 兼容性：schemaVersion 管理演进；未知字段严格拒绝；CLI 参数保持稳定。
- 可服务性：每条可见问题可定位和修复；详细报告可归档。
- 可测试性：Host、Bundle、Collector、CEL、DAG 和 Renderer 可独立注入和测试。
- 性能：Fact 按 `(subject,key)` 去重；并发有硬上限；输入和输出有大小限制。

## 8. 用户文档

本特性只提交中文用户文档：

| 文档 | 内容 |
|---|---|
| `docs/zh/cluster_installation_guide/environment_preparation/before_you_start.md` | 安装前推荐运行对应 preflight，说明只读和 envCheck 边界 |
| `docs/zh/cluster_installation_guide/appendix/command_reference.md` | 命令、参数、配置、报告和退出码 |
| `docs/zh/cluster_installation_guide/environment_preparation/environment_pre_check_tool_guide.md` | 区分统一 preflight 与旧 envCheck 的能力和风险 |

## 9. 测试与验收

### 9.1 测试范围

- 共享 Resolver/Validate、PlanHash、Binding 和适用条件。
- 73个固定模板与两个阶段清单、policy、Bundle 和 SourceLock 一致。
- Bundle generate/verify、摘要、SourceLock、DAG、类型和篡改检测。
- local、artifact、endpoint、Kubernetes、SSH、custom Collector 契约和只读性。
- CEL 内置 operator、custom expression、类型和成本边界。
- 输入/Bundle/内部错误、依赖 SKIP、超时、取消和退出码。
- 15/60/72/80/120/220列终端、中文显示、text/JSON/output-file 和单 JSON 文档。
- Linux AMD64/ARM64 的真实 CLI、SSH、Kubernetes 和只读审计；fake collector 不替代真实 E2E。

### 9.2 验收标准

1. preflight 与业务命令使用等价输入时，最终业务值和检查对象一致。
2. 固定规则数量、ID、条件、阈值、依赖和来源与 Bundle 一致。
3. 输入、Bundle、类型和框架错误不被渲染为普通环境 FAIL。
4. 根依赖失败后不继续无意义探测，不扩散多条可见 FAIL。
5. 每条 FAIL/WARN 可定位目标并给出证据支持的修复建议。
6. 文本在真实终端宽度下可读，JSON 可稳定解析和归档。
7. 默认执行不产生副作用；命令、临时磁盘写入、TCP监听和远端官方探测程序暂存必须分别显式授权，且不产生安装、修复、清理和 Kubernetes 写副作用。
8. Bundle 可确定性重建、verify 和篡改检测，发布入口执行对应门禁。

## 10. 需求分配与修改日志

| 需求 | 主要承载模块 | 验收入口 |
|---|---|---|
| R01、R02 | BKE CLI / Host Adapter | 两阶段 CLI 与业务 Resolver/Validate 一致性测试 |
| R03、R05、R06 | SDK、Collector、CEL、DAG | SDK 单测、类型/依赖/超时测试和真实环境验证 |
| R04 | Bundle compiler、SourceLock、generator | generate/verify、摘要和篡改测试 |
| R07 | Runtime policy、custom Collector | 完整配置、兼容性、安全和 customChecks 测试 |
| R08、R09 | Report、Renderer、中文用户文档 | text/JSON golden、终端宽度和可修复性验收 |
| R10 | Makefile 与发布入口 | 单元测试、静态检查和 Linux E2E |

| 版本 | 说明 |
|---|---|
| v1 | 统一 init 与 cluster-create 前置检查的首版提案 |

## 11. 参考资料

- [init 阶段检查项提案](./init阶段检查项提案.md)
- [cluster-create 阶段检查项提案](./cluster-create阶段检查项提案.md)
- [校验框架 SDK 提案](./校验框架SDK提案.md)
