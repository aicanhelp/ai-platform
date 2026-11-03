# 校验框架 SDK 提案

## 1. 特性描述

前置检查 SDK 提供与产品命令解耦的计划校验、事实采集编排、CEL 规则计算和报告契约。BKE 的
CLI、Provider、节点角色、默认值和业务输入由 Host Adapter 负责，SDK 只理解通用 OperationPlan、
RuleBinding、Bundle、Fact、依赖和 Report。

```text
BKE CLI / 配置
  → BKE Host Adapter
  → ResolveResult（Plan、Bindings、Facts、Issues）
  → 通用 SDK + PreflightBundle + Collector
  → Report
```

SDK 是 bkeadm 中的独立 Go module `gopkg.openfuyao.cn/preflight`。它不依赖 Cobra，不读取产品
配置或环境变量，不调用 `os.Exit`，不直接写 stdout/stderr，也不硬编码 init/cluster-create 的解析。

### 1.1 目标

- 统一表达业务输入、事实、规则、依赖、问题和结果。
- 同一 Plan、Bundle 和 Facts 得到确定性相同的 Report。
- 将输入错误、事实不可得、规则不满足和框架违约分为不同错误域。
- 所有环境访问通过显式 Collector 注入，SDK core 无隐式 I/O。
- 通过 DAG 抑制单根因多 FAIL，并保留可审计的 SKIP 链路。
- Expected、Actual、Evidence 和 Remediation 足以直接生成可修复报告。
- 固定 operator 和 custom rule 使用同一 CEL 计算内核。

### 1.2 非目标

- SDK 不承诺后续安装成功，也不替代 Provider 最终校验。
- 不在 SDK 中解析 BKE CLI、cluster/nodes 或 Kubernetes CRD。
- 不提供通用表达式 DSL、CEL I/O 函数或任意 shell 执行能力。
- 不允许 renderer、Collector 或 Host 各自维护第二套状态计算。

### 1.3 依赖与 License

SDK 核心直接依赖 Go context、CEL-Go、go-pretty、`x/term` 和 YAML 库。Kubernetes client-go 与
`x/crypto/ssh` 属于 bkeadm Host 层，通过 Collector/executor 接口注入。SDK 与 bkeadm 使用木兰
宽松许可证第2版；新增第三方依赖必须经过仓库 License、漏洞和供应链审查。

## 2. 架构与包职责

| 包 | 职责 |
|---|---|
| `api` | Plan、Binding、Fact、RunIssue、Result、Report、Bundle 等 DTO |
| `normalize` | Plan 规范化与 PlanHash |
| `bundle` | Bundle 解析、摘要和静态契约校验 |
| `celcheck` | CEL 环境、表达式长度、成本、编译和求值 |
| `engine` | DAG、Fact 去重采集、类型校验、求值和 Overall |
| `collectors` | local、artifact、endpoint、Kubernetes、SSH、custom 等事实能力 |
| `render` | 动态终端文本、详细文本和 JSON 渲染 |
| Host Adapter | 位于 bkeadm 根模块，处理产品输入、Provider、凭据和 Collector 组装 |

依赖方向必须保持：CLI/Host 可以依赖 SDK；SDK 不得反向依赖 Cobra、bkeadm 产品包、Provider
runtime 或具体 Kubernetes client 实现。

## 3. Host API

### 3.1 主入口

```go
type CheckInput struct {
    Resolved     api.ResolveResult
    Release      api.ReleaseRef
    Bundle       []byte
    ConfigDigest api.Digest
    Runtime      api.RuntimeOptions
    Progress     api.ProgressFunc
}

type Dependencies struct {
    Collector            api.Collector
    ExpectedRelease      string
    ExpectedBundleDigest api.Digest
    BundleValidator      func(*api.PreflightBundle) error
}

Check(ctx context.Context, in CheckInput) (api.Report, error)
```

`CheckInput` 不携带 Credentials、Internal、AppliesWhen 等死字段。凭据只存在于 Host 注入的短生命
周期 executor/client 中。

预期的输入和 Bundle 问题写入 Report，分别形成 `INPUT_ERROR` 和 `BUNDLE_ERROR`；非 nil Go error
仅用于 SDK 或 Collector 契约故障，并由最外层转为结构化 `INTERNAL_ERROR`。

### 3.2 Host 调用顺序

```text
总体 context
  → 严格解析和业务 Validate
  → Adapter ResolveResult
  → SDK 校验 Plan/Bindings
  → Bundle 解析、摘要和静态校验
  → 过滤不适用项并构建实例 DAG
  → Collector 采集和 CEL 计算
  → Report
  → Host renderer / exit code
```

Host 不得在 Check 后用第二套业务条件重算结果，也不得在 management/SSH setup 失败时切换到不
支持该事实的本地 Collector。

### 3.3 ResolveResult

```go
type ResolveResult struct {
    Plan     OperationPlan
    Issues   []RunIssue
    Bindings []RuleBinding
    Facts    []Fact
}
```

Adapter 已能只读取得的计划级事实可通过 Facts 传入；动态本机、节点和管理集群事实仍由 Collector
按请求采集。

## 4. OperationPlan 与 RuleBinding

### 4.1 OperationPlan

```go
type OperationPlan struct {
    SchemaVersion string
    Operation     Operation
    Subjects      []Subject
    Values        map[string]Value
    Sources       []Source
    Origin        PlanOrigin
    PlanHash      Digest
}
```

- Subject 是本机、cluster 范围或节点等稳定目标，包含 ID、role、address 和非敏感 metadata；
  管理集群事实可绑定到 cluster 范围并用 Evidence locator 定位对象。
- Value 是 Adapter 解析后的 typed 最终业务值，不包含 comparator、level 或表达式。
- Source 描述本次操作真实使用的 endpoint、kind、objectPath 和非敏感 AuthRef。
- PlanHash 覆盖 operation、Subject、role、address、metadata、Values 和 Sources 的规范化内容。
- RuleBinding 必须完全复用 Plan 中的 canonical Subject，不能用相同 ID 替换地址或 SSH 端口。
- 密码、Token、私钥、完整 kubeconfig 和带凭据 URL 禁止进入 Plan 和摘要。

Subjects、roles、metadata、Values 和 Sources 均需稳定排序。只改变输入列表或 map 遍历顺序不应
改变 PlanHash；节点、角色、端口、来源或功能条件变化应改变 PlanHash。

### 4.2 RuleBinding

```go
type RuleBinding struct {
    TemplateID string
    InstanceID string
    Name       string
    Fix        string
    Subject    Subject
    Fact       string
    Applicable bool
}
```

Binding 只负责将可信模板绑定到具体 Subject 和 fact key，不能覆盖 expected、level、unknown、
operator、dependsOn 或 SourceRef。`Applicable=false` 在建图前过滤，不产生 Result 或 SKIP，也不能
因为无效模板触发 Bundle 错误。

## 5. Bundle 与规则契约

```go
type RuleTemplate struct {
    ID         string
    Name       string
    Operation  Operation
    Fact       string
    Op         Op
    Expected   any
    Level      FailureSeverity
    Unknown    UnknownPolicy
    Source     string
    Fix        string
    DependsOn  []string
    Expression string
}
```

### 5.1 状态策略

| 字段 | 值 | 结果语义 |
|---|---|---|
| level | `ERROR` | 不满足期望时 FAIL |
| level | `WARNING` | 不满足期望时 WARN |
| level | `INFO` | 满足时 PASS，不满足时观察型 SKIP |
| unknown | `BLOCK` | 事实不可取得时 FAIL |
| unknown | `WARN` | 事实不可取得时 WARN |
| unknown | `IGNORE` | 事实不可取得时 SKIP |

只有 `Fact.Available=false` 才进入 unknown 策略。已取得但不满足的事实、规则类型错误和 Collector
契约错误不能伪装成 UNKNOWN。

### 5.2 CEL 计算

公开 operator 为 `eq`、`in`、`gte`、`lte`、`range`、`exists`、`free`、`reachable`、
`relation` 和 `cel`。前九种由 SDK 映射为固定 CEL 表达式；只有 custom rule 携带受限 expression。

CEL 仅暴露 `actual: dyn`，不注册文件、网络、时间、命令或产品函数。表达式最长2048字节，执行
成本上限10000，结果必须为 bool。配置编译错误属于 INPUT_ERROR，Bundle 编译错误属于
BUNDLE_ERROR，运行期类型或求值错误属于 INTERNAL_ERROR。

### 5.3 Bundle 静态校验

加载时必须依次校验：

1. 严格单文档 YAML、大小、schemaVersion、release 和 operation。
2. PolicyDigest、BundleDigest 和预期 release 配对。
3. Rule 必填字段、唯一 ID、operator/expected、level、unknown 和 ValueType。
4. DependsOn 悬空、重复、跨 operation、自依赖和环。
5. 每个 Source ID 可解析到完整 SourceLock。
6. artifacts、verifiedPlatforms、签名和审批约束。
7. Host 注入的 policy/release facts 与产品目录一致性。

失败形成单一可定位 setup issue，不启动 Collector。

## 6. Fact 与 Collector 契约

```go
type Fact struct {
    Key       string
    Subject   string
    Value     any
    Type      ValueType
    Available bool
    Evidence  []Evidence
    Error     *FactError
    Display   *DisplayValue
}
```

### 6.1 请求与返回

- 一个请求恰好包含一个 `(subject,key)`。
- Collector 成功时恰好返回一个相同 identity 的 Fact。
- 零个、多个、错误 key、错误 subject、非法 availability/error 组合均为 INTERNAL_ERROR。
- `Available=false` 必须携带受支持的 NOT_FOUND、PERMISSION_DENIED、UNREACHABLE、TIMEOUT、
  UNSUPPORTED 或 PARSE_ERROR。
- Collector 只返回事实和安全 Evidence，不决定 PASS、FAIL 或 WARN。

### 6.2 类型与展示

SDK ValueType 包括 bool、integer、bytes、duration、string、enum、address、set、range、relation 和
object。Fact.Type 为空时才允许推断；非空时必须合法且 Value 与声明类型一致。规则 expected 类型
错误属于 BUNDLE_ERROR，Collector 类型错误属于 INTERNAL_ERROR。

Result.Expected 和 Result.Actual 必须传播 DisplayValue.Type。Fact.Display 是安全投影；evaluator
读取完整 Value，Result 优先展示 Display，防止文件原文和 object 私有字段泄露。

### 6.3 Collector 能力

| Collector | 能力 | 边界 |
|---|---|---|
| Local | identity、平台、资源、工具、路径、挂载、端口、文件、TCP | 不创建路径或测试文件，不改系统 |
| Artifact | 离线制品存在、可读、格式和受信属性 | 不解压、安装或执行制品 |
| Kubernetes | API、Discovery、CRD、Secret、Provider defaults | 只允许 GET/LIST/Discovery |
| SSH | 独立transport/auth/privilege、固定探测命令、端口、节点侧TCP连接；显式授权后暂存官方探测程序 | 不安装；当前不验证host key |
| Endpoint | host:port TCP dial | 不做 HTTP、TLS、认证或对象查询 |
| Custom | fileContent、预装绝对路径 exec | 不接受 raw shell/inline/`bash -c`/upload/sudo/env；64 KiB 和安全投影 |

Router 的匹配顺序是安全契约：远端节点必须先匹配 SSH route，setup 失败不得落入本地 route。

## 7. DAG、并发与 Context

- 引擎先构建模板 DAG，再按 Subject 绑定为实例 DAG。
- 默认优先查找同 Subject 依赖，必要时回退到计划级依赖。
- 根依赖 FAIL/WARN 后，下游适用项为 SKIP 并引用最早根因。
- 不同独立分支可继续执行；同一 `(subject,key)` 只采集一次。
- `MaxConcurrency=0` 为有界自适应，合法范围0..64；并发不得改变结果排序。
- 总体 context 由 Host 创建，覆盖 management defaults、Adapter、SDK 和客户端。
- 每个 probe 从总体 context 派生 FactTimeout；不得使用 `context.Background()` 丢失取消链。
- timeout 和 cancel 分别产生 TIMED_OUT、CANCELED，进度不得显示100%完成。

## 8. RunIssue、Report 与 Renderer

```go
type RunIssue struct {
    Code       string
    Severity   INFO | WARN | ERROR
    Stage      INPUT | PLAN | SETUP | EXECUTION
    Effect     BLOCK_START | CONTINUE | INTERRUPT
    Field      string
    Cause      string
    Action     string
    SourceRefs []string
}
```

缺参数、严格 YAML、Provider defaults、Bundle 配对和框架错误使用 RunIssue；环境事实不满足使用
Result。`BLOCK_START` 统一展示“前置检查未启动”，用户可修复问题不能只返回裸 Go error。

```go
type Report struct {
    SchemaVersion string
    Operation     Operation
    PlanHash      Digest
    ConfigDigest  Digest
    BundleDigest  Digest
    Overall       Overall
    ExitCode      int
    Summary       Summary
    Issues        []RunIssue
    Results       []Result
}
```

公开状态只有 PASS、FAIL、WARN、SKIP。Summary 必须满足
`total = pass + fail + warn + skip = len(results)`；不适用项不进入 total。FAIL/WARN 必须有原因或
修复建议，依赖 SKIP 必须引用根因。

Renderer 只消费 Report，不访问 Collector、Bundle 或业务配置，也不重算阈值、状态、Overall 和
退出码。宽终端使用“状态、检查项、目标、期望、实际、原因及建议”六列；窄终端降级为纵向布局。
JSON 必须是单一文档。

## 9. 自定义扩展

customChecks 只覆盖常见现场差异，不追求任意能力：

| 字段 | 约束 |
|---|---|
| key | operation 内唯一，生成固定 Rule ID 和 fact key |
| description/expected/remediation | 必填，分别用于名称、期望和修复建议 |
| failure/unknown | fail、warn、skip，映射到 SDK 级别和 unknown 策略 |
| roles/when | init 禁止 roles；cluster 仅允许 master/worker/etcd/all，all 不混用；when 是 Plan bool key，false 不实例化 |
| fileContent | clean absolute path，最多64 KiB，原文不进入报告 |
| exec | 预装绝对路径、结构化 args、普通可执行且非 group/world writable |
| resultType | bool、integer、string、stringList、integerList、object |
| publicFields | object 必填，只有声明字段进入报告 |
| expression | 只读取 actual，长度和成本受限，结果必须 bool |

最小配置示例：

```yaml
customChecks:
  - key: runtime-health
    description: 容器运行时状态
    expected: 运行时健康
    collector:
      type: exec
      path: /usr/local/libexec/check-runtime
      args: [--format, fact]
      resultType: object
      publicFields: [runtime, healthy]
    expression: actual.healthy == true
    failure: fail
    unknown: fail
    remediation: 检查容器运行时服务后重试
```

脚本 stdout 必须只包含一个 `preflight.openfuyao.io/fact/v1` JSON。可用事实示例：

```json
{
  "apiVersion": "preflight.openfuyao.io/fact/v1",
  "available": true,
  "type": "object",
  "value": {"runtime": "containerd", "healthy": true, "privateDetail": "不会进入报告"}
}
```

不可用事实示例：

```json
{
  "apiVersion": "preflight.openfuyao.io/fact/v1",
  "available": false,
  "error": {"code": "PERMISSION_DENIED", "message": "cannot inspect runtime"}
}
```

`type` 与 resultType 对应；value 分别使用 JSON bool、整数、字符串、同类字符串/整数数组或最多32个
字段的 object。错误码限 NOT_FOUND、PERMISSION_DENIED、UNREACHABLE、TIMEOUT、UNSUPPORTED 和
PARSE_ERROR。未知字段、多份 JSON、类型不匹配或超过64 KiB 均拒绝。脚本返回 Fact 而不是
PASS/FAIL；用户不填写 digest，由 Host 计算观测 SHA-256。框架不接受用户提供的 raw shell、inline、
`bash -c`、上传、下载、隐式 sudo 或自定义环境。fileContent 原文和 object 非公开字段只存在于本次
内存 Fact/CEL，报告使用安全 Display/publicFields；bool、integer、string 和 list 的 value 会作为
实际值进入报告。脚本提供者必须审核业务逻辑与输出内容，不能输出凭据或其他敏感值。

增加发布固定规则时优先复用已有 Fact；只有现有能力无法表达且有明确来源时才增加新的固定 Fact
或 Collector。新增 operation 必须建立独立 Host Adapter、输入规范、规则提案、SourceLock 和真实 E2E，
不能自动继承 init/cluster-create 规则。

## 10. 威胁模型与质量属性

| 风险 | 控制 |
|---|---|
| 凭据泄露 | Host 安装凭据只在 executor/client，不进入 Plan/Fact/Report；custom 内容按 Display/publicFields 投影，scalar/list 视为公开输出 |
| 任意代码执行 | 固定规则使用 allowlist；custom exec 禁止 raw shell/inline/上传/sudo/env，并校验权限 |
| 恶意或超大输入 | 严格 schema、文件大小、输出大小、CEL 长度和成本限制 |
| 网络误诊 | transport、auth、privilege 分层；TCP 不解释为 HTTP/TLS/auth |
| 单根因告警风暴 | 显式 DAG 和依赖 SKIP |
| 非确定性结果 | 规范化、Fact 去重、稳定排序、摘要和 renderer 单一事实源 |
| 超时失控 | Host 总体 context、单 Fact timeout 和有界并发 |

质量属性要求：核心无全局可变状态；同输入结果确定；错误可定位；新增 Fact/规则可独立测试；schema
和 Report 通过版本字段演进；敏感字段默认不进入公开 DTO。

## 11. 测试与验收

必须覆盖：

- PlanHash 规范化、Subject/Value/Source/Binding identity 和敏感信息门禁。
- Bundle schema、digest、SourceLock、operator/expected、DAG 和 CEL 静态校验。
- Collector 的零/多 Fact、identity、available/error、type/value 和 Evidence 契约。
- DAG 根因传播、去重、并发、稳定排序、超时和取消。
- 内置 operator 与 custom CEL 的正向、失败、未知、类型和成本边界。
- local、artifact、endpoint、Kubernetes、SSH、custom 的只读和脱敏测试。
- Report 完整性、Overall/退出码、动态终端宽度、中文显示和单 JSON 文档。
- Host 的真实 CLI、SSH、Kubernetes E2E；fake collector 只作为进程内集成测试。

## 12. 参考资料

- [openFuyao 统一前置检查提案](./openFuyao统一前置检查提案.md)
- [init 阶段检查项提案](./init阶段检查项提案.md)
- [cluster-create 阶段检查项提案](./cluster-create阶段检查项提案.md)
