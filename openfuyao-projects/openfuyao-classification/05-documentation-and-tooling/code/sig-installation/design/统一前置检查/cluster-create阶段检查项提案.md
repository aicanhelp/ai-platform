# cluster-create 阶段检查项提案

## 1. 阶段描述

`bke preflight cluster-create` 在执行 `bke cluster create` 前，使用相同的 cluster/nodes 文件和
管理集群 kubeconfig，检查输入、管理集群只读依赖、目标节点、角色端口和安装来源。

命令只读解析文件、读取管理集群并通过 SSH 采集目标节点事实，不创建或修改 Kubernetes 资源，
不上传脚本，不执行安装和清理。

### 1.1 目标

- 单次读取并严格解析 cluster/nodes，复用 Provider 的纯默认化和本地校验语义。
- 根据最终节点、角色、端口和来源动态生成 OperationPlan 和 RuleBinding。
- 准确区分输入、管理集群、SSH 网络、认证、权限和环境不满足。
- 根依赖失败后停止下游探测，避免一个问题扩散成多条 FAIL。
- YAML 错误和规则结果均尽可能定位到具体字段、节点、端口、路径或来源。

### 1.2 输入与优先级

```bash
bke --kubeconfig /path/to/management.kubeconfig \
  preflight cluster-create \
  --file ./cluster.yaml \
  --nodes ./nodes.yaml

bke --kubeconfig /path/to/management.kubeconfig \
  preflight cluster-create \
  --preflight-config ./preflight.yaml \
  --file ./cluster.yaml --nodes ./nodes.yaml \
  --output json --output-file report.json
```

v1 不提供 `--request`。运行值优先级为：cluster/nodes 显式字段或管理集群 Provider 值 > 当前生效
policy 的 fallback。指定`--preflight-config`时当前policy就是该外部完整文件；未指定时才是内嵌默认
文件。policy 不修改输入对象或后续安装参数。

## 2. 计划与依赖设计

```text
cluster.yaml + nodes.yaml + management kubeconfig
  → 严格解析和 Provider 纯默认化/校验
  → 管理集群只读 defaults/facts
  → OperationPlan（cluster、nodes、values、sources）
  → RuleBinding
  → SDK Collector / CEL / Report
```

Plan Subject 必须稳定标识 cluster 和每个目标节点。管理集群事实绑定到 `cluster` 范围，并通过
Evidence locator 标识 API、CRD、Secret 或字段；不建立独立的 management Subject。节点地址、SSH
端口、角色、业务端口、来源和非敏感 metadata 进入 PlanHash；密码、Token、私钥和完整 kubeconfig
不得进入。

主要根依赖：

```text
Management API → Discovery → BKECluster/BKENode CRD
Management API → Secret GET → Secret data → kubeconfig parse → source endpoint
SSH TCP transport → SSH auth → root/sudo privilege → 节点事实、端口、来源和 customChecks
```

管理默认值或客户端初始化失败不得静默回退到可能错误的静态值，也不得将远端事实路由到本地
Collector。`Applicable=false` 不实例化；只有适用项被依赖、超时或取消阻塞时才产生 SKIP。

## 3. 固定检查项

固定目录包含46个模板。`来源`为 Bundle Source ID，并由 SourceLock 映射到固定 repo、revision、
path、locator 和 sourceURL。
除 `CLUSTER-HA-COLOCATE` 使用 `WARNING/WARN` 外，其余固定模板使用 `ERROR/BLOCK`。

### 3.1 输入唯一性与拓扑（6项）

| # | Rule ID | 检查对象 | 期望值/判断 | 来源 |
|---:|---|---|---|---|
| 1 | `CLUSTER-UNIQUE-IP` | 节点 IP | 非空 IP 不重复 | `CAPBKE-VALIDATION` |
| 2 | `CLUSTER-UNIQUE-HOSTNAME` | 节点 hostname | 仅非空 hostname 不重复；空值不失败 | `CAPBKE-VALIDATION` |
| 3 | `CLUSTER-TOPO-MASTER` | master 数量 | 不少于1 | `CAPBKE-VALIDATION` |
| 4 | `CLUSTER-TOPO-MASTER-ODD` | master 数量 | 为奇数 | `CAPBKE-VALIDATION` |
| 5 | `CLUSTER-TOPO-ETCD` | etcd 数量 | 不少于1 | `CAPBKE-VALIDATION` |
| 6 | `CLUSTER-TOPO-WORKER` | worker 能力数量 | 不少于1；master+worker 计入 worker | `CAPBKE-VALIDATION` |

不存在 `CLUSTER-UNIQUE-NAME`。Provider 没有节点对象名称唯一性约束时，不增加该门禁。

### 3.2 管理集群与本地输入（10项）

| # | Rule ID | 检查对象 | 期望值/判断 | 依赖 | 来源 |
|---:|---|---|---|---|---|
| 7 | `CLUSTER-MGMT-API` | 管理集群 API | 可连接 | 根依赖 | `CAPBKE-VALIDATION` |
| 8 | `CLUSTER-MGMT-DISCOVERY` | API Discovery | 可读取 | API | `CAPBKE-VALIDATION` |
| 9 | `CLUSTER-MGMT-CRD-CLUSTER` | BKECluster CRD | 已注册 | Discovery | `CAPBKE-VALIDATION` |
| 10 | `CLUSTER-MGMT-CRD-NODE` | BKENode CRD | 已注册 | Discovery | `CAPBKE-VALIDATION` |
| 11 | `CLUSTER-MGMT-AGENT-KUBECONFIG-GET` | `kube-system/localkubeconfig` | Secret 可 GET | API | `CAPBKE-VALIDATION` |
| 12 | `CLUSTER-MGMT-AGENT-KUBECONFIG-DATA` | Secret 数据 | 目标键非空 | Secret GET | `CAPBKE-VALIDATION` |
| 13 | `CLUSTER-MGMT-AGENT-KUBECONFIG-PARSE` | Agent kubeconfig | 可解析 | Secret data | `CAPBKE-VALIDATION` |
| 14 | `CLUSTER-MGMT-AGENT-KUBECONFIG-SOURCE` | kubeconfig server | endpoint 语法有效 | kubeconfig parse | `CAPBKE-VALIDATION` |
| 15 | `CLUSTER-MGMT-LOCAL-CLUSTER` | 本地 BKECluster 输入 | schema/业务校验通过 | 独立 | `CAPBKE-VALIDATION` |
| 16 | `CLUSTER-MGMT-LOCAL-NODE` | 本地 BKENode 输入 | schema/业务校验通过 | 独立 | `CAPBKE-VALIDATION` |

### 3.3 本地路径、网络、HA 和 addon（9项）

| # | Rule ID | 检查对象 | 期望值/判断 | 条件 | 来源 |
|---:|---|---|---|---|---|
| 17 | `CLUSTER-LOCAL-PATH-CREATE` | 集群输出路径 | 已存在或父路径允许创建 | 始终 | `CAPBKE-VALIDATION` |
| 18 | `CLUSTER-LOCAL-PATH-ACCESS` | 集群输出路径 | 可访问 | 始终 | `CAPBKE-VALIDATION` |
| 19 | `CLUSTER-LOCAL-PATH-MOUNT` | 输出路径挂载点 | 非只读挂载 | 始终 | `CAPBKE-VALIDATION` |
| 20 | `CLUSTER-NET-POD-CIDR` | Pod subnet | CIDR 语法有效 | 已配置 | `CAPBKE-VALIDATION` |
| 21 | `CLUSTER-NET-SERVICE-CIDR` | Service subnet | CIDR 语法有效 | 已配置 | `CAPBKE-VALIDATION` |
| 22 | `CLUSTER-HA-COLOCATE` | bootstrap 与 HA master | 可证明同一稳定节点时提示风险 | WARNING；master>1 且可识别 bootstrap | `CAPBKE-VALIDATION` |
| 23 | `CLUSTER-REF-VERSION` | Provider/版本引用 | 属于当前发布支持范围 | 始终 | `CAPBKE-VALIDATION` |
| 24 | `CLUSTER-ADDON-CLOSURE` | addon 依赖 | 依赖闭包完整 | 按 addon | `CAPBKE-VALIDATION` |
| 25 | `CLUSTER-ADDON-SOURCE` | addon 来源配置 | 必需来源信息配置完整 | 按 addon | `CAPBKE-VALIDATION` |

CIDR 本期只校验语法，不推测网段重叠。`CLUSTER-ADDON-SOURCE` 只检查配置完整，不表示网络可达；
来源连通性由 `CLUSTER-SRC-REACHABLE` 负责。普通 master+worker AIO 不等于 bootstrap 共部署。v1
生产 Host 当前不提供可独立证明的 bootstrap 节点映射，因此不实例化 HA 告警；只有后续 Host 能
可信设置 `BootstrapNodeID` 时，才按本表条件实例化。

`CLUSTER-HA-COLOCATE` 的 `CAPBKE-VALIDATION` 只追溯 Provider 的角色和 HA 拓扑语义；该规则不表示
Provider 禁止共部署，其 WARNING 策略及“必须先证明 bootstrap 身份”的限制由本提案定义。

### 3.4 SSH、权限、资源与平台（10项）

| # | Rule ID | 检查对象 | 期望值/判断 | 依赖 | 来源 |
|---:|---|---|---|---|---|
| 26 | `CLUSTER-SSH-TRANSPORT` | SSH host:port | 纯 TCP 可连接 | 根依赖 | `DOC-QUICKSTART` |
| 27 | `CLUSTER-SSH-AUTH` | SSH 认证 | 最终凭据可建立会话 | transport | `DOC-QUICKSTART` |
| 28 | `CLUSTER-SSH-PRIVILEGE` | 执行权限 | root 或无交互 sudo 可用 | auth | `DOC-QUICKSTART` |
| 29 | `CLUSTER-AGENT-SYSTEMD` | systemd | 可用 | privilege | `DOC-QUICKSTART` |
| 30 | `CLUSTER-RES-CPU` | 逻辑 CPU | 默认不少于8核 | privilege | `DOC-QUICKSTART` |
| 31 | `CLUSTER-RES-MEMORY` | 可用内存 | 默认不少于16 GiB | privilege | `DOC-QUICKSTART` |
| 32 | `CLUSTER-RES-SYSTEM-DISK` | 系统盘容量 | 默认不少于100 GiB | privilege | `DOC-QUICKSTART` |
| 33 | `CLUSTER-PLAT-OS` | OS | Linux | privilege | `DOC-QUICKSTART` |
| 34 | `CLUSTER-PLAT-DISTRO` | 发行版 | 命中当前策略平台矩阵 | privilege | `DOC-QUICKSTART` |
| 35 | `CLUSTER-PLAT-KERNEL` | 内核 | 命中发行版对应内核前缀 | privilege | `DOC-QUICKSTART` |

transport 只做 TCP；认证必须单独建立 SSH 会话。凭据错误不能误报为网络失败。v1 Host 不校验
SSH host key，只能证明会话建立，不能描述为已完成目标节点身份或 host-key 信任校验。

### 3.5 角色端口和来源（11项）

| # | Rule ID | 检查对象 | 期望值/判断 | 角色/条件 | 来源 |
|---:|---|---|---|---|---|
| 36 | `CLUSTER-PORT-APISERVER` | kube-apiserver | 实际端口空闲，默认6443/TCP | master | `MANIFEST` |
| 37 | `CLUSTER-PORT-CONTROLLER` | controller-manager | 10257/TCP 空闲 | master | `MANIFEST` |
| 38 | `CLUSTER-PORT-SCHEDULER` | scheduler | 10259/TCP 空闲 | master | `MANIFEST` |
| 39 | `CLUSTER-PORT-LOCAL-HA` | 本地 HA | 实际控制面入口端口空闲，默认36443/TCP | Provider计划需要负载均衡的 master | `MANIFEST` |
| 40 | `CLUSTER-PORT-ETCD-CLIENT` | etcd client | 实际端口空闲，默认2379/TCP | master 或 etcd | `MANIFEST` |
| 41 | `CLUSTER-PORT-ETCD-PEER` | etcd peer | 实际端口空闲，默认2380/TCP | master 或 etcd | `MANIFEST` |
| 42 | `CLUSTER-PORT-ETCD-METRICS` | etcd metrics | 实际端口空闲，默认2381/TCP | master 或 etcd | `MANIFEST` |
| 43 | `CLUSTER-PORT-KUBELET-SECURE` | kubelet secure | 10250/TCP 空闲 | master 或 worker | `MANIFEST` |
| 44 | `CLUSTER-PORT-KUBELET-HEALTH` | kubelet health | 10248/TCP 空闲 | master 或 worker | `MANIFEST` |
| 45 | `CLUSTER-PORT-BKEAGENT` | BKEAgent | 实际健康检查端口空闲，默认58080/TCP | 所有目标节点 | `MANIFEST` |
| 46 | `CLUSTER-SRC-REACHABLE` | 节点到实际安装来源 | 可建立脱敏 host:port TCP 连接 | 每节点、每来源 | `DOC-QUICKSTART` |

远端来源通过已有SSH会话在目标节点执行受限的TCP连接探测，使用目标节点自己的DNS和`/etc/hosts`
解析结果；不依赖节点安装curl或nc，也不执行HTTP、TLS、认证和对象下载。

### 3.6 自定义检查（0..64项）

cluster-create 自定义检查位于 `operations.clusterCreate.customChecks`，按 roles 展开到匹配节点；
roles 为空或 `[all]` 时覆盖所有目标节点：

| Rule ID | Collector | 判断 | 固定依赖 | 来源 |
|---|---|---|---|---|
| `CONFIG-CLUSTER-CREATE-CUSTOM-<KEY>` | `fileContent` 或受控 `exec` | `expression(actual)` 必须为 true | `CLUSTER-SSH-PRIVILEGE` | `RUNTIME-PREFLIGHT-CONFIG` |

脚本必须预装在节点，不上传、不下载、不隐式 sudo。用户不配置 digest，由框架按节点记录观测摘要。

## 4. 输入错误与报告

- cluster/nodes 必须是普通文件，分别受大小限制并严格解析。
- 未知字段和类型错误使用路径式字段，如
  `cluster.spec.clusterConfig.cluster.networking.podSubnet`、`nodes[0].spec.ip`。
- YAML 语法错误无法恢复字段路径时保留文件名、行号和解析详情。
- Provider 本地业务校验结果应携带可确定的同类字段 locator。
- 缺少文件、解析失败或无法形成计划时显示“前置检查未启动”，不产生伪环境 FAIL。

目标列必须说明“哪个节点的什么”，例如：

- `cluster / management Secret kube-system/localkubeconfig`
- `cluster.yaml / spec.clusterConfig.cluster.networking.podSubnet`
- `master-01 (192.0.2.10:22) / SSH authentication`
- `master-01 (192.0.2.10) / TCP 6443`
- `master-01 (192.0.2.10) / Image repository example.com:443`

## 5. 只读、安全与质量属性

- Kubernetes 只允许 Discovery、GET 和 LIST，不得 CREATE、UPDATE、PATCH、DELETE 或 server-side dry-run。
- SSH 固定规则只使用 allowlist 只读命令；凭据不进入 Plan、Fact、Evidence、Report 或日志。
- 输出路径检查不创建目录或测试文件。
- 总体 context 覆盖 management defaults、客户端、Adapter、SSH、TCP、SDK 和渲染前阶段。
- 同一 `(subject,factKey)` 只采集一次；并发不影响结果顺序和计数。
- 连接、认证、权限、超时和业务不满足必须保持不同错误归属。

## 6. 测试与验收

必须覆盖：

- cluster/nodes 严格解析、字段路径、文件行号、Provider 默认化和优先级。
- 空 hostname、重复 hostname/IP、普通 AIO、Host 未提供 bootstrap 映射时不实例化，以及 Adapter
  契约中可证明 bootstrap+HA 的正反场景。
- management API/Discovery/CRD 与 Secret/kubeconfig 两条依赖链。
- SSH transport PASS/auth FAIL、privilege FAIL 和下游 SKIP。
- 每个角色端口、实际端口覆盖、同节点多角色去重和多来源展开。
- 远端来源在目标节点发起TCP连接并使用节点侧名称解析，不执行HTTP或依赖curl/nc。
- custom fileContent/exec 的角色、依赖、类型、权限、超限、摘要和脱敏。
- timeout/cancel、text/JSON、窄终端、中英文宽度和 Summary 算术一致。
- Linux 真实 CLI、SSH 和管理集群 Kubernetes 只读链路；fake collector 测试不能替代真实 E2E。

## 7. 参考资料

- [openFuyao 统一前置检查提案](./openFuyao统一前置检查提案.md)
- [校验框架 SDK 提案](./校验框架SDK提案.md)
