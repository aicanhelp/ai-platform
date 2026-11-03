# 安装部署并发场景 E2E 用例设计（后端）

## 1. 文档目的

本文档为 openFuyao **安装部署（installation）** 模块补充**后端 Ginkgo** 并发端到端用例的设计基线。

**范围限定**：

- 仅后端（`e2e/installation/`），不含前端 Playwright；
- 仅覆盖四类生命周期操作：**集群创建**、**集群删除**、**扩容**、**缩容**；
- 不涉及升级、Addon 变更、离线专项、诊断任务等场景；
- **全部用例均在管理集群场景下执行**：并发对象为管理集群上的**业务集群**；管理集群为前置环境中已就绪的 **1Master**，与**引导节点同机共部署**（见 §4.1）。

目标：

- 验证管理集群控制器在**多业务集群操作同时发生**时的正确性、互斥性与可恢复性；
- 与现有**串行**功能用例形成互补，而非重复 `performance-test/` 中的性能压测；
- 为后续实现提供统一的用例编号、标签、环境要求与断言标准。

---

## 2. 现状与缺口

### 2.1 已有能力

| 区域 | 现状 | 与并发的关系 |
|------|------|--------------|
| `management_cluster_test.go` | 引导节点同机 **1Master** 管理集群安装（`mgmt-1master`） | 前置环境拓扑参考 |
| `cluster_1m1w_helpers.go` | 管理集群上创建 1M1W **业务**集群 | 业务集群创建可复用 |
| `utils/cluster_manager.go` | `CreateClusterInBackground*`、`ScaleOut/In`、`DeleteCluster*` | 具备后台创建与并发组合能力 |
| `health_mechanism_test.go` 等 | 假定管理集群已存在并拉取 kubeconfig | 前置环境接入参考 |

### 2.2 缺口定义

「并发」在本模块指：**在已就绪的 1Master 管理集群上，短时间窗口内对 2 个及以上业务集群发起创建 / 删除 / 扩容 / 缩容操作**，包括：

1. 多业务集群同时创建或删除；
2. 创建与删除交叉进行（不同业务集群）；
3. 同一业务集群上同时扩容、缩容多个节点。

当前**无**专门 Describe/标签体系覆盖上述**管理集群业务面并发**场景。

> **不纳入 E2E 缺口**：节点资源争用（同一 IP 被多个集群声明）——由 Webhook/控制器准入拦截；正向用例通过节点隔离即可（见附录 B A1）。

---

## 3. 设计原则

1. **管理集群由前置环境提供**：**不在用例或 `BeforeAll` 中创建/删除管理集群**；流水线或人工已完成后，引导节点上存在与引导节点**同机共部署**的 **1Master** 管理集群（含 Cluster API），且 `state == Healthy`。
2. **统一运行面**：业务集群操作统一走 `mgmtKubeconfigPath`；命令经引导节点 SSH 下发。
3. **功能正确性优先**：断言业务集群 BC/BKENode 终态与节点占用，而非仅统计耗时。
4. **节点隔离**：管理集群仅占引导节点（`GUIDE_NODE_HOST`）；业务集群从 **`TEST_NODE2` 起**分配，不得与引导节点 IP 重叠。
5. **可清理**：`AfterEach` **仅**清理本用例创建的业务集群；**禁止**删除或 reset 管理集群。
6. **标签一致**：运行阶段标签表示**前置集群状态**（递进关系，见 §5.1）；Suite 仅声明本套件所需的**最高阶段** `with-management-cluster`，It 仅声明场景与 `online`。
7. **扩缩容验收最终一致**：仅在操作全部完成后断言节点集合与集群健康（见附录 B A3）。

---

## 4. 环境与资源要求

### 4.1 统一前置（所有用例）

**由流水线前置 Job 或人工一次性准备**（并发 Suite **不负责**安装管理集群）：

| 步骤 | 要求 | 对应标签（流水线 Job） |
|------|------|------------------------|
| 1 | 引导节点完成 `bke init` | `post-init`（`prescript` / 引导初始化 Job） |
| 2 | 在引导节点**同机**安装 **1Master** 管理集群（`mgmt-1master` 拓扑，含 Cluster API） | `with-management-cluster`（`mgmt-1master` Job） |
| 3 | 管理集群 `state == Healthy` 且 `clusterStatus == Ready` | `with-management-cluster` |
| 4 | 执行机 / 引导节点可获取管理集群 kubeconfig | `with-management-cluster` |

> **并发 Suite 标签**：仅打 `with-management-cluster`（已隐含 `post-init`，不再重复声明）。

**并发 Suite 启动时**（`BeforeAll` / `BeforeSuite`）**仅做校验与接入**，不创建集群：

| 项 | 要求 |
|----|------|
| 引导节点 SSH | `GUIDE_NODE_*` 可连接 |
| 管理集群健康 | 引导集群视角 BC 为 Healthy；管理集群自管 BC（`bke-{MGMT_CLUSTER_NAME}`，继承引导侧逻辑名，见 #224）为 Healthy |
| kubeconfig | `ClusterChecker.SaveKubeconfigToFile()` 或环境已配置的 `mgmtKubeconfigPath` |
| 管理集群名称 | 环境变量 `MGMT_CLUSTER_NAME`（实现阶段补充 `.env.template`），或按引导节点 `bke cluster list` 发现 |
| 业务集群操作 | `CreateClusterInBackgroundWithKubeconfig(..., mgmtKubeconfigPath)` 等 |

**拓扑示意**：

```
【前置环境已就绪，用例不改动左侧】

引导节点（同机共部署）       业务节点池（用例使用）
┌─────────────────┐         ┌──────────────────────────┐
│ bke init        │         │ TEST_NODE2 … TEST_NODE5  │
│ 1Master 管理集群 │ ──────► │ 并发创建/扩缩容的业务集群  │
└─────────────────┘         └──────────────────────────┘
```

> **与 `mgmt-1master` 用例关系**：前置环境安装步骤对齐 `management_cluster_test.go`；并发 Suite 只消费该环境，不重复执行安装。

### 4.2 节点规模分级

管理集群固定占用 **1 台**（引导节点）。业务节点从 **`TEST_NODE2`** 起。

| 级别 | 业务可用节点 | 适用用例 | 说明 |
|------|-------------|----------|------|
| L1（P0） | `TEST_NODE2`–`TEST_NODE5`（4 台） | CONC-01、04～09 | 2×1Master 业务集群并发；单集群扩缩容 |
| L1+（P1） | `TEST_NODE2`–`TEST_NODE6`（5 台） | CONC-10 | 2×1M1W 业务集群并发创建 + 跨集群并发缩容/扩容 |
| L2（P1） | 8 台（`TEST_NODE2` 起） | CONC-02、03、11 | 2～3×1M1W 业务集群并发 |
| L3（P2） | 12+ 台 | 三集群并发 + 删除复用 | 扩展夜间流水线 |

环境变量：沿用 `e2e/.env`（`GUIDE_NODE_*`、`TEST_NODE{n}_*`）；实现时补充 `MGMT_CLUSTER_NAME`（可选）。

> **环境检查**：Suite 入口断言 `GUIDE_NODE_HOST` 与 `TEST_NODE1_IP` 一致（若均配置）；管理集群不健康时 **Skip 全套件** 并输出明确日志（提示先跑 `mgmt-1master` 前置或检查流水线）。

### 4.3 超时建议

| 场景 | `SpecTimeout` / Suite timeout |
|------|-------------------------------|
| Suite `BeforeAll` 校验 + 拉取 kubeconfig | ≤ 5min（无安装耗时） |
| 双业务集群并发创建/删除 | `InstallationItTimeout` |
| 三业务集群及以上 | `2 * InstallationItTimeout` 或 `ConcurrentItTimeout` |

---

## 5. 用例总览

共 **11** 条用例；前置均为**已就绪的引导节点同机 1Master 管理集群**上的业务集群操作。

### 5.1 标签规范

运行阶段标签表示**前置环境集群就绪状态**（与 `installation/README.md` 一致），呈递进关系：`pre-init` → `post-init` → `with-management-cluster` → `with-workload-cluster`。**Suite 只声明本套件所需的最高阶段**，低阶段不再重复打标。

| 标签 | 含义 | 本模块是否使用 |
|------|------|----------------|
| `pre-init` | 引导节点初始化前 | **否** |
| `post-init` | 引导节点已就绪 | **否**（已由 `with-management-cluster` 隐含；前置 Job 如 `prescript` 使用） |
| `with-management-cluster` | 管理集群已就绪 | **是**（Suite 唯一运行阶段标签） |
| `with-workload-cluster` | 业务集群已就绪 | **否**（业务集群在用例 / `BeforeEach` 内创建） |

**分层声明**（对齐 `prescript_installation_test.go` 等）：

| 层级 | 声明位置 | 标签 |
|------|----------|------|
| **Suite** | 顶层 `Describe` | `concurrent`, `mgmt-1master`, **`with-management-cluster`**, `online` |
| **It** | 各 `It` | `concurrent`, `<场景>`, `[拓扑]`, `<P0\|P1\|P2>`, `online` |

说明：

- `installation` 由 `SIGDescribe` 自动注入。
- **运行阶段标签只打在 Suite**，且**仅声明最高阶段** `with-management-cluster`；**It 不重复**。
- **不使用 `pre-init`、`post-init`、`with-workload-cluster`**。
- `mgmt-1master` 描述管理集群拓扑（1Master 同机共部署），与运行阶段标签互补。
- **所有 It 统一包含 `online`**；优先级 `P0` / `P1` / `P2` 在 It 级声明。

**Suite 示例**：

```go
var _ = SIGDescribe("安装部署并发场景",
    Label("concurrent", "mgmt-1master", "with-management-cluster", "online"), func() {
```

**It 示例**（CONC-01）：

```go
It("...", Label("concurrent", "multi-cluster-create", "1master", "P0", "online"), func(ctx SpecContext) {
```

### 5.2 用例清单

| 编号 | 操作类型 | 用例名称 | 优先级 | 环境 | Suite 前置标签 | It 标签 |
|------|----------|----------|--------|------|----------------|---------|
| CONC-01 | 创建 | 管理集群上 2 个 1Master 业务集群并发创建成功 | P0 | L1 | `with-management-cluster` | `concurrent`, `multi-cluster-create`, `1master`, `P0`, `online` |
| CONC-02 | 创建 | 管理集群上 2 个 1M1W 业务集群并发创建成功 | P0 | L2 | 同上 | `concurrent`, `multi-cluster-create`, `1m1w`, `P0`, `online` |
| CONC-03 | 创建 | 管理集群上 3 个业务集群并发创建（节点完全隔离） | P1 | L2 | 同上 | `concurrent`, `multi-cluster-create`, `1m1w`, `P1`, `online` |
| CONC-04 | 创建 | 管理集群上并发 `bke cluster create` CLI（不同业务集群配置） | P0 | L1 | 同上 | `concurrent`, `cluster-create`, `1master`, `P0`, `online` |
| CONC-05 | 删除 | 管理集群上 2 个已就绪业务集群并发删除成功 | P1 | L1 | 同上 | `concurrent`, `multi-cluster-delete`, `delete`, `P1`, `online` |
| CONC-06 | 创建+删除 | 业务集群 A 创建过程中对业务集群 B 并发删除 | P1 | L1 | 同上 | `concurrent`, `create-delete-race`, `P1`, `online` |
| CONC-07 | 删除+创建 | 两业务集群并发删除后，节点可被新业务集群复用创建 | P1 | L1 | 同上 | `concurrent`, `cleanup-reuse`, `P1`, `online` |
| CONC-08 | 扩容 | 同一业务集群并发扩容 2 个不同 Worker（`kubectl apply` BKENode） | P0 | L1 | 同上 | `concurrent`, `scale`, `scale-out`, `P0`, `online` |
| CONC-09 | 缩容 | 同一业务集群并发缩容 2 个不同 Worker | P1 | L1 | 同上 | `concurrent`, `scale`, `scale-in`, `P1`, `online` |
| CONC-10 | 创建+扩容+缩容 | 两业务集群并发创建后跨集群并发缩容与扩容（不同集群、不同节点） | P1 | L1+（5 台） | 同上 | `concurrent`, `scale`, `scale-race`, `P1`, `online` |
| CONC-11 | 创建+扩容 | 业务集群 A 创建过程中对业务集群 B 并发扩容 | P2 | L2 | 同上 | `concurrent`, `create-scale-race`, `P2`, `online` |

> **CONC-08 合并说明**：原 CONC-08（`ScaleOutNode`）与 CONC-09（直接 `kubectl apply` BKENode）底层均为 BKENode 声明式扩容，合并为一条；实现采用 `ScaleOutNodeWithKubeconfig` 并发调用即可（内部即 `kubectl apply` BKENode）。

> Suite 前置标签由顶层 `Describe` 统一声明一次；上表「Suite 前置标签」列仅为说明各用例对环境的共同要求。

---

## 6. 详细用例设计

> **通用说明**（各用例均继承，不再逐条重复）：
>
> - Suite `BeforeAll` 已完成 §6.0 接入，持有 `mgmtKubeconfigPath`；
> - 命令经引导节点 SSH 执行，`kubectl` / `bke` 均带 `mgmtKubeconfigPath`；
> - 业务节点从 `TEST_NODE2` 起分配，**不得占用** `GUIDE_NODE_HOST`；
> - **用例内不创建、不删除管理集群**；`AfterEach` 仅清理本用例业务集群。

### 6.0 Suite 级环境接入（非安装）

**前置条件**：

- 流水线或人工已完成 §4.1 前置（引导节点 init + 同机 1M 管理集群 Healthy）；
- `e2e/.env` 中 `GUIDE_NODE_*`、`TEST_NODE{n}_*` 已配置且 SSH 可达。

**用例步骤**（`BeforeAll`，全套件一次）：

1. SSH 连接引导节点。
2. 解析管理集群名称（`MGMT_CLUSTER_NAME` 或 `bke cluster list`）。
3. 校验引导集群视角 BC：`GetClusterFullStatusWithKubeconfig(mgmtName, "")` → `state == Healthy` 且 `clusterStatus == Ready`。
4. `ClusterChecker.SaveKubeconfigToFile()` 得到 `mgmtKubeconfigPath`。
5. 校验管理集群自管 BC：`GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, mgmtName)` → Healthy。

**预期结果**：

- 上述校验全部通过，后续用例可复用 `mgmtKubeconfigPath`；
- 任一失败 → `Skip` 全套件，日志提示检查 `mgmt-1master` 前置 Job。

**清理**：无 `AfterAll` 管理集群清理。

---

### 6.1 集群创建

#### CONC-01 管理集群上 2 个 1Master 业务集群并发创建成功

**标签**：`concurrent`, `multi-cluster-create`, `1master`, `P0`, `online`

**前置条件**：

- §6.0 已完成，`mgmtKubeconfigPath` 可用；
- 环境 **L1**：`TEST_NODE2`、`TEST_NODE3` 空闲，分别作为两个业务集群的唯一 Master；
- 两业务集群名称唯一，与环境中已有 BC 不冲突。

**用例步骤**：

1. 分别为集群 `wl-a`、`wl-b` 生成 1Master 拓扑的 BKECluster + BKENode 配置，上传引导节点。
2. 使用 `sync.WaitGroup` 启动 2 个 goroutine，同时调用 `CreateClusterWithKubeconfig(..., mgmtKubeconfigPath)`（声明式 CR：`kubectl apply` BKENode → BKECluster）。
3. `WaitGroup.Wait()` 等待两条创建命令均返回（不等待集群就绪）。
4. 对 `wl-a`、`wl-b` 分别 `Eventually` 轮询 `GetClusterFullStatusWithKubeconfig` 直至 `state == Healthy` 且 `clusterStatus == Ready`。
5. 在管理集群执行 `kubectl get bc -A`（带 `mgmtKubeconfigPath`），确认存在两个业务 BC。
6. 校验两集群 BKENode 所用 IP 分别为 `TEST_NODE2`、`TEST_NODE3`，且均 ≠ `GUIDE_NODE_HOST`。

**预期结果**：

- 两个业务集群均创建成功并达 Healthy/Ready；
- 可见 2 个业务 BC，节点 IP 互不重叠、未占用引导节点。

**清理**：`AfterEach` 并发 `DeleteClusterWithKubeconfig(wl-a/b, mgmtKubeconfigPath)`，直至 BC 不存在。

---

#### CONC-02 管理集群上 2 个 1M1W 业务集群并发创建成功

**标签**：`concurrent`, `multi-cluster-create`, `1m1w`, `P0`, `online`

**前置条件**：

- §6.0 已完成；
- 环境 **L2**（至少 8 台业务节点，`TEST_NODE2` 起）；
- `PartitionWorkloadNodes` 切分为 2 组，每组 1 Master + 1 Worker，节点互不重叠。

**用例步骤**：

1. 为集群 `wl-a`、`wl-b` 各生成 1M1W 配置（BKECluster + BKENode），上传引导节点。
2. `sync.WaitGroup` + 2 goroutine 并发 `CreateClusterWithKubeconfig(..., mgmtKubeconfigPath)`。
3. 等待创建命令返回后，并行 `Eventually` 轮询两集群直至 Healthy/Ready。
4. 校验各集群 BKENode 数量 = 2（1 Master + 1 Worker），且两组节点 IP 集合无交集。

**预期结果**：

- 2 个 1M1W 业务集群均 Healthy/Ready；
- 共 4 个 BKENode 分布在 4 台不同业务节点上。

**清理**：`AfterEach` 并发删除 `wl-a`、`wl-b`。

---

#### CONC-03 管理集群上 3 个业务集群并发创建（节点完全隔离）

**标签**：`concurrent`, `multi-cluster-create`, `1m1w`, `P1`, `online`

**前置条件**：

- §6.0 已完成；
- 环境 **L2**（至少 6 台业务节点，推荐 8 台）；
- `PartitionWorkloadNodes(..., mastersPerCluster=1, workersPerCluster=1, clusterCount=3)` 成功返回 3 组互斥节点。

**用例步骤**：

1. 为集群 `wl-a`、`wl-b`、`wl-c` 各生成 1M1W 配置并上传。
2. `sync.WaitGroup` + 3 goroutine 并发 `CreateClusterWithKubeconfig(..., mgmtKubeconfigPath)`。
3. 等待命令返回后，对 3 个集群并行 `Eventually` 轮询直至均 Healthy/Ready。
4. 汇总 3 组 BKENode IP，断言并集大小 = 6 且无重复。

**预期结果**：

- 3 个业务集群全部创建成功并达 Healthy/Ready；
- 6 个节点各归属唯一集群，无 IP 冲突。

**清理**：`AfterEach` 并发删除 3 个业务集群。

---

#### CONC-04 管理集群上并发 `bke cluster create` CLI（不同业务集群配置）

**标签**：`concurrent`, `cluster-create`, `1master`, `P0`, `online`

**前置条件**：

- §6.0 已完成；
- 环境 **L1**：`TEST_NODE2`、`TEST_NODE3` 空闲；
- 与 CONC-01 节点规划相同，但走 **CLI 路径**（非 CONC-01 的 CR apply 路径）。

**用例步骤**：

1. 为 `wl-a`、`wl-b` 各生成 1Master 集群配置（`-f` 集群 YAML、`-n` 节点 YAML），上传引导节点。
2. `sync.WaitGroup` + 2 goroutine 并发 `CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, mgmtKubeconfigPath)`（`bke --kubeconfig <mgmt> cluster create`）。
3. 等待后台命令触发完成后，并行 `Eventually` 轮询两集群直至 Healthy/Ready。
4. 校验 `kubectl get bc -A` 含 2 个新业务 BC；节点 IP 互斥。

**预期结果**：

- 通过 **BKE CLI** 并发创建的两业务集群均达 Healthy/Ready；
- 验证 CLI 编排链路与 CR 路径（CONC-01）均可承受并发。

**清理**：`AfterEach` 并发删除两业务集群。

---

### 6.2 集群删除

#### CONC-05 管理集群上 2 个已就绪业务集群并发删除成功

**标签**：`concurrent`, `multi-cluster-delete`, `delete`, `P1`, `online`

**前置条件**：

- §6.0 已完成；
- 环境 **L1**；
- 用例 `BeforeEach` 已串行或并行创建 `wl-a`、`wl-b`（各 1Master），且均已 **Healthy/Ready**。

**用例步骤**：

1. 确认 `wl-a`、`wl-b` 的 `GetClusterFullStatusWithKubeconfig` 均为 Healthy/Ready。
2. `sync.WaitGroup` + 2 goroutine 并发 `DeleteClusterWithKubeconfig(clusterName, mgmtKubeconfigPath)`。
3. `WaitGroup.Wait()` 后，对两集群 `Eventually` 轮询 `ClusterExistsWithKubeconfig` 直至均为 `false`。
4. 可选：`WaitForClusterDeleted` 确认 BC/BKENode 资源已清理。

**预期结果**：

- 两个业务集群均被成功删除，对应 BC 不再存在；
- 删除过程无 hang。

**清理**：`AfterEach` 幂等确认两集群已不存在（`BeforeEach` 创建的资源已在本用例内删除）。

---

#### CONC-06 业务集群 A 创建过程中对业务集群 B 并发删除

**标签**：`concurrent`, `create-delete-race`, `P1`, `online`

**前置条件**：

- §6.0 已完成；
- 环境 **L1**：`TEST_NODE2`（集群 A）、`TEST_NODE3`（集群 B）；
- `BeforeEach` 已创建集群 **B**（`wl-b`，1Master）并达 **Healthy/Ready**。

**用例步骤**：

1. 为集群 **A**（`wl-a`）生成 1Master 配置并上传。
2. 在 goroutine-1 中调用 `CreateClusterInBackgroundWithKubeconfig` 启动 **A** 的创建（后台执行）。
3. 几乎同时，在 goroutine-2 中调用 `DeleteClusterWithKubeconfig(wl-b, mgmtKubeconfigPath)` 删除 **B**。
4. `WaitGroup.Wait()` 等待两操作均返回。
5. `Eventually` 轮询：**A** 达 Healthy/Ready；**B** 的 `ClusterExistsWithKubeconfig` 为 `false`。

**预期结果**：

- 创建与删除交叉进行互不影响：**A** 最终创建成功，**B** 最终被删除。

**清理**：`AfterEach` 删除 **A**（**B** 已在用例中删除）。

---

#### CONC-07 两业务集群并发删除后，节点可被新业务集群复用创建

**标签**：`concurrent`, `cleanup-reuse`, `P1`, `online`

**前置条件**：

- §6.0 已完成；
- 环境 **L1**：`TEST_NODE2`、`TEST_NODE3` 用于初始两集群；
- `BeforeEach` 已创建 `wl-a`、`wl-b`（各 1Master）并均 Healthy/Ready。

**用例步骤**：

1. 确认 `wl-a`、`wl-b` 分别占用 `TEST_NODE2`、`TEST_NODE3`。
2. `sync.WaitGroup` + 2 goroutine 并发 `DeleteClusterWithKubeconfig` 删除 `wl-a`、`wl-b`。
3. `Eventually` 等待两集群 BC 均不存在。
4. 使用原 `TEST_NODE2` 节点，创建新业务集群 `wl-reuse`（1Master，`CreateClusterWithKubeconfig` 或 CLI 均可）。
5. `Eventually` 轮询 `wl-reuse` 直至 Healthy/Ready。

**预期结果**：

- 并发删除完成后，原节点资源已释放；
- 新业务集群可在原节点上成功创建并达 Healthy/Ready；
- 验证删除清理与节点复用链路正确。

**清理**：`AfterEach` 删除 `wl-reuse`。

---

### 6.3 扩缩容

#### CONC-08 同一业务集群并发扩容 2 个不同 Worker（`kubectl apply` BKENode）

**标签**：`concurrent`, `scale`, `scale-out`, `P0`, `online`

**前置条件**：

- §6.0 已完成；
- 环境 **L1**：至少 4 台业务节点；
- `BeforeEach` 已创建 1M1W 业务集群 `wl-scale`（Master=`TEST_NODE2`，Worker=`TEST_NODE3`）并 **Healthy/Ready**；
- `TEST_NODE4`、`TEST_NODE5` 作为待扩容 Worker，角色 `node`。

**用例步骤**：

1. 确认 `wl-scale` 当前 BKENode 数为 2 且状态 Healthy/Ready。
2. `sync.WaitGroup` + 2 goroutine 并发调用 `ScaleOutNodeWithKubeconfig(wl-scale, node4, mgmtKubeconfigPath)` 与 `ScaleOutNodeWithKubeconfig(wl-scale, node5, mgmtKubeconfigPath)`（内部 `kubectl apply` BKENode）。
3. `WaitGroup.Wait()` 等待两次扩容命令返回。
4. `Eventually`（`eventuallyClusterScaleSettled`，超时 `InstallationItTimeout`）联合轮询直至：
   - `wl-scale` 非 `Scaling*` / `EnsureWorker*` 等中间态，且 `state == Healthy`、`clusterStatus == Ready`；
   - BKENode 共 4 个，含 `TEST_NODE4`、`TEST_NODE5`。

**预期结果**（按附录 B **A3**，仅验收**终态**）：

- 并发扩容完成后，集群最终 Healthy/Ready；
- 节点集合包含原 2 节点 + 新增 2 Worker，无重复 IP；
- 不断言扩容过程中的中间态或操作顺序。

**清理**：`defer cleanupScaleWorkloadClusters` 删除 `wl-scale`（删除前 `waitClusterQuiescent`，删除等待 `scaleUninstallTimeout`）。

---

#### CONC-09 同一业务集群并发缩容 2 个不同 Worker

**标签**：`concurrent`, `scale`, `scale-in`, `P1`, `online`

**前置条件**：

- §6.0 已完成；
- 环境 **L1**：至少 4 台业务节点；
- `BeforeEach` 已创建 1M3W 业务集群 `wl-scale`（1 Master + 3 Worker，占用 `TEST_NODE2`–`TEST_NODE5`）并 **Healthy/Ready**；
- 待缩容目标为其中 2 个 Worker 的 IP（保留 1 Master + 1 Worker）。

**用例步骤**：

1. 确认 `wl-scale` BKENode 数为 4 且 Healthy/Ready。
2. `sync.WaitGroup` + 2 goroutine 并发 `ScaleInNodeWithKubeconfig(wl-scale, workerIP1, mgmtKubeconfigPath)` 与 `ScaleInNodeWithKubeconfig(wl-scale, workerIP2, mgmtKubeconfigPath)`。
3. `WaitGroup.Wait()` 等待两次缩容命令返回。
4. `Eventually`（`eventuallyClusterScaleSettled`）联合轮询直至集群非扩缩容中间态、Healthy/Ready，BKENode 数为 2，且被缩容节点 IP 已不在 BKENode 列表中。

**预期结果**（A3，仅终态）：

- 并发缩容完成后集群 Healthy/Ready；
- 最终保留 1 Master + 1 Worker，共 2 个 BKENode；
- 不校验缩容过程中的瞬时状态。

**清理**：`defer cleanupScaleWorkloadClusters` 删除 `wl-scale`。

---

#### CONC-10 两业务集群并发缩容与扩容（不同集群、不同节点）

**标签**：`concurrent`, `scale`, `scale-race`, `P1`, `online`

**前置条件**：

- §6.0 已完成；
- 环境 **L1+**：至少 **5** 台业务节点（`TEST_NODE2`～`TEST_NODE6`）；
- **wl-shrink** 拓扑：1M1W，Master=`TEST_NODE2`，Worker=`TEST_NODE3`；
- **wl-grow** 拓扑：1M1W，Master=`TEST_NODE4`，Worker=`TEST_NODE5`；
- `TEST_NODE6` 为 wl-grow 待扩容 Worker；wl-shrink 待缩容 Worker 为 `TEST_NODE3`。

**用例步骤**：

1. 为 **wl-shrink**、**wl-grow** 分别生成 1M1W 配置并上传。
2. `sync.WaitGroup` + 2 goroutine 并发 `CreateClusterWithKubeconfig` 创建两集群。
3. `waitClustersHealthyReady` 轮询直至两集群均 Healthy/Ready。
4. goroutine-1：`ScaleInNodeWithKubeconfig(wl-shrink, TEST_NODE3, mgmtKubeconfigPath)`；
5. goroutine-2：`ScaleOutNodeWithKubeconfig(wl-grow, TEST_NODE6, mgmtKubeconfigPath)`；
6. 两 goroutine 通过 `sync.WaitGroup` 几乎同时触发，`Wait()` 等待返回。
7. `eventuallyClusterScaleSettled` 分别轮询终态：
   - **wl-shrink**：非扩缩容中间态 + Healthy/Ready；BKENode 仅含 Master（`TEST_NODE2`）；
   - **wl-grow**：非扩缩容中间态 + Healthy/Ready；BKENode 含 `TEST_NODE4`、`TEST_NODE5`、`TEST_NODE6`。
8. 断言两集群 BKENode IP 互不重叠。

**预期结果**（A3，仅终态）：

- 两业务集群可并发创建；创建完成后，不同集群上的缩容与扩容可并发执行，最终均 Healthy/Ready；
- wl-shrink 缩容后仅保留 Master；wl-grow 扩容后新增 Worker；
- 不因 BKENode CR 已创建但 Agent 未就绪（如 `2/3`）或 BC 处于 `ScalingWorkerNodes*` 即提前进入清理。

**清理**：`defer cleanupScaleWorkloadClusters` 删除 `wl-shrink`、`wl-grow`（删除前对各集群 `waitClusterQuiescent`）。

---

### 6.4 交叉场景

#### CONC-11 业务集群 A 创建过程中对业务集群 B 并发扩容

**标签**：`concurrent`, `create-scale-race`, `P2`, `online`

**前置条件**：

- §6.0 已完成；
- 环境 **L2**：至少 5 台业务节点；
- `BeforeEach` 已创建 1M1W 集群 **B**（`wl-b`）并 **Healthy/Ready**（如 `TEST_NODE2` Master、`TEST_NODE3` Worker）；
- 集群 **A** 使用 `TEST_NODE4`（1Master）；**B** 的扩容节点为 `TEST_NODE5`。

**用例步骤**：

1. 为集群 **A**（`wl-a`）生成 1Master 配置并上传。
2. goroutine-1：`CreateClusterInBackgroundWithKubeconfig` 启动 **A** 创建。
3. goroutine-2：`ScaleOutNodeWithKubeconfig(wl-b, node5, mgmtKubeconfigPath)` 对 **B** 扩容。
4. `WaitGroup.Wait()` 等待两操作返回。
5. `eventuallyClusterScaleSettled` 分别轮询终态：
   - **A**：非中间态 + Healthy/Ready；BKENode 含 `TEST_NODE4`（1 Master）；
   - **B**：非中间态 + Healthy/Ready；BKENode 含 `TEST_NODE2`、`TEST_NODE3`、`TEST_NODE5`。

**预期结果**：

- 不同业务集群上的「创建」与「扩容」可并发执行，最终均成功；
- **A**、**B** 终态均为 Healthy/Ready。

**清理**：`defer cleanupScaleWorkloadClusters` 删除 `wl-a`、`wl-b`。

---

> **不纳入 E2E 范围**：同集群名重复创建（Webhook 拦截，见附录 B A2）；两集群声明同一节点（Webhook 拦截，见 A1）。

---

## 7. 技术实现方案

### 7.1 建议目录与文件

```
e2e/installation/concurrent/
├── concurrent_installation_test.go
├── concurrent_scale_test.go
├── concurrent_installation_helpers.go
└── doc/concurrent-test-design.md
```

- 顶层 `SIGDescribe`：`Label("concurrent", "mgmt-1master", "with-management-cluster", "online")`；各 `It` 标签见 §5.2。
- **`BeforeAll`**：仅 `ResolvePreprovisionedMgmtCluster`（校验 + kubeconfig），**不**调用 `CreateClusterInBackground`。

### 7.2 Helper 设计草案

```go
// concurrent_installation_helpers.go（示意）

type MgmtClusterContext struct {
    MgmtClusterName    string
    MgmtKubeconfigPath string
    GuideNodeIP        string
    SSHExecutor        *executor.SSHExecutor
    ClusterManager     *utils.ClusterManager
}

// ResolvePreprovisionedMgmtCluster 接入前置环境：校验 1M 管理集群 Healthy 并返回 kubeconfig
// 管理集群不存在或不健康时返回 error，Suite 层 Skip
func ResolvePreprovisionedMgmtCluster(
    cm *utils.ClusterManager,
    localExec *executor.LocalExecutor,
    mgmtClusterName string,
) (*MgmtClusterContext, error)

// PartitionWorkloadNodes 从 TEST_NODE2 起切分，排除引导节点 IP
func PartitionWorkloadNodes(...) ([][]config.NodeInfo, error)

// eventuallyClusterScaleSettled 扩缩容终态联合校验（Eventually）：
// 非 Scaling*/EnsureWorker* 中间态 + Healthy/Ready + BKENode IP 集合
func eventuallyClusterScaleSettled(clusterName string, wantIPs, excludeIPs []string, timeout time.Duration)

// waitClusterQuiescent 删除前等待集群离开扩缩容中间态并达 Healthy/Ready
func waitClusterQuiescent(clusterName string, timeout time.Duration)

// cleanupScaleWorkloadClusters 扩缩容用例清理：quiescent → 并发删除 → scaleUninstallTimeout（30min）
func cleanupScaleWorkloadClusters(jobs []WorkloadClusterJob)
```

### 7.3 并发执行模式

| 模式 | 适用 | 实现参考 |
|------|------|----------|
| `BeforeAll` 校验前置 1M 管理集群 | 全套件 | `health_mechanism_test.go`、`cluster_test_helpers.go` |
| goroutine + WaitGroup | 业务集群并发操作 | `performance-test/installation_performance_test.go` |
| `CreateClusterInBackgroundWithKubeconfig(..., mgmt)` | 业务集群创建 | `cluster_manager.go` |

### 7.4 断言标准

| 检查项 | 手段 |
|--------|------|
| 前置 1M 管理集群可用 | Suite 入口 `GetManagementClusterSelfFullStatus` → Healthy |
| 业务集群健康 | `GetClusterFullStatusWithKubeconfig(name, mgmtKubeconfig)` |
| 节点互斥 | 业务 BKENode IP ≠ `GUIDE_NODE_HOST` |
| 扩缩容终态 | `eventuallyClusterScaleSettled`：非中间态 + Healthy/Ready + BKENode IP 集合；不校验操作顺序 |
| 扩缩容清理 | `cleanupScaleWorkloadClusters`：删除前 `waitClusterQuiescent`；删除等待 `scaleUninstallTimeout`（30min） |

### 7.5 运行命令

```bash
# 前置流水线：post-init Job → mgmt-1master Job → concurrent Suite
cd e2e && ginkgo -v --timeout=8h --label-filter="concurrent && with-management-cluster && online" ./installation/concurrent
cd e2e && ginkgo -v --timeout=8h --label-filter="concurrent && P0 && online" ./installation/concurrent
cd e2e && ginkgo -v --timeout=8h --label-filter="concurrent && scale && online" ./installation/concurrent

# 仅扩缩容与交叉场景（concurrent_scale_test.go，CONC-08~11）
cd e2e && ginkgo -v --timeout=8h --label-filter="concurrent && (scale || create-scale-race)" ./installation/concurrent
```

**流水线建议顺序**：

```bash
ginkgo -v --label-filter="post-init && online" ./installation                    # 引导节点 init（若未完成）
ginkgo -v --label-filter="mgmt-1master && post-init && online" ./installation    # 同机 1M 管理集群
ginkgo -v --label-filter="concurrent && with-management-cluster && online" ./installation/concurrent
```

---

## 8. 实施路线

| 阶段 | 用例 | 产出 |
|------|------|------|
| **Phase 1** | `ResolvePreprovisionedMgmtCluster` + CONC-01、04、08 | L1：接入前置 1M 管理集群 + 并发创建/扩容 |
| **Phase 2** | CONC-05、06、07、09 | 并发删除、竞态、缩容 |
| **Phase 3** | CONC-02、03、10 | 多 1M1W 并发、扩缩容竞态 |
| **Phase 4** | CONC-11 + README / 流水线文档 | 交叉场景与前置 Job 说明 |

---

## 9. 风险与依赖

1. **强依赖前置环境**：无引导节点同机 1M 管理集群时全套件 Skip；需在 CI 文档中明确前置 Job（`mgmt-1master`）。
2. **引导节点共机**：业务集群不得占用 `GUIDE_NODE_HOST`。
3. **节点不足**：L1 业务节点 4 台（`TEST_NODE2`–`5`）；1M1W×2 需 L2。
4. **清理边界**：`AfterEach` **禁止** `DeleteCluster` / `DeleteManagementClusterSelfBC` 作用于管理集群。
5. **管理集群名称**：多集群并存时需 `MGMT_CLUSTER_NAME` 明确指定，避免 kubeconfig 拉错对象。

---

## 10. 验收标准

- [ ] 前置环境（`with-management-cluster`）下，`ginkgo --label-filter="concurrent && with-management-cluster && online"` 通过率 ≥ 95%（连跑 3 次）
- [ ] 无前置管理集群时 Suite 明确 Skip 并提示
- [ ] 用例结束后本用例创建的业务 BC 均已清理
- [ ] 业务节点未占用引导节点 IP
- [ ] `installation/README.md` 补充：并发用例依赖前置 `mgmt-1master` 共部署环境

---

## 11. 参考实现

| 参考 | 路径 |
|------|------|
| 前置 1M 管理集群安装（流水线 Job，非并发 Suite） | `management_cluster_test.go`（`mgmt-1master`） |
| 管理集群 kubeconfig / 自管 BC 校验 | `health_mechanism_test.go`、`cluster_test_helpers.go` |
| 业务集群 1M1W 模板 | `cluster_1m1w_helpers.go` → `Create1M1WWorkloadCluster` |
| 引导节点配置 | `bke-config/cluster_config.go` → `LoadGuideNodeFromEnv` |
| 业务节点 offset | `installation_abnormal_test.go`（业务从 `TEST_NODE2` / offset=2 起） |

---

## 附录 A：标签映射

### 运行阶段标签（递进，Suite 仅声明最高阶段）

| 标签 | 含义 | 本模块 |
|------|------|--------|
| `post-init` | 引导节点已就绪 | 隐含于 `with-management-cluster`；前置 Job 使用 |
| `with-management-cluster` | 管理集群已就绪 | **Suite 唯一运行阶段标签** |
| `pre-init` | 引导节点初始化前 | 不使用 |
| `with-workload-cluster` | 业务集群已就绪 | 不使用（用例内创建） |

### Suite（Describe）

```
concurrent + mgmt-1master + with-management-cluster + online
# installation 由 SIGDescribe 注入
```

### It（场景 + 优先级 + online，不含运行阶段标签）

```
concurrent + multi-cluster-create + 1master|1m1w + P0|P1 + online
concurrent + cluster-create + 1master + P0 + online
concurrent + multi-cluster-delete + delete + P1 + online
concurrent + create-delete-race + P1 + online
concurrent + cleanup-reuse + P1 + online
concurrent + scale + scale-out + P0 + online
concurrent + scale + scale-in + P1 + online
concurrent + scale + scale-race + P1 + online
concurrent + create-scale-race + P2 + online
```

### CI 常用筛选

```
concurrent && with-management-cluster && online
concurrent && with-management-cluster && P0 && online
concurrent && scale && online
```

## 附录 B：已确认产品行为

| ID | 结论 | 对 E2E 的影响 |
|----|------|---------------|
| **A1** | 不能出现同一节点被两个集群声明 | `PartitionWorkloadNodes` 排除引导节点；业务节点互斥 |
| **A2** | 同名集群由 Webhook 拦截 | 不写负向并发用例 |
| **A3** | 扩缩容保证最终一致性 | CONC-08～11 用 `eventuallyClusterScaleSettled` 联合验收终态；CONC-10 含并发创建与跨集群并发缩容+扩容 |
