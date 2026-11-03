# 统一前置检查 Preflight E2E 测试

## 概述

本测试套件用于验证 BKE 统一前置检查（preflight）功能的端到端行为，基于 Ginkgo v2 测试框架实现。覆盖 `bke preflight init`（引导节点初始化前检查）与 `bke preflight cluster-create`（创建集群前检查）两个操作，验证 JSON 检查报告的整体结果、summary 四态算术、检查项状态（PASS/FAIL/WARN/SKIP）、期望/实际值契约、修复建议（remediation）以及 issues[] 输入校验定位。

关联用例里程碑：`26.09`。套件位于 `installation/` 特性下，作为独立子目录维护（独立 Suite 入口）。

绝大多数自动化用例通过 `--preflight-config` 显式传入包内 `preflight.yaml`（资源阈值：init `cpu≥2 / memory≥6GiB / systemDisk≥40GiB`，cluster-create `cpu≥4 / memory≥6GiB / systemDisk≥40GiB`），与 bke 内嵌默认策略解耦，避免环境规格误 FAIL。

## 测试范围

### init 操作（Label: `preflight` / `pre-init` / `26.09`，在节点一 TEST_NODE1 上执行）

| 用例 | 优先级 | 场景 | 关键断言 |
|------|--------|------|----------|
| 【PREFLIGHT-INIT-001】 | P0 | 外部策略正常执行 | Overall=PASS exit 0，全检查项 PASS，summary 四态算术一致 |
| 【PREFLIGHT-INIT-002】 | P1 | 端口冲突（占用 38080） | FAIL exit 3，单根因端口项 FAIL 含 remediation，释放后重跑 PASS |
| 【PREFLIGHT-INIT-003】 | P1 | `--kubernetesPort abc/70000` | INPUT_ERROR exit 4，提示"前置检查未启动"，不产生伪造检查项 |
| 【PREFLIGHT-INIT-004】 | P2 | 外部策略改 cpu minimum 2→99 | CPU 项 FAIL，configDigest 与默认不同，其余项不受影响 |
| 【PREFLIGHT-INIT-006】 | P1 | 离线模式正向与条件实例化 | 与 001 相同传入外部策略；物料用例内生成（tar.gz 制品包/自签 CA/本机 TLS 仓库监听），带离线参数 → PASS，INIT-OFFLINE/REPO-CA/INIT-SRC 各项 PASS 且目标列指向实际物料；去掉离线参数重跑 → 对应项 0 条、无 SKIP |

### cluster-create 操作（Label: `preflight` / `with-management-cluster` / `26.09`，在节点一 TEST_NODE1 上执行）

| 用例 | 优先级 | 场景 | 关键断言 |
|------|--------|------|----------|
| 【CLUSTER-001】 | P0 | 外部策略正常执行 | 管理集群检查链（API/Discovery/CRD/Secret）≥10 项全 PASS，节点 SSH 采集链全 PASS，资源实际值与 nproc/free/df 交叉核对 |
| 【CLUSTER-002】 | P1 | 节点资源不足（cpu minimum 4→999） | RES-CPU FAIL 期望/实际准确，其余项不受影响 |
| 【CLUSTER-003】 | P1 | imageRepo.domain 改为不可达地址 | 单条 FAIL 含 remediation，还原后重跑 PASS |
| 【CLUSTER-004】 | P1 | SSH 密码错误 | SSH-TRANSPORT PASS、SSH-AUTH FAIL，下游检查 SKIP，修正后 PASS |
| 【CLUSTER-005】 | P2 | 无效 kubeconfig | API FAIL，管理集群下游 SKIP，节点 SSH 采集不受影响 |
| 【CLUSTER-006】 | P2 | 未知字段 / 非法 podSubnet CIDR | 未知字段 → INPUT_ERROR + issues 定位 cluster.unknownField（BLOCK_START）；非法 CIDR → CLUSTER-NET-POD-CIDR FAIL（目标列经 evidence.locator 校验） |
| 【CLUSTER-008】 | P2 | 自定义 exec 检查 | fact JSON 公共字段按节点展开 PASS，detail 敏感值不泄漏，脚本缺失/非法输出 FAIL 不泄敏感 |
| 【CLUSTER-009】 | P1 | 端口规则按节点实例化 | master（TEST_NODE2）占 6443 → 仅 master 一条 FAIL（bool 契约 true→false），KUBELET/HEALTH 每节点各一条，SRC-REACHABLE 每节点每来源各一条，释放后 PASS |
| 【CLUSTER-010】 | P2 | 偶数 master 拓扑 | CLUSTER-TOPO-MASTER-ODD FAIL（relation 契约 expected=true/actual=false，subject=cluster），其余正常执行 |
| 【CLUSTER-011】 | P2 | 唯一性校验 | 同 IP → UNIQUE-IP 恰一条 FAIL；空 hostname → 相关检查项 PASS、无 SKIP/FAIL；同 hostname → UNIQUE-HOSTNAME 恰一条 FAIL |

未自动化：PREFLIGHT-INIT-005/007（用例 CSV 标记为人工）、PREFLIGHT-UT-001~016（UT 范畴）。

## 环境要求

| 组件 | 要求 |
|------|------|
| **Go** | 1.24.5（`e2e/go.mod`） |
| **Ginkgo** | v2 |
| **bke** | 已包含 `preflight init` / `preflight cluster-create` 子命令，默认路径 `/usr/local/bin/bke` |
| **测试节点** | 4 个 Linux 节点，SSH 密码认证互通 |

### 节点编排约定

```
节点一 TEST_NODE1  ← init 用例执行机（已创建引导节点，bke 可执行）
节点一 TEST_NODE1  ← cluster-create 用例执行机（已安装管理集群，kubeconfig 可用）
节点二 TEST_NODE2  ← 创建集群的 master-01（hostname 经 TEST_NODE2_HOSTNAME 提供给用例）
节点三 TEST_NODE3  ← 创建集群的 worker-01
```

- 节点实际规格建议 ≥8C/16G/100G（物理规格，非剩余容量）；用例外部策略阈值放宽为 memory≥6GiB、systemDisk≥40GiB，避免资源项误 FAIL。
- cluster.yaml 无需人工准备：cluster-create 用例直接读取 bke 自带模板 `/bke/cluster/1master1node-cluster.yaml`（节点一优先，缺失时回退引导节点），用例内通过字段替换注入故障（imageRepo.domain、podSubnet、cpu minimum、密码、未知字段等）。
- nodes.yaml 由测试代码按 `TEST_NODE2_*` / `TEST_NODE3_*` 环境变量渲染 BKENode 多文档 YAML，无需模板。

## 环境变量

统一使用 `e2e/.env`（复制自 `e2e/.env.template`），套件 `init()` 通过 `framework/env.LoadEnv(".env")` 加载，本模块不单独维护环境变量文件。

### 必需

| 变量 | 说明 |
|------|------|
| `TEST_NODE1_IP/HOSTNAME/PORT/USER/PASSWORD` | 节点一 SSH 凭据（init/cluster-create 用例执行机） |
| `TEST_NODE2_IP/HOSTNAME/PORT/USER/PASSWORD` | 节点二 SSH 凭据与 hostname（master-01，nodes.yaml 渲染数据源） |
| `TEST_NODE3_IP/HOSTNAME/PORT/USER/PASSWORD` | 节点三 SSH 凭据与 hostname（worker-01，nodes.yaml 渲染数据源） |

### 可选

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PREFLIGHT_BKE_BIN` | `/usr/local/bin/bke` | bke 可执行文件路径（两节点统一） |
| `KUBECONFIG` | `/root/.kube/config` | cluster-create 使用的 kubeconfig 路径（节点一上） |
| `PREFLIGHT_INIT_HOST_IP` | `TEST_NODE1_IP` | `bke preflight init --hostIP` 参数 |

### 可选覆盖（策略文件）

默认使用包内 `preflight.yaml`（`//go:embed`），落盘到远程临时目录后传 `--preflight-config`。设置下列环境变量时优先使用节点上的外部路径：

| 变量 | 关联用例 | 要求 |
|------|----------|------|
| `PREFLIGHT_INIT_POLICY_TEMPLATE` | INIT-001/002/004 等 | 完整合法策略，init cpu `minimum: 2` |
| `PREFLIGHT_CLUSTER_POLICY_TEMPLATE` | CLUSTER-001/002 等 | 完整合法策略，clusterCreate cpu `minimum: 4` |
| `PREFLIGHT_CLUSTER_POLICY_CUSTOM_EXEC` | CLUSTER-008 | 完整合法策略，含 `/opt/openfuyao/check-runtime` exec 采集器（publicFields runtime/healthy） |

> **Skip 语义**：仅当"使用包内 preflight.yaml 且报告 overall=INPUT_ERROR"（与当前 bke 版本不兼容）时对应用例自动 Skip 并附 issues code 提示；通过环境变量显式提供的外部策略被拒时照常走断言，不吞失败。

## 运行测试

```bash
cd e2e

# init 用例（节点一）
ginkgo -v --label-filter="preflight && pre-init" ./installation/preflight/

# cluster-create 用例（节点一）
ginkgo -v --label-filter="preflight && with-management-cluster" ./installation/preflight/

# 全部 preflight 用例（同一台机器上执行时请分开跑，用例运行位置由编排决定）
ginkgo -v --label-filter="preflight" --skip-label="with-management-cluster" ./installation/preflight/
```

> **串行运行**：用例涉及端口占用、集群 YAML 字段改写、/opt/openfuyao 下脚本部署等同一节点上的环境操作，请勿使用 `-p` 并行（用例之间无顺序依赖、可乱序执行，但不能并发争抢同一节点资源）。

## 文件结构

```
preflight/
├── README.md                         ← 本文件
├── preflight.yaml                    # 外部策略（go:embed，默认 --preflight-config）
├── preflight_suite_test.go           # 套件入口（TestPreflight + AfterSuite 统一清理）
├── preflight_init_test.go            # init 操作 5 条用例（节点一执行）
├── preflight_cluster_create_test.go  # cluster-create 操作 10 条用例（节点一执行）
└── preflight_helpers.go              # 公共 helper：JSON 报告结构解析、harness、
                                      #   策略落盘、nodes.yaml 渲染、YAML 字段替换、
                                      #   端口占用/释放、节点事实交叉核对等
```

## 报告契约速查

用例断言基于 `--output json --output-file <report>` 产生的 JSON 报告，关键契约：

| 字段 | 说明 |
|------|------|
| `overall` | `PASS` / `FAIL` / `INPUT_ERROR` / ... |
| `exitCode` | 与进程退出码一致（PASS=0、FAIL=3、INPUT_ERROR=4） |
| `summary` | `{total, pass, fail, warn, skip}`，`total == pass+fail+warn+skip` 且等于 `results` 长度 |
| `results[].instanceID` | `<规则ID>@<主体>`，节点类检查为 `@<节点名>`，来源可达检查为 `@<节点名>:<来源>` |
| `results[].expected/actual` | `{type, value}`；端口/拓扑类为 bool 契约（如端口空闲=true/占用=false、奇数=true） |
| `results[].evidence[].locator` | 部分检查项（如 CLUSTER-NET-POD-CIDR）的字段路径在 evidence.locator 中 |
| `issues[]` | INPUT_ERROR 时的结构化定位（`code` / `effect` / `field`） |

## 常见问题

### Q: 用例 Skip 了？
**A:** 检查 Skip 原因信息：
1. 包内 `preflight.yaml` 被当前版本 bke 拒绝（overall=INPUT_ERROR）→ 更新 yaml 的 `metadata.release` 与 bke 对齐，或通过 `PREFLIGHT_*` 环境变量提供与当前 bke 匹配的策略文件；
2. `PREFLIGHT_CLUSTER_POLICY_CUSTOM_EXEC` 等可选覆盖变量指向的文件不存在 → 修正路径或删除该变量改用包内策略。

### Q: 资源类检查项 FAIL 混入失败输出？
**A:** 确认用例是否带上了 `--preflight-config`（包内阈值为 6GiB/40GiB）。若仍 FAIL，说明节点实际可用资源低于策略阈值，属环境问题。

### Q: cluster.yaml 模板读取失败？
**A:** 用例优先从节点一读取 `/bke/cluster/1master1node-cluster.yaml`，缺失时回退引导节点（节点一）；确认对应节点的 bke 安装包含该模板文件。

### Q: CLUSTER-003 提示 imageRepo 结构无法改写？
**A:** 该用例通过替换 `imageRepo.domain` 子键注入不可达地址，仅支持映射结构（domain/ip/port）；bke 模板结构变更时需同步调整用例。

## 相关文件

- `e2e/.env.template` — 统一前置检查环境变量段
- `e2e/framework/executor/` — SSH 执行器框架
- `e2e/framework/env/` — `.env` 加载
- `e2e/installation/README.md` — installation 特性总览（含子目录规范）
