# BKE 安装部署 E2E 测试

## 概述

本测试套件用于验证 BKE（Bocloud Kubernetes Engine）集群安装部署功能的端到端测试，基于 Ginkgo v2 测试框架实现。测试涵盖了从基础集群创建、扩缩容到高级特性（如节点标签、Addon 亲和性部署）的完整生命周期验证。

> **重要**：BKE 已完成 BKECluster/BKENode CRD 拆分（简称"bc 拆分"）。集群拓扑中的节点信息从 `BKECluster` 资源中分离到独立的 `BKENode` 资源，创建集群时需要分别提供集群配置和节点配置两个文件。

## 测试范围

| 测试场景          | 描述                                 | Label                                                             |
|---------------|------------------------------------|-------------------------------------------------------------------|
| 创建 1Master 集群 | 创建单 Master 节点的集群并验证基础组件            | `1master`                                                         |
| 集群扩容          | 向已有集群添加新节点（创建 BKENode）并验证 Ready 状态 | `scale`                                                           |
| 集群缩容          | 从已有集群移除节点（删除 BKENode）并验证资源回收       | `scale`                                                           |
| 节点标签功能        | 验证 BKENode 上的自定义标签、多标签、非法字符过滤      | `node-labels`                                                     |
| Addon 亲和性     | 验证 Addon 根据节点标签进行定向调度              | `node-labels-addon-affinity`                                      |
| NUMA 亲和性组件    | 验证 NUMA 亲和性扩展组件（Volcano）的安装        | `numa-affinity`                                                   |
| 集群删除          | 验证集群销毁及相关资源的清理                     | `delete`                                                          |
| 引导节点初始化       | 验证 bke init 命令初始化引导节点              | `guide-install`                                                   |
| 离线引导节点初始化     | 验证使用本地离线包初始化引导节点 | `offline`, `pre-init`                                           |
| 管理集群安装        | 验证创建带 cluster-api 的管理集群            | `mgmt-1master`, `mgmt-3master`                                    |
| 业务集群单跳升级      | 管理集群部署的 FIT 业务集群从 vfit-pre 升级到 vfit-cur（1m1w/3m1w，按 Label 筛选） | `declarative-upgrade-fit`, `declarative-upgrade-upgrade`, `with-management-cluster`, `single-upgrade` |
| 管理集群单跳升级      | FIT 管理集群从 vfit-pre 升级到 vfit-cur（1m1w/3m1w，按 Label 筛选） | `declarative-upgrade-fit`, `declarative-upgrade-mgmt-upgrade`, `post-init`, `single-upgrade` |
| 业务集群两跳升级      | 管理集群部署的 FIT 业务集群从 vfit-pre 升级到 vfit-post（1m1w/3m1w，按 Label 筛选） | `declarative-upgrade-fit`, `declarative-upgrade-upgrade`, `with-management-cluster`, `multi-upgrade`       |
| 业务集群 Gate 关闭升级 | 全局 Gate 关闭 + Legacy/注解覆盖（DAG_EXEC_FIT_3/7，1m1w/3m1w 业务+管理，按 Label 筛选） | `declarative-upgrade-fit-gate-closed`, `gate-closed`, `with-management-cluster` |
| 管理集群两跳升级      | FIT 管理集群从 vfit-pre 升级到 vfit-post（1m1w/3m1w，按 Label 筛选） | `declarative-upgrade-fit`, `declarative-upgrade-mgmt-upgrade`, `post-init`, `multi-upgrade`  |
| 安装部署并发场景      | 管理集群上多业务集群并发创建/删除/扩缩容及竞态（CONC-01~11） | `concurrent`, `with-management-cluster`, `online`               |
| KubeOS 不可变 OS 适配 | 验证 KubeOS 混合 OS 集群安装、节点标签、只读目录、addon 调度与故障恢复 | `kubeos`, `post-init`                                           |
| 统一前置检查      | 验证 `bke preflight init`（节点一）与 `bke preflight cluster-create`（节点二）检查报告契约（overall/summary 四态/检查项期望实际值/issues 定位/敏感信息不泄漏） | `preflight`                                                       |

## 目录结构

```
installation/
├── installation_suite_test.go               # 测试套件入口及全局配置
├── installation_test.go                     # 网络断开与无效凭据场景（待第二批迁移）
├── installation_abnormal_test.go            # 异常处理完整流程测试用例
├── installation_abnormal_framework_test.go  # 异常处理测试框架与 CaseContext
├── management_cluster_test.go               # 管理集群安装测试
├── cluster_basic_test.go                    # 集群基础创建测试
├── cluster_test_helpers.go                  # 测试辅助函数（3M1W 集群等）
├── cluster_1m1w_helpers.go                  # 1M1W 集群辅助函数
├── diagnosis_handle_test.go                 # 诊断与扩容异常处理测试
├── health_mechanism_test.go                 # 健康检查机制测试
├── hosts_integrity_test.go                  # hosts 完整性测试
├── kubeos_functional_test.go                # KubeOS 不可变 OS 功能测试（IT-001/006/007）
├── kubeos_integration_test.go               # KubeOS 不可变 OS 集成测试（IT-002/003/004/005/008/DFR-001/DFD-002）
├── prescript_installation_test.go           # 前置脚本安装测试
├── postscript_installation_test.go          # 后置脚本安装测试
├── env_check_test.go                        # 环境检查测试
├── large_scale_cluster_test.go              # 大规模集群测试
├── bke_failure_artifact_capture_test.go     # BKE 失败诊断产物收集钩子
├── framework.go                             # SIG 描述函数封装
├── .env.bke.template                        # 环境变量配置模板
├── README.md                                # 本文档
├── bke-config/
│   ├── types.go                             # 类型定义（模板路径、节点信息等）
│   └── cluster_config.go                    # 配置加载函数（从环境变量读取）
├── addons/                                 # Addon 顺序、节点标签、HTTPS 与 OCI Chart
├── bke-commands/                           # 在线/离线引导初始化与补丁构建部署
├── bootstrap-optimization/                 # 引导节点部署优化
├── custom-installation/                    # 二进制配置、证书、HTTP Chart 与自定义/多镜像仓库
├── f5-loadbalancer/                        # haproxy 模拟 F5 的安装场景
├── declarative-upgrade/                     # 声明式升级（FIT 升级/异常，安装阶段在升级 BeforeAll 内覆盖）
│   ├── declarative_upgrade_test.go              # ClusterVersion/ReleaseImage/UpgradePath 契约
│   ├── declarative_upgrade_helpers.go           # 基础契约 helper
│   ├── upgradepath_test.go                      # UpgradePath OCI 拉取与生命周期
│   ├── declarative_upgrade_install_helpers.go     # 安装与升级公共函数
│   ├── declarative_upgrade_fit_upgrade_test.go    # FIT 业务集群升级（单跳/两跳，Label: declarative-upgrade-upgrade）
│   ├── declarative_upgrade_fit_mgmt_upgrade_test.go # FIT 管理集群升级（单跳/两跳，Label: declarative-upgrade-mgmt-upgrade）
│   ├── declarative_upgrade_fit_gate_closed_test.go  # FIT Gate 关闭 Legacy 升级（FIT_3/7，Label: declarative-upgrade-fit-gate-closed）
│   ├── declarative_upgrade_fit_gate_helpers.go      # Gate 关闭/Legacy 断言 helper
│   ├── declarative_upgrade_fit_abnormal_test.go   # FIT 升级异常场景
│   ├── declarative_upgrade_fit_helpers.go         # FIT 升级 helper
│   ├── declarative_upgrade_abnormal_test.go       # 升级异常场景测试用例
│   ├── declarative_upgrade_suite_test.go          # 测试套件入口及全局配置
│   ├── bke_failure_artifact_capture_test.go       # BKE 失败诊断产物收集钩子
│   └── framework.go                               # SIG 描述函数封装
├── concurrent/                              # 安装部署并发场景
│   ├── concurrent_installation_helpers.go         # 公共函数
│   ├── concurrent_installation_test.go            # 集群创建与删除测试用例（CONC-01~07）
│   ├── concurrent_scale_test.go                   # 扩缩容与交叉场景测试用例（CONC-08~11）
│   ├── concurrent_suite_test.go                   # 测试套件入口及全局配置
│   ├── bke_failure_artifact_capture_test.go       # BKE 失败诊断产物收集钩子
│   └── framework.go                               # SIG 描述函数封装
├── abnormal-retry/                          # 异常重试处理（FIT_1~9）
│   ├── abnormal_retry_helpers.go                  # 公共函数（BKENode 状态查询、故障注入、retry 注解、轮询等待）
│   ├── abnormal_retry_test.go                     # RetryCount 递增/重置与集群 Failed 终态测试用例（FIT_1~5）
│   ├── abnormal_retry_annotation_test.go          # Retry 注解清除 NeedSkip/ManualInterventionRequired 测试用例（FIT_6~9）
│   ├── abnormal_retry_suite_test.go               # 测试套件入口及全局配置
│   ├── bke_failure_artifact_capture_test.go       # BKE 失败诊断产物收集钩子
│   └── framework.go                               # SIG 描述函数封装
├── f5-system-integration/                   # F5 系统集成
│   ├── f5_system_integration_test.go              # F5 负载均衡系统集成测试用例
│   ├── f5_system_integration_suite_test.go        # 测试套件入口及全局配置
│   └── framework.go                               # SIG 描述函数封装
├── performance-test/                        # 性能测试
│   ├── installation_performance_test.go           # 安装部署性能测试用例
│   ├── performance_test_suite_test.go             # 测试套件入口及全局配置
│   └── constants.go                               # 性能测试常量定义
├── system-integration/                      # 跨组件系统集成
│   ├── system_integration_test.go                 # 系统集成测试用例
│   ├── system_integration_suite_test.go           # 测试套件入口及全局配置
│   ├── system_integration_service_test.go         # 服务集成测试用例
│   ├── system_integration_etcd_apiserver_test.go  # etcd/APIServer 集成测试用例
│   ├── system_integration_dns_test.go             # DNS 集成测试用例
│   ├── installation_reboot_test.go                # 重启恢复测试用例
│   ├── system_integration_utils.go                # 系统集成辅助函数
│   ├── system_integration_utils/
│   │   └── helper_function.go                     # 系统集成 helper 函数
│   ├── constants.go                               # 系统集成常量定义
│   └── framework.go                               # SIG 描述函数封装
├── utils/
    ├── yaml_helper.go                       # BKECluster + BKENode 配置生成工具
    ├── cluster_checker.go                   # 集群状态检查工具
    ├── cluster_manager.go                   # 集群生命周期管理工具
    ├── kubeos_checker.go                    # KubeOS 适配检查工具
    ├── resource_manager.go                  # 资源管理工具（Namespace/ConfigMap）
    ├── node_executor.go                     # 远程节点命令执行工具
    ├── script_manager.go                    # 脚本管理工具
    ├── loadbalancer_helper.go               # 负载均衡器辅助工具
    ├── declarative_upgrade_helper.go        # 声明式升级辅助工具
    ├── bke_failure_artifact_capture.go        # BKE 失败诊断产物收集工具（BC/BN CRD + controller-manager 日志）
    ├── cert_verifier.go                     # 证书验证工具
    └── cert_generator.go                    # 证书生成工具
└── preflight/                                # 统一前置检查
    ├── preflight_suite_test.go               # 测试套件入口及全局配置
    ├── preflight_init_test.go                # preflight init 测试用例（节点一执行）
    ├── preflight_cluster_create_test.go      # preflight cluster-create 测试用例（节点二执行）
    ├── preflight_helpers.go                  # 公共函数（报告解析/harness/策略 fixture/nodes.yaml 渲染）
    └── README.md                             # 运行说明、环境要求与环境变量文档
```

> **新特性用例目录规范**：对于新特性的 e2e 用例，应在 `installation/` 下创建独立的子目录（如 `abnormal-retry/`、`concurrent/`、`declarative-upgrade/`），不要平铺为单个文件。每个子目录包含独立的 Suite 入口、测试用例文件、helper 文件和 `framework.go`，保持与现有模块结构一致。

## 前置条件

1. **引导集群已部署**：需要一个已运行的引导集群（管理集群）。
2. **BKE 模板文件就位**：引导集群上需要有 BKECluster 模板，如 `/bke/cluster/1master-cluster.yaml`。BKENode 配置由测试代码动态生成，无需模板。
3. **yq 工具已安装**：引导节点必须安装 `yq` (v4+)，用于动态修改集群配置。
4. **jq工具已安装**：补丁包构建与镜像校验相关用例需要`jq` 解析 JSON 输出。
5. **skopeo工具已安装**：镜像 digest 一致性校验用例需要`skopeo inspect` 访问源镜像、OCI layout 和目标 registry。
6. **测试节点已准备**：用于创建业务集群的节点需满足 BKE 部署要求，且 SSH 互通。
7. **SSH 访问权限**：测试执行机需具备通过 SSH 访问引导节点的权限。
8. **KubeOS worker VM（仅 KubeOS 用例）**：运行 `kubeos` 标签用例时，需准备 1 台普通 openEuler master 和 1 台 KubeOS worker VM（镜像内置 bkeagent/kubelet/containerd），并在 `.env` 中配置对应的 `TEST_NODE` 信息。可选配置 `KUBEOS_WORKER_*` 环境变量单独指定 KubeOS worker 节点。

## 环境配置

1. 复制环境变量模板：
   ```bash
   cp .env.template .env
   ```
2. 编辑 `.env` 填写实际环境信息（引导节点信息、测试节点 IP/密码等）。

## 运行测试

### 安装依赖
```bash
cd e2e/installation
go mod tidy
```

### 使用 Ginkgo CLI 运行（推荐）
```bash
# 安装 ginkgo CLI
go install github.com/onsi/ginkgo/v2/ginkgo@latest

# 运行全部测试
ginkgo -v ./...

# 运行特定特性的测试
ginkgo -v --label-filter="node-labels" ./...
ginkgo -v --label-filter="numa-affinity" ./...
ginkgo -v --label-filter="install-upgrade-delete-chart-from-oci-repo" ./...
ginkgo -v --label-filter="install-chart-from-https-repo" ./...
ginkgo -v --label-filter="install-chart-from-http-repo" ./...

# 运行引导节点初始化测试
ginkgo -v --label-filter="guide-install" ./...

# 运行离线引导节点初始化测试
ginkgo -v --label-filter="offline && pre-init" ./bke-commands

# 运行管理集群测试
ginkgo -v --label-filter="mgmt-1master || mgmt-3master" ./...

# 运行安装部署并发场景（需前置 mgmt-1master Job，整包建议 8h 超时）
ginkgo -v --timeout=8h --label-filter="concurrent && with-management-cluster && online" ./concurrent

# 运行 KubeOS 不可变 OS 适配测试（需 KubeOS worker VM 环境）
ginkgo -v --label-filter="kubeos" ./...
ginkgo -v --label-filter="kubeos && P0" ./...          # 仅 P0 用例
ginkgo -v --label-filter="kubeos && dfr" ./...         # 仅 DFR 故障恢复用例

# 仅扩缩容与交叉场景（concurrent_scale_test.go，CONC-08~11）
ginkgo -v --timeout=8h --label-filter="concurrent && (scale || create-scale-race)" ./concurrent

# 运行单条并发用例示例
ginkgo -v --timeout=8h --label-filter="concurrent && multi-cluster-create && 1master && P0" ./concurrent
ginkgo -v --timeout=8h --label-filter="concurrent && scale-race && P1" ./concurrent

# 增加超时时间运行全量测试（防止大规模集群创建超时）
ginkgo -v --timeout=3h ./...
```

## 用例标签说明

安装部署自动化用例会根据以下标签进行筛选用例运行：其中**场景标签**和**运行标签**和**节点数标签**是必填项，筛选时需要同时明确这两类标签。

### 场景标签

| 标签 | 含义 |
|------|------|
| `offline` | 离线安装场景 |
| `custom-repo` | 自定义仓库场景 |
| `online` | 在线安装场景 |

### 运行阶段标签

| 标签 | 含义 |
|------|------|
| `pre-init` | 引导节点初始化前 |
| `post-init` | 引导节点已就绪 |
| `with-management-cluster` | 管理集群已就绪 |
| `with-workload-cluster` | 业务集群已就绪 |

### 节点数标签（required_nodes）

格式为 `required_nodes:<n>`（n = 0~4），表示用例运行所需的**测试节点数**，用于 CI 按环境规模分组调度，例如：

```bash
# 只跑需要 2 个测试节点的 online 用例（引导节点 + 2 测试节点的环境）
ginkgo --label-filter="online && required_nodes:2" ./installation
```

**取值口径**（不含引导节点；管理集群与引导节点共部署时不计入）：

| 值 | 典型场景 |
|----|----------|
| `0` | 仅引导节点操作（bke init/envCheck/补丁包制备等） |
| `1` | 创建 1Master 集群 |
| `2` | 创建 1Master1Worker 集群、1M 集群 + 扩容目标节点 |
| `3` | 创建 3Master 集群、复用预建 1M2W 基线的校验用例 |
| `4` | 创建 3Master1Worker 集群、需要独立 LB 节点的场景 |

**设置时必须同时考虑以下两部分，取最大值**：

1. **套件级初始化的节点占用**：`BeforeSuite`/`BeforeAll` 中创建的集群或 SSH 目标节点对包内/容器内所有用例生效，即使用例自身只复用已有集群，也要计入 setup 的占用
   > 例：`performance-optimize` 的 `BeforeSuite` 会在 TEST_NODE2 上建管理集群、TEST_NODE3/4 上建 1M1W 目标集群（3 节点），其 calico FIT 用例另需 TEST_NODE5，故标 `required_nodes:4`
2. **用例自身的峰值占用**：动态创建的集群规模、扩容目标节点、SSH 操作的目标节点等

若代码中存在环境门禁（如 `len(nodes) < N` 时 `Skip`，或断言 `TEST_NODE<n>_IP` 非空），环境配置不满足时用例会跳过/失败，此时**按门禁要求取值**，即使实际占用更少（例：NP-DEP-003 实际只用 1 master + 1 worker，但门禁要求 4 个测试节点，标 `required_nodes:4`）。

套件内所有用例节点数一致时，可在 `SIGDescribe`/`DescribeTable` 的容器 `Label` 上统一声明，由用例继承（例：`compliance-operator` 套件级 `required_nodes:4`）。

### 其他标签

| 标签 | 含义 |
|------|------|
| `skip-temporarily` | 未就绪用例，临时跳过 |
| `kubeos` | KubeOS 不可变 OS 适配场景 |
| `dfr` | KubeOS 故障恢复（DFR）用例 |
| `concurrent` | 安装部署并发场景（`concurrent/` 子目录） |
| `multi-cluster-create` | 多业务集群并发创建 |
| `multi-cluster-delete` | 多业务集群并发删除 |
| `create-delete-race` | 创建与删除竞态 |
| `cleanup-reuse` | 删除后节点复用创建 |
| `scale-out` / `scale-in` / `scale-race` | 并发扩容 / 缩容 / 并发创建 + 跨集群并发缩容+扩容 |
| `create-scale-race` | 创建与扩容竞态 |

扩缩容用例（CONC-08~11）终态通过 `eventuallyClusterScaleSettled` 联合校验（非 `Scaling*` 中间态 + Healthy/Ready + BKENode IP）；清理使用 `cleanupScaleWorkloadClusters`（删除前 `waitClusterQuiescent`，删除等待 30min）。

> 并发用例详细设计见 `doc/concurrent-test-design.md`。

## 核心功能说明

### BKECluster / BKENode 资源拆分

集群创建现在涉及两类 CRD 资源：

- **BKECluster**：集群级别配置（网络、版本、Addon 等），由引导节点上的模板生成。
- **BKENode**：每个节点对应一个 BKENode 资源，包含 hostname、IP、密码、角色等信息，通过 label `cluster.x-k8s.io/cluster-name` 关联到所属集群。BKENode 配置由测试代码根据节点列表动态生成。

创建命令变更：
```bash
# 旧命令（已废弃）
bke cluster create -f <cluster.yaml>

# 新命令
bke cluster create -f <cluster.yaml> -n <nodes.yaml>
```

扩缩容变更：
- **扩容**：直接创建新的 BKENode 资源（`kubectl apply`），无需修改 BKECluster。
- **缩容**：根据节点 IP 查找对应的 BKENode 资源后删除（`kubectl delete`）。

### 节点标签与亲和性验证
测试套件会自动在 `BKENode` 资源中注入自定义标签，并验证：
- 标签是否正确同步到业务集群的 `Node` 对象。
- Addon（如 Redis）是否根据 `nodeSelector` 调度到指定标签的节点。
- 非法标签（含空格等）是否被正确过滤。

### NUMA 亲和性组件 (Volcano)
验证 `numa-affinity-package` Addon 的安装：
- 检查 `numa-affinity-package` 命名空间下的安装器状态。
- 检查 `volcano-system` 命名空间下的核心组件（Scheduler, Controller, Admission）是否全部就绪。

### KubeOS 不可变 OS 适配

验证普通 openEuler master + KubeOS worker VM 的混合 OS 集群安装部署能力。KubeOS worker 使用只读 rootfs，二进制（kubelet/containerd/bkeagent）在镜像制作阶段内置，安装器通过 `--immutable-os=kubeos` 参数进入 KubeOS 适配分支，跳过二进制推送只保留配置渲染。

**前置条件**：
- 1 台普通 openEuler master 节点
- 1 台 KubeOS worker VM（镜像内置 bkeagent/kubelet/containerd/os-agent）
- 引导节点已就绪，`bkeadm` 已支持 `--immutable-os` 参数

**用例说明**：

| 用例编号 | 类型 | 描述 | 验证点 |
|---------|------|------|--------|
| KubeOS-IT-001 | 功能 | bkeadm 注入 immutable-os 标志 | BKECluster annotations 含 `bke.bocloud.com/immutable-os=kubeos` |
| KubeOS-IT-002 | 集成 | 混合 OS 集群安装——worker 加入成功 | master/worker 均 Ready，bkeagent/kubelet/containerd 服务 active |
| KubeOS-IT-003 | 集成 | 只读目录未被写入 | `/persist`、`/etc`、`/var`、`/opt/cni` 可写；二进制镜像内置；安装日志无 read-only 错误 |
| KubeOS-IT-004 | 集成 | worker 节点标签自动打标 | worker 带 `node.openfuyao.io/os=kubeos`、`upgrade.openeuler.org/node-selector=default-worker-pool`；master 不带 |
| KubeOS-IT-005 | 集成 | 集群可用性验证——Pod 调度与 CNI | Pod 调度到 KubeOS worker 并 Running；Calico CNI 插件存在；`/opt/cni/bin` 可写 |
| KubeOS-IT-006 | 功能 | webhook 校验——同 IP 角色冲突 | 同一 IP 同时配 master 和 worker 角色时集群创建被拒绝 |
| KubeOS-IT-007 | 回归 | 普通模式不受影响 | 不传 `--immutable-os` 时集群正常安装，worker 不带 KubeOS 标签 |
| KubeOS-IT-008 | 集成 | KubeOS addon 安装与调度 | os-operator Running；os-proxy Pod 数 == KubeOS worker 数且只调度到 KubeOS worker；不调度到 master |
| KubeOS-DFR-001 | DFR | os-proxy Pod 故障后自动恢复 | 删除 os-proxy Pod 后 DaemonSet 自动重建，numberReady 恢复到 desiredNumberScheduled |
| KubeOS-DFD-002 | DFD | os-proxy 访问 os-agent 失败可定位 | 停止 os-agent 后 os-proxy 输出诊断信息；恢复后 os-proxy 自动恢复 |

**测试组织**：
- 功能测试（IT-001/006/007）各自独立创建集群，因为验证点不同（传参/拒绝/普通模式）。
- 集成测试（IT-002~005、008、DFR-001、DFD-002）使用 `Ordered` + `BeforeAll`/`AfterAll` 共享一个 KubeOS 混合集群，避免重复安装。DFR-001 和 DFD-002 复用 IT-008 已安装的 KubeOS addon。

**关键检查工具**（`utils/kubeos_checker.go`）：
- `GetImmutableOSAnnotation`：读取 BKECluster 的 immutable-os annotation
- `CheckNodeHasKubeOSLabels`：检查节点 KubeOS 标签
- `CheckDirWritableOnNode` / `CheckBinaryOnNode` / `CheckServiceActiveOnNode`：通过 SSH 检查 KubeOS worker 目录可写性、二进制存在性、服务状态
- `CheckOSOperatorReady` / `GetOSProxyDaemonSetStatus`：检查 KubeOS addon 组件状态
- `CheckOSProxyScheduledOnKubeOSWorkers` / `CheckOSProxyNotOnMaster`：检查 os-proxy 调度落点

### 升级适配 Helm FIT 用例

升级适配 Helm 用例集中在 `declarative-upgrade/` 目录，覆盖管理集群升级、业务集群升级和业务集群负向升级。用例会在触发升级前自动准备 `fit-helm-e2e` chart，并通过 `cvo.openfuyao.cn/helm-component=true` 注解开启 Helm 组件升级路径。

**Chart 准备与仓库配置**：
- 引导节点上创建或复用 `/opt/fit-helm-e2e`，打包并上传 `0.1.1`、`0.1.2`、`0.1.3` 版本到 `http://<GUIDE_NODE>:38080`，上传前会确保 `/bke/mount/charts` 可写。
- 管理集群和业务集群正向升级会创建 `/opt/chartrepo-tls`，生成以 `GUIDE_NODE` IP 为 CN/SAN 的自签证书，并启动 `chartmuseum-tls-proxy` nginx 容器，把 `38443` 反代到 `38080`。
- 用例会创建或更新 `chartrepo-tls` Secret，并把 BKECluster 的 `spec.clusterConfig.cluster.chartRepo` patch 到 `ip=<GUIDE_NODE>`、`port=38443`、`tlsSecretRef={name: chartrepo-tls, caKey: ca.crt, certKey: cert.crt, keyKey: key.key}`。
- 清理阶段会删除 TLS Secret、回滚 chartRepo 到 `38080`、删除 `/opt/chartrepo-tls`，并移除 `chartmuseum-tls-proxy` 容器。

**正向升级验证**：
- 管理集群单跳升级 `vfit-pre -> vfit-cur`：在第一个管理 master 上校验 `fit-helm-e2e` Pod、`helm status`、`helm history`、`helm get manifest` 和 Deployment YAML；同时在重启 `cluster-system/bke-controller-manager` 前后校验 TLS 仓库信息日志与 release 状态。
- 管理集群两跳升级 `vfit-pre -> vfit-post`：第一次触发目标版本 `vfit-post`，调谐器按升级路径完成两跳；第一跳完成后只在升级后的自管集群里重新创建 Secret 并 patch `tlsSecretRef`，不重复操作引导节点 TLS 代理；最终校验 pod label 中包含 `0.1.2`。
- 业务集群正向升级：触发升级和 controller-manager 日志/重启动作在引导节点执行，Helm release、manifest、Deployment YAML、pod label 校验在第一个业务 master 上执行。目前业务集群升级用例只保留 Helm 相关验证，YAML/Inline 组件断言暂不执行。

**负向升级验证**：
- `vfit-cur-neg-helm-chart`：模拟 `fit-helm-e2e` chart 版本 `9.9.9` 不存在，在引导节点 `bke-controller-manager` 日志中校验 `chart "fit-helm-e2e" version "9.9.9" not found in`。
- `vfit-cur-neg-helm-hc`：模拟 Helm healthCheck 失败，在第一个业务 master 上执行 `kubectl -n fit-helm-e2e get pod --show-labels`，校验包含 `helm-healthcheck-fail`。
- `version-not-exist`：模拟目标版本不存在，在引导节点 `bke-controller-manager` 日志中校验 `version not found in upgrade`。

### 离线安装测试

离线安装测试用于验证在断网环境下从本地离线包进行安装的场景：

**离线引导节点初始化测试（`bke-commands/`，标签 `offline`、`pre-init`）**：
| 用例 | 描述 | 预期结果 |
|------|------|----------|
| 指定正确的 oFVersion | 使用 `--oFVersion v25.10` | 初始化成功 |
| 指定不存在的 oFVersion | 使用 `--oFVersion v99.99` | 初始化失败，报错 `failed to start cluster API` |
| 不指定 oFVersion | 使用默认的 latest 版本 | 初始化成功 |

离线测试使用 `bke init` 命令（与在线一致），镜像从引导节点预装的离线包加载，而非从远程仓库拉取。离线包会在测试环境中提前制备好。

## 测试流程

### 1Master 集群创建测试流程

```
1. 从引导节点读取 BKECluster 模板文件 (/bke/cluster/1master-cluster.yaml)
2. 复制模板到临时文件
3. 使用 yq 修改集群名称（并移除模板中残留的节点信息）
4. 根据节点列表动态生成 BKENode 配置文件（多文档 YAML）
5. 执行 bke cluster create -f <cluster.yaml> -n <nodes.yaml>
6. 轮询检查 BKECluster 资源状态直到 Ready
7. 通过 kubectl 验证业务集群状态：
   - 节点数量和状态
   - 系统组件 Pod 状态
   - CoreDNS 运行状态
   - Calico 网络状态
8. 测试完成后删除集群并清理临时文件（集群配置 + 节点配置）
```

## 工具说明

### YAMLModifier

用于在引导节点上生成 BKECluster 和 BKENode 配置文件：

```go
modifier := utils.NewYAMLModifier(sshExecutor)

// 生成集群配置 + 节点配置（返回两个文件路径）
clusterConfigPath, nodeConfigPath, err := modifier.GenerateAndUpload(
    "/bke/cluster/1master-cluster.yaml",  // BKECluster 模板路径
    "my-cluster",                          // 集群名称
    nodes,                                 // 节点列表
)

// 为扩容生成单个 BKENode YAML
nodePath, err := modifier.GenerateSingleBKENodeYAML("my-cluster", newNode)
```

### ClusterChecker

用于检查业务集群状态：

```go
checker := utils.NewClusterChecker(sshExecutor, "my-cluster")

// 获取节点数量
nodeCount, _ := checker.GetNodeCount()

// 检查系统 Pod 状态
allRunning, notRunning, _ := checker.CheckSystemPods()

// 获取资源状态概览
status, _ := checker.GetClusterResourceStatus()
```

### ClusterManager

封装集群生命周期管理操作：

```go
manager := utils.NewClusterManager(sshExecutor, dynamicClient)

// 创建集群（需要同时传入集群配置和节点配置路径）
manager.CreateCluster(clusterConfigPath, nodeConfigPath)

// 等待集群就绪
manager.WaitForClusterReady(ctx, "my-cluster", 30*time.Minute)

// 扩容节点（创建 BKENode 资源）
manager.ScaleOutNode("my-cluster", newNode)

// 缩容节点（根据 IP 查找并删除 BKENode 资源）
manager.ScaleInNode("my-cluster", nodeIP)

// 删除集群
manager.DeleteCluster("my-cluster", true)
```

## 故障排查

### Kubeconfig 获取失败
**现象**：报错 `error loading config file ... no such file or directory`。
**原因**：测试脚本尝试从引导节点获取业务集群的 kubeconfig。
**解决**：脚本已支持从 Secret (`bke-<name>-kubeconfig`) 和文件路径 (`/root/.bke/...`) 多重获取。请确保引导集群的 `kubectl` 权限正常。

### 测试套件超时 (Suite Timeout)
**现象**：`[TIMEDOUT] A suite timeout occurred`。
**原因**：全量测试耗时较长（每个集群创建/删除约 15-20 分钟），超过了默认的 1 小时限制。
**解决**：运行命令时添加 `--timeout=3h` 参数。

### 缩容节点失败
**现象**：节点数量未减少，测试超时。
**原因**：可能是根据 IP 未找到对应的 BKENode 资源，或 BKE 控制器未触发节点重置。
**解决**：检查引导集群上的 BKENode 资源是否存在：`kubectl get bkenode -A`，确认节点 IP 与 BKENode spec 中的 IP 一致。

### SSH 连接失败

检查：
- 引导节点 IP 和端口是否正确
- SSH 用户名密码是否正确
- 防火墙是否开放 SSH 端口

### yq 命令不存在

在引导节点安装 yq：

```bash
# 使用 snap（Ubuntu）
sudo snap install yq

# 或下载二进制
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq
chmod +x /usr/local/bin/yq
```

### 集群创建超时

- 检查引导节点日志：`journalctl -u bke-controller`
- 检查 BKECluster 资源状态：`kubectl get bkecluster -A -o yaml`
- 检查 BKENode 资源状态：`kubectl get bkenode -A -o yaml`
- 检查测试节点网络连通性

## 扩展开发

### 添加新 Addon 测试
在 `installation_test.go` 的 `Describe("Addon 扩展组件安装", ...)` 块中添加新的 `It` 用例，并使用 `checker.CheckNamespacePodsReady` 进行状态验证。

### 添加新的测试用例

1. 在 `installation_test.go` 中添加新的 `Describe` 块
2. 使用 `Label()` 标记测试类别
3. 复用 `ClusterManager` 和 `ClusterChecker` 工具

示例：

```go
Describe("创建 3Master 高可用集群", Label("3master", "ha"), func() {
    It("应该成功创建 3 Master 集群", func() {
        // 生成配置
        clusterConfigPath, nodeConfigPath, err := modifier.GenerateAndUpload(
            templatePaths.ThreeMasterClusterTemplate,
            "ha-cluster",
            nodes,
        )
        Expect(err).NotTo(HaveOccurred())

        // 创建集群
        err = manager.CreateCluster(clusterConfigPath, nodeConfigPath)
        Expect(err).NotTo(HaveOccurred())

        // 等待就绪并验证
        manager.WaitForClusterReady(ctx, "ha-cluster", 30*time.Minute)
    })
})
```
