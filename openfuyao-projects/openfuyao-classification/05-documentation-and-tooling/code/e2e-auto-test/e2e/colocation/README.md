# Colocation 后端功能测试（Ginkgo）

本目录为在离线混部特性的后端自动化测试（Ginkgo/Go），聚焦 K8s 资源行为验证（部署、NUMA 亲和优化插件）。

> **注意**：`e2e/colocation/system-integration` 为独立的集成测试目录，不在本文档范围。

## 用例覆盖

后端当前覆盖 **39 个功能用例编号**（25 个部署测试 + 14 个 NUMA 插件测试）：

### 按功能分类

| 功能分类 | 覆盖用例编号 | 数量 | 状态 |
| --- | --- | --- | --- |
| 部署与基础校验 | 002~021, 029~033 | 25 | ✅ 稳定 |
| NUMA 亲和优化插件 | NUMA2506-001~027 | 14 | ✅ 稳定 |

### 按测试文件分类

| 文件 | 覆盖编号 | 说明 |
| --- | --- | --- |
| `colocation_suite_test.go` | 002, 003, 004 | Volcano 未安装时 HLS/LS/BE 应保持 Pending |
| `deployment_test.go` | 005~021, 029~033 | HLS/LS/BE 资源注入与高级特性 |
| `numa_affinity_test.go` | NUMA2506-001~027 | NUMA 亲和优化插件（各类工作负载验证） |

### NUMA 亲和优化插件测试详情

| 用例编号 | 工作负载类型 | 验证内容 |
| --- | --- | --- |
| NUMA2506-001 | - | NUMA 插件启用验证 |
| NUMA2506-002 | HLS Pod | 使用原生 CPU 管理 |
| NUMA2506-003 | LS Pod | NUMA 亲和应用 |
| NUMA2506-005 | 无 QoS Pod | 正常运行（跳过 NUMA） |
| NUMA2506-006 | HLS Deployment | 使用原生 CPU 管理 |
| NUMA2506-007 | LS Deployment | NUMA 亲和应用 |
| NUMA2506-009 | 无 QoS Deployment | 正常运行 |
| NUMA2506-011 | LS StatefulSet | NUMA 亲和应用 |
| NUMA2506-013 | 无 QoS StatefulSet | 正常运行 |
| NUMA2506-015 | LS DaemonSet | NUMA 亲和应用 |
| NUMA2506-019 | LS Job | NUMA 亲和应用 |
| NUMA2506-021 | 无 QoS Job | 正常运行 |
| NUMA2506-023 | LS CronJob | NUMA 亲和应用 |
| NUMA2506-027 | LS Deployment 多副本 | 多 Pod NUMA 亲和 |


## 自动安装/卸载流程

套件在 `BeforeSuite` 自动执行：

1. 清理历史测试 Pod（hls-pod, ls-pod, be-pod, stress-ng-1 等）
2. 卸载已存在的 `colocation-package` 与 `numa-affinity-package`
3. 安装 `colocation-package`
4. 启用所有节点混部
5. 验证 Volcano 未安装时 HLS/LS/BE 业务处于 Pending
6. 安装 `numa-affinity-package`（包含 Volcano）
7. 等待 Volcano webhook 服务就绪

在 `AfterSuite` 自动卸载：

- `colocation-package`
- `numa-affinity-package`

## 前置条件

1. 本机具备 `kubectl`、`helm`、`go`、`ginkgo` 命令
2. 运行环境可访问镜像仓库与 chart 源（`cr.openfuyao.cn`）
3. `KUBECONFIG` 环境变量已正确配置

## 环境变量

在 `e2e/.env` 中配置（可参考 `.env.template`）：

```bash
# Kubernetes 配置
KUBECONFIG=/path/to/kubeconfig

# SSH 连接（用于 kubelet 配置修改等高级测试，可选）
TEST_NODE2_IP=192.168.1.100
TEST_NODE2_PORT=22
TEST_NODE2_USER=root
TEST_NODE2_PASSWORD=your_password
```

## 运行方式

### 运行全部测试

```bash
cd e2e
go mod tidy
ginkgo -v ./colocation
```

### 按标签运行

```bash
# 部署与高级特性
ginkgo -v --label-filter="colocation-deployment" ./colocation

# NUMA 亲和优化插件
ginkgo -v --label-filter="colocation-numa-affinity" ./colocation

# 抢占（已跳过）
ginkgo -v --label-filter="colocation-pod-preemption" ./colocation

# 真实负载调度（已跳过）
ginkgo -v --label-filter="colocation-usage-threshold" ./colocation

# 驱逐（已跳过）
ginkgo -v --label-filter="colocation-eviction" ./colocation
```

### 使用 go test

```bash
cd e2e/colocation
go test -v -timeout 30m
```

## 关键实现说明

- `usage_threshold_test.go` 与 `eviction_test.go` 采用“低阈值触发 + 高阈值恢复”的双阶段校验，降低资源波动导致的误判。
- `deployment_test.go` 新增 `006/007/008`：通过 SSH 修改 kubelet 的 `cpuManagerPolicy` 与 `topologyManagerPolicy`，验证 HLS 在不同策略下调度行为。

## 已知限制

1. 抢占/驱逐/阈值类用例对集群资源敏感，若资源背景负载过高可能导致波动。
2. `stress-ng` 镜像需可拉取（`colinianking/stress-ng`）。
3. `006/007/008` 会重启 kubelet，建议在专用测试环境执行。
4. 抢占场景依赖“第二个低优先级 Pod 进入 Pending”的前置条件；若集群资源充足导致该条件不成立，用例会自动 Skip。
5. usage/eviction 场景依赖 stress Pod 启动；当镜像拉取或资源不足导致 stress Pod 不可用时，相关场景会自动 Skip。

