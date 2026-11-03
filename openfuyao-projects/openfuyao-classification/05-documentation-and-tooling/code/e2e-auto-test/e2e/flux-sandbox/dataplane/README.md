# FluxSandbox 数据面 E2E 测试（Ginkgo）

本目录为 flux-sandbox 数据面（节点 Agent 与 SandboxWatcher）的后端自动化测试（Ginkgo/Go），覆盖 Agent HTTP API、Sandbox 生命周期、Reconcile、并发与孤儿清理，共 36 个用例（`DP-E2E-001` ~ `DP-E2E-036`，用例定义详见 `dataplane-test.md`）。

> 说明：测试代码与 API 类型（`api/v1alpha1`、`agentapi`）同步自 flux-sandbox 产品仓库 `test/e2e/`，按本仓库模块规范做了适配。

## 目录结构

```
e2e/flux-sandbox/
├── framework/      # 可复用测试基础设施（普通 Go 库，带单测）
├── dataplane/      # Ginkgo 套件
├── api/v1alpha1/   # Sandbox 等 CRD 类型
└── agentapi/       # Agent HTTP API 请求/响应类型
```

## 用例覆盖

| 文件 | 覆盖编号 | 说明 |
| --- | --- | --- |
| `agent_api_test.go` | DP-E2E-001~004, 006~013 | Agent HTTP API（健康检查、创建/删除/状态/日志） |
| `agent_special_test.go` | DP-E2E-005 | 小容量 agent（环境门控） |
| `sandbox_lifecycle_test.go` | DP-E2E-014~025 | Sandbox 生命周期 |
| `reconcile_test.go` | DP-E2E-026~030 | 重启收编与并发 |
| `orphan_cleanup_test.go` | DP-E2E-031~036 | SandboxWatcher 孤儿清理 |

## 套件行为

- `BeforeSuite` 自动部署 Namespace / ServiceAccount / RBAC / Agent Pod / Watcher DaemonSet（create-only 幂等），等待 agent 就绪并建立 port-forward，清理残留沙箱。
- `AfterSuite` 释放 port-forward；`E2E_CLEANUP` 非空时删除部署组件。
- 部署为 create-only：修改部署参数（如 grace period）后需手动删除对应 Agent Pod / Watcher DaemonSet 再运行，让套件重建。

## 前置条件

- k8s 集群就绪；测试机具备 `kubectl`、`go`、`ginkgo` 命令
- 测试机可访问 registry（推送/节点拉取镜像）、可访问 apiserver、可 SSH 到节点
- 组件部署（Agent Pod / Watcher DaemonSet / 小 agent pod）由套件在 `BeforeSuite` 自动完成（create-only 幂等），无需手工部署；修改部署参数（如 grace period）后需手动删除对应组件再运行，让套件重建

## 完整执行步骤（标准 Linux + k8s，一次跑完全部用例）

### 1. 构建并推送镜像

在 flux-sandbox 产品仓库执行（任意有 docker 且可访问 registry 的机器）：

```bash
docker build -f build/Dockerfile.flux-sandbox-agent -t <REGISTRY>/flux-sandbox/agent:v1 .
docker build -f build/Dockerfile.sandbox-watcher -t <REGISTRY>/flux-sandbox/sandbox-watcher:v1 .
docker push <REGISTRY>/flux-sandbox/agent:v1
docker push <REGISTRY>/flux-sandbox/sandbox-watcher:v1
```

同时确保节点可拉取 `opensandbox/execd:v1.0.14` 与沙箱镜像 `docker.io/library/busybox:latest`，内网环境可预先导入节点。

### 2. 安装 CRD（flux-sandbox 产品仓库内）

```bash
kubectl apply -k config/crd
```

### 3. 打通测试机到节点的 SSH 免密

套件通过 SSH 在 Agent 所在节点执行 `ctr` 等命令（`BatchMode=yes`，仅支持免密登录）：

```bash
ssh-copy-id root@<节点InternalIP>
# 节点 InternalIP 不可直连时：对跳板机执行 ssh-copy-id，并导出 E2E_NODE_SSH_HOST=<跳板机IP>
```

### 4. 配置环境变量并一次执行全部用例

```bash
cd e2e-auto-test/e2e
export KUBECONFIG=/path/to/kubeconfig      # 套件内的 kubectl port-forward 子进程依赖它
export E2E_KUBECONFIG=$KUBECONFIG
export E2E_AGENT_IMAGE=<REGISTRY>/flux-sandbox/agent:v1
export E2E_WATCHER_IMAGE=<REGISTRY>/flux-sandbox/sandbox-watcher:v1
export E2E_SMALL_CAPACITY_AGENT=true       # 套件自建 capacity=2 小 agent（DP-E2E-005/029）
export E2E_NODE_SSH_READY=true             # 启用节点级用例（DP-E2E-025/036）
export E2E_EXECD_REACHABLE=true            # 启用 execd 可达检查（DP-E2E-021）

ginkgo -v -timeout 30m ./flux-sandbox/dataplane
```

预期结果：43 个 spec 全部 Passed（0 Skipped，含 DescribeTable 展开），用时约 7 分钟。

### 5. 失败用例单独复跑（可选）

```bash
ginkgo -v -timeout 15m -focus "DP-E2E-034" ./flux-sandbox/dataplane
```

## 常用运行方式

```bash
cd e2e

# 按用例编号 / 特性组聚焦
ginkgo -v -timeout 15m -focus "DP-E2E-014" ./flux-sandbox/dataplane
ginkgo -v -timeout 30m --label-filter="watcher" ./flux-sandbox/dataplane

# 框架单测（无需集群）
go test ./flux-sandbox/framework/
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `E2E_KUBECONFIG` | `~/.kube/config` | kubeconfig 路径 |
| `E2E_NAMESPACE` | `flux-sandbox` | 测试命名空间 |
| `E2E_AGENT_IMAGE` | `flux-sandbox/agent:latest` | Agent 镜像 |
| `E2E_WATCHER_IMAGE` | `flux-sandbox/sandbox-watcher:latest` | Watcher 镜像 |
| `E2E_EXECD_IMAGE` | `opensandbox/execd:v1.0.14` | execd 注入镜像 |
| `E2E_SKIP_DEPLOY` | (空) | 非空则跳过组件部署 |
| `E2E_CLEANUP` | (空) | 非空则测试后删除部署组件 |
| `E2E_NODE_ACCESS` | `ssh` | 节点访问方式：`ssh` / `docker` / `local` |
| `E2E_NODE_SSH_HOST` | Agent 节点 InternalIP | ssh 模式目标主机 |
| `E2E_NODE_SSH_USER` | `root` | ssh 模式用户 |
| `E2E_AGENT_AUTH_TOKEN` | (空) | Agent Bearer 认证 token（可选） |
| `E2E_ORPHAN_GRACE_PERIOD` | `30s` | 孤儿清理宽限期（生产 3m，e2e 可调小加速） |
| `E2E_SCAN_INTERVAL` | `30s` | watcher 扫描间隔（加速测试） |
| `E2E_SMALL_CAPACITY_AGENT_URL` | (空) | 外部部署的 capacity=2 agent 地址（与 `E2E_SMALL_CAPACITY_AGENT` 二选一） |
| `E2E_SMALL_CAPACITY_AGENT` | (空) | 非空则套件自建 capacity=2 小 agent pod（AfterSuite 自动清理），无需外部部署 |
| `E2E_NODE_SSH_READY` | (空) | 非空启用节点级用例（DP-E2E-025 / 036） |
| `E2E_EXECD_REACHABLE` | (空) | 非空启用 execd 可达检查（DP-E2E-021） |

## 关键实现说明

- Agent 部署为固定名裸 Pod（非 DaemonSet）：重启用例删除后由框架按原名重建（仅 UID 变化），对齐生产调度器形态下 agent pod 名稳定的行为——agent 的 reconcile 按 `agent-name` 标签收编沙箱，名字稳定才能覆盖原地重启路径（DP-E2E-026/027）；watcher 对同名不同 UID 的 pod 视为旧 agent 已消亡，孤儿清理（DP-E2E-032）不受影响。
- `framework/deploy.go` 在 agent Pod 上挂载 emptyDir 到 /tmp：agent 镜像基于 scratch 无 /tmp，而 containerd client 创建容器时需要在进程内创建临时目录。
- DP-E2E-005 / 029 的小容量 agent 可由套件托管（`E2E_SMALL_CAPACITY_AGENT=true`）：BeforeSuite 自动部署固定名 `flux-sandbox-agent-small` pod（capacity=2），经本地 kubectl port-forward（17508）访问，AfterSuite 自动删除；标准集群与本地环境均无需节点端口，也可用 `E2E_SMALL_CAPACITY_AGENT_URL` 指向外部部署的 agent。
- 时序类用例的等待时间由配置推导：孤儿不清理观察窗为 `grace-5s`；等待 watcher 清理的超时为 `orphanTimeout + grace + 2*scan + 60s`。

## 已知限制

1. DP-E2E-005 / 029 需小容量 agent（设 `E2E_SMALL_CAPACITY_AGENT=true` 由套件自建，或指定外部 agent 地址），两者均未提供时自动 Skip。
2. DP-E2E-034（CR 收编停止孤儿跟踪）依赖 watcher 扫描与 Sandbox CR 创建之间的时序竞争，偶发抖动，单独重跑即可通过。
3. E2B/CRI 运行时路径不覆盖（标准 k8s 无 E2B shim，由产品仓 UT 覆盖）。
4. 控制面（Controller / Scheduler）不在数据面测试范围。
