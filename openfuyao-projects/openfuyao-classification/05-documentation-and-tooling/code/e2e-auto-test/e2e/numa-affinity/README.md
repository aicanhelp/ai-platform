# NUMA Affinity Functional Test

本目录为 NUMA 插件后端功能测试（Ginkgo），用于验证在虚拟机环境下的核心逻辑可用性。

## 说明

- 本目录根下为 functional test。
- `system-integration` 子目录是组件集成测试目录，和这里的功能测试分开维护。
- 用例以“稳定可通过”为优先，复杂物理机 NUMA 拓扑校验不作为阻塞条件。

## 当前覆盖

- `NUMA2503-052` 独立部署后 volcano 基础功能验证
- `NUMA2503-053` 独立部署 helm 卸载验证（支持自动恢复）
- `NUMA2503-054` 独立部署-环境正交-openEuler22.03-x86（根据架构自动跳过）
- `NUMA2503-055` 独立部署-环境正交-openEuler22.03-arm（根据架构自动跳过）
- `NUMA2503-033` 运行时 Pod 亲和优化基础检查（NRI / net_rship）
- `NUMA2503-040` 重启 containerd 对现有业务影响
- `NUMA2503-025` 重启 kubelet 对现有业务影响
- `NUMA2503-009` 重启 kubelet 对现有业务影响（static策略未打开）
- `VCS-INT-001` 分层开关-总开关关闭（集成测试模块）
- `VCS-INT-002` 分层开关-volcano 主组件关闭（集成测试模块）
- `VCS-INT-003` 分层开关-resourceExporter 关闭（集成测试模块）

PS：
- 开关交互与策略切换相关用例优先在前端 Playwright 覆盖（通过页面真实点击链路执行）。
- 后端 functional 主要覆盖服务稳定性、重启影响和安装/卸载能力。

## 环境变量

优先从以下文件加载：

1. `e2e/.env`

关键变量示例：

- `FUNCTIONAL_TEST_IP`
- `FUNCTIONAL_TEST_USER`
- `FUNCTIONAL_TEST_PASSWORD`

说明：

- 若未显式设置 `FUNCTIONAL_TEST_*`，测试会尝试复用 `TEST_NODE2_*`。
- 套件会在 `BeforeSuite` 自动安装 `numa-affinity-package`（`latest`），并在 `AfterSuite` 自动卸载。

## 运行方式

在仓库根目录执行：

```bash
cd e2e
ginkgo -v ./numa-affinity
```

按标签执行示例：

```bash
cd e2e
ginkgo -v --label-filter="NUMA2503-052 || NUMA2503-053 || NUMA2503-033 || NUMA2503-040 || NUMA2503-025" ./numa-affinity
```
