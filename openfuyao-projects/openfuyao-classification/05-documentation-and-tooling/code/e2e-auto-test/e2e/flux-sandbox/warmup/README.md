# FluxSandbox 快照预热 E2E 测试

本目录覆盖 flux-sandbox 控制器内嵌快照预热子系统，共 29 个用例（`WARM-E2E-001` ~ `WARM-E2E-029`）。用例定义见 [warmup-test.md](./warmup-test.md)。

## 前置条件

- Go `1.24.7` 或更新版本、`kubectl`、`ginkgo` CLI 可用（`go install github.com/onsi/ginkgo/v2/ginkgo@v2.28.1`，与本仓库 go.mod 版本一致）。
- `E2E_KUBECONFIG` 指向目标集群 kubeconfig；未设置时默认使用 `~/.kube/config`。
- flux 相关组件已部署：`flux-system` 中名为 `flux-sandbox-controller` 的 Deployment 处于 Ready，且 CRD/RBAC 已随 flux 部署安装。
- 当前用户有创建 Namespace/Deployment/Service/PVC、patch controller Deployment、port-forward Pod 的权限；集群无 StorageClass 时还需创建/删除 PersistentVolume（cluster-scoped）的权限。
- 集群节点可拉取 `flux-sandbox/fake-e2b:latest`、`postgres:16-alpine`、`quay.io/coreos/etcd:v3.5.10`。
- 本机端口 `18085`、`15432`、`12379`、`12380` 空闲（不要与 dataplane 套件并行运行：两者都会 port-forward flux-sandbox-controller，且本套件会临时修改 controller 启动参数）。

### 存储（自动处理，无需手工准备）

postgres 需要一块 PVC 保存 warm 表数据。套件在 BeforeSuite 自动检测集群是否有 StorageClass：

- 有 StorageClass：普通 PVC 由默认 StorageClass 自动供给，无需任何操作。
- 无 StorageClass（常见于单节点/裸机测试集群）：套件自动创建名为 `warm-postgres-static-pv` 的 hostPath 静态 PV（路径 `/var/lib/warm-e2e-pv`，Retain）并预绑定 PVC；AfterSuite 在 PVC 消失后删除该 PV，集群不残留测试资源。可用 `WARM_E2E_PG_PV_NAME` / `WARM_E2E_PG_HOSTPATH` 覆盖 PV 名与路径。

### Before/After 职责一览

所有前置准备与还原均由套件自动完成：

| 阶段 | 动作 |
| --- | --- |
| BeforeSuite | 创建 mock 命名空间；无 StorageClass 时自动准备静态 PV + 预绑定 PVC；部署 PostgreSQL / fake MoonCake etcd / fake E2B API-Server；执行 warm 表迁移并 TRUNCATE（每次运行从空库开始）；patch controller Deployment 注入 `--warm-*` 参数指向 mock 服务；建立 4 条本地 port-forward |
| AfterSuite | 恢复 controller 原始启动参数并等待其重新可用；删除 mock 命名空间（连带 Deployment/Service/PVC）；删除套件创建的静态 PV；关闭全部 port-forward 与 etcd 连接 |

## 一条指令执行 29 个用例

```bash
cd /path/to/e2e-auto-test/e2e
ginkgo -v --tags e2e -timeout 45m ./flux-sandbox/warmup
```

> 套件源码带 `//go:build e2e` 构建标签（与产品仓约定一致），`ginkgo` 需加 `--tags e2e`；等价的 `go test -tags=e2e -v -timeout=45m ./flux-sandbox/warmup` 亦可用于无 ginkgo CLI 的环境。

预期结果：

```text
Ran 29 of 29 Specs in <duration>
SUCCESS! -- 29 Passed | 0 Failed | 0 Pending | 0 Skipped
```

失败用例单独复跑：

```bash
ginkgo -v --tags e2e -timeout 15m -focus "WARM-E2E-007" ./flux-sandbox/warmup
```
