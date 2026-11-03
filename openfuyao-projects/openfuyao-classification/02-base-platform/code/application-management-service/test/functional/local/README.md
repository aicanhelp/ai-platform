# Operation Envtest 集成测试

本目录包含 `application-management-service` 的 Operation 层（`pkg/helm`）集成测试。测试基于 `envtest` 启动的真实 Kubernetes 控制面（`kube-apiserver` + `etcd`）运行，用于验证 Helm 真实实现与 Kubernetes API 的端到端兼容性。

## 测试范围

| 场景 | 描述 |
|------|------|
| Release 查询 | 获取 Release 详情、Manifest、Values |
| Release 生命周期 | Install、Upgrade、Rollback、Uninstall |
| 依赖 Mock 边界 | 仅 Mock Chart 来源（`marketplace-service`），通过注入 `FakeChartGetter` 实现 |

说明：
- 不 Mock `pkg/helm` 内部业务逻辑。
- 不 Mock Helm SDK Action 执行（如 `action.Run()`）。
- 不 Mock Kubernetes 控制面（由 `envtest` 提供）。

## 目录结构

```plain
test/functional/local
├── application_suite_test.go    # Suite 入口
├── envtest_context_test.go      # envtest 上下文与通用初始化逻辑
├── application_test.go          # 集成测试用例
├── README.md                    # 本文档
└── (依赖) chartgetter/
    └── fake_chart_getter.go     # FakeChartGetter：Mock Chart 获取
```

## 前置条件

1. 本机可运行 `envtest` 依赖的 `kube-apiserver` 与 `etcd` 二进制。
2. 已安装 Go（版本以项目 `go.mod` 为准）。
3. 可正常联网下载 `envtest` 资源（或提前缓存）。

## 环境准备

`envtest` 需要通过环境变量 `KUBEBUILDER_ASSETS` 指定二进制所在目录。

### 1. 安装 setup-envtest

`setup-envtest` 是 controller-runtime 提供的工具，用于下载/管理 `envtest` 所需二进制。

```bash
go install sigs.k8s.io/controller-runtime/tools/setup-envtest@latest
```

注意：确保 `$(go env GOPATH)/bin` 已加入系统 `PATH`，否则终端无法直接执行 `setup-envtest`。(export PATH="/root/go/bin:$PATH")

### 2. 下载二进制并设置 KUBEBUILDER_ASSETS

在运行测试的同一个终端会话中设置环境变量。

Windows (PowerShell)：

```powershell
$env:KUBEBUILDER_ASSETS = (setup-envtest use -p path)
```

Windows (CMD)：

```cmd
for /f "delims=" %i in ('setup-envtest use -p path') do set KUBEBUILDER_ASSETS=%i
```

Linux / macOS (Bash)：

```bash
export KUBEBUILDER_ASSETS="$(setup-envtest use -p path)"
```


## 运行测试

在项目根目录执行：

```bash
go test ./test/functional/local -v
```