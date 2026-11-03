# System Integration Tests - OpenFuyao E2E 测试套件

## 项目概述

本项目是针对 **OpenFuyao** 容器云平台的完整系统集成测试套件。通过 Go + Ginkgo v2 框架，对多节点 Kubernetes 集群进行分层次、全方位的功能验证和稳定性测试。

### 测试拓扑

- **1 个引导节点** (Bootstrap Node) - 单节点安装与配置验证
- **1 个 3-Master 管理集群** (Management Cluster) - HA 高可用集群检测
- **1 个 2-Master 业务集群** (Workload Cluster) - 工作负载集群稳定性测试

### 核心测试范围

- ✓ 集群健康状态检查 (节点、Pod、组件)
- ✓ Service 发现与连通性验证
- ✓ DNS 解析与网络通信测试
- ✓ 故障恢复能力验证 (Master/Worker 宕机)
- ✓ 高压负载测试 (etcd、API Server)

---

## 环境要求

| 组件 | 版本要求 |
|------|---------|
| **Go** | >= 1.19 |
| **Ginkgo** | v2 |
| **kubectl** | >= 1.20 |
| **Kubernetes** | >= 1.20 |
| **etcdctl** | v3.5+ (用于负载测试) |

### 节点与网络要求

四个 Linux 节点，通过 SSH 连接（含密码认证）：

```
BootstrapNodeInfo           引导节点
├── Host: <IP/Hostname>
├── Port: 22
├── User: <SSH 用户>
└── Pass: <密码>

ManagementClusterNodeInfo   管理集群主节点
├── Host: <IP/Hostname>
├── Port: 22
├── User: <SSH 用户>
└── Pass: <密码>

WorkloadClusterNodeInfo     业务集群节点
├── Host: <IP/Hostname>
├── Port: 22
├── User: <SSH 用户>
└── Pass: <密码>

FailWorkLoadNodeInfo        故障测试节点
├── Host: <IP/Hostname>
├── Port: 22
├── User: <SSH 用户>
└── Pass: <密码>
```

---

## 测试套件详解

### 1️⃣ System Integration Tests (`system_integration_test.go`)

**标签**: `system_integration`

**目标**: 验证三层集群的核心健康状态与故障恢复能力。

#### 测试用例

| # | 名称 | 验证项 |
|----|------|-------|
| 1.1 | **Bootstrap Node Configuration** | Pod 状态、镜像版本、配置文件完整性 |
| 1.2 | **Management Cluster Health Check** | 节点 Ready、Pod 副本、镜像版本匹配 |
| 1.3 | **Workload Cluster Health Check** | 同上 |
| 1.4 | **Management Cluster Single Master Failure Test** | Master 宕机后的自动恢复、数据完整性 |
| 1.5 | **Workload Cluster Worker Node Failure Test** | Worker 失效后 Pod 自动转移 |

**关键验证**:
```
✓ Node Ready 状态 = true
✓ CoreDNS Pod Ready >= 2
✓ etcd Pod Ready >= 3 (管理集群)
✓ API Server 可用性
✓ 故障恢复时间 < 5 分钟
```

---

### 2️⃣ Service Status and Connectivity Tests (`system_integration_service_test.go`)

**标签**: `system_integration_services`

**目标**: 全面验证 Service 对象与网络连通性。

#### 测试用例

| # | 名称 | 验证项 |
|----|------|-------|
| 2.1 | **Service Status and Connectivity Tests** | Service 存在性、类型、ClusterIP、连通率 |

**关键验证**:
```
✓ Service 发现率 = 100% (所有期望 Service 已部署)
✓ Service 类型正确 (ClusterIP/NodePort/LoadBalancer)
✓ ClusterIP 有效分配
✓ 关键 Service HTTP 响应 < 500
✓ 连接超时 < 10s
```

**覆盖的关键 Service**:
- `kubernetes.default` - Kubernetes API Server
- `kube-dns` - CoreDNS
- `prometheus-k8s` - Prometheus 监控
- `alertmanager-main` - 告警管理器
- `console` / `application-management` - 自定义应用服务

---

### 3️⃣ DNS and Network Tests (`system_integration_dns_test.go`)

**标签**: `system_integration_dns_network` / `system_integration_network_only`

**目标**: DNS 域名解析与 Pod 间网络通信验证。

#### 测试用例

| # | 名称 | 验证项 |
|----|------|-------|
| 3.1 | **CoreDNS Resolution Tests** | 同/跨命名空间域名解析、外部域名解析 |
| 3.2 | **Pod Network Communication Tests** | Pod 间通信、延迟分布、带宽性能 |

**关键验证**:
```
✓ CoreDNS Pod Ready >= 2
✓ 同命名空间完整域名解析成功 (FQDN)
✓ 跨命名空间 Service 可访问
✓ 外部域名 (如 google.com) 解析成功
✓ Pod 间平均延迟 (RTT) < 10ms
✓ 跨节点通信带宽 > 100Mbps
```

---

### 4️⃣ Workload Cluster Load Testing (`system_integration_etcd_apiserver_test.go`)

**标签**: `system_integration_etcd_apiserver_load_test`

**目标**: 数据平面与控制平面的高压负载与稳定性测试。

#### 测试用例

| # | 名称 | 验证项 |
|----|------|-------|
| 4.1 | **etcd Load Testing** | 并发写入、读取延迟分布 (P50/P95/P99) |
| 4.2 | **API Server Load Testing** | QPS 吞吐量、响应延迟、错误率 |

**关键验证**:
```
✓ etcd 并发写入成功率 >= 95%
✓ etcd 读取 P95 延迟 < 50ms
✓ API Server QPS 峰值 >= 1000
✓ 99% 请求延迟 < 200ms
✓ 长时间运行无内存泄漏
```

---

## 快速开始

### 步骤 1: 配置常量文件

编辑 `constants.go`，填入实际环境信息：

```go
// 节点 SSH 连接信息
var BootstrapNodeInfo = struct {
    Host string
    Port int
    User string
    Pass string
}{
    Host: "192.168.1.100",
    Port: 22,
    User: "root",
    Pass: "your-password",
}

var ManagementClusterNodeInfo = struct{...}{...}
var WorkloadClusterNodeInfo = struct{...}{...}
var FailWorkLoadNodeInfo = struct{...}{...}

// 集群名称（不含 bke- 前缀）
const (
    ManagementClusterName = "mgmt"
    WorkloadClusterName   = "workload"
)

// 期望的核心 Pod 列表 (在 BootstrapNodeExpectations 等中)
var BootstrapNodeExpectations = map[string]ComponentExpectation{
    "etcd": {
        Name:         "etcd",
        MinReplicas:  1,
        Image:        "registry.example.com/etcd:v3.5.x",
        ExpectedReady: true,
        Labels: map[string]string{"component": "etcd"},
    },
    // ... 更多组件
}
```

### 步骤 2: 运行完整测试

```bash
cd e2e/installation/system_integration

# 全部测试
go run github.com/onsi/ginkgo/v2/ginkgo@latest -v ./...

# 运行特定标签的测试
go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="system_integration_bootstrap" ./...
go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="system_integration_management" ./...
go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="system_integration_services" ./...
go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="system_integration_dns_network" ./...
go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="system_integration_etcd_apiserver_load_test" ./...
```

### 步骤 3: 查看测试结果

```bash
# 编译检查
go build ./...

# 运行并输出详细日志
go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --timeout=30m ./... 2>&1 | tee test-results.log
```

---

## 文件结构

```
system_integration/
├── README.md                                  ← 本文件
├── constants.go                               ← 节点信息、集群名称、期望值配置
├── system_integration_suite_test.go           ← Ginkgo 套件入口
├── system_integration_test.go                 ← 主测试：集群检查 + 故障恢复
├── system_integration_service_test.go         ← Service 连通性测试
├── system_integration_dns_test.go             ← DNS 和网络通信测试
├── system_integration_etcd_apiserver_test.go  ← 高压负载测试
├── utils/
│   ├── cluster_checker.go                     ← 集群检查工具
│   └── ...
└── get-config.sh                              ← 导出 kubeconfig 脚本
```

---

## 核心工具与组件

### SSH Executor (`executor` 包)

在远程节点执行命令：
```go
executor, _ := executor.NewSSHExecutor(host, port, user, pass)
result, _ := executor.Exec("kubectl get pods -A")
```

### Cluster Checker (`utils/cluster_checker.go`)

获取集群 kubeconfig 和创建客户端：
```go
checker := utils.NewClusterChecker(sshExecutor, clusterName)
kubeconfigStr, _ := checker.GetKubeconfig()
clientset, _ := checker.CreateClientset(kubeconfigStr)
```

### 期望配置 (`constants.go`)

定义期望的组件、Service、标签等：
```go
var BootstrapNodeExpectations = map[string]ComponentExpectation{
    "etcd": {...},
    "apiserver": {...},
    // ...
}

var ServiceExpectations = map[string]ServiceExpectation{
    "kube-dns": {Namespace: "kube-system", Name: "kube-dns", Type: "ClusterIP"},
    // ...
}
```

---

## 常见问题排查

### ❌ SSH 连接失败

```
Error: dial tcp x.x.x.x:22: connection refused
```

**检查清单**:
- [ ] 网络连通性：`ping <host>`
- [ ] SSH 端口开放：`telnet <host> 22`
- [ ] 用户/密码正确
- [ ] 防火墙规则允许 SSH

---

### ❌ kubeconfig 获取失败

```
Error: 无法获取业务集群 workload 的 kubeconfig
```

**检查清单**:
- [ ] 集群名称无 `bke-` 前缀 (在 `constants.go` 中)
- [ ] 集群已部署并可通过 `kubectl` 访问
- [ ] SSH 用户有权访问 kubeconfig 文件

---

### ❌ Pod 验证失败

```
Expected Pod count: 3, Actual: 2
```

**检查清单**:
- [ ] `constants.go` 中的期望 Pod 列表与实际部署一致
- [ ] 运行 `kubectl get pods -A` 查看实际 Pod
- [ ] 更新期望值或修复部署配置

---

### ❌ DNS 测试超时

```
nslookup: no answer
```

**检查清单**:
- [ ] CoreDNS Pod 是否运行：`kubectl get pods -n kube-system | grep coredns`
- [ ] DNS Service 是否存在：`kubectl get svc kube-dns -n kube-system`
- [ ] Pod 网络插件 (Flannel/Calico) 是否就绪

---

## 性能基准参考

基于标准部署的预期指标：

| 指标 | 目标值 | 备注 |
|------|--------|------|
| **Service 发现率** | 100% | 所有期望 Service 已部署 |
| **DNS 同 NS 延迟** | < 50ms | 完整域名解析 |
| **Pod 间 RTT** | < 10ms | 跨节点通信 |
| **etcd 写成功率** | ≥ 95% | 并发写入 |
| **API Server QPS** | > 1000 | 标准负载 |
| **Master 恢复时间** | < 5min | 从宕机到可用 |

---

## 扩展与自定义

### 添加新的验证函数

```go
// 在对应的 _test.go 文件中
func verifyCustomComponent(ctx context.Context, clientset kubernetes.Interface) {
    pods, _ := clientset.CoreV1().Pods("custom-ns").List(ctx, metav1.ListOptions{})
    // 自定义验证逻辑
    GinkgoWriter.Printf("Custom check passed\n")
}
```

### 添加新的测试场景

```go
It("应该验证自定义业务逻辑", func() {
    By("步骤 1: 准备环境")
    // 初始化
    
    By("步骤 2: 执行操作")
    // 执行操作
    
    By("步骤 3: 验证结果")
    Expect(result).To(Equal(expected))
})
```

### 更新期望配置

```bash
# 在 constants.go 中修改期望值
var BootstrapNodeExpectations = map[string]ComponentExpectation{
    "component-name": {
        Name:         "component-name",
        MinReplicas:  2,
        Image:        "new-image:v1.0",
        ExpectedReady: true,
    },
}
```

---

## 贡献指南

1. **代码规范**: 遵循 Go 最佳实践 (`gofmt`, `golint`)
2. **文档**: 为新函数添加详细中文注释
3. **测试**: 本地验证后再提交
4. **提交信息**: 清晰描述改动内容

---

## 许可证

参见项目根目录 `LICENSE` 文件。

---

## 相关文档

- [Ginkgo 官方文档](https://onsi.github.io/ginkgo/)
- [Kubernetes 文档](https://kubernetes.io/docs/)
- [etcd 官方文档](https://etcd.io/docs/)

---

**最后更新**: 2026-01-30  
**版本**: 1.0.0  
**状态**: ✅ 生产可用go
// 引导节点信息
BootstrapNodeInfo = struct {
	Host string
	User string
}{
	Host: "192.168.201.40",
	User: "root",
}

// 集群配置信息
ClusterConfigs = struct {
	ManagementCluster struct {
		Name              string
		ConfigPath        string
		ExpectedMasterNum int
	}
	// ...
}{
	ManagementCluster: struct {
		Name              string
		ConfigPath        string
		ExpectedMasterNum int
	}{
		Name:              "bke-cluster-mgmt",
		ConfigPath:        "/bke/cluster/bke-cluster-mgmt/bke-cluster-mgmt.yaml",
		ExpectedMasterNum: 3,
	},
	// ...
}

// Kubernetes 核心组件期望值
K8sComponentExpectations = struct {
	APIServer struct {
		Name           string
		Namespace      string
		ExpectedCount  int
		ExpectedImage  string
		ExpectedLabels map[string]string
	}
	// ...
}{
	APIServer: struct {
		Name           string
		Namespace      string
		ExpectedCount  int
		ExpectedImage  string
		ExpectedLabels map[string]string
	}{
		Name:          "kube-apiserver",
		Namespace:     "kube-system",
		ExpectedCount: 3, // 根据实际修改
		ExpectedImage: "k8s.gcr.io/kube-apiserver:v1.28.0",
		ExpectedLabels: map[string]string{
			"component": "kube-apiserver",
		},
	},
	// ...
}

// BKE 集群配置版本
BKEClusterConfigVersions = struct {
	ManagementCluster struct {
		Name            string
		Version         string
		KubernetesImage string
	}
	// ...
}{
	ManagementCluster: struct {
		Name            string
		Version         string
		KubernetesImage string
	}{
		Name:            "bke-cluster-mgmt",
		Version:         "v1beta1",
		KubernetesImage: "k8s.gcr.io/kube-apiserver:v1.28.0",
	},
	// ...
}
```

### 2. 配置环境变量

需要配置 SSH 连接信息，可以通过以下方式：

- **环境变量方式**：
  ```bash
  export SSH_HOST=192.168.201.40
  export SSH_USER=root
  export SSH_PORT=22
  ```

- **配置文件方式**：
  在项目根目录创建 `.env.system-integration` 文件：
  ```
  SSH_HOST=192.168.201.40
  SSH_USER=root
  SSH_PORT=22
  ```

### 3. 运行测试

```bash
# 运行所有系统集成测试
ginkgo -v ./e2e/installation/system_integration/...

# 运行特定测试
ginkgo -v ./e2e/installation/system_integration/... --focus="Bootstrap Node"

# 运行管理集群测试
ginkgo -v ./e2e/installation/system_integration/... --focus="Management Cluster"

# 运行业务集群1测试
ginkgo -v ./e2e/installation/system_integration/... --focus="Workload Cluster 1"

# 运行业务集群2测试
ginkgo -v ./e2e/installation/system_integration/... --focus="Workload Cluster 2"

# 运行并生成报告
ginkgo -v --junit-report=report.xml ./e2e/installation/system_integration/...
```

## 测试验证项目

### Bootstrap Node Configuration
- ✅ SSH 连接验证
- ✅ /bke/cluster 目录存在性
- ✅ 各集群配置文件验证（文件存在、名称匹配、命名空间配置）

### Management/Workload Cluster Health Check
- ✅ Master 节点数量验证
- ✅ 节点 Ready 状态验证
- ✅ K8s 核心组件：
  - kube-apiserver
  - kube-controller-manager
  - kube-scheduler
  - etcd
  - kubelet
  - kube-proxy
  - 其他自定义组件
- ✅ Pod 状态验证（Running、Ready）
- ✅ Pod 个数验证
- ✅ Pod 镜像版本验证
- ✅ Pod 标签验证

## 常见问题

### Q: SSH 连接失败？
**A:** 
1. 检查引导节点 IP 地址和用户名是否正确
2. 检查 SSH 密钥是否正确配置
3. 检查防火墙是否允许 SSH 连接

### Q: kubeconfig 获取失败？
**A:**
1. 确保业务集群已成功创建
2. 检查 kubeconfig 是否存在于默认路径或 Secret 中
3. 检查集群名称是否配置正确

### Q: 核心组件 Pod 数量不符合预期？
**A:**
1. 检查集群实际的 Master 节点数量
2. 根据节点数量调整 `ExpectedCount` 值
3. 某些组件可能是 DaemonSet，数量应该等于 Worker 节点数

### Q: 镜像版本检查失败？
**A:**
1. 获取实际的镜像版本：
   ```bash
   kubectl get pods -n kube-system -o wide | grep -E "apiserver|controller|scheduler"
   ```
2. 更新 `K8sComponentExpectations` 中的 `ExpectedImage` 值

## 扩展指南

如需添加更多的 K8s 组件检查，参考以下步骤：

1. 在 `constants.go` 的 `K8sComponentExpectations` 中添加新的组件配置
2. 在 `system_integration_test.go` 的验证函数中添加相应的检查逻辑
3. 根据组件类型（Pod、DaemonSet、Deployment）编写对应的验证代码

## 依赖项

- Ginkgo v2
- Kubernetes client-go
- executor 框架（用于 SSH 连接和命令执行）
- cluster_checker 工具（用于集群信息获取）

## 相关文件

- `e2e/installation/utils/cluster_checker.go` - 集群检查工具
- `e2e/framework/executor/` - 执行器框架
