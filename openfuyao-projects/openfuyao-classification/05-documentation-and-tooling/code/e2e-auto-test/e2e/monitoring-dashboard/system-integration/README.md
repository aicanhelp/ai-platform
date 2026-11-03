# Monitoring-Dashboard System Integration Tests

该目录包含针对 **Monitoring Dashboard** 组件的系统级集成测试。
测试同样使用 Go + Ginkgo v2，对集群环境中的相关 Pod、Service、安装冲突等行为进行验证。
所有测试代码都可以访问本目录下的文件，因此可以通过修改 `constants.go` 来影响测试行为。

## 快速入门

1. 编辑 `constants.go`：填写 SSH 节点地址 (`MonitoringClusterNodeInfo`)、
   `ExpectedImages`、安装/卸载 Helm 命令等。
2. 执行以下命令运行测试：
   ```pwsh
   go run github.com/onsi/ginkgo/v2/ginkgo@latest -v \
       --label-filter="monitoring_dashboard_system_integration" ./...
   ```
3. 根据输出调整常量或环境，失败的断言会打印上下文信息。

## 测试内容

主要包括：

* 检查 Dashboard 相关 Pod 是否正常运行并使用预期镜像
* 删除 Pod 以验证 Operator 的自愈能力
* 安装其他组件（NUMA、Colocation、RAY 等）并观察是否与 Dashboard 冲突
* 资源释放后检查集群 Pod 状态

更多细节请参考 `system_integration_monitoring_dashboard_test.go` 文件。

## 环境要求

与根目录 `installation/system-integration/README.md` 中列出的环境要求一致。请确保
SSH 密钥/密码可用，并具备对目标集群的 kubeconfig 访问权限。
