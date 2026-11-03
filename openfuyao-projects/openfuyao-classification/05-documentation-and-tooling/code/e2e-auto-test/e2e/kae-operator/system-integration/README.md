# Kae-Operator System Integration Tests

本目录为 **Kae Operator** 提供一组系统集成测试，用于验证 Operator
本身及其与 NUMA、NPU 等其他组件的运行兼容性。测试通过 SSH 访问集群
节点，执行 Kubernetes 操作并检查响应。

测试文件可访问本目录下任意文件，因此请确保只包含非敏感信息。

## 准备工作

1. 编辑 `constants.go`：填写 `ExpectedImages`、安装/卸载 Helm 命令，
    或者直接在 `kae-operator-config/.env.kae-operator` 中设置 SSH 环境变量。
2. 确保目标测试节点可通过 SSH 连接，并可执行 `kubectl` 命令。

## 运行命令

```pwsh
go run github.com/onsi/ginkgo/v2/ginkgo@latest -v \
    --label-filter="kae_operator_system_integration" ./...
```

可以组合 `--focus`/`--skip` 等指令限制执行范围。

## 测试内容概览

* Pod 状态与镜像校验
* 组件删除后的自动恢复测试
* KAE 与 NUMA/NPU 的冲突安装测试

详细逻辑请参见 `system_integration_kae_operator.go`。

## 环境要求

参照根级 `installation/system-integration/README.md`，一般需要 Go、
kubectl、Ginkgo 及有效的 SSH/Kube 配置。测试会访问目录下所有文件，
请勿保存敏感资料。

此外为了 SSH 连接，请在环境变量中指定 `KAE_TEST_IP`/`PORT`/`USER`/`PASSWORD`
，可以通过 `kae-operator-config/.env.kae-operator` 方便地加载。
