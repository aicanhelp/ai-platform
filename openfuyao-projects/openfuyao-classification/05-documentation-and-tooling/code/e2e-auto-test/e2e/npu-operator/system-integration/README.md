# NPU-Operator System Integration Tests

本目录包含针对 **NPU Operator** 在 OpenFuyao 平台上运行的系统级集成测试。
这些测试基于 Go + Ginkgo v2 构建，通过 SSH 连接到集群节点并执行 kubectl 命令，验证关键组件
的状态、镜像版本、恢复能力以及与其他组件的兼容性。

> 📁 该目录下的测试代码允许访问同级和子目录中的所有文件（例如 `constants.go`、
> SSH 密钥、yaml 清单等），因此请确保不在此处存放敏感数据。

## 目录结构

```
constants.go               // 示例常量：SSH 节点信息、期望镜像、安装命令等
system_integration_npu-operator_test.go  // 实际测试实现
README.md                  // 本文件
```

## 运行环境要求

与 [installation/system-integration/README.md](../installation/system-integration/README.md) 中列出的
要求基本一致：

* Go >=1.19
* Ginkgo v2
* kubectl 与 client-go 可访问目标集群
* SSH 可连通指定节点（`constants.go` 中配置）

## 快速开始

1. 编辑 `constants.go`：填写 `NPUClusterNodeInfo`（主机/端口/用户/密码）、
   `ExpectedImages` 等。
2. 可选：调整安装/卸载命令以符合你的 helm 仓库与版本。
3. 在包目录运行：
   ```pwsh
   go run github.com/onsi/ginkgo/v2/ginkgo@latest -v \
       --label-filter="npu_operator_system_integration" ./...
   ```
4. 查看输出日志，错误信息会打印在标准输出。

## 测试内容概览

* 验证 NPU Operator 和相关组件 pod 是否 Running 且镜像符合期望
* 检查组件在被删除后是否自动恢复
* 与其他组件（比如 NUMA、KAE、Colocation 等）的安装冲突测试
* 统计 & 保存 exporter 监控指标以便调试

请参阅源代码以获取详细实现与更多示例。
