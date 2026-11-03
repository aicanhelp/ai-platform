# Eagle-Eye System Integration Tests

这个目录包含针对 **Eagle-Eye** 服务的系统集成测试代码。
测试用例通过 SSH 执行 kubectl 命令，验证 Eagle-Eye 服务的 Pods、NATS 日志、
硬件检测功能等，属于更高层级的 end‑to‑end 测试。

测试文件可以访问本目录及子目录的所有资源（例如 `constants.go`、yaml 清单），
请勿在此存放敏感密码。

## 使用说明

1. 在 `constants.go` 中配置 `EagleEyeClusterNodeInfo`、`ExpectedImages` 及
   组件测试需要的 SSH/Kube 环境，环境中需要提前下载eagle-eye包并安装到集群中。
2. 运行所有测试：
   ```pwsh
   go run github.com/onsi/ginkgo/v2/ginkgo@latest -v \
       --label-filter="eagle_eye_system_integration" ./...
   ```
3. 运行单个测试用例：
   ```pwsh
   go run github.com/onsi/ginkgo/v2/ginkgo@latest -v \
       --label-filter="eagle_eye_system_integration" \
       --focus="测试用例名称" ./...
   ```

## 测试用例列表

以下是所有可用的测试用例及其执行命令：

| 测试用例名称 | 执行命令 |
|------------|----------|
| eagle-eye & logging-package | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="eagle-eye & logging-package" ./...` |
| eagle-eye & colocation-package | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="eagle-eye & colocation-package" ./...` |
| eagle-eye & many-core-scheduler | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="eagle-eye & many-core-scheduler" ./...` |
| eagle-eye & numa-affinity-package | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="eagle-eye & numa-affinity-package" ./...` |
| eagle-eye & multi-cluster-service | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="eagle-eye & multi-cluster-service" ./...` |
| eagle-eye & ray-package | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="eagle-eye & ray-package" ./...` |
| eagle-eye & kae-operator | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="eagle-eye & kae-operator" ./...` |
| eagle-eye & npu-operator | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="eagle-eye & npu-operator" ./...` |
| eagle-eye & infernex | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="eagle-eye & infernex" ./...` |
| eagle-eye & monitoring-dashboard | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="eagle-eye & monitoring-dashboard" ./...` |
| should have Eagle-Eye related pods running and images match expectations | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="should have Eagle-Eye related pods running and images match expectations" ./...` |
| should have NATS service running and logs show readiness | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="should have NATS service running and logs show readiness" ./...` |
| should have hardware-diagnosis service running and logs show events detection | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="should have hardware-diagnosis service running and logs show events detection" ./...` |
| should have hardware-monitor service running and logs show hardware metrics collection | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="should have hardware-monitor service running and logs show hardware metrics collection" ./...` |
| should deploy subscriber and publisher to verify NATS functionality | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="should deploy subscriber and publisher to verify NATS functionality" ./...` |
| should automatically recover Eagle-Eye components after pod deletion | `go run github.com/onsi/ginkgo/v2/ginkgo@latest -v --label-filter="eagle_eye_system_integration" --focus="should automatically recover Eagle-Eye components after pod deletion" ./...` |

4. 仔细阅读失败用例输出，可以定位问题所在（镜像不匹配、日志关键字缺失等）。

## 覆盖测试范围

* Eagle-Eye 核心组件 Pod 状态与镜像验证
* NATS 服务读取并检查就绪日志
* Hardware-diagnosis 和 hardware-monitor 日志关键词检查
* NATS 发布/订阅功能完整性测试

更多细节在 `system_integration_eagle_eye_test.go`中。

## 依赖与环境

请参考 `installation/system-integration/README.md` 以获取一般环境要求，
包括 Go、Ginkgo、kubectl 版本以及 SSH 访问权限。