# BKE 安装部署性能测试

本目录包含 BKE 安装部署性能测试用例，用于测试 BKE 在线安装和集群创建的性能。

## 测试内容

1. **在线安装性能测试**：测试引导节点初始化的性能
2. **集群创建性能测试**：测试从引导节点创建集群并在后端节点验证的性能
3. **业务集群资源占用测试**：测试业务节点上Pod的CPU和内存资源占用情况
4. **多节点集群创建测试**：同时测试5个不同配置的集群创建（1master1worker、1master2worker、1master3worker、1master4worker、3master2worker）

## 环境准备

### 1. 配置环境变量

复制环境变量模板文件并填写实际的机器信息：

```bash
cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e\installation\performance-test
cp .env.performance.template .env.performance

编辑 `.env.performance` 文件，填写以下信息：

- **引导节点信息**：用于初始化 BKE
- **前端节点信息**：目前未使用，可以留空
- **后端节点信息**：用于创建集群
- **业务节点信息**：用于业务集群资源占用测试
- **多节点集群测试配置**：5个引导节点和19台机器的配置信息

### 2. 安装依赖

确保您已经安装了 Go 环境（版本 1.18 或更高），然后安装测试依赖：

```bash
cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e
# 安装项目依赖
go mod tidy

## 运行测试

### 1. 运行全部性能测试

```bash
cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e\installation\performance-test
# 使用 ginkgo 运行测试
go run github.com/onsi/ginkgo/v2/ginkgo -v --timeout=6h ./...

### 2. 运行特定的测试套件

- 运行 BKE 在线安装性能测试套件：

  ```bash
  cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e\installation\performance-test
  go run github.com/onsi/ginkgo/v2/ginkgo -v --timeout=6h --label-filter="online-installation-performance"
  ```

- 运行业务集群资源占用测试套件：

  ```bash
  cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e\installation\performance-test
  go run github.com/onsi/ginkgo/v2/ginkgo -v --timeout=6h --label-filter="business-cluster-resource"
  ```

- 运行多节点集群创建测试套件：

  ```bash
  cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e\installation\performance-test
  go run github.com/onsi/ginkgo/v2/ginkgo -v --timeout=6h --label-filter="multi-node,cluster-creation"
  ```

### 3. 运行特定的测试用例

- 只运行在线安装性能测试：

  ```bash
  cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e\installation\performance-test
  go run github.com/onsi/ginkgo/v2/ginkgo -v --timeout=6h --focus="should complete bke init with performance metrics"
  ```

- 只运行集群创建性能测试：

  ```bash
  cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e\installation\performance-test
  go run github.com/onsi/ginkgo/v2/ginkgo -v --timeout=6h --focus="should create cluster from guide node and verify on backend node"
  ```

- 只运行业务集群资源占用测试：

  ```bash
  cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e\installation\performance-test
  go run github.com/onsi/ginkgo/v2/ginkgo -v --timeout=6h --focus="should check resource usage on business node"
  ```

- 只运行多节点集群创建测试：

  ```bash
  cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e\installation\performance-test
  go run github.com/onsi/ginkgo/v2/ginkgo -v --timeout=6h --focus="should create multi-node clusters with different configurations"
  ```

### 4. 运行测试并生成报告

```bash
cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e\installation\performance-test
# 运行测试并输出 JSON 格式的报告
go run github.com/onsi/ginkgo/v2/ginkgo -v --timeout=6h --json-report=performance-test-report.json ./...

## 测试结果

测试完成后，您将在控制台看到详细的测试结果，包括：

- 每个步骤的耗时
- 性能指标统计
- 集群资源占用情况（按命名空间和总体汇总）
- 多节点集群创建测试的就绪状态统计
- 测试是否通过

测试结果同时会保存到 `performance_test_results.txt` 文件中，方便后续查看和分析。

## 注意事项

1. 确保所有测试节点都可以通过 SSH 访问
2. 测试可能需要较长时间（建议设置至少 6 小时超时）
3. 测试过程中不要中断，否则可能影响测试结果
4. 如果测试失败，可以查看详细的日志输出来定位问题
5. 资源占用测试会显示详细的调试信息，方便验证统计结果的准确性

## 自定义配置

您可以根据需要修改 `constants.go` 文件中的常量：

- `BKE_DOWNLOAD_COMMAND`：BKE 二进制文件下载命令
- `BKE_INIT_COMMAND`：BKE 初始化命令
- 其他超时和间隔时间设置