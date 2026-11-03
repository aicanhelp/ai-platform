# Eagle Eye 性能测试

## 概述

本性能测试用于评估 Eagle Eye 组件的安装性能和运行性能，包括：
1. 安装时间统计
2. Pod 就绪时间统计
3. 资源使用情况统计

支持同时测试310机器和910机器。

## 测试内容

### 1. 环境变量获取
从环境变量中获取测试机器信息，包括：
- 服务器地址、端口、用户名、密码
- Kubernetes 配置文件路径

### 2. Helm 安装性能
使用 Helm 安装 Eagle Eye 组件，并统计：
- 安装总时长
- 各个 Pod（hardware-diagnosis、hardware-monitor、nats）从安装到就绪的时长
- 最多等待 5 分钟，15 秒轮询一次检查 Pod 状态

### 3. 资源使用统计
统计 Eagle Eye 相关 Pod 的 CPU 和内存使用情况。

## 环境变量配置

### 配置文件方式

在当前目录（`e2e/eagle-eye/performance-test/`）下创建 `.env` 文件并配置环境变量：

```bash
# Eagle Eye 性能测试环境配置
# 310机器配置
EAGLE_EYE_310_TEST_IP=192.168.1.310
EAGLE_EYE_310_TEST_PORT=22
EAGLE_EYE_310_TEST_USER=root
EAGLE_EYE_310_TEST_PASSWORD=password123
EAGLE_EYE_310_KUBECONFIG_PATH=/root/.kube/config

# 910机器配置
EAGLE_EYE_910_TEST_IP=192.168.1.910
EAGLE_EYE_910_TEST_PORT=22
EAGLE_EYE_910_TEST_USER=root
EAGLE_EYE_910_TEST_PASSWORD=password123
EAGLE_EYE_910_KUBECONFIG_PATH=/root/.kube/config
```

### 直接设置环境变量

也可以直接设置环境变量：

```bash
# 310机器配置
export EAGLE_EYE_310_TEST_IP="your-310-server-ip"
export EAGLE_EYE_310_TEST_PORT="22"
export EAGLE_EYE_310_TEST_USER="root"
export EAGLE_EYE_310_TEST_PASSWORD="your-password"
export EAGLE_EYE_310_KUBECONFIG_PATH="/path/to/310/kubeconfig"

# 910机器配置
export EAGLE_EYE_910_TEST_IP="your-910-server-ip"
export EAGLE_EYE_910_TEST_PORT="22"
export EAGLE_EYE_910_TEST_USER="root"
export EAGLE_EYE_910_TEST_PASSWORD="your-password"
export EAGLE_EYE_910_KUBECONFIG_PATH="/path/to/910/kubeconfig"

## 运行测试

### 1. 配置环境变量

编辑 `eagle-eye-config/.env.eagle-eye` 文件，配置310和910机器的信息。

### 2. 运行测试

```bash
# 进入性能测试目录
cd d:\VscodeWorkspace\e2e-auto-test-xc\e2e\eagle-eye\performance-test

# 运行所有测试
ginkgo run -v

# 只运行310机器测试
ginkgo run -v -focus="310"

# 只运行910机器测试
ginkgo run -v -focus="910"

## 测试结果

### 控制台输出
测试运行过程中，控制台会输出详细的测试步骤和结果，包括：
- 安装进度
- Pod 就绪状态和时间
- 资源使用情况

### 结果文件
测试完成后，会生成以下结果文件：

#### 1. 性能结果文件
- 310机器结果：`eagle-eye-performance-result-310-YYYYMMDDHHMMSS.json`
- 910机器结果：`eagle-eye-performance-result-910-YYYYMMDDHHMMSS.json`
- 包含基本的测试结果，包括安装时间、Pod 就绪时间和资源使用情况

## 测试说明

1. **测试环境要求**：
   - 已安装 Kubernetes 集群
   - 已配置 Helm
   - 测试服务器可以访问互联网（用于下载 Helm 包）

2. **测试时间**：
   - 每台机器的安装和 Pod 就绪检查：约 5 分钟
   - 总测试时间（两台机器）：约 10 分钟（并行运行时约 5 分钟）

3. **测试清理**：
   - 测试完成后，默认不会卸载 Eagle Eye 组件
   - 如果需要自动卸载，可以修改测试代码，取消注释相关卸载代码

## 自定义配置

### 修改 Helm 配置
如果需要修改 Eagle Eye 的 Helm 配置，可以在 `helm_config.yaml` 文件中添加或修改 `eagle-eye` 部分的配置。

### 修改测试参数
可以在 `eagle_eye_performance_test.go` 文件中修改以下参数：
- 轮询间隔时间
- 最大等待时间
- 需要监控的 Pod 列表