# vNPU

## 概述

vNPU项目专注构建容器平台之上的昇腾NPU算力切分和动态调度能力，支持多容器共享设备的同时具备显存和AICore资源的隔离能力，提升资源利用率。通过提供多样化调度策略、多样化算力切分模式、可观测等能力，打造一体化极简易用的昇腾NPU算力接入和纳管方案。

## 关键特性

- 多样化算力切分模式
  - 软切分：基于CANN Runtime API劫持的用户态虚拟化，切分精度更细更灵活
    - 最大1切20，AICore切分粒度最小1%，显存切分粒度最小1Gi
    - AICore和显存隔离。ACore为时分复用，非严格隔离，算力隔离波动约小于整卡的10%。显存严格隔离
    - 支持只限制显存，不限制AICore，每个容器可独立设置
    - 多种AICore调度策略，支持固定配额模式、弹性模式、争抢模式
    - 性能损耗小于5%
  - 硬切分：基于昇腾HDK的虚拟化，将NPU切分为多份vNPU挂载到容器中使用
- 多样化集群调度策略
  - 基于volcano扩展NPU资源调度插件，支持资源共享调度
  - 支持binpack调度
  - 支持整卡调度，同一节点部分容器使用整卡，部分容器共享一张卡
  - [基于DRA的调度](https://gitcode.com/openFuyao/npu-dra-plugin)
- 可观测
  - 支持采集vNPU相关监控指标上报对接Prometheus

## 约束和限制

**软切分约束和限制：**

1. 当前限制一个容器最多只能申请一个vNPU。
2. 当前限制一个容器只能有一个进程可以使用vNPU。
3. 只支持切分和限制AICore及显存容量，不支持AI CPU、VPC、VDEC、JEPGD等核的切分控制。
4. 业务容器不允许为特权容器。

## 快速开始

### 代码结构

- volcano-xpu-plugin：volcano插件，实现NPU虚拟化资源调度。
- xpu-device-plugin：设备插件，实现NPU资源上报和分配。
- client_update：负责软切分劫持库的部署。
- xpu-exporter：负责vNPU指标的上报。
- ci：编译构建脚本。
- charts：vNPU组件部署相关的文件。

### 编译构建&部署

请参考[用户手册](https://gitcode.com/openFuyao/vNPU/blob/master/docs/user-guide.md)。

### POD使用NPU

业务Pod只需声明所需的vNPU资源，无需修改业务代码，vNPU的调度器和device plugin负责自动调度Pod到合适的节点合适的NPU设备上并挂载设备。

例如，Pod使用软切分，声明1个vNPU，50%的AICore，8G显存：

```yaml
spec:
    schedulerName: volcano
    containers:
        resources:
          limits:
            huawei.com/vnpu-number: 1  
            huawei.com/vnpu-cores: 50  
            huawei.com/vnpu-memory.1Gi: 8 
          requests:
            huawei.com/vnpu-number: 1
            huawei.com/vnpu-cores: 50
            huawei.com/vnpu-memory.1Gi: 8
```

使用硬切分，申请50%的AICore：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-mode: "hard"
    huawei.com/vnpu-hard-aicpu-level: "low"
spec:
    schedulerName: volcano
    containers:
        resources:
          limits:
            huawei.com/vnpu-number: 1  
            huawei.com/vnpu-cores: 50  # AI Core百分比
          requests:
            huawei.com/vnpu-number: 1
            huawei.com/vnpu-cores: 50
```

Pod声明使用两张整卡：

```yaml
spec:
    schedulerName: volcano
    containers:
        resources:
          limits:
            huawei.com/vnpu-number: 2 
          requests:
            huawei.com/vnpu-number: 2
```

更多使用方法请参考[用户手册](https://gitcode.com/openFuyao/vNPU/blob/master/docs/user-guide.md)。

## FAQ

**支持的NPU型号**

| 芯片型号 | 软切分 | 硬切分 | HDK版本配套      | CANN版本配套（软切分） |
| -------- | ------ | ------ | ---------------- | ---------------------- |
| 310P3    | 支持   | 支持   | 25.5.0及以上版本 | 8.5.0、9.1.0           |
| 910B4    | 支持   | 支持   | 25.5.0及以上版本 | 8.5.0、9.1.0           |
| 910C     | 不支持 | 不支持 | NA               | NA                     |

说明：以上芯片型号及软件版本配套已经过充分验证，不在此范围内的软硬件未经验证，可能存在兼容性问题。

**支持的volcano版本**

当前支持volcano 1.15.0版本，最新版本待规划。

## 贡献与支持

### 贡献指南

欢迎开发者参与项目优化与迭代，如需贡献代码，请通过以下方式提交：

- 访问GitCode仓库：https://gitcode.com/openFuyao/vNPU.git。
- 提交拉取请求（Pull Request），并附详细的功能说明或问题修复描述。
- 提交问题（Issue），反馈使用过程中遇到的Bug或功能需求。

### 支持渠道

如在部署、使用过程中遇到问题，请在GitCode项目页面打开Issue，我们会及时响应并提供技术支持。 感谢您的使用与反馈，助力vNPU项目持续优化！
