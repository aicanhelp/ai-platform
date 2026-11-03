<div align="center">
  <h1 align="center">
      eagle-eye
  </h1>

  <p><b>面向AI推理场景的可观测体系</b></p>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_eagle_eye/user_guide/eagle_eye_for_ai_inference.md)
[![License](https://img.shields.io/badge/License-Mulan_PSL_v2-blue.svg)](./LICENSE)
[![Helm](https://img.shields.io/badge/Helm-chart-00a1d6.svg)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_eagle_eye/user_guide/eagle_eye_for_ai_inference.md#%E5%BC%80%E5%A7%8B%E5%AE%89%E8%A3%85)

</div>

<hr>

## Updates

- [26-09] 新增基于OpenTelemetry的端到端分布式链路追踪能力，支持从AI网关到vLLM-Ascend推理引擎的非侵入式请求追踪；已支持vLLM-Ascend 0.18.0，0.23.0尚在适配中。

## Overview

Eagle Eye是面向AI推理场景的可观测体系，提供全链路指标采集、近实时传输、智能诊断与端到端分布式链路追踪能力。

指标监控方面，融合了Prometheus的周期性指标采集与分布式消息队列系统的低延迟推送机制，实现从AI网关、推理引擎、Mooncake到基础设施（Ray、K8s、硬件）的全方位可观测，既支持扩缩容决策的趋势分析，也满足时效性要求高的模块（如智能路由）对秒级数据更新的需求。通过独立的硬件健康诊断模块，实现对NPU/GPU、温度、功耗、错误码等底层指标的持续监测与异常识别，构建"采集—传输—诊断—评估"的闭环监控能力。

链路追踪方面，基于OpenTelemetry提供非侵入式的端到端链路追踪能力，追踪推理请求从AI网关到推理引擎（vLLM-Ascend）的完整执行路径，记录各环节耗时和关键业务属性，支持性能调优和故障诊断。

**核心功能**

- 多层指标覆盖：覆盖AI网关（如性能、资源消耗、安全与合规审计、治理策略执行记录）、推理引擎（API Server、模型输入输出、推理过程、推理引擎状态）、Mooncake（Mooncake master、transfer engine、Mooncake client）和基础设施（Ray、K8s、硬件），实现全链路观测。

- 近实时指标传输：面向对时效性要求高的模块，通过分布式消息队列系统实现秒级指标推送，确保指标能够被及时感知并影响决策。

- 扩缩容决策支撑：将采集到的系统与运行态指标同步上报至Prometheus，用于周期性计算与趋势评估。

- 硬件健康检查与诊断：构建独立的硬件健康诊断模块，周期性采集NPU/GPU温度、功耗、错误码等底层指标，并通过分布式消息队列系统实时上报。诊断模块订阅并分析采集数据，结合设备型号、驱动与固件信息，基于阈值规则与异常指标分析，识别典型故障模式并输出诊断结论与处置建议，实现从数据采集到健康评估的闭环。

- 端到端分布式链路追踪：基于OpenTelemetry标准，提供从AI网关到推理引擎（vLLM-Ascend）的非侵入式请求级追踪能力，支持采集、存储与查询，记录关键性能指标和业务属性。

## Quick Start

### Prerequisites

#### 硬件要求

Eagle Eye本身对硬件环境无特殊要求。

#### 软件要求

Kubernetes v1.33.0及以上版本。

推理引擎（如vLLM-Ascend 0.18.0）

### Binary Deployment

1. 拉取项目安装包。
    ```bash
    helm pull oci://cr.openfuyao.cn/charts/eagle-eye --version xxx
    ```
   其中`xxx`需替换为具体项目安装包版本，如`0.0.0-latest`。拉取得到的安装包为压缩包形式。

2. 解压安装包。
    ```bash
    tar -xzvf eagle-eye-xxx.tgz
    ```
   其中`xxx`需替换为具体项目安装包版本，如`0.0.0-latest`。

3. 安装部署。

   以命名空间`eagle-eye`、release名称`eagle-eye`为例，在`eagle-eye`同级目录下执行如下命令：
   ```bash
   helm install -n eagle-eye eagle-eye ./eagle-eye
   ```

### Source Deployment

1. 从仓库拉取项目。

   ```bash
   git clone https://gitcode.com/openFuyao/eagle-eye.git
   ```

2. 安装部署。

   以命名空间`eagle-eye`、release名称`eagle-eye`为例，在`eagle-eye`同级目录下执行如下命令：
   ```bash
   cd eagle-eye/charts/eagle-eye
   helm install -n eagle-eye eagle-eye .
   ```

### Configuration
使用链路追踪请参考eagle-eye用户手册的[AI推理鹰眼](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_eagle_eye/user_guide/eagle_eye_for_ai_inference.md#%E4%BD%BF%E7%94%A8%E9%93%BE%E8%B7%AF%E8%BF%BD%E8%B8%AA)章节。
