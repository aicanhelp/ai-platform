<div align="center">
  <h1 align="center">
      eagle-eye
  </h1>

  <p><b>Observability System for AI Inference Scenarios</b></p>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_eagle_eye/user_guide/eagle_eye_for_ai_inference.md)
[![License](https://img.shields.io/badge/License-Mulan_PSL_v2-blue.svg)](./LICENSE)
[![Helm](https://img.shields.io/badge/Helm-chart-00a1d6.svg)](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_eagle_eye/user_guide/eagle_eye_for_ai_inference.md#%E5%BC%80%E5%A7%8B%E5%AE%89%E8%A3%85)

</div>

<hr>

## Updates

- [26-09] Added end-to-end distributed tracing capability based on OpenTelemetry, supporting non-intrusive request tracing from AI gateway to vLLM-Ascend inference engine; fully supports vLLM-Ascend 0.18.0, adaptation for 0.23.0 is in progress.

## Overview

Eagle Eye is an observability system for AI inference scenarios, providing full-link metric collection, near real-time transmission, intelligent diagnosis, and end-to-end distributed tracing capabilities.

In terms of metric monitoring, it integrates Prometheus's periodic metric collection with the low-latency push mechanism of distributed message queue systems, achieving comprehensive observability from AI gateway, inference engine, Mooncake to infrastructure (Ray, K8s, hardware). It supports both trend analysis for scaling decisions and meets the second-level data update requirements of modules with high time-sensitivity (such as intelligent routing). Through an independent hardware health diagnosis module, it achieves continuous monitoring and anomaly identification of underlying metrics such as NPU/GPU, temperature, power consumption, error codes, etc., building a closed-loop monitoring capability of "collection-transmission-diagnosis-evaluation".

In terms of distributed tracing, it provides non-intrusive end-to-end distributed tracing based on OpenTelemetry, tracking the complete execution path of inference requests from the AI gateway to the inference engine (vLLM-Ascend), recording latency and key business attributes at each stage, supporting performance tuning and fault diagnosis.

**Core Functions**

- Multi-layer metric coverage: Covers AI gateway (such as performance, resource consumption, security and compliance audit, governance policy execution tracking), inference engine (API Server, model input/output, inference process, inference engine status), Mooncake (Mooncake master, transfer engine, Mooncake client) and infrastructure (Ray, K8S, hardware), achieving full-link observability.

- Near real-time metric transmission: For modules with high time-sensitivity requirements, achieves second-level metric push through distributed message queue systems, ensuring metrics can be timely perceived and affect decisions.

- Scaling decision support: Synchronously reports collected system and runtime metrics to Prometheus for periodic calculation and trend evaluation.

- Hardware health check and diagnosis: Builds an independent hardware health diagnosis module that periodically collects underlying metrics such as NPU/GPU temperature, power consumption, error codes, etc., and reports them in real-time through distributed message queue systems. The diagnosis module subscribes to and analyzes collected data, combines device model, driver and firmware information, identifies typical failure patterns based on threshold rules and abnormal metric analysis, and outputs diagnosis conclusions and handling recommendations, achieving a closed loop from data collection to health evaluation.

- End-to-end distributed tracing: Based on the OpenTelemetry standard, provides non-intrusive request-level tracing from AI gateway to inference engine (vLLM-Ascend), supporting collection, storage and querying, recording key performance metrics and business attributes.

## Quick Start

### Prerequisites

#### Hardware Requirements

Eagle Eye has no special requirements for the hardware environment.

#### Software Requirements

Kubernetes v1.33.0 or above.

Inference engine (such as vLLM-Ascend 0.18.0)

### Binary Deployment

1. Pull the project installation package.
    ```bash
    helm pull oci://cr.openfuyao.cn/charts/eagle-eye --version xxx
    ```
   Where `xxx` needs to be replaced with the specific project installation package version, such as `0.0.0-latest`. The pulled installation package is in compressed form.

2. Extract the installation package.
    ```bash
    tar -xzvf eagle-eye-xxx.tgz
    ```
   Where `xxx` needs to be replaced with the specific project installation package version, such as `0.0.0-latest`.

3. Install and deploy.

   Using namespace `eagle-eye` and release name `eagle-eye` as an example, execute the following command in the same directory as `eagle-eye`:
   ```bash
   helm install -n eagle-eye eagle-eye ./eagle-eye
   ```

### Source Deployment

1. Clone the project from the repository.

   ```bash
   git clone https://gitcode.com/openFuyao/eagle-eye.git
   ```

2. Install and deploy.

   Using namespace `eagle-eye` and release name `eagle-eye` as an example, execute the following command in the same directory as `eagle-eye`:
   ```bash
   cd eagle-eye/charts/eagle-eye
   helm install -n eagle-eye eagle-eye .
   ```

### Configuration
For distributed tracing usage, please refer to the [AI Inference Eagle Eye](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_eagle_eye/user_guide/eagle_eye_for_ai_inference.md#%E4%BD%BF%E7%94%A8%E9%93%BE%E8%B7%AF%E8%BF%BD%E8%B8%AA) chapter in the eagle-eye user manual.
