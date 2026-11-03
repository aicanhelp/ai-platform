---
title: "openFuyao v25.09发布"
date: 2025-09-30
category: blog
tags:
    - 版本发布
sig: sig-release
archives: 2025-09
author:
    - Release-management Maintainer
summary: "一大波全新特性来袭！Kubernetes全新升级，更有AI推理优化"
---

# openFuyao v25.09发布

2025年09月30日


openFuyao社区致力于构建面向多样化算力集群的开放软件生态，专注于推动云原生与AI原生技术的高效协同，促进有效算力的极致释放。

社区发行版v25.09发布了诸多新特性，并对若干存量特性进行优化，以下介绍具体的特性新增与变动：

## openFuyao Kubernetes全新升级

[SIG-orchestration-engine](https://gitcode.com/openFuyao/sig-orchestration-engine)作为openFuyao社区核心SIG，致力于构建多样化集群的核心容器编排引擎组件。

v25.09版本中，容器编排SIG推出多项Kubernetes增强，全面提升性能、运维能力和可靠性。

### Kubernetes升级：1.28 → 1.33

一年一度的K8s版本焕新。

### 性能优化：kubelet支持服务启动时CPU垂直扩容加速Java程序启动

部分类型的应用在启动时对资源需求更高，超过稳态的limit限制，所以导致启动耗时很长，如Java、大模型推理、科学计算等。该特性能够在保障稳态较少资源占用的前提下，通过启动时CPU扩容加快应用启动速度，加速安装部署、重启、故障恢复等场景Java程序的就绪。

在资源允许的情况下，该特性对测试应用的启动速度提升可超过50%，对于部分特定应用提升可超过100%。

### 运维能力提升：Kubernetes证书管理增强-热加载

Kubernetes/etcd/coredns原生缺乏证书热加载能力，openFuyao Kubernetes为coredns实现了证书热加载能力，从而减少证书轮换对业务的影响。

### 运维能力提升：StatefulSet PVC模板支持扩容

在Kubernetes集群中，存储类基础设施使用StatefulSet类型工作负载，而随着集群业务规模的扩大，工作负载的存储空间也需要进行扩容。原生扩容方式为修改集群中StatefulSet关联的PVC对象，而该特性旨在通过修改StatefulSet规约中PVC的信息，来触发StatefulSet PVC的自动扩容，从而简化运维。

### 可靠性增强：日志轮转支持和可靠性增强

原生kube-log-runner只提供日志重定向能力，fuyao-log-runner对原生机制进行功能扩展，提供字节级精细日志轮转，即使日志磁盘空间不足，也能保障业务正常服务不退出；日志文件被删除时，支持自动恢复保障日志正常输出。

### 性能优化：容器高密部署

在高规格裸机容器场景下，单个节点会部署大量的Pod。K8s原生只支持到每节点100~300Pod，本特性实现了针对容器运行时和kubelet探针的资源消减方案，从而降低容器底座底噪和运行时开销，提升单节点部署密度到1000以上。

### 环境OS兼容性检测

提供全面的环境诊断，无侵入式输出openFuyao在当前操作系统下的兼容性检查结果。

## AI推理优化更进一步

[SIG-ai-inference](https://gitcode.com/openFuyao/sig-ai-inference)旨在构建一个开放、高效、面向未来的云原生LLM推理加速体系，该SIG组孵化中的ai-inference-integration在v25.09正式和大家见面。该特性将提供AI推理场景下的端到端加速解决方案，包含智能路由模块、推理后端模块、全局KV Cache管理模块，实现推理吞吐量提升和时延降低，为AI服务部署提供高效、可靠的技术支撑。

### 端到端AI加速：AI推理集成部署

AI推理集成部署是一个专为云原生环境下AI推理服务优化所设计的端到端集成部署方案。该方案基于Helm Chart将智能路由（hermes-router）、高性能推理后端（Inference Backend）、全局KV Cache管理（cache-indexer）三大AI推理加速模块无缝集成，提供完整的AI推理加速组件的部署链路，从请求接入到推理执行，再到资源管理，实现一站式部署体验。

#### Smart Router智能路由

智能路由选取kv-aware路由策略，与round-robin相比，kv-aware动态权衡KV Cache匹配与VLLM实例负载情况，显著优化了推理性能：
- 平均首token延迟降低56.9%。
- 输出吞吐率提升133.0%。

### 最佳实践集成：AI推理软件套件

AI推理软件套件提供了AI一体机集成解决方案，初步支持基础LLM推理全栈与DeepSeek；真正做到开箱即用，硬件驱动-推理框架-推理模型全链路打通，现提供NPU卡或GPU卡部分型号的驱动支持。

## 存量特性优化

### 在离线混部：Qos保障能力提升

在离线混部v25.09进一步提升了QoS保障能力，通过集成rubik弹性限流、内存异步分级回收等多项能力，在维持30~50%资源利用率提升的同时进一步降低QoS波动。
与此同时，[SIG-colocation](https://gitcode.com/openFuyao/sig-colocation)组织了结构化代码重构，简化仓库架构，提升代码质量。

### Cluster-API：安装方式优化

[SIG-installation](https://gitcode.com/openFuyao/sig-installation)为版本带来全新的安装体验：
- 离线安装包制作更简单：通过配置可选择性附带扩展组件离线安装包。
- 安装规格更轻量：支持引导节点和管理集群共部署，只有1个节点也可以创建集群！

[参考链接](https://docs.openfuyao.cn/docs/%E5%8F%91%E8%A1%8C%E8%AF%B4%E6%98%8E/%E7%89%88%E6%9C%AC%E4%BB%8B%E7%BB%8D)
