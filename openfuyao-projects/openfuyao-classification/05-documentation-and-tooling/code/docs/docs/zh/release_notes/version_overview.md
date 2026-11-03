---
sidebar_position: 4
---

# 版本介绍

## 组件变更说明

**表1**  openFuyao v26.09组件变更说明

| **组件名（Helm Chart包名）** | 组件变更类型<br>（新增、增强、修复、变更、废弃、移除）<br> | **变更说明** | **SIG** |
| --- | --- | --- | --- |
| hermes-router | 增强 | 新增cacheIndexer.blockSize配置参数，供kv-cache-aware等策略统一配置缓存块大小下发。修复前缀缓存统计问题：仅统计完整Block，避免与cache-indexer语义不一致导致的评分偏差。 | sig-ai-inference |
| cache-indexer | 增强 | 新增Decode节点KV Cache感知能力，将decode Pod纳入L1索引发现链路；L1 kvevent订阅兼容vLLM 0.28的map编码事件，避免缓存索引缺失。 | sig-ai-inference |
| infernex | 增强 | 1、infernex增强。新增vLLM引擎PodMonitor指标采集，通过Prometheus采集prefill/decode推理实例的引擎指标。支持traceparent请求头透传，打通eagle-eye分布式链路追踪，监控与追踪组件可独立开关。<br/>2、infernex-bridge增强。对齐KServe调度器契约，移除scheduler配置webhook补丁，精简KServe部署示例并兼容restricted PSA策略。<br/>3、infernex-checker增强。新增可选网络连通性检查开关，msprof基准测试增加HCCL通信数据采集，辅助慢卡定位。 | sig-ai-inference |
| opensandbox | 新增 | 新增opensandbox组件，兼容opensandbox API，提供fluxsandbox provide路由请求到高性能沙箱调度引擎。 | sig-agent-sandbox |
| fluxsandbox | 新增 | 新增sandboxsched组件，高性能Kubernetes沙箱调度引擎，作为OpenSandbox的runtime后端，面向AI Agent工作负载提供低时延、高吞吐的沙箱生命周期管理能力。<br/>SandboxSched不直接面向用户暴露API，而是通过OpenSandbox Server的flux runtime接入：用户使用OpenSandbox SDK创建沙箱，由Server通过gRPC转发给fluxSandbox controller完成调度与生命周期管理。 | sig-agent-sandbox |
| npu-dra-plugin | 增强 | 支持按资源容量（core、显存）、按设备ID、按设备类型多种调度方式，同时支持整卡、硬切分、软切分多种调度模式，支持节点内拓扑亲和调度，提升推理性能。 | sig-orchestration-engine |
| eagle-eye | 增强 | 新增面向vLLM-Ascend的无侵入式分布式链路追踪能力；修复chart无条件创建命名空间问题，仅在启用时创建。 | sig-ai-inference |
| kubevirt | 增强 | 新增支持kubevirt在鲲鹏环境下的高阶能力，包括CDI对接和memorydump能力，以及支持自定义参数安装部署能力。 | sig-orchestration-engine |
| ubs-k8s-enable | 增强 | 修复并发更新和清理场景下的冲突问题，升级适配底层UBS相关SDK版本适配，增强可用性；支持超节点的可观测能力，上报超节点拓扑、内存借用、内存共享、URMA设备等相关指标，支持超节点的拓扑感知调度能力。 | sig-orchestration-engine |
| superpod-exporter | 新增 | 新增超节点可观测，支持超节点的可观测能力，上报超节点拓扑、内存借用、内存共享、URMA设备等相关指标。 | sig-orchestration-engine |
| compliance-operator | 增强 | 新增基于Compliance Operator的安全加固与回退工具，将扫描结果中直接修改对集群无直接影响的FAIL项进行加固，帮助用户在使用安装部署工具安装了Kubernetes后按需进行安全整改。 | sig-installation |
| cluster-api-provider-bke | 增强 | 1. 支持安装时worker节点为KubeOS不可变OS。<br/>2. 重构DAG升级框架，支持Yaml和Helm两种组件升级接入。<br/>3. 完成并发安全、安装状态归一、配置文档梳理、日志可观测性、性能、异常场景等一系列加固，提升安装部署性能、可靠性。<br/> | sig-installation |
| bke-manifests | 增强 | 支持版本组件yaml变更适配。 | sig-installation |
| bkeadm | 增强 | 1. 支持安装时worker节点为KubeOS不可变OS。<br/>2.bkeadm工具加固，配置文档梳理、日志可观测性加固，提升安装部署可靠性，易用性。<br/>3.前置检查能力归一，支持bke preflight引导节点初始化以及集群创建的前置检查校验。 | sig-installation |
| release-image | 增强 | 修改componentversion字段，用以支持Yaml和Helm升级，增加版本内容。 | sig-installation |
| upgrade-path | 增强 | 增加版本升级路径，支持按社区发布包的组件清单和依赖关系，自动编排、校验并执行升级。 | sig-installation |
| weight-dispatcher | 增强 | 重构RDMA数据传输代码实现，优化RDMA链路数据传输性能。 | sig-ai-inference |
| npu-operator | 增强 | 新增组件VNPU与NPU DRA plugin的一键式安装、管理、卸载能力，使能集群接入不同资源申请与切分能力。 |  sig-orchestration-engine |

## 接口变更说明

无

## 版本特性说明

openFuyao v26.09主要功能如[表2](#table-container-platform)和[表3](#table-independent-component)所示，功能特性的具体信息请参见[《安装部署》](https://gitcode.com/openFuyao/sig-installation/blob/master/docs/zh/cluster_installation_guide/read_before_installation/installation_process.md)、[《管理面操作》](../management_plane_operations/registration_and_login.md)和[《特性用户指南》](../user_guide/kae_operator.md)。

**表2** 容器平台<a id="table-container-platform" href="#"></a>

| **特性名** | **中文简介** | **特性介绍** | **SIG** |
| --- | --- | --- | --- |
| 安装部署 | 安装部署 | 对接标准Cluster-API的安装部署工具，支持一键式安装业务集群，管理集群在统一管理面上提供多场景交互式业务集群生命周期管理能力，包括单/多节点安装（含高可用）、在/离线安装、集群扩缩容、Kubernetes原地升级等。 | sig-installation |
| 容器编排 | 容器编排 | 提供openFuyao Kubernetes，兼容K8s 1.34，提供高密部署、启动加速、日志增强、证书管理增强等增强功能。 | sig-orchestration-engine |
| 管理面 | 管理面 | 提供开箱即用的控制台，支持应用管理、应用市场、扩展组件管理、资源管理、仓库管理、监控、告警、用户管理、命令行交互等功能。 | sig-container-platform |

**表3** 独立组件<a id="table-independent-component" href="#"></a>

| **特性名** | **中文简介** | **特性介绍** | **SIG** |
| --- | --- | --- | --- |
| infernex | AI推理加速套件 | InferNex是面向云原生场景的AI推理端到端集成部署方案，通过Helm Chart将开源网关、智能路由、高性能推理后端、KVCache索引管理、扩缩容决策框架及推理可观测体系等核心组件无缝集成，提供从请求接入、动态路由、推理执行到资源管理与监控的完整加速链路，旨在提升推理吞吐量并降低推理时延，实现一站式的高效AI服务部署与使用体验。 | sig-ai-inference |
| eagle-eye | AI推理可观测 | Eagle Eye是面向AI推理场景的可观测体系，实现从AI网关、推理引擎、Mooncake到基础设施（Ray、K8s、硬件）的全链路指标采集、近实时传输与智能诊断。该体系融合了Prometheus的周期性指标采集与分布式消息队列系统的低延迟推送机制，既支持扩缩容决策的趋势分析，也满足时效性要求高的模块（如智能路由）对秒级数据更新的需求。通过独立的硬件健康诊断模块，实现对NPU/GPU、温度、功耗、错误码等底层指标的持续监测与异常识别，构建"采集—传输—诊断—评估"的闭环监控能力，为AI推理系统的稳定性、性能优化与资源调度提供坚实的数据支撑。 | sig-ai-inference |
| hermes-router | AI推理智能路由 | Hermes-router是基于K8s GIE框架的AI推理智能路由组件，提供KVCache aware、时延预测等多种路由策略，用于接收推理请求并转发至最佳推理服务后端，帮助用户在多种场景下提升AI推理性能、集群资源利用率与服务稳定性。 | sig-ai-inference |
| cache-indexer | AI推理KVCache索引管理 | cache-indexer是面向AI推理场景的全局KVCache索引管理组件，统一维护推理实例本地HBM与内存两层KVCache视图，支持智能路由查询候选推理实例的KVCache命中率，进行更准确的请求级调度。 | sig-ai-inference |
| pd-orchestrator | 弹性调度组件集 | 面向Kubernetes的弹性调度组件集，包含3个核心组件：Elastic Scaler提供通用扩缩容决策能力，ResourceScalingGroup提供工作负载分组扩缩与资源编排能力，Tidal提供基于时间规则的潮汐扩缩能力。 | sig-ai-inference |
| weight-dispatcher | 模型权重分发 | Weight-Dispatcher是面向AI推理场景的模型权重预热与分发组件，用于在推理实例启动前，将模型权重从源节点或外部模型仓库分发到目标推理节点的本地缓存目录，减少实例启动阶段反复下载大模型权重带来的等待时间和带宽压力。 | sig-ai-inference |
| kae-operator | KAE接入使能 | 实现分钟级鲲鹏KAE硬件的自动化管理能力，包含KAE硬件特征发现，驱动、固件、硬件设备插件等部件的自动化管理与安装能力，五分钟内可完成KAE部署到可用。 | sig-orchestration-engine |
| logging-package | 日志组件 | 日志作为扩展组件可查看Pod、容器的日志信息，并且支持配置日志的采集源、采集任务及告警规则。openFuyao日志系统能够有效提升平台识别错误的效率，增强整个平台的监控能力。 | sig-container-platform |
| colocation-package | 在离线混部调度增强 | 支持在线/离线业务混合部署，保障在线业务在使用高峰时期的调度与对离线业务的压制，同时使能离线业务在在线业务低谷时期使用超卖资源提升集群的资源利用率，利用率提高30%~50%，且QoS无明显影响，抖动低于5%。 | sig-orchestration-engine |
| many-core-orchestrator | 众核混部调度 | 面向众核混部集群，提供宿主机干扰指标采集、节点干扰分析、Volcano干扰感知调度，以及可选的Kata Containers虚拟机级隔离能力。目标是降低在线业务在I/O、LLC缓存和内存压力混部场景下的长尾延迟抖动，并提高节点安全混部密度。 | sig-orchestration-engine |
| matrixagent | 内存借用节点侧功能 | matrixagent定时采集节点内存使用率上报到matrixcontroller，同时侦听逃生决策或内存借用，调用VirtAgent进行内存借用，平衡集群内存使用，提升资源利用率。 | sig-ub-enable |
| matrixcontroller | 内存借用中心侧功能 | matrixcontroller侦听matrixagent上报数据，计算内存水位线，根据阈值自动发起内存借用请求，平衡集群内存使用，提升资源利用率。 | sig-ub-enable |
| matrixshm | 共享内存CSI | 为Pod提供了基于CR声明式隔离的内存共享方案，仅支持全量映射与解映射，容量受配置块大小和NUMA借出比例双因子约束，Pod调度受节点标签共享域控制。其生命周期管理强绑定于Pod退出，通过shm-agent侦听、引用计数、操作失败回滚及启动状态恢复等多层机制防止资源泄漏，并配合路径校验、权限严控、原子写缓存和任务队列过载保护来确保安全与高可用。 | sig-ub-enable |
| superpod-exporter | 超节点可观测 | 支持超节点的可观测能力，上报超节点拓扑、内存借用、内存共享、URMA设备等相关指标。 | sig-orchestration-engine |
| monitoring-dashboard | 自定义监控看板 | 用于呈现可拓展指标，满足企业对高度可定制与可拓展的监控系统的需求。 | sig-container-platform |
| multi-cluster-service | 多集群管理 | 提供集群列表界面与全生命周期管理功能，支持集群的纳管、扩缩容、配置更新及安全销毁。用户可借助集群凭证实现快速访问，并通过标签管理灵活编辑集群标签。该组件实现跨集群安全访问，助力在混合环境中高效统一管理多个Kubernetes集群。 | sig-container-platform |
| npu-operator | NPU接入使能 | NPU Operator使用Kubernetes中的Operator Framework来自动管理配置昇腾设备所需的所有昇腾驱动程序和固件，使能构建集群全流程运行，支持集群作业调度、运维监控、故障恢复等功能。通过安装相应的组件，可以实现NPU资源管理、工作负载的优化调度、训推任务的容器化支持，使能AI作业以容器形式部署并运行在NPU设备上。 | sig-orchestration-engine |
| numa-affinity-package | numa亲和调度 | 在集群级以及节点级实现硬件NUMA的拓扑感知，并基于NUMA亲和性对应用进行NUMA亲和调度，提升应用性能。 | sig-container-platform |
| ray-package | openFuyao/ray | 源自于ray-project/ray的“平行宇宙”项目，用于扩展AI和Python应用程序的统一分布式计算框架，致力于面向Ray的使用者、运维者，提供稳定可靠、国产硬件亲和的LTS (Long-Term Support)版本。 | sig-distributed-framework |
| compliance-operator | 合规扫描 | Compliance Operator是一个基于Kubernetes Operator模式的自动化合规扫描工具，支持使用CIS Benchmark和DISA STIG标准对Kubernetes集群进行安全合规性扫描。 | sig-installation |
| opensandbox | 沙箱接入 | 兼容opensandbox API，提供fluxsandbox provide路由请求到高性能沙箱调度引擎。 | sig-agent-sandbox |
| fluxsandbox | 高性能沙箱调度引擎 | 新增sandboxsched组件，高性能Kubernetes沙箱调度引擎，作为OpenSandbox的runtime后端，面向AI Agent工作负载提供低时延、高吞吐的沙箱生命周期管理能力。<br/>SandboxSched不直接面向用户暴露API，而是通过OpenSandbox Server的flux runtime接入：用户使用OpenSandbox SDK创建沙箱，由Server通过gRPC转发给fluxSandbox controller完成调度与生命周期管理。 | sig-agent-sandbox |
| vNPU | NPU虚拟化 | 通过昇腾docker runtime实现NPU软硬切分，支持310P，910B，910C卡型号，提升集群算力资源利用率。 | sig-orchestration-engine |
| npu-dra-plugin | 亲和昇腾NPU的DRA插件 | 基于kubernetes的DRA机制，支持310P，910B，910C卡型号；支持按资源容量（core、显存）、按设备ID、按设备类型多种调度方式，同时支持整卡、硬切分、软切分多种调度模式，支持节点内拓扑亲和调度，提升推理性能。 | sig-orchestration-engine |
| kubevirt | 容器虚机共管组件 | kubevirt在鲲鹏环境下的虚机容器共管能力，支撑vm全生命周期管理，支持macvlan、支持SR-IOV、支持SR-IOV实时迁移、支持网卡热插拔能力、支持虚拟机热迁移、支持内存导出。验证当前kubevirt不支持鲲鹏UEFI secure boot启动。支持自定义参数安装部署。 | sig-orchestration-engine |
