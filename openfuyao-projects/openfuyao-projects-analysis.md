# openFuyao 开源项目代码分析文档

> 文档生成时间：2026-11-02  
> 分析目录：`/udata/workspace/github-aican/ai-platform/openfuyao-projects`  
> 说明：本目录为 openFuyao 庞大代码仓库的本地聚合/工作区，包含约 80+ 个组件/子仓库；下文按 SIG 与功能域做系统梳理，涵盖项目定位、技术栈、核心能力、目录结构、对外接口与关键依赖。

---

## 1. 项目总体概览

### 1.1 项目定位

`openFuyao` 是一个面向 AI/高性能计算场景的 **云原生容器平台 / 私有云发行版**，基于 Kubernetes 做了全栈增强与扩展。平台核心能力包括：

- 容器编排底座：Kubernetes、containerd、runc、etcd、CoreDNS 的维护/适配版本；
- AI 训练与分布式计算：Ray 及其生命周期管理；
- AI 大模型推理服务化：InferNex、Hermes Router、Mooncake KV Cache 分离、vLLM 昇腾适配；
- 异构算力管理：昇腾 NPU 全栈运维（驱动、固件、Runtime、Device Plugin、vNPU、DRA、调度器）；
- 在离线混部与资源超卖：Rubik / Volcano 协同的 colocation 方案；
- NUMA 亲和调度：Volcano 拓扑感知 + kubelet 配置联动；
- 多集群/大规模集群管理：Karmada 多集群算力池、KWOK 仿真、超大规模集群最佳实践；
- 管理面与可观测：console、monitoring、logging、web-terminal、oauth、marketplace、plugin 等微服务。

### 1.2 仓库组织方式

当前目录下并不是单一 git 仓库，而是按 **SIG（特别兴趣组）+ 功能域** 拆分的多个子仓库/组件。常见命名约定：

| 命名后缀 | 含义 |
|---|---|
| `-service` | 后端微服务（Go） |
| `-website` | 前端用户界面（React/OpenInula，微前端插件） |
| `-operator` | Kubernetes Operator（Go，controller-runtime/kubebuilder） |
| `-controller` | 控制器/调度器组件 |
| `-device-plugin` / `feature-discovery` | K8s 设备插件/节点特征发现 |
| `-installer` | 节点级二进制/容器化安装器 |
| `sig-*` | 社区治理、设计文档、路线图 |

### 1.3 核心 SIG 与对应组件

| SIG | 负责方向 | 代表性组件 |
|---|---|---|
| sig-Installation | 集群安装部署、生命周期管理 | bkeadm, bkeagent, cluster-api-provider-bke, installer, installer-service, bke-manifests, openfuyao-system-controller |
| sig-Container-Platform | 容器平台管理面、认证、监控 | console-service/website, oauth-server/webhook/proxy, monitoring-service, marketplace-service, plugin-management-service, logging-operator/website |
| sig-orchestration-engine | 容器编排引擎、运行时、CNI/CSI | kubernetes, containerd, runc, etcd, coredns, node-checker |
| sig-Large-Scale-Cluster | 超大规模集群、批量调度、自动化部署 | volcano-ext, clusters-computing-operator, kwok, mind-cluster |
| sig-AI-Inference | AI 推理服务化 | InferNex, hermes-router, mooncake, cache-indexer, eagle-eye, vllm-ascend, mis-management |
| sig-Distributed | 分布式训练/计算框架 | ray, ray-service, ray-website |
| sig-Colocation | 在离线混部 | colocation-management/service/website |
| sig-numa-affinity | NUMA 亲和调度 | volcano-config-service/website, 火山 NUMA 扩展 |
| sig-ub-enable | UB 超节点 AI 云原生使能 | ub-network-device-plugin, ub-network-bandwidth-operator, ub-enable-quickstart |
| sig-qa / 其他 | 质量保证、社区治理、文档 | e2e-auto-test, qa-tool, community, docs, ofep, release-management |

---

## 2. 基础设施与安装部署

### 2.1 组件清单

| 项目 | 定位 | 技术栈 | 主要产物/接口 | 关键依赖 |
|---|---|---|---|---|
| **bkeadm** | 集群生命周期 CLI（创建、扩缩容、升级、重置） | Go 1.25, Cobra, Helm, netlink | 二进制 `bkeadm` | cluster-api-provider-bke |
| **bkeagent** | 节点级 in-cluster 代理，执行 `Command` CR | controller-runtime, Go | 镜像：manager/launcher/deployer | bkecommon |
| **bkecommon** | 安装相关共享 Go 库 | Go | 复用包 | 无 |
| **bke-console-website** | 集群生命周期前端 | OpenInula / React, Vite, Ant Design | Helm chart | installer-service |
| **bke-manifests** | 各版本 Deployment YAML、init 脚本、RPM 清单 | Shell, Makefile, YAML | 静态文件 | 无 |
| **cluster-api-provider-bke** | 集群 API Provider / Operator | Go 1.24, controller-runtime | CRD: `BKECluster`, `BKENode`, `ClusterVersion`, `UpgradePath` 等 | K8s Cluster API |
| **installer** | 管理面/业务集群命令行安装器 | Go, Cobra | 二进制 `installer` | 无 |
| **installer-service** | 可视化安装后端服务 | Go 1.24, go-restful, controller-runtime | Helm chart，REST API | cluster-api-provider-bke |
| **installer-website** | 可视化安装前端 | OpenInula / React, Vite | Helm chart，微前端 | installer-service |
| **openfuyao-system-controller** | 在 K8s 集群上快速部署/卸载 openFuyao 管理面 | Shell, YAML | 部署脚本/YAML | 无 |
| **env-check** | 安装前环境检测工具 | Go | `env-check` 二进制/配置 | 无 |
| **node-checker** | 节点 OS 一致性/就绪检查 | Go | `node-checker` 服务 | 无 |
| **dependency-database** | 组件版本/依赖元数据数据库 | Go | `dependency-database` 服务 | 无 |
| **release-management** | 版本发布定义、组件清单、规则白皮书 | Markdown, drawio | `component-list.md`, `openFuyao-arch.drawio.svg` | 无 |

### 2.2 关键流程

- `bkeadm` → 调用 `cluster-api-provider-bke` CRD → `bkeagent` 在目标节点上执行命令 → 拉取 `bke-manifests` 中的 YAML/脚本 → 完成 K8s 与管理面部署。
- `installer-service` 提供 Web UI 驱动的替代入口，与 `installer-website` 配合。
- `openfuyao-system-controller` 提供一键式管理面安装/卸载能力。

### 2.3 关键目录结构说明

- `bkeadm/{cmd,pkg,configs,examples}`：命令、业务逻辑、预检配置、用例。
- `bkeagent/{api/v1beta1,controllers,launcher,deployer,config}`：Operator API、控制器、节点执行器。
- `cluster-api-provider-bke/{api/v1alpha1,controllers,config/crd}`：CRD 与调谐器。

---

## 3. 管理面与平台服务

### 3.1 平台后端服务

| 项目 | 定位 | 技术栈 | 关键能力 | 对外接口/产物 |
|---|---|---|---|---|
| **console-service** | 容器平台管理面后端 | Go 1.24.5, go-restful, JWT, WebSocket | 集群/资源/应用/告警等管理 | REST API, Helm chart |
| **application-management-service** | 应用及服务生命周期管理 | Go 1.24.5, gRPC, go-restful, Helm | 应用发布、Helm 部署、审计、日志 | REST/gRPC, Helm chart |
| **monitoring-service** | 监控与告警服务 | Go 1.24.5, Prometheus client, go-restful | 指标查询、Targets、告警规则 | REST API, Helm chart |
| **monitoring-dashboard-service** | 监控大盘后端 | Go 1.24.5, K8s client | 大盘配置、数据源 | REST API, Helm chart |
| **logging-operator** | 日志采集管理 Operator | Go 1.24.5, controller-runtime | 日志采集器 CRD、配置管理 | CRD, Helm chart |
| **multi-cluster-service** | 多集群纳管后端 | Go 1.24.5, Karmada 1.12, controller-runtime | 集群纳管/解纳管、集群标签、跨集群访问 | CRD, REST API, Helm chart |
| **marketplace-service** | 应用市场后端 | Go 1.24.5, gRPC, go-restful | 应用商店、Chart 管理、灰度发布 | REST/gRPC, Helm chart |
| **plugin-management-service** | 扩展插件管理后端 | Go 1.24.5, go-restful | 插件生命周期、安装/卸载/升级 | REST API, Helm chart |
| **web-terminal-service** | 容器集群 Web 终端服务 | Go 1.24.5, controller-runtime | Web 终端模板、Pod 终端 | CRD, REST API |
| **oauth-server** | 内置认证服务器 | Go 1.24.5, gorilla, JWT | 登录、Token、密码策略 | REST API |
| **oauth-webhook** | kube-apiserver 认证 Webhook | Go 1.24.5, Cobra | 校验 access_token | K8s Authentication Webhook |
| **oauth-proxy** | 扩展组件认证鉴权边车 | Go 1.25, OAuth2, K8s clients | 边车代理认证 | 容器镜像 |

### 3.2 平台前端项目

| 项目 | 技术栈 | 接入方式 |
|---|---|---|
| console-website | OpenInula / React, Vite, Ant Design, xterm, CodeMirror | 平台总入口 |
| monitoring-dashboard-website | OpenInula, ECharts, Ant Design | 微前端插件 |
| multi-cluster-website | OpenInula, CodeMirror, Ant Design | 微前端插件 |
| logging-website | OpenInula, Ant Design, js-yaml | 微前端插件 |
| installer-website | OpenInula, CodeMirror, Ant Design | 安装引导界面 |
| bke-console-website | OpenInula, Ant Design, js-yaml | 集群生命周期引导前端 |
| colocation-website | OpenInula, ECharts, Ant Design | 在离线混部插件 |
| volcano-config-website | React / OpenInula, ECharts, Ant Design, qiankun | NUMA 监控配置插件 |
| ray-website | React / OpenInula, CodeMirror, Ant Design | Ray 管理插件 |

### 3.3 共同技术特征

- 后端统一采用 **Go 1.24.5** 为主体版本，使用 `go-restful/v3`（v2 用于部分项目）、`zap` 日志、`viper` 配置、`ginkgo/gomega` 做单元/集成测试。
- 大量服务直接以 **Helm chart** 形式发布，chart 通常放在 `<项目>/charts/` 下。
- 前端统一采用 **OpenInula**（openFuyao 版 React 兼容框架）或原生 React + **Vite + Ant Design**，多作为 **qiankun 微前端子应用** 嵌入到 `console-website`。
- 认证统一走 `oauth-server` 签发 JWT，`oauth-webhook` 注入到 `kube-apiserver` 做认证后端，`oauth-proxy` 给扩展组件做边车鉴权。

---

## 4. AI 训练、推理与分布式框架

### 4.1 分布式框架

| 项目 | 定位 | 技术栈 | 核心能力 | 产物/接口 |
|---|---|---|---|---|
| **ray** | openFuyao 维护的 Ray 分支（基于 2.48.0） | Python/C++/Java, Bazel | 昇腾 310duo 拓扑感知、节点常驻稳定性、Ray Serve/LLM、TimeTable Dashboard | Python wheel / CLI / Dashboard |
| **ray-service** | K8s 上 Ray 集群/作业管理后端 | Go 1.24.5, KubeRay, go-restful, GORM+SQLite | RayCluster/RayJob/RayService 生命周期、模板、告警、多租户 | REST API, Helm chart |
| **ray-website** | Ray 管理前端 | React / OpenInula, Vite, Ant Design | 集群/作业/服务可视化、日志/监控 | 微前端插件 |

### 4.2 AI 推理服务化

| 项目 | 定位 | 技术栈 | 关键能力 | 产物/接口 |
|---|---|---|---|---|
| **InferNex** | 端到端 AI 推理部署框架 | Go/Python, Gateway API, LWS | Bridge 网关、PD-Orchestrator 预填-解码分离、cache-indexer、proxy-server、eagle-eye | Helm chart `infernex` |
| **hermes-router** | 智能推理路由 | Go 1.25, gateway-api-inference-extension, controller-runtime | KV-aware 路由、资源/负载感知、PD 分离支持 | 容器镜像 / EPP |
| **developer-experience** | Hermes Router 引擎 | Go 1.24.5, gateway-api-inference-extension | scorer/picker 插件、多种路由策略 | Helm chart `inferencepool` |
| **cache-indexer** | 分布式 KV Cache 全局索引 | Go 1.24, ZeroMQ, msgpack, CBOR | KV Cache 索引、一致性哈希/路由 | 容器镜像 / Go 库 |
| **mooncake** | 面向 LLM 推理的分离式 KV Cache 架构 | C++/Python, RDMA/TCP, gRPC | transfer-engine、store、ep、p2p-store、vLLM/SGLang/lmdeploy 集成 | Python wheel `mooncake-transfer-engine` |
| **eagle-eye** | 推理场景近实时可观测系统 | Go | 请求追踪、Token 级监控、推理性能观测 | 可观测后端 |

### 4.3 关键集成思路

- **流量入口** → `InferNex-Bridge` / `Hermes Router` → 基于 KV Cache 感知与负载策略分发到具体推理实例；
- **PD 分离** 架构由 `PD-Orchestrator` 与 `developer-experience` 协同实现，对大模型推理降低 TTFT / 提高吞吐；
- **KV Cache 规模化** 由 `cache-indexer` 做全局索引、`mooncake` 做分布/分离式存储；
- **观测** 由 `eagle-eye` 提供推理链路跟踪。

---

## 5. 异构算力与资源调度

### 5.1 昇腾 NPU 全栈

| 项目 | 定位 | 技术栈 | 关键能力 | 产物/接口 |
|---|---|---|---|---|
| **npu-operator** | 昇腾 NPU 全栈运维 Operator | Go 1.24.5, kubebuilder, controller-runtime | 驱动/固件/运行时/设备插件/调度器/Volcano/NPU exporter/MindIO/vNPU/DRA 部署与生命周期 | CRD `NPUClusterPolicy`, Helm chart |
| **npu-container-toolkit** | 容器化 ascend-docker-runtime 部署 | Rust, OCI 钩子 | 修改 containerd/cri-o 配置、NPU 设备/库注入 | 二进制 / 镜像 |
| **npu-driver-installer** | 驱动/固件容器化安装 | Shell, Docker, RPM/DEB | 驱动版本管理、离线包支持 | DaemonSet / 镜像 |
| **npu-feature-discovery** | NPU 标签发现 | Go 1.24.5, NFD 框架 | 节点 NPU 型号/拓扑标签 | DaemonSet / Helm chart |
| **npu-node-provision** | 昇腾 MindIO 组件安装 | Shell, Docker | MindIO TFT/ACP 组件安装 | DaemonSet/Job / 镜像 |
| **npu-dra-plugin** | NPU 动态资源分配（DRA）插件 | 规划中 | 软切分、统一调度、指标监控 | 规划中 |
| **vNPU** | NPU 算力虚拟化与细粒度调度 | Go 1.24, C++, Volcano 插件 | 软切分（1% AICore / 1Gi 显存粒度）、硬切分、Volcano 调度器插件、设备插件、exporter | 资源名 `huawei.com/vnpu-*`, Helm chart |

### 5.2 混部与资源调度

| 项目 | 定位 | 技术栈 | 关键能力 | 产物/接口 |
|---|---|---|---|---|
| **volcano-ext** | Volcano 批量调度系统 openFuyao 分支 | Go 1.22, K8s 1.30 | Gang-Scheduling、Queue、PodGroup、NUMA 拓扑感知、Job/JobFlow、弹性调度 | CRD `*.volcano.sh`, Helm chart |
| **colocation-management** | 在离线混部资源/准入管理 | Go 1.24.5, kubebuilder, containerd NRI, Rubik | 超卖资源计算、准入控制、NRI 插件、QoS 压制、水位线驱逐 | Operator / DaemonSet / Helm chart |
| **colocation-service** | 在离线混部后端服务 | Go 1.24.5, go-restful | 节点加入/移除、策略配置、状态聚合、监控 | REST `/rest/colocation/v1/*` |
| **colocation-website** | 混部可视化面板 | OpenInula, ECharts | 混部概览、节点/策略/监控管理 | 微前端插件 |

### 5.3 NUMA 亲和与 UB 超节点

| 项目 | 定位 | 技术栈 | 关键能力 | 产物/接口 |
|---|---|---|---|---|
| **volcano-config-service** | NUMA 拓扑/资源采集与调度配置后端 | Go, kubebuilder, go-restful, Prometheus | NUMA 发现、实时资源、容器分配、拓扑策略配置 | REST API / Helm chart |
| **volcano-config-website** | NUMA 监控与调度配置前端 | React/OpenInula, ECharts, qiankun | NUMA 拓扑图、资源监控、策略参数配置 | 微前端插件 |
| **ub-network-device-plugin** | UB 网络设备插件 | Go 1.25, Kubelet Device Plugin gRPC | 上报/分配 `unifiedbus.com/ub_net_device` | DaemonSet / Helm chart |
| **ub-network-bandwidth-operator** | UB 网络 QoS Operator | Go 1.24, CRI, CNI QoS | 按优先级配置带宽限制、ConfigMap 热加载 | ConfigMap / 镜像 / Helm chart |
| **ub-enable-quickstart** | UB 使能 SIG 快速开始 | Markdown/示例代码 | 内存借用、共享内存教程与 demo | 文档/示例 |

### 5.4 关键资源类型

- NPU: `huawei.com/NPU`（整卡）, `huawei.com/vnpu-number/cores/memory.1Gi`（虚拟卡）
- KAE: `/k8s-device-plugin.openfuyao.com/kae`, `device.openfuyao.com/KAEPolicy`
- UB 网络: `unifiedbus.com/ub_net_device`
- Volcano 调度: `volcano.sh/PodGroup`, `volcano.sh/Queue`, `volcano.sh/Command`, `volcano.sh/Numatopology`

---

## 6. K8s 底座与社区治理

### 6.1 上游适配/内源项目

| 项目 | 来源/定位 | openFuyao 侧主要工作 |
|---|---|---|
| **kubernetes** | K8s 上游 v1.34.3 | `of-v1.34.3` 分支；`fylogs` 日志增强、`fy-kube-log-runner`；日志/调度/CRI 增强 |
| **containerd** | containerd v2.1.1 | 分支 `2.1.1-of.1`；版本维护与依赖升级 |
| **runc** | opencontainers/runc | master 分支；`--track-mem` 热迁移增强、路径调整 |
| **etcd** | CNCF etcd | 分支 `of-v3.6.7`；安全依赖升级、CVE 修复 |
| **coredns** | CoreDNS | 分支 `of-v1.12.2`；`fypkg/crypto`、`fypkg/dynamicloader` 安全/证书增强 |
| **kwok** | sigs.k8s.io/kwok | 大规模仿真集群；额外 list-watch 客户端 |
| **node-feature-discovery** | kubernetes-sigs NFD | 节点硬件/拓扑发现，OAT 配置与依赖升级 |

### 6.2 多算力/大规模集群

- **clusters-computing-operator**：多算力租户集群 Operator，基于 Kubebuilder，定义 `TenantCluster` / `TenantManagedCluster` CRD，借助 Karmada 部署控制面并聚合成员集群状态。
- **sig-large-scale-cluster**：沉淀单集群 128k 卡/1.6 万节点、未来 50 万卡目标的最佳实践，指向 `kubernetes`, `volcano-ext`, `kwok`, `mind-cluster` 等仓。

### 6.3 社区、测试与交付

| 项目 | 定位 | 内容 |
|---|---|---|
| **community** | 社区治理入口 | SIG 列表、成员、贡献指南、PR/Issue/许可证规范 |
| **docs** | 官方文档库 | 发行说明、安装指南、设计文档、写作规范 |
| **ofep** | 特性提案仓库（类似 KEP） | 提案模板、审批流程、里程碑与 feature gate |
| **sig-Container-Platform** | 容器平台 SIG | 基础平台能力治理、设计文档、子项目清单 |
| **sig-orchestration-engine** | 编排引擎 SIG |  roadMap、容器运行时/网络/存储规划、OFEP 管理 |
| **e2e-auto-test** | 端到端/自动化测试 | Go+Ginkgo 后端测试、TypeScript+Playwright 前端 UI 测试 |
| **qa-tool** | QA 辅助工具集 | GitCode 看板/issue 自动化、镜像安全离线检查、性能/内存泄漏监控 |

---

## 7. 关键依赖与技术栈全景

### 7.1 后端服务统一技术栈

- **语言**：Go 1.24.5 为主，少量 Python、Shell、Rust、C++。
- **Web 框架**：`emicklei/go-restful/v3` 系列。
- **K8s 客户端**：client-go 0.28 - 0.34 各服务不一，controller-runtime 0.16 / 0.19 / 0.23 等。
- **Operator 脚手架**：Kubebuilder、controller-runtime。
- **日志/配置**：`go.uber.org/zap`、`spf13/viper`、`natefinch/lumberjack`。
- **测试**：`onsi/ginkgo/v2` + `gomega`。
- **认证**：`golang-jwt/jwt/v4`。
- **gRPC**：`google.golang.org/grpc` 1.79+，用于部分服务间通信。

### 7.2 前端统一技术栈

- **框架**：OpenInula（类 React）、React 18。
- **构建**：Vite 4/5。
- **组件库**：Ant Design 5。
- **图表**：ECharts。
- **代码编辑器/终端**：CodeMirror、xterm.js。
- **微前端**：qiankun。
- **国际化/请求**：inula-intl, inula-request, inula-router。

### 7.3 底层/系统组件技术栈

- **容器运行时**：containerd / runc / CRI / OCI 运行时。
- **CNI / 网络**：villus（openFuyao 自研 CNI，仓库未在当前目录）。
- **存储**：localcsi（规划中）。
- **调度器**：Volcano + 大量调度扩展插件。
- **服务发现/存储**：etcd、CoreDNS。
- **监控/可观测**：Prometheus、Grafana、VictoriaMetrics Controller（当前目录包含 `victoriametrics-controller`）。

### 7.4 openFuyao 私有依赖

- `gopkg.openfuyao.cn/common-modules/ologger`：统一日志模块。
- `github.com/openeuler/ubs-engine.git` / 昇腾 CANN/HDK：面向 UB / 昇腾硬件的私有依赖。

---

## 8. 组件依赖关系简图

```text
用户
  │
  ├─ console-website / installer-website / bke-console-website / 各微前端插件
  │     │
  │     ▼
  ├─ console-service / installer-service / 各 *-service 后端
  │     │
  │     ▼
  ├─ oauth-server + oauth-webhook (认证鉴权)
  │     │
  │     ▼
  ├─ cluster-api-provider-bke / K8s API Server ──► kube-apiserver / etcd
  │     │
  │     ▼
  ├─ bkeadm / bkeagent / openfuyao-system-controller (生命周期)
  │     │
  │     ▼
  ├─ kubernetes / containerd / runc / coredns (编排底座)
  │     │
  │     ├─ npu-operator / vNPU / npu-* (昇腾算力)
  │     ├─ volcano-ext / colocation-* (调度/混部)
  │     ├─ volcano-config-service / ub-network-* (NUMA/UB 超节点)
  │     ├─ ray / ray-service (分布式训练/计算)
  │     ├─ InferNex / hermes-router / mooncake / cache-indexer / eagle-eye (AI 推理)
  │     └─ monitoring-service / logging-operator / marketplace-service / plugin-management-service (平台服务)
```

---

## 9. 典型使用场景

### 9.1 场景 A：私有化交付 K8s + openFuyao 管理面

1. 使用 `env-check` / `node-checker` 完成环境预检；
2. 通过 `bkeadm create cluster` 或 `installer-website` 引导安装；
3. `cluster-api-provider-bke` + `bkeagent` 下发安装指令；
4. 安装完成后使用 `console-website` 与 `console-service` 进行平台管理。

### 9.2 场景 B：昇腾 NPU 集群的 AI 训练与调度

1. 部署 `npu-operator`，通过 `NPUClusterPolicy` 声明驱动/固件/运行时/调度器；
2. `npu-feature-discovery` 为节点打上 NPU 标签；
3. 使用 `volcano-ext` 提交 `Job` / `PodGroup` 进行批量训练；
4. 可选启用 `vNPU` 实现细粒度算力切分。

### 9.3 场景 C：大模型推理服务化

1. 部署 `InferNex` / `mooncake` / `cache-indexer` 构建 KV Cache 基础设施；
2. 在 `hermes-router` / `developer-experience` 配置流量路由策略（KV-aware / 负载感知）；
3. 通过 `PD-Orchestrator` 做 Prefill/Decode 分离；
4. `eagle-eye` 提供可观测。

---

## 10. 需要注意的事项

1. **子仓库多、依赖版本不一**：client-go / controller-runtime / K8s API 版本在不同服务间有差异（0.28 ~ 0.34），需要协调升级。
2. **前端大多为微前端插件**：依赖 `console-website` 主应用与 qiankun 框架，单独运行通常需要基座。
3. **硬件依赖强**：昇腾、KAE、UB 等组件依赖特定系统环境、内核、驱动包与离线包；文档中（如 npu-operator、vNPU）对 OS 版本和 HDK/CANN 版本有明确要求。
4. **安装面与业务面分开**：“管理面/业务面”安装由 `installer`/`installer-service` 负责，与 cluster-api 流程互补。
5. **部分项目处于规划中**：如 `npu-dra-plugin`、`localcsi-controller/node`、`pod-lm-controller/agent` 当前目录下尚未包含完整代码或仅含 README。
6. **分支/tag 管理**：多个上游 Fork（kubernetes、etcd、containerd、runc、coredns、kwok、NFD）使用 openFuyao 专用分支，需要关注与上游的同步策略。

---

## 11. 目录清单速查

> 按功能域整理的主要代码仓（不含 `*-website` 前端，已在上文表格覆盖）。

- **安装与生命周期**：`bkeadm`, `bkeagent`, `bkecommon`, `bke-manifests`, `cluster-api-provider-bke`, `installer`, `installer-service`, `openfuyao-system-controller`, `env-check`, `node-checker`, `dependency-database`
- **容器编排底座**：`kubernetes`, `containerd`, `runc`, `etcd`, `coredns`, `kwok`, `node-feature-discovery`
- **管理面服务**：`console-service`, `application-management-service`, `monitoring-service`, `monitoring-dashboard-service`, `logging-operator`, `multi-cluster-service`, `marketplace-service`, `plugin-management-service`, `web-terminal-service`
- **认证鉴权**：`oauth-server`, `oauth-webhook`, `oauth-proxy`
- **AI 分布式**：`ray`, `ray-service`
- **AI 推理**：`InferNex`, `hermes-router`, `developer-experience`, `cache-indexer`, `mooncake`, `eagle-eye`
- **昇腾算力**：`npu-operator`, `npu-container-toolkit`, `npu-driver-installer`, `npu-feature-discovery`, `npu-node-provision`, `npu-dra-plugin`, `vNPU`
- **KAE 加速**：`kae-operator`, `kae-device-plugin`, `kae-driver-installer`
- **调度与混部**：`volcano-ext`, `volcano-config-service`, `colocation-management`, `colocation-service`
- **超节点/UB**：`ub-network-device-plugin`, `ub-network-bandwidth-operator`, `ub-enable-quickstart`
- **多集群/大规模**：`clusters-computing-operator`
- **可观测/监控存储**：`victoriametrics-controller`
- **测试/QA/社区**：`e2e-auto-test`, `qa-tool`, `community`, `docs`, `ofep`, `sig-*`, `release-management`

---

## 12. 参考文档

- `release-management/component-manage/component-list.md` — openFuyao 全量组件清单
- 各子项目 `README-zh.md` / `README.md` / `Chart.yaml` / `go.mod`
- 各 `sig-*/docs/` 下的设计文档与 RoadMap
- `e2e-auto-test/AGENTS.md` — 测试规范（如存在）

---

## 附录 A：openFuyao 对原生 Kubernetes 及其核心生态的改动与增强

> 本章基于对 `kubernetes/`, `coredns/`, `etcd/`, `runc/`, `kwok/`, `containerd/` 等上游 Fork 仓库的源码只读分析，汇总 openFuyao 在 K8s 底座上的主要增强点。

### A.1 总体结论

openFuyao **不是直接使用原生 Kubernetes**，而是维护了自己的下游分支 `of-v1.34.3`，并对 K8s 及核心运行时/网络/存储组件做了面向企业级、国产硬件、大规模 AI 集群的增强。增强方向集中在：

1. **安全与证书管理**：动态证书热加载、密文私钥解密、TLS CA 自动更新、敏感数据清零。
2. **日志与可观测性**：独立 access log、日志文件按权限/轮转/压缩、磁盘满保护。
3. **高可用与性能**：Etcd 快速探测、watch 进度通知、runc 热迁移内存追踪、大规模调度仿真增强。
4. **底座轻量化**：单进程 `kubeone` 模式，允许 apiserver 直接复用同进程 EtcdServer。
5. **安全合规**：依赖升级、CVE 修复。

对外 API 层面仍然兼容原生 Kubernetes，但二进制产物和默认配置经过了 openFuyao 定制。

### A.2 Kubernetes 主仓库（`kubernetes`）

**基准版本**：`tag 1.34.3-of.1`，分支 `of-v1.34.3`；同时存在一个面向超轻量部署的 tag `v1.34.3-kubeone`。

#### A.2.1 新增核心内部模块 `k8s.io/fy-ext`

位于 `kubernetes/staging/src/k8s.io/fy-ext/`，是一个与 K8s 核心解耦、可被各组件复用的扩展库：

| 子包 | 功能 |
|------|------|
| `dynamicloader` | 动态证书 / CA / 文件内容加载；监听文件变更并热更新 tls.Config |
| `dynamicloader/certificate` | 服务端/客户端动态证书、CA 池、缓存管理 |
| `dynamicloader/contentprovider` | 文件内容提供器、监听器、watch 机制 |
| `security/crypto/decryption` | 插件化私钥/证书解密（适配外部 KMS/密码机） |
| `security/sensitive` | 敏感字节内存清零 `CleanSensitiveData` |
| `signal` | 全局 context 管理，把 `<-chan struct{}` 转成可取消的 `context.Context` |

#### A.2.2 各核心组件接入 `fy-ext` 的情况

| 接入点 | 文件 | 增强内容 |
|--------|------|----------|
| `kube-apiserver` | `cmd/kube-apiserver/app/server.go` | 启动时设置全局 context (`fy-ext/signal`) |
| `kube-apiserver` | `pkg/controlplane/apiserver/options/fy_option.go` | 监听 `--service-account-issuer-key-file` 变更，支持**密文私钥**透明解密并热更新 token 签名器 |
| `kube-apiserver` | `staging/src/k8s.io/apiserver/pkg/server/options/fy_access.go` | 新增 `--fy-access-log-*` 系列参数，输出独立 access log，支持权限、轮转、压缩、磁盘满保护 |
| `kube-apiserver` | `staging/src/k8s.io/apiserver/plugin/pkg/authenticator/token/oidc/fy_oidc.go` | OIDC CA Bundle 动态更新 |
| `kube-controller-manager` | `cmd/kube-controller-manager/app/fy_controllermanager.go` | 监听 root-ca-file / service-account-key-file 变化并热更新 ServiceAccount token 控制器 |
| `kubelet` | `cmd/kubelet/app/server.go` | 构造 `tls.Config` 后通过 `dynamicloader` 开启动态服务端证书 + 客户端 CA |
| `kubelet` + apiserver 调用链 | `staging/src/k8s.io/client-go/transport/transport.go` | 新增 `TLS.FYReloadCAFile` 字段，客户端 CA 文件可动态重载 |
| etcd3 后端 | `staging/src/k8s.io/apiserver/pkg/storage/storagebackend/factory/fy_etcd3.go` | 动态 CA + 快速感知节点 unhealthy（`FY_FAST_PERCEIVE_NODENOTREADY_CONFIG`）；暴露 `SetEtcdServer` 给 **kubeone** |
| serviceaccount | `pkg/serviceaccount/fy_jwt.go` | 动态公钥加载器 `FyDynamicPublicKeysGetter` |

#### A.2.3 日志增强 `fylogs`

新增包 `staging/src/k8s.io/component-base/fylogs/` 与新的独立二进制 `logs/fy-kube-log-runner/fy_kube_log_runner.go`：

- 原 `kube-log-runner` 只支持 `--log-file`；openFuyao 版本加入：
  - `--fy-file-permission`：日志文件权限控制（默认 0644，推荐 0640）。
  - `--fy-compress-permission`：压缩后日志文件权限（默认 0440）。
  - `--fy-rotate-enable`、`--fy-each-file-size`、`--fy-total-size-limit`、`--fy-total-count-limit`：日志轮转与总量限制。
  - `--fy-ignore-nospace-error`：磁盘满时进程不退出，记录故障时间，待磁盘恢复后继续写日志。
  - `--fy-check-log-deleted-period`：周期性检测日志文件是否被误删。
  - `--fy-enable-daemon-subprocess`：子进程崩溃后自动重启。
- `fylogs` 包实现了文件包装、轮转器、错误处理器、磁盘恢复逻辑。

#### A.2.4 为 `kubeone` 做的轻量化适配

- `staging/src/k8s.io/apiserver/pkg/storage/storagebackend/factory/fy_etcd3.go` 中导出 `SetEtcdServer(server *etcdserver.EtcdServer)`：
  - 在 kubeone 模式下，apiserver 不再通过 gRPC 连接外部 etcd，而是直接复用同一进程内的 EtcdServer 实例。
- git tag 直接命名为 `v1.34.3-kubeone`，说明这是为超紧凑单节点/边缘场景定制的 K8s 底座。

#### A.2.5 对我方业务校验的适配

- 部分 `pkg/apis/.../validation/` 代码里增加了 `// +k8s:verify-mutation:reason=clone` 注释，配合静态校验器保证深拷贝逻辑与单元测试安全。
- `pkg/fyprivatekeyspublisher/`：新增「私钥发布器」，在 apiserver 的私钥热更新后把新的私钥内容发布给订阅方，可能用于多副本同步或审计场景。

#### A.2.6 依赖/构建/安全补丁

- 从提交历史可见近期 MR 包括：
  - 修复 SCA（静态代码分析）问题。
  - 修复 PLEG 构建失败、修复 UT 错误。
  - 更新 kubeone 依赖。

### A.3 CoreDNS（`coredns`）

**基准版本**：`tag 1.12.2-of.1`，分支 `of-v1.12.2`。

#### A.3.1 与 K8s 同源的 `fypkg` 能力

在 CoreDNS 中新增 `fypkg/` 目录，几乎复用了 `k8s.io/fy-ext` 中同类能力：

| 子包 | 功能 |
|------|------|
| `fypkg/crypto` | 密码学钩子 + 解密插件适配（CaaS Core 安全模块） |
| `fypkg/crypto/caascore/security/crypto/decryption/...` | 基于 UDS/gRPC 的外部解密服务客户端，支持从外部 KMS 或密码机解密 TLS 私钥 |
| `fypkg/dynamicloader` | 与 `k8s.io/fy-ext/dynamicloader` 等价的动态证书/文件内容加载实现 |
| `fypkg/util/timer` | 可重试 runner |

#### A.3.2 实际接入点

- `plugin/kubernetes/fy_kubernetes.go`：Kubernetes 插件初始化时使用 `fypkg/crypto` 进行私钥解密。
- `plugin/pkg/tls/fy_tls.go`：TLS 配置接入动态证书加载，支持证书/CA 变更后自动重载。

#### A.3.3 CVE 修复

- 提交记录中有明确说明：「fix: upgrade dependencies to address 22 CVEs」。
- Go 构建版本升级到 1.25.0，并修复测试中存在的竞态条件。

### A.4 containerd（`containerd`）

**基准版本**：`tag 2.1.1-of.1`，分支 `main`。当前仓库中可见的改动以版本同步为主，源码层的新增 `fy-*` 逻辑较少。

- openFuyao 维护的 branch 保证了与上游 v2.1.1 同步，并可能包含针对国产 OS、安全加固、CVE 修复的补丁。
- 因为 CRI 接口由 K8s 定义，openFuyao 对 CRI 的增强更多体现在 `kubernetes` 的 kubelet/CRI 客户端与 `npu-container-toolkit` 这类设备插件侧，而不是在 containerd 本体大量改源码。

### A.5 runc（`runc`）

**当前分支**：`master`。

- 提交记录显示核心改动为：**新增 `--track-mem` 参数，调整 `stats-dump` 文件路径**。
- 在 `libcontainer/container_linux.go` 与 `checkpoint.go` 中可以看到：
  - `TrackMem` 字段被传递给 CRIU，用于 **容器热迁移/检查点时追踪内存变化**，加速增量 checkpoint 或 live-migration。
- 其余提交主要是初始同步与合并。runc 本体改动相对聚焦在容器迁移能力上。

### A.6 etcd（`etcd`）

**基准版本**：`tag 3.6.7-of.1`，分支 `of-v3.6.7`。

- 与安全强相关：
  - `!15 security: bump 5 deps to fix 27 CVEs (align upstream v3.6.13)`。
- 与高可用相关：
  - `fix：serveClients recover to 3rd`——描述虽简短，指向 **etcd 客户端服务异常时的恢复逻辑**。
- Go 版本升级到 1.25.x（与 k8s 1.34 对齐）。
- 与 K8s 的联动：Kubeadm 默认配置 etcd 的 `--watch-progress-notify-interval=5s`（替换了上游的 1s/未设），让 apiserver watch cache 更快感知 etcd 变化。

### A.7 kwok（`kwok`）

用于大规模 K8s 控制面压测与节点仿真。

- 增强大规模 List-Watch 仿真能力：
  - 新增 `AdditionalListWatchClients` / `AdditionalListWatchClients` 配置。
  - `pkg/utils/informer/listwatchs.go`：模拟多组客户端对 Pod/Service/Node/ConfigMap/Secret/Endpoints/Deployment 等资源进行 List-Watch，逼近真实 kubelet 对 apiserver 的压力。
  - `experimental-enable-kubelet-list-watch` 标志用于开启对 kubelet list-watch 行为的仿真。
- 用于验证 openFuyao 大规模集群下调度器、apiserver、CRD operator 的稳定性。

### A.8 增强点一览表

| 维度 | 原上游能力 | openFuyao 增强 |
|------|------------|----------------|
| 私钥/证书 | 启动时加载一次 | **运行时动态重载**；支持**密文私钥**解密 |
| 客户端 CA | 启动时加载一次 | **动态 CA 更新** |
| 日志 | 简单的 `--log-file` | **文件权限、轮转、压缩、磁盘满保护、子进程守护** |
| access log | 无单独文件 | **独立 access log 文件 + 轮转压缩** |
| etcd 后端 | 默认 watch 进度 | **快速节点不健康感知** + 单进程 EtcdServer 适配 |
| 运行时 | 标准 CRIU 检查点 | **支持 `--track-mem` 内存变化追踪**，优化热迁移 |
| 轻量底座 | 多组件多进程 | **`kubeone` 单进程模式** |
| 安全合规 | 依赖版本基准 | **主动升级依赖修复 CVE**（etcd 27 个、CoreDNS 22 个等） |
| 大规模测试 | 普通 kwok 节点模拟 | **模拟 kubelet 多客户端 List-Watch 负载** |

### A.9 对用户/运维的影响

- **配置差异**：部署 openFuyao 时会出现 `--fy-*` 系列启动参数，以及访问日志/日志权限相关默认值与原生 K8s 不同。
- **升级注意**：openFuyao 分支的 tag（如 `1.34.3-of.1`）在原版 v1.34.3 基础上叠加了补丁，**不能直接用上游 v1.34.3 二进制替换**。
- **二进制产物**：除 `kube-apiserver` / `kubelet` 等原生二进制外，新增了 `fy-kube-log-runner`。
- **依赖替换**：所有 `k8s.io/*` staging 模块都改指向 `gitcode.com/openfuyao/kubernetes.git/staging/src/k8s.io/...`（含 `k8s.io/fy-ext`）。

---

## 附录 B：openFuyao 对大规模集群与多集群纳管能力的增强

> 回应常见问题：openFuyao 是否在 K8s 底座层有针对「大规模集群纳管」的优化？

### B.1 一句话结论

- **Kubernetes 主仓库本身**：源码层没有明显调大控制面并发极限（如 `--max-requests-inflight` 仍保持 400），也没有新增专门的大规模控制面 CRD。其主要增强在**证书/安全/日志/热迁移/轻量化底座**。
- **openFuyao 平台层**：在 K8s 之上通过 `clusters-computing-operator` + `multi-cluster-service` + Karmada + `kwok`，构成了**面向万级节点、多集群统一纳管**的完整方案。

因此，大规模集群纳管能力更多是 **openFuyao 在上层平台工程化的结果**，而非单纯依赖 K8s 主仓库源码魔改。

### B.2 大规模多集群纳管核心组件

#### B.2.1 `clusters-computing-operator` — 万级节点分片编排

这是一个 Kubebuilder 风格的 Operator，职责是：**把一个由万台物理节点组成的「租户集群」自动拆分成若干可管理的子 K8s 集群，并完成部署、纳管、生命周期管理。**

| 能力 | 具体实现 |
|------|----------|
| 万级节点声明 | 支持节点范围表达式，如 `nodeList: ["node[1-10000]"]`；通过 `ExpandNode()` 展开为具体主机名 |
| 自动分片 | 默认每个子集群 5000 节点（`CLUSTER_SIZE` 默认 5000），每个子集群 3 个管理节点（`NUM_MANAGEMENT_NODES` 默认 3） |
| 节点聚合逻辑 | `buildManagedClusters()` 将总节点数按 `clusterSize` 拆分；剩余节点若超过 `minClusterSize`（默认 100）则再建一个子集群，否则合并到最后一个子集群 |
| 大规模安装参数 | 为每个子集群生成 Ansible INI 时注入 `[large_scale]` 配置：`SUB_GROUP_MAX_SIZE=5`、`AUTO_GROUP_BY_SUBNET=0`，用于底层批量安装时的子网分组优化 |
| 控制面 | 为每个租户集群部署独立的 **Karmada** 控制面，作为多集群联邦调度与统一 API 入口 |
| 纳管模式 | 支持 `managed`（由 openFuyao 自动安装并 join）与 `unmanaged`（用户已有集群，自行 join 到 Karmada）两种模式 |

**典型流程**：

```
TenantCluster (万节点声明)
    │
    ├── 部署 SuperClusterD（超级集群守护组件）
    ├── 部署 Karmada 实例
    └── buildManagedClusters()
            │
            ├── node[1-5000]   → TenantManagedCluster-member-1
            ├── node[5001-10000] → TenantManagedCluster-member-2
            │
            └── 每个 TenantManagedCluster 通过 Job 完成 openFuyao 安装并 join 到 Karmada
```

#### B.2.2 `multi-cluster-service` — 多集群统一管理与跨集群服务

这是一个 **openFuyao 扩展组件**，通过应用市场安装，基于 **Karmada** 实现：

- 集群列表：统一查看所有被纳管集群状态与基本信息。
- 集群生命周期：纳管新集群、调整规模、更新配置、安全取消纳管。
- 集群凭证：提供被纳管集群的 kubeconfig，实现快速访问。
- 集群标签管理。
- **跨集群服务访问**：在单一主集群入口访问所有被纳管集群上的服务（`ServiceExport/ServiceImport` 类能力）。

这个项目把 Karmada 的底层多集群能力包装成了 openFuyao 控制台可操作的 UI/权限/流程。

#### B.2.3 `kwok` — 大规模控制面压测与仿真

openFuyao 维护的 kwok 分支增强了仿真真实 kubelet 对控制面压力的能力：

- 原生 kwok 主要模拟节点 lifecycle。
- openFuyao 增强：
  - `AdditionalListWatchClients` / `AdditionalListWatchResources`
  - `--experimental-enable-kubelet-list-watch`
  - 模拟多组客户端同时对 Pod/Service/Node/ConfigMap/Secret/Endpoints/Deployment/ReplicaSet/DaemonSet/StatefulSet/Job/CronJob 进行 List-Watch
- 用途：在实验室里低成本验证万级节点规模下，apiserver、etcd、scheduler、CRD operator 的稳定性与性能瓶颈。

### B.3 Kubernetes 底座对大规模场景的间接支撑

虽然 K8s 源码没有直接改大 `--max-requests-inflight` 等参数，但 openFuyao 的底层改动**间接提升了大规模集群的可运维性**：

| 改动 | 对大规模的意义 |
|------|----------------|
| `fy-ext/dynamicloader` 动态证书 | 万节点场景下证书续期无需滚动重启 apiserver/kubelet |
| etcd watch 进度通知默认 5s | 控制面缓存更快同步，降低节点状态不一致时间窗口 |
| `kubeone` 单进程模式 | 边缘/小规模节点可作为一体化控制面快速铺开，减少组件数量 |
| runc `--track-mem` | 支持容器热迁移，便于大规模集群节点维护与负载腾挪 |
| `kwok` 增强 List-Watch 仿真 | 上线前就能在单机上模拟万节点控制面压力 |

### B.4 与原生 Kubernetes 大规模能力的对比

| 能力 | 原生 K8s | openFuyao |
|------|----------|-----------|
| 单集群节点规模 | 官方建议 5,000 节点 | 通过子集群分片 + Karmada，可支持 **万级乃至更多** |
| 多集群统一纳管 | 无原生方案（需自行集成 Karmada/Fleet/OCM） | 提供 `clusters-computing-operator` + `multi-cluster-service` 工程化方案 |
| 跨集群服务访问 | 需依赖 Karmada MCS/Submariner 等 | 内置在 `multi-cluster-service` 中 |
| 大规模安装 | 需自行编写 Ansible/Terraform | `clusters-computing-operator` 自动生成安装清单，支持 `[large_scale]` 子网分组 |
| 大规模压测 | 需自行搭搭建测试集群 | `kwok` 增强支持多客户端 List-Watch 仿真 |

### B.5 对用户/架构师的意义

1. 如果你的目标是**单集群 10,000 台物理节点**，openFuyao 的答案是：**不会把压力全部压在同一个 K8s apiserver/etcd 上，而是通过分片成多个 5000 节点级子集群，再用 Karmada 联邦纳管**。
2. 如果你的目标是**跨地域/多中心的统一平台管理**，`multi-cluster-service` 提供了纳管、访问、服务跨集群的能力。
3. openFuyao 在 K8s 主仓库里主要是为了让这整套多集群架构**更安全、可热更新证书、易于维护**，而不是单纯拼 K8s 单控制面极限。

---

## 附录 C. 真实超大规模单集群实验中的控制面水平拆分实践

> 本节内容基于团队内部真实实验经验整理，用于补充 openFuyao 仓库公开文档中未完整描述的部署实践。

### C.1 实验目标与结果

| 阶段 | 节点规模 | 控制面架构 |
|---|---|---|
| 阶段1 | 2,500 节点 | 3 控制面主机、单 etcd 集群 |
| 阶段2 | 12,000 节点 | 9 控制面主机、3 套 etcd（pods / events / data）、9 个 apiserver 实例 |
| 阶段3 | **20,000 节点** | 同上 + **apiserver 按 API 资源类型分组** + HAProxy 七层路由 |

同步规模还包括：**100,000 级 Pod**、**500,000 级指标/秒**。

### C.2 openFuyao 公开文档中的「三 etcd 拆分」

openFuyao `sig-large-scale-cluster` 单集群目标为 **16,000 节点 / 128,000 卡**，推荐架构：

- **etcd-pods**：承载 `/registry/pods` 等 Pod 相关数据
- **etcd-events**：承载 `/registry/events`、`/registry/leases` 等高写入数据
- **etcd-data**：承载其他控制面数据

这一拆分依赖的是 **Kubernetes 原生能力 `--etcd-servers-overrides`**（见 `kubernetes/pkg/kubeapiserver/default_storage_factory_builder.go:148` 的 `ConfigForResource` 处理），openFuyao 在该路径上**没有源码级改造**。

### C.3 内部实验补充：apiserver 按 API 资源分组

在 openFuyao 公开文档中，apiserver 高可用方案是 **keepalived + HAProxy 四层 TCP 轮询**（`best_practices_for_kube_apiserver_load_balancing_in_the_ultra_large_scale_cluster_scenario.md`），所有客户端都通过同一个 VIP 访问一组对等的 apiserver 实例。

团队内部实验进一步做了优化：

> **单一 VIP 入口，后端 HAProxy 按 URL path 做七层路由，把 `/api/v1/pods`、`/api/v1/services`、`/api/v1/nodes` 等请求转发到不同的 apiserver 实例组。**

转发规则示意：

```
frontend k8s-apiserver
  bind 0.0.0.0:6443
  acl is_pods  path_beg /api/v1/pods
  acl is_nodes path_beg /api/v1/nodes
  acl is_svc   path_beg /api/v1/services
  use_backend apiserver-pods  if is_pods
  use_backend apiserver-nodes if is_nodes
  use_backend apiserver-svc   if is_svc
  default_backend apiserver-core
```

### C.4 为什么能减少 apiserver 的 cache 压力

kube-apiserver 的内存主要被以下几部分占用：

- **watch cache**：每个启用的资源类型都会维护一个 cache，用于快速 list/watch。
- **etcd watch 连接**：每个 cacher 都与 etcd 保持一个 watch 长连接。
- **客户端长连接**：kubelet、controller-manager、scheduler、operator 都会与 apiserver 建立 watch。
- **反序列化/索引**：大量对象会触发 Go GC 与内存分配。

当所有请求和长连接都集中在同一组 apiserver 上时，每个 apiserver 节点需要同时维护所有资源类型的 cache 和所有客户端连接。通过 HAProxy 七层按资源分组：

| 优化对象 | 效果 |
|---|---|
| 长连接数量 | 不同客户端/请求被分散到不同 apiserver 组，单组连接数下降 |
| watch cache 热数据 | 某个 apiserver 组主要处理 Pod 请求，其 Pod cache 使用更集中，其他 cache 处于"冷"状态 |
| 资源争抢 | 重资源（Pod/Node）不会拖垮处理轻量资源（如 ConfigMap、Secret）的 apiserver 组 |
| 故障域 | 单组异常只影响部分 API，提高整体可用性 |

实际实验结果验证了这一点：引入该分组后，控制面从 12,000 节点进一步提升到 **20,000 节点**。

### C.5 这是 K8s 源码改造还是部署架构创新

**属于部署架构创新，不需要修改 Kubernetes 源码。**

原因：

- 每个 apiserver 实例仍然使用标准的 `kube-apiserver` 二进制镜像；
- 只是前端 HAProxy 根据 HTTP 路径做反向代理路由；
- 所有 apiserver 仍连接同一套或多套 etcd，etcd 层不需要额外改动。

> 原生 Kubernetes（同样的版本、同样的编译参数）配合同等的硬件、同等的 etcd 拆分、同等的 HAProxy 七层配置，理论上可以复现类似规模。

### C.6 局限与注意事项

1. **apiserver 进程仍会初始化全量 storage**：`kube-apiserver` 启动时会为所有启用的 API 资源注册 storage 并建立 etcd watch。HAProxy 按路径路由只能分散**客户端流量**，无法让某个 apiserver 组"完全不加载"某些资源。如果追求完全隔离 cache，需要结合 `--runtime-config` 禁用部分 API group，或者对 apiserver 做更深度改造。
2. **注意跨资源一致性**：某些操作会级联影响多种资源（如 Deployment 创建会触发 ReplicaSet、Pod）。按资源分组后，需要确保所有相关 apiserver 组都可用且数据一致。
3. **HAProxy 自身成为瓶颈**：七层路由比四层转发更耗 CPU，HAProxy 节点需保证足够性能与冗余。

### C.7 实践总结

| 做法 | 是否依赖 K8s 源码改造 | openFuyao 是否已公开 |
|---|---|---|
| 三 etcd 集群拆分（pods / events / data） | 否（原生 `--etcd-servers-overrides`） | 是，sig-large-scale-cluster 详细最佳实践 |
| 多 apiserver 实例 + 四层 TCP VIP 负载均衡 | 否 | 是，keepalived + HAProxy 文档 |
| **apiserver 按 API 资源做七层分组** | **否** | **否（团队内部实验实践）** |

### C.8 从 12,000 节点到 20,000 节点的真实瓶颈演进

> 本节整理自团队内部真实调优过程，展示了不同阶段瓶颈的转移和对应决策。

#### C.8.1 阶段一：12,000 节点 —— apiserver cache 瓶颈

在 9 控制面主机、3 套 etcd（pods / events / data）、9 个 apiserver 实例的架构下，集群规模达到 12,000 节点时出现主要问题：

- apiserver 内存占用持续升高；
- watch cache 压力过大；
- 控制面响应变慢，接近单点性能上限。

**根因**：所有资源类型的请求和长连接压在同一组 apiserver 上，每个实例都需要维护全资源对象的 watch cache。

**解决方案**：启用 **HAProxy 按 API 资源类型做七层路由分组**。

实施后，不同 API 路径的请求被分发到专门的 apiserver 后端：

- Pod 相关请求 → `apiserver-pods`
- Node 相关请求 → `apiserver-nodes`
- 其他核心资源   → `apiserver-core`

这样每组 apiserver 的 cache 工作集更集中，内存压力和连接数被分散，控制面瓶颈得以突破。

#### C.8.2 阶段二：20,000 节点 / 100,000 Pod —— 状态上报通道瓶颈

突破到 20,000 节点、部署 100,000 个 Pod 后，出现新的问题：

- 真实节点上 Pod 运行正常；
- 但 kubectl 中部分 Pod 显示为 **Not Ready**；
- 实际上是 kubelet 上报的 Pod status 写请求被阻塞或延迟，控制面和真实状态不一致。

**根因转移**：从「cache 装不下」变成「Pod 写通道处理不过来」。

当时讨论过的候选方案包括：

| 方案 | 评估 |
|---|---|
| 给 kubelet 单独一个 VIP/4 层入口 | 能保护写通道，但会让 kubelet 专用组重新变成全资源 cache，破坏阶段一的优化成果 |
| 调大 kubelet QPS / APF 阈值 | 有效，但单独作用有限 |
| **增加 apiserver-pods 组的实例数量** | **推荐**：在保留 7 层资源分组的前提下，直接提升 Pod 相关请求的总处理能力 |

#### C.8.3 最终决策

**保留 7 层按资源分组架构，重点增加负责 Pod 相关资源的 apiserver 组实例数。**

原因：

1. 12,000 节点时的核心矛盾是 **cache 分散问题**，资源分组有效解决了这个问题。
2. 20,000 节点时的核心矛盾是 **Pod 写吞吐问题**，不能通过牺牲 cache 分散性来换取写通道隔离。
3. 在现有分组内增加实例数，可以在不破坏 cache 优化成果的前提下，线性提升处理能力和冗余度。

预期调整方向：

| 组 | 调整建议 |
|---|---|
| `apiserver-pods` | 重点扩增，从 3 实例扩至 6~9 实例或更多 |
| `apiserver-nodes` | 次要扩增 |
| `apiserver-core`  | 按需扩增 |
| 7 层 HAProxy 本身 | 如果后端实例数超过其处理上限，需同步升级规格 |

同时辅助调整：

- kubelet `--kube-api-qps` / `--kube-api-burst`：从默认 5/10 提升；
- apiserver `--max-requests-inflight` / `--max-mutating-requests-inflight`：提升限流阈值；
- kubelet 状态上报频率：适当放宽，降低写频率。

#### C.8.4 对架构师的启示

1. **单集群超大节点规模（10,000+）不是 K8s 源码无法突破的硬限制**，它可以通过控制面水平拆分来缓解。
2. 拆分的维度已经从 **etcd 存储层** 延伸到 **apiserver 请求处理层**。
3. 大规模场景下的优化要**分清当前主要矛盾**：cache、QPS、连接数、延迟，不同阶段的优化手段可能完全不同，甚至互相冲突。
4. openFuyao 在大规模场景下的核心价值不仅是"能让 K8s 跑多大"，更是把这套复杂架构**工程化、可安装、可运维、可仿真**；而你们内部的 apiserver 七层分组实践，是对这套架构的进一步补充和验证。

---

## 附录 D：openFuyao 插件化架构与国产异构算力通用管理框架

> **一句话定位**：如果把 CRD 当成“本体定义”、把 Controller 当成“本体驱动”，那么 openFuyao 的强项在于已经跑通了**一套可复用的组件本体+驱动模式**，但目前只把它用在了昇腾一个具体本体上。下面把它拆成两层：A 层（通用生命周期/安装管理框架）、B 层（跨厂商算力资源抽象与调度语义层）。两层相互独立，也可以组合使用。

---

### D.1 层 A：通用 CRD + Controller 组件管理框架（可复用的“本体驱动”生命周期层）

#### D.1.1 已存在的模式

openFuyao 目前有两套并行的扩展机制：

1. **应用市场/Helm 包层**：`marketplace-service` + `plugin-management-service`，用 OCI 仓库存放并安装 Helm Chart 组件包。该层只负责**包的发现、安装、升级、卸载**，不定义组件运行期如何被表达。
2. **控制台扩展层**：`plugin-management-service` 定义了 `ConsolePlugin` CRD (`console.openfuyao.com/v1beta1`)，用于把组件的 UI 注册到 openFuyao 控制台。字段包括：
   - `PluginName`、`DisplayName`、`Order`
   - `Entrypoint`（`Nav`/`Side`）
   - `Backend`（目前只支持 `Service`）
   - `SubPages`、`Enabled`

这两层解决的是**“包怎么来、UI 怎么挂”**的问题，还没解决**“组件本身的状态怎么声明、怎么调和”**的问题。

#### D.1.2 `npu-operator` 已经跑通的“本体驱动”模式

`npu-operator` 实际上已经给出了一个完整的答案，只是被固化在昇腾语义里：

- **本体定义**：`NPUClusterPolicy`（`api/v1/npuclusterpolicy_types.go`）
  - 只能有一个实例，名字必须是 `cluster`。
  - Spec 里列出昇腾各子组件：Driver、OCIRuntime、DevicePlugin、NodeD、Trainer、Volcano、InferOperator、Exporter、ClusterD 等。
  - 每个子组件都有 `Managed bool` + 镜像/环境变量/命令/资源/Placement 等公共字段——这正是 `ComponentWorkloadSpec` 在 `component_types.go` 中抽象的内容。
  - Status 里有 `Phase`（Ready/NotReady）以及每个组件的 `ComponentStatus`（pending/deploying/running/terminated/unmanaged）。

- **驱动实现**：`internal/controller/npuclusterpolicy_controller.go` 的 Reconciler 流水线：
  1. 读 `NPUClusterPolicy`。
  2. 校验冲突/版本/状态。
  3. 根据 spec 渲染 `/var/lib/npu-operator/components`（或源码 `assets/`）里的静态模板。
  4. 按 `Managed=true/false` 决定 apply 或 delete 子组件。
  5. 同步资源实际状态并回写 `Conditions` 和 `ComponentStatuses`。

- **核心可复用抽象**：
  - `ComponentWorkloadSpec`（一致性组件设置：启用、镜像、命令、环境变量、卷、资源、调度）。
  - `ComponentStatus` + `ComponentState`（统一状态机）。
  - `UpdateConditions` / `UpdateComponentStatus`（Status 回写辅助函数）。
  - 以目录为单元的静态 manifest 渲染（`assets/` → `/var/lib/npu-operator/components`）。

#### D.1.3 通用化后的目标形态

把 `NPUClusterPolicy` 从“昇腾专属本体”变为“通用组件套件本体”，可新增如下 CRD：

```yaml
apiVersion: management.openfuyao.com/v1beta1
kind: ClusterComponentPolicy
metadata:
  name: cluster
spec:
  vendor: ascend          # 未来可改为 kunlunxin / haiguang / mthreads / pingtouge
  version: "8.0.RC2"
  runtimeClass: ascend    # 由具体 vendor 解释
  components:
    - name: driver
      managed: true
      workload:
        imageSpec: { registry: "", repository: "ascend-driver", tag: "8.0.RC2" }
        nodeSelector: { "node-role.openfuyao.com/accelerator": "ascend" }
    - name: device-plugin
      managed: true
      # ...
  consolePlugin:
    register: true          # 安装完成后是否自动创建 ConsolePlugin
    pluginName: ascend-dashboard
    displayName: 昇腾硬件管理
    entrypoint: Side
    backend:
      service:
        name: ascend-dashboard-svc
        namespace: openfuyao-system
        port: 80
```

通用 Operator 的 Reconciler 流程可以复用 `npu-operator` 的骨架，只是把：

- **组件名 / 静态模板目录 / 渲染参数**——从代码里硬编码改为按 `spec.vendor` + `spec.version` 查表；
- **冲突检测 / 特殊 preflight**——下沉到 vendor 特定的 `Validator` 钩子；
- **镜像构建 / 命令参数**——保留为 `ComponentWorkloadSpec` 的公共字段，vendor 只提供默认值。

这样 `npu-operator` 的当前实现就退化为**一个 vendor driver 实例**，而通用框架成为 openFuyao 的**`component-management-operator`** 或 **`cluster-policy-operator`**。

#### D.1.4 与 openFuyao 现有能力的衔接

- **安装入口**：保留 `marketplace-service` 的 Helm/OCI 分发。通用 operator 本身也可以被打成一个 Chart 上架；vendor 组件包作为 Chart 或 assets 包被引用。
- **UI 注册**：当组件包声明 `consolePlugin.register=true` 时，通用 operator 在安装完成后自动创建一个 `ConsolePlugin`，无需组件自己写 Console 集成代码。
- **多集群场景**：`clusters-computing-operator` 已经管理了 `TenantCluster`/`TenantManagedCluster` 生命周期。将来可以在 `ClusterComponentPolicy` 上增加 `clusterSelector`，把组件下发逻辑委托给多集群控制器。

---

### D.2 层 B：跨厂商异构算力资源抽象与调度语义层

#### D.2.1 为什么层 B 必须独立

层 A 只解决**“把各厂商的 daemonset/operator 装起来、管起来”**。但上层 AI 平台（训练调度器 `Volcano`、推理网关 `InferNex`、Pod 资源申请）真正关心的是：

- 这个节点有什么算力？
- 一个 Pod 要申请“一张卡”或“一部分卡”时，资源名应该是什么？
- 调度时应该看内存、算力、拓扑、健康状态中的哪些维度？

目前 openFuyao 里这些语义是**隐式且厂商私有**的：昇腾设备插件会暴露 `huawei.com/Ascend910` 这类 extended resource，其他厂商会暴露各自的名字。上层组件如果不小心写死厂商资源名，就丧失通用性。

因此层 B 要做的是**“统一资源命名与能力描述规范”**，让厂商 device-plugin 和 scheduler 之间有一座桥。

#### D.2.2 建议的抽象对象

可以在 openFuyao 中定义一组**不依赖具体 vendor operator 的轻量 CRD / 标签约定**：

1. **节点标签约定（-node topology descriptor）**
   - `accelerator.openfuyao.com/vendor` = `ascend|kunlunxin|haiguang|mthreads|pingtouge|...`
   - `accelerator.openfuyao.com/device-type` = 芯片型号/代际。
   - `accelerator.openfuyao.com/capability` =  Capability Profile 名，例如 `ai-inference-heavy`、`ai-training-high-mem`。
   - `accelerator.openfuyao.com/health` = `healthy|degraded|unhealthy`。

2. **统一资源名约定（resource naming）**
   - 基础资源：`openfuyao.com/gpu` 或 `fuyao.com/accelerator` 表示“一张完整加速卡”。
   - 细分资源：`openfuyao.com/accelerator-memory` 表示显存，`openfuyao.com/accelerator-compute` 表示算力单元。
   - 厂商可以保持自己的 extended resource，但应通过 `fuyao.com/` 前缀别名对外暴露，便于上层 Pod 申请。

3. **DeviceProfile CRD（可选）**
   描述一类加速设备的抽象能力，供调度器读而不是给人写：
   ```yaml
   apiVersion: accelerator.openfuyao.com/v1beta1
   kind: DeviceProfile
   metadata:
     name: ascend-910b
   spec:
     vendor: ascend
     deviceType: Ascend910B
     computeUnits: 8
     memoryGB: 32
     memoryBandwidthGBps: 400
     topologyLabel: ring      # ring / switch / none
   ```
   - 设备插件在启动时可以根据本节点硬件型号自动创建/更新对应的 `DeviceProfile`；
   - 调度器通过读取 `DeviceProfile` 做通用化调度，而不是解析每个厂商的配置。

4. **NodeDeviceInventory CRD（可选）**
   每个节点一个实例，汇总该节点上所有加速设备的**逻辑容量**（来自 device-plugin 上报）与**健康状态**，与厂商 device-plugin  loosely coupled。

#### D.2.3 与调度系统的接口

- 对 `Volcano` 的 `nodeorder`、`predicate` 插件，提供**基于 DeviceProfile 的通用评分函数**，而不是按 vendor 单独实现。
- 对 k8s 原生 scheduler，提供基于 `fuyao.com/` extended resource 的统一资源名，让普通 Pod 可以写：
  ```yaml
  resources:
    limits:
      fuyao.com/gpu: 1
  ```
- 对上层 AI 平台（训练/推理编排），提供统一查询接口：按 `capability` 选节点，而不是按 `huawei.com/Ascend910` 选。

---

### D.3 两层的关系：独立与组合

| 维度 | 层 A：通用组件管理框架 | 层 B：跨厂商算力抽象 |
|---|---|---|
| **关注点** | component 的安装、升级、卸载、状态回写 | 资源名、节点标签、能力规格、调度语义 |
| **输入** | `ClusterComponentPolicy` / Helm Chart | 节点硬件信息、DeviceProfile、Pod 资源请求 |
| **输出** | DaemonSet/Deployment/ConfigMap/Service 等 | 统一 extended resource、节点标签、调度约束 |
| **是否可以单独存在** | ✅ 可以。只装 vendor operator，不定义统一资源名。 | ✅ 可以。只发布规范，让现有 operator 按规范暴露别名。 |
| **组合后的形态** | A 把 vendor 组件装完之后，自动驱动该 vendor 的设备插件/导出器按 B 的规范注册资源与标签。 | B 作为通用抽象，被 A 安装的组件自然实现，无需上层各自对接。 |
| **与现有 openFuyao 的关系** | 复用 `marketplace-service` + `ConsolePlugin` + `npu-operator` 控制器模式。 | 在 `npu-operator` 之上新增可选 CRD/标签层，与 `Volcano` / 原生 scheduler 对接。 |

**推荐的落地顺序**：

1. **先 A 后 B**：先抽出通用 `ClusterComponentPolicy` 框架，让昇腾、昆仑芯、海光等 operator 都能以相同模式接入 openFuyao。这一步对现有代码改动最小，因为很多骨架已经在 `npu-operator` 里。
2. **再定义 B 的最小公约**：先不统一所有细节，而是约定几个最核心的字段（`vendor` 标签、`fuyao.com/gpu` extended resource、`DeviceProfile` 结构），逐步让各厂商 operator 兼容。
3. **组合**：通用 operator 安装组件时，可选地创建 `ConsolePlugin` + `DeviceProfile` + `NodeDeviceInventory`，形成“一次安装、统一暴露”的体验。

---

### D.4 可复用代码/结构清单

| 来源文件 | 可复用内容 |
|---|---|
| `npu-operator/api/v1/npuclusterpolicy_types.go` | `Phase`、`ComponentStatus/State`、`UpdateConditions`、`UpdateComponentStatus`、状态机。 |
| `npu-operator/api/v1/component_types.go` | `ComponentWorkloadSpec`（`Managed`、`ImageSpec`、`Env`、`Command`、`VolumeSpec`、`Resources`、`PlacementSpec`）。 |
| `npu-operator/internal/controller/npuclusterpolicy_controller.go` | Reconcile 骨架：读 CR → 校验 → 渲染静态 manifest → apply/delete → 回写 status。 |
| `npu-operator/internal/controller/resources.go` / `components.go` | 静态模板渲染 + 组件冲突检测逻辑。 |
| `npu-operator/assets/` | 静态组件模板组织方式（按子组件目录存放）。 |
| `plugin-management-service/pkg/plugin/consoleplugin_type.go` | `ConsolePlugin` 模型，供通用框架在组件安装后自动注册 UI。 |
| `marketplace-service/README-zh.md` | Helm/OCI 包分发规范，作为通用组件包的交付格式。 |
| `clusters-computing-operator/internal/controller/tenantcluster_controller.go` | 多集群模式下下发组件的扩展点。 |

---

*本文档基于对仓库内 README、go.mod、Chart.yaml、源码目录与 CRD 的只读分析生成，未修改任何业务代码。*
