# openFuyao v26.12 特性用户指南文档优化需求设计

| 版本 | 完成日期 | 作者 | 变更说明 |
|------|------|------|---------|
| v1.0 | 2026-12-10 | 各开发组SIG & Docs SIG | 基于当前的特性用户指南平铺问题进行架构和单本资料的优化 |

## 1. 背景与目标

### 1.1 背景

openFuyao v26.12“特性用户指南”目录下现有31本特性文档，分布在`docs/docs/zh/user_guide`（本地2本）和5个SIG仓的`upstream`引用（跨仓29本）。

当前官网呈现问题如下：
- **整体架构**：31个条目全部平铺在`_toc.yaml`二级目录下，无二级分组，用户难以定位。
- **单本内容**：15本文档超过300行（最长1453行），缺少验证/卸载章节，特性介绍为超长单段落，参数表未按场景分组等问题。

### 1.2 目标

| 目标 | 衡量标准 |
|------|---------|
| 目录分类清晰可导航 | 31个条目按5个二级目录分组，每组≤10本，二级目录与SIG维护边界对齐 |
| 单本文档长度可控 | 拆分后主文档≤500行，参考类文档≤300行 |
| 章节结构统一 | 每本文档包含：特性介绍→安装→验证→使用→卸载/回退 |
| 内容质量达标 | 无文档低错、用户不易获取等问题 |
| 中英文一致 | 拆分后文档同步产出中英文版本 |

### 1.3 范围

- **涉及目录**：`docs/docs/zh/user_guide/` 及对应 `en/user_guide/`
- **涉及SIG仓**：sig-orchestration-engine、sig-ai-inference、sig-container-platform、sig-agent-sandbox、sig-large-scale-cluster
- **不涉及**：管理面操作类文档（注册登录、应用市场等），仅涉及特性用户指南部分。

## 2. 目录架构优化

### 2.1 现状问题

当前 `_toc.yaml` 共121行、29个section（含`超大规模集群`），全部平铺在二级目录下，无任何分组层级。条目来自6个不同SIG仓，用户无法感知特性归属，也难以按技术领域定位。

### 2.2 SIG与文档归属映射

31本特性文档来自6个SIG仓，分布如下：

| SIG仓 | SIG全称 | 文档数 | 涉及文档 |
|--------|--------|--------|---------|
| sig-ai-inference | AI推理SIG | 10 | AI推理赫尔墨斯路由、AI推理集成部署、AI推理鹰眼、AI推理弹性扩缩系统、AI推理权重分发加速、AI推理KVCache索引管理、Mooncake Store热点缓存优化、Mooncake昇腾NPU存储管理层、Mooncake P2P架构客户端远程调用、Mooncake Store V3架构双向数据传输 |
| sig-orchestration-engine | 编排引擎SIG | 15 | Serverlessdb Operator、容器启动时CPU加速、证书热更新、容器高密部署、kube-log-runner日志轮转、OS一致性检查工具、修改statefulset实现PVC扩容、npu-dra-plugin、vNPU、基于UB容器网络的设备插件、SuperPod、内存借用、内存共享、Agent沙箱Checkpoint与Restore、KubeVirt |
| sig-container-platform | 容器平台SIG | 2 | 日志框架ologger、鲲鹏机密容器 |
| sig-agent-sandbox | Agent沙箱SIG | 1 | FluxSandbox沙箱调度引擎 |
| docs仓（本地） | DOC SIG | 2 | KAE Operator、NPU Operator |
|sig-large-scale-cluster|超大规模集群|1（13）|超大规模集群用户指南、最佳实践|

**设计原则**：二级目录用用户可理解的技术领域命名，但与SIG维护边界对齐。sig-ai-inference10本文档收纳到二级目录AI推理与KVCache加速，sig-orchestration-engine的15本按技术领域拆分到3个二级目录（硬件设备管理/沙箱与虚拟化/集群与容器增强），避免单目录过臃肿；sig-agent-sandbox（1本）和sig-container-platform的机密容器（1本）合入沙箱与虚拟化目录，sig-large-scale-cluster中的超大规模集群归入集群与容器增强目录。

### 2.3 目标分类方案（5个二级目录，与SIG对齐）

```yaml
label: 特性用户指南
sections:
  # ============================================================
  # 二级目录1：AI推理与KVCache加速
  # 维护SIG：sig-ai-inference（纯净边界，独立维护）
  # 文档数：10本（5本平铺 + 1个三级目录AI推理KVCache含5本）
  # ============================================================
  - label: AI推理与KVCache加速
    sections:
      - label: AI推理赫尔墨斯路由
        href:
          upstream: https://gitcode.com/openFuyao/sig-ai-inference/blob/openFuyao-v26.12/docs/zh/ai_inference_hermes_routing/user_guide/ai_inference_hermes_routing.md
          path: ./
      - label: AI推理集成部署
        href:
          upstream: https://gitcode.com/openFuyao/sig-ai-inference/blob/openFuyao-v26.12/docs/zh/ai_inference_infernex/user_guide/_toc.yaml
          path: ./ai_inference_infernex
      - label: AI推理鹰眼
        href:
          upstream: https://gitcode.com/openFuyao/sig-ai-inference/blob/openFuyao-v26.12/docs/zh/ai_inference_eagle_eye/user_guide/eagle_eye_for_ai_inference.md
          path: ./
      - label: AI推理弹性扩缩系统
        href:
          upstream: https://gitcode.com/openFuyao/sig-ai-inference/blob/openFuyao-v26.12/docs/zh/ai_inference_elastic_scaling_system/_toc.yaml
          path: ./ai_inference_elastic_scaling_system
      - label: AI推理权重分发加速
        href:
          upstream: https://gitcode.com/openFuyao/sig-ai-inference/blob/openFuyao-v26.12/docs/zh/ai_inference_weight_dispatcher/ai_inference_weight_dispatcher.md
          path: ./
      # --- 三级目录：AI推理KVCache（5本）---
      - label: AI推理KVCache
        sections:
          - label: AI推理KVCache索引管理
            href:
              upstream: https://gitcode.com/openFuyao/sig-ai-inference/blob/openFuyao-v26.12/docs/zh/ai_inference_cache_indexer/user_guide/ai_inference_cache-indexer.md
              path: ./
          - label: Mooncake Store热点缓存优化
            href:
              upstream: https://gitcode.com/openFuyao/sig-ai-inference/blob/openFuyao-v26.12/docs/zh/mooncake_store_hotspot_cache_optimization/user_guide/mooncake_store_hotspot_cache_optimization.md
              path: ./
          - label: Mooncake昇腾NPU存储管理层
            href:
              upstream: https://gitcode.com/openFuyao/sig-ai-inference/blob/openFuyao-v26.12/docs/zh/mooncake_ascend_npu_storage_management_layer/user_guide/mooncake_ascend_npu_storage_management_layer.md
              path: ./
          - label: Mooncake P2P架构客户端远程调用
            href:
              upstream: https://gitcode.com/openFuyao/sig-ai-inference/blob/openFuyao-v26.12/docs/zh/mooncake_p2p_architecture_client_remote_invocation/user_guide/mooncake_p2p_architecture_client_remote_invocation.md
              path: ./
          - label: Mooncake Store V3架构双向数据传输
            href:
              upstream: https://gitcode.com/openFuyao/sig-ai-inference/blob/openFuyao-v26.12/docs/zh/mooncake_store_v3_bidirectional_transfer/user_guide/mooncake_store_v3_bidirectional_transfer.md
              path: ./

  # ============================================================
  # 二级目录2：NPU与硬件设备管理
  # 维护SIG：DOC SIG（KAE/NPU Operator 2本）+ sig-orchestration-engine（6本）
  # 文档数：8本（平铺，无三级，数量可接受）
  # 合并逻辑：KAE/NPU Operator与npu-dra-plugin/vNPU强关联，
  #          内存借用/共享基于UB硬件池化机制，同属硬件资源管理
  # ============================================================
  - label: NPU与硬件设备管理
    sections:
      - label: KAE Operator
        href: ./kae_operator.md
      - label: NPU Operator
        href: ./npu_operator.md
      - label: npu-dra-plugin
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/npu_dra_plugin_user_guide/npu_dra_plugin.md
          path: ./
      - label: vNPU
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/vnpu_user_guide/vnpu_user_guide.md
          path: ./
      - label: 基于UB容器网络的设备插件
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/device_plug_in_based_on_ub_container_network/device_plug_in_based_on_ub_container_network.md
          path: ./
      - label: SuperPod
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/superpod_user_guide/superpod_user_guide.md
          path: ./
      - label: 内存借用
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/memory_borrowing_user_guide/memory_borrowing.md
          path: ./memory_borrowing_user_guide
      - label: 内存共享
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/memory_sharing_user_guide/memory_sharing.md
          path: ./memory_sharing_user_guide

  # ============================================================
  # 二级目录3：沙箱与虚拟化
  # 维护SIG：sig-orchestration-engine（2本，牵头）+ sig-container-platform（1本）+ sig-agent-sandbox（1本）
  # 文档数：4本（平铺，无三级）
  # 合并逻辑：均提供隔离运行环境（沙箱/虚拟机/TEE），技术上有交叉
  # ============================================================
  - label: 沙箱与虚拟化
    sections:
      - label: Agent沙箱Checkpoint与Restore
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/sandbox_checkpoint_restore_user_guide/use_checkpoint_and_restore.md
          path: ./
      - label: FluxSandbox沙箱调度引擎
        href:
          upstream: https://gitcode.com/openFuyao/sig-agent-sandbox/blob/openFuyao-v26.12/doc/zh/flux_sandbox.md
          path: ./
      - label: KubeVirt
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/kubevirt_user_guide/kubevirt.md
          path: ./kubevirt_user_guide
      - label: 鲲鹏机密容器
        href:
          upstream: https://gitcode.com/openFuyao/sig-container-platform/blob/openFuyao-v26.12/docs/zh/kunpeng_confidential_container/user_guide/kunpeng_confidential_container.md
          path: ./

  # ============================================================
  # 二级目录4：集群与容器增强
  # 维护SIG：sig-orchestration-engine（5本，牵头）+ sig-container-platform（1本）+ sig-large-scale-cluster（1本）
  # 文档数：7本（平铺，无二级）
  # 合并逻辑：均为集群/容器层面的增强能力
  # ============================================================
  - label: 集群与容器增强
    sections:
      - label: 容器启动时CPU加速
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/cpu_acceleration_during_container_startup_user_guide/cpu_acceleration_during_container_startup.md
          path: ./
      - label: 容器高密部署
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/high_density_container_deployment_user_guide/high-density_container_deployment.md
          path: ./
      - label: 证书热更新
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/certificate_hot_reload_user_guide/certificate_hot_reload.md
          path: ./
      - label: kube-log-runner日志轮转可靠性增强
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/kube_log_runner_user_guide/kube-log-runner_log_rotation_reliability_enhancement.md
          path: ./
      - label: OS一致性检查工具
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/oschecktool_user_guide/os_consistency_check_tool.md
          path: ./
      - label: 修改statefulset实现PVC扩容
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/pvc_expansion_user_guide/pvc_expansion_through_statefulset_modification.md
          path: ./
      - label: 日志框架ologger
        href:
          upstream: https://gitcode.com/openFuyao/sig-container-platform/blob/openFuyao-v26.12/docs/zh/ologger/user_guide/ologger_log_framework.md
          path: ./
      - label: Serverlessdb Operator
        href:
          upstream: https://gitcode.com/openFuyao/sig-orchestration-engine/blob/openFuyao-v26.12/docs/zh/serverlessdb_operator_user_guide/serverlessdb_operator.md
          path: ./

  # ============================================================
  # 二级目录5：超大规模集群
  # 维护SIG：sig-large-scale-cluster
  # 文档数：1本（独立二级目录，与其他二级目录并列）
  # ============================================================
  - label: 超大规模集群
    sections:
      - label: 超大规模集群
        href:
          upstream: https://gitcode.com/openFuyao/sig-large-scale-cluster/blob/openFuyao-v26.12/docs/zh/_toc.yaml
          path: ./sig-large-scale-cluster
```

### 2.4 目录结构总览

```
特性用户指南
├── 二级目录1：AI推理与KVCache加速（10本，sig-ai-inference独立维护）
│   AI推理赫尔墨斯路由 / AI推理集成部署 / AI推理鹰眼
│   / AI推理弹性扩缩系统 / AI推理权重分发加速
│   └── 三级目录：AI推理KVCache（5本）
│       AI推理KVCache索引管理 / Mooncake Store热点缓存优化
│       / Mooncake昇腾NPU存储管理层 / Mooncake P2P架构客户端远程调用
│       / Mooncake Store V3架构双向数据传输
│
├── 二级目录2：NPU与硬件设备管理（8本，DOC SIG + sig-orchestration-engine）
│   KAE Operator / NPU Operator / npu-dra-plugin / vNPU
│   / 基于UB容器网络的设备插件 / SuperPod / 内存借用 / 内存共享
│
├── 二级目录3：沙箱与虚拟化（4本，sig-orchestration-engine牵头 + 2个小SIG）
│   Agent沙箱Checkpoint与Restore / FluxSandbox沙箱调度引擎
│   / KubeVirt / 鲲鹏机密容器
│
├── 二级目录4：集群与容器增强（8本，sig-orchestration-engine牵头 + sig-container-platform）
│   容器启动时CPU加速 / 容器高密部署 / 证书热更新
│   / kube-log-runner日志轮转 / OS一致性检查工具 / 修改statefulset实现PVC扩容
│   / 日志框架ologger / Serverlessdb Operator
│
└── 二级目录5：超大规模集群（1本，sig-large-scale-cluster独立维护）
    超大规模集群
```

### 2.5 SIG维护边界与协调说明

| 二级目录 | 主要维护SIG | 跨SIG文档 | 协调说明 |
|---------|-----------|----------|---------|
| AI推理与KVCache加速 | sig-ai-inference | 无 | 纯净边界，sig-ai-inference独立维护全部10本 |
| NPU与硬件设备管理 | DOC SIG + sig-orchestration-engine | docs仓2本（KAE/NPU Operator） + orchestration-engine 6本 | KAE/NPU Operator在docs仓，npu-dra-plugin/vNPU在orchestration-engine仓，需两SIG协同；内存借用/共享基于UB硬件池化机制，从原“集群增强”移入此目录 |
| 沙箱与虚拟化 | sig-orchestration-engine（牵头） | sig-container-platform 1本 + sig-agent-sandbox 1本 | 3个SIG各出1-2本，建议由orchestration-engine牵头维护目录结构 |
| 集群与容器增强 | sig-orchestration-engine（牵头） | sig-container-platform 1本 | 2个SIG，建议由orchestration-engine牵头维护目录结构 |
| 超大规模集群 | sig-large-scale-cluster | 无 | 纯净边界，sig-large-scale-cluster独立维护 |

### 2.6 与现状对比

| 维度 | 现状（31本全部平铺在二级目录） | 新方案（5个二级目录，SIG对齐） |
|------|-------------------------------|-------------------------------|
| 目录层级 | 1级（全部平铺，无分组） | 2级（5个二级目录，其中AI推理目录含1个三级目录（AI推理KVCache）） |
| 最大目录条目数 | 31本（全部平铺在一层） | 10本（AI推理与KVCache加速，有二级分组） |
| SIG维护边界 | 目录与SIG无对应关系，5个SIG的文档混杂排列 | 每个二级目录与SIG维护边界对齐，sig-ai-inference和sig-large-scale-cluster各独立维护1个目录 |
| sig-ai-inference（10本） | 10本与其他SIG文档混排，无分组 | 10本全部归在“AI推理与KVCache加速”二级目录下，独立维护 |
| sig-orchestration-engine（15本） | 15本与其他SIG文档混排，无分组 | 15本按技术领域拆到3个二级目录（硬件设备管理6本/沙箱与虚拟化2本/集群与容器增强7本） |
| sig-container-platform（2本） | 2本分散在不同位置 | 鲲鹏机密容器→沙箱与虚拟化；ologger→集群与容器增强 |
| sig-agent-sandbox（1本） | 1本混排在容器增强中 | 合入沙箱与虚拟化目录 |
| sig-large-scale-cluster（1本） | 1本混排在末尾 | 独立为二级目录“超大规模集群”，sig-large-scale-cluster独立维护 |
| NPU相关特性 | KAE/NPU Operator、npu-dra-plugin、vNPU分散在不同位置 | 4本NPU相关文档聚在“NPU与硬件设备管理”目录，用户从NPU Operator入口可顺次找到 |
| Mooncake系列（4本） | 4本与其他AI推理文档混排 | 归入“AI推理与KVCache加速”目录下的“AI推理KVCache”二级目录 |
| 内存借用/共享 | 与容器增强类混排 | 移入“NPU与硬件设备管理”（基于UB硬件池化机制） |
| 用户导航成本 | 高（31个条目平铺，需逐个扫描） | 中（5个目录各≤10本，可按技术领域快速定位） |

### 2.7 _toc.yaml 修复项

| 问题 | 位置 | 修复 |
|------|------|------|
| `AI推理KVCache索引管理` 缺少 `path` 字段 | 原第43-45行 | 补充 `path: ./` |
| `修改statefulset实现PVC扩容` path末尾多余空格 | 原第73行 | 去除尾随空格 |
| `npu-dra-plugin` path末尾多余空格 | 原第77行 | 去除尾随空格 |
| `vNPU` path末尾多余空格 | 原第81行 | 去除尾随空格 |
| `基于UB容器网络的设备插件` path末尾多余空格 | 原第85行 | 去除尾随空格 |

---

## 3. 过长文档拆分方案

### 3.1 需拆分文档清单（5本）

| 序号 | 文档 | 当前行数 | 拆分后主文档行数 | 拆出文件数 | 优先级 | 所属SIG仓 |
|------|------|---------|----------------|-----------|--------|-----------|
| 1 | NPU Operator | 1453 | ~400 | 2 | P0 | docs仓（本地） |
| 2 | AI推理鹰眼 | 1373 | ~500 | 2 | P0 | sig-ai-inference |
| 3 | KubeVirt | 1150 | ~400 | 2 | P1 | sig-orchestration-engine |
| 4 | FluxSandbox沙箱调度引擎 | 1109 | ~500 | 2 | P1 | sig-agent-sandbox |
| 5 | Serverlessdb Operator | 581 | ~350 | 1 | P3 | sig-orchestration-engine |

> **说明**：以上拆分方案为初步规划，具体拆分粒度和内容归属待与各文档责任人确认后最终确定。

### 3.2 逐本拆分方案

#### 3.2.1 NPU Operator（1453行 → 3本）

| 文件 | 内容 | 预估行数 |
|------|------|---------|
| `npu_operator.md`（主文档） | 特性介绍（拆为要点列表） + 组件表（增加"是否必装"列） + 版本配套关系 + 安装方式决策矩阵 + 验证 + 卸载 | ~400 |
| `npu_operator_installation.md` | 安装参考（包含全部安装方式：openFuyao平台在线/离线、独立部署在线/离线） + 常见定制选项表（按组件分组：NFD/驱动/设备插件/训练组件/vNPU/DRA） + 镜像列表 | ~600 |
| `npu_operator_usage.md` | 使用指南（NPU资源使用：整卡/硬切分/软切分 + vNPU使用 + DRA使用 + CEL表达式 + 拓扑亲和） | ~400 |

**拆分后 `_toc.yaml` 调整**：
```yaml
- label: NPU Operator
  href: ./npu_operator.md
  sections:
    - label: 安装参考
      href: ./npu_operator_installation.md
    - label: 使用指南
      href: ./npu_operator_usage.md
```

#### 3.2.2 AI推理鹰眼（1373行 → 3本）

| 文件 | 内容 | 预估行数 |
|------|------|---------|
| `eagle_eye_for_ai_inference.md`（主文档） | 特性介绍 + 安装（openFuyao平台/独立部署） + 使用指标监控 + 使用链路追踪（概览） | ~500 |
| `eagle_eye_metrics_reference.md` | 全量数据列表（HTML表格转Markdown多表，按指标类别分组） | ~400 |
| `eagle_eye_trace_reference.md` | vLLM-Ascend Span结构 + Span说明 + 数据流图 | ~300 |

#### 3.2.3 KubeVirt（1150行 → 3本）

| 文件 | 内容 | 预估行数 |
|------|------|---------|
| `kubevirt.md`（主文档） | 特性介绍（标题层级修正） + 安装 + 基础使用 + 验证 + 卸载 | ~400 |
| `kubevirt_advanced_features.md` | 网卡热插拔 + macvlan + SR-IOV + SR-IOV实时迁移 + 虚拟机热迁移 | ~500 |
| `kubevirt_ui_operations.md` | 使用界面操作管理虚机能力 | ~250 |

#### 3.2.4 FluxSandbox（1109行 → 3本）

| 文件 | 内容 | 预估行数 |
|------|------|---------|
| `flux_sandbox.md`（主文档） | 特性介绍 + 安装（在线/离线） + 快速开始 + 使用沙箱 | ~500 |
| `flux_sandbox_snapshot_warmup.md` | 快照预热（开启/创建/查询/常见错误） | ~300 |
| `flux_sandbox_api_reference.md` | SandboxGroup Spec/Status字段 + SDK操作与REST API对照 | ~300 |

#### 3.2.5 Serverlessdb Operator（581行 → 2本）

| 文件 | 内容 | 预估行数 |
|------|------|---------|
| `serverlessdb_operator.md`（主文档） | 特性介绍 + 前提条件 + 操作步骤（增加子标题导航） + 验证 + 卸载 | ~350 |
| `serverlessdb_operator_reference.md` | DBInstance状态说明 + HTTP API错误码 + Operator配置参数 | ~200 |

---

## 4. 单本内容优化

### 4.1 章节结构统一规范

每本特性文档应包含以下章节（按顺序）：

| 序号 | 章节标题 | 层级 | 说明 |
|------|---------|------|------|
| 1 | 特性介绍 | `##` | 包含应用场景、能力范围、亮点特征、实现原理、与相关特性的关系 |
| 2 | 前提条件 | `##` | 软硬件要求、依赖组件，若无特殊要求可写“无” |
| 3 | 安装 | `##` | 安装方式概览 + 步骤 |
| 4 | 使用[特性名] | `##` | 操作步骤 + 使用限制 + 相关操作 |
| 5 | 验证 | `##` | 独立验证章节，非嵌在步骤中 |
| 6 | 卸载/回退 | `##` | 组件卸载步骤 + 数据清理 |
| 7 | FAQ（可选） | `##` | 常见问题 |

**规则**：
- `前提条件`和`使用限制`若无特殊要求可写“无”
- `应用场景`/`能力范围`/`亮点特征`/`实现原理`统一为`###`级（在`## 特性介绍`下）
- 统一使用`前提条件`（不使用`前置条件`）

### 4.2 内容修复清单

> **说明**：以下修复清单为基于文档检视发现的问题汇总，具体修复优先级和方案待与各文档责任人确认后最终确定。

#### P0级修复（影响功能正确性）

| 序号 | 文档 | 问题 | 修复方案 |
|------|------|------|---------|
| 1 | 容器高密部署 | `背景信息`内容错误（从CPU加速文档复制粘贴了InPlacePodVerticalScaling） | 替换为高密部署的正确背景说明 |
| 2 | AI推理鹰眼 | 第937行代码围栏损坏 `` ```cd mo `` | 改为 `` ```bash `` |
| 3 | AI推理鹰眼 | 第730行拼写错误 `performace` | 改为 `performance` |
| 4 | Agent沙箱Checkpoint | 步骤编号冲突（第399行和第438行均为`4.3`） | 重新编号 |
| 5 | Agent沙箱Checkpoint | 第386行拼写错误 `pod_annnotations` | 改为 `pod_annotations` |
| 6 | KubeVirt | 第807行命令`kubectl label node node node...` | 改为 `kubectl label node <node-name>...` |
| 7 | KubeVirt | 第652行YAML缩进错误 | 修正`networkName`缩进到`multus:`下 |
| 8 | Mooncake Store热点缓存优化 | 环境变量名不一致：正文`MC_STORE_LOCAL_HOT_CACHE_SIZE` vs 代码`LOCAL_HOT_CACHE_SIZE` | 统一为同一个名称 |
| 9 | 内存借用 | `remote-mem-allocation-ratio`约束矛盾：max=25 vs (0,200] | 统一约束值 |
| 10 | 内存共享 | URL域名`atomgit.com`与其他文档`gitcode.com`不一致 | 统一为`gitcode.com` |
| 11 | 鲲鹏机密容器 | 第262行代码围栏4个反引号 | 改为3个反引号 |
| 12 | AI推理权重分发加速 | 第442行代码块缺少闭合反引号 | 补充闭合反引号 |
| 13 | 基于UB容器网络的设备插件 | 镜像tag硬编码`br_noncom_container_20260228` | 改为稳定tag |
| 14 | AI推理赫尔墨斯路由 | 表编号跳号：表6→表8（缺表7） | 重新编号表7 |
| 15 | SuperPod | 表编号跳号：缺表1和表4 | 重新编号 |
| 16 | KAE Operator | `RELEASE_TAG=v24.09`硬编码旧版本号 | 更新为当前版本 |

#### P1级修复（影响用户易用性）

| 序号 | 文档 | 问题 | 修复方案 |
|------|------|------|---------|
| 17 | NPU Operator | 特性介绍第8行400字超长单段落 | 拆为要点列表 |
| 18 | NPU Operator | 4种安装方式无决策矩阵 | 文件开头增加安装方式决策表格 |
| 19 | NPU Operator | 离线安装前提条件重复在线安装全部内容 | 改为引用方式 |
| 20 | KAE Operator | 开头无快速入门决策引导 | 增加平台部署vs Helm部署决策引导 |
| 21 | SuperPod | 特性介绍为200+字超长单段落 | 拆为2-3个子段落 |
| 22 | 内存借用 | 特性介绍仅1句话 | 补充UBS、UBSE、池化机制说明 |
| 23 | 内存共享 | 特性介绍仅1句话 | 补充内存池化、UBS-Core说明 |
| 24 | 内存共享 | `亮点特征`仅1条 | 补充完整 |
| 29 | 基于UB容器网络的设备插件 | `前置条件`→`前提条件` | 统一术语 |
| 30 | AI推理鹰眼 | `前置条件`与`前提条件`混用 | 统一为`前提条件` |

#### P2级修复（格式与规范）

| 序号 | 文档 | 问题 | 修复方案 |
|------|------|------|---------|
| 31 | 23/31本文档 | 图片alt文本为占位符`输入图片说明`/`image` | 替换为描述性文字 |
| 32 | NPU Operator | 常见定制选项表75行参数未分组 | 按组件分组（NFD/驱动/设备插件/训练/vNPU/DRA） |
| 33 | KAE Operator | 常见定制选项表28行参数未分组 | 按场景分组（基础/NFD/驱动/设备插件/安全） |
| 34 | 容器启动时CPU加速 | 新增参数表无表标题 | 补充`**表1**`标题 |
| 35 | 容器高密部署 | 新增参数表无表标题 | 补充`**表1**`标题 |
| 36 | OS一致性检查工具 | 表格无表标题 | 补充表标题 |
| 37 | OS一致性检查工具 | 图标题格式`**图1:**`不一致 | 改为`**图1**`（无冒号） |
| 38 | 证书热更新 | 能力范围表25+行过大 | 按组件拆分为多个小表 |
| 39 | ologger日志框架 | 表1单元格含HTML `<ul><li>` | 改为Markdown列表或纯文本 |
| 40 | 证书热更新/kube-log-runner | 实现原理含Go源码 | 移至开发指南或附录 |
| 41 | 鲲鹏机密容器 | 标题嵌套达6级`######` | 扁平化为`####` |
| 42 | KubeVirt | 示例值`ttl.sh/openfuyao`、`v17`为测试值 | 替换为正式镜像地址 |

#### 4.3 跨文档共性修复（全量31本）

> **说明**：以下共性修复项需各SIG统一推进，具体修复范围和优先级待与各SIG责任人确认。

| 序号 | 共性问题 | 影响文档数 | 修复方案 |
|------|---------|-----------|---------|
| 43 | 无卸载/回退章节 | 27/31 | 补充`## 卸载`章节（含Helm uninstall + CRD清理 + 数据清理） |
| 44 | 无独立验证章节 | 23/31 | 补充`## 验证`章节（含`kubectl get`验证命令） |
| 45 | 无`基本概念`章节 | 21/31 | 在特性介绍下补充`### 基本概念`，定义关键术语 |
| 46 | 无FAQ章节 | 25/31 | 补充`## FAQ`章节 |
| 47 | 应用场景过于简略（1-2句） | 18/31 | 扩充应用场景为3-5条具体场景 |
| 49 | 表格无表标题 | 9/31 | 补充`**表N**`标题 |
| 50 | 特性介绍为超长单段落 | 8/31 | 拆为要点列表 |
| 51 | 实现原理中包含源码 | 6/31 | 移至开发指南或附录 |

---

## 5. 实施计划

### 5.1 分阶段实施

本次优化分为两大里程碑：

- **里程碑一：架构修改完成** — 完成时间 **10月31日**
- **里程碑二：单本优化完成** — 完成时间 **12月10日**

| 阶段 | 内容 | 涉及文档数 | 里程碑 | 完成时间 |
|------|------|-----------|--------|---------|
| 阶段一 | `_toc.yaml` 目录重构（5个二级目录） + path字段修复 | 1文件 | 里程碑一 | 10月31日 |
| 阶段二 | P0级内容修复（功能正确性） | 12本 | 里程碑二 | 12月10日 |
| 阶段三 | 过长文档拆分（P0优先级） | 2本（NPU Operator + AI推理鹰眼） | 里程碑二 | 12月10日 |
| 阶段四 | 过长文档拆分（P1优先级） | 3本（KubeVirt + FluxSandbox + 赫尔墨斯路由） | 里程碑二 | 12月10日 |
| 阶段五 | P1级内容修复（用户易用性） | 14本 | 里程碑二 | 12月10日 |
| 阶段六 | 过长文档拆分（P2优先级） | 4本（npu-dra-plugin + Checkpoint + SuperPod + Serverlessdb） | 里程碑二 | 12月10日 |
| 阶段七 | P2级格式修复 + 跨文档共性修复 | 全部31本 | 里程碑二 | 12月10日 |
| 阶段八 | 中英文同步 | 拆分/修改的文档 | 里程碑二 | 12月10日 |

**里程碑一（10月31日前）交付内容**：
- `_toc.yaml` 完成目录架构重构，31个条目按5个二级目录分组
- path字段修复完成
- 各SIG确认目录归属和维护边界

**里程碑二（12月10日前）交付内容**：
- 5本过长文档拆分完成
- 51项内容修复全部完成（P0/P1/P2）
- 跨文档共性修复完成（卸载章节、验证章节、基本概念、FAQ等）
- 中英文同步完成

### 5.2 责任分工

| SIG组 | 负责二级目录 | 负责文档 | 文档数 |
|--------|------------|---------|--------|
| sig-ai-inference | AI推理与KVCache加速 | AI推理系列10本（独立维护，纯净边界） | 10 |
| DOC SIG + sig-orchestration-engine | NPU与硬件设备管理 | KAE/NPU Operator（docs仓2本）+ npu-dra-plugin/vNPU/UB设备插件/SuperPod/内存借用/内存共享（orchestration-engine 6本） | 8 |
| sig-orchestration-engine（牵头）+ sig-container-platform + sig-agent-sandbox | 沙箱与虚拟化 | Agent沙箱Checkpoint/Restore + KubeVirt（orchestration-engine 2本）+ 鲲鹏机密容器（container-platform 1本）+ FluxSandbox（agent-sandbox 1本） | 4 |
| sig-orchestration-engine（牵头）+ sig-container-platform | 集群与容器增强 | CPU加速/高密部署/证书热更新/kube-log-runner/OS检查/PVC扩容/Serverlessdb（orchestration-engine 7本）+ ologger（container-platform 1本） | 8 |
| sig-large-scale-cluster | 超大规模集群 | 超大规模集群（独立维护，纯净边界） | 1 |
| DOC SIG | _toc.yaml 目录重构 | 目录架构 + path字段修复 | 1文件 |

### 5.3 验收标准

| 验收项 | 标准 |
|--------|------|
| 目录架构 | `_toc.yaml` 包含5个二级目录（与SIG对齐），每组≤10本，AI推理目录含1个三级目录，超大规模集群独立为二级目录，无path字段问题 |
| SIG维护边界 | sig-ai-inference独立维护“AI推理与KVCache加速”目录；sig-large-scale-cluster独立维护“超大规模集群”目录；sig-orchestration-engine牵头维护其余3个目录 |
| 文档长度 | 拆分后主文档≤500行，参考文档≤300行 |
| 章节完整性 | 每本文档包含特性介绍→前提条件→安装→验证→卸载/回退 |
| 内容正确性 | 无P0级问题（代码围栏损坏、拼写错误、约束矛盾、命令错误等） |
| 格式规范 | 表格有标题、文档描述、结构风格一致 |
| 中英文一致 | 所有拆分/修改的文档同步产出中英文版本 |
