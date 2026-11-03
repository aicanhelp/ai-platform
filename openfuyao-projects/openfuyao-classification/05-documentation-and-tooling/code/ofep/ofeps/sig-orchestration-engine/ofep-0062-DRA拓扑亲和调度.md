---
title: DRA拓扑亲和调度
ofep-number: 0062
authors:
  - "@ruanhanqing"
owning-sig: sig-orchestration-engine
participating-sigs:

status: implementable
creation-date: 2025-07-20
reviewers:

approvers:

see-also:

replaces:

replaced-by:

stage: alpha
latest-milestone: "v26.09"
milestone:
  alpha: "v26.09"
  beta: "v26.12"
  stable: "v26.12"
feature-gates:
disable-supported: 
metrics:
---
<!--
**注意：**当你的 oFEP 完成时，应删除所有这些注释块。

要开始使用此模板：
- [ ] **选择一个托管 SIG。**
  确保该问题领域是 SIG 感兴趣的。如果没有 SIG 的赞助，oFEP 不应提交。
- [ ] **在 openfuyao/ofep 中创建问题**
  提交增强功能跟踪问题时，请务必填写该模板中的所有字段。其中一个字段要求提供指向 oFEP 的链接。你可以留空该字段，直到提交此 oFEP 后再返回到增强功能并添加链接。
- [ ] **复制此模板目录。**
  将此模板复制到所属 SIG 的目录中，并将其命名为"ofep-NNNN-short-descriptive-title"，其中"NNNN"是分配给上述增强功能的问题编号（没有前导零填充）。
- [ ] **尽可能多地填写以上oFEP元数据。**
  至少，你应该填写"标题"、"作者"、"SIG所有者"、"状态"和与日期相关的字段。
- [ ] **请尽可能详细地填写此文件。**
  至少，你应该填写"摘要"和"动机"部分。如果你已经和相关的 SIG 进行过前期沟通和想法验证，那么这两部分应该会很容易完成。
- [ ] **为此 oFEP 创建 PR。**
  将其指派给正在支持该流程的 SIG（特别兴趣小组）成员。
- [ ] **尽早合并并迭代。**
  避免纠结于具体细节，而应致力于明确 oFEP 的目标并快速合并。最好的方法是从概要部分开始，然后在后续的 PR 中逐步完善细节。

oFEP 合并并不意味着它已完成或获得批准。任何标记为"临时"的 oFEP 都是工作文档，可能会发生变更。你可以按以下方式标记正在积极讨论的部分：

编辑 oFEPS 时，请尽量使用范围明确、主题单一的 PR，以保持讨论的集中性。如果你不同意文档中已有的内容，请提交新的 PR 并提出修改建议。

一个 oFEP 对应其整个生命周期内的一项"功能"或"增强"。例如，从 Beta 版升级到 GA 版无需新的 oFEP。如果出现属于 oFEP 的新细节，请编辑 oFEP。一旦某个功能被"实现"，重大变更应该获得新的 oFEP。

最新（以及该文件的可能来源）的规范位置是[0000-ofep-template.md]（/0000-ofep-template.md）。

**注意：**任何将 oFEP 推进为"implementable"状态的 PR，或在标记为"implementable"后的重大更改，都必须得到每个 oFEP 批准人的批准。如果这些批准人均不合适（例如离开社区、角色变更等），则该列表的更改应由其余批准人和/或所属 SIG 批准。
-->
# oFEP-0062：DRA拓扑亲和调度

<!--
这是你的 oFEP 的标题。请保持简短、简洁且描述性强。一个好的标题可以帮助传达什么是 oFEP，以及有助于审查与追踪。
-->

<!--
目录（TOC）有助于快速跳转到 oFEP 的各个部分，同时突出显示超出标准模板所提供的其他信息。

确保目录已用<code>&lt;!-- toc --&rt;&lt;!-- /toc --&gt;</code>标签，然后用`hack/update-toc.sh`生成。
-->

<!-- toc -->
- [发布签核清单](#release-signoff-checklist)
- [摘要](#summary)
- [动机](#motivation)
  - [目标](#goals)
  - [非目标](#non-goals)
- [提案](#proposal)
  - [用户故事（可选）](#user-stories-optional)
    - [故事 1](#story-1)
    - [故事 2](#story-2)
    - [故事 3](#story-3)
  - [注释/约束/警告（可选）](#notesconstraintscaveats-optional)
  - [风险与缓解措施](#risks-and-mitigations)
- [设计细节](#design-details)
  - [拓扑关系查询](#拓扑关系查询)
  - [DRA驱动上报拓扑关系](#dra驱动上报拓扑关系)
  - [通过ResourceClaim实现亲和调度](#通过resourceclaim实现亲和调度)
  - [预置ResourceClaimTemplate](#预置resourceclaimtemplate)
  - [测试计划](#test-plan)
      - [先决条件测试更新](#prerequisite-testing-updates)
      - [单元测试](#unit-tests)
      - [集成测试](#integration-tests)
      - [e2e 测试](#e2e-tests)
  - [毕业标准](#graduation-criteria)
  - [升级/降级策略](#upgrade--downgrade-strategy)
  - [版本倾斜策略](#version-skew-strategy)
- [生产准备情况评审问卷](#production-readiness-review-questionnaire)
  - [功能启用和回滚](#feature-enablement-and-rollback)
  - [推出、升级和回滚规划](#rollout-upgrade-and-rollback-planning)
  - [监控要求](#monitoring-requirements)
  - [依赖项](#dependencies)
  - [可扩展性](#scalability)
  - [故障排除](#troubleshooting)
- [实施历史](#implementation-history)
- [缺点](#drawbacks)
- [替代方案](#alternatives)
- [所需基础设施（可选）](#infrastructure-needed-optional)
<!-- /toc -->

## 发布签核清单
<!--
**需要采取的行动：**为了将代码合并到一个版本中，在[openfuyao/ofep]引用此oFEP并在目标版本的[增强冻结]之前瞄准发布里程碑**中必须存在问题。

对于对核心代码或流程/程序进行更改的增强功能，例如：[openfuyao/openfuyao]，我们需要完成以下发布签署清单。

完成后勾选这些，以便发布团队跟踪。为了发布增强，必须更新这些检查表项。
-->

标记有（R）的项目*在达到里程碑/发布*之前是必需的。
- [ ]（R）发布里程碑中的增强问题，链接到 [openfuyao/ofep] 中的 oFEP 目录
- [ ] (R) oFEP 审批者已批准 oFEP 状态为"可实施"
- [ ] (R) 设计细节已适当记录
- [ ]（R）测试计划已到位，并考虑了 SIG 架构和 SIG 测试的输入（包括测试重构）
  - [ ] 针对所有 Beta API 操作（端点）进行 e2e 测试
  - [ ] (R) 确保 GA e2e 测试满足一致性测试的要求
  - [ ] (R) GA e2e 测试至少需要两周时间才能证明测试结果无 flake（不稳定或偶发失败）
- [ ] (R) 毕业标准已设定
  - [ ] (R) 所有 GA 端点必须通过[一致性测试]
- [ ] (R) 生产准备情况审查完成
- [ ] (R) 生产准备情况审查已获批准
- [ ] "实施历史"部分已更新里程碑
- [ ] 面向用户的文档已在 [openfuyao/docs] 创建，以便发布到 [openfuyao.com]
- [ ] 支持文档 - 例如，额外的设计文档、邮件列表讨论/SIG 会议链接、相关 PR/问题、发行说明

<!--
**注意：**此清单是迭代的，每次考虑将此增强功能作为里程碑时都应进行审查和更新。
-->

- [openfuyao.com](https://openfuyao.com/)
- [openfuyao/ofep](https://gitcode.com/openfuyao/ofep)
- [openfuyao/docs](https://gitcode.com/openfuyao/docs)

## 概括
<!--
这部分对于生成高质量、以用户为中心的文档（如发行说明或开发路线图）非常重要。应该在实现开始之前收集这些信息，以避免要求实现者在编写发行说明和实现功能本身之间分散注意力。oFEP 编辑器和SIG文档应该有助于确保"摘要"部分的语气和内容对广泛的受众有用。

好的摘要可能至少有一段长度。

在本节和下一节中，请遵循[文档样式指南]的指导方针。特别是，将代码行包装到合理的长度，使审阅者更容易引用特定的部分，并尽量减少更新的差异。
-->

在昇腾AI处理器集群环境中，不同NPU卡之间的拓扑连接方式以及NPU与CPU的NUMA亲和性对AI训练/推理任务的性能有显著影响。本提案旨在基于Kubernetes动态资源分配（DRA）机制，实现NPU设备的拓扑感知亲和调度，通过在ResourceSlice中上报拓扑关系属性，并利用ResourceClaim的属性约束能力，优化多卡AI训练/推理任务的设备分配策略，提升训练/推理性能。

## 动机
<!--
本节用于明确列出该oFEP的动机、目标和非目标。描述变更的重要性以及对用户的好处。
-->

### 目标
<!--
列出oFEP的具体目标。它想要达到什么目标？我们怎么知道这已经成功了？
-->

- DRA驱动能够上报NPU设备的拓扑关系属性（包括NUMA节点、HCCS环、PCIe Root等）
- 支持通过ResourceClaim约束实现多种拓扑亲和调度策略（同HCCS环、同NUMA节点、同PCIe Root等）
- 支持310p、910b、910c的拓扑关系解析和上报

### 非目标
<!--
什么超出了这个oFEP的范围？列出非目标有助于集中讨论并取得进展。
-->

- 不涉及跨节点的拓扑亲和调度（仅关注节点内拓扑）
- 不涉及DRA调度器节点打分能力的实现（优先占满原则）
- 不涉及异构设备间（如NPU与RDMA网卡）的亲和调度优化

## 提案
<!--
**这是我们真正进入提案具体内容的部分。**  
这一部分应包含足够的细节，使评审人员能清晰理解你到底在提出什么建议，  
但不应涉及 API 设计或具体实现细节。  

请阐明：
- **预期目标是什么？**  
- **我们如何衡量成功？**

请将更详细的设计和实现细节放在下方的 "设计细节" 部分中。
-->

### 用户故事（可选）
<!--
详细说明如果该 oFEP 被实施，用户将能够做哪些事情。请尽可能提供细节，以便人们理解系统将"如何"运作。此部分的目标是：让用户对提案有真实的感受，而不是陷入技术细节的泥淖中。
-->

#### 故事 1

AI训练团队需要在Atlas 200T A2 Box16服务器上运行一个8卡训练任务，希望8张NPU卡位于同一个HCCS环内以获得最佳通信性能。通过引用预置的ResourceClaimTemplate，团队可以直接申领8张卡，系统自动确保这8张卡来自同一个HCCS环，无需手动指定具体的NPU卡编号。

#### 故事 2

AI工程师需要运行一个跨环的12卡训练任务，希望将12张卡平均分配到两个HCCS环，并且环间对应的NPU卡通过PIX连接以获得较好的跨环通信性能。通过使用具有多个约束的ResourceClaim，工程师可以声明12张卡的拓扑约束，系统自动满足对称分配要求。

#### 故事 3

性能优化工程师需要验证NUMA亲和性对训练性能的影响。通过使用支持优先级排序的ResourceClaim，工程师可以声明优先选择同HCCS环的NPU卡，如果无法满足则退而求其次选择同NUMA节点的卡，系统按照优先级自动选择最优拓扑组合。

### 注释/限制/警告（可选）
<!--
该提案有哪些注意事项或潜在限制？
有没有上文未能充分表达的重要细节？
请在此处根据需要尽可能详细地展开说明。
这部分也非常适合用于讲解一些核心概念及它们之间的关联关系。
-->

**限制：**

- 不支持运行时动态调整已分配设备的拓扑约束

### 风险与缓解措施
<!--
这个提案存在哪些风险？我们将如何加以缓解？请从广泛的角度思考，例如包括安全性问题，以及它可能对更大范围的 openfuyao 生态系统产生的影响。 安全性将由谁进行评审，以及如何评审？用户体验（UX）将由谁进行评审，以及如何评审？建议考虑邀请 SIG 外部或子项目之外的相关人员参与评估。
-->

**风险1：拓扑信息上报不准确导致调度失败或性能下降**

缓解措施：
- DRA驱动启动时对拓扑信息进行校验，确保上报数据的准确性
- 提供拓扑信息查询和验证工具，方便用户核对

**风险2：约束过于严格导致资源碎片化**

缓解措施：
- 通过DRA框架的优先级排序机制，允许业务在严格约束无法满足时降级到较宽松约束

## 设计细节
<!--
本节应包含足够的信息，以便读者能够清楚理解你所提出的变更具体是什么。这可能包括 API 规格说明（虽然并非必须）或代码片段。如果对该提案将如何实施存在任何疑问，应在此处进行详细讨论。
-->

### 拓扑关系查询

通过以下命令可以查询昇腾NPU设备CPU和NPU的亲和性关系、多NPU之间的拓扑结构：

```bash
npu-smi info -t topo
```

拓扑关系说明：**按照性能从快到慢排序**，NPU卡之间的通信方式有以下几种。

- X：表示卡自身与自身通信
- HCCS：表示两张NPU卡之间通过HCCS连接，速度最快。
- PIX：在同一个NUMA节点下通过**单个PCIe switch**连接两个NPU之间的拓扑互联关系，速度极快
- PXB：在同一个NUMA节点下通过**多个PCIe switches**连接两个NPU之间的拓扑互联关系，速度较快
- PHB（PCIe Host Bridge）：在同一个NUMA节点下的NPU之间的拓扑互联关系，表示两张卡连接在同一个CPU的**同一个PCIe主机桥下**，数据传输不需要跨越NUMA节点，直接在PCIe局部总线层完成交互，通信速度较慢。
- SYS：表示两张卡位于不同的**不同的NUMA节点下**。通信速度最慢。

参考资料：[Atlas A2 中心推理和训练硬件 26.0.RCx npu-smi 命令参考 02 - 华为](https://support.huawei.com/enterprise/zh/doc/EDOC1100568421/20aa1a9d?idPath=23710424|251366513|254884019|261408772|261457531)

**示例1：一个有8张310P3 NPU卡（一卡两芯）的服务器，拓扑关系如下**

```
           NPU0       NPU1       NPU2       NPU3       NPU4       NPU5       NPU6       NPU7       CPU Affinity
NPU0       X          SYS        SYS        SYS        PHB        SYS        SYS        SYS        144-167
NPU1       SYS        X          SYS        SYS        SYS        PHB        SYS        SYS        96-119
NPU2       SYS        SYS        X          SYS        SYS        SYS        PHB        SYS        48-71
NPU3       SYS        SYS        SYS        X          SYS        SYS        SYS        PHB        0-23
NPU4       PHB        SYS        SYS        SYS        X          SYS        SYS        SYS        144-167
NPU5       SYS        PHB        SYS        SYS        SYS        X          SYS        SYS        96-119
NPU6       SYS        SYS        PHB        SYS        SYS        SYS        X          SYS        48-71
NPU7       SYS        SYS        SYS        PHB        SYS        SYS        SYS        X          0-23

Legend:

  X    = Self
  SYS  = Path traversing PCIe and NUMA nodes. Nodes are connected through SMP, such as QPI, UPI.
  PHB  = Path traversing PCIe and the PCIe host bridge of a CPU.
  HCCS = Connection traversing HCCS.
  NA   = Unknown relationship.
```

以上拓扑关系解读为：

8张NPU卡两两配对：

- NPU0和NPU4互为PHB：NPU0和NPU4插在同一个PCIe桥下。
- NPU1和NPU5互为PHB。
- NPU2和NPU6互为PHB。
- NPU3和NPU7互为PHB。

CPU亲和性关系：与NPU具有亲和性关系的CPU。

- NPU0和NPU4共享CPU亲和核心144-167。当NPU0和NPU4与CPU进行数据交互时，最快、路径最短的CPU逻辑核心编号是144-167。当某个AI进程绑定了NPU0或NPU4，那么把该进程的CPU核心限制在144-167范围内性能最好。
- NPU1和NPU5共享CPU亲和核心96-119
- NPU2和NPU6共享CPU亲和核心48-71
- NPU3和NPU7共享CPU亲和核心0-23

**示例2：一个有8张910B4卡的服务器（Atlas 800I A2 HCCS款），NPU拓扑关系输出如下**

```
[root@n2 ~]# npu-smi info -t topo
         NPU0       NPU1       NPU2       NPU3       NPU4       NPU5       NPU6       NPU7       CPU Affinity
NPU0       X          HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       144-167
NPU1       HCCS       X          HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       144-167
NPU2       HCCS       HCCS       X          HCCS       HCCS       HCCS       HCCS       HCCS       96-119
NPU3       HCCS       HCCS       HCCS       X          HCCS       HCCS       HCCS       HCCS       96-119
NPU4       HCCS       HCCS       HCCS       HCCS       X          HCCS       HCCS       HCCS       0-23
NPU5       HCCS       HCCS       HCCS       HCCS       HCCS       X          HCCS       HCCS       0-23
NPU6       HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       X          HCCS       48-71
NPU7       HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       X          48-71

Legend:

  X    = Self
  SYS  = Path traversing PCIe and NUMA nodes. Nodes are connected through SMP, such as QPI, UPI.
  PHB  = Path traversing PCIe and the PCIe host bridge of a CPU.
  PIX  = Path traversing a single PCIe switch
  PXB  = Path traversing multipul PCIe switches
  HCCS = Connection traversing HCCS.
  NA   = Unknown relationship.
```

NPU-to-NPU拓扑关系解读：

8张NPU卡通过HCCS组成一个8P Full Mesh完全互联架构，任意NPU卡之间都是直达的。

说明：一些老款Atlas 800训练服务器是4P Full Mesh架构，4张NPU卡组成HCCS环，环间通过PCIe通信。

CPU-to-NPU亲和关系解读：

4个CPU，每个物理CPU直连2张NPU卡，0和1，2和3，4和5，6和7分别共享一个NUMA节点。



**示例3：一个Atlas 200T A2 Box16异构子框服务器的拓扑关系如下**

```
        NPU0    NPU1    NPU2    NPU3    NPU4    NPU5    NPU6    NPU7    NPU8    NPU9    NPU10    NPU11    NPU12    NPU13    NPU14    NPU15    CPU Affinity
NPU0    X       HCCS    HCCS    HCCS    HCCS    HCCS    HCCS    HCCS    PIX     PHB     PHB      PHB      SYS      SYS      SYS      SYS      0-51,104-155
NPU1    HCCS    X       HCCS    HCCS    HCCS    HCCS    HCCS    HCCS    PHB     PIX     PHB      PHB      SYS      SYS      SYS      SYS      0-51,104-155
NPU2    HCCS    HCCS    X       HCCS    HCCS    HCCS    HCCS    HCCS    PHB     PHB     PIX      PHB      SYS      SYS      SYS      SYS      0-51,104-155
NPU3    HCCS    HCCS    HCCS    X       HCCS    HCCS    HCCS    HCCS    PHB     PHB     PHB      PIX      SYS      SYS      SYS      SYS      0-51,104-155
NPU4    HCCS    HCCS    HCCS    HCCS    X       HCCS    HCCS    HCCS    SYS     SYS     SYS      SYS      PIX      PHB      PHB      PHB      52-103,156-207
NPU5    HCCS    HCCS    HCCS    HCCS    HCCS    X       HCCS    HCCS    SYS     SYS     SYS      SYS      PHB      PIX      PHB      PHB      52-103,156-207
NPU6    HCCS    HCCS    HCCS    HCCS    HCCS    HCCS    X       HCCS    SYS     SYS     SYS      SYS      PHB      PHB      PIX      PHB      52-103,156-207
NPU7    HCCS    HCCS    HCCS    HCCS    HCCS    HCCS    HCCS    X       SYS     SYS     SYS      SYS      PHB      PHB      PHB      PIX      52-103,156-207
NPU8    PIX     PHB     PHB     PHB     SYS     SYS     SYS     SYS     X       HCCS    HCCS     HCCS     HCCS     HCCS     HCCS     HCCS     0-51,104-155
NPU9    PHB     PIX     PHB     PHB     SYS     SYS     SYS     SYS     HCCS    X       HCCS     HCCS     HCCS     HCCS     HCCS     HCCS     0-51,104-155
NPU10   PHB     PHB     PIX     PHB     SYS     SYS     SYS     SYS     HCCS    HCCS    X        HCCS     HCCS     HCCS     HCCS     HCCS     0-51,104-155
NPU11   PHB     PHB     PHB     PIX     SYS     SYS     SYS     SYS     HCCS    HCCS    HCCS     X        HCCS     HCCS     HCCS     HCCS     0-51,104-155
NPU12   SYS     SYS     SYS     SYS     PIX     PHB     PHB     PHB     HCCS    HCCS    HCCS     HCCS     X        HCCS     HCCS     HCCS     52-103,156-207
NPU13   SYS     SYS     SYS     SYS     PHB     PIX     PHB     PHB     HCCS    HCCS    HCCS     HCCS     HCCS     X        HCCS     HCCS     52-103,156-207
NPU14   SYS     SYS     SYS     SYS     PHB     PHB     PIX     PHB     HCCS    HCCS    HCCS     HCCS     HCCS     HCCS     X        HCCS     52-103,156-207
NPU15   SYS     SYS     SYS     SYS     PHB     PHB     PHB     PIX     HCCS    HCCS    HCCS     HCCS     HCCS     HCCS     HCCS     X        52-103,156-207
```

NPU-to-NPU拓扑关系解读：

16张卡组成两个HCCS环，环间通过PCIe Switch互联。

- NPU0~NPU7组成第一个HCCS环。NPU8~NPU15组成第二个HCCS环。
- NPU0和NPU8互为PIX，表示这两张卡插在同一个PCIe Switch下。NPU1和NPU9，。。。，NPU7和NPU15两两配对，都是通过PIX连接。
- NPU0和NPU9/10/11通过PHB互联，表示处于同一个NUMA节点下，但是走物理CPU的PCIe桥，性能比PIX慢。
- NPU0和NPU12/13/14/15通过SYS互联，表示处于不同的NUMA节点下，性能最差。

CPU亲和性分析：

这台服务器有2个超大的NUMA节点。

NUMA 0（核心0-51,104-155）：亲和NPU 0/1/2/3/8/9/10/11

NUMA 1（核心52-103,156-207）：亲和NPU 4/5/6/7/12/13/14/15

针对这种拓扑关系，亲和调度应该分优先级：

第一优先级：HCCS环内分配。如果申请小于等于8张卡，应该优先在同一个HCCS环内分配。如果小于等于4张卡，再考虑NUMA亲和，应该优先分配同一个NUMA节点的卡，例如NPU 0/1/2/3。

第二优先级：跨HCCS环的对称分配。如果申请大于8张卡，应该优先保证环间通信走PIX通道，例如NPU0和NPU8。



**示例4：Atlas 800T A3 超节点的拓扑关系示例如下**

```
           Phy-ID0    Phy-ID1    Phy-ID2    Phy-ID3    Phy-ID4    Phy-ID5    Phy-ID6    Phy-ID7
Phy-ID0    X          SIO        HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW
Phy-ID1    SIO        X          HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW
Phy-ID2    HCCS_SW    HCCS_SW    X          SIO        HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW
Phy-ID3    HCCS_SW    HCCS_SW    SIO        X          HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW
Phy-ID4    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    X          SIO        HCCS_SW    HCCS_SW
Phy-ID5    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    SIO        X          HCCS_SW    HCCS_SW
Phy-ID6    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    X          SIO
Phy-ID7    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    HCCS_SW    SIO        X
Legend:
  X    = Self
  SYS  = Path traversing PCIe and NUMA nodes. Nodes are connected through SMP, such as QPI, UPI.
  PHB  = Path traversing PCIe and the PCIe host bridge of a CPU.
  PIX  = Path traversing a single PCIe switch
  PXB  = Path traversing multipul PCIe switches
  HCCS = Connection traversing HCCS.
  SIO  = Path traversing the SIO bus
  HCCS_SW = Connection traversing HCCS through a switch
  NA   = Unknown relationship.
```

拓扑关系解读：

- 两个昇腾AI处理器之间通过SIO互联形成一个HiAM模组，例如NPU0和NPU1形成一个HiAM模组，NPU2和NPU3形成一个HiAM模组。
- HiAM模组之间通过HCCS互联，HCCS_SW表示通过HCCS经交换机连接的两个NPU之间的拓扑互联关系。
- 同一个NPU与所有CPU的亲和性相同，不体现NUMA亲和性关系。


### DRA驱动上报拓扑关系

DRA驱动需将拓扑关系作为ResourceSlice的属性上报。针对NUMA节点、HCCS环、PIX、PXB、PHB、SYS不同的连接方式，定义对应的属性，呈现完整拓扑结构。

- resource.kubernetes.io/pciBusID：Kubernetes社区定义的[DRA标准属性](https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/4381-dra-structured-parameters)，表示设备的PCI总线地址，用于唯一设备分配。pciBusID可通过`npu-smi info`命令获取（Bus-Id）。
- resource.kubernetes.io/pcieRoot：Kubernetes社区定义的[DRA标准属性](https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/4381-dra-structured-parameters)，表示设备的PCIe根复合体。一个NUMA节点下可以有多个PCIe Root。PIX连接即表示两个NPU卡处于相同的PCIe Root。通过约束具有相同的PCIe Root可以实现NPU和RDMA NIC之间的异构设备亲和调度，参考[issue213](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu/issues/213)。pcieRoot可通过`readlink -f /sys/bus/pci/devices/${pciBusID}`获取设备的物理绝对路径，再通过awk抓取含有`pciXXXX:XX`的根节点。
- numaNode：表示NPU卡所属的numa节点。如果没有，则不上报该属性。Kubernetes社区的[6072-dra-standard-numanode](https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/6072-dra-standard-numanode)提案计划将numaNode作为DRA标准属性，当前pr还处于open状态，因此，暂时还是使用厂商自定义的属性，待Kubernetes社区支持标准numaNode属性后，再进行迁移。获取方式：`cat /sys/bus/pci/devices/${pciBusID}/numa_node`。
- topoHccsRingID：表示NPU卡所属的HCCS环。如果没有，则不上报该属性。获取方式：通过解析`npu-smi info -t topo`命令输出。
- topoSioID：表示NPU卡所属的SIO bus ID。如果没有，则不上报该属性。获取方式：通过解析`npu-smi info -t topo`命令输出。
- topoPixID：表示NPU卡所属的pcie-switch-pair-id。通过PIX连接的设备的pcieRoot相同，因此可以使用resource.kubernetes.io/pcieRoot字段替代。当前可不上报该属性。
- topoPxbID：表示NPU卡所属的pxb ID。典型昇腾服务器当前没有看到有PXB连接方式。因此，该字段当前也可以不上报。
- topoPhbID：表示NPU卡所属的phb ID。通过PHB连接的设备处于同一个NUMA节点，因此可以使用numaNode字段替代。当前可不上报该属性。
- sys：sys是最慢的连接方式，没有必要设置对应的属性。

**说明**：

1. HCCS环拓扑关系需通过解析环境真实topo关系获得，不能根据服务器型号硬编码，同时要考虑不同服务器的拓扑差异。
2. 优先调用DCMI接口获取NPU相关信息，次选调用npu-smi命令获取。


示例：以上面的16卡服务器为例，各卡取值如下：

| 属性           | NPU0 | NPU1 | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| numaNode       | 0    | 0    | 0    | 0    | 1    | 1    | 1    | 1    | 0    | 0    | 0    | 0    | 1    | 1    | 1    | 1    |
| topoHccsRingID | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 1    | 1    | 1    | 1    | 1    | 1    | 1    |
| topoPixID      | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
| topoPxbID      | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |
| topoPhbID      | 0    | 0    | 0    | 0    | 1    | 1    | 1    | 1    | 0    | 0    | 0    | 0    | 1    | 1    | 1    | 1    |



ResourceSlice资源参考：

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceSlice
metadata:
  creationTimestamp: "2026-02-27T01:55:37Z"
  generateName: master-npu.huawei.com
  generation: 1
  name: master-npu.huawei.com-9gv8l
  ownerReferences:
  - apiVersion: v1
    controller: true
    kind: Node
    name: master
    uid: 6ef76e72-da36-44e3-b9c3-93f44684a859
  resourceVersion: "2225369"
  uid: 0c1b399e-4fa8-4279-93f8-b92a1faeff6f
spec:
  devices:
  - attributes:
      chipName:
        string: 910B4
      physicalId:  # NPU 0
        int: 0
      resource.kubernetes.io/pciBusID:   
        string: '0000:81:00:0'
      resource.kubernetes.io/pcieRoot:   
        string: 'pci0000:80'
      numaNode:   # NPU卡所属numa节点
        int: 0
      topoHccsRingID:  # NPU卡所属的HCCS环
        int: 0
    capacity:
      memCapacity:
        value: 32Gi
    name: npu-0
  - attributes:
      chipName:
        string: 910B4
      physicalId:  # NPU 8
        int: 8  
      resource.kubernetes.io/pciBusID:   
        string: '0000:91:00:0'
      resource.kubernetes.io/pcieRoot:   
        string: 'pci0000:80'
      numaNode:   # NPU卡所属numa节点
        int: 0
      topoHccsRingID:  # NPU卡所属的HCCS环
        int: 1
    capacity:
      memCapacity:
        value: 32Gi
    name: npu-8
  - attributes:
      chipName:
        string: 910B4
      physicalId:  # NPU 11
        int: 11
      resource.kubernetes.io/pciBusID:   
        string: '0000:95:00:0'
      resource.kubernetes.io/pcieRoot:   
        string: 'pci0000:94'
      numaNode:   # NPU卡所属numa节点
        int: 0
      topoHccsRingID:  # NPU卡所属的HCCS环
        int: 1
    capacity:
      memCapacity:
        value: 32Gi
    name: npu-11
  - attributes:
      chipName:
        string: 910B4
      physicalId:  # NPU 14
        int: 14
      resource.kubernetes.io/pciBusID:   
        string: '0000:99:00:0'
      resource.kubernetes.io/pcieRoot:   
        string: 'pci0000:98'
      numaNode:   # NPU卡所属numa节点
        int: 1
      topoHccsRingID:  # NPU卡所属的HCCS环
        int: 1
    capacity:
      memCapacity:
        value: 32Gi
    name: npu-14
... ...   # 其它NPU不一一罗列
  driver: npu.huawei.com
  nodeName: master
  pool:
    generation: 1
    name: master
    resourceSliceCount: 1
```

### 通过ResourceClaim实现亲和调度

利用DRA的constraints约束能力，可以实现多卡亲和调度。

例如，POD声明8张卡，要求8张卡处于同一个HCCS环内。配置参考如下：

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: npu-topo-affinity-claim
  namespace: ascend-dra
spec:
  devices:
    requests:
    - name: npu 
      exactly:
        deviceClassName: npu.huawei.com
        allocationMode: ExactCount
        count: 8
        selectors:
        - cel:
            expression: |-
              device.attributes["npu.huawei.com"].chipName == '910B4'
    constraints:  # 约束8张卡都具有topoHccsRingID属性，并且属性值相同
    - requests: ["npu"]
      matchAttribute: "npu.huawei.com/topoHccsRingID"
    
```

POD声明4张卡，要求4张卡处于同一个HCCS环内，并且处于同一个NUMA节点。配置参考如下：

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: npu-topo-affinity-claim
  namespace: ascend-dra
spec:
  devices:
    requests:
    - name: npu 
      exactly:
        deviceClassName: npu.huawei.com
        allocationMode: ExactCount
        count: 4
        selectors:
        - cel:
            expression: |-
              device.attributes["npu.huawei.com"].chipName == '910B4'
    constraints:  
    - requests: ["npu"]
      matchAttribute: "npu.huawei.com/topoHccsRingID"
    - requests: ["npu"]
      matchAttribute: "npu.huawei.com/numaNode"
    
```

利用DRA的优先级排序能力([KEP-4816](https://github.com/kubernetes/enhancements/tree/master/keps/sig-scheduling/4816-dra-prioritized-list))，可以实现POD声明2张卡，优先要求2张卡处于同一个HCCS环内，如果满足不了，则退而求其次，要求2张卡通过单个PCIe switch连接。

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: npu-topo-affinity-claim
  namespace: ascend-dra
spec:
  devices:
    requests:
    - name: npu
      firstAvailable:
      - name: npu-same-hccs-ring 
        deviceClassName: npu.huawei.com
        allocationMode: ExactCount
        count: 2
        selectors:
        - cel:
            expression: |-
              device.attributes["npu.huawei.com"].chipName == '910B4'
      - name: npu-same-pcieroot
        deviceClassName: npu.huawei.com
        allocationMode: ExactCount
        count: 2
        selectors:
        - cel:
            expression: |-
              device.attributes["npu.huawei.com"].chipName == '910B4'
    constraints:  
    - requests: ["npu/npu-same-hccs-ring"]
      matchAttribute: "npu.huawei.com/topoHccsRingID"
    - requests: ["npu/npu-same-pcieroot"]
      matchAttribute: "resource.kubernetes.io/pcieRoot" 
    
```

POD申领12张卡，要求每个HCCS环内分配6张，并且环间两两配对。

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: npu-topo-affinity-claim
  namespace: ascend-dra
spec:
  devices:
    requests:
    - name: npu1 
      exactly:
        deviceClassName: npu.huawei.com
        allocationMode: ExactCount
        count: 1
        selectors:
        - cel:
            expression: |-
              device.attributes["npu.huawei.com"].chipName == '910B4'
    - name: npu2 
      exactly:
        deviceClassName: npu.huawei.com
        allocationMode: ExactCount
        count: 1
        selectors:
        - cel:
            expression: |-
              device.attributes["npu.huawei.com"].chipName == '910B4'
    # npu3 ~ npu11 不一一罗列
    - name: npu12 
      exactly:
        deviceClassName: npu.huawei.com
        allocationMode: ExactCount
        count: 1
        selectors:
        - cel:
            expression: |-
              device.attributes["npu.huawei.com"].chipName == '910B4'
    constraints:  
    - requests: ["npu1,npu2,npu3,npu4,npu5,npu6"]
      matchAttribute: "npu.huawei.com/topoHccsRingID"
    - requests: ["npu7,npu8,npu9,npu10,npu11,npu12"]
      matchAttribute: "npu.huawei.com/topoHccsRingID"
    - requests: ["npu1,npu7"]
      matchAttribute: "resource.kubernetes.io/pcieRoot"
    - requests: ["npu2,npu8"]
      matchAttribute: "resource.kubernetes.io/pcieRoot"
    ... # npu3，npu4，npu5，npu6，npu9，npu10，npu11，npu12类似一一配对
```

业务可以针对各种服务器硬件预置典型场景的ResourceClaimTemplate，业务POD直接引用即可。

说明：DRA driver组件不负责预置典型场景的ResourceClaimTemplate，仅提供examples，部署DRA driver也不会部署这些ResourceClaimTemplate。

### 参考资料
#### 昇腾MindCluster节点内亲和调度参考

昇腾MindCluster基于设备插件和volcano调度器扩展实现了[节点内的NPU亲和调度和跨节点的亲和调度](https://www.hiascend.com/document/detail/zh/mindcluster/2600/clustersched/dlug/docs/zh/scheduling/usage/basic_scheduling/01_affinity_scheduling/03_ascend_ai_processor_based_affinity.md)。

针对每种硬件服务器，因为拓扑结构不同，所以提供了不同的拓扑亲和调度策略和算法。

以“Atlas 200T A2 Box16 异构子框和Atlas 200I A2 Box16 异构子框”为例。

| 优先级 | 策略名称         | 策略描述                                                     |
| ------ | ---------------- | ------------------------------------------------------------ |
| 1      | HCCS互联分配原则 | 如果申请昇腾AI处理器的个数为1~8，则需要调度到同一个HCCS互联。如果申请昇腾AI处理器的个数为10、12、14，需要将所需的昇腾AI处理器平均分配到两个环，相对的物理地址也一致。 |
| 2      | 优先占满原则     | 优先分配已经分配过昇腾AI处理器的节点，减少碎片。以1、2、4、8为例，具体如下：<br/>如果申请1个昇腾AI处理器，优先申请HCCS互联可用昇腾AI处理器数量为1的节点，其次是可用数量为2个，3个，一直到8个。相同数量优先选择节点昇腾AI处理器总数量少的节点。<br/>如果申请2个昇腾AI处理器，优先申请HCCS互联可用昇腾AI处理器数量为2的节点，其次是可用数量为3个，4个，一直到8个。相同数量优先选择节点昇腾AI处理器总数量少的节点。<br/>如果申请4个昇腾AI处理器，优先申请HCCS互联可用昇腾AI处理器数量为4的节点，其次是可用数量为5个，6个，一直到8个。相同数量优先选择节点昇腾AI处理器总数量少的节点。<br/>如果申请8个昇腾AI处理器，只申请HCCS互联可用昇腾AI处理器数量为8的节点。相同数量优先选择节点昇腾AI处理器总数量少的节点。<br/>下发分布式任务时，任务存在未按照优先占满调度原则占满某个节点。说明如下：<br/>现象说明：如在两台Atlas 200T A2 Box16 异构子框或Atlas 200I A2 Box16 异构子框集群中，同时下发5卡、4卡、3卡任务，存在4卡和3卡任务调度到同一个节点，5卡任务调度到另一个节点的问题。<br/>原因分析：因为Volcano调度完一个任务后，Ascend Device Plugin上报调度后的昇腾AI处理器的拓扑结构到mindx-dl-deviceinfo-${node_name}存在时延，导致Volcano校验该节点昇腾AI处理器数量失败，将任务调度到其他节点上。 |

资源申请约束：

根据业务模型，对Atlas 200T A2 Box16 异构子框和Atlas 200I A2 Box16 异构子框训练任务资源申请作如下要求：

- 训练任务申请的昇腾AI处理器数量不能大于节点昇腾AI处理器总数。
- 训练任务申请的昇腾AI处理器数量为1~8、10、12、14和16。
- 当训练任务申请的昇腾AI处理器数量不大于8个时，需要选取同一个HCCS互联内的昇腾AI处理器。
- 当训练任务申请的昇腾AI处理器数量为10、12、14时，需要将所需的昇腾AI处理器平均分配到两个环，相对的物理地址也一致。
- 当训练任务申请的昇腾AI处理器数量为16个时，需要将节点的昇腾AI处理器全部分配给该任务。
- 遵循Volcano开源部分的其他约束。

**总结：**

1. 拓扑亲和调度，优先调度到同一个HCCS环，其次调度到使用PCIe互联的NPU。
2. 申请的昇腾AI处理器数量只能为1~8、10、12、14和16。小于8个时，必须在同一个HCCS环内分配。超过8个时，将所需的昇腾AI处理器平均分配到两个环，相对的物理地址也一致。例如声明10个，则分配0,1,2,3,4,8,9,10,11,12。
3. 支持优先占满原则，依赖节点打分。DRA当前尚不支持节点打分能力（参考[issue4970](https://github.com/kubernetes/enhancements/issues/4970)）
4. 拓扑关系根据NPU的logicID硬编码，非动态获取。

其它硬件服务器不一一说明，请参考MindCluster文档。


#### DRA亲和调度业界参考

[AWS Neuron Dynamic Resource Allocation (DRA) Beta](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.27.0/containers/neuron-dra.html)  AWS Neuron DRA亲和调度

[Simplify AI infrastructure for AWS Trainium and Elastic Fabric Adapter with Kubernetes Dynamic Resource Allocation](https://aws.amazon.com/cn/blogs/containers/simplify-ai-infrastructure-for-aws-trainium-and-elastic-fabric-adapter-with-kubernetes-dynamic-resource-allocation/) 异构设备亲和调度

[Optimizing RDMA performance for AI workloads on AKS with DRANET](https://blog.aks.azure.com/2026/04/01/dranet-rdma-optimization-for-ai-on-aks) GPU和RDMA亲和调度

### 测试计划
<!--
**注意：**在该提案尚未被纳入某个正式版本前，此部分不是必需的。
其目标是确保我们不会接收缺乏充分测试的增强功能。
所有代码都应具备充分的测试（最终也应满足测试覆盖率要求）。在撰写测试计划时，请遵循 openFuyao 测试指南。
-->

[ ] 我/我们理解，相关组件的所有者可能会要求更新已有的测试，以便在提交实现该增强功能所需的更改之前，使代码达到足够稳固的质量标准。

##### 先决条件测试更新
<!--
根据评审者的反馈，描述在实施此项增强功能之前需要补充哪些额外的测试，以确保该增强特性也具备稳固的基础。
-->

##### 单元测试
<!--
原则上，所有新增的代码都应具有完整的单元测试覆盖率，因此列出确切的测试项并不会带来额外价值。
但如果无法实现完整的单元测试覆盖，请说明原因，并解释为何在这种情况下这是可以接受的。
-->

- 

单元测试覆盖：
- 拓扑信息采集和解析逻辑
- ResourceSlice属性上报逻辑

##### 集成测试
<!--
集成测试允许控制用于启动被测二进制文件的配置参数。
这与不允许配置参数的 e2e 测试不同。
这样做可以测试非默认选项以及多个不同的、可能冲突的命令行选项。

如果集成测试不是必要的或有用的，请解释原因。
-->

- 测试名称：`dra-topology-integration`

集成测试场景：
- DRA驱动正确上报拓扑属性到ResourceSlice
- ResourceClaim约束能够正确限制设备分配
- 不同拓扑约束组合的设备分配结果符合预期

##### e2e 测试
<!--
当该功能计划纳入某个正式版本时，应填写此问题。
- 对于 Alpha 阶段，请描述将添加哪些测试，以确保该增强功能具备良好的质量保障。
- 对于 Beta 和 GA 阶段，需要说明测试已经编写、被定期执行，且结果稳定。

可通过以下方式提供证明材料： 
- 指向 gitcode 源代码的永久链接
- 指向定期测试作业的链接，并按测试名称过滤
- 在 openfuyao 缺陷追踪工具中进行搜索。

作为进入 GA（正式可用）阶段的标准，我们期望过去一个月内不存在任何非基础设施相关的 flaky 测试（不稳定测试）。
如果你认为无需添加端到端测试（e2e），请解释其原因及合理性。
-->

| 用例名称 | 前置条件 | 用例步骤 | 后置操作 | 预期结果 |
|---------|---------|---------|---------|---------|
| 单HCCS环8卡调度 | 集群有Atlas 200T A2 Box16节点 | 1.创建8卡ResourceClaim（同HCCS环约束）<br>2.创建Pod引用该ResourceClaim<br>3.检查分配的NPU设备 | 删除Pod和ResourceClaim | 分配的8张NPU位于同一个HCCS环 |
| NUMA亲和4卡调度 | 集群有Atlas 200T A2 Box16节点 | 1.创建4卡ResourceClaim（同HCCS环+同NUMA约束）<br>2.创建Pod引用该ResourceClaim<br>3.检查分配的NPU设备 | 删除Pod和ResourceClaim | 分配的4张NPU位于同一个HCCS环且同一个NUMA节点 |
| 跨环对称12卡调度 | 集群有Atlas 200T A2 Box16节点 | 1.创建12卡ResourceClaim（双环对称约束）<br>2.创建Pod引用该ResourceClaim<br>3.检查分配的NPU设备 | 删除Pod和ResourceClaim | 分配的12张NPU平均分配到两个HCCS环 |
| 优先级调度降级测试 | 集群有Atlas 200T A2 Box16节点，HCCS环已占用 | 1.创建优先级ResourceClaim<br>2.创建Pod引用该ResourceClaim<br>3.检查分配的NPU设备 | 删除Pod和ResourceClaim | 无法满足高优先级约束时，降级到低优先级约束成功分配 |

- 测试名称：`dra-topology-affinity-e2e`

### 毕业标准
<!--
> **注意：** *在功能尚未计划纳入某个正式版本时，本节无需填写。*
> 请在此处定义该功能的毕业（Graduation）里程碑。
> 毕业条件可以基于 API 成熟度、[Feature Gate] 的推进阶段，或其他方式来定义。此处应保持高层次，重点说明评估是否可以毕业时将参考哪些信号（信心指标）。
> 在制定毕业标准时，请参考以下内容：
> - [成熟度等级（`alpha`、`beta`、`stable`）]
> - [Feature Gate 生命周期][feature gate]
> - [弃用政策][deprecation-policy]
> 请明确说明"毕业"的定义。
> 通常我们倾向于无论功能通过何种方式访问，均采用相同的阶段划分（alpha、beta、GA）。

#### 🔹 Alpha 阶段
- 功能已通过 Feature Gate 实现（默认关闭）  
- 初步的端到端（e2e）测试已完成并启用  
#### 🔸 Beta 阶段
- 收集开发者反馈及用户调研结果  
- 完成核心功能 A、B、C  
- 额外的测试已加入 Testgrid，并在 oFEP 中有链接说明  
- 实现更严格的测试形式，例如降级测试和可扩展性测试  
- 所有功能均已实现  
- 所有安全相关机制均已完备  
- 所有监控要求均已实现  
- 所有测试要求均已满足  
- 所有预发布阶段的问题与缺陷均已修复  

> **注意：** Beta 阶段的评估标准必须包含所有功能、安全性、监控与测试的要求，并解决所有已知问题或差距。

#### 🟢 GA 阶段
- 有 N 个真实生产环境中的使用示例  
- 有 N 次实际安装部署记录  
- 已留出反馈窗口期，收集充分用户反馈  
- 所有在 Beta 阶段反馈的问题与缺陷都已解决  

> **注意：** GA 阶段的毕业标准不得再包含功能、安全性、监控或测试方面的要求，这些应在 Beta 阶段全部完成。
> **注意：** 通常，我们在 Beta 和 GA 之间至少间隔两个版本周期，以便留出时间获取用户反馈和发现潜在问题，避免在连续发布中遗漏反馈环节。
> **对于非可选（默认启用）功能在进入 GA 阶段时，毕业标准必须包含 [一致性测试（Conformance tests）]。**

#### 🧯 弃用（Deprecation）
- 宣布弃用现有功能标志（flag）并说明支持策略  
- 自引入替代功能以来，已过去两个版本（以解决版本偏差问题）  
- 处理来自 gitcode Issues 等反馈渠道中的用法变更或行为差异问题  
- 正式弃用原有标志 （flag）
-->

#### Alpha阶段
- DRA驱动支持上报拓扑关系属性（pciBusID, pcieRoot, numaNode, topoHccsRingID）
- 支持通过ResourceClaim约束实现同HCCS环调度
- 初步e2e测试覆盖基础调度场景

#### Beta阶段
- 完整测试覆盖各种拓扑约束组合
- 收集用户反馈和性能数据

#### GA阶段
- 至少1个生产环境使用案例
- 监控和诊断工具完备
- 文档和最佳实践完善

### 升级/降级策略
<!--
> 指导提案人在功能设计时兼顾向前兼容性、配置迁移、Feature Gate 管理等问题。
-->
<!--
如果适用，该组件在升级和降级时将如何处理？请确保在测试计划中包含这部分内容。

在制定该增强功能的升级/降级策略时，请考虑以下问题：
- 为了保持现有行为不变，集群在升级时是否需要进行调用方式、配置或 API 使用上的任何更改？
- 为了使用该增强功能，集群在升级时是否需要对调用方式、配置或 API 使用做出调整？
-->

**升级策略：**
- 新增的拓扑属性字段对现有ResourceClaim无影响
- 已有的ResourceClaim在不使用拓扑约束时行为不变
- 建议升级后逐步为业务Pod添加拓扑约束以优化性能

**降级策略：**
- 使用拓扑约束的ResourceClaim在调度时会失败
- 建议在降级前移除所有拓扑约束

### 版本倾斜策略
<!--
确保你的提案在 openfuyao 集群中升级时能够兼容不同版本组件之间的运行差异。
-->
<!--
如果适用，该组件在面对与其他组件的版本不一致（version skew）时将如何处理？有哪些兼容性保证？请确保在测试计划中包括这一部分。

在为此增强功能制定版本偏差应对策略时，请考虑以下问题：
- 该功能是否涉及控制面（Control Plane）与节点（Node）之间的协同行为？
- 当使用该功能时，版本落后三个版本（n-3）的 kubelet 或 kube-proxy 会如何表现？
- 当使用该功能时，版本落后一个版本（n-1）的 kube-controller-manager 或 kube-scheduler 会有何行为？
- 节点上的其他组件是否会发生变更？ 例如，CSI（容器存储接口）、CRI（容器运行时接口）或 CNI（容器网络接口）是否需要在 kubelet 之前被更新？
-->

**版本偏差处理：**
- DRA驱动需要与kube-scheduler版本匹配才能使用拓扑约束
- 当kube-scheduler版本落后时，不支持拓扑约束的ResourceClaim会调度失败
- 建议先升级kube-scheduler，再升级DRA驱动

## 生产可用性审查
<!--
**生产可用性审查（Production Readiness Review，PRR）** 的目的是确保即将合并到 openfuyao 中的功能：
- 可观测（observable）、可扩展（scalable）、可支持（supportable）；
- 能在生产环境中安全运行；
- 在出现故障时能够被禁用或回滚。

**要使 oFEP 进入 `implementable` 状态并被纳入发布版本，必须完成并通过生产可用性审查问卷（PRR Questionnaire）。**

在某些情况下，元数据中也应包含这些问题的答案：
- 这样可以便于自动化工具验证是否进行了审查；
- 同时有助于减少评审负担并降低审查延迟。
-->

### 功能启用和回滚
<!--
当针对 alpha 版本发布时，必须完成此部分。
-->

###### 如何在实时集群中启用/禁用此功能？
<!--
选择其中一个并删除其余的。
-->

- [x] **功能开关（Feature gate）**（请同时在元数据中填写相应字段）  
  - 功能开关名称：  
  - 依赖该功能开关的组件：

###### 启用该功能会改变任何默认行为吗？
<!--
任何默认行为的变更都可能让用户感到意外，或破坏现有的自动化流程，因此在这方面必须格外小心。
-->

启用该功能不会改变默认行为。只有当ResourceClaim明确声明拓扑约束时才会触发拓扑感知调度。

###### 该功能一旦启用，是否可以禁用（即我们可以回滚启用）？
<!--
**请描述该功能对现有工作负载可能造成的影响**（例如，如果这是一个运行时特性，它是否可能破坏现有应用程序的行为？）。

通常，通过将功能开关（Feature Gate）设置为 `false` 并重启相应组件，即可禁用该功能。除这个操作之外，不应再需要其他变更来完成禁用。

**注意：**在元数据中，也请将 `disable-supported` 字段设置为 `true` 或 `false`。
-->

###### 如果该功能之前已回滚，现在我们重新启用它会发生什么情况？

重新启用后，DRA驱动重新上报拓扑属性，调度器可以处理拓扑约束。

###### 是否有任何针对功能启用/禁用的测试？
<!--
当前的端到端测试框架（e2e framework）**尚不支持启用或禁用 Feature Gate**。  
然而，对于处理数据的每个组件，必须编写包含**启用和未启用该功能场景的单元测试**。

如果该功能修改了 API 类型，**至少应该考虑添加转换测试（conversion tests）**。

此外，如果该功能引入了新的 API 字段，**还必须编写测试 Feature Gate 开关行为的单元测试**——也就是验证以下情形：  
- "当我先启用 Feature Gate 并写入了包含新字段的对象后，随后将其禁用，会发生什么？"
-->

### 推出、升级和回滚规划
<!--
当该功能计划从 Beta 阶段发布至正式版本时，必须填写本节内容。
-->

###### 部署或回滚为何会失败？这会影响正在运行的工作负载吗？
<!--
**尽可能保持警觉和审慎**——例如，假设在发布过程中某些组件会中途重启，会发生什么？

请务必考虑以下场景：
- **高可用（HA）集群**：在该场景下，功能开关（Feature Flag）可能仅在部分 API Server 上启用，而其他仍为禁用状态；
- **大型集群**：在这种情况下，功能的启用/禁用可能会分批分节点地推进，需考虑该过程中的一致性与兼容性问题。
-->

部署失败场景：
- DRA驱动启动失败：ResourceSlice不会更新，调度器使用缓存的旧数据，工作负载不受影响
- kube-scheduler配置错误：调度器无法启动，需要修复配置后重启

回滚失败场景：
- 回滚时存在使用拓扑约束的工作负载：新Pod调度失败，需要先移除约束

###### 哪些具体指标应该通知回滚？
<!--
当该功能还处于早期阶段时，用户应关注哪些信号，以便及早发现可能存在的严重问题？
-->

回滚指标：
- DRA驱动启动失败率
- ResourceSlice更新失败率
- 使用拓扑约束的ResourceClaim调度失败率
- 设备分配耗时显著增加

###### 升级和回滚测试了吗？升级->降级->升级的路径测试了吗？
<!--
请描述已完成的手动测试及其结果。
从长远来看，我们可能会要求实施自动化的升级/回滚测试，但目前我们仍缺少相关的机制和工具，因此暂时无法实现。
-->

手动测试计划：
- 升级测试：禁用->启用->验证拓扑调度功能
- 回滚测试：启用->禁用->验证已运行工作负载正常
- 路径测试：禁用->启用->禁用->启用

###### 推出时是否伴随任何功能、API、API 类型的字段、标志等的弃用和/或删除？
<!--
即使应用弃用政策，仍可能会让一些用户感到惊讶。
-->

无弃用或删除。

### 监控要求
<!--
当计划将功能从 Beta 阶段纳入某个正式版本（release）时，必须完成此部分内容。
对于 GA（正式可用）阶段，该部分也是必须填写的：审批人应能够基于实际生产环境中的经验，确认之前各项回答的准确性。
-->

###### 操作员如何确定该功能是否正在被工作负载使用？
<!--
理想情况下，应使用指标（metric）来实现。
通过对 Kubernetes API 的操作（例如检查是否存在设置了字段 X 的对象）应作为最后手段。
请避免将日志或事件用于此目的。
-->

监控指标：
- 使用拓扑约束的ResourceClaim数量
- 各种拓扑约束类型的分布（同HCCS环、同NUMA等）
- 拓扑约束调度成功率和失败率

###### 使用此功能的人如何知道它适用于他们的实例？
<!--
例如，如果这是一个与 Pod 相关的功能，则应能够针对每个 Pod 确定该功能是否正常工作。
请从以下选项中选择一项并删除其余内容。
请详细描述所有对最终用户可见的内容，确保他们能够验证该功能是否已正确启用并正常运行。
> 请注意：最终用户通常无法查看组件日志或访问系统指标（metrics）。
-->

- [x] 其他（作为最后手段）
  - 细节：用户可以通过Pod的resource claim状态查看分配的设备拓扑信息，或通过`kubectl get resourceslice -o yaml`查看节点上报的拓扑属性

###### 增强的合理 SLO（服务级别目标）是什么？
<!--
这是你定义该功能"正常服务质量"（Quality of Service，QoS）表现的机会。
我们无法提供全面的指导，但从高层角度来看（还需要更精确定义），这些表现可能包括：
- 每天返回 5XX 错误的 API 调用比例 ≤ 1%  
- CronJob 的任务实际创建时间与预期创建时间之差的绝对值在一天内的 99 百分位 ≤ 10%  
- 每天 99.9% 的 `/health` 请求返回 HTTP 200 状态码  

这些目标将有助于你在下一个问题中确定需要衡量的服务指标（SLIs）。
-->

拓扑约束调度延迟不超过普通调度的2倍。

拓扑约束调度成功率不低于95%（受资源碎片影响）。

###### 运维人员可以使用哪些服务级别指标（SLIs）来判断服务的健康状况？
<!--
请选择以下选项中的一个，并删除其余内容。
-->

- [x] 指标
  - 指标名称：
  - [可选] 聚合方法：平均值、计数
  - 暴露指标的组件：

###### 是否存在任何尚未覆盖的指标（metrics），可以用来进一步提升该功能的可观测性？
<!--
请描述这些指标本身，以及未添加它们的原因（例如：成本高、实现复杂等）。
-->

### 依赖项
<!--
当计划将该功能从 Beta 阶段纳入某个正式版本（release）时，必须完成本节内容。
-->

###### 此功能是否依赖于集群中运行的任何特定服务？
<!--
请同时考虑集群级别的服务（例如 metrics-server）以及节点级别的代理（例如某个特定版本的 CRI）。  
重点关注该功能所依赖的 **外部或可选服务**。  
例如，如果该功能依赖云服务商的 API、或外部的软件定义存储（SDS）或网络控制面板等服务，则应明确列出。

对于每一项依赖项，请填写以下内容：
- 当前用户工作负载的运行情况  
- 新建工作负载的创建情况  
- 集群级别的服务（例如 DNS）

填写格式如下：
```
- [依赖项名称]
  - 使用说明：
    - 若该服务发生中断，对该功能的影响：
    - 若该服务性能下降或错误率升高，对该功能的影响：
```
-->

- [kube-scheduler]
  - 使用说明：调度器需要支持DRA和拓扑约束处理
    - 若该服务发生中断，对该功能的影响：新Pod无法调度，已运行Pod不受影响
    - 若该服务性能下降或错误率升高，对该功能的影响：调度延迟增加

- [DRA驱动（npu.huawei.com）]
  - 使用说明：驱动需要上报拓扑属性
    - 若该服务发生中断，对该功能的影响：ResourceSlice不更新，调度器使用缓存数据
    - 若该服务性能下降或错误率升高，对该功能的影响：拓扑信息可能不准确

### 可扩展性
<!--
对于 Alpha 阶段，鼓励填写本节内容：评审人员应考虑这些问题，并尝试给出回答。
对于 Beta 阶段，本节为必填项：评审人员必须回答这些问题。
对于 GA（正式发布）阶段，本节同样为必填项：审批人员应能根据实际生产经验，确认之前给出的所有回答。
-->

###### 启用/使用此功能会导致任何新的 API 调用吗？
<!--
**请描述相关 API 调用，包括以下信息：**
- **API 调用类型**（例如：`PATCH pods`）  
- **预估调用频率（吞吐量）**  
- **调用发起组件**（例如：`Kubelet`、`Feature-X-controller`）

重点关注以下场景：
- **组件开始列出（list）或监听（watch）以前未处理的资源**
- **某些 Kubernetes 资源发生变化后，触发新的 API 请求行为**  
  例如：更新对象 X 后又触发了对对象 Y 的修改或创建
- **用于状态对齐（state reconciliation）的定期 API 请求**  
  例如：周期性获取资源状态、心跳上报、领导者选举等
-->

无新增API调用。拓扑信息作为ResourceSlice的属性字段上报，不增加额外的API请求。

###### 启用/使用此功能是否会导致引入新的 API 类型？
<!--
请描述它们（指代 API 对象或资源），并提供以下信息：
- **API 类型（API type）**  
- **每个集群支持的最大对象数量**
- **每个命名空间支持的最大对象数量**（仅适用于具备命名空间作用域的对象）
-->

无新增API类型。使用现有的ResourceSlice和ResourceClaim API。

###### 启用/使用此功能是否会导致对云提供商的任何新的 API 调用？
<!--
请进行如下描述：
- **涉及哪些 API：**  
- **预估调用增加量：**
-->

不涉及云提供商API调用。

###### 启用/使用此功能是否会导致现有 API 对象的大小或数量增加？
<!--
**请描述新增或受影响的资源对象，提供以下信息：**
- **API 类型（API type(s)）：**  
- **预估对象体积增加量：**（例如：新增注解字段，大小为 32 字节）  
- **预估新增对象数量：**（例如：为每个现有 Pod 创建一个新的对象 X）
-->

- API类型：ResourceSlice
- 预估对象体积增加量：每个设备增加约100字节的拓扑属性字段
- 预估新增对象数量：无新增对象

###### 启用/使用此功能是否会导致现有 SLI/SLO 所涵盖操作的耗时增加？
<!--
请思考是否会新增额外操作或在现有流程中引入新的中间步骤（例如：为了启动一个容器，现在需要先执行步骤 X 等）。
请在下方详细描述这些新增内容。
-->

Pod调度耗时可能因拓扑约束验证而略有增加，预计影响小于10%。

###### 启用/使用此功能是否会导致任何组件的资源使用率（CPU、RAM、磁盘、IO 等）不可忽略的增加？
<!--
需要注意的事项包括：
- 增加的内存状态（in-memory state）；
- 新引入的耗时计算操作；
- 频繁的磁盘访问（包括日志量增加）；
- 大量发送或接收的网络数据流量等。

请从小型和大型集群的不同规模角度，**全面评估这些资源消耗**，并结合 Kubernetes 的[支持上限（supported limits）](https://git.k8s.io/community/sig-scalability/configs-and-limits/thresholds.md)进行考量。
-->

- 内存：kube-scheduler需要缓存拓扑信息，每个设备约增加100字节，影响很小
- CPU：拓扑约束验证的计算开销很小，不影响整体性能
- 磁盘/IO：无额外磁盘访问

###### 启用/使用此功能是否会导致某些节点资源（PID、套接字、inode 等）耗尽？
<!--
**请不要只关注理想情况，更重要的是评估异常或极端情况**，例如：
- 探针响应时间从毫秒级变成分钟级；
- 异常或失败的 Pod 持续占用资源等。

**如果该功能可能会导致某些资源被耗尽**，请说明：
- 如何通过 Kubernetes 已有的资源限制机制进行缓解（例如每节点最大 Pod 数）；
- 或该 oFEP 是否引入了新的限制来控制这些风险。

此外，还请说明：
- **是否已经进行了性能相关测试（或计划进行），用于更好地理解性能特征，并验证所声明的资源使用上限？**
-->

不会导致节点资源耗尽。拓扑信息存储在etcd和调度器内存中，不占用节点资源。

### 故障排除
<!--
**当该功能计划从 Beta 阶段发布至正式版本（release）时，本节必须填写。**

对于 **GA（正式发布）阶段**，本节同样为必填项：审批人员应能够根据实际生产环境中的经验，确认之前的各项回答。

当前的 **"排障（Troubleshooting）" 部分**，在功能发布流程中相当于临时承担了 **"运维手册（Playbook）"** 的角色。  
未来可能会将其拆分为一个专门的 `Playbook` 文档（可能还会包含一些监控信息）。但目前仍将其保留在此处。
-->

###### 如果 API 服务器和/或 etcd 不可用，此功能如何反应？

如果API服务器/etcd不可用：
- DRA驱动无法更新ResourceSlice，使用缓存数据
- 调度器无法处理新的调度请求，已运行的工作负载不受影响

###### 其他已知故障模式有哪些？
<!--
**对于每种故障模式，请使用以下模板逐项填写信息：**
- **[故障模式简要描述]**  
  - **检测方式（Detection）：**  
    如何通过指标（metrics）检测该问题？换句话说：  
    操作人员**如何在不登录 master 或 worker 节点的情况下排障**？
  - **缓解手段（Mitigations）：**  
    尤其对于已在运行的用户工作负载，可采取哪些措施止血/减缓影响？
  - **诊断信息（Diagnostics）：**  
    有哪些有用的日志消息？其对应的**日志级别**（logging level）是多少？  
    ✅ *注：日志诊断信息在功能进入 Beta 阶段前可不填写。*
  - **测试情况（Testing）：**  
    针对该故障是否有测试用例？若无，请说明原因。
-->

###### 如果未满足 SLO，应采取哪些步骤来确定问题？

## 实施历史
<!--
**在本节中应记录一个 oFEP 生命周期中的主要里程碑。**

可能包括但不限于以下内容：
- `摘要` 和 `动机` 部分被合并，表示 SIG 已接受该提案  
- `提案` 部分被合并，表示对设计方案达成一致  
- 实现工作的启动日期  
- 首个包含该 oFEP 初始版本的 openfuyao 发布版本  
- 该 oFEP 成功毕业为正式可用（GA）的 openfuyao 版本  
- oFEP 被废弃或被其他提案取代的时间  
-->

- 2025-07-20：oFEP创建，摘要和动机部分完成

## 缺点
<!--
为什么不实施这个 oFEP？从反面角度分析该提案可能带来的负面影响、权衡成本、潜在风险或争议点。
-->

1. **资源碎片化风险**：严格的拓扑约束可能导致资源碎片化，降低整体资源利用率。

2. **运维复杂度增加**：需要用户了解硬件拓扑结构和约束配置，增加了使用门槛。

3. **硬件依赖性强**：不同服务器型号的拓扑结构不同，需要预置不同的模板，维护成本高。

4. **调度延迟增加**：拓扑约束验证会增加调度延迟，虽然影响较小但仍存在。

## 替代方案
<!--
**你还考虑过哪些其他方案？为什么将它们排除？**  
这些备选方案不需要像最终提案那样详尽，但应提供足够的信息来阐述其基本思路，并说明为什么它们不可接受。
-->

**替代方案1：使用节点标签和节点亲和性**

在节点上打标签标记拓扑信息，使用节点亲和性调度。

缺点：
- 无法实现细粒度的设备级拓扑约束
- 节点标签信息有限，难以表达复杂的拓扑关系
- 与DRA机制不兼容

**替代方案2：使用调度器扩展**

实现自定义调度器扩展，在调度时查询拓扑信息并决策。

缺点：
- 需要额外的调度器扩展开发和维护
- 无法利用DRA的原生约束机制
- 与Kubernetes生态集成度低

**替代方案3：使用设备管理工具（如MindCluster）**

使用专业的设备管理工具实现拓扑调度。

缺点：
- 引入额外依赖，增加系统复杂度
- 与Kubernetes原生调度不集成
- 不符合云原生理念

## 所需基础设施（可选）
<!--
**如果你需要从项目或 SIG 获得资源支持，请在本节中列出。** 例如：
- 请求创建新的子项目（subproject）  
- 新建 gitcode 仓库（repos）  
- 配置 gitcode 相关权限或细节（如 team 成员或 CI 权限等）

提前在这里列出这些需求，有助于 SIG 尽早启动相关流程，加快资源配置效率。
-->