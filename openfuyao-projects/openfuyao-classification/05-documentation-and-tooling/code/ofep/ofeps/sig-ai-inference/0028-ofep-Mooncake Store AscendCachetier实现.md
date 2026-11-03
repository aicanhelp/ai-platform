---
# 提案标题
title: oFEP Mooncake Store AscendCacheTier实现
# 提案编号
ofep-number: 0028
# 提案作者
authors:
  - "@ccccccxy @chenwenxiao"
# 提案主导SIG
owning-sig: sig-ai-inference
# 提案协作SIG
participating-sigs:
# 提案状态(初步草案|准备开始实现|已在SIG中实现并合并|提案被暂缓|提案被否决|作者主动撤回提案|被另一个oFEP取代)
status: provisional
# 创建日期
creation-date: 2025-12-03
# 评审人
reviewers:
  - TBD
# 批准人
approvers:
  - TBD
# 相关联的其他oFEP
see-also:
# 当前oFEP替代了哪些已有提案
replaces:
# 当前oFEP被哪些提案替代，当status为replaced时需要填写此字段
replaced-by:
# 此oFEP当前开发阶段
stage: alpha
# 最近一次推进的版本
latest-milestone:
# 各阶段目标版本
milestone:
  alpha: TBD
  beta: TBD
  stable: TBD
# 功能开关及影响组件
feature-gates:
  - name: TBD
    components:
      - TBD
# 是否支持关闭该功能
disable-supported: true
# 该功能引入的监控指标
metrics:
---

# oFEP-0028：Mooncake Store AscendCacheTier实现

<!-- toc -->
- [发布签核清单](#release-signoff-checklist)
- [摘要](#summary)
- [动机](#motivation)
  - [目标](#goals)
  - [非目标](#non-goals)
- [提案](#proposal)
  - [用户故事](#user-stories-optional)
    - [故事 1](#story-1)
    - [故事 2](#story-2)
  - [注释/约束/警告](#notesconstraintscaveats-optional)
  - [风险与缓解措施](#risks-and-mitigations)
- [设计细节](#design-details)
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
- [所需基础设施](#infrastructure-needed-optional)
<!-- /toc -->

## 发布签核清单

标记有（R）的项目*在达到里程碑/发布*之前是必需的。
- [ ]（R）发布里程碑中的增强问题，链接到 [openfuyao/ofep] 中的 oFEP 目录
- [ ] (R) oFEP 审批者已批准 oFEP 状态为“可实施”
- [ ] (R) 设计细节已适当记录
- [ ]（R）测试计划已到位，并考虑了 SIG 架构和 SIG 测试的输入（包括测试重构）
  - [ ] 针对所有 Beta API 操作（端点）进行 e2e 测试
  - [ ] (R) 确保 GA e2e 测试满足一致性测试的要求
  - [ ] (R) GA e2e 测试至少需要两周时间才能证明测试结果无 flake（不稳定或偶发失败）
- [ ] (R) 毕业标准已设定
  - [ ] (R) 所有 GA 端点必须通过[一致性测试]
- [ ] (R) 生产准备情况审查完成
- [ ] (R) 生产准备情况审查已获批准
- [ ] “实施历史”部分已更新里程碑
- [ ] 面向用户的文档已在 [openfuyao/docs] 创建，以便发布到 [openfuyao.cn]
- [ ] 支持文档 - 例如，额外的设计文档、邮件列表讨论/SIG 会议链接、相关 PR/问题、发行说明

<!--
**注意：**此清单是迭代的，每次考虑将此增强功能作为里程碑时都应进行审查和更新。
-->

- [openfuyao.cn](https://openfuyao.cn/)
- [openfuyao/ofep](https://gitcode.com/openfuyao/ofep)
- [openfuyao/docs](https://gitcode.com/openfuyao/docs)

## 概括

本提案旨在为Mooncake Store的分层缓存系统（Tiered Caching）实现AscendCacheTier，这是一个基于昇腾（Ascend）NPU设备内存的缓存层实现。AscendCacheTier作为CacheTier接口的具体实现，采用**Allocate/Free资源管理模型**，提供了在昇腾NPU设备内存中分配和管理缓存空间的能力，支持与其他缓存层（如DRAM）之间的数据拷贝操作。

该实现通过ACL（Ascend Computing Language）运行时库与昇腾设备进行交互，提供了设备内存的分配、释放、以及主机内存与设备内存之间的数据拷贝功能。核心设计包括：

- **RAII模式**：通过`AscendBuffer`类实现自动资源管理，析构时自动释放设备内存
- **原子操作**：使用CAS（Compare-And-Swap）原子操作进行线程安全的容量管理
- **设备上下文管理**：通过`AscendUnifiedPointer`结构将设备指针与设备ID绑定，支持多设备场景
- **拷贝函数注册**：通过`CopierRegistry`静态注册DRAM与ASCEND_NPU之间的拷贝函数

## 动机
<!--
本节用于明确列出该oFEP的动机、目标和非目标。描述变更的重要性以及对用户的好处。
-->
近年来，随着大语言模型（LLM）技术的快速发展，推理服务在实际应用中面临着越来越高的性能要求。特别是KV Cache（键值缓存）的高效管理，已成为大规模推理场景中影响系统吞吐与响应速度的核心问题。如何在保障服务质量的同时，提升缓存体系的带宽与容量利用率，是诸多用户关心的重点。

为解决上述难题Mooncake Store推出了全新的分层缓存系统架构。该架构设计采用了“两级元数据”管理方案，并充分利用多种存储介质（如 GPU、VRAM、DRAM、SSD 等）在多个节点之间协同，目的是实现大容量、高速且成本可控的分布式缓存。

在此架构基础上，本提案提出：基于CacheTier抽象接口新增实现AscendCacheTier。这样，开发者和用户就可以将高访问频率的数据直接存入NPU设备内存，充分发挥昇腾硬件高带宽、低延迟的优势，有效提升大模型推理场景下的整体系统性能和资源利用率。

### 目标

1. **实现基于昇腾NPU设备内存的缓存层**：实现CacheTier接口，支持Init、Allocate、Free等基本操作，能够成功在昇腾设备上分配和释放内存；采用RAII模式进行资源管理，并通过原子操作保证线程安全。
2. **支持条件编译和降级处理**：通过`USE_ASCEND_CACHE_TIER`宏控制功能启用，未启用时有友好日志提示并回退到主机内存，避免运行时崩溃。
3. **实现数据拷贝功能**：通过`CopierRegistry`注册DRAM与ASCEND_NPU之间的双向拷贝函数（`CopyDramToAscend`、`CopyAscendToDram`），能够正确执行跨内存类型的数据拷贝操作，支持跨层数据传输。

### 非目标

1. **不实现缓存淘汰策略**：缓存淘汰和迁移策略由TieredBackend和CacheScheduler负责，AscendCacheTier仅负责单层缓存的基本存储操作。
2. **不提供设备间数据拷贝**：当前实现仅支持单个设备内的操作，跨设备的数据传输不在本提案范围内。

## 提案
<!--
**这是我们真正进入提案具体内容的部分。**  
这一部分应包含足够的细节，使评审人员能清晰理解你到底在提出什么建议，  
但不应涉及 API 设计或具体实现细节。  

请阐明：
- **预期目标是什么？**  
- **我们如何衡量成功？**

请将更详细的设计和实现细节放在下方的 “设计细节” 部分中。
-->

### 用户故事
<!--
详细说明如果该 oFEP 被实施，用户将能够做哪些事情。请尽可能提供细节，以便人们理解系统将"如何"运作。此部分的目标是：让用户对提案有真实的感受，而不是陷入技术细节的泥淖中。
-->

本提案基于Mooncake Store V3的TieredBackend架构（参考[Mooncake PR #1022](https://github.com/kvcache-ai/Mooncake/pull/1022)），通过实现CacheTier接口为昇腾NPU设备提供缓存层支持。

#### 用例视图

@startuml
top to bottom direction

actor "推理服务开发者" as dev

rectangle "AscendCacheTier" {
    usecase "Init\n初始化设备" as UC_INIT
    usecase "Allocate\n分配NPU内存" as UC_ALLOC
    usecase "Free\n释放内存" as UC_FREE
    usecase "DataCopier\n跨层数据拷贝" as UC_COPY
}

dev --> UC_INIT
dev --> UC_ALLOC
dev --> UC_FREE
dev --> UC_COPY

@enduml


#### 故事1：在昇腾NPU设备上分配缓存空间

**角色：** 推理服务开发者

**背景：** 开发基于昇腾NPU的LLM推理服务时，KV Cache是影响性能的关键数据。传统方案将KV Cache存放在主机DRAM中，每次推理都需要通过PCIe传输到设备，造成显著延迟。

**需求：**
- 通过TieredBackend在NPU设备内存中分配缓存空间
- 使用Allocate/Free模式进行资源管理
- 通过RAII机制自动释放设备内存

**场景：** 在Prefill阶段需要存储KV Cache时，先通过`Allocate`在AscendCacheTier分配设备内存空间，获得`DataSource`对象。然后通过`DataCopier`将数据从DRAM拷贝到设备内存。使用完成后，`AscendBuffer`析构时自动释放设备内存。

**价值：** 充分利用NPU HBM的高带宽特性，减少PCIe瓶颈带来的延迟开销。

---

#### 故事 2：跨层数据拷贝

**角色：** 推理服务开发者

**背景：** 在推理的Decode阶段，需要在不同缓存层之间传输数据。例如，将DRAM中的数据写入NPU设备内存，或从NPU设备内存读取数据到DRAM。

**需求：**
- 支持DRAM到ASCEND_NPU的数据拷贝（Host-to-Device）
- 支持ASCEND_NPU到DRAM的数据拷贝（Device-to-Host）
- 自动管理设备上下文，简化多设备编程

**场景：** 通过`DataCopier`调用已注册的拷贝函数，自动处理设备上下文切换和内存拷贝操作。拷贝函数通过`CopierRegistry`在静态初始化阶段自动注册。

**价值：** 统一的数据拷贝接口，简化跨层数据传输的编程复杂度。

---

#### 故事 3：RAII自动资源管理

**角色：** 推理服务开发者

**背景：** 多租户LLM服务平台中，需要频繁分配和释放设备内存。手动管理内存容易导致内存泄漏或重复释放。

**需求：**
- 分配的设备内存自动关联到`AscendBuffer`对象
- `AscendBuffer`析构时自动调用`aclrtFree`释放设备内存
- 支持移动语义，避免不必要的拷贝

**场景：** 调用`Allocate`后获得包含`AscendBuffer`的`DataSource`。当`DataSource`超出作用域或显式调用`Free`时，`AscendBuffer`的析构函数自动释放设备内存。

**价值：** 通过RAII机制避免内存泄漏，提高代码的安全性和可维护性。

---

### 注释/限制/注意事项

#### 内存管理

- 设备内存通过`AscendUnifiedPointer`结构进行封装，同时保存设备指针、
- 设备ID和分配大小
- `AscendBuffer`类实现RAII模式，析构时自动调用`ReleaseMemory()`释放设备内存
- 使用原子操作（CAS）进行容量管理，分配失败时自动回滚已预留的空间
- 支持移动语义（move constructor/assignment），禁用拷贝语义

#### 线程安全

- `current_usage_`使用`std::atomic<size_t>`类型，通过CAS操作保证线程安全
- 内存分配采用两阶段提交：先原子预留空间，再执行实际分配
- 初始化过程通过`init_mutex_`互斥锁保护，支持幂等初始化

#### 条件编译支持

- 通过`USE_ASCEND_CACHE_TIER`宏控制功能启用
- 未启用时，内存分配回退到`std::malloc`，拷贝操作回退到`std::memcpy`
- 这允许代码在不支持昇腾的环境中编译和运行测试

### 风险与缓解措施

#### 1. ACL 运行时库依赖风险

**风险：** AscendCacheTier强依赖 ACL 运行时库，如果库版本不兼容或缺失，可能导致运行时错误。

**缓解措施：**
- 通过条件编译（`USE_ASCEND_CACHE_TIER`）控制功能启用
- 在未启用时提供明确的错误提示，避免静默失败
- 在初始化阶段检查设备可用性，失败时返回明确的错误信息

#### 2. 设备内存不足风险

**风险：** 当设备内存不足时，分配操作可能失败，影响缓存功能。

**缓解措施：**
- 使用原子CAS操作在分配前检查可用容量
- 在 `Allocate` 操作中，如果空间不足，返回 `NO_AVAILABLE_HANDLE` 错误码
- 设备分配失败时自动回滚已预留的空间，避免容量计数不一致

#### 3. 多线程环境下的设备上下文风险

**风险：** 在多线程环境中，如果设备上下文设置不正确或容量管理出现竞态条件，可能导致数据损坏或操作失败。

**缓解措施：**
- 使用原子操作（CAS）进行线程安全的容量预留和更新
- 所有 ACL 操作都通过统一的辅助函数自动设置设备上下文
- 采用两阶段提交模式：先原子预留空间，再执行实际分配
- 分配失败时通过 `fetch_sub` 原子操作回滚预留的空间
- 初始化过程使用互斥锁保护，支持幂等初始化


## 设计细节
<!--
本节应包含足够的信息，以便读者能够清楚理解你所提出的变更具体是什么。这可能包括 API 规格说明（虽然并非必须）或代码片段。如果对该提案将如何实施存在任何疑问，应在此处进行详细讨论。
-->

### 系统架构视图

#### 部署视图

下图展示了AscendCacheTier在实际部署环境中的组件分布:

@startuml
!include <C4/C4_Deployment>

Deployment_Node(node, "工作节点", "Physical/Virtual Machine") {
    Deployment_Node(process, "Mooncake Store 进程", "C++ Application Process") {
        Container(backend, "TieredBackend", "C++", "多层缓存协调器")
        Container(tier0, "AscendCacheTier", "C++", "昇腾NPU缓存层\nAllocate/Free模型")
        Container(buffer, "AscendBuffer", "C++", "RAII设备内存包装")
        Container(copier, "DataCopier", "C++", "跨内存数据传输")
        Container(registry, "CopierRegistry", "C++", "拷贝函数静态注册")
    }

    Deployment_Node(aclruntime, "ACL运行时", "Ascend Computing Language") {
        Container(aclrt, "ACL Runtime", "Library", "设备管理与内存操作")
    }

    Deployment_Node(device0, "昇腾NPU设备", "Ascend NPU Hardware") {
        ContainerDb(vram0, "设备内存", "VRAM", "缓存数据存储")
    }
}

Rel(backend, tier0, "管理")
Rel(tier0, buffer, "创建")
Rel(tier0, copier, "使用")
Rel(copier, registry, "查询")
Rel(tier0, aclrt, "调用")
Rel(buffer, aclrt, "调用")
Rel(copier, aclrt, "调用")
Rel(aclrt, vram0, "控制")

@enduml

#### 组件视图

下图展示了系统的逻辑组件及其关系:


@startuml
skinparam componentStyle rectangle

package "Mooncake Store 分层缓存系统" {

    component "用户应用" as app

    package "协调层" {
        component "TieredBackend" as backend
    }

    package "缓存层接口" {
        interface "CacheTier" as itier
        note right of itier
          Init / Allocate / Free
          使用tl::expected返回错误
        end note
    }

    package "缓存层实现" {
        component "AscendCacheTier" as ascend
        component "DRAMCacheTier" as dram
    }

    package "缓冲区" {
        interface "BufferBase" as ibuffer
        component "AscendBuffer" as ascendbuf
        component "DRAMBuffer" as drambuf
    }

    package "数据传输" {
        component "DataCopier" as copier
        component "CopierRegistry" as registry
    }

    package "硬件设备" {
        database "昇腾NPU\n设备内存" as npumem
        database "主机DRAM" as hostmem
    }
}

app --> backend : 缓存请求
backend --> itier : 委托操作
itier <|.. ascend : 实现
itier <|.. dram : 实现

ascend --> ascendbuf : 创建/管理
dram --> drambuf : 创建/管理
ibuffer <|.. ascendbuf : 实现
ibuffer <|.. drambuf : 实现

backend --> copier : 数据拷贝
copier --> registry : 查找函数

ascendbuf --> npumem : RAII内存管理
drambuf --> hostmem : 内存管理

note right of ascend
  线程安全: atomic<size_t> + CAS
  RAII: AscendBuffer自动释放内存
  Allocate/Free资源管理模型
end note

@enduml

#### 类图视图

下图展示了核心类的结构和关系:


@startuml
class BufferBase <<abstract>> {
    + {abstract} data(): uint64_t
    + {abstract} size(): size_t
}

class AscendBuffer {
    - unified_ptr_: unique_ptr<AscendUnifiedPointer>
    --
    + AscendBuffer(unified_ptr)
    + ~AscendBuffer()
    + data(): uint64_t
    + size(): size_t
    + GetDeviceId(): int
    + GetDevicePtr(): void*
    + GetUnifiedPointer(): AscendUnifiedPointer*
    - ReleaseMemory()
}

class DRAMBuffer {
    - dram_buffer_: unique_ptr<AllocatedBuffer>
    --
    + data(): uint64_t
    + size(): size_t
}

struct AscendUnifiedPointer {
    + device_ptr: void*
    + device_id: int
    + size: size_t
}

struct DataSource {
    + buffer: unique_ptr<BufferBase>
    + type: MemoryType
}

class CacheTier <<abstract>> {
    # backend_: TieredBackend*
    --
    + {abstract} Init(backend, engine): expected<void, ErrorCode>
    + {abstract} Allocate(size, data): expected<void, ErrorCode>
    + {abstract} Free(data): expected<void, ErrorCode>
    + {abstract} GetTierId(): UUID
    + {abstract} GetCapacity(): size_t
    + {abstract} GetUsage(): size_t
    + {abstract} GetMemoryType(): MemoryType
    + {abstract} GetTags(): vector<string>
}

class AscendCacheTier {
    - tier_id_: UUID
    - capacity_: size_t
    - tags_: vector<string>
    - device_id_: int
    - current_usage_: atomic<size_t>
    - is_initialized_: bool
    - init_mutex_: mutex
    --
    + Init(backend, engine): expected<void, ErrorCode>
    + Allocate(size, data): expected<void, ErrorCode>
    + Free(data): expected<void, ErrorCode>
    + GetDeviceId(): int
    - AllocateDeviceMemory(size): unique_ptr<AscendUnifiedPointer>
    - HasSpace(size): bool
}

class TieredBackend {
    - tiers_: map<UUID, unique_ptr<CacheTier>>
    - data_copier_: DataCopier
    --
    + Init(config): expected<void, ErrorCode>
    + Allocate(tier_id, size, data): expected<void, ErrorCode>
    + Free(data): expected<void, ErrorCode>
    + GetDataCopier(): DataCopier&
}

class DataCopier {
    - copy_functions_: map<pair<MemoryType,MemoryType>, CopyFunction>
    --
    + Copy(src, dst): expected<void, ErrorCode>
}

class CopierRegistry <<singleton>> {
    - memory_type_regs_: vector<MemoryTypeRegistration>
    --
    + {static} GetInstance(): CopierRegistry&
    + RegisterMemoryType(type, to_dram, from_dram)
}

BufferBase <|-- AscendBuffer
BufferBase <|-- DRAMBuffer
AscendBuffer *-- AscendUnifiedPointer
DataSource *-- BufferBase

CacheTier <|-- AscendCacheTier
TieredBackend o-- CacheTier
TieredBackend *-- DataCopier
CopierRegistry ..> DataCopier : builds

note right of AscendCacheTier
  线程安全: atomic<size_t> + CAS
  RAII: AscendBuffer自动释放内存
  返回类型: tl::expected<void, ErrorCode>
end note

@enduml

**说明：**
- `BufferBase`是所有缓冲区的抽象基类，定义了`data()`和`size()`接口
- `AscendBuffer`封装`AscendUnifiedPointer`，实现RAII模式自动释放设备内存
- `AscendUnifiedPointer`将设备指针与设备ID绑定，支持多设备场景
- `DataSource`包含`BufferBase`指针和`MemoryType`，用于跨层数据传输
- `CacheTier`使用`tl::expected<void, ErrorCode>`返回类型进行错误处理
- `CopierRegistry`通过静态注册机制收集拷贝函数，由`DataCopierBuilder`构建`DataCopier`
### 时序图

#### Allocate操作时序图

下图展示了完整的Allocate操作流程，包括容量检查、原子预留、内存分配：


@startuml
actor 客户端 as client
participant "TieredBackend" as backend
participant "AscendCacheTier" as tier
participant "ACL Runtime" as acl
database "设备内存" as devmem

client -> backend: Allocate(tier_id, size)
activate backend

backend -> tier: Allocate(size, DataSource&)
activate tier

== 初始化检查 ==
tier -> tier: 检查is_initialized_
alt 未初始化
    tier --> backend: return unexpected(INTERNAL_ERROR)
    backend --> client: Allocate失败
end

== 原子容量预留 (CAS) ==
tier -> tier: current = current_usage_.load(acquire)
loop CAS循环
    tier -> tier: 检查: current + size <= capacity_
    alt 容量不足
        tier --> backend: return unexpected(NO_AVAILABLE_HANDLE)
        backend --> client: Allocate失败
    end
    tier -> tier: compare_exchange_weak(current, current + size)
    note right: acq_rel成功, acquire失败重试
end

== 设备内存分配 ==
tier -> tier: AllocateDeviceMemory(size)
activate tier
tier -> acl: aclrtSetDevice(device_id)
tier -> acl: aclrtMalloc(size, HUGE_FIRST)
acl -> devmem: 分配设备内存
devmem --> acl: device_ptr
acl --> tier: device_ptr
deactivate tier

alt 分配失败
    tier -> tier: current_usage_.fetch_sub(size, release)
    note right: 回滚预留空间
    tier --> backend: return unexpected(NO_AVAILABLE_HANDLE)
    backend --> client: Allocate失败
end

== 创建AscendBuffer ==
tier -> tier: 创建AscendUnifiedPointer{device_ptr, device_id, size}
tier -> tier: 创建AscendBuffer(unique_ptr<AscendUnifiedPointer>)
tier -> tier: data.buffer = std::move(ascend_buffer)
tier -> tier: data.type = ASCEND_NPU

tier --> backend: return expected<void>
deactivate tier

backend --> client: Allocate成功, DataSource
deactivate backend

note over tier, devmem
  关键特性:
  - 原子CAS: 线程安全的容量预留
  - 两阶段提交: 先预留后分配
  - 失败回滚: 分配失败时恢复预留
  - RAII: AscendBuffer管理设备内存
end note

@enduml

**流程说明：**
1. TieredBackend将Allocate请求转发给目标tier
2. AscendCacheTier检查初始化状态
3. 使用CAS原子操作预留空间（先检查后更新，失败重试）
4. 调用`AllocateDeviceMemory`分配设备内存
5. 如果设备分配失败，通过`fetch_sub`回滚预留的空间
6. 创建`AscendBuffer`封装设备内存，通过`DataSource`返回给调用者
7. `AscendBuffer`实现RAII，析构时自动释放设备内存

#### Free操作时序图


@startuml
actor 客户端 as client
participant "TieredBackend" as backend
participant "AscendCacheTier" as tier
participant "AscendBuffer" as buffer
participant "ACL Runtime" as acl

client -> backend: Free(DataSource)
activate backend

backend -> tier: Free(DataSource)
activate tier

== 参数验证 ==
alt buffer为空
    tier -> tier: LOG(WARNING)
    tier --> backend: return expected<void>
    note right: 空buffer是no-op
end

== 获取大小 ==
tier -> tier: freed_size = data.buffer->size()

== RAII释放 (函数返回时) ==
tier -> tier: DataSource析构触发
tier -> buffer: ~AscendBuffer()
activate buffer
buffer -> buffer: ReleaseMemory()
buffer -> acl: aclrtSetDevice(device_id)
buffer -> acl: aclrtFree(device_ptr)
acl --> buffer: 完成
buffer -> buffer: unified_ptr_.reset()
deactivate buffer

== 更新使用量 ==
tier -> tier: current_usage_.fetch_sub(freed_size, acq_rel)

tier --> backend: return expected<void>
deactivate tier

backend --> client: Free成功
deactivate backend

note over tier, acl
  关键特性:
  - RAII: AscendBuffer析构自动释放内存
  - 原子更新: fetch_sub更新使用量
  - 幂等性: 空buffer是安全的no-op
end note

@enduml

**流程说明：**
1. TieredBackend将Free请求转发给目标tier
2. 检查buffer是否为空（空buffer是安全的no-op）
3. 获取待释放的内存大小
4. 当`DataSource`超出作用域时，`AscendBuffer`析构函数被调用
5. `ReleaseMemory()`调用`aclrtFree`释放设备内存
6. 通过原子操作更新`current_usage_`

#### 数据拷贝时序图


@startuml
participant "TieredBackend" as backend
participant "DataCopier" as copier
participant "CopyDramToAscend" as copy_func
participant "ACL Runtime" as acl
database "DRAM" as dram
database "设备内存" as devmem

== DRAM -> ASCEND_NPU 拷贝 ==
backend -> copier: Copy(src_DataSource, dst_DataSource)
activate copier

copier -> copier: 查找拷贝函数\n(DRAM, ASCEND_NPU)
copier -> copy_func: CopyDramToAscend(src, dst)
activate copy_func

== 参数验证 ==
copy_func -> copy_func: 验证src.buffer非空
copy_func -> copy_func: 验证dst.buffer非空
copy_func -> copy_func: dynamic_cast<AscendBuffer*>(dst.buffer)
alt 类型不匹配
    copy_func --> copier: return unexpected(INVALID_PARAMS)
end

== 获取指针 ==
copy_func -> copy_func: src_ptr = src.buffer->data()
copy_func -> copy_func: ascend_ptr = dst->GetUnifiedPointer()
copy_func -> copy_func: 验证dst.size >= src.size

== 执行拷贝 ==
copy_func -> acl: aclrtSetDevice(device_id)
copy_func -> acl: aclrtMemcpy(device_ptr, dst_size,\n                src_ptr, src_size,\n                HOST_TO_DEVICE)
acl -> dram: 读取数据
acl -> devmem: 写入数据
acl --> copy_func: 成功

copy_func --> copier: return expected<void>
deactivate copy_func

copier --> backend: return expected<void>
deactivate copier

note over copier, acl
  拷贝函数注册机制:
  - CopierRegistrar在静态初始化时注册
  - RegisterMemoryType(ASCEND_NPU, to_dram, from_dram)
  - DataCopier根据src/dst类型查找拷贝函数
end note

@enduml

**流程说明：**
1. `DataCopier`根据源/目标的`MemoryType`查找对应的拷贝函数
2. `CopyDramToAscend`函数验证参数有效性
3. 通过`dynamic_cast`确保目标buffer是`AscendBuffer`类型
4. 从`AscendUnifiedPointer`获取设备ID和设备指针
5. 设置设备上下文并执行`aclrtMemcpy`（HOST_TO_DEVICE）
6. 拷贝函数通过`CopierRegistrar`在静态初始化阶段自动注册

### 测试计划
<!--
**注意：**在该提案尚未被纳入某个正式版本前，此部分不是必需的。
其目标是确保我们不会接收缺乏充分测试的增强功能。
所有代码都应具备充分的测试（最终也应满足测试覆盖率要求）。在撰写测试计划时，请遵循 openFuyao 测试指南。
-->

[ ] 我/我们理解，相关组件的所有者可能会要求更新已有的测试，以便在提交实现该增强功能所需的更改之前，使代码达到足够稳固的质量标准。

##### 先决条件测试更新

在实施AscendCacheTier之前，需要确保以下基础组件的测试覆盖：

1. **CacheTier 接口测试**
   - 确保 CacheTier 接口定义完整且稳定
   - 验证接口契约的清晰性，特别是错误处理约定

2. **DataCopier 和 CopierRegistry 测试**
   - 验证 CopierRegistry 能够正确注册和查找拷贝函数
   - 确保 DataCopier 能够正确处理不同 MemoryType 之间的转换

3. **TieredBackend 基础功能测试**
   - 确保 TieredBackend 能够正确初始化和管理多个缓存层
   - 验证后端能够正确调用 DataCopier 进行数据拷贝

##### 单元测试
<!--
原则上，所有新增的代码都应具有完整的单元测试覆盖率，因此列出确切的测试项并不会带来额外价值。
但如果无法实现完整的单元测试覆盖，请说明原因，并解释为何在这种情况下这是可以接受的。
-->

<!--
此外，对于 Alpha 阶段，请尽量列出为了实现该增强功能将涉及的核心包（core package），并提供这些包当前的单元测试覆盖率，格式如下：
- <软件包>: <日期> - <当前测试覆盖率>
这可以帮助我们在扩展生产代码、实施该增强功能之前，识别并推进某些测试覆盖率的改进工作。
-->

- `mooncake-store/src/tiered_cache/ascend_tier.cpp`: `2025-12-03` - 目标覆盖率 > 80%

**测试文件：** `mooncake-store/tests/ascend_tier_test.cpp`

**测试类别：**

1. **MemoryType测试**
   - `MemoryTypeIncludesAscendNpu`: 验证MemoryType枚举包含ASCEND_NPU
   - `MemoryTypeToStringAllTypes`: 测试所有内存类型的字符串转换

2. **AscendBuffer测试**（需要`USE_ASCEND_CACHE_TIER`宏）
   - `AscendBufferLifecycle`: 测试构造、访问器方法、析构
   - `AscendBufferMoveConstructor`: 测试移动构造和所有权转移
   - `AscendBufferMoveAssignment`: 测试移动赋值和清理

3. **AscendCacheTier测试**
   - `AscendCacheTierConstruction`: 验证所有构造参数
   - `AscendCacheTierInit`: 测试初始化和空backend处理
   - `AscendCacheTierAllocateAndFree`: 基本分配/释放周期
   - `AscendCacheTierCapacityLimit`: 容量溢出检测
   - `AscendCacheTierAllocateZeroSize`: 无效参数处理
   - `AscendCacheTierMultipleAllocations`: 多次同时分配

4. **TieredBackend集成测试**
   - `TieredBackendAscendTierInit`: ASCEND_NPU tier配置
   - `TieredBackendMixedTiers`: ASCEND_NPU + DRAM tier组合
   - `TieredBackendAllocateOnAscend`: 通过backend分配
   - `TieredBackendWriteToAscend`: DRAM → Ascend数据拷贝

5. **数据生命周期测试**
   - `AscendTierCompleteLifecycle`: Allocate → Write → Get → Free 周期
   - `CopyAscendToDramWithVerification`: 写入测试数据到Ascend，拷贝回DRAM，验证数据完整性

6. **线程安全测试**
   - `ThreadSafetyAllocateFree`: 4线程 × 10次操作的并发分配/释放

7. **边界条件测试**
   - `AscendCacheTierInitTwice`: 幂等初始化
   - `AscendCacheTierInitWithInvalidDeviceId`: 设备ID边界验证
   - `AscendCacheTierInitWithZeroCapacity`: 零容量处理
   - `AscendCacheTierAllocateWithoutInit`: 未初始化tier操作
   - `AscendCacheTierFreeWithoutInit`: 未初始化tier的Free操作

8. **多态性测试**
   - `DeleteViaBasePointer`: 虚析构函数正确性

9. **资源管理测试**
   - `DestructorCleansUpAllocations`: RAII清理验证
   - `DestructorWithOutstandingAllocations`: 未释放分配的警告

##### 集成测试
当前暂不支持集成测试。

### 毕业标准

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

**功能实现：**
- [x] AscendCacheTier类实现完成，所有 CacheTier 接口方法已实现（Init、Allocate、Free）
- [x] AscendBuffer类实现RAII模式，自动管理设备内存生命周期
- [x] 设备内存分配和释放功能正常
- [x] 跨内存类型数据拷贝功能（CopyAscendToDram、CopyDramToAscend）已实现
- [x] 通过CopierRegistry静态注册拷贝函数
- [x] 使用原子操作（CAS）实现线程安全的容量管理
- [x] ACL框架初始化和清理（aclInit、aclFinalize）

**测试要求：**
- [x] 单元测试覆盖核心功能（构造函数、Init、Allocate、Free）
- [x] 单元测试覆盖AscendBuffer生命周期和移动语义
- [x] 单元测试覆盖错误处理场景（空指针、容量不足、设备错误等）
- [x] 单元测试覆盖数据拷贝并验证数据完整性
- [x] 线程安全测试（多线程并发Allocate/Free）
- [ ] 集成测试验证与 TieredBackend 的完整集成
- [ ] 初步的 e2e 测试已完成（需要昇腾设备环境）

**文档要求：**
- [x] 代码注释完整，关键函数有详细说明
- [x] oFEP 提案文档基本完成

#### 🔸 Beta 阶段

**功能完善：**
- [ ] 所有 Alpha 阶段功能稳定运行
- [ ] 性能优化完成（如需要）
- [ ] 错误处理和日志记录完善
- [ ] 支持多设备配置和并发访问

**测试要求：**
- [ ] 完整的单元测试覆盖（覆盖率 > 80%）
- [ ] 集成测试覆盖所有主要使用场景
- [ ] e2e 测试在真实环境中稳定运行至少两周
- [ ] 多线程并发测试通过
- [ ] 压力测试和性能测试完成

**监控和可观测性：**
- [ ] 关键操作的指标已添加
- [ ] 错误日志和诊断信息完善
- [ ] 性能指标收集机制就绪

**用户反馈：**
- [ ] 收集至少 3 个内部用户的反馈
- [ ] 解决所有已知问题和缺陷

#### 🟢 GA 阶段

**生产就绪：**
- [ ] 在至少 2 个生产环境中稳定运行超过 1 个月
- [ ] 所有 Beta 阶段反馈的问题已解决
- [ ] 性能指标满足预期（如：缓存命中率、数据传输延迟等）
- [ ] 无严重 bug 或稳定性问题

**文档和培训：**
- [ ] 用户文档完整，包含使用示例和最佳实践
- [ ] API 文档完整
- [ ] 故障排除指南已编写

**社区和生态：**
- [ ] 有真实用户在生产环境中使用
- [ ] 社区反馈积极，无重大投诉

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
### 升级/降级策略

#### 升级策略

**向前兼容性：**
- AscendCacheTier作为 CacheTier 接口的实现，与现有 TieredBackend 完全兼容
- 升级时无需修改现有代码，只需在配置中添加AscendCacheTier层
- 如果未启用 `USE_ASCEND_CACHE_TIER` 宏，代码会降级为使用主机内存，不会导致编译或运行时错误

**配置迁移：**
- 新版本支持在配置文件中添加AscendCacheTier层
- 现有配置无需修改，AscendCacheTier 是可选的缓存层
- 如果配置中未指定 AscendCacheTier，系统行为与之前完全一致

**功能启用：**
- 通过编译时宏 `USE_ASCEND_CACHE_TIER` 控制功能启用
- 运行时通过配置文件决定是否使用 AscendCacheTier
- 如果设备不可用或初始化失败，系统会记录警告但不会崩溃

#### 降级策略

**功能禁用：**
- 如果编译时未定义 `USE_ASCEND_CACHE_TIER`，相关功能自动禁用
- 运行时如果AscendCacheTier初始化失败，TieredBackend 会跳过该层，继续使用其他缓存层
- 禁用AscendCacheTier不会影响其他缓存层的功能

**数据迁移：**
- 降级时，AscendCacheTier 中的数据不会自动迁移
- 建议在降级前清空AscendCacheTier中的数据，或等待数据自然过期
- 如果降级后重新启用，需要重新填充缓存数据

**测试要求：**
- 升级测试：验证从旧版本升级到包含AscendCacheTier的新版本
- 降级测试：验证从新版本降级到旧版本（AscendCacheTier 数据会被丢弃）
- 升级->降级->升级循环测试：验证多次升级降级循环的稳定性

### 版本倾斜策略

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

- [x] **编译时宏控制**（`USE_ASCEND_CACHE_TIER`）  
  - 功能开关名称：`USE_ASCEND_CACHE_TIER`  
  - 依赖该功能开关的组件：`mooncake-store`
  - 启用方式：在编译时定义 `USE_ASCEND_CACHE_TIER` 宏
  - 禁用方式：不定义该宏，相关功能会自动降级
  
- [x] **运行时配置控制**  
  - 描述机制实现方式：通过 TieredBackend 配置文件控制是否创建AscendCacheTier层
  - 启用/禁用该功能是否会导致控制平面需要停机？否，只需重启 Mooncake Store 进程
  - 启用/禁用该功能是否会导致节点需要停机或重新部署（reprovision）？否，只需重启进程

###### 启用该功能会改变任何默认行为吗？

**不会改变默认行为。**

- AscendCacheTier是可选的缓存层，只有在配置文件中明确添加时才会使用
- 如果未配置 AscendCacheTier，系统行为与之前完全一致
- 如果AscendCacheTier初始化失败，系统会记录警告但继续使用其他缓存层
- 所有操作都是向后兼容的，不会影响现有功能

###### 该功能一旦启用，是否可以禁用（即我们可以回滚启用）？

**可以禁用，支持回滚。**

**禁用方式：**
1. **编译时禁用**：重新编译时不定义 `USE_ASCEND_CACHE_TIER` 宏
2. **运行时禁用**：从配置文件中移除AscendCacheTier配置，重启进程

**对现有工作负载的影响：**
- 禁用后，AscendCacheTier 中的数据不会被自动迁移，会丢失
- 系统会自动回退到使用其他缓存层（如 DRAM）
- 不会破坏现有应用程序的行为，只是缓存性能可能下降
- 建议在禁用前等待缓存数据自然过期，或手动清空缓存

**注意：**在元数据中，`disable-supported` 字段已设置为 `true`。

###### 如果该功能之前已回滚，现在我们重新启用它会发生什么情况？

**重新启用时的行为：**

1. **编译时重新启用**：重新编译时定义 `USE_ASCEND_CACHE_TIER` 宏，功能会重新可用
2. **运行时重新启用**：在配置文件中重新添加AscendCacheTier配置，重启进程后会重新初始化
3. **数据状态**：重新启用后，AscendCacheTier 是空的，需要重新填充缓存数据
4. **初始化检查**：如果设备不可用或 ACL 库版本不兼容，初始化会失败，但不会影响其他功能
5. **无缝切换**：重新启用后，系统会立即开始使用 AscendCacheTier，无需额外配置

###### 是否有任何针对功能启用/禁用的测试？

**单元测试覆盖：**

1. **启用场景测试**（`USE_ASCEND_CACHE_TIER` 定义时）：
   - 测试所有核心功能（Put、Get、Delete、Contains、AsDataSource）
   - 测试设备内存分配和释放
   - 测试跨内存类型数据拷贝
   - 测试错误处理和边界条件

2. **禁用场景测试**（`USE_ASCEND_CACHE_TIER` 未定义时）：
   - 测试拷贝函数返回明确的错误信息
   - 测试系统能够正常编译和运行（虽然功能受限）
   - 验证不会导致运行时崩溃

### 推出、升级和回滚规划
<!--
当该功能计划从 Beta 阶段发布至正式版本时，必须填写本节内容。
-->

###### 部署或回滚为何会失败？这会影响正在运行的工作负载吗？

**部署失败场景：**

1. **编译时失败**：
   - ACL 运行时库缺失或版本不兼容
   - 编译环境不支持昇腾设备
   - **影响**：无法编译，不会影响运行中的工作负载

2. **初始化失败**：
   - 昇腾设备不可用或设备ID无效
   - ACL 库初始化失败
   - 设备内存不足
   - **影响**：AscendCacheTier 初始化失败，系统会记录警告但继续使用其他缓存层，不影响运行中的工作负载

3. **运行时失败**：
   - 设备内存分配失败
   - 设备上下文设置失败
   - **影响**：相关操作会失败，但不会导致进程崩溃，系统会回退到其他缓存层

**回滚失败场景：**

1. **数据丢失**：
   - 回滚时AscendCacheTier中的数据会丢失
   - **影响**：缓存命中率可能暂时下降，但不会影响数据正确性

2. **配置错误**：
   - 配置文件格式错误
   - **影响**：TieredBackend 初始化可能失败，需要修复配置

**对运行中工作负载的影响：**

- **最小影响**：AscendCacheTier 是可选的缓存层，禁用后系统会继续使用其他缓存层
- **性能影响**：禁用后缓存性能可能下降，但不会导致功能失效
- **数据一致性**：所有操作都是原子性的，不会导致数据不一致

**高可用和大型集群考虑：**

- Mooncake Store 是进程内组件，不涉及多节点协调
- 每个进程独立管理自己的缓存层，不存在跨节点一致性问题
- 可以逐个节点进行部署和回滚，互不影响

###### 哪些具体指标应该通知回滚？

**关键指标：**

1. **错误率指标**：
   -AscendCacheTier操作失败率 > 5%
   - 设备内存分配失败次数持续增加
   - ACL 操作错误率上升

2. **性能指标**：
   - 缓存操作延迟显著增加（> 2倍基线）
   - 设备内存拷贝速度异常下降
   - 整体缓存命中率下降

3. **资源指标**：
   - 设备内存使用率异常（接近 100% 但仍有分配失败）
   - 设备上下文设置失败次数增加
   - 进程内存泄漏迹象

4. **稳定性指标**：
   - 进程崩溃或异常退出
   - 设备驱动异常
   - ACL 库版本不兼容错误

**回滚触发条件：**

- 连续 5 分钟内错误率 > 10%
- 设备不可用持续时间 > 10 分钟
- 严重内存泄漏导致系统不稳定
- 数据一致性错误

**监控建议：**

- 记录所有 ACL 操作的错误码和错误信息
- 监控设备内存使用情况
- 跟踪缓存命中率和性能指标
- 设置告警阈值，及时发现问题

###### 升级和回滚测试了吗？升级->降级->升级的路径测试了吗？

**测试状态：Alpha 阶段**

**已完成的测试：**

1. **编译时启用/禁用测试**：
   - 测试了定义和未定义 `USE_ASCEND_CACHE_TIER` 宏时的编译和运行
   - 验证了功能降级的正确性

**待完成的测试：**

1. **运行时配置变更测试**：
   - 需要测试添加和移除AscendCacheTier配置后的行为
   - 需要验证重新初始化机制

2. **数据迁移测试**：
   - 需要测试启用/禁用时的数据状态
   - 需要验证缓存数据的正确清理

3. **升级->降级->升级循环测试**：
   - 需要在实际环境中测试多次升级降级循环
   - 验证系统稳定性和数据一致性

4. **长时间运行测试**：
   - 测试在启用状态下长时间运行（> 24 小时）的稳定性
   - 验证内存泄漏和资源管理

5. **压力测试**：
   - 在高负载下测试启用/禁用操作
   - 验证系统在压力下的稳定性

**测试计划：**
- 在 Beta 阶段完成所有升级/回滚测试
- 建立自动化测试流程（如可能）
- 在实际部署环境中进行验证

###### 推出时是否伴随任何功能、API、API 类型的字段、标志等的弃用和/或删除？

**无弃用或删除。**

- 本提案是新增功能，不涉及任何现有功能的弃用或删除
- 所有现有 API 和接口保持不变
- 不修改任何现有配置格式或标志
- 完全向后兼容，不影响现有用户

### 监控要求
<!--
当计划将功能从 Beta 阶段纳入某个正式版本（release）时，必须完成此部分内容。
对于 GA（正式可用）阶段，该部分也是必须填写的：审批人应能够基于实际生产环境中的经验，确认之前各项回答的准确性。
-->

###### 操作员如何确定该功能是否正在被工作负载使用？

**通过配置检查：**
- 检查 TieredBackend 配置文件中是否包含AscendCacheTier配置
- 检查进程启动日志中是否有AscendCacheTier初始化成功的消息

###### 使用此功能的人如何知道它适用于他们的实例？

**通过日志信息：**
- 初始化成功日志：`AscendCacheTier {tier_id} initialized with capacity {capacity} bytes on device {device_id}`
- 操作日志：记录关键操作（Put/Get/Delete）的成功和失败
- 错误日志：记录设备错误、内存分配失败等异常情况

###### 增强的合理 SLO（服务级别目标）是什么？

**核心 SLO：**

1. **可用性 SLO**：
   -AscendCacheTier操作成功率 ≥ 99.5%（排除设备硬件故障）
   - 设备初始化成功率 ≥ 99.9%

2. **性能 SLO**：
   - Put 操作延迟：P99 ≤ 10ms（对于 1MB 数据）
   - Get 操作延迟：P99 ≤ 1ms（零拷贝返回指针）
   - 内存拷贝性能：H2D 和 D2H 拷贝速度 ≥ 理论带宽的 80%

3. **可靠性 SLO**：
   - 内存泄漏：24 小时内内存使用增长 ≤ 1%
   - 设备上下文错误率 ≤ 0.1%
   - 数据一致性：100%（所有操作都是原子性的）

4. **容量 SLO**：
   - 容量利用率：在正常负载下，使用率 ≤ 90%（预留 10% 缓冲）
   - 容量不足时的错误处理：100% 返回明确的错误信息，不导致崩溃

**注意：** 这些 SLO 需要在 Beta 阶段通过实际测试验证和调整。


**健康检查规则：**
- 成功率 < 99%：警告
- 成功率 < 95%：严重
- 延迟 P99 > 50ms：警告
- 内存使用率 > 95%：警告
- 设备错误率 > 1%：警告

### 依赖项
<!--
当计划将该功能从 Beta 阶段纳入某个正式版本（release）时，必须完成本节内容。
-->

###### 此功能是否依赖于集群中运行的任何特定服务？

**依赖项列表：**

```
- [昇腾 NPU 设备和驱动]
  - 使用说明：AscendCacheTier 需要访问昇腾 NPU 设备进行内存操作
  - 若该服务发生中断，对该功能的影响：
    -AscendCacheTier初始化失败，系统会记录错误但继续使用其他缓存层
    - 已缓存的数据可能无法访问，但不会导致进程崩溃
    - 新建工作负载的创建情况：无法使用 AscendCacheTier，回退到其他缓存层
  - 若该服务性能下降或错误率升高，对该功能的影响：
    - 缓存操作延迟增加，可能触发超时
    - 内存分配失败率上升，影响缓存功能
    - 系统会自动降级，使用其他缓存层

- [ACL 运行时库]
  - 使用说明：提供设备内存管理和数据拷贝的底层 API
  - 若该服务发生中断，对该功能的影响：
    - 编译时：如果库缺失，功能无法编译（通过条件编译控制）
    - 运行时：如果库加载失败，初始化会失败，系统会降级
  - 若该服务性能下降或错误率升高，对该功能的影响：
    - 所有设备操作都会受到影响
    - 错误率上升会导致缓存操作失败率增加
    - 系统会记录错误并可能触发告警

- [TieredBackend]
  - 使用说明：AscendCacheTier 通过 TieredBackend 进行初始化和数据拷贝
  - 若该服务发生中断，对该功能的影响：
    -AscendCacheTier无法初始化，功能不可用
    - 这是进程内依赖，不会单独中断
  - 若该服务性能下降或错误率升高，对该功能的影响：
    - 数据拷贝操作可能变慢
    - 可能影响缓存层的协调和管理
```

### 可扩展性
<!--
对于 Alpha 阶段，鼓励填写本节内容：评审人员应考虑这些问题，并尝试给出回答。
对于 Beta 阶段，本节为必填项：评审人员必须回答这些问题。
对于 GA（正式发布）阶段，本节同样为必填项：审批人员应能根据实际生产经验，确认之前给出的所有回答。
-->

###### 启用/使用此功能会导致任何新的 API 调用吗？

**无新的 Kubernetes API 调用。**

AscendCacheTier 是 Mooncake Store 的内部组件，不涉及 Kubernetes API 调用。所有操作都是进程内的，通过函数调用和内存操作完成。

**内部 API 调用：**

1. **ACL 运行时库调用**：
   - `aclrtSetDevice`：设置设备上下文（每次设备操作前调用）
   - `aclrtMalloc`：分配设备内存（Put 操作时调用）
   - `aclrtFree`：释放设备内存（Delete 操作或析构时调用）
   - `aclrtMemcpy`：内存拷贝（Put/Get 操作时调用）
   - **调用频率**：与缓存操作频率相关
   - **调用发起组件**：`AscendCacheTier` 类的方法

2. **TieredBackend 接口调用**：
   - `GetDataCopier()`：获取数据拷贝器（Put 操作时调用）
   - **调用频率**：与 Put 操作频率相同
   - **调用发起组件**：`AscendCacheTier::Put`

**注意：** 这些调用都是同步的、进程内的函数调用，不涉及网络或外部服务。

###### 启用/使用此功能是否会导致引入新的 API 类型？

**不引入新的 Kubernetes API 类型。**

AscendCacheTier 是内部实现组件，不暴露任何 Kubernetes API。所有配置和状态都是进程内的。

**内部数据结构：**

1. **AscendUnifiedPointer**：
   - 类型：C++ 结构体（非 API 对象）
   - 用途：封装设备指针和设备 ID
   - 数量：与缓存条目数量相同，由容量和条目大小决定

2. **AscendMemoryEntry**：
   - 类型：C++ 结构体（非 API 对象）
   - 用途：存储缓存条目的元数据
   - 数量：与缓存条目数量相同，由容量和条目大小决定

**配置数据结构：**

-AscendCacheTier的配置通过 TieredBackend 的配置文件（JSON）指定
- 不涉及 Kubernetes 资源对象（如 ConfigMap、CustomResource 等）

**注意：** 这些数据结构都是进程内的，不涉及 API Server 或 etcd。

###### 启用/使用此功能是否会导致对云提供商的任何新的 API 调用？

**无云提供商 API 调用。**

AscendCacheTier 直接与本地昇腾 NPU 设备交互，通过 ACL 运行时库访问设备，不涉及任何云服务 API 调用。

**设备访问方式：**
- 通过 PCIe 或专用总线直接访问昇腾 NPU 设备
- 使用设备驱动和 ACL 运行时库进行通信
- 所有操作都是本地的，不涉及网络或云服务

**注意：** 即使在云环境中部署，昇腾设备也是作为本地硬件资源存在的，不需要通过云 API 访问。

###### 启用/使用此功能是否会导致现有 API 对象的大小或数量增加？

**不影响任何 Kubernetes API 对象。**

AscendCacheTier 是内部实现，不修改任何 Kubernetes 资源对象的定义、大小或数量。

**内部内存使用：**

1. **每个AscendCacheTier实例**：
   - 固定开销：约 100-200 字节（成员变量、互斥锁等）
   - 动态开销：`storage_` 哈希表，每个条目约 50-100 字节（键字符串 + AscendMemoryEntry）

2. **每个缓存条目**：
   - 键字符串：取决于键的长度（通常 10-100 字节）
   - AscendMemoryEntry：约 24 字节（指针 + size_t）
   - AscendUnifiedPointer：约 16 字节（指针 + int）
   - 设备内存：实际数据大小

**注意：** 这些内存使用都是进程内的，不涉及 etcd 存储或 API 对象序列化。

###### 启用/使用此功能是否会导致现有 SLI/SLO 所涵盖操作的耗时增加？

**对现有操作的影响：**

1. **缓存 Put 操作**：
   - **新增步骤**：设备内存分配 + H2D 内存拷贝
   - **耗时增加**：约 1-5ms（取决于数据大小和设备性能）
   - **影响**：Put 操作延迟会增加，但这是可选的缓存层，不影响核心功能

2. **缓存 Get 操作**：
   - **新增步骤**：设备上下文设置（约 0.1ms）
   - **耗时增加**：最小（零拷贝返回指针）
   - **影响**：Get 操作延迟几乎不增加，性能优于 DRAM 到主机的拷贝

3. **缓存 Delete 操作**：
   - **新增步骤**：设备内存释放
   - **耗时增加**：约 0.1-0.5ms
   - **影响**：Delete 操作延迟略有增加，但影响很小

**性能权衡：**

- **短期**：Put 操作延迟增加（需要设备内存拷贝）
- **长期**：Get 操作性能提升（零拷贝，避免 D2H 传输）
- **整体**：对于频繁访问的数据，使用AscendCacheTier可以显著提升性能

**注意：** 这些操作都是可选的，如果性能不满足要求，可以禁用AscendCacheTier或调整缓存策略。

###### 启用/使用此功能是否会导致任何组件的资源使用率（CPU、RAM、磁盘、IO 等）不可忽略的增加？

**资源使用分析：**

1. **CPU 使用率**：
   - **增加量**：轻微增加（< 1%）
   - **原因**：
     - 设备上下文设置开销（每次操作约 0.1ms）
     - 内存拷贝操作（由设备 DMA 处理，CPU 开销小）
     - 互斥锁竞争（多线程环境下）
   - **影响**：可忽略，CPU 开销主要由设备硬件加速处理

2. **内存（RAM）使用率**：
   - **增加量**：每个AscendCacheTier实例约 1-10MB（取决于配置和条目数量）
   - **原因**：
     - `storage_` 哈希表的内存开销
     - 键字符串存储
     - 元数据结构（AscendMemoryEntry、AscendUnifiedPointer）
   - **注意**：实际数据存储在设备内存中，不在主机 RAM 中

3. **设备内存使用率**：
   - **增加量**：等于配置的 `capacity_`
   - **原因**：缓存数据存储在昇腾 NPU 设备内存中
   - **影响**：这是功能的核心目的，使用设备内存而不是主机内存

4. **磁盘 I/O**：
   - **增加量**：无增加
   - **原因**：所有操作都在内存中，不涉及磁盘

5. **网络 I/O**：
   - **增加量**：无增加
   - **原因**：所有操作都是进程内的，不涉及网络

6. **日志 I/O**：
   - **增加量**：轻微增加（每次错误操作记录日志）
   - **原因**：错误和警告日志
   - **影响**：可忽略，正常操作不记录详细日志

**不同规模的影响：**

- **小型集群**（< 10 节点）：资源增加可忽略
- **中型集群**（10-100 节点）：资源增加可忽略
- **大型集群**（> 100 节点）：资源增加可忽略，每个节点独立管理

**注意：** 主要资源使用是设备内存，这是功能的核心目的，而不是副作用。

###### 启用/使用此功能是否会导致某些节点资源（PID、套接字、inode 等）耗尽？

**资源耗尽风险评估：**

1. **PID（进程 ID）**：
   - **风险**：无风险
   - **原因**：AscendCacheTier 不创建新进程，所有操作都在现有进程内

2. **套接字（Socket）**：
   - **风险**：无风险
   - **原因**：不涉及网络通信，所有操作都是进程内的

3. **inode（文件描述符）**：
   - **风险**：低风险
   - **原因**：不直接打开文件，但 ACL 库可能使用文件描述符
   - **缓解措施**：通过设备数量限制间接控制（每个设备可能有少量文件描述符）

4. **设备内存**：
   - **风险**：中等风险（这是功能的核心资源）
   - **原因**：设备内存是有限的，如果配置不当可能导致耗尽
   - **缓解措施**：
     - 通过 `capacity_` 参数限制最大使用量
     - 实现容量检查机制（`HasSpace`）
     - 由 TieredBackend 负责数据淘汰和迁移
   - **限制**：每个AscendCacheTier实例的容量受配置限制

5. **主机内存（用于元数据）**：
   - **风险**：低风险
   - **原因**：元数据存储（哈希表）可能随着条目数量增长
   - **缓解措施**：
     - 容量限制间接限制了条目数量
     - 每个条目的元数据开销很小（约 100 字节）

**异常情况处理：**

1. **设备不可用**：
   - 初始化失败，系统降级，不占用资源
   - 已分配的内存会在析构时释放

2. **内存分配失败**：
   - Put 操作失败，返回错误，不导致资源泄漏
   - 已分配的资源会通过 RAII 自动清理

3. **设备上下文设置失败**：
   - 操作失败，返回错误，不影响其他资源
   - 不会导致资源泄漏

**性能测试计划：**

- [ ] **压力测试**：测试在最大容量下的资源使用情况
- [ ] **长时间运行测试**：验证资源泄漏（24+ 小时）
- [ ] **并发测试**：测试多线程环境下的资源竞争
- [ ] **设备内存耗尽测试**：验证容量限制机制的有效性

**注意：** 主要风险是设备内存耗尽，这通过容量限制和上层淘汰策略进行控制。

### 故障排除
<!--
**当该功能计划从 Beta 阶段发布至正式版本（release）时，本节必须填写。**

对于 **GA（正式发布）阶段**，本节同样为必填项：审批人员应能够根据实际生产环境中的经验，确认之前的各项回答。

当前的 **“排障（Troubleshooting）” 部分**，在功能发布流程中相当于临时承担了 **“运维手册（Playbook）”** 的角色。  
未来可能会将其拆分为一个专门的 `Playbook` 文档（可能还会包含一些监控信息）。但目前仍将其保留在此处。
-->

###### 如果 API 服务器和/或 etcd 不可用，此功能如何反应？

**不受影响。**

AscendCacheTier 是 Mooncake Store 的内部组件，不依赖 Kubernetes API Server 或 etcd。所有操作都是进程内的，直接与昇腾 NPU 设备交互。

**功能独立性：**
- 不读取或写入 etcd
- 不调用 Kubernetes API
- 不依赖集群状态
- 配置在进程启动时加载，运行时不需要外部服务

**故障场景：**
- API Server 不可用：AscendCacheTier 继续正常工作
- etcd 不可用：AscendCacheTier 继续正常工作
- 网络分区：AscendCacheTier 继续正常工作（本地设备访问）

**注意：** 如果 Mooncake Store 进程本身不可用，AscendCacheTier 当然也会不可用，但这是进程级别的问题，不是组件级别的问题。

###### 其他已知故障模式有哪些？

**故障模式 1：设备不可用或初始化失败**

- **检测方式（Detection）：**
  - 日志：初始化错误消息
- **缓解手段（Mitigations）：**
  - 系统自动降级，使用其他缓存层（如 DRAM）
  - 不影响现有工作负载，只是缓存性能下降
  - 可以禁用AscendCacheTier配置，完全回退
- **诊断信息（Diagnostics）：**
  - 日志级别：ERROR
  - 日志消息：`AscendCacheTier {tier_id} initialization failed: {error_message}`
  - 包含设备 ID、错误码、ACL 库版本等信息
- **测试情况（Testing）：**
  - 有测试用例：测试设备ID不可用时的初始化失败

**故障模式 2：设备内存分配失败**

- **检测方式（Detection）：**
  - 日志：设备内存分配错误消息
- **缓解手段（Mitigations）：**
  - Allocate 操作返回 `NO_AVAILABLE_HANDLE` 错误码
  - 原子操作自动回滚已预留的空间
  - 由 TieredBackend 触发数据淘汰或迁移
  - 可以增加容量配置或减少其他缓存层的使用
- **诊断信息（Diagnostics）：**
  - 日志级别：ERROR
  - 日志消息：`Insufficient space in tier {tier_id}: requested={size}, available={available}`
  - 或：`Failed to allocate device memory for tier {tier_id}`
  - 包含tier_id、所需大小、可用空间等信息
- **测试情况（Testing）：**
  - 有测试用例：测试容量限制和空间检查（`AscendCacheTierCapacityLimit`）

**故障模式 3：设备上下文设置失败**
- **检测方式（Detection）：**
  - 日志：设备上下文设置错误消息
- **缓解手段（Mitigations）：**
  - 操作失败，返回错误，不导致数据损坏
  - 系统会记录错误并可能触发告警
  - 可以重启进程或检查设备状态
- **诊断信息（Diagnostics）：**
  - 日志级别：ERROR
  - 日志消息：`{operation_name}: aclrtSetDevice failed, device: {device_id}, error: {error_code}`
  - 包含操作名称、设备 ID、错误码等信息
- **测试情况（Testing）：**
  - 有测试用例：测试设备上下文设置的错误处理
  - 需要模拟设备错误场景（可能需要 mock）

**故障模式 4：内存拷贝失败**
- **检测方式（Detection）：**
  - 日志：拷贝错误消息
- **缓解手段（Mitigations）：**
  - Copy 操作返回 `DATA_COPY_FAILED` 错误码
  - 通过 RAII 机制不会导致内存泄漏
  - 可以重试操作或检查设备状态
- **诊断信息（Diagnostics）：**
  - 日志级别：ERROR
  - 日志消息：`CopyDramToAscend: aclrtMemcpy failed, error: {error}`
  - 或：`CopyAscendToDram: destination buffer too small ({dst_size} < {src_size})`
  - 包含拷贝方向、错误信息等
- **测试情况（Testing）：**
  - 有测试用例：测试拷贝失败时的错误处理
  - 有测试用例：测试数据完整性验证（`CopyAscendToDramWithVerification`）


###### 如果未满足 SLO，应采取哪些步骤来确定问题？

**问题诊断步骤：**

1. **检查日志**：
   - 查看 ERROR 级别日志，查找错误消息
   - 查看 WARNING 级别日志，查找警告信息
   - 检查初始化日志，确认初始化是否成功
   - 查看操作日志，查找失败的操作

2. **检查设备状态**：
   - 验证昇腾 NPU 设备是否可用：`npu-smi info`
   - 检查设备驱动是否正常加载
   - 检查 ACL 运行时库版本是否兼容
   - 验证设备内存是否充足

3. **检查配置**：
   - 验证AscendCacheTier配置是否正确
   - 检查 `capacity_` 是否合理（不应超过设备可用内存）
   - 检查 `device_id` 是否有效
   - 验证 TieredBackend 配置是否正确

4. **检查资源使用**：
   - 检查设备内存使用情况
   - 检查主机内存使用情况（元数据）
   - 检查 CPU 使用率是否异常

5. **性能分析**：
   - 分析操作延迟分布，找出慢操作
   - 检查是否存在CAS重试过多（通过日志或性能分析工具）
   - 分析内存拷贝性能，检查是否达到预期带宽
   - 检查缓存命中率，评估缓存效果

6. **故障隔离**：
   - 如果问题持续，可以临时禁用 AscendCacheTier
   - 检查问题是否特定于某个设备或节点
   - 验证问题是否与特定工作负载相关

**常见问题和解决方案：**

- **成功率低**：检查设备状态、内存是否充足、ACL 库版本
- **延迟高**：检查设备性能、是否存在锁竞争、数据大小
- **内存不足**：调整容量配置、触发数据淘汰、检查内存泄漏
- **设备错误**：检查设备驱动、重启设备、联系硬件支持

## 实施历史

- **2025-12-03**：oFEP-0028 创建，提案进入初步草案（provisional）状态
- **2026-01-27**：更新提案以反映实际代码实现
  - 架构从 Put/Get/Delete 模型更新为 Allocate/Free 模型
  - 新增 AscendBuffer 类，实现 RAII 模式
  - 线程安全从互斥锁改为原子操作（CAS）
  - 新增 CopierRegistry 拷贝函数注册机制
  - 更新类图、时序图和测试计划

## 缺点

1. **硬件依赖**：
   - 需要昇腾 NPU 设备，增加了硬件要求
   - 在不支持昇腾的环境中无法使用（虽然可以通过条件编译禁用）
   - 设备故障会影响缓存功能

2. **复杂性增加**：
   - 引入了设备内存管理的复杂性
   - 需要处理设备上下文、内存拷贝等底层细节
   - 增加了代码维护成本

4. **调试困难**：
   - 设备内存问题难以调试（无法直接访问）
   - 需要设备特定的工具和知识
   - 错误信息可能不够直观

5. **资源消耗**：
   - 占用设备内存，可能影响其他使用设备的应用
   - 需要合理配置容量，避免资源浪费

6. **兼容性限制**：
   - 依赖特定版本的 ACL 运行时库
   - 不同设备型号可能有兼容性问题
   - 升级 ACL 库可能需要代码调整

## 替代方案

### 方案 1：使用主机内存（DRAM）缓存

**描述**：不使用设备内存，只使用主机 DRAM 作为缓存层。

**为什么排除**：
- 无法利用设备内存的高带宽特性
- 对于频繁访问的数据，仍然需要 H2D 传输
- 无法充分利用昇腾设备的硬件优势

## 所需基础设施
**代码仓库**
 - 使用现有的Mooncake代码仓库，无需新建
