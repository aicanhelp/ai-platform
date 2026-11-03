# 声明式集群版本升级方案-支持 Helm 组件

## 目录

1. [概述](#1-概述)
   - 1.1 设计目标
   - 1.2 设计范围
   - 1.3 设计约束
   - 1.4 术语表
2. [整体架构设计](#2-整体架构设计)
   - 2.1 系统架构图
   - 2.2 组件交互关系
3. [ComponentVersion CRD 详细设计](#3-componentversion-crd-详细设计)
   - 3.1 ComponentVersion 类型定义
   - 3.2 YAML 类型字段定义
   - 3.3 Helm 类型字段定义
   - 3.4 Inline 类型字段定义
   - 3.5 CRD YAML 定义
   - 3.6 CRD 版本迁移设计
4. [HelmInstaller 详细设计](#4-helminstaller-详细设计)
   - 4.1 核心组件架构
   - 4.2 HelmInstaller 执行流程图
   - 4.3 核心接口定义
   - 4.4 健康检查
   - 4.5 Hooks 执行引擎
5. [YamlInstaller 详细设计](#5-yamlinstaller-详细设计)
   - 5.1 核心组件架构
   - 5.2 YamlInstaller 执行流程图
   - 5.3 核心接口定义
   - 5.4 健康检查
   - 5.5 YAML Uninstall 流程
6. [HealthCheck 共享包设计](#6-healthcheck-共享包设计)
   - 6.1 类型定义
7. [模板变量系统与 TemplateContext 详细设计](#7-模板变量系统与-templatecontext-详细设计)
   - 7.1 TemplateContext 扩展策略
   - 7.2 模板变量系统
   - 7.3 TemplateContext 构建流程
8. [DAG 集成详细设计](#8-dag-集成详细设计)
   - 8.1 执行器注册
   - 8.2 ComponentVersionStore
   - 8.3 DAG 构建与执行流程
   - 8.4 核心接口定义
   - 8.5 状态模型、幂等性与兼容性设计
9. [完整安装流程详细设计](#9-完整安装流程详细设计)
   - 9.1 安装流程图
10. [完整升级流程详细设计](#10-完整升级流程详细设计)
    - 10.1 升级流程图
11. [迁移策略详细设计](#11-迁移策略详细设计)
    - 11.1 迁移流程图
    - 11.2 Feature Gate 设计
12. [错误处理与恢复](#12-错误处理与恢复)
    - 12.1 错误处理流程图
13. [测试设计](#13-测试设计)
    - 13.1 单元测试
    - 13.2 集成测试
    - 13.3 E2E 测试
14. [工作量与任务拆解](#14-工作量与任务拆解)
    - 14.1 工作量评估
    - 14.2 Sprint 计划
    - 14.3 里程碑
15. [附录](#15-附录)
    - 15.1 参考文档
    - 15.2 术语表
16. [Story 分解与最小集/增强集拆解](#16-story-分解与最小集增强集拆解)
    - 16.1 设计原则
    - 16.2 原始 Story 规格
    - 16.3 最小集（5 个 Milestone）
    - 16.4 增强集（最小集完成后迭代）
    - 16.5 对比总览
    - 16.6 Milestone 依赖关系

## 1. 概述

### 1.1 设计目标

本设计文档提供完整的实现方案，包括：

- **HelmInstaller**: Helm 组件的 Chart 获取、渲染、部署
- **DAG 集成**: 执行器注册与调度流程

### 1.2 设计范围

| 范围 | 说明 |
| ------ | ------ |
| CRD 扩展 | ComponentVersion 新增 helm 类型的完整字段定义 |
| 核心安装器 | HelmInstaller、YamlInstaller 的完整实现 |
| DAG 集成 | HelmComponentExecutor |
| 迁移策略 | Feature Gate、向后兼容、灰度发布 |

### 1.3 设计约束

| 约束 | 说明 |
| ------ | ------ |
| 向后兼容 | 必须支持从现有硬编码 Phase 平滑迁移 |
| 离线环境 | Helm Chart 支持本地缓存 |
| 接口复用 | 复用现有 NeedExecute() 接口 |
| 安全性 | 制品必须支持 checksum 校验 |

### 1.4 术语表

| 术语 | 定义 |
| ------ | ------ |
| **HelmInstaller** | 负责 Helm Chart 获取、渲染、部署的安装器 |
| **ComponentVersion** | 组件版本 CRD，定义组件的类型、配置、依赖等 |
| **ReleaseImage** | 发布版本清单 CRD，定义安装和升级的组件列表 |

## 2. 整体架构设计

### 2.1 系统架构图

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BKECluster                                         │
│  spec.desiredVersion: v2.6.0                                                    │
└────────────────────────────────┬────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            ReleaseImage                                         │
│  spec.install.components: [coredns/v1.11.1, openfuyao-addon/v26.03, ...]        │
│  spec.upgrade.components: [coredns/v1.11.1, openfuyao-addon/v26.03, ...]        │
└────────────────────────────────┬────────────────────────────────────────────────┘
                                  │ 按 (name, version) 定位
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         bke-manifests (ComponentVersion)                        │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ coredns/v1.11.1/component.yaml             ← type: helm                 │    │
│  │   ├── helm.chart.oci: registry/charts/coredns                           │    │
│  │   ├── helm.values: (带模板变量)                                          │    │
│  │   └── helm.healthCheck: PodReady + EndpointReady                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└────────────────────────────────┬────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            DAG Scheduler                                        │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        Component Executor Factory                       │    │
│  │                                                                         │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐                     │    │
│  │  │    Helm      │  │   Inline     │  │   YAML     │                     │    │
│  │  │   Component  │  │   Component  │  │  Component │                     │    │
│  │  │   Executor   │  │   Executor   │  │  Executor  │                     │    │
│  │  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘                     │    │
│  │         │                 │                │                             │    │
│  └─────────┼─────────────────┼────────────────┼────────────────────────────┘    │
│            │                 │                │                                 │
│            ▼                 ▼                ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                          Installer Layer                                │    │
│  │                                                                         │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │                       HelmInstaller                             │    │    │
│  │  │  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐    │    │    │
│  │  │  │    Chart       │  │    Values      │  │    Helm         │    │    │    │
│  │  │  │   Fetcher      │  │   Renderer     │  │  Action Exec    │    │    │    │
│  │  │  └────────┬───────┘  └────────┬───────┘  └────────┬────────┘    │    │    │
│  │  │           │                   │                   │             │    │    │
│  │  │           └───────────────────┼───────────────────┘             │    │    │
│  │  │                               ▼                                 │    │    │
│  │  │                      ┌────────────────┐                         │    │    │
│  │  │                      │ HealthChecker  │                         │    │    │
│  │  │                      └────────────────┘                         │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                         │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │                     YamlInstaller                               │    │    │
│  │  │  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐    │    │    │
│  │  │  │  BundleStore   │  │  YAML Parser   │  │  K8s Client     │    │    │    │
│  │  │  │(清单加载)      │  │  (解析/分组)    │  │  (Apply/Delete) │    │    │    │
│  │  │  └────────────────┘  └────────────────┘  └─────────────────┘    │    │    │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │    │    │
│  │  │  │              HealthChecker (Pod/Endpoint/Custom)        │    │    │    │
│  │  │  └─────────────────────────────────────────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 组件交互关系

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              组件交互关系图                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │  BKECluster      │
                    │  Reconciler      │
                    └────────┬─────────┘
                              │
                              │ 1. 解析 ReleaseImage
                              ▼
                     ┌──────────────────┐
                     │  ReleaseImage    │
                     │  Parser          │
                     └────────┬─────────┘
                              │
                              │ 2. 加载 ComponentVersion
                              ▼
                     ┌──────────────────┐
                     │  ManifestStore   │
                     └────────┬─────────┘
                              │
                              │ 3. 构建 DAG
                              ▼
                     ┌──────────────────┐
                     │  DAG Builder     │
                     └────────┬─────────┘
                              │
                              │ 4. 调度执行
                              ▼
                     ┌──────────────────┐
                     │  DAG Scheduler   │
                     └────────┬─────────┘
                              │
                ┌─────────────┼──────────────┐
                │             │              │
                ▼             ▼              ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │    Helm      │ │   Inline     │ │    YAML      │
        │   Component  │ │   Component  │ │   Component  │
        │  Executor    │ │   Executor   │ │   Executor   │
        └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
               │                │                │
               ▼                ▼                ▼
       ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
       │    Helm      │ │  Component   │ │    Yaml      │
       │  Installer   │ │  Factory     │ │  Installer   │
       └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
              │                │                │
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │ Helm SDK     │ │ Phase        │ │ K8s Client   │
      │ (helm/v3)    │ │ Execute()    │ │ (Apply)      │
      └──────────────┘ └──────────────┘ └──────────────┘
```

## 3. ComponentVersion CRD 详细设计

### 3.1 ComponentVersion 类型定义

> **复用说明**：现有 `api/v1alpha1/componentversion_types.go` 已定义 `ComponentVersionSpec`/`ComponentType`/`InlineSpec`/`SubComponent`/`CompatibilitySpec`/`Constraint`/`Dependency`/`UpgradeStrategySpec`/`ResourceSpec` 等类型。本节中这些类型为**复用现有**（仅新增 `Helm`/`YAML` 两个字段及对应 `*Spec` 类型），下文以「✅复用」「🆕新增」标注。路径修正：原设计写 `pkg/api/v1alpha1/...` 有误，实际为 `api/v1alpha1/...`（无 `pkg/` 前缀）。

```go
// api/v1alpha1/componentversion_types.go

// ComponentVersionSpec 定义组件版本规格 ✅复用现有，仅新增 Helm/YAML 字段
type ComponentVersionSpec struct {
    // 组件名称
    Name string `json:"name"`
    
    // 组件类型: yaml, helm, inline
    Type ComponentType `json:"type"`
    
    // 组件版本
    Version string `json:"version"`
    
    // Helm 类型配置 (type=helm 时必填)
    Helm *HelmSpec `json:"helm,omitempty"`
    
    // YAML 类型配置 (type=yaml 时必填)
    YAML *YAMLSpec `json:"yaml,omitempty"`
    
    // Inline 类型配置 (type=inline 时必填)
    Inline *InlineSpec `json:"inline,omitempty"`
    
    // 兼容性约束
    Compatibility CompatibilitySpec `json:"compatibility,omitempty"`
    
    // 依赖关系
    Dependencies []Dependency `json:"dependencies,omitempty"`
    
    // 子组件列表
    // type=yaml 时: 全包含语义, 所有子组件都会被安装
    SubComponents []SubComponent `json:"subComponents,omitempty"`
    
    // 升级策略
    UpgradeStrategy UpgradeStrategySpec `json:"upgradeStrategy,omitempty"`
    
    // Kubernetes 资源定义列表
    // 注意: 此字段位于顶层是历史原因——最初无独立 YAML 类型, Resources 用于所有类型
    // 新增 YAML 类型后, 理论上应迁移至 YAMLSpec, 但为保持向后兼容暂不移动
    Resources []ResourceSpec `json:"resources,omitempty"`
}

// ComponentType 定义组件类型 ✅复用现有 (含 helm/yaml/inline 三值)
type ComponentType string

const (
    ComponentTypeYAML     ComponentType = "yaml"
    ComponentTypeHelm     ComponentType = "helm"
    ComponentTypeInline   ComponentType = "inline"
)

// CompatibilitySpec 定义兼容性约束 ✅复用现有
type CompatibilitySpec struct {
    // 约束列表
    Constraints []Constraint `json:"constraints,omitempty"`
}

// Constraint 定义单个兼容性约束 ✅复用现有
type Constraint struct {
    // 依赖组件名称
    Component string `json:"component"`
    
    // 版本规则 (semver range, 如 ">=1.26.0")
    Rule string `json:"rule"`
}

// Dependency 定义组件间依赖关系 ✅复用现有
type Dependency struct {
    // 依赖组件名称
    Name string `json:"name"`
    
    // 依赖阶段 (Install / Upgrade)
    Phase string `json:"phase,omitempty"`
}

// UpgradeStrategySpec 定义升级策略 ✅复用现有
// 这是 DAG 调度层策略, 适用于所有组件类型 (helm/yaml/inline)
// 与各类型的专属策略互补:
// - Helm:   HelmStrategySpec 控制 helm 命令参数, UpgradeStrategy 控制调度和失败处理
// - YAML:  无专属策略, 仅使用 UpgradeStrategy
// 两者的 Mode 字段含义不同:
// - UpgradeStrategy.Mode = Rolling/Parallel/Batch (节点并发策略)
// - HelmStrategySpec.Mode = Install/Upgrade/Rollback (Helm 操作类型)
type UpgradeStrategySpec struct {
    // 升级模式: Rolling / Parallel / Batch (节点并发策略)
    Mode string `json:"mode,omitempty"`
    
    // 批量大小 (Batch 模式下每批节点数)
    BatchSize int `json:"batchSize,omitempty"`
    
    // 超时时间
    Timeout string `json:"timeout,omitempty"`
    
    // 失败策略: FailFast / Continue / Rollback
    // FailFast: 立即终止整个组件执行
    // Continue: 记录警告, 继续执行下一个节点/批次
    // Rollback: 回滚后继续 (Helm 执行 helm rollback)
    //           注意: Helm 若 Strategy.Atomic=true, Helm SDK 已自动回滚, 无需额外调用
    FailurePolicy string `json:"failurePolicy,omitempty"`
}

// SubComponent 定义子组件引用 ✅复用现有
type SubComponent struct {
    // 子组件名称
    Name string `json:"name"`
    
    // 子组件版本
    Version string `json:"version"`
}

// ResourceSpec 定义 Kubernetes 资源 ✅复用现有
//
// 设计思路 — Data、StringData 与 Manifest 三种资源定义方式:
//
// ResourceSpec 支持三种方式定义 K8s 资源内容, 通过 Kind 字段分发:
//
// 1. Data (ConfigMap 专属): key-value 字符串, 直接作为 ConfigMap.data
//    - 场景: 简单的配置文件, 如 bkeagent 的 logLevel/snapshotter 等参数
//    - 优势: 结构化、可校验, 无需写完整 ConfigMap YAML
//    - 代码: provisionConfigMap() 读取 Data 字段创建 ConfigMap 对象
//
// 2. StringData (Secret 专属): key-value 明文字符串, 自动转 base64 编码
//    - 场景: Secret 中的证书、密钥等敏感数据
//    - 优势: 用户写明文, 代码自动转 base64, 避免手动编码出错
//    - 如果用 Manifest 方式, Secret.data 需要 base64 编码, 不友好且易错
//    - 代码: provisionSecret() 读取 StringData, 自动转为 Secret.data (base64)
//
// 3. Manifest (通用): 原始 YAML 字符串, 解析为 unstructured.Unstructured 后创建
//    - 场景: Deployment/Service/CRD 等复杂资源, 声明式字段无法表达完整定义
//    - 优势: 支持任意 K8s 资源, 灵活性最高
//    - 代码: provisionFromManifest() 解析 YAML 并创建资源
//    - 额外用途: CollectComponentManifests() 收集 Manifest 作为 YAML 组件的清单
//
// 分发逻辑 (provisionResource):
//   - Manifest 非空且 Kind 非 ConfigMap/Secret → 直接用 Manifest
//   - Kind == ConfigMap → 用 Data 创建
//   - Kind == Secret → 用 StringData 创建 (避免手动 base64)
//   - 其他 Kind 且无 Manifest → 报错
//
// ConfigMap 用 Data 的原因: 与 Secret 对称设计, 且略简洁 (无需写完整 YAML)
// Secret 用 StringData 的原因: 避免 base64 编码, 这是核心价值
// 其他资源用 Manifest 的原因: 声明式字段无法表达完整 K8s 资源定义
type ResourceSpec struct {
    // 资源类型 (ConfigMap / Secret / Deployment / CRD 等)
    Kind string `json:"kind"`
    
    // API 版本
    APIVersion string `json:"apiVersion"`
    
    // 命名空间
    Namespace string `json:"namespace,omitempty"`
    
    // 资源名称
    Name string `json:"name"`
    
    // 标签 (自动注入到创建的资源中)
    Labels map[string]string `json:"labels,omitempty"`
    
    // ConfigMap 数据 (Kind=ConfigMap 时使用, 直接作为 ConfigMap.data)
    Data map[string]string `json:"data,omitempty"`
    
    // Secret 明文数据 (Kind=Secret 时使用, 自动转 base64 编码为 Secret.data)
    // 核心价值: 用户写明文, 避免手动 base64 编码
    StringData map[string]string `json:"stringData,omitempty"`
    
    // 原始 Manifest 内容 (通用, 任意 Kind 均可使用)
    // ConfigMap/Secret 优先用 Data/StringData, 其他 Kind 用 Manifest
    // 也用于 YAML 组件的清单收集 (CollectComponentManifests)
    Manifest string `json:"manifest,omitempty"`
}
```

### 3.2 YAML 类型字段定义

```go
// YAMLSpec 定义 YAML 清单组件规格 🆕新增 (api/v1alpha1 扩展)
//
// 设计思路 — 移除外部 URL 清单支持:
// YAMLSpec 不再支持通过外部 URL 引用 YAML 清单文件。所有清单通过以下两种方式提供:
// 1. Bundle 文件: components/<name>/<version>/*.yaml (由 BundleStore 加载)
// 2. 内联 Resources: ComponentVersionSpec.Resources (顶层字段)
// 这消除了对外部网络的依赖，简化了缓存管理，且 Bundle 已包含所有清单。
type YAMLSpec struct {
    // 注意: 内联 K8s 资源定义通过 ComponentVersionSpec.Resources (顶层) 提供
    // YAML 清单通过 Bundle 文件 (components/<name>/<version>/*.yaml) 加载
    // 不再支持外部 URL 引用，所有清单均从本地 Bundle 获取
    
    // 部署目标命名空间
    Namespace string `json:"namespace,omitempty"`
    
    // 应用策略: ServerSideApply, Replace, CreateOnly
    ApplyStrategy string `json:"applyStrategy,omitempty"`
    
    // 是否启用裁剪 (按 label selector 删除不再需要的资源)
    Prune bool `json:"prune,omitempty"`
    
    // 裁剪使用的标签选择器
    PruneLabelSelector map[string]string `json:"pruneLabelSelector,omitempty"`
    
    // 健康检查配置 (应用清单后验证 Pod/Endpoint 就绪)
    // 类型定义见第 6 章 HealthCheck 共享包设计 (6.1 节)
    HealthCheck *HealthCheckSpec `json:"healthCheck,omitempty"`
}
```

### 3.3 Helm 类型字段定义

```go
// HelmSpec 定义 Helm 组件规格 🆕新增 (api/v1alpha1 扩展)
type HelmSpec struct {
    // Chart 配置
    Chart ChartSpec `json:"chart"`
    
    // 命名空间
    Namespace string `json:"namespace"`
    
    // Release 名称
    ReleaseName string `json:"releaseName"`
    
    // Values 模板 (支持模板变量)
    Values map[string]interface{} `json:"values,omitempty"`
    
    // 自定义 Values 文件列表
    ValuesFiles []string `json:"valuesFiles,omitempty"`
    
    // 安装策略
    Strategy HelmStrategySpec `json:"strategy,omitempty"`
    
    // 健康检查配置
    // 类型定义见第 6 章 HealthCheck 共享包设计 (6.1 节)
    HealthCheck HealthCheckSpec `json:"healthCheck,omitempty"`
    
    // 回滚配置
    Rollback RollbackSpec `json:"rollback,omitempty"`
    
    // 卸载配置
    Uninstall UninstallSpec `json:"uninstall,omitempty"`
    
    // 前置安装钩子
    PreInstallHooks []HookSpec `json:"preInstallHooks,omitempty"`
    
    // 前置卸载钩子
    PreUninstallHooks []HookSpec `json:"preUninstallHooks,omitempty"`
}

// ChartSpec 定义 Chart 规格
type ChartSpec struct {
    // OCI Registry 配置
    OCI *OCIChartSpec `json:"oci,omitempty"`
    
    // HTTP URL (支持模板变量，如 "{{imageRegistry}}/charts/coredns-{{componentVersion}}.tgz")
    URL string `json:"url,omitempty"`
    
    // 本地路径
    LocalPath string `json:"localPath,omitempty"`
    
    // 校验和
    Checksum string `json:"checksum,omitempty"`
}

// OCIChartSpec 定义 OCI Chart 规格
type OCIChartSpec struct {
    // OCI 仓库地址 (支持模板变量，如 "{{imageRegistry}}/charts/coredns")
    Repository string `json:"repository"`
    
    // Chart 标签 (支持模板变量，如 "{{componentVersion}}")
    Tag string `json:"tag"`
}

// HelmStrategySpec 定义 Helm 安装策略
//
// 设计思路 — 与 UpgradeStrategySpec 的分工:
// HelmStrategySpec 是 Helm SDK 层策略, 控制 helm 命令参数 (--wait/--atomic/--timeout)
// UpgradeStrategySpec 是 DAG 调度层策略, 控制节点间并发和失败处理 (Rolling/Batch/FailFast)
// 两者互补: UpgradeStrategy 决定"何时/如何调度", HelmStrategy 决定"helm 命令怎么执行"
//
// Mode 字段的三种值:
// - Install:  显式指定 helm install (Release 不存在时)
// - Upgrade:  显式指定 helm upgrade (Release 已存在时)
// - Rollback: 显式触发 helm rollback (恢复到上一个 Release 版本)
//
// Mode 为空时, 由 VersionContext 自动判定:
//   HasCurrent=false → Install, HasCurrent=true → Upgrade
//
// Mode=Rollback 的使用场景:
// 1. 用户显式回滚: 升级后发现问题, 修改 ComponentVersion 指定 Mode=Rollback 触发回滚
// 2. 失败自动回滚: 当 UpgradeStrategy.FailurePolicy=Rollback 且 helm upgrade 失败时,
//    若 Strategy.Atomic=false (Helm SDK 未自动回滚), 系统调用 i.rollback() 执行 helm rollback
//    若 Strategy.Atomic=true, Helm SDK 已在 upgrade 命令内部自动回滚, 无需额外调用
type HelmStrategySpec struct {
    // 安装模式: Install / Upgrade / Rollback (空=自动判定)
    Mode string `json:"mode,omitempty"`
    
    // 是否等待就绪 (对应 helm --wait)
    Wait bool `json:"wait,omitempty"`
    
    // 等待超时时间 (对应 helm --timeout)
    WaitTimeout string `json:"waitTimeout,omitempty"`
    
    // 是否原子操作 (对应 helm --atomic, 失败时 Helm SDK 自动回滚)
    Atomic bool `json:"atomic,omitempty"`
    
    // 失败时是否清理 (对应 helm --cleanup-on-fail, 配合 Atomic 使用)
    CleanupOnFail bool `json:"cleanupOnFail,omitempty"`
}

// 健康检查相关类型定义见第 6 章 HealthCheck 共享包设计 (6.1 节)
// HealthCheckSpec / HealthCheckItemSpec / PodReadyCheckSpec /
// EndpointReadyCheckSpec / CustomCheckSpec
// HelmSpec 通过引用 pkg/healthcheck 包中的类型使用

// RollbackSpec 定义回滚规格
type RollbackSpec struct {
    // 是否启用回滚
    Enabled bool `json:"enabled"`
    
    // 保留历史版本数
    MaxHistory int `json:"maxHistory,omitempty"`
}

// UninstallSpec 定义卸载规格
type UninstallSpec struct {
    // 卸载前钩子
    PreUninstallHooks []HookSpec `json:"preUninstallHooks,omitempty"`
}

// HookSpec 定义钩子规格
type HookSpec struct {
    // 钩子名称
    Name string `json:"name,omitempty"`
    
    // 钩子类型: Job
    Type string `json:"type"`
    
    // 钩子 Manifest
    Manifest string `json:"manifest"`
}
```

### 3.4 Inline 类型字段定义

```go
// InlineSpec 定义内联执行器配置 ✅复用现有 (api/v1alpha1 已定义)
// Inline 组件通过 ComponentFactory 注册的 handler 执行, 无需制品下载/模板渲染
// handler 名称对应 ComponentFactory.Register() 注册的 key
type InlineSpec struct {
    // Handler 名称 (对应 ComponentFactory 注册的 handler)
    Handler string `json:"handler"`
    
    // Handler 版本
    Version string `json:"version"`
}
```

### 3.5 CRD YAML 定义

```yaml
# config/crd/bases/config.openfuyao.cn_componentversions.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: componentversions.config.openfuyao.cn
spec:
  group: config.openfuyao.cn
  names:
    kind: ComponentVersion
    listKind: ComponentVersionList
    plural: componentversions
    singular: componentversion
    shortNames:
      - cv
  scope: Namespaced
  versions:
    # v1alpha1: 唯一版本, 直接扩展 (storage=true)
    # 包含 helm/yaml 字段定义, 所有新字段均为 omitempty, 向后兼容
    - name: v1alpha1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            apiVersion:
              type: string
            kind:
              type: string
            metadata:
              type: object
            spec:
              type: object
              properties:
                name:
                  type: string
                type:
                  type: string
                  enum: [yaml, helm, inline]
                version:
                  type: string
                helm:
                  type: object
                  properties:
                    chart:
                      type: object
                      properties:
                        oci:
                          type: object
                          properties:
                            repository:
                              type: string
                            tag:
                              type: string
                          required: [repository, tag]
                        url:
                          type: string
                        localPath:
                          type: string
                        checksum:
                          type: string
                    namespace:
                      type: string
                    releaseName:
                      type: string
                    values:
                      type: object
                      x-kubernetes-preserve-unknown-fields: true
                    valuesFiles:
                      type: array
                      items:
                        type: string
                    strategy:
                      type: object
                      properties:
                        mode:
                          type: string
                        wait:
                          type: boolean
                        waitTimeout:
                          type: string
                        atomic:
                          type: boolean
                        cleanupOnFail:
                          type: boolean
                    healthCheck:
                      type: object
                      properties:
                        enabled:
                          type: boolean
                        timeout:
                          type: string
                        interval:
                          type: string
                        checks:
                          type: array
                          items:
                            type: object
                            properties:
                              type:
                                type: string
                              podReady:
                                type: object
                                properties:
                                  namespace:
                                    type: string
                                  labelSelector:
                                    type: string
                                  minReady:
                                    type: integer
                                required: [namespace, labelSelector]
                              endpointReady:
                                type: object
                                properties:
                                  namespace:
                                    type: string
                                  serviceName:
                                    type: string
                                  port:
                                    type: integer
                                required: [namespace, serviceName]
                              custom:
                                type: object
                                properties:
                                  command:
                                    type: string
                                required: [command]
                            required: [type]
                    rollback:
                      type: object
                      properties:
                        enabled:
                          type: boolean
                        maxHistory:
                          type: integer
                    uninstall:
                      type: object
                      properties:
                        preUninstallHooks:
                          type: array
                          items:
                            type: object
                            properties:
                              name:
                                type: string
                              type:
                                type: string
                              manifest:
                                type: string
                            required: [type, manifest]
                    preInstallHooks:
                      type: array
                      items:
                        type: object
                        properties:
                          name:
                            type: string
                          type:
                            type: string
                          manifest:
                            type: string
                        required: [type, manifest]
                    preUninstallHooks:
                      type: array
                      items:
                        type: object
                        properties:
                          name:
                            type: string
                          type:
                            type: string
                          manifest:
                            type: string
                        required: [type, manifest]
                  required: [chart, namespace, releaseName]
                yaml:
                  type: object
                  properties:
                    namespace:
                      type: string
                    applyStrategy:
                      type: string
                      enum: [ServerSideApply, Replace, CreateOnly]
                    prune:
                      type: boolean
                    pruneLabelSelector:
                      type: object
                      additionalProperties:
                        type: string
                    healthCheck:
                      type: object
                      properties:
                        enabled:
                          type: boolean
                        timeout:
                          type: string
                        interval:
                          type: string
                        checks:
                          type: array
                          items:
                            type: object
                            properties:
                              type:
                                type: string
                              podReady:
                                type: object
                                properties:
                                  namespace:
                                    type: string
                                  labelSelector:
                                    type: string
                                  minReady:
                                    type: integer
                                required: [namespace, labelSelector]
                              endpointReady:
                                type: object
                                properties:
                                  namespace:
                                    type: string
                                  serviceName:
                                    type: string
                                  port:
                                    type: integer
                                required: [namespace, serviceName]
                              custom:
                                type: object
                                properties:
                                  command:
                                    type: string
                                required: [command]
                            required: [type]
                      required: [enabled]
                inline:
                  type: object
                  properties:
                    handler:
                      type: string
                    version:
                      type: string
                  required: [handler, version]
                subComponents:
                  type: array
                  items:
                    type: object
                    properties:
                      name:
                        type: string
                      version:
                        type: string
                    required: [name, version]
                compatibility:
                  type: object
                  properties:
                    constraints:
                      type: array
                      items:
                        type: object
                        properties:
                          component:
                            type: string
                          rule:
                            type: string
                        required: [component, rule]
                dependencies:
                  type: array
                  items:
                    type: object
                    properties:
                      name:
                        type: string
                      phase:
                        type: string
                    required: [name]
                upgradeStrategy:
                  type: object
                  properties:
                    mode:
                      type: string
                    batchSize:
                      type: integer
                    timeout:
                      type: string
                    failurePolicy:
                      type: string
                resources:
                  type: array
                  items:
                    type: object
                    properties:
                      kind:
                        type: string
                      apiVersion:
                        type: string
                      namespace:
                        type: string
                      name:
                        type: string
                      labels:
                        type: object
                        additionalProperties:
                          type: string
                      data:
                        type: object
                        additionalProperties:
                          type: string
                      stringData:
                        type: object
                        additionalProperties:
                          type: string
                      manifest:
                        type: string
                    required: [kind, apiVersion, name]
              required: [name, type, version]
            status:
              type: object
              properties:
                phase:
                  type: string
                conditions:
                  type: array
                  items:
                    type: object
                    properties:
                      type:
                        type: string
                      status:
                        type: string
                        enum: ["True", "False", Unknown]
                      lastTransitionTime:
                        type: string
                        format: date-time
                      reason:
                        type: string
                      message:
                        type: string
                    required: [type, status, lastTransitionTime, reason, message]
      subresources:
        status: {}
```

### 3.6 CRD 版本迁移设计

**设计思路 - 直接扩展 v1alpha1，不引入 v1alpha2**：

原设计拟新增 `api/v1alpha2/` + conversion 函数。但经审视，现有 `api/v1alpha1/componentversion_types.go` 的 `Type` 字段已支持 `yaml/helm/inline` 三值（仅 enum，无 schema 定义），且所有新字段（`Helm`/`YAML`）均为 `omitempty` 指针类型，向旧数据完全兼容。引入 v1alpha2 会带来：① 双版本 conversion 维护成本；② `(*v1alpha2.InlineSpec)(src.Spec.Inline)` 等跨包指针强转的脆弱性；③ 现有引用 v1alpha1 的代码全部需评估迁移。

因此**直接在 v1alpha1 上扩展**：新增 `Helm *HelmSpec`/`YAML *YAMLSpec` 两个 omitempty 字段及对应 CRD schema。旧 ComponentVersion（无这两个字段）反序列化后新字段为 nil，行为不变；新 ComponentVersion 带新字段，旧控制器忽略（omitempty）。无需 conversion，零迁移风险，最大化复用现有类型文件与 deepcopy 生成代码。

**扩展内容**：

| 改动位置 | 内容 |
| --------- | ------ |
| `api/v1alpha1/componentversion_types.go` | 新增 `HelmSpec`/`YAMLSpec`/`ChartSpec`/`OCIChartSpec`/`HelmStrategySpec`/`RollbackSpec`/`UninstallSpec`/`HookSpec` 等类型；`ComponentVersionSpec` 增加 `Helm *HelmSpec`/`YAML *YAMLSpec` 字段 |
| `api/v1alpha1/zz_generated.deepcopy.go` | 重新 `make` 生成 DeepCopy 方法 |
| `config/crd/bases/...componentversions.yaml` | v1alpha1 schema 新增 helm/yaml 字段定义（见 3.5 节） |

**兼容性保证**：

- 新字段全部 `omitempty` + 指针类型，旧 YAML 不填则为 nil
- `Type` 字段 enum 已含 `helm`/`yaml`，无需改 enum
- 旧控制器代码不读取新字段，不受影响
- Feature Gate 关闭时即使新字段存在也不走新路径（见 11.2）

**迁移步骤（简化）**：

| 步骤 | 操作 | 风险 | 回滚方案 |
| ------ | ------ | ------ | --------- |
| 1 | 在 `api/v1alpha1/componentversion_types.go` 新增 HelmSpec/YAMLSpec 及子类型 | 无 | 删除新增类型 |
| 2 | `ComponentVersionSpec` 增加 `Helm/YAML` 字段 + 重新生成 deepcopy | 低 | 删除字段 |
| 3 | CRD schema 合并 helm/yaml 定义到 v1alpha1 版本 | 低 | 还原 schema |
| 4 | 控制器按 Feature Gate 读取新字段 | 中 | 关闭 Feature Gate |

**注意事项**：

- 若未来字段规模膨胀确需 v1alpha2，再按标准 conversion 流程引入，此时 v1alpha1 已是稳定存储版本

## 4. HelmInstaller 详细设计

### 4.1 核心组件架构

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             HelmInstaller                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         核心组件                                         │    │
│  │                                                                         │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │                        ChartFetcher                             │    │    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐    │    │    │
│  │  │  │  OCI Client  │  │  HTTP Client │  │  Local Loader       │    │    │    │
│  │  │  │  (OCI拉取)   │  │  (HTTP下载)   │  │  (本地加载)          │   │     │    │
│  │  │  └──────────────┘  └──────────────┘  └─────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                         │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │                       ValuesRenderer                            │    │    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐    │    │    │
│  │  │  │  Template    │  │  Values      │  │  Merge Strategy     │    │    │    │
│  │  │  │  Resolver    │  │  File Loader │  │  (合并策略)          │    │    │    │
│  │  │  └──────────────┘  └──────────────┘  └─────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                         │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │                    Helm Action Executor                         │    │    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐    │    │    │
│  │  │  │  Install     │  │  Upgrade     │  │  Rollback           │    │    │    │
│  │  │  │  Action      │  │  Action      │  │  Action             │    │    │    │
│  │  │  └──────────────┘  └──────────────┘  └─────────────────────┘    │    │    │
│  │  │  ┌──────────────┐  ┌──────────────┐                             │    │    │
│  │  │  │  Uninstall   │  │  Wait/Atomic │                             │    │    │
│  │  │  │  Action      │  │  Control     │                             │    │    │
│  │  │  └──────────────┘  └──────────────┘                             │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                         │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │                       HealthChecker                             │    │    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐    │    │    │
│  │  │  │  PodReady    │  │  Endpoint    │  │  Custom Check       │    │    │    │
│  │  │  │  Check       │  │  Ready Check │  │                     │    │    │    │
│  │  │  └──────────────┘  └──────────────┘  └─────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 HelmInstaller 执行流程图

**设计思路**：HelmInstaller 的执行流程分为 7 个主要步骤：获取 Chart、校验 Checksum、渲染 Values、加载自定义 Values、合并 Values、执行 Helm Action、健康检查。整个流程支持 OCI/HTTP/本地三种 Chart 来源，并提供原子操作和自动回滚能力。

**关键设计点**：

- **多来源支持**：OCI Registry、HTTP URL、本地路径三种 Chart 获取方式
- **Values 渲染**：支持模板变量替换，可动态生成配置
- **原子操作**：通过 `atomic: true` 配置，失败时自动回滚
- **健康检查**：安装/升级后执行 PodReady/EndpointReady 检查

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          HelmInstaller 执行流程                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   Install()      │
                              │   入口函数        │
                              └────────┬─────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  1. 获取 Chart                       │
                    │  getChart(ctx, helm.Chart)           │
                    └────────────────────┬─────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
          │  OCI Registry   │  │  HTTP URL       │  │  Local Path     │
          │  helm pull      │  │  http.Get()     │  │  os.Open()      │
          │  oci://...      │  │  https://...    │  │  /path/to/chart │
          └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                   │                    │                    │
                   └────────────────────┼────────────────────┘
                                        │
                                        ▼
                    ┌──────────────────────────────────────┐
                    │  2. 校验 Chart Checksum              │
                    │  verifyChecksum(chart, expected)     │
                    └────────────────────┬─────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                         校验通过                校验失败
                              │                     │
                              ▼                     ▼
                    ┌─────────────────┐   ┌─────────────────┐
                    │  继续           │   │  返回错误        │
                    └────────┬────────┘   └─────────────────┘
                             │
                             ▼
                    ┌──────────────────────────────────────┐
                    │  3. 渲染 Values                      │
                    │  renderValues(ctx, helm.Values, opts)│
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  4. 加载自定义 Values 文件            │
                    │  loadValuesFiles(ctx, helm.Values    │
                    │                   Files, opts)       │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  5. 合并 Values                      │
                    │  mergeValues(base, custom)           │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  6. 执行 Helm Action                 │
                    │  executeAction(ctx, actionConfig,    │
                    │                chart, values, helm)  │
                    └────────────────────┬─────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
          │  Install        │  │  Upgrade        │  │  Rollback       │
          │  action.New     │  │  action.New     │  │  action.New     │
          │  Install()      │  │  Upgrade()      │  │  Rollback()     │
          └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                   │                    │                    │
                   └────────────────────┼────────────────────┘
                                        │
                                        ▼
                    ┌──────────────────────────────────────┐
                    │  7. 执行健康检查                      │
                    │  runHealthCheck(ctx, helm.           │
                    │                HealthCheck, opts)    │
                    └────────────────────┬─────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
          │  PodReady       │  │  EndpointReady  │  │  Custom         │
          │  检查 Pod 状态   │  │  检查 Endpoint  │  │  自定义检查      │
          └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                   │                    │                    │
                   └────────────────────┼────────────────────┘
                                        │
                              ┌─────────┴───────────┐
                              │                     │
                         检查通过                检查失败
                              │                     │
                              ▼                     ▼
                    ┌─────────────────┐   ┌─────────────────┐
                    │  返回成功        │   │  返回错误       │
                    │  return nil     │   │  (触发回滚)      │
                    └─────────────────┘   └─────────────────┘
```

### 4.3 核心接口定义

**设计思路**：HelmInstaller 的接口设计复用现有的 `manifest.TemplateContext`。DAG 调度器构建的 TemplateContext 直接传递给 HelmInstaller，用于渲染 Helm Values。

```go
// pkg/helminstaller/installer.go

// HelmInstaller Helm 组件安装器
type HelmInstaller struct {
    client     client.Client
    clientset  kubernetes.Interface  // 目标集群 clientset，健康检查 (Pod/Endpoint) 用
                                      // 注意: client.Client (controller-runtime) 无 CoreV1() 方法，
                                      // 健康检查需 typed clientset，故单独注入
    restConfig *rest.Config
    cacheDir   string
    httpClient *http.Client
    chartAuth  *kube.AuthConfig      // 镜像仓库认证，传给 kube.FetchChartUniversal
    logger     *bkev1beta1.BKELogger
}

// HelmInstallerConfig HelmInstaller 构建配置
type HelmInstallerConfig struct {
    Client     client.Client
    Clientset  kubernetes.Interface  // 目标集群 typed clientset
    RestConfig *rest.Config
    CacheDir   string
    HttpClient *http.Client
    ChartAuth  *kube.AuthConfig      // 复用 pkg/kube.AuthConfig
    Logger     *bkev1beta1.BKELogger
}

// NewHelmInstaller 创建 Helm 组件安装器
func NewHelmInstaller(cfg HelmInstallerConfig) *HelmInstaller {
    return &HelmInstaller{
        client:     cfg.Client,
        clientset:  cfg.Clientset,
        restConfig: cfg.RestConfig,
        cacheDir:   cfg.CacheDir,
        httpClient: cfg.HttpClient,
        chartAuth:  cfg.ChartAuth,
        logger:     cfg.Logger,
    }
}

// InstallOptions Helm 安装选项
type InstallOptions struct {
    Component   *ComponentVersion
    TemplateCtx manifest.TemplateContext  // 复用现有 TemplateContext
    Action      HelmAction  // Install / Upgrade / Uninstall / Rollback
    Timeout     time.Duration
    DryRun      bool
}

// HelmAction Helm 操作类型
type HelmAction string

const (
    HelmActionInstall   HelmAction = "Install"
    HelmActionUpgrade   HelmAction = "Upgrade"
    HelmActionUninstall HelmAction = "Uninstall"
    HelmActionRollback  HelmAction = "Rollback"
)

// Install 执行 Helm 组件安装/升级
func (i *HelmInstaller) Install(ctx context.Context, opts InstallOptions) error {
    component := opts.Component
    helm := component.Spec.Helm
    tmplCtx := opts.TemplateCtx  // 复用 DAG 调度器传递的 TemplateContext
    
    // 1. 渲染 Chart 来源字段中的模板变量
    //    支持 {{imageRegistry}}、{{componentVersion}} 等变量在 Repository/Tag/URL 中
    renderedChart, err := i.renderChartSpec(ctx, helm.Chart, tmplCtx)
    if err != nil {
        return fmt.Errorf("failed to render chart spec: %w", err)
    }
    
    // 2. 获取 Chart (使用渲染后的来源)
    chart, err := i.getChart(ctx, renderedChart)
    if err != nil {
        return fmt.Errorf("failed to get chart: %w", err)
    }
    
    // 3. 渲染 Values (使用 TemplateContext)
    values, err := i.renderValues(ctx, helm.Values, tmplCtx)
    if err != nil {
        return fmt.Errorf("failed to render values: %w", err)
    }
    
    // 4. 加载自定义 Values 文件
    for _, vf := range helm.ValuesFiles {
        customValues, err := i.loadValuesFile(ctx, vf, tmplCtx)
        if err != nil {
            log.Warn("failed to load values file %s: %v", vf, err)
            continue
        }
        values = mergeValues(values, customValues)
    }
    
    // 5. 获取 Action Config
    actionConfig, err := i.getActionConfig(ctx, helm.Namespace)
    if err != nil {
        return fmt.Errorf("failed to get action config: %w", err)
    }
    
    // 6. 执行 Helm Action
    switch opts.Action {
    case HelmActionInstall:
        return i.install(ctx, actionConfig, chart, values, helm, opts)
    case HelmActionUpgrade:
        return i.upgrade(ctx, actionConfig, chart, values, helm, opts)
    case HelmActionUninstall:
        return i.uninstall(ctx, actionConfig, helm, opts)
    case HelmActionRollback:
        return i.rollback(ctx, actionConfig, helm, opts)
    }
    
    return nil
}

// getChart 获取 Chart
func (i *HelmInstaller) getChart(ctx context.Context, chartSpec ChartSpec) (*chart.Chart, error) {
    if chartSpec.OCI != nil {
        return i.getChartFromOCI(ctx, chartSpec.OCI)
    }
    if chartSpec.URL != "" {
        return i.getChartFromURL(ctx, chartSpec.URL)
    }
    if chartSpec.LocalPath != "" {
        return i.getChartFromLocal(ctx, chartSpec.LocalPath)
    }
    return nil, errors.New("no chart source specified")
}

// install 执行 Helm Install
func (i *HelmInstaller) install(ctx context.Context, actionConfig *action.Configuration, 
    chart *chart.Chart, values map[string]interface{}, helm *HelmSpec, opts InstallOptions) error {
    
    client := action.NewInstall(actionConfig)
    client.ReleaseName = helm.ReleaseName
    client.Namespace = helm.Namespace
    client.CreateNamespace = true
    client.Wait = helm.Strategy.Wait
    client.Timeout, _ = time.ParseDuration(helm.Strategy.WaitTimeout)
    client.Atomic = helm.Strategy.Atomic
    client.DryRun = opts.DryRun
    
    release, err := client.Run(chart, values)
    if err != nil {
        if helm.Strategy.CleanupOnFail && release != nil {
            uninstallClient := action.NewUninstall(actionConfig)
            uninstallClient.Run(release.Name)
        }
        return fmt.Errorf("helm install failed: %w", err)
    }
    
    // 执行健康检查
    if helm.HealthCheck.Enabled {
        if err := i.runHealthCheck(ctx, helm.HealthCheck, opts); err != nil {
            return fmt.Errorf("health check failed after install: %w", err)
        }
    }
    
    return nil
}

// upgrade 执行 Helm Upgrade
func (i *HelmInstaller) upgrade(ctx context.Context, actionConfig *action.Configuration,
    chart *chart.Chart, values map[string]interface{}, helm *HelmSpec, opts InstallOptions) error {
    
    client := action.NewUpgrade(actionConfig)
    client.Namespace = helm.Namespace
    client.Wait = helm.Strategy.Wait
    client.Timeout, _ = time.ParseDuration(helm.Strategy.WaitTimeout)
    client.Atomic = helm.Strategy.Atomic
    client.MaxHistory = helm.Rollback.MaxHistory
    client.DryRun = opts.DryRun
    
    release, err := client.Run(helm.ReleaseName, chart, values)
    if err != nil {
        if helm.Strategy.CleanupOnFail && release != nil {
            uninstallClient := action.NewUninstall(actionConfig)
            uninstallClient.Run(release.Name)
        }
        return fmt.Errorf("helm upgrade failed: %w", err)
    }
    
    // 执行健康检查
    if helm.HealthCheck.Enabled {
        if err := i.runHealthCheck(ctx, helm.HealthCheck, opts); err != nil {
            return fmt.Errorf("health check failed after upgrade: %w", err)
        }
    }
    
    return nil
}
```

**设计思路 — Helm `--wait` 与自定义 `healthCheck` 的关系**：

两者**同时生效**，是两层递进检查，非互斥关系：

| 层级 | 机制 | 触发时机 | 检查内容 | 局限 |
| ------ | ------ | --------- | --------- | ------ |
| **第一层: Helm `--wait`** | Helm SDK 原生 (`strategy.wait: true`) | `helm install/upgrade` 命令内部 | Pod Ready 状态（全部 Pod） | 仅检查 Pod Ready，不检查 Endpoint；要求全部 Ready 才通过 |
| **第二层: 自定义 `healthCheck`** | Installer 实现 (`healthCheck.enabled: true`) | Helm 命令返回后 | PodReady + EndpointReady + Custom | 支持 `minReady` 部分就绪；支持 Endpoint 检查；支持自定义命令 |

**为什么需要两层**：

1. **Helm `--wait` 的局限**：仅检查 Pod `Ready` condition，不验证 Service Endpoint 是否就绪（Pod Ready 但 Endpoint 未注册的情况存在）；要求全部 Pod Ready，不支持部分就绪场景。
2. **自定义 `healthCheck` 的补充**：支持 `minReady`（3 副本中 2 个就绪即可通过）、`EndpointReady`（验证服务可达性）、`Custom`（执行自定义检查命令）。
3. **执行顺序**：Helm `--wait` 先执行（在 `helm install/upgrade` 命令内部），通过后才执行自定义 `healthCheck`。如果 `--wait` 失败且 `atomic: true`，Helm 自动回滚，不会执行自定义 `healthCheck`。
4. **配置建议**：如果 `strategy.wait: true` 已启用，`healthCheck` 可仅配置 `EndpointReady` 或 `Custom` 检查（`PodReady` 已由 Helm `--wait` 覆盖），避免重复检查。

```go
// runHealthCheck 执行 Helm 安装后的自定义健康检查
// 与 Helm --wait 的关系: 两者同时生效, --wait 先执行 (Helm 命令内部),
// 自定义 healthCheck 后执行 (Helm 命令返回后), 互补非互斥
//
// 实现委托给共享 pkg/healthcheck 包 (见 6)，避免与 YamlInstaller 重复实现
// PodReady/EndpointReady/Custom 检查逻辑。
func (i *HelmInstaller) runHealthCheck(
    ctx context.Context,
    hc HealthCheckSpec,
    opts InstallOptions,
) error {
    // 委托共享包: 第 6 章 HealthCheck 共享包设计
    // HealthCheckSpec/PodReadyCheckSpec/EndpointReadyCheckSpec/CustomCheckSpec
    // 类型定义迁移到 pkg/healthcheck/types.go，Helm/YAML 共用
    return healthcheck.Run(ctx, i.clientset, hc)
}
```

```go
// getActionConfig 初始化 Helm Action 配置 (每次调用创建新实例, 避免状态残留)
func (i *HelmInstaller) getActionConfig(ctx context.Context, namespace string) (*action.Configuration, error) {
    actionConfig := new(action.Configuration)
    if err := actionConfig.Init(i.restConfig, namespace, "secret", func(format string, v ...interface{}) {
        // Helm 日志输出到标准日志
    }); err != nil {
        return nil, fmt.Errorf("failed to init helm action config: %w", err)
    }
    return actionConfig, nil
}

// uninstall 执行 Helm Uninstall
func (i *HelmInstaller) uninstall(ctx context.Context, actionConfig *action.Configuration,
    helm *HelmSpec, opts InstallOptions) error {
    client := action.NewUninstall(actionConfig)
    client.Wait = helm.Strategy.Wait
    client.Timeout, _ = time.ParseDuration(helm.Strategy.WaitTimeout)
    _, err := client.Run(helm.ReleaseName)
    if err != nil {
        return fmt.Errorf("helm uninstall failed: %w", err)
    }
    return nil
}

// rollback 执行 Helm Rollback
func (i *HelmInstaller) rollback(ctx context.Context, actionConfig *action.Configuration,
    helm *HelmSpec, opts InstallOptions) error {
    client := action.NewRollback(actionConfig)
    client.Wait = helm.Strategy.Wait
    client.Timeout, _ = time.ParseDuration(helm.Strategy.WaitTimeout)
    if err := client.Run(helm.ReleaseName); err != nil {
        return fmt.Errorf("helm rollback failed: %w", err)
    }
    return nil
}

// 复用现有 pkg/kube/chart.go 的 Chart 拉取能力，避免重复实现 OCI/HTTP/本地加载。
//
// 现有可复用函数 (pkg/kube/chart.go):
//   - FetchChartUniversal(chartRepo, chartName, version, auth, logger) (*chart.Chart, error)
//     支持 OCI Registry 与传统 HTTP 仓库，自动识别 oci:// 前缀 (chart.go:442-465)
//   - FetchChartOCI(...) (chart.go:525-582)
//   - FetchChartTraditional(...) (chart.go:688-728)
//   - initActionConfig(namespace) (*action.Configuration, error) (chart.go:113-121)
//   - releaseExists(actionConfig, releaseName, ns) (bool, error) (chart.go:123-136)
//
// HelmInstaller 不再直接调用 ocipuller/loader，而是通过 kube.FetchChartUniversal 拉取。
// 这与现有 addon helm 部署链路 (pkg/kube/chart.go installChartAddon) 保持一致，
// 后续可将 chart.go 中通用部分进一步抽取到 pkg/helminstaller 共享。

// getChartFromOCI 从 OCI Registry 拉取 Chart (复用 kube.FetchChartOCI)
func (i *HelmInstaller) getChartFromOCI(ctx context.Context, ociSpec *OCIChartSpec) (*chart.Chart, error) {
    // 复用 pkg/kube/chart.go: FetchChartOCI(repository, tag, auth, logger)
    // auth 来自集群镜像仓库认证配置 (由 HelmInstaller.chartAuth 提供)
    return kube.FetchChartOCI(ociSpec.Repository, ociSpec.Tag, i.chartAuth, i.logger.NormalLogger)
}

// getChartFromURL 从 HTTP URL 下载 Chart (复用 kube.FetchChartTraditional)
func (i *HelmInstaller) getChartFromURL(ctx context.Context, url string) (*chart.Chart, error) {
    // 复用 pkg/kube/chart.go: FetchChartTraditional(repo, name, version, auth, logger)
    // url 形如 https://repo/charts/name-version.tgz，解析出 repo/name/version 后调用
    repo, name, version := parseChartURL(url)
    return kube.FetchChartTraditional(repo, name, version, i.chartAuth, i.logger.NormalLogger)
}

// getChartFromLocal 从本地路径加载 Chart (复用 helm loader.Load)
func (i *HelmInstaller) getChartFromLocal(ctx context.Context, path string) (*chart.Chart, error) {
    ch, err := loader.Load(path)
    if err != nil {
        return nil, fmt.Errorf("failed to load chart from %s: %w", path, err)
    }
    return ch, nil
}
```

> 说明：原 `listPods`/`checkPodReady`/`checkEndpointReady`/`checkCustom`/`runHealthCheck` 的实现逻辑已移至 **第 6 章 HealthCheck 共享包设计**（参数化为 `kubernetes.Interface`，Helm/YAML 共用）。HelmInstaller 仅保留 `runHealthCheck` 一行委托（见上方）。

`HelmSpec` 提供两种 Values 来源，合并优先级从低到高：

```txt
Chart 默认值 → ValuesFiles[0] → ValuesFiles[1] → ... → Values 内联字段 (最高优先级)
```

1. **Values（内联）**：`map[string]interface{}`，直接在 CRD YAML 中定义，支持模板变量渲染（如 `{{componentVersion}}`）。优先级最高，覆盖所有文件值。
2. **ValuesFiles（外部文件）**：`[]string`，文件路径列表，路径支持模板变量（如 `values-{{arch}}.yaml`）。从控制器 Pod 本地文件系统读取（通常通过 ConfigMap 挂载）。逐个加载后按顺序合并，后者覆盖前者。
3. **mergeValues 合并规则**：递归合并 map（src 覆盖 dst 同名键），非 map 类型（list/string/int）整体替换而非追加。与 Helm 自身 `--set` 和 `-f` 合并行为一致。
4. **文件加载失败为警告而非错误**：ValuesFiles 是可选补充配置，文件可能按条件存在（如 `values-{{arch}}.yaml` 在某些架构上不存在），缺失时使用内联 Values 即可。

**设计思路 — 本地文件系统 vs ConfigMap/URL**：

当前从控制器 Pod 本地文件系统读取 Values 文件。在 Kubernetes 控制器场景下，文件通常通过 ConfigMap 或 Secret 挂载到 Pod 中。选择本地文件而非直接读取 ConfigMap API 的原因是：

- Values 文件可能很大（超过 etcd 1MB 限制），ConfigMap 不适合存储大型 YAML
- 文件路径支持模板变量渲染，动态选择不同文件（如 `values-{{arch}}.yaml`）
- 后续可扩展支持 HTTP URL 和 ConfigMap 引用作为 Values 来源

```go
// renderValues 渲染 Values 中的模板变量
// 将 helm.Values (map[string]interface{}) 中的字符串值进行模板变量替换
// 支持 {{componentVersion}}, {{clusterName}}, {{imageRegistry}} 等变量
func (i *HelmInstaller) renderValues(
    ctx context.Context,
    values map[string]interface{},
    tmplCtx manifest.TemplateContext,
) (map[string]interface{}, error) {
    rendered := make(map[string]interface{})
    for key, val := range values {
        switch v := val.(type) {
        case string:
            // 字符串值: 渲染模板变量
            renderedVal, err := i.renderTemplateString(v, tmplCtx)
            if err != nil {
                return nil, fmt.Errorf("failed to render value %q: %w", key, err)
            }
            rendered[key] = renderedVal
        case map[string]interface{}:
            // 嵌套 map: 递归渲染
            nested, err := i.renderValues(ctx, v, tmplCtx)
            if err != nil {
                return nil, err
            }
            rendered[key] = nested
        default:
            // 非字符串类型 (int/bool/float/list): 原样保留
            rendered[key] = val
        }
    }
    return rendered, nil
}

// renderTemplateString 渲染字符串中的模板变量
// 支持 {{componentVersion}} 简单替换 和 {{.clusterName}} Go template 两种语法
func (i *HelmInstaller) renderTemplateString(s string, tmplCtx manifest.TemplateContext) (string, error) {
    if !strings.Contains(s, "{{") {
        return s, nil // 无模板变量, 直接返回
    }
    // 简单变量替换: {{componentVersion}} → tmplCtx.ComponentVersion
    result := s
    result = strings.ReplaceAll(result, "{{componentVersion}}", tmplCtx.ComponentVersion)
    result = strings.ReplaceAll(result, "{{clusterName}}", tmplCtx.ClusterName)
    result = strings.ReplaceAll(result, "{{namespace}}", tmplCtx.Namespace)
    result = strings.ReplaceAll(result, "{{kubernetesVersion}}", tmplCtx.KubernetesVersion)
    result = strings.ReplaceAll(result, "{{imageRegistry}}", tmplCtx.ImageRegistry)
    return result, nil
}

// renderChartSpec 渲染 ChartSpec 中的模板变量
// 支持 OCI.Repository、OCI.Tag、URL 字段使用模板变量
// 返回深拷贝，不修改原始 ChartSpec
func (i *HelmInstaller) renderChartSpec(
    ctx context.Context,
    spec ChartSpec,
    tmplCtx manifest.TemplateContext,
) (ChartSpec, error) {
    rendered := spec // 深拷贝
    
    // 渲染 OCI 配置
    if spec.OCI != nil {
        renderedOCI := *spec.OCI // 深拷贝
        
        // 渲染 Repository
        if spec.OCI.Repository != "" {
            repo, err := i.renderTemplateString(spec.OCI.Repository, tmplCtx)
            if err != nil {
                return rendered, fmt.Errorf("render OCI repository %q: %w", spec.OCI.Repository, err)
            }
            renderedOCI.Repository = repo
        }
        
        // 渲染 Tag
        if spec.OCI.Tag != "" {
            tag, err := i.renderTemplateString(spec.OCI.Tag, tmplCtx)
            if err != nil {
                return rendered, fmt.Errorf("render OCI tag %q: %w", spec.OCI.Tag, err)
            }
            renderedOCI.Tag = tag
        }
        
        rendered.OCI = &renderedOCI
    }
    
    // 渲染 URL
    if spec.URL != "" {
        url, err := i.renderTemplateString(spec.URL, tmplCtx)
        if err != nil {
            return rendered, fmt.Errorf("render URL %q: %w", spec.URL, err)
        }
        rendered.URL = url
    }
    
    return rendered, nil
}

// renderChartSpec 设计说明:
// Chart 来源字段 (OCI.Repository、OCI.Tag、URL) 支持模板变量的原因:
// 1. 镜像仓库地址因环境而异: 在线环境使用公共仓库，离线环境使用私有仓库
//    例如: "{{imageRegistry}}/charts/coredns" 在不同环境渲染为不同地址
// 2. Chart 版本与组件版本一致: tag 通常等于 componentVersion
//    例如: tag: "{{componentVersion}}" 避免硬编码版本号
// 3. 统一配置管理: 与 Values 模板变量保持一致的渲染机制
// 4. 渲染时机: 在 getChart 之前渲染，确保拉取正确的 Chart 来源

// loadValuesFile 加载自定义 Values 文件
// 1. 渲染文件路径中的模板变量 (如 values-{{arch}}.yaml)
// 2. 从控制器本地文件系统读取文件 (通常通过 ConfigMap 挂载)
// 3. 解析 YAML 为 map[string]interface{}
func (i *HelmInstaller) loadValuesFile(
    ctx context.Context,
    valuesFile string,
    tmplCtx manifest.TemplateContext,
) (map[string]interface{}, error) {
    // 1. 渲染文件名中的模板变量
    renderedFile, err := i.renderTemplateString(valuesFile, tmplCtx)
    if err != nil {
        return nil, fmt.Errorf("failed to render values file path %s: %w", valuesFile, err)
    }

    // 2. 读取文件内容
    data, err := os.ReadFile(renderedFile)
    if err != nil {
        return nil, fmt.Errorf("failed to read values file %s: %w", renderedFile, err)
    }

    // 3. 解析 YAML
    values := make(map[string]interface{})
    if err := yaml.Unmarshal(data, &values); err != nil {
        return nil, fmt.Errorf("failed to parse values file %s: %w", renderedFile, err)
    }

    return values, nil
}

// mergeValues 合并两份 Values (dst 被 src 覆盖)
// 策略: 递归合并 map, src 中的值覆盖 dst 中的同名键
// 非 map 类型 (list/string/int/bool) 整体替换, 不追加
// 与 Helm 自身 --set 和 -f 合并行为一致
func mergeValues(dst, src map[string]interface{}) map[string]interface{} {
    result := make(map[string]interface{})
    for k, v := range dst {
        result[k] = v
    }
    for k, v := range src {
        if dstMap, ok := dst[k].(map[string]interface{}); ok {
            if srcMap, ok := v.(map[string]interface{}); ok {
                // 两边都是 map: 递归合并
                result[k] = mergeValues(dstMap, srcMap)
                continue
            }
        }
        // src 覆盖 dst (非 map 类型整体替换, list 不追加)
        result[k] = v
    }
    return result
}
```

#### 4.3.1 Values / ValuesFiles 使用样例

**场景 1**：values + valuesFiles 同时使用（按架构区分）

```yaml
# bke-manifests/coredns/v1.11.1/component.yaml
spec:
  name: coredns
  type: helm
  version: v1.11.1
  helm:
    chart:
      oci:
        repository: "{{imageRegistry}}/charts/coredns"   # 支持模板变量
        tag: "{{componentVersion}}"                       # 支持模板变量
    namespace: kube-system
    releaseName: coredns
    
    # 内联 Values (优先级最高, 覆盖文件中的同名键)
    values:
      image:
        repository: "registry.openfuyao.cn/coredns/coredns"
        tag: "{{componentVersion}}"
      replicaCount: 2                        # 覆盖 values-base.yaml 中的 replicaCount: 1
    
    # 外部 Values 文件 (按顺序合并, 后者覆盖前者)
    valuesFiles:
      - "/etc/bke-values/coredns/values-base.yaml"       # 基础配置
      - "/etc/bke-values/coredns/values-{{arch}}.yaml"   # 按架构区分
```

**控制器 Pod 中的文件结构**（通常通过 ConfigMap 挂载）：

```txt
/etc/bke-values/coredns/
├── values-base.yaml          # 基础配置 (所有架构共享)
├── values-amd64.yaml         # amd64 架构特定配置
└── values-arm64.yaml         # arm64 架构特定配置
```

**合并结果**（优先级：Chart 默认 < values-base.yaml < values-amd64.yaml < 内联 values）：

```yaml
image:
  repository: "registry.openfuyao.cn/coredns/coredns"  # 内联 values 覆盖 amd64 文件
  tag: "v1.11.1"                                        # 内联 values (模板渲染后)
  pullPolicy: IfNotPresent                              # 来自 values-base.yaml
resources:
  limits:
    cpu: "100m"                                         # 来自 values-base.yaml
    memory: "128Mi"                                     # 来自 values-base.yaml
  requests:
    cpu: "50m"                                          # 来自 values-base.yaml
    memory: "64Mi"                                      # 来自 values-base.yaml
replicaCount: 2                                          # 内联 values 覆盖 base 的 1
service:
  type: ClusterIP                                        # 来自 values-base.yaml
nodeSelector:
  kubernetes.io/arch: amd64                              # 来自 values-amd64.yaml
```

**场景 2**：仅 values 内联（简单场景）

```yaml
helm:
  values:
    image:
      repository: "registry.openfuyao.cn/coredns/coredns"
      tag: "{{componentVersion}}"
    replicaCount: 2
    resources:
      limits:
        cpu: "100m"
        memory: "128Mi"
```

无需外部文件，所有配置内联在 CRD 中。适用于配置简单、无需按环境区分的场景。

### 4.4 健康检查

HelmInstaller 的健康检查通过 `runHealthCheck` 方法委托共享 `pkg/healthcheck` 包执行（`return healthcheck.Run(ctx, i.clientset, hc)`）。PodReady/EndpointReady/Custom 三种检查的接口定义与实现、与 Helm `--wait` 的关系、类型归属等详见 **第 6 章 HealthCheck 共享包设计**。

### 4.5 Hooks 执行引擎

**设计思路 — PreInstallHooks 与 PreUninstallHooks 统一设计**：

Helm 组件支持两种钩子：`PreInstallHooks`（安装/升级前执行）和 `PreUninstallHooks`（卸载前执行）。两者共享相同的执行逻辑（创建 Job → 等待完成 → 清理），仅触发时机不同。统一设计为 `HookExecutor` 接口，避免重复实现。

**钩子执行流程**：

```txt
Hooks 执行流程:
┌─────────────────────────────────────────────────────────────┐
│ 1. 遍历 hooks 列表                                           │
│    - 过滤支持的 hook.Type (目前仅 "Job")                     │
├─────────────────────────────────────────────────────────────┤
│ 2. 渲染 Hook Manifest                                       │
│    - 支持模板变量 ({{clusterName}}, {{namespace}} 等)        │
├─────────────────────────────────────────────────────────────┤
│ 3. 创建 Job 资源                                            │
│    - Job 名称: pre-install-<hookName>-<timestamp>           │
│    - 或: pre-uninstall-<hookName>-<timestamp>               │
├─────────────────────────────────────────────────────────────┤
│ 4. 等待 Job 完成                                            │
│    - 超时: 5 分钟 (可配置)                                   │
│    - 失败处理: 清理 Job 后返回错误                            │
├─────────────────────────────────────────────────────────────┤
│ 5. 清理 Job 资源                                            │
│    - 删除 Job (保留 Pod 日志用于调试)                         │
└─────────────────────────────────────────────────────────────┘
```

**代码实现**：

```go
// pkg/helmhooks/executor.go

// HookExecutor 钩子执行器接口
type HookExecutor interface {
    Execute(ctx context.Context, hooks []HookSpec, opts HookOptions) error
}

// JobHookExecutor Job 类型钩子执行器
type JobHookExecutor struct {
    clientset kubernetes.Interface
    renderer  *HookRenderer
    logger    *bkev1beta1.BKELogger
}

// HookOptions 钩子执行选项
type HookOptions struct {
    Namespace string
    Timeout   time.Duration
    Prefix    string  // "pre-install" 或 "pre-uninstall"
    // 模板变量
    TemplateData map[string]interface{}
}

func (e *JobHookExecutor) Execute(ctx context.Context, hooks []HookSpec, opts HookOptions) error {
    for _, hook := range hooks {
        if hook.Type != "Job" {
            e.logger.Warn("unsupported hook type: %s, skipping", hook.Type)
            continue
        }
        
        // 1. 渲染 Hook Manifest
        manifest, err := e.renderer.Render(hook.Manifest, opts.TemplateData)
        if err != nil {
            return fmt.Errorf("render hook %s manifest: %w", hook.Name, err)
        }
        
        // 2. 创建 Job
        jobName := fmt.Sprintf("%s-%s-%d", opts.Prefix, hook.Name, time.Now().Unix())
        if err := e.createJob(ctx, jobName, manifest, opts.Namespace); err != nil {
            return fmt.Errorf("create hook job %s: %w", jobName, err)
        }
        
        // 3. 等待 Job 完成
        if err := e.waitForJob(ctx, jobName, opts.Namespace, opts.Timeout); err != nil {
            e.cleanupJob(ctx, jobName, opts.Namespace)
            return fmt.Errorf("hook job %s failed: %w", jobName, err)
        }
        
        // 4. 清理 Job
        e.cleanupJob(ctx, jobName, opts.Namespace)
    }
    return nil
}

// HookRenderer 钩子 Manifest 渲染器
type HookRenderer struct {
    funcMap template.FuncMap
}

func (r *HookRenderer) Render(manifest string, data map[string]interface{}) (string, error) {
    t, err := template.New("hook").Funcs(r.funcMap).Parse(manifest)
    if err != nil {
        return "", err
    }
    var buf bytes.Buffer
    if err := t.Execute(&buf, data); err != nil {
        return "", err
    }
    return buf.String(), nil
}
```

**在 HelmInstaller 中集成**：

```go
// pkg/helminstaller/installer.go

func (i *HelmInstaller) install(ctx context.Context, actionConfig *action.Configuration,
    helm *HelmSpec, opts InstallOptions) error {
    
    // 1. 执行 PreInstallHooks
    if len(helm.PreInstallHooks) > 0 {
        hookOpts := HookOptions{
            Namespace:    helm.Namespace,
            Timeout:      5 * time.Minute,
            Prefix:       "pre-install",
            TemplateData: buildHookTemplateData(opts),
        }
        if err := i.hookExecutor.Execute(ctx, helm.PreInstallHooks, hookOpts); err != nil {
            return fmt.Errorf("pre-install hooks failed: %w", err)
        }
    }
    
    // 2. 执行 Helm Install
    client := action.NewInstall(actionConfig)
    // ... 原有逻辑
}

func (i *HelmInstaller) uninstall(ctx context.Context, actionConfig *action.Configuration,
    helm *HelmSpec, opts InstallOptions) error {
    
    // 1. 执行 PreUninstallHooks
    if len(helm.PreUninstallHooks) > 0 {
        hookOpts := HookOptions{
            Namespace:    helm.Namespace,
            Timeout:      5 * time.Minute,
            Prefix:       "pre-uninstall",
            TemplateData: buildHookTemplateData(opts),
        }
        if err := i.hookExecutor.Execute(ctx, helm.PreUninstallHooks, hookOpts); err != nil {
            return fmt.Errorf("pre-uninstall hooks failed: %w", err)
        }
    }
    
    // 2. 执行 Helm Uninstall
    client := action.NewUninstall(actionConfig)
    // ... 原有逻辑
}
```

**使用示例**：

```yaml
helm:
  releaseName: coredns
  namespace: kube-system
  
  # 安装前钩子：备份现有配置
  preInstallHooks:
    - name: backup-config
      type: Job
      manifest: |
        apiVersion: batch/v1
        kind: Job
        spec:
          template:
            spec:
              containers:
              - name: backup
                image: bitnami/kubectl:latest
                command: ["/bin/sh", "-c", "kubectl get configmap coredns -n kube-system -o yaml > /backup/coredns.yaml"]
  
  # 卸载前钩子：清理依赖资源
  preUninstallHooks:
    - name: cleanup-dependencies
      type: Job
      manifest: |
        apiVersion: batch/v1
        kind: Job
        spec:
          template:
            spec:
              containers:
              - name: cleanup
                image: bitnami/kubectl:latest
                command: ["/bin/sh", "-c", "kubectl delete configmap coredns-custom -n kube-system --ignore-not-found"]
```

## 5. YamlInstaller 详细设计

### 5.1 核心组件架构

```txt
┌─────────────────────────────────────────────────────────────────┐
│                    YamlInstaller                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   BundleStore    │    │  YAML Parser     │                   │
│  │                  │    │                  │                   │
│  │ • bundle文件加载  │    │ • 多文档解析      │                   │
│  │ • 内联Resources  │    │ • GVK 识别       │                   │
│  │                  │    │ • 资源分组        │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌──────────────────────────────────────────┐                   │
│  │       ApplyStrategy Engine               │                   │
│  │                                          │                   │
│  │ • ServerSideApply (默认, 声明式字段管理)   │                  │
│  │ • Replace (删除+重建)                    │                   │
│  │ • CreateOnly (仅创建)                    │                   │
│  └──────────────────┬───────────────────────┘                  │
│                     │                                          │
│                     ▼                                          │
│  ┌──────────────────────────────────────────┐                  │
│  │            K8s Applier                   │                  │
│  │                                          │                  │
│  │ • 应用清单到目标集群                      │                   │
│  │ • Prune 裁剪废弃资源 (按 label selector)  │                   │
│  └──────────────────────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 YamlInstaller 执行流程图

**设计思路**：YamlInstaller 的执行流程分为 4 个主要步骤：加载清单、应用清单、健康检查、返回结果。与 HelmInstaller（7 步）相比更简单——无制品下载/SSH 执行/Helm SDK 等复杂子层，仅 Store 加载 + Applier 应用 + healthcheck。

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         YamlInstaller 执行流程                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   Apply()        │
                              │   入口函数        │
                              └────────┬─────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────────┐
                     │  1. 加载清单                           │
                     │  store.GetComponentManifests(         │
                     │    ctx, name, version, tmplCtx)       │
                     │  → BundleStore → CollectComponent     │
                     │    Manifests (bundle文件+内联Resources)│
                     └────────────────────┬──────────────────┘
                                          │
                                          ▼
                     ┌────────────────────────────────────────┐
                     │  2. 应用清单                            │
                     │  applier.ApplyComponent(ctx, pkg)      │
                     │  → ClusterApplier → ApplyYaml          │
                     │    (ServerSideApply/Replace/CreateOnly)│
                     └────────────────────┬───────────────────┘
                                          │
                                          ▼
                     ┌──────────────────────────────────────┐
                     │  3. 健康检查                          │
                     │  healthcheck.Run(ctx, clientset, hc) │
                     │  (见第 6 章, hc.Enabled=true 时执行)  │
                     └────────────────────┬─────────────────┘
                                          │
                               ┌──────────┴──────────┐
                               │                     │
                          检查通过                检查失败
                               │                     │
                               ▼                     ▼
                     ┌─────────────────┐   ┌─────────────────┐
                     │  返回成功        │   │  返回错误       │
                     │  return nil     │   │  return err     │
                     └─────────────────┘   └─────────────────┘
```

### 5.3 核心接口定义

**设计思路 — YAML 组件两层模型**：

YAML 组件采用两层结构：`YamlComponentExecutor`（dagexec 调度层，见 8.4.3.3）+ `YamlInstaller`（`pkg/yamlinstaller` 引擎层，本节）。`YamlInstaller` 持有 `manifest.Applier`，负责清单 Apply + 健康检查；与 `HelmInstaller` 命名对称。区别仅在 `YamlInstaller` 内部更简单——无下载/缓存等子层，仅 Apply + healthcheck。

| 层 | Helm | YAML |
| ---- | ------ | ------ |
| dagexec 执行器 | `HelmComponentExecutor` | `YamlComponentExecutor`（见 8.4.3.3） |
| 独立包引擎 | `helminstaller.HelmInstaller` | `yamlinstaller.YamlInstaller`（本节） |

**Installer（引擎层）完成的逻辑**（`YamlInstaller.Apply`）：

- 构建 ComponentPackage → `applier.ApplyComponent()` → 健康检查（PodReady/EndpointReady/Custom）

**Executor（调度层）完成的逻辑**（`YamlComponentExecutor`，见 8.4.3.3）：

- 获取 ComponentVersion → VersionContext 判断是否需要执行 → 委托 `YamlInstaller.Apply()` → 处理失败策略
- 无节点级调度（应用到集群而非单节点）
- 无回滚机制（SSA 天然支持幂等，重新 Apply 上一版本即可回滚）

```go
// pkg/yamlinstaller/installer.go

// YamlInstaller YAML 组件安装器（引擎层，对称 HelmInstaller）
// 持有 manifest.Store (加载清单) + manifest.Applier (应用清单)，负责清单 Apply + 健康检查。
type YamlInstaller struct {
    store   manifest.Store            // 加载清单 (复用 BundleStore.GetComponentManifests)
    applier manifest.Applier          // 应用清单 (现有 manifest.ClusterApplier 实现)
    logger  *bkev1beta1.BKELogger
}

type YamlInstallerConfig struct {
    Store   manifest.Store
    Applier manifest.Applier
    Logger  *bkev1beta1.BKELogger
}

func NewYamlInstaller(cfg YamlInstallerConfig) *YamlInstaller {
    return &YamlInstaller{store: cfg.Store, applier: cfg.Applier, logger: cfg.Logger}
}

// Apply 应用 YAML 清单 + 健康检查 (由 YamlComponentExecutor 委托调用)
func (i *YamlInstaller) Apply(ctx context.Context, cv *configv1alpha1.ComponentVersion, execCtx *ExecutionContext) error {
    // 通过 manifest.Store 加载清单 (复用 BundleStore.GetComponentManifests 逻辑:
    // bundle 文件 (components/<name>/<version>/*.yaml) + cv.Spec.Resources 内联清单)
    pkg, err := i.store.GetComponentManifests(ctx, cv.Spec.Name, cv.Spec.Version, execCtx.TemplateContext)
    if err != nil {
        return fmt.Errorf("get manifests for %s: %w", cv.Spec.Name, err)
    }

    // 应用 Manifest
    if err := i.applier.ApplyComponent(ctx, pkg); err != nil {
        return fmt.Errorf("apply manifests for %s: %w", cv.Spec.Name, err)
    }

    // 健康检查 (应用清单后验证 Pod/Endpoint 就绪)
    // 复用共享 pkg/healthcheck 包 (见 6)，避免与 HelmInstaller 重复
    if cv.Spec.YAML != nil && cv.Spec.YAML.HealthCheck != nil && cv.Spec.YAML.HealthCheck.Enabled {
        if err := healthcheck.Run(ctx, execCtx.TargetClient, *cv.Spec.YAML.HealthCheck); err != nil {
            return fmt.Errorf("health check failed for %s: %w", cv.Spec.Name, err)
        }
    }
    return nil
}
```

> 健康检查实现委托共享 `pkg/healthcheck` 包（`healthcheck.Run(ctx, execCtx.TargetClient, *cv.Spec.YAML.HealthCheck)`）。共享包的接口定义与 PodReady/EndpointReady/Custom 检查实现详见 **第 6 章 HealthCheck 共享包设计**。

**设计思路 — YamlInstaller 健康检查 clientset 来源**：

`YamlInstaller.Apply` 中调用 `healthcheck.Run(ctx, execCtx.TargetClient, ...)` 需要 `kubernetes.Interface` (typed clientset)。当前 `YamlInstallerConfig` 仅注入 `Store` + `Applier`，未注入 clientset。解决方案：通过 `ExecutionContext.TargetClient` 传递，无需修改 `YamlInstallerConfig`。`ExecutionContext.TargetClient` 由 controllers 层在 `buildExecutionContext` 中注入，是目标集群的 typed clientset。YamlInstaller 通过 `execCtx.TargetClient` 访问，无需在 `YamlInstallerConfig` 中重复注入。这与 `HelmInstaller` 的设计一致——`HelmInstaller` 在 Config 中直接注入 `Clientset`，因为 HelmInstaller 不接收 `ExecutionContext`。

### 5.4 健康检查

YamlInstaller 的健康检查在 `Apply` 方法内部调用共享 `pkg/healthcheck` 包执行（`healthcheck.Run(ctx, execCtx.TargetClient, *cv.Spec.YAML.HealthCheck)`）。当 `cv.Spec.YAML.HealthCheck.Enabled` 为 true 时，清单 Apply 后执行 PodReady/EndpointReady/Custom 检查。接口定义与实现详见 **第 6 章 HealthCheck 共享包设计**。

### 5.5 YAML Uninstall 流程

**设计思路 — YAML 组件卸载与 Helm 的区别**：

YAML 组件的卸载流程与 Helm 有本质区别：

- **Helm**：通过 Helm SDK 执行 `helm uninstall`，删除 Release
- **YAML**：通过 K8s API 删除资源，支持 Prune 裁剪

YAML 组件没有"服务"概念，卸载就是删除已应用的 Kubernetes 资源。

**卸载流程图**：

```txt
YAML Uninstall 流程:
┌─────────────────────────────────────────────────────────────┐
│ 1. 加载已安装资源清单                                         │
│    - 从 ComponentVersion 获取 Resources                      │
│    - 或从 ManifestStore 获取已应用的清单                      │
├─────────────────────────────────────────────────────────────┤
│ 2. 删除资源 (逆序)                                           │
│    - 按 GVK 依赖关系逆序删除                                 │
│    - CRD 最后删除                                           │
│    - 使用 DeletePropagationBackground 或 Foreground         │
├─────────────────────────────────────────────────────────────┤
│ 3. 可选：Prune 裁剪                                          │
│    - 按 PruneLabelSelector 查找资源                          │
│    - 删除不在当前清单中的资源                                 │
├─────────────────────────────────────────────────────────────┤
│ 4. 验证删除完成                                              │
│    - 检查资源是否已删除                                       │
└─────────────────────────────────────────────────────────────┘
```

**代码实现**：

```go
// pkg/yamlinstaller/installer.go

// Uninstall 卸载 YAML 组件
func (i *YamlInstaller) Uninstall(ctx context.Context, cv *configv1alpha1.ComponentVersion, execCtx *ExecutionContext) error {
    // 1. 获取已安装资源
    pkg, err := i.store.GetComponentManifests(ctx, cv.Spec.Name, cv.Spec.Version, execCtx.TemplateContext)
    if err != nil {
        return fmt.Errorf("get manifests for %s: %w", cv.Spec.Name, err)
    }

    // 2. 逆序删除资源
    if err := i.applier.DeleteComponent(ctx, pkg); err != nil {
        return fmt.Errorf("delete manifests for %s: %w", cv.Spec.Name, err)
    }

    // 3. 可选：Prune 裁剪（按 label selector 列出资源，与当前清单比对后删除多余资源）
    if cv.Spec.YAML != nil && cv.Spec.YAML.Prune {
        namespace := ""
        if cv.Spec.YAML != nil {
            namespace = cv.Spec.YAML.Namespace
        }
        if err := i.applier.PruneResources(ctx, cv.Spec.YAML.PruneLabelSelector, namespace, pkg.Manifests); err != nil {
            return fmt.Errorf("prune resources for %s: %w", cv.Spec.Name, err)
        }
    }

    return nil
}
```

**Applier 扩展接口**：

```go
// pkg/manifest/applier.go

// Applier 扩展接口，支持删除和裁剪
type Applier interface {
    // ApplyComponent 应用组件清单
    ApplyComponent(ctx context.Context, pkg *ComponentPackage) error
    
    // DeleteComponent 删除组件资源（按 GVK 依赖逆序删除，IsNotFound 幂等）
    DeleteComponent(ctx context.Context, pkg *ComponentPackage) error
    
    // PruneResources 裁剪不在清单中的资源（按 label selector 列出，与 currentManifests 比对后删除多余资源）
    // namespace: 限定裁剪的命名空间范围（空字符串表示集群级别）
    // currentManifests: 当前版本的清单，用于构建 wantSet 比对
    PruneResources(ctx context.Context, selector map[string]string, namespace string, currentManifests [][]byte) error
}
```

**Applier 接口实现**：

```go
// pkg/manifest/applier.go

// DeleteComponent 删除组件资源（按 GVK 依赖逆序删除，IsNotFound 幂等）
// 复用现有基础设施:
// - kubeClient() 获取远端 client (与 ApplyComponent 对称)
// - addonutil.SortUninstallUnstructuredByKind() GVK 逆序排序 (复用 Helm UninstallOrder)
// - dynamic.Interface.Delete() 执行删除
func (a *ClusterApplier) DeleteComponent(ctx context.Context, pkg *ComponentPackage) error {
    if pkg == nil || len(pkg.Manifests) == 0 {
        return nil
    }

    kubeClient, err := a.kubeClient()
    if err != nil {
        return fmt.Errorf("get kube client for delete %s: %w", pkg.Name, err)
    }
    clientset, dynamicClient := kubeClient.KubeClient()
    if clientset == nil || dynamicClient == nil {
        return fmt.Errorf("kube client not initialized for %s", pkg.Name)
    }

    // 1. 解析所有清单为 unstructured 对象
    var resources []unstructured.Unstructured
    for _, doc := range pkg.Manifests {
        parsed, err := parseMultiDocYAML(doc)
        if err != nil {
            return fmt.Errorf("parse manifest for delete %s: %w", pkg.Name, err)
        }
        resources = append(resources, parsed...)
    }
    if len(resources) == 0 {
        return nil
    }

    // 2. 按 GVK 依赖逆序排序 (复用 Helm UninstallOrder)
    //    排序规则: Namespace → CRD → ... → Deployment → Service → ConfigMap → Secret → RBAC
    //    实现: addonutil.SortUninstallUnstructuredByKind() 使用 releaseutil.UninstallOrder
    resources = addonutil.SortUninstallUnstructuredByKind(resources)

    // 3. 逐个删除资源
    for _, res := range resources {
        gvr, err := a.resolveGVR(clientset, res)
        if err != nil {
            a.logger.Warn("skip delete: cannot resolve GVR",
                "kind", res.GetKind(), "name", res.GetName(), "error", err)
            continue // 无法解析 GVR 时跳过，不阻塞后续删除
        }

        namespace := res.GetNamespace()
        var dr dynamic.ResourceInterface
        if namespace != "" {
            dr = dynamicClient.Resource(gvr).Namespace(namespace)
        } else {
            dr = dynamicClient.Resource(gvr)
        }

        // 使用 DeletePropagationBackground: 删除请求立即返回，K8s 后台级联删除依赖资源
        propagation := metav1.DeletePropagationBackground
        err = dr.Delete(ctx, res.GetName(), metav1.DeleteOptions{
            PropagationPolicy: &propagation,
        })
        if err != nil {
            if apierrors.IsNotFound(err) {
                // 资源不存在视为成功 (幂等)
                continue
            }
            return fmt.Errorf("delete %s %s/%s: %w",
                res.GetKind(), res.GetNamespace(), res.GetName(), err)
        }
    }

    return nil
}

// PruneResources 裁剪不在当前清单中的资源
// 流程:
// 1. 按 label selector 列出目标集群中的资源
// 2. 从 currentManifests 构建 wantSet (GVK + namespace + name)
// 3. 删除 wantSet 之外的资源 (仅限 PruneableGVKs 安全范围)
func (a *ClusterApplier) PruneResources(
    ctx context.Context,
    selector map[string]string,
    namespace string,
    currentManifests [][]byte,
) error {
    kubeClient, err := a.kubeClient()
    if err != nil {
        return fmt.Errorf("get kube client for prune: %w", err)
    }
    clientset, dynamicClient := kubeClient.KubeClient()
    if clientset == nil || dynamicClient == nil {
        return fmt.Errorf("kube client not initialized for prune")
    }

    // 1. 构建 wantSet: 当前清单中所有资源的 (GVK, namespace, name) 集合
    wantSet := make(map[resourceKey]struct{})
    for _, doc := range currentManifests {
        resources, err := parseMultiDocYAML(doc)
        if err != nil {
            return fmt.Errorf("parse manifest for prune: %w", err)
        }
        for _, res := range resources {
            key := resourceKey{
                gvk:       res.GroupVersionKind(),
                namespace: res.GetNamespace(),
                name:      res.GetName(),
            }
            wantSet[key] = struct{}{}
        }
    }

    // 2. 按 PruneableGVKs 逐类型列出集群中的资源
    for _, gvk := range PruneableGVKs {
        gvr, err := a.resolveGVR(clientset, unstructured.Unstructured{Object: map[string]interface{}{
            "apiVersion": gvk.Group + "/" + gvk.Version,
            "kind":       gvk.Kind,
        }})
        if err != nil {
            continue // 目标集群不支持此 GVK，跳过
        }

        var list *unstructured.UnstructuredList
        if namespace != "" {
            list, err = dynamicClient.Resource(gvr).Namespace(namespace).List(ctx, metav1.ListOptions{
                LabelSelector: labels.Set(selector).String(),
            })
        } else {
            list, err = dynamicClient.Resource(gvr).List(ctx, metav1.ListOptions{
                LabelSelector: labels.Set(selector).String(),
            })
        }
        if err != nil {
            if apierrors.IsNotFound(err) {
                continue
            }
            return fmt.Errorf("list resources for prune (GVK %s): %w", gvk, err)
        }

        // 3. 删除不在 wantSet 中的资源
        for i := range list.Items {
            item := &list.Items[i]
            key := resourceKey{
                gvk:       item.GroupVersionKind(),
                namespace: item.GetNamespace(),
                name:      item.GetName(),
            }
            if _, wanted := wantSet[key]; wanted {
                continue // 在当前清单中，保留
            }

            // 删除多余资源
            var dr dynamic.ResourceInterface
            if item.GetNamespace() != "" {
                dr = dynamicClient.Resource(gvr).Namespace(item.GetNamespace())
            } else {
                dr = dynamicClient.Resource(gvr)
            }
            propagation := metav1.DeletePropagationBackground
            if err := dr.Delete(ctx, item.GetName(), metav1.DeleteOptions{
                PropagationPolicy: &propagation,
            }); err != nil && !apierrors.IsNotFound(err) {
                return fmt.Errorf("prune %s %s/%s: %w",
                    item.GetKind(), item.GetNamespace(), item.GetName(), err)
            }
        }
    }

    return nil
}

// resourceKey 资源唯一标识 (用于 wantSet 比对)
type resourceKey struct {
    gvk       schema.GroupVersionKind
    namespace string
    name      string
}

// PruneableGVKs 可裁剪的资源类型 (安全范围，避免误删 CRD 等核心资源)
var PruneableGVKs = []schema.GroupVersionKind{
    {Group: "", Version: "v1", Kind: "ConfigMap"},
    {Group: "", Version: "v1", Kind: "Secret"},
    {Group: "", Version: "v1", Kind: "Service"},
    {Group: "", Version: "v1", Kind: "ServiceAccount"},
    {Group: "apps", Version: "v1", Kind: "Deployment"},
    {Group: "apps", Version: "v1", Kind: "DaemonSet"},
    {Group: "apps", Version: "v1", Kind: "StatefulSet"},
    {Group: "batch", Version: "v1", Kind: "Job"},
    {Group: "batch", Version: "v1", Kind: "CronJob"},
    {Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "Role"},
    {Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "RoleBinding"},
    {Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "ClusterRole"},
    {Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "ClusterRoleBinding"},
    {Group: "networking.k8s.io", Version: "v1", Kind: "Ingress"},
    {Group: "policy", Version: "v1", Kind: "PodDisruptionBudget"},
}

// parseMultiDocYAML 解析多文档 YAML 为 unstructured 对象列表
// 复用 pkg/kube/yaml.go 中的解析逻辑
func parseMultiDocYAML(data []byte) ([]unstructured.Unstructured, error) {
    var resources []unstructured.Unstructured
    decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewReader(data), 4096)
    for {
        var obj unstructured.Unstructured
        err := decoder.Decode(&obj)
        if err == io.EOF {
            break
        }
        if err != nil {
            return nil, fmt.Errorf("parse YAML document: %w", err)
        }
        if obj.Object == nil {
            continue // 跳过空文档
        }
        resources = append(resources, obj)
    }
    return resources, nil
}

// resolveGVR 从 GVK 解析 GVR (GroupVersionResource)
// 复用 restmapper 发现 API 资源
func (a *ClusterApplier) resolveGVR(
    clientset kubernetes.Interface,
    res unstructured.Unstructured,
) (schema.GroupVersionResource, error) {
    gvk := res.GroupVersionKind()
    restMapper := restmapper.NewDeferredDiscoveryRESTMapper(
        memory.NewMemCacheClient(clientset.Discovery()),
    )
    mapping, err := restMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
    if err != nil {
        return schema.GroupVersionResource{}, fmt.Errorf("resolve GVR for %s: %w", gvk, err)
    }
    return mapping.Resource, nil
}
```

**设计说明**：

| 设计点 | 说明 |
| ------- | ------ |
| **GVK 删除顺序** | `gvkDeleteOrder` 定义优先级，数值越大越先删除。Deployment(50) > Service(40) > ConfigMap(30) > RBAC(10) > Namespace(-10) > CRD(-30)。未列出的 Kind 优先级为 0，在 RBAC 和 Namespace 之间删除 |
| **DeletePropagationBackground** | 使用 Background 传播策略，删除请求立即返回，K8s 后台级联删除依赖资源。显式设置 PropagationPolicy 以确保级联行为可控 |
| **IsNotFound 幂等** | 删除时资源不存在视为成功，支持重复执行 Uninstall |
| **Prune 安全范围** | `PruneableGVKs` 显式列出可裁剪的资源类型，避免误删非组件管理的资源。CRD 不在列表中，防止裁剪时意外删除 CRD 导致数据丢失 |
| **复用基础设施** | 复用 `ClusterApplier.kubeClient()` 获取远端 client，复用 `restmapper` 发现 API 资源，与 `ApplyComponent` 对称 |

**与 Helm 卸载的对比**：

| 组件类型 | 卸载机制 | 是否支持 Prune |
| --------- | --------- | --------------- |
| Helm | Helm SDK `helm uninstall` | ❌ |
| YAML | K8s API 删除资源 | ✅ 按 label 裁剪 |

## 6. HealthCheck 共享包设计

**设计思路 — 横切关注点独立成包**：

PodReady/EndpointReady/Custom 三种 K8s 资源就绪检查是 HelmInstaller（4.3）与 YamlInstaller（5.3）共用的横切关注点。两处各自实现 `runHealthCheck`/`checkPodReady`/`checkEndpointReady`/`checkCustom`，逻辑近乎相同。现抽取为独立共享包 `pkg/healthcheck`，两处均委托调用，消除重复。

**作用范围**：

- ✅ `HelmInstaller`（4.3）：Helm `install/upgrade` 返回后执行自定义健康检查
- ✅ `YamlInstaller`（5.3）：清单 Apply 后执行健康检查

**与 Helm `--wait` 的关系**（仅 Helm 侧）：两者同时生效，是两层递进检查非互斥——Helm `--wait`（`strategy.wait: true`）在 `helm install/upgrade` 命令内部先执行（仅检查全部 Pod Ready）；自定义 `healthCheck` 在 Helm 命令返回后执行（支持 `minReady` 部分就绪、Endpoint、Custom）。若 `--wait` 失败且 `atomic: true`，Helm 自动回滚，不执行自定义 `healthCheck`。

**执行流程图**：

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         HealthCheck 执行流程                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   Run()          │
                              │  入口函数        │
                              └────────┬─────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────────┐
                     │  设置超时与重试间隔                   │
                     │  timeout = hc.Timeout (默认 3m)      │
                     │  interval = hc.Interval (默认 5s)    │
                     │  deadline = now + timeout            │
                     └────────────────────┬─────────────────┘
                                          │
                                          ▼
                     ┌──────────────────────────────────────┐
                     │  now < deadline ?                    │
                     └────────────────────┬─────────────────┘
                                          │
                            ┌─────────────┴─────────────┐
                            │                           │
                           是                          否
                            │                           │
                            ▼                           ▼
              ┌─────────────────────────┐   ┌─────────────────┐
              │  遍历 hc.Checks         │   │  返回超时错误    │
              │  allReady = true        │   │  return err     │
              └────────────┬────────────┘   └─────────────────┘
                           │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │  PodReady    │ │EndpointReady │ │   Custom     │
   │ list Pods    │ │ get Endpoints│ │ exec command │
   │ count Ready  │ │ check Addrs  │ │ exit code 0  │
   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
               ┌─────────────────────────┐
               │  allReady ?             │
               └────────────┬────────────┘
                    ┌───────┴───────┐
                    │               │
                   是              否
                    │               │
                    ▼               ▼
          ┌─────────────────┐ ┌─────────────────┐
          │  返回成功        │ │  Sleep(interval)│
          │  return nil     │ │  继续重试循环    │
          └─────────────────┘ └─────────────────┘
                                    │
                                    └──→ 回到 now < deadline?
```

```go
// pkg/healthcheck/healthcheck.go

// Run 执行健康检查，按 hc.Checks 遍历 PodReady/EndpointReady/Custom，
// 重试直到全部通过或超时。HelmInstaller 与 YamlInstaller 共用此入口。
// client 为目标集群 typed clientset (kubernetes.Interface)，非 controller-runtime client。
func Run(ctx context.Context, client kubernetes.Interface, hc HealthCheckSpec) error {
    timeout := parseDurationDefault(hc.Timeout, 3*time.Minute)
    interval := parseDurationDefault(hc.Interval, 5*time.Second)
    deadline := time.Now().Add(timeout)

    for time.Now().Before(deadline) {
        allReady := true
        for _, check := range hc.Checks {
            switch check.Type {
            case "PodReady":
                if check.PodReady == nil {
                    return fmt.Errorf("PodReady check requires 'podReady' config")
                }
                ready, err := checkPodReady(ctx, client, check.PodReady)
                if err != nil || !ready {
                    allReady = false
                }
            case "EndpointReady":
                if check.EndpointReady == nil {
                    return fmt.Errorf("EndpointReady check requires 'endpointReady' config")
                }
                ready, err := checkEndpointReady(ctx, client, check.EndpointReady)
                if err != nil || !ready {
                    allReady = false
                }
            case "Custom":
                if check.Custom == nil {
                    return fmt.Errorf("Custom check requires 'custom' config")
                }
                ready, err := checkCustom(ctx, check.Custom)
                if err != nil || !ready {
                    allReady = false
                }
            }
        }
        if allReady {
            return nil
        }
        time.Sleep(interval)
    }

    return fmt.Errorf("health check timed out after %s", timeout)
}

// checkPodReady 检查 Pod Ready 状态 (支持 minReady 部分就绪)
func checkPodReady(ctx context.Context, client kubernetes.Interface, spec *PodReadyCheckSpec) (bool, error) {
    selector, err := labels.Parse(spec.LabelSelector)
    if err != nil {
        return false, fmt.Errorf("invalid label selector %q: %w", spec.LabelSelector, err)
    }
    podList, err := client.CoreV1().Pods(spec.Namespace).List(ctx, metav1.ListOptions{
        LabelSelector: selector.String(),
    })
    if err != nil {
        return false, err
    }
    minReady := int(spec.MinReady)
    if minReady == 0 {
        minReady = len(podList.Items) // 默认要求全部 Ready
    }
    readyCount := 0
    for _, pod := range podList.Items {
        for _, cond := range pod.Status.Conditions {
            if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
                readyCount++
                break
            }
        }
    }
    return readyCount >= minReady, nil
}

// checkEndpointReady 检查 Service Endpoint 是否有就绪端点
func checkEndpointReady(ctx context.Context, client kubernetes.Interface, spec *EndpointReadyCheckSpec) (bool, error) {
    endpoints, err := client.CoreV1().Endpoints(spec.Namespace).Get(ctx, spec.ServiceName, metav1.GetOptions{})
    if err != nil {
        return false, err
    }
    for _, subset := range endpoints.Subsets {
        if len(subset.Addresses) > 0 {
            return true, nil // 有就绪端点
        }
    }
    return false, nil
}

// checkCustom 执行自定义检查命令 (在控制器 Pod 中执行, 退出码 0 = 通过)
func checkCustom(ctx context.Context, spec *CustomCheckSpec) (bool, error) {
    cmd := exec.CommandContext(ctx, "/bin/sh", "-c", spec.Command)
    if err := cmd.Run(); err != nil {
        return false, nil // 非零退出码 = 未就绪
    }
    return true, nil
}
```

**类型归属**：`HealthCheckSpec`/`HealthCheckItemSpec`/`PodReadyCheckSpec`/`EndpointReadyCheckSpec`/`CustomCheckSpec` 等类型定义应迁移到 `pkg/healthcheck/types.go`，供 Helm/YAML 共用；`api/v1alpha1` 的 CRD 字段类型可内嵌或别名这些类型，避免在 `helminstaller`/`yamlinstaller` 多处重复定义。

**调用方**：

- `HelmInstaller.runHealthCheck`（4.3）：`return healthcheck.Run(ctx, i.clientset, hc)`
- `YamlInstaller.Apply`（5.3）：`healthcheck.Run(ctx, execCtx.TargetClient, *cv.Spec.YAML.HealthCheck)`

### 6.1 类型定义

**设计思路**：健康检查类型定义独立于具体 Installer，供 Helm/YAML 组件共用。CRD 字段类型引用此处的定义，避免在多处重复定义。

```go
// pkg/healthcheck/types.go

// HealthCheckSpec 定义健康检查规格
type HealthCheckSpec struct {
    // 是否启用健康检查
    Enabled bool `json:"enabled"`
    
    // 健康检查超时时间
    Timeout string `json:"timeout,omitempty"`
    
    // 健康检查间隔
    Interval string `json:"interval,omitempty"`
    
    // 健康检查项列表
    Checks []HealthCheckItemSpec `json:"checks,omitempty"`
}

// HealthCheckItemSpec 定义健康检查项规格
// 按检查类型嵌套子结构体, 消除 Name 字段双重含义
// Type 决定使用哪个子结构体, 其他子结构体应为 nil
type HealthCheckItemSpec struct {
    // 检查类型: PodReady / EndpointReady / Custom
    Type string `json:"type"`
    
    // PodReady 检查配置 (type=PodReady 时使用)
    PodReady *PodReadyCheckSpec `json:"podReady,omitempty"`
    
    // EndpointReady 检查配置 (type=EndpointReady 时使用)
    EndpointReady *EndpointReadyCheckSpec `json:"endpointReady,omitempty"`
    
    // Custom 检查配置 (type=Custom 时使用)
    Custom *CustomCheckSpec `json:"custom,omitempty"`
}

// PodReadyCheckSpec 定义 Pod 就绪检查规格
type PodReadyCheckSpec struct {
    // 命名空间
    Namespace string `json:"namespace"`
    
    // 标签选择器 (如 "k8s-app=kube-dns")
    LabelSelector string `json:"labelSelector"`
    
    // 最小就绪数量 (0=要求全部 Ready, 1=至少 1 个 Ready)
    MinReady int32 `json:"minReady,omitempty"`
}

// EndpointReadyCheckSpec 定义 Endpoint 就绪检查规格
type EndpointReadyCheckSpec struct {
    // 命名空间
    Namespace string `json:"namespace"`
    
    // Service 名称
    ServiceName string `json:"serviceName"`
    
    // 端口 (可选, 检查特定端口是否就绪)
    Port int32 `json:"port,omitempty"`
}

// CustomCheckSpec 定义自定义检查规格
type CustomCheckSpec struct {
    // 检查命令 (在控制器 Pod 中执行, 退出码 0=通过, 如 "curl -s http://.../healthz")
    Command string `json:"command"`
}
```

## 7. 模板变量系统与 TemplateContext 详细设计

### 7.1 TemplateContext 扩展策略

**设计思路**：为避免重复造轮子，本设计复用并扩展现有的 `pkg/manifest.TemplateContext` 结构体。现有 TemplateContext 用于 YAML/Manifest 组件的模板渲染，包含 4 个基础字段。在此基础上扩展，增加 Helm 组件所需的版本信息和镜像仓库字段。

**复用策略**：

- **向后兼容**：现有 TemplateContext 的 4 个字段保持不变，YAML 组件代码无需修改
- **扩展字段**：新增字段均为可选，YAML 组件不使用时留空即可
- **统一接口**：HelmInstaller 和 YamlInstaller 共享同一个 TemplateContext，简化 DAG 调度器的数据传递

**现有 TemplateContext** (`pkg/manifest/types.go`)：

```go
type TemplateContext struct {
    ClusterName       string  // 已有：集群名称
    Namespace         string  // 已有：命名空间
    KubernetesVersion string  // 已有：K8s 版本
    OpenFuyaoVersion  string  // 已有：OpenFuyao 版本
}
```

**扩展后的 TemplateContext**：

```go
type TemplateContext struct {
    // 现有字段 (保持向后兼容，快捷访问)
    ClusterName       string
    Namespace         string
    KubernetesVersion string
    OpenFuyaoVersion  string
    
    // 新增：完整配置引用 (通用性与可扩展性)
    // 注入 BKEConfig，模板可访问任意配置字段
    // 简单场景使用快捷字段: {{.ClusterName}}
    // 复杂场景使用完整引用: {{.Config.Cluster.ContainerRuntime.CRI}}
    Config            *confv1beta1.BKEConfig
    
    // 新增：集群扩展信息
    APIServer         string
    ServiceCIDR       string
    PodCIDR           string
    DNSDomain         string
    
    // 新增：版本信息
    ComponentVersion          string
    ComponentPreviousVersion  string
    
    // 新增：镜像仓库
    ImageRegistry     string
    ImagePullSecret   string
}
```

**设计决策 — 混合设计（快捷字段 + 完整配置）**：

| 方案 | 优点 | 缺点 | 选择 |
| ------ | ------ | ------ | ------ |
| 仅快捷字段 | 模板简洁、类型安全 | 新增字段需修改 TemplateContext | ❌ |
| 仅完整配置 | 通用性强、可扩展 | 模板访问路径长、与结构耦合 | ❌ |
| **混合设计** | **兼顾简洁与灵活、向后兼容** | **需维护两套访问方式** | **✅** |

**使用示例**：

```yaml
# 简单场景：使用快捷字段
values:
  clusterName: "{{.ClusterName}}"
  imageRegistry: "{{.ImageRegistry}}"

# 复杂场景：使用完整配置
values:
  imageRepo: "{{.Config.Cluster.ImageRepo.Domain}}"
  podCIDR: "{{.Config.Cluster.Networking.PodCIDR}}"
```

#### 7.1.1 TemplateContext 实现规格

**设计思路**：明确 `TemplateContext` 扩展字段的注入时机和边界条件，确保实现的一致性和可测试性。

**目标代码** (`pkg/manifest/types.go`)：

```go
// TemplateContext carries cluster fields used to render component templates.
type TemplateContext struct {
    // 现有字段 (保持向后兼容，快捷访问)
    ClusterName       string
    Namespace         string
    KubernetesVersion string
    OpenFuyaoVersion  string
    
    // 新增：完整配置引用 (通用性与可扩展性)
    Config            *confv1beta1.BKEConfig
    
    // 新增：集群扩展信息
    APIServer         string
    ServiceCIDR       string
    PodCIDR           string
    DNSDomain         string
    
    // 新增：版本信息
    ComponentVersion          string
    ComponentPreviousVersion  string
    
    // 新增：镜像仓库
    ImageRegistry     string
    ImagePullSecret   string
}
```

**注入时机**：

| 字段组 | 注入时机 | 注入位置 | 说明 |
| -------- | --------- | --------- | ------ |
| 基础字段 | `NewExecutionContext` 初始化时 | `pkg/dagexec/execution_context.go` | ClusterName, Namespace, KubernetesVersion, OpenFuyaoVersion |
| `Config` | `NewExecutionContext` 初始化时 | `pkg/dagexec/execution_context.go` | 注入完整 BKEConfig 引用 |
| 集群扩展字段 | `NewExecutionContext` 初始化时 | `pkg/dagexec/execution_context.go` | APIServer, ServiceCIDR, PodCIDR, DNSDomain |
| 版本信息 | `ComponentExecutor.ExecuteComponent` 时 | 各 Executor 内部 | ComponentVersion 从 CV 对象填充 |
| 镜像仓库 | `NewExecutionContext` 初始化时 | `pkg/dagexec/execution_context.go` | ImageRegistry, ImagePullSecret |

**边界条件处理**：

| 场景 | 处理方式 |
| ------ | --------- |
| `ClusterConfig` 为 nil | 基础字段留空 |
| `BKECluster` 为 nil | `NewExecutionContext` 返回错误，不创建 ExecutionContext |

### 7.2 模板变量系统

模板变量系统支持 Helm 组件所需的模板变量，覆盖集群、版本和镜像仓库。

**变量与 TemplateContext 字段映射**：

#### 1. 集群信息变量 (Cluster Variables)

| 变量 | 说明 | TemplateContext 字段 | 示例值 |
| ------ | ------ | --------------------- | -------- |
| `{{clusterName}}` | 集群名称 | `.ClusterName` | `my-cluster` |
| `{{clusterNamespace}}` | 集群命名空间 | `.Namespace` | `default` |
| `{{kubernetesVersion}}` | 集群 Kubernetes 版本 | `.KubernetesVersion` | `v1.29.0` |
| `{{openFuyaoVersion}}` | OpenFuyao 版本 | `.OpenFuyaoVersion` | `v2.6.0` |

#### 2. 版本信息变量 (Version Variables)

| 变量 | 说明 | TemplateContext 字段 | 示例值 |
| ------ | ------ | --------------------- | -------- |
| `{{componentVersion}}` | 当前组件版本 | `.ComponentVersion` | `v1.11.1` |
| `{{componentPreviousVersion}}` | 上一组件版本 | `.ComponentPreviousVersion` | `v1.10.1` |

#### 3. 镜像仓库变量 (Registry Variables)

| 变量 | 说明 | TemplateContext 字段 | 示例值 |
| ------ | ------ | --------------------- | -------- |
| `{{imageRegistry}}` | 镜像仓库地址 | `.ImageRegistry` | `registry.openfuyao.cn` |
| `{{imagePullSecret}}` | 镜像拉取 Secret | `.ImagePullSecret` | `registry-secret` |

### 7.3 TemplateContext 构建流程

**设计思路**：DAG 调度器在执行组件前构建 TemplateContext，包含集群信息、版本信息等。

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TemplateContext 构建流程                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────┐
                    │  DAG Scheduler.ExecuteDAG()          │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  构建基础 TemplateContext             │
                    │                                      │
                    │  tmpl := manifest.TemplateContext{   │
                    │    ClusterName:       cluster.Name,  │
                    │    Namespace:         cluster.NS,    │
                    │    KubernetesVersion: cluster.K8sVer,│
                    │    OpenFuyaoVersion:  cluster.OFVer, │
                    │    APIServer:         cluster.API,   │
                    │    ServiceCIDR:       cluster.SvcCIDR│
                    │    ImageRegistry:     cluster.ImgReg  │
                    │  }                                   │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  遍历执行批次                         │
                    │  for _, batch := range batches       │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  对每个组件:                          │
                    │  executeComponent(ctx, node, tmpl)   │
                    └────────────────────┬─────────────────┘
                                         │
                      ┌──────────────────┼────────────────────┐
                      │                   │                    │
                      ▼                   ▼                    ▼
            ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
            │  YAML 组件      │  │  Helm 组件       │  │  Inline 组件     │
            │                 │  │                 │  │                 │
            │  使用基础字段    │  │  使用基础字段    │  │  使用基础字段    │
            │  - ClusterName  │  │  + APIServer    │  │  - ClusterName  │
            │  - Namespace    │  │  + ServiceCIDR  │  │  - Namespace    │
            │  - K8sVersion   │  │                 │  │  - K8sVersion   │
            │                 │  │  渲染 Values     │  │                 │
            │  渲染 Manifest  │  │  helm install   │  │  Phase 执行     │
            │  应用到集群      │  │                 │  │  (无需模板渲染)  │
            └─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 8. DAG 集成详细设计

### 8.1 执行器注册

**设计思路**：DAG 调度器根据 ComponentVersion 的类型选择对应的执行器。系统支持三种组件类型：Helm（Helm Chart）、Inline（内联代码）、YAML（清单文件）。每种类型对应一个专门的 Executor，负责该类型组件的完整生命周期管理。

**关键设计点**：

- **类型分发**：根据 `cv.Spec.Type` 选择对应的 Executor
- **执行器注册**：每个 Executor 实现 `ComponentExecutor` 接口
- **依赖注入**：Executor 通过构造函数注入所需的 Installer/Applier

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DAG 执行器注册流程                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │  DAG Scheduler   │
                              │  接收组件节点     │
                              └────────┬─────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  获取 ComponentVersion               │
                    │  manifestStore.GetComponentVersion() │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  判断组件类型                         │
                    │  switch cv.Spec.Type                 │
                    └────────────────────┬─────────────────┘
                                         │
               ┌──────────────┬──────────┼──────────┐
               │              │           │          │
               ▼              ▼           ▼          │
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │ ComponentType│ │ ComponentType│ │ ComponentType│
     │ Helm         │ │ Inline       │ │ YAML         │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │ HelmCompo-   │ │ InlineCompo- │ │ YamlCompo-   │
      │ nentExecutor │ │ nentExecutor │ │ nentExecutor │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
               ┌──────────────────────────────────────┐
               │  注册到 DAG                          │
               │  dag.AddNode(name, executor,         │
               │              dependencies, policy)   │
               └──────────────────────────────────────┘
```

#### 8.1.1 当前代码分析

当前 `pkg/dagexec/scheduler.go` 中的 `executeComponent()` 仅二路分发，无执行器注册机制：

```go
// 当前代码: 仅 Inline vs Manifest 两条路径 (scheduler.go:326-337)
func (s *Scheduler) executeComponent(
    ctx context.Context,
    phaseCtx *phaseframe.PhaseContext,
    oldCluster, newCluster *bkev1beta1.BKECluster,
    node *topology.ComponentNode,
    tmpl manifest.TemplateContext,
) error {
    if node.Inline != nil {
        return s.executeInline(phaseCtx, oldCluster, newCluster, node)
    }
    return s.executeManifest(ctx, phaseCtx, node, tmpl)
}
```

**问题**：

- 无 Helm/YAML 执行器，Helm 和 YAML 组件类型无法处理
- 分发逻辑硬编码在 if-else 中，新增类型需修改 Scheduler 代码
- 无执行器注册机制，无法按需注入

#### 8.1.2 执行器注册表设计

**设计思路 - 为什么用注册表而非 switch-case**：

当前代码用 `if node.Inline != nil` 硬编码分发，新增组件类型需修改 `executeComponent()` 方法。引入 `ExecutorRegistry` 注册表后，新增类型只需调用 `registry.Register()` 注册新执行器，Scheduler 代码无需修改——符合开闭原则。

注册表还支持按需注入：Feature Gate OFF 时不注册 Helm/YAML 执行器，`registry.Get()` 返回错误，自动回退到旧路径。

```go
// pkg/dagexec/registry.go

// ExecutorRegistry 执行器注册表 (按组件类型注册)
type ExecutorRegistry struct {
    executors map[string]ComponentExecutor
}

// NewExecutorRegistry 创建空注册表
func NewExecutorRegistry() *ExecutorRegistry {
    return &ExecutorRegistry{
        executors: make(map[string]ComponentExecutor),
    }
}

// Register 注册执行器 (按组件类型)
func (r *ExecutorRegistry) Register(componentType string, executor ComponentExecutor) {
    r.executors[componentType] = executor
}

// Get 获取执行器 (未注册返回错误)
func (r *ExecutorRegistry) Get(componentType string) (ComponentExecutor, error) {
    executor, ok := r.executors[componentType]
    if !ok {
        return nil, fmt.Errorf("no executor registered for component type %q", componentType)
    }
    return executor, nil
}

// Has 检查是否已注册某类型
func (r *ExecutorRegistry) Has(componentType string) bool {
    _, ok := r.executors[componentType]
    return ok
}
```

#### 8.1.3 执行器分发实现

**设计思路 - 彻底去除 phaseframe 依赖**：

现有 `Scheduler.ExecuteDAG(ctx, phaseCtx *phaseframe.PhaseContext, ...)` 直接接收 phaseframe 类型，使 `pkg/dagexec` 被 phaseframe 绑定。重构后：

1. `ExecuteDAG` 改为接收 phaseframe-free 的 `ExecutionContext`（由 controllers 层从 `phaseframe.PhaseContext` 构建适配，phaseframe 类型不进入 dagexec 包）。
2. `InlinePhaseRunner`（签名含 `*phaseframe.PhaseContext`）替换为 phaseframe-free 的 `InlineRunner` 接口（见 8.4.3.3），由 controllers 层 `InlinePhaseRunnerAdapter` 桥接。
3. 这样 `pkg/dagexec` 不再 import `pkg/phaseframe`，可独立编译测试。

```go
// pkg/dagexec/scheduler.go 重构 (phaseframe-free)

// ExecuteDAG 执行 DAG (phaseframe-free 入口)
// execCtx 由 controllers 层从 phaseframe.PhaseContext 构建适配后传入
func (s *Scheduler) ExecuteDAG(
    ctx context.Context,
    execCtx *ExecutionContext,
    dag *topology.UpgradeDAG,
) error {
    batches := dag.TopologicalBatches()
    for _, batch := range batches {
        if err := s.executeBatchParallel(ctx, execCtx, batch); err != nil {
            return err
        }
    }
    return nil
}

// executeComponent 三路分发 (Feature Gate ON)
func (s *Scheduler) executeComponent(
    ctx context.Context,
    execCtx *ExecutionContext,
    node *topology.ComponentNode,
    tmpl manifest.TemplateContext,
) error {
    // 1. 解析组件类型 (优先 node 快速路径，回退到 CV 加载)
    componentType := ""
    if t, ok := node.ComponentType(); ok {
        componentType = string(t)
    } else if s.cvStore != nil {
        cv, err := s.cvStore.GetComponentVersion(ctx, node.Component.Name, node.Component.Version)
        if err != nil {
            return fmt.Errorf("failed to load ComponentVersion for %s: %w", node.Component.Name, err)
        }
        componentType = string(cv.Spec.Type)
    }

    // 2. Feature Gate OFF 或类型未注册: 回退到旧路径 (Inline 二路分发)
    executor, err := s.registry.Get(componentType)
    if err != nil {
        return s.executeComponentLegacy(ctx, execCtx, node, tmpl)
    }

    // 3. 填充 TemplateContext
    execCtx.TemplateContext = tmpl
    return executor.ExecuteComponent(ctx, node, execCtx)
}

// executeComponentLegacy 旧路径 (类型未注册时的二路分发，兼容未迁移组件)
func (s *Scheduler) executeComponentLegacy(
    ctx context.Context,
    execCtx *ExecutionContext,
    node *topology.ComponentNode,
    tmpl manifest.TemplateContext,
) error {
    if node.Inline != nil {
        inlineExec := &InlineComponentExecutor{runner: s.inlineRunner}
        return inlineExec.ExecuteComponent(ctx, node, execCtx)
    }
    return s.executeYaml(ctx, execCtx, node, tmpl)
}
```

**controllers 层桥接（phaseframe 仅存在于此）**：

```go
// controllers/capbke/bkecluster_upgrade_dag.go
// buildExecutionContext 从 phaseframe.PhaseContext 构建 dagexec.ExecutionContext
func (r *BKEClusterReconciler) buildExecutionContext(
    ctx context.Context,
    phaseCtx *phaseframe.PhaseContext,            // phaseframe 限定在 controllers 包
    oldCluster, newCluster *bkev1beta1.BKECluster,
    targetClient kubernetes.Interface,
) *dagexec.ExecutionContext {
    return dagexec.NewExecutionContext(
        oldCluster,
        newCluster,
        dagexec.NewBKEComponentStatusUpdater(r.Client),
        phaseCtx.Log,
        phaseCtx.VersionContext,                  // 复用 upgrade.VersionContext
        targetClient,
    )
}
```

#### 8.1.4 Scheduler 初始化与执行器注入

```go
// pkg/dagexec/scheduler.go 扩展

// Config 扩展: 新增 Helm/YAML 执行器依赖
// 注意: InlineRunner 使用 dagexec.InlineRunner 接口 (phaseframe-free)，
//       不再使用 dagexec.InlinePhaseRunner (其签名含 *phaseframe.PhaseContext)
type Config struct {
    InlineRunner             InlineRunner            // phaseframe-free, 由 controllers 适配
    CVStore                  ComponentVersionStore   // 加载 ComponentVersion (类型分发用)
    ManifestStore            manifest.Store
    ManifestApplier          manifest.Applier
    HelmInstaller            *helminstaller.HelmInstaller      // 新增
    YAMLInstaller            *yamlinstaller.YamlInstaller      // 新增 (pkg/yamlinstaller, 对称 Helm)
    ComponentStatusUpdater   ComponentStatusUpdater // 新增 (组件状态更新，所有组件类型使用)
    MaxParallelPerBatch      int
}

// NewScheduler 创建调度器, 按需注册执行器
func NewScheduler(cfg Config) *Scheduler {
    maxParallel := cfg.MaxParallelPerBatch
    if maxParallel == 0 {
        maxParallel = defaultMaxParallelPerBatch
    }

    registry := NewExecutorRegistry()

    // Inline 执行器: 始终注册
    registry.Register("inline", &InlineComponentExecutor{
        runner: cfg.InlineRunner,
    })

    // Helm 执行器: 依赖已注入时注册
    if cfg.HelmInstaller != nil {
        registry.Register("helm", &HelmComponentExecutor{
            installer:              cfg.HelmInstaller,
            cvStore:                cfg.CVStore,
            componentStatusUpdater: cfg.ComponentStatusUpdater,
        })
    }

    // YAML 执行器: 依赖已注入时注册
    if cfg.YAMLInstaller != nil {
        registry.Register("yaml", &YamlComponentExecutor{
            installer:              cfg.YAMLInstaller,
            cvStore:                cfg.CVStore,
            componentStatusUpdater: cfg.ComponentStatusUpdater,
        })
    }

    return &Scheduler{
        inlineRunner:        cfg.InlineRunner,
        cvStore:             cfg.CVStore,
        ManifestStore:       cfg.ManifestStore,
        ManifestApplier:     cfg.ManifestApplier,
        registry:            registry,
        MaxParallelPerBatch: 0, // 0 = defaultMaxParallelPerBatch (8)
    }
}
```

### 8.2 ComponentVersionStore

**设计思路 — 跨执行器共享的 CV 加载抽象**：

`ComponentVersionStore` 是 Helm/YAML 两个 Executor 共用的横切关注点——执行器需要加载 `ComponentVersion` 对象以获取组件类型（`cv.Spec.Type`）和规格（`cv.Spec.Helm`/`cv.Spec.YAML`）。接口定义独立成文件 `pkg/dagexec/component_version_store.go`，与 `scheduler.go` 分离，便于多执行器引用。

```go
// pkg/dagexec/component_version_store.go

// ComponentVersionStore 加载 ComponentVersion (phaseframe-free)
// 由 pkg/manifest.BundleStore 扩展实现 (BundleStore 同时满足 manifest.Store 与本接口)
type ComponentVersionStore interface {
    GetComponentVersion(ctx context.Context, name, version string) (*configv1alpha1.ComponentVersion, error)
}
```

**BundleStore 扩展实现**：

现有 `pkg/manifest/bundle_store.go` 的 `BundleStore` 已持有 `*releasemanifest.Bundle` 且 `GetComponentManifests` 内部已做 `bundle.Components[ComponentKey(name, version)]` 查找（验证 CV 存在）——只是未返回 CV 对象本身。因此直接在 `BundleStore` 上扩展 `GetComponentVersion` 方法，使其同时实现 `manifest.Store` 与 `dagexec.ComponentVersionStore` 两个接口，无需独立类型：

| 接口 | 方法 | 返回 | 用途 |
| ------ | ------ | ------ | ------ |
| `manifest.Store` | `GetComponentManifests` | `*ComponentPackage`（清单字节） | YAML 执行器应用清单 |
| `dagexec.ComponentVersionStore` | `GetComponentVersion` | `*ComponentVersion`（CV 对象） | 类型分发 + Executor 读取 spec |

```go
// pkg/manifest/bundle_store.go 扩展

func (s *BundleStore) GetComponentVersion(context.Context, name, version string) (*configv1alpha1.ComponentVersion, error) {
    if s == nil || s.bundle == nil {
        return nil, fmt.Errorf("release bundle is not initialized")
    }
    key := releasemanifest.ComponentKey(name, version)
    cv, ok := s.bundle.Components[key]
    if !ok {
        return nil, fmt.Errorf("component %s not found in release bundle", key)
    }
    return &cv, nil
}
```

### 8.3 DAG 构建与执行流程

**设计思路**：DAG 调度采用"批次间串行、批次内并行"的执行策略。首先通过拓扑排序将 DAG 分解为多个批次，然后按顺序执行每个批次，批次内的组件可以并行执行。这种策略既保证了依赖关系的正确性，又最大化了并行度。

**关键设计点**：

- **拓扑排序**：使用 Kahn 算法将 DAG 分解为执行批次
- **批次串行**：批次之间严格按顺序执行，确保依赖满足
- **批次并行**：批次内组件通过 errgroup 并行执行，可配置最大并行数
- **失败策略**：支持 FailFast（立即终止）、Continue（继续执行）、Rollback（回滚后继续）

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DAG 调度执行流程                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │  ExecuteDAG()    │
                              │  入口函数         │
                              └────────┬─────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  1. 拓扑排序                          │
                    │  batches := dag.TopologicalBatches() │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  2. 遍历执行批次 (串行)               │
                    │  for _, batch := range batches       │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  3. 执行批次内组件 (并行)              │
                    │  executeBatchParallel(ctx, batch)    │
                    └────────────────────┬─────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
          │  组件 A          │  │  组件 B         │  │  组件 C         │
          │  (并行执行)      │  │  (并行执行)      │  │  (并行执行)     │
          └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                   │                    │                    │
                   └────────────────────┼────────────────────┘
                                        │
                                        ▼
                    ┌──────────────────────────────────────┐
                    │  4. 检查批次结果                      │
                    │  if err != nil                       │
                    └────────────────────┬─────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                         执行成功                执行失败
                              │                     │
                              ▼                     ▼
                    ┌─────────────────┐   ┌─────────────────┐
                    │  继续下一批次    │   │  检查失败策略    │
                    └─────────────────┘   └────────┬────────┘
                                                   │
                                        ┌──────────┼──────────┐
                                        │          │          │
                                        ▼          ▼          ▼
                              ┌─────────────┐ ┌─────────┐ ┌─────────┐
                              │  FailFast   │ │Continue │ │Rollback │
                              │  立即终止    │ │继续执行 │ │回滚后    │
                              │  返回错误    │ │下一批次 │ │继续      │
                              └─────────────┘ └─────────┘ └─────────┘
```

### 8.4 核心接口定义

**设计思路 - 接口分层与解耦**：

当前 `pkg/dagexec` 包依赖 `pkg/phaseframe` 包，导致 DAG 调度器无法独立编译和测试。本节通过两层接口设计实现完全解耦：

1. **上下文层（ExecutionContext）**：替代 `phaseframe.PhaseContext`，携带组件执行所需的全部上下文信息（集群、版本、模板）。Executor 仅依赖此接口，不直接依赖 phaseframe。
2. **执行层（ComponentExecutor）**：统一组件执行器接口，Helm/YAML/Inline 三种执行器各自实现。Scheduler 通过此接口多态分发，不关心具体实现类型。

**接口间关系**：

- `Scheduler` 负责创建 `ExecutionContext`（内含 `VersionContext` + `Cluster` + `TemplateContext`）
- `Scheduler` 根据 `ComponentNode.ComponentType()` 选择对应的 `ComponentExecutor`
- `ComponentExecutor` 从 `ExecutionContext` 获取上下文信息，**自主决定**操作类型（Install/Upgrade/Skip），而非由调用方下达指令
- `VersionContext` 提供版本事实，`ComponentExecutor` 基于事实做决策——这是声明式协调模式的核心

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           核心接口架构                                           │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  Scheduler   │
  │  (调度入口)   │
  └──────┬───────┘
          │
          │ 创建
          ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    ExecutionContext                      │
  │                                                          │
  │  ┌─────────────────┐  ┌────────────┐                     │
  │  │ VersionContext  │  │  Cluster   │                     │
  │  │ (版本事实)       │  │ (集群信息)  │                     │
  │  │                 │  │            │                     │
  │  │ HasCurrent()    │  │            │                     │
  │  │ HasTarget()     │  │            │                     │
  │  │ NeedsUpgrade()  │  │            │                     │
  │  └─────────────────┘  └────────────┘                     │
  │                                                          │
  │  ┌────────────────────────────────────────────────────┐  │
  │  │              TemplateContext                       │  │
  │  │              (模板渲染上下文)                       │  │
  │  └────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────┘
          │
          │ 传递 ExecutionContext
          ▼
  ┌──────────────────────────────────────────────────────────┐
  │              ComponentExecutor (统一接口)                 │
  │                                                          │
  │  ExecuteComponent(ctx, node, execCtx) error              │
  │  GetComponentType() ComponentType                        │
  └──────────────────────────┬───────────────────────────────┘
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
    ┌──────────────┐  ┌────────────┐  ┌────────────┐
    │    Helm      │  │    YAML    │  │   Inline   │
    │ Component    │  │ Component  │  │ Component  │
    │ Executor     │  │ Executor   │  │ Executor   │
    └──────┬───────┘  └─────┬──────┘  └─────┬──────┘
           │                │               │
           ▼                ▼               ▼
   ┌──────────────┐  ┌────────────┐  ┌────────────┐
   │ 从 VC 决定   │  │ 从 VC 决定  │  │ 从 VC 决定 │
   │ Install或    │  │ 是否 Skip   │  │ Install或  │
   │ Upgrade      │  │            │  │ Upgrade    │
   └──────────────┘  └────────────┘  └────────────┘

   数据流: Scheduler → ExecutionContext → ComponentExecutor → Installer
   控制流: ComponentExecutor 根据 VersionContext 自主决定操作类型
```

#### 8.4.1 ExecutionContext 定义

**设计思路 - 为什么用 VersionContext 而非 IsUpgrade bool 或 OperationType 枚举**：

1. **声明式协调**：Kubernetes 控制器应基于"当前状态 vs 期望状态"自主决定操作，而非由调用方显式下达操作指令。VersionContext 提供版本事实（已安装版本、目标版本），Executor 根据 `HasCurrent`/`HasTarget`/`NeedsUpgrade` 自主推断 Install/Upgrade/Skip。
2. **避免概念重复**：`HelmAction` (Install/Upgrade/Rollback) 已在 Executor 中定义。ExecutionContext 中再放 OperationType 枚举会造成两套枚举需要映射，增加维护负担。
3. **扩展性**：后续支持 Rollback 时，只需在 VersionContext 中新增版本历史记录 (`previousVersions map`)，Executor 即可推断 Rollback 操作，无需修改 ExecutionContext 接口。
4. **与实际代码一致**：当前 `PhaseContext` 已使用 `VersionContext` 进行版本判断，设计文档与实现保持一致。

```go
// pkg/upgrade/context.go 扩展

// HasCurrent reports whether a current version is recorded for the component.
func (vc *VersionContext) HasCurrent(name string) bool {
    if vc == nil {
        return false
    }
    vc.mu.RLock()
    defer vc.mu.RUnlock()
    _, ok := vc.Current[name]
    return ok
}

// CurrentVersion returns the current version and whether it exists.
func (vc *VersionContext) CurrentVersion(name string) (string, bool) {
    if vc == nil {
        return "", false
    }
    vc.mu.RLock()
    defer vc.mu.RUnlock()
    v, ok := vc.Current[name]
    return v, ok
}

// TargetVersion returns the target version and whether it exists.
func (vc *VersionContext) TargetVersion(name string) (string, bool) {
    if vc == nil {
        return "", false
    }
    vc.mu.RLock()
    defer vc.mu.RUnlock()
    v, ok := vc.Target[name]
    return v, ok
}
```

```go
// ExecutionContext 组件执行上下文 (完全独立于 phaseframe)
//
// 设计说明:
// - 不引用 phaseframe 任何类型；Inline 路径通过 InlineRunner 接口桥接
//   (见 8.4.3.3)，由 controllers 层提供适配实现
// - OldCluster 供 InlineComponentExecutor 调用 InlineRunner.Execute 时传入
// - TargetClient 供 Helm/YAML 健康检查访问目标集群 (Pod/Endpoint)
type ExecutionContext struct {
    // 旧集群状态 (Inline 执行器需要，对应原 InlinePhaseRunner.Execute 的 oldCluster 参数)
    OldCluster *bkev1beta1.BKECluster

    // 新集群状态 (期望状态)
    Cluster *bkev1beta1.BKECluster

    // 组件状态更新器 (所有组件类型使用，更新组件级状态)
    ComponentStatusUpdater ComponentStatusUpdater

    // 日志记录器
    Log *bkev1beta1.BKELogger

    // 版本上下文 (复用 pkg/upgrade.VersionContext，携带版本事实供 Executor 自主决定操作)
    VersionContext *upgrade.VersionContext

    // 模板上下文 (复用 manifest.TemplateContext)
    TemplateContext manifest.TemplateContext

    // 目标集群 Kubernetes clientset (Helm/YAML 健康检查用)
    // 由 Scheduler 从 manifest.ClusterApplier 复用的远端 client 注入
    TargetClient kubernetes.Interface
}

// NewExecutionContext 创建执行上下文
func NewExecutionContext(
    oldCluster, cluster *bkev1beta1.BKECluster,
    componentStatusUpdater ComponentStatusUpdater,
    log *bkev1beta1.BKELogger,
    versionContext *upgrade.VersionContext,
    targetClient kubernetes.Interface,
) *ExecutionContext {
    ctx := &ExecutionContext{
        OldCluster:             oldCluster,
        Cluster:                cluster,
        ComponentStatusUpdater: componentStatusUpdater,
        Log:                    log,
        VersionContext:         versionContext,
        TargetClient:           targetClient,
    }
    
    // 初始化 TemplateContext
    ctx.TemplateContext = manifest.TemplateContext{}
    
    // 填充基础字段
    if cluster != nil {
        ctx.TemplateContext.ClusterName = cluster.Name
        ctx.TemplateContext.Namespace = cluster.Namespace
        
        // 填充集群扩展字段
        if cluster.Spec.ClusterConfig != nil {
            ctx.TemplateContext.Config = cluster.Spec.ClusterConfig
            
            spec := cluster.Spec.ClusterConfig.Cluster
            ctx.TemplateContext.KubernetesVersion = spec.KubernetesVersion
            ctx.TemplateContext.OpenFuyaoVersion = spec.OpenFuyaoVersion
            ctx.TemplateContext.APIServer = spec.APIServer
            ctx.TemplateContext.ServiceCIDR = spec.Networking.ServiceCIDR
            ctx.TemplateContext.PodCIDR = spec.Networking.PodCIDR
            ctx.TemplateContext.DNSDomain = spec.Networking.DNSDomain
            ctx.TemplateContext.ImageRegistry = spec.ImageRepo.Domain
            ctx.TemplateContext.ImagePullSecret = spec.ImageRepo.AuthSecretRef.Name
        }
    }
    
    return ctx
}
```

#### 8.4.2 ComponentExecutor 接口 (解耦后)

```go
// pkg/dagexec/executor.go

// ComponentExecutor 组件执行器接口 (不再依赖 phaseframe)
type ComponentExecutor interface {
    // ExecuteComponent 执行组件
    ExecuteComponent(ctx context.Context, node *ComponentNode, execCtx *ExecutionContext) error
    
    // GetComponentType 获取组件类型
    GetComponentType() ComponentType
}
```

**设计思路 — Executor 与 Installer 的分层设计**：

每种组件类型都有两层：**Executor（调度层）** 和 **Installer（执行层）**，遵循单一职责原则（SRP）：

| 层级 | 职责 | 关心什么 | 可独立性 |
| ------ | ------ | --------- | --------- |
| **Executor** | DAG 调度 | 什么顺序、失败怎么办（FailFast/Continue/Rollback） | mock Installer 测试调度逻辑 |
| **Installer** | 执行 | 怎么获取 Chart/清单、渲染、部署 | mock HTTP/K8s 测试安装逻辑 |

**各类型 Executor 的特点**：

| Executor | Installer | 单集群执行 | 健康检查 |
| ---------- | ----------- | ----------- | --------- |
| HelmComponentExecutor | HelmInstaller | Helm SDK install/upgrade | PodReady/EndpointReady |
| YamlComponentExecutor | YamlInstaller | K8s API Apply | PodReady/EndpointReady |
| InlineComponentExecutor | (无独立 Installer) | InlineRunner.Execute | Phase 自身逻辑 |

##### 8.4.2.1 HelmComponentExecutor

**设计思路 — Helm 组件无需节点级调度**：

Helm 组件通过 Helm SDK 部署到目标集群（`helm install/upgrade`），不涉及逐节点 SSH 操作。因此 HelmComponentExecutor 无 Rolling/Parallel/Batch 策略，直接调用 `HelmInstaller.Install()` 一次完成。

**与 HelmInstaller 的协作**：

- Executor 负责：获取 ComponentVersion → VersionContext 判断 Install/Upgrade → 读取 `Strategy.Mode` 覆盖 Action → 构建 InstallOptions → 调用 `HelmInstaller.Install()` → FailurePolicy=Rollback 时触发 `helm rollback`
- HelmInstaller 负责：拉取 Chart → 渲染 Values → 执行 helm install/upgrade（含 `--wait`/`--atomic`）→ 健康检查
- 与 Helm 的区别：Helm 无节点级并发（部署到集群而非单节点），但需处理 `Strategy.Mode=Rollback` 和 `FailurePolicy=Rollback` 两种回滚场景

```go
// HelmComponentExecutor Helm 组件执行器
type HelmComponentExecutor struct {
    installer              *helminstaller.HelmInstaller
    cvStore                ComponentVersionStore    // 加载 ComponentVersion
    componentStatusUpdater ComponentStatusUpdater   // 组件状态更新器
}

func (e *HelmComponentExecutor) GetComponentType() ComponentType {
    return ComponentTypeHelm
}

// ExecuteComponent 执行 Helm 组件
func (e *HelmComponentExecutor) ExecuteComponent(ctx context.Context, node *ComponentNode, execCtx *ExecutionContext) error {
    component := node.Component
    
    // 1. 获取 ComponentVersion
    cv, err := e.cvStore.GetComponentVersion(ctx, component.Name, component.Version)
    if err != nil {
        return fmt.Errorf("failed to get component version: %w", err)
    }
    
    // 2. 确认是 Helm 类型
    if cv.Spec.Type != configv1alpha1.ComponentTypeHelm {
        return fmt.Errorf("component %s is not a helm component", component.Name)
    }
    
    // 3. 根据 VersionContext 判断操作类型
    vc := execCtx.VersionContext
    if vc != nil && !vc.NeedsUpgrade(component.Name) {
        execCtx.Log.Info("component %s already at target version, skipping", component.Name)
        return nil
    }
    
    // 4. 标记组件开始安装
    if e.componentStatusUpdater != nil {
        if err := e.componentStatusUpdater.MarkPending(ctx, execCtx.Cluster, component.Name, "helm"); err != nil {
            execCtx.Log.Warn("failed to mark component %s as pending: %v", component.Name, err)
        }
    }
    
    // 5. 确定操作类型 (Strategy.Mode 优先, 为空时由 VersionContext 自动判定)
    action := helminstaller.HelmActionInstall
    switch cv.Spec.Helm.Strategy.Mode {
    case "Rollback":
        action = helminstaller.HelmActionRollback
    case "Install":
        action = helminstaller.HelmActionInstall
    case "Upgrade":
        action = helminstaller.HelmActionUpgrade
    default:
        if vc != nil && vc.HasCurrent(cv.Spec.Name) {
            action = helminstaller.HelmActionUpgrade
        }
    }
    
    // 6. 填充 TemplateContext 的版本信息
    tmpl := execCtx.TemplateContext
    tmpl.ComponentVersion = cv.Spec.Version
    
    // 7. 执行 Helm 操作
    opts := helminstaller.InstallOptions{
        Component:   cv,
        TemplateCtx: tmpl,
        Action:      action,
        Timeout:     cv.Spec.UpgradeStrategy.Timeout,
    }
    
    if err := e.installer.Install(ctx, opts); err != nil {
        // FailurePolicy=Rollback: 升级失败后执行 helm rollback
        if cv.Spec.UpgradeStrategy.FailurePolicy == "Rollback" &&
           action == helminstaller.HelmActionUpgrade &&
           !cv.Spec.Helm.Strategy.Atomic {
            if e.componentStatusUpdater != nil {
                if markErr := e.componentStatusUpdater.MarkRollingBack(ctx, execCtx.Cluster, component.Name, bkev1beta1.ComponentTypeCluster, ""); markErr != nil {
                    execCtx.Log.Warn("failed to mark component %s as rolling back: %v", component.Name, markErr)
                }
            }
            
            execCtx.Log.Warn("helm upgrade failed, attempting rollback: %v", err)
            rollbackOpts := helminstaller.InstallOptions{
                Component:   cv,
                TemplateCtx: tmpl,
                Action:      helminstaller.HelmActionRollback,
                Timeout:     cv.Spec.UpgradeStrategy.Timeout,
            }
            if rbErr := e.installer.Install(ctx, rollbackOpts); rbErr != nil {
                if e.componentStatusUpdater != nil {
                    if markErr := e.componentStatusUpdater.MarkFailed(ctx, execCtx.Cluster, component.Name, bkev1beta1.ComponentTypeCluster, "", rbErr); markErr != nil {
                        execCtx.Log.Warn("failed to mark component %s as rollback failed: %v", component.Name, markErr)
                    }
                }
                return fmt.Errorf("upgrade failed: %w; rollback also failed: %v", err, rbErr)
            }
            
            if e.componentStatusUpdater != nil {
                oldVersion := ""
                if vc != nil {
                    if v, ok := vc.CurrentVersion(component.Name); ok {
                        oldVersion = v
                    }
                }
                if markErr := e.componentStatusUpdater.MarkInstalled(ctx, execCtx.Cluster, component.Name, bkev1beta1.ComponentTypeCluster, "", oldVersion); markErr != nil {
                    execCtx.Log.Warn("failed to mark component %s as rolled back: %v", component.Name, markErr)
                }
            }
            execCtx.Log.Info("helm rollback succeeded after upgrade failure")
            return fmt.Errorf("upgrade failed but rollback succeeded: %w", err)
        }
        
        if e.componentStatusUpdater != nil {
            if markErr := e.componentStatusUpdater.MarkFailed(ctx, execCtx.Cluster, component.Name, bkev1beta1.ComponentTypeCluster, "", err); markErr != nil {
                execCtx.Log.Warn("failed to mark component %s as failed: %v", component.Name, markErr)
            }
        }
        return err
    }
    
    // 8. 标记组件安装成功
    if e.componentStatusUpdater != nil {
        if err := e.componentStatusUpdater.MarkInstalled(ctx, execCtx.Cluster, component.Name, bkev1beta1.ComponentTypeCluster, "", cv.Spec.Version); err != nil {
            execCtx.Log.Warn("failed to mark component %s as success: %v", component.Name, err)
        }
    }
    
    return nil
}
```

##### 8.4.2.2 YamlComponentExecutor

**设计思路 — YAML 组件两层模型（与 Helm 对称）**：

YAML 组件采用两层结构：`YamlInstaller`（`pkg/yamlinstaller` 引擎层，负责清单 Apply + 健康检查，**详见第 5 章**）+ `YamlComponentExecutor`（dagexec 调度层，本节）。本节仅描述调度层 `YamlComponentExecutor`，引擎层接口定义见 **5.3 核心接口定义**。

```go
// pkg/dagexec/yaml_component_executor.go

// YamlComponentExecutor YAML 组件执行器（调度层）
type YamlComponentExecutor struct {
    installer              *yamlinstaller.YamlInstaller
    cvStore                ComponentVersionStore
    componentStatusUpdater ComponentStatusUpdater
}

func (e *YamlComponentExecutor) GetComponentType() ComponentType {
    return ComponentTypeYAML
}

// ExecuteComponent 执行 YAML/Manifest 组件
func (e *YamlComponentExecutor) ExecuteComponent(ctx context.Context, node *ComponentNode, execCtx *ExecutionContext) error {
    component := node.Component

    // 1. 获取 ComponentVersion
    cv, err := e.cvStore.GetComponentVersion(ctx, component.Name, component.Version)
    if err != nil {
        return fmt.Errorf("failed to get component version: %w", err)
    }

    // 2. 确认是 YAML 类型
    if cv.Spec.Type != configv1alpha1.ComponentTypeYAML {
        return fmt.Errorf("component %s is not a yaml component", component.Name)
    }

    // 3. 根据 VersionContext 判断是否需要执行
    vc := execCtx.VersionContext
    if vc != nil && !vc.NeedsUpgrade(component.Name) {
        execCtx.Log.Info("component %s already at target version, skipping", component.Name)
        return nil
    }

    // 4. 标记组件开始安装
    if e.componentStatusUpdater != nil {
        if err := e.componentStatusUpdater.MarkPending(ctx, execCtx.Cluster, component.Name, bkev1beta1.ComponentTypeCluster, ""); err != nil {
            execCtx.Log.Warn("failed to mark component %s as pending: %v", component.Name, err)
        }
    }

    // 5. 委托 YamlInstaller 执行 Apply + 健康检查
    if err := e.installer.Apply(ctx, cv, execCtx); err != nil {
        if e.componentStatusUpdater != nil {
            if markErr := e.componentStatusUpdater.MarkFailed(ctx, execCtx.Cluster, component.Name, bkev1beta1.ComponentTypeCluster, "", err); markErr != nil {
                execCtx.Log.Warn("failed to mark component %s as failed: %v", component.Name, markErr)
            }
        }
        return err
    }

    // 6. 标记组件安装成功
    if e.componentStatusUpdater != nil {
        if err := e.componentStatusUpdater.MarkInstalled(ctx, execCtx.Cluster, component.Name, bkev1beta1.ComponentTypeCluster, "", cv.Spec.Version); err != nil {
            execCtx.Log.Warn("failed to mark component %s as success: %v", component.Name, err)
        }
    }

    return nil
}
```

> 健康检查由 `YamlInstaller.Apply` 内部委托共享 `pkg/healthcheck` 包完成（见 **第 6 章 HealthCheck 共享包设计**）。

##### 8.4.2.3 InlineComponentExecutor

**设计思路 — Inline 是最简执行器，无 Installer、无调度策略**：

Inline 组件通过 `ComponentFactory` 注册的 handler 执行（如 `EnsureMasterInit`/`EnsureWorkerJoin`），是已有 Phase 逻辑的适配层。无制品下载、无模板渲染，直接委托给 `InlineRunner.Execute()`。

```go
// InlineRunner 内联组件执行接口 (不依赖 phaseframe)
type InlineRunner interface {
    Execute(ctx context.Context, oldCluster, newCluster *bkev1beta1.BKECluster, handler, version string) error
}

// InlineComponentExecutor 内联组件执行器
type InlineComponentExecutor struct {
    runner InlineRunner
}

func (e *InlineComponentExecutor) GetComponentType() ComponentType {
    return ComponentTypeInline
}

func (e *InlineComponentExecutor) ExecuteComponent(
    ctx context.Context,
    node *ComponentNode,
    execCtx *ExecutionContext,
) error {
    if node.Inline == nil {
        return fmt.Errorf("component %q has no inline ref", node.Name)
    }

    handler := node.Inline.Handler
    version := node.Inline.Version
    if handler == "" {
        return fmt.Errorf("inline component %q missing handler", node.Name)
    }
    if version == "" {
        version = defaultComponentVersion
    }

    return e.runner.Execute(ctx, execCtx.OldCluster, execCtx.Cluster, handler, version)
}
```

**controllers 层适配实现示例** (phaseframe 类型仅出现在此适配层，不泄漏到 dagexec):

```go
// controllers/capbke/inline_runner_adapter.go
package capbke

// InlinePhaseRunnerAdapter 把 componentfactory.PhaseRunner (依赖 phaseframe)
// 适配为 dagexec.InlineRunner (phaseframe-free)。
type InlinePhaseRunnerAdapter struct {
    phaseCtx *phaseframe.PhaseContext   // phaseframe 限定在 controllers 包内
    runner   *componentfactory.PhaseRunner
}

func NewInlinePhaseRunnerAdapter(phaseCtx *phaseframe.PhaseContext, runner *componentfactory.PhaseRunner) *InlinePhaseRunnerAdapter {
    return &InlinePhaseRunnerAdapter{phaseCtx: phaseCtx, runner: runner}
}

// Execute 实现 dagexec.InlineRunner，桥接回 phaseframe.PhaseContext
func (a *InlinePhaseRunnerAdapter) Execute(ctx context.Context, oldCluster, newCluster *bkev1beta1.BKECluster, handler, version string) error {
    return a.runner.Execute(a.phaseCtx, oldCluster, newCluster, handler, version)
}
```

| 维度 | 修改前 | 修改后 |
| ------ | -------- | -------- |
| **接口依赖** | `phaseframe.PhaseContext` | `dagexec.ExecutionContext` |
| **集群信息** | `phaseCtx.BKECluster` | `execCtx.Cluster` |
| **日志记录** | `phaseCtx.Log` | `execCtx.Log` |
| **操作类型** | `phaseCtx.IsUpgrade` | `execCtx.VersionContext` (携带版本事实，Executor 自主决定操作) |
| **包依赖** | `pkg/dagexec` → `pkg/phaseframe` | `pkg/dagexec` 独立，无 phaseframe 依赖 |

### 8.5 状态模型、幂等性与兼容性设计

#### 8.5.1 状态数据模型

**新增 `LifecyclePhase` 状态枚举**：

```go
// api/bkecommon/v1beta1/lifecycle_phase.go

// LifecyclePhase 生命周期阶段（集群/节点/组件共用）
type LifecyclePhase string

// 组件生命周期
const (
    ComponentLifecyclePending      LifecyclePhase = "Pending"
    ComponentLifecycleInstalling   LifecyclePhase = "Installing"
    ComponentLifecycleInstalled    LifecyclePhase = "Installed"
    ComponentLifecycleUpgrading    LifecyclePhase = "Upgrading"
    ComponentLifecycleRollingBack  LifecyclePhase = "RollingBack"
    ComponentLifecycleUninstalling LifecyclePhase = "Uninstalling"
    ComponentLifecycleRemoved      LifecyclePhase = "Removed"
    ComponentLifecycleFailed       LifecyclePhase = "Failed"
)

// ComponentType 组件类型
type ComponentType string

const (
    ComponentTypeNode    ComponentType = "node"    // 节点级组件
    ComponentTypeCluster ComponentType = "cluster" // 集群级组件
)
```

**新增组件状态字段**（与 kep6-state-machine-v3.md 对齐）：

```go
// api/bkecommon/v1beta1/bkecluster_status.go 扩展

type BKEClusterStatus struct {
    // ... 现有字段保持不变 ...

    // === v3 新增字段 ===

    // ClusterComponentStatuses 记录集群级组件的状态
    // key: componentName -> ComponentLifecycleStatus
    ClusterComponentStatuses map[string]ComponentLifecycleStatus `json:"clusterComponentStatuses,omitempty"`
}

// ComponentLifecycleStatus 组件生命周期状态
type ComponentLifecycleStatus struct {
    // 组件名称
    Name string `json:"name"`

    // 节点 IP（节点级组件必填，集群级组件留空）
    NodeIP string `json:"nodeIP,omitempty"`

    // 组件类型（node/cluster）
    ComponentType ComponentType `json:"componentType"`

    // 生命周期阶段
    Phase LifecyclePhase `json:"phase"`

    // 当前版本
    CurrentVersion string `json:"currentVersion,omitempty"`

    // 目标版本
    TargetVersion string `json:"targetVersion,omitempty"`

    // 最后更新时间
    LastTransitionTime *metav1.Time `json:"lastTransitionTime,omitempty"`

    // 错误信息
    Message string `json:"message,omitempty"`
}
```

**ComponentStatusUpdater 接口**：

```go
// pkg/dagexec/component_status_updater.go

// ComponentStatusUpdater 组件级状态更新接口
type ComponentStatusUpdater interface {
    // MarkPending 标记组件等待安装
    MarkPending(ctx context.Context, cluster *bkev1beta1.BKECluster, componentName string, componentType ComponentType, nodeIP string) error
    
    // MarkInstalled 标记组件安装成功
    MarkInstalled(ctx context.Context, cluster *bkev1beta1.BKECluster, componentName string, componentType ComponentType, nodeIP string, version string) error
    
    // MarkFailed 标记组件失败
    MarkFailed(ctx context.Context, cluster *bkev1beta1.BKECluster, componentName string, componentType ComponentType, nodeIP string, err error) error
    
    // MarkRollingBack 标记组件正在回滚
    MarkRollingBack(ctx context.Context, cluster *bkev1beta1.BKECluster, componentName string, componentType ComponentType, nodeIP string) error
    
    // MarkUninstalling 标记组件正在卸载
    MarkUninstalling(ctx context.Context, cluster *bkev1beta1.BKECluster, componentName string, componentType ComponentType, nodeIP string) error
    
    // MarkRemoved 标记组件已卸载
    MarkRemoved(ctx context.Context, cluster *bkev1beta1.BKECluster, componentName string, componentType ComponentType, nodeIP string) error
}
```

## 9. 完整安装流程详细设计

### 9.1 安装流程图

**设计思路**：完整安装流程从用户创建 BKECluster 开始，经过 ReleaseImage 解析、ComponentVersion 加载、DAG 构建、DAG 执行，最终完成所有组件的安装。流程中 Helm/YAML/Inline 三种类型的组件通过各自的 Executor 并行执行——Helm 组件通过 Helm SDK 部署 Chart，YAML 组件通过 YamlComponentExecutor 将 Kubernetes 清单直接应用到目标集群，Inline 组件通过内联执行器完成 Kubernetes 集群初始化。所有组件安装完成后通过健康检查确认安装成功。

**关键设计点**：

- **声明式安装**：通过 ReleaseImage 声明需要安装的组件列表
- **DAG 调度**：根据组件依赖关系构建 DAG，按拓扑顺序执行
- **多类型支持**：Helm、YAML、Inline
- **健康检查**：安装完成后执行 PodReady/EndpointReady 检查
- **YAML 清单应用**：YAML 类型组件通过 YamlComponentExecutor 应用 Kubernetes 清单，支持 ServerSideApply/Replace/CreateOnly 三种策略

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           完整安装流程                                           │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │  用户创建         │
                              │  BKECluster      │
                              │  desiredVersion: │
                              │  v2.6.0          │
                              └────────┬─────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  BKEClusterReconciler.Reconcile()    │
                    │  检测到新集群创建                     │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  1. 解析 ReleaseImage v2.6.0         │
                    │  releaseImage.GetInstallComponents() │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  install.components:                 │
                      │  ├── coredns/v1.11.1 (helm)          │
                      │  ├── openfuyao-adddon/v26.03 (yaml)  │
                      │  ├── kubernetes-master/v1.29.0       │
                      │  │                   (inline)        │
                      │  └── kubernetes-worker/v1.29.0       │
                      │                      (inline)        │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  2. 加载 ComponentVersion            │
                    │  manifestStore.GetComponentVersion() │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  3. 构建安装 DAG                      │
                    │  BuildInstallDAG(releaseImage)       │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  DAG 结构:                           │
                      │  finalizer → ... → dryrun            │
                      │                   → kubernetes-master│
                      │                     (inline)         │
                      │                   → kubernetes-worker│
                      │                     (inline)         │
                      │                   → coredns (helm)   │
                      │                   → openfuyao-addon  │
                      │                     (yaml)           │
                      │                   → addon            │
                      │                   → postprocess      │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  4. DAG Scheduler 执行               │
                    │  scheduler.ExecuteDAG(ctx, dag)      │
                    └────────────────────┬─────────────────┘
                                         │
               ┌──────────────────────────┼──────────────────────────┐
               │                          │                          │
               ▼                          ▼                          ▼
     ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
     │  Batch 1:       │      │  Batch 2:       │      │  Batch 3:       │
     │  CommonPhases   │      │  DeployPhases   │      │  PostPhases     │
     │  (前置判断)      │      │  (核心部署)     │      │  (后置处理)      │
     └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────────┐
                     │  5. HelmComponentExecutor            │
                     │  执行 coredns 安装                    │
                     └────────────────────┬─────────────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     │                    │                    │
                     ▼                    ▼                    ▼
            ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
            │  拉取 Chart      │  │  渲染 Values    │  │  Helm Install   │
            │  OCI Registry   │  │  模板变量        │  │  --atomic       │
            └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                     │                    │                    │
                     └────────────────────┼────────────────────┘
                                          │
                                          ▼
                      ┌──────────────────────────────────────┐
                      │  6. YamlComponentExecutor            │
                      │  执行 YAML 类型组件安装               │
                      └────────────────────┬─────────────────┘
                                           │
                      ┌────────────────────┼────────────────────┐
                      │                    │                    │
                      ▼                    ▼                    ▼
            ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
            │  获取清单        │  │  解析多文档      │  │  按策略应用     │
            │  ManifestStore  │  │  YAML Parser    │  │  ServerSideApply│
            │  或 URL 下载     │  │  → Unstructured│  │  → K8s Applier  │
            └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                     │                    │                    │
                     └────────────────────┼────────────────────┘
                                          │
                                          ▼
                      ┌──────────────────────────────────────┐
                      │  7. 健康检查                          │
                      │  PodReady + EndpointReady            │
                      └────────────────────┬─────────────────┘
                                          │
                               ┌──────────┴──────────┐
                               │                     │
                          检查通过                检查失败
                               │                     │
                               ▼                     ▼
                     ┌─────────────────┐   ┌─────────────────┐
                     │  安装完成        │   │  返回错误       │
                     │  ClusterStatus  │   │  触发回滚       │
                     │  = Ready        │   │                 │
                     └─────────────────┘   └─────────────────┘
```

## 10. 完整升级流程详细设计

### 10.1 升级流程图

**设计思路**：完整升级流程从用户修改 ClusterVersion 的 desiredVersion 开始，经过版本对比、DAG 构建、DAG 执行，最终完成所有组件的升级。流程中通过对比当前 ReleaseImage 和目标 ReleaseImage 确定需要升级的组件，Helm 组件使用 `helm upgrade --atomic` 确保原子性。

**关键设计点**：

- **版本对比**：对比当前和目标 ReleaseImage 确定升级范围
- **原子升级**：Helm 组件使用 `--atomic` 标志，失败自动回滚
- **失败策略**：支持 FailFast/Continue/Rollback 三种策略
- **YAML 清单升级**：YAML 类型组件通过 ServerSideApply 增量更新，支持 Prune 裁剪不再需要的资源

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           完整升级流程                                           │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │  用户修改        │
                              │  ClusterVersion  │
                              │  desiredVersion: │
                              │  v2.5.0 → v2.6.0 │
                              └────────┬─────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  ClusterVersionReconciler            │
                    │  检测到版本变更                       │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  1. 解析 ReleaseImage v2.6.0         │
                    │  releaseImage.GetUpgradeComponents() │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  2. 解析当前 ReleaseImage v2.5.0      │
                    │  currentReleaseImage                 │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  3. 对比版本，确定需要升级的组件       │
                    │  compareVersions(current, target)    │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  需要升级的组件:                      │
                      │  ├── coredns: v1.10.1 → v1.11.1      │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  4. 构建升级 DAG                      │
                    │  BuildUpgradeDAG(releaseImage)       │
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  DAG 结构:                           │
                      │  provider → coredns (helm)           │
                      │            → etcd (inline)           │
                      │            → kubernetes-worker       │
                      │              (inline)                │
                      │            → kubernetes-master       │
                      │              (inline)                │
                      │            → component → cluster     │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  5. DAG Scheduler 执行 (按拓扑批次)   │
                    └────────────────────┬─────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
          │  Batch 1:       │  │  Batch 2:       │  │  Batch 3:       │
          │  provider       │  │  coredns (helm) │  │  openfuyao-addon│
          │                 │  │  helm upgrade   │  │  (yaml)         │
          │                 │  │  --atomic       │  │  ServerSideApply│
          └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                   │                    │                    │
                   └────────────────────┼────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │  Batch 4: etcd → worker → master     │
                     │  (inline Phase 执行)                 │
                     └────────────────────┬─────────────────┘
                                          │
                                          ▼
                     ┌──────────────────────────────────────┐
                     │  Batch 5: component → cluster        │
                     │  最终健康检查                         │
                     └───────────────────┬──────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                         升级成功                升级失败
                              │                     │
                              ▼                     ▼
                    ┌─────────────────┐   ┌─────────────────┐
                    │  升级完成        │   │  根据策略处理    │
                    │  ClusterStatus  │   │  FailFast/      │
                    │  = Ready        │   │  Continue/      │
                    │                 │   │  Rollback       │
                    └─────────────────┘   └─────────────────┘
```

## 11. 迁移策略详细设计

### 11.1 迁移流程图

**设计思路**：迁移策略通过 Feature Gate 控制新旧两种执行路径。启用 Feature Gate 后，新集群使用 DAG + HelmInstaller 的新路径，旧集群保持原有的硬编码 Phase 路径。兼容层在 reconcile 时根据 Feature Gate 状态选择执行路径，确保平滑迁移。

**关键设计点**：

- **Feature Gate**：通过 `HelmComponentSupport` 控制
- **双轨运行**：新旧路径可以并存，通过 Feature Gate 切换
- **兼容层**：在 reconcile 入口根据 Feature Gate 选择执行路径
- **灰度发布**：可以先在测试环境启用，验证后再推广到生产环境

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           迁移策略流程                                           │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │  Feature Gate    │
                              │  检查            │
                              │  HelmComponent   │
                              │  Support         │
                              └────────┬─────────┘
                                       │
                              ┌────────┴────────┐
                              │                 │
                         启用 (true)        禁用 (false)
                              │                 │
                              ▼                 ▼
                    ┌─────────────────┐ ┌─────────────────┐
                    │  新路径          │ │  旧路径         │
                    │  DAG +          │ │  硬编码 Phase   │
                    │  HelmInstaller  │ │  执行           │
                    └────────┬────────┘ └────────┬────────┘
                             │                   │
                             └─────────┬─────────┘
                                       │
                                       ▼
                      ┌──────────────────────────────────────┐
                      │  兼容层处理                           │
                      │  executeHelmComponent()              │
                      │  {                                   │
                      │    if HelmComponentEnabled(cluster)  │
                      │      return ctrl.Result{}, nil       │
                      │      // DAG helm 节点处理升级         │
                      │    }                                 │
                      │    return rolloutHelm()              │
                      │    // 旧路径: 硬编码 helm 操作       │
                      │  }                                   │
                      └──────────────────────────────────────┘
```

### 11.2 Feature Gate 定义

**设计思路 - 复用现有 `pkg/featuregate` 注解/flag 模式**：

现有 `pkg/featuregate/features.go` 并非 Kubernetes 标准 `featuregate.MutableFeatureGate` 注册表，而是"注解 + 全局 flag"模式：

- `DeclarativeUpgradeEnabled(obj client.Object) bool`：全局 `config.DeclarativeUpgrade` 为 true **或** 对象带 `DeclarativeUpgradeAnnotationKey: "true"` 注解时启用。
- `UpgradeReady(obj client.Object) (string, bool)`：读取 `CVOUpgradeReady` 注解。

本设计沿用同一模式新增 Helm 开关，**不**引入 `featuregate.Enabled(string)` 这种与现有包不符的 API。

```go
// pkg/featuregate/features.go 扩展

const (
    // HelmComponentAnnotationKey 控制是否启用 Helm 组件 (HelmInstaller) 路径
    HelmComponentAnnotationKey = "cvo.openfuyao.cn/helm-component"
)

// HelmComponentEnabled 判断是否启用 Helm 组件路径
func HelmComponentEnabled(obj client.Object) bool {
    if annotations.Has(obj, HelmComponentAnnotationKey) {
        return annotations.Get(obj, HelmComponentAnnotationKey) == "true"
    }
    return config.HelmComponentSupport
}
```

**全局 flag 注册**：在 `utils/capbke/config` 中新增 `HelmComponentSupport` bool 变量，与现有 `DeclarativeUpgrade` 一致，通过控制器启动参数注入。

## 12. 错误处理与恢复

### 12.1 错误处理流程图

**设计思路**：错误处理流程首先对错误进行分类（可重试/不可重试/部分失败），然后根据错误类型和 FailurePolicy 决定后续行为。可重试错误在重试次数未耗尽时自动重试，不可重试错误立即返回，部分失败根据策略决定是继续、终止还是回滚。

**关键设计点**：

- **错误分类**：区分可重试错误（网络超时等）和不可重试错误（配置错误等）
- **重试机制**：可重试错误在重试次数内自动重试，支持指数退避
- **FailurePolicy**：支持 FailFast（立即终止）、Continue（继续执行）、Rollback（回滚后继续）
- **状态记录**：所有错误都会记录到组件状态中，便于排查

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           错误处理与恢复流程                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │  组件执行失败     │
                              └────────┬─────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  1. 错误分类                          │
                    │  classifyError(err)                  │
                    └────────────────────┬─────────────────┘
                                         │
               ┌──────────────────────────┼──────────────────────────┐
               │                          │                          │
               ▼                          ▼                          ▼
     ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
     │  可重试错误      │      │  不可重试错误    │      │  部分失败        │
     │  (网络超时等)    │      │  (配置错误等)    │      │  (部分节点失败)  │
     └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
              │                        │                        │
              ▼                        ▼                        ▼
     ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
     │  检查重试次数    │      │  检查失败策略    │      │  检查失败策略    │
     │  retryCount <   │      │  FailurePolicy  │      │  FailurePolicy  │
     │  maxRetries?    │      │                 │      │                 │
     └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
              │                        │                        │
     ┌────────┴────────┐      ┌────────┴────────┐      ┌────────┴────────┐
     │                 │      │                 │      │                 │
     ▼                 ▼      ▼                 ▼      ▼                 ▼
   是                否    FailFast          Continue  FailFast        Continue
   │                 │      │                 │      │                 │
   ▼                 ▼      ▼                 ▼      ▼                 ▼
 ┌─────────┐  ┌─────────┐ ┌─────────┐  ┌─────────┐ ┌─────────┐  ┌─────────┐
 │ 重试执行 │  │ 返回错误│ │ 立即终止 │  │ 记录错误 │ │ 立即终止 │  │ 记录错误│
 │ retry() │  │ return  │ │ return  │  │ 继续执行 │ │ return  │  │ 继续执行│
 └─────────┘  │  err    │ │  err    │  │ 下一节点 │ │  err    │  │ 下一节点│
              └─────────┘ └─────────┘  └─────────┘ └─────────┘  └─────────┘
```

## 13. 测试设计

### 13.1 单元测试

| 测试模块 | 测试场景 | 覆盖目标 |
| --------- | --------- | --------- |
| **HelmInstaller** | OCI/HTTP/本地 Chart 获取、Values 渲染、Install/Upgrade/Rollback | >85% |
| **YamlInstaller** | Bundle 清单加载/解析、Apply 策略、健康检查 | >85% |
| **HealthCheck** | PodReady/EndpointReady/Custom 检查、超时/重试 | >90% |
| **ExecutionContext** | TemplateContext 初始化、边界条件（nil ClusterConfig） | >90% |
| **HelmComponentExecutor** | Install/Upgrade 流程、FailurePolicy、回滚 | >85% |
| **YamlComponentExecutor** | Apply 流程、幂等性检查 | >85% |

#### 13.1.1 ExecutionContext 测试用例

**测试文件**: `pkg/dagexec/execution_context_test.go`

| 测试名称 | 场景 | 输入 | 验证点 |
| --------- | ------ | ------ | -------- |
| `TestNewExecutionContext_BasicFields` | 正常场景 | 完整的 BKECluster | `ClusterName`、`Namespace`、`KubernetesVersion`、`OpenFuyaoVersion` 正确填充 |
| `TestNewExecutionContext_ClusterExtFields` | 正常场景 | 完整的 BKECluster | `APIServer`、`ServiceCIDR`、`PodCIDR`、`DNSDomain`、`ImageRegistry` 正确填充 |
| `TestNewExecutionContext_NilCluster` | cluster 为 nil | `cluster = nil` | 函数不 panic，`TemplateContext` 仍正确初始化 |
| `TestNewExecutionContext_NilClusterConfig` | ClusterConfig 为 nil | `cluster.Spec.ClusterConfig = nil` | 基础字段留空 |
| `TestNewExecutionContext_ConfigInjection` | 正常场景 | 完整的 BKECluster | `ctx.TemplateContext.Config` 不为 nil，且等于 `cluster.Spec.ClusterConfig` |
| `TestNewExecutionContext_ConfigAccess` | 使用完整配置 | 完整的 BKECluster | 可通过 `ctx.TemplateContext.Config.Cluster.ImageRepo.Domain` 访问配置 |

### 13.2 集成测试

| 测试场景 | 验证内容 | 预期结果 |
| --------- | --------- | --------- |
| **全新安装 (helm)** | coredns + kube-proxy 安装 | Chart 正确部署，Pod Ready，Endpoint Ready |
| **全新安装 (yaml)** | openfuyao-addon 安装 | YAML 清单正确 Apply，资源创建 |
| **升级 (helm)** | coredns v1.10.1 → v1.11.1 | helm upgrade 成功，Pod 滚动更新 |
| **升级 (yaml)** | openfuyao-addon v26.01 → v26.03 | ServerSideApply 增量更新 |
| **回滚 (helm)** | helm upgrade 失败后 rollback | 自动回滚到上一版本 |
| **离线环境** | 无网络时使用本地缓存 | 安装/升级正常完成 |

### 13.3 E2E 测试

| 测试场景 | 集群规模 | 验证内容 |
| --------- | --------- | --------- |
| **小规模安装** | 1 Master + 2 Worker | 完整安装流程，所有组件正常 |
| **中规模安装** | 3 Master + 10 Worker | 并行安装性能，无资源竞争 |
| **跨版本升级** | 3 Master + 5 Worker | v2.5.0 → v2.6.0 完整升级 |
| **升级失败恢复** | 3 Master + 3 Worker | 模拟节点失败，验证 Continue/Rollback 策略 |

## 14. 工作量与任务拆解

### 14.1 工作量评估

| 任务 | 预估工时 | 风险等级 | 依赖 |
| ------ | --------- | --------- | ------ |
| **HelmInstaller 核心实现** | 5 人日 | 中 | 无 |
| **YamlInstaller 核心实现** | 4 人日 | 中 | 无 |
| **ApplyStrategy 引擎实现** | 2 人日 | 中 | YamlInstaller |
| **Prune 裁剪功能实现** | 2 人日 | 中 | ApplyStrategy 引擎 |
| **PreInstallHooks 执行引擎** | 3 人日 | 中 | HelmInstaller |
| **Helm 健康检查实现** | 1 人日 | 低 | HelmInstaller |
| **YAML 健康检查实现** | 1 人日 | 低 | YamlInstaller (复用 Helm 健康检查逻辑) |
| **ComponentVersion CRD 扩展** | 3 人日 | 低 | 无 |
| **VersionContext 扩展方法实现** | 1 人日 | 低 | 无 |
| **VersionContext 与 ExecutionContext 实现** | 3 人日 | 中 | 无 |
| **HelmComponentExecutor 集成** | 3 人日 | 中 | HelmInstaller |
| **YamlComponentExecutor 集成** | 2 人日 | 中 | YamlInstaller |
| **DAG 调度器适配** | 3 人日 | 低 | Executor 集成 |
| **Feature Gate 实现** | 1 人日 | 低 | 无 |
| **兼容层实现** | 3 人日 | 中 | DAG 调度器适配 |
| **错误分类与恢复机制** | 3 人日 | 中 | 核心实现完成 |
| **单元测试** | 5 人日 | 低 | 核心实现完成 |
| **集成测试** | 5 人日 | 中 | 单元测试完成 |
| **E2E 测试** | 8 人日 | 中 | 集成测试完成 |
| **迁移验证** | 2 人日 | 中 | 兼容层实现 |
| **文档编写** | 3 人日 | 低 | 无 |
| **代码审查与修复** | 3 人日 | 中 | 测试完成 |
| **总计** | **63 人日 (约 3 人月)** | | |

### 14.2 Sprint 计划

#### Sprint 1 (第1-2周): HelmInstaller 核心实现

| 任务 | 负责人 | 交付物 |
| ------ | -------- | -------- |
| HelmInstaller 结构定义 | 开发A | `pkg/helminstaller/installer.go` |
| ChartFetcher 实现 | 开发A | OCI/HTTP/本地 Chart 获取 |
| ValuesRenderer 实现 | 开发A | Values 模板渲染 |
| Helm Action Executor 实现 | 开发A | Install/Upgrade/Rollback/Uninstall |
| HealthCheck 实现 | 开发A | PodReady/EndpointReady 检查 |
| 单元测试 (HelmInstaller) | 开发A | 测试覆盖率 >85% |

#### Sprint 2 (第3-4周): YamlInstaller 与 DAG 集成

| 任务 | 负责人 | 交付物 |
| ------ | -------- | -------- |
| YamlInstaller 结构定义 | 开发B | `pkg/yamlinstaller/installer.go` |
| ApplyStrategy 引擎实现 | 开发B | ServerSideApply/Replace/CreateOnly |
| ComponentVersion CRD 扩展 | 开发A | helm/yaml 字段定义 |
| VersionContext 扩展方法 | 开发B | HasCurrent/CurrentVersion/TargetVersion |
| HelmComponentExecutor | 开发A | `pkg/dagexec/helm_component_executor.go` |
| YamlComponentExecutor | 开发B | `pkg/dagexec/yaml_component_executor.go` |
| DAG 调度器适配 | 开发B | 执行器注册与调度 |
| Feature Gate 实现 | 开发A | 开关控制逻辑 |

#### Sprint 3 (第5-6周): 测试与发布

| 任务 | 负责人 | 交付物 |
| ------ | -------- | -------- |
| 兼容层实现 | 开发A | 旧 Phase 兼容逻辑 |
| 集成测试 | 开发A+B | 安装/升级/回滚场景（5 人日） |
| E2E 测试 | 开发A+B | Helm/YAML 场景（8 人日） |
| 迁移验证 | 开发A | 验证清单所有项目通过 |
| 文档编写 | 开发B | 用户文档 + 开发文档 |
| 代码审查与修复 | 开发A+B | 测试完成后的最终修复 |

### 14.3 里程碑

| 里程碑 | 时间 | 交付内容 | 验收标准 |
| -------- | ------ | --------- | --------- |
| **M1: HelmInstaller 完成** | 第2周末 | HelmInstaller 核心功能 + 单元测试 | 单元测试覆盖率 >85% |
| **M2: YamlInstaller + DAG 集成完成** | 第4周末 | YamlInstaller + Executor 集成 + Feature Gate | 集成测试通过 |
| **M3: Beta 发布** | 第6周末 | E2E 测试 + 迁移验证 + 文档 | E2E 通过率 >95% |

## 15. 附录

### 15.1 参考文档

- [Helm Action API](https://pkg.go.dev/helm.sh/helm/v3/pkg/action)

### 15.2 术语表

| 术语 | 定义 |
| ------ | ------ |
| **HelmInstaller** | 负责 Helm Chart 获取、渲染、部署的安装器 |
| **YamlInstaller** | 负责 YAML 清单加载、解析、应用的安装器 |
| **ComponentVersion** | 组件版本 CRD，定义组件的类型、配置、依赖等 |
| **ReleaseImage** | 发布版本清单 CRD，定义安装和升级的组件列表 |
| **DAG** | 有向无根图，用于表示组件依赖关系和执行顺序 |
| **Feature Gate** | 功能开关，用于控制新功能的启用/禁用 |
| **Rolling Update** | 滚动更新，逐节点执行升级操作 |
| **HealthCheck** | 健康检查，验证组件安装/升级后是否正常运行 |

## 16. Story 分解与最小集/增强集拆解

### 16.1 设计原则

**最小集目标**：跑通所有三种组件类型（Inline + Helm + YAML）的端到端路径，含完整的健康检查共享包和 TemplateContext 扩展。验证架构可行性后，仅保留 Hooks、高级错误处理、大规模测试等作为增强迭代。

**选择 Helm 作为首个组件类型**：Helm 组件（coredns）是集群安装的关键路径，且 HelmInstaller 的 Chart 获取/Values 渲染/Action 执行链路最完整，能充分验证框架设计。

### 16.2 原始 Story 规格

以下为完整的 Story 分解，每个 Story 对应一个或多个 Milestone/Enhancement。

#### Story 1：基础框架 + Scheduler 重构 + Inline 路径打通

**对应 Milestone**：M1

**规格**：
- **phaseframe 解耦**：
  - 新增 `ExecutionContext`（phaseframe-free），携带 OldCluster/Cluster/VersionContext/TemplateContext/TargetClient/Log/ComponentStatusUpdater
  - 新增 `InlineRunner` 接口（`Execute(ctx, oldCluster, newCluster, handler, version) error`），替代原 `InlinePhaseRunner`
  - 在 controllers 层实现 `InlinePhaseRunnerAdapter` 桥接，phaseframe 类型仅限 controllers 包
  - 重构 `Scheduler.ExecuteDAG` 签名，接收 `*ExecutionContext` 而非 `*phaseframe.PhaseContext`
- **执行器注册表**：
  - 新增 `ExecutorRegistry`（`Register`/`Get`/`Has`），替代硬编码 if-else 分发
  - 重构 `executeComponent` 为三路分发：注册表获取 → 未注册则回退 `executeComponentLegacy`
- **ComponentExecutor 接口与 Inline 实现**：
  - 定义 `ComponentExecutor` 接口（`ExecuteComponent` + `GetComponentType`）
  - 实现 `InlineComponentExecutor`：委托 `InlineRunner.Execute`，无 Installer
- **ComponentVersionStore**：
  - 定义 `ComponentVersionStore` 接口（`GetComponentVersion(ctx, name, version) (*ComponentVersion, error)`）
  - 扩展 `BundleStore` 实现该接口
- **ComponentLifecycleStatus 状态模型**：
  - 扩展 `BKEClusterStatus` 新增 `ClusterComponentStatuses map[string]ComponentLifecycleStatus`
  - 新增 `LifecyclePhase` 枚举（Pending/Installing/Installed/Upgrading/RollingBack/Uninstalling/Removed/Failed）
  - 新增 `ComponentType` 枚举（node/cluster）
  - 定义 `ComponentStatusUpdater` 接口（MarkPending/MarkInstalled/MarkFailed/MarkRollingBack/MarkUninstalling/MarkRemoved）
  - 实现 `BKEComponentStatusUpdater`（通过 client.Patch 更新 BKECluster.Status）
- **TemplateContext 完整扩展**：
  - 新增字段：Config（*BKEConfig）、APIServer、ServiceCIDR、PodCIDR、DNSDomain、ComponentVersion、ComponentPreviousVersion、ImageRegistry、ImagePullSecret
  - 在 `NewExecutionContext` 中填充所有扩展字段，处理 nil ClusterConfig 边界条件
- **Scheduler 初始化扩展**：
  - `Config` 新增 InlineRunner/CVStore/HelmInstaller/YAMLInstaller/ComponentStatusUpdater/MaxParallelPerBatch 字段
  - `NewScheduler` 按需注册执行器（依赖非 nil 时注册，Feature Gate OFF 时不注册 Helm/YAML）

**验收标准**：
1. `pkg/dagexec` 包不再 import `pkg/phaseframe`，可独立编译测试
2. `ExecutorRegistry.Get()` 对未注册类型返回错误，触发 `executeComponentLegacy` 回退旧路径
3. InlineComponentExecutor 通过 Adapter 正确调用原有 Phase 逻辑
4. ComponentStatusUpdater 正确更新 ClusterComponentStatuses，Patch 操作幂等
5. BundleStore.GetComponentVersion 正确返回 CV 对象，key 不存在时返回明确错误
6. Scheduler 初始化时，依赖为 nil 时不注册对应执行器
7. TemplateContext 扩展字段在 `NewExecutionContext` 中正确填充
8. nil ClusterConfig 时不 panic，基础字段留空
9. 集成测试：Inline 组件（kubernetes-master/worker）执行通过
10. 单元测试覆盖率 ≥85%

#### Story 2：HelmInstaller + HelmComponentExecutor + CRD Helm 扩展

**对应 Milestone**：M2

**规格**：
- **HelmInstaller 核心引擎**：
  - 实现 `HelmInstaller` 结构体（client/clientset/restConfig/cacheDir/httpClient/chartAuth/logger）
  - Chart 获取：复用 `pkg/kube/chart.go` 的 `FetchChartUniversal`/`FetchChartOCI`/`FetchChartTraditional`，支持 OCI/HTTP/本地三种来源
  - Chart 来源字段模板渲染：`renderChartSpec` 渲染 OCI.Repository/Tag/URL 中的模板变量
  - Values 渲染：`renderValues` 递归渲染模板变量（`{{componentVersion}}`/`{{clusterName}}`/`{{imageRegistry}}` 等），支持嵌套 map
  - ValuesFiles 加载与合并：`loadValuesFile` + `mergeValues`（递归合并 map，src 覆盖 dst，非 map 类型整体替换）
  - Helm Action：`install`/`upgrade`/`uninstall`/`rollback`，支持 `--wait`/`--atomic`/`--cleanup-on-fail`/`--timeout`
  - 健康检查：委托 `healthcheck.Run(ctx, clientset, hc)`
- **HelmComponentExecutor**：
  - 持有 HelmInstaller + CVStore + ComponentStatusUpdater
  - 根据 VersionContext 自主决定操作：HasCurrent=false → Install，HasCurrent=true → Upgrade，Strategy.Mode=Rollback → Rollback
  - FailurePolicy=Rollback + atomic=false 时，升级失败自动触发 `helm rollback`
  - 执行前后调用 ComponentStatusUpdater 标记状态（Pending → Installing → Installed/Failed/RollingBack）
- **CRD Helm 扩展**：
  - `ComponentVersionSpec` 新增 `Helm *HelmSpec` 字段（omitempty 指针）
  - 新增类型：`HelmSpec`/`ChartSpec`/`OCIChartSpec`/`HelmStrategySpec`/`RollbackSpec`/`UninstallSpec`/`HookSpec`
  - `ChartSpec.OCI.Repository`/`Tag`/`URL` 支持模板变量
  - 更新 CRD schema（v1alpha1 直接扩展）
  - 重新生成 deepcopy

**验收标准**：
1. HelmInstaller 三种 Chart 来源均能正确拉取并校验 Checksum
2. ChartSpec 来源字段模板变量渲染正确
3. Values 模板变量渲染正确，支持嵌套 map 递归渲染
4. ValuesFiles 合并结果与 Helm `--set`/`-f` 行为一致（递归合并，后者覆盖前者）
5. Install/Upgrade/Uninstall/Rollback 四种 Action 执行正确，参数映射到 Helm SDK 正确
6. `atomic=true` 时 Helm SDK 自动回滚，无需额外调用
7. `cleanup-on-fail=true` 时安装失败自动清理 Release
8. HelmComponentExecutor 根据 VersionContext 自动判定 Install/Upgrade，Strategy.Mode 优先
9. FailurePolicy=Rollback + atomic=false 时，升级失败正确触发 helm rollback，状态标记为 Installed（回滚到旧版本）
10. CRD 扩展后旧 ComponentVersion 反序列化不受影响（Helm 字段为 nil）
11. 集成测试：coredns Helm 组件全新安装 + 升级通过
12. 单元测试覆盖率 ≥85%


#### Story 3：HealthCheck 共享包 + YamlInstaller + YamlComponentExecutor + CRD YAML 扩展 + TemplateContext

**对应 Milestone**：M3

**规格**：
- **HealthCheck 共享包**：
  - 抽取 `pkg/healthcheck` 包，类型定义迁移至 `pkg/healthcheck/types.go`
  - 类型：`HealthCheckSpec`/`HealthCheckItemSpec`/`PodReadyCheckSpec`/`EndpointReadyCheckSpec`/`CustomCheckSpec`
  - 实现 `Run(ctx, client, hc)`：超时重试循环，遍历 Checks，全部通过或超时返回
  - 实现 `checkPodReady`（支持 minReady 部分就绪）、`checkEndpointReady`、`checkCustom`
  - HelmInstaller/YamlInstaller 均委托共享包执行健康检查
- **YamlInstaller 核心引擎**：
  - 实现 `YamlInstaller` 结构体（store + applier + logger）
  - `Apply` 方法：通过 `store.GetComponentManifests` 加载清单（Bundle 文件 + 内联 Resources）→ `applier.ApplyComponent` → 健康检查
  - `DeleteComponent`：GVK 逆序删除（复用 `addonutil.SortUninstallUnstructuredByKind`），使用 `DeletePropagationBackground`，IsNotFound 幂等
  - `PruneResources`：按 label selector 列出资源 → 构建 wantSet → 删除不在 wantSet 中的资源（仅限 `PruneableGVKs` 安全范围，排除 CRD/Namespace）
  - 扩展 `Applier` 接口：新增 `DeleteComponent` + `PruneResources` 方法签名
  - ApplyStrategy 引擎：ServerSideApply / Replace / CreateOnly 三种策略
- **YamlComponentExecutor**：
  - 持有 YamlInstaller + CVStore + ComponentStatusUpdater
  - 根据 VersionContext.NeedsUpgrade 判断是否跳过
  - 委托 `YamlInstaller.Apply` 执行清单应用 + 健康检查
  - 执行前后调用 ComponentStatusUpdater 标记状态
  - 无回滚机制（SSA 天然幂等，重新 Apply 上一版本即可）
- **CRD YAML 扩展**：
  - `ComponentVersionSpec` 新增 `YAML *YAMLSpec` 字段（omitempty 指针）
  - 新增类型：`YAMLSpec`（Namespace/ApplyStrategy/Prune/PruneLabelSelector/HealthCheck）
  - 更新 CRD schema（v1alpha1 直接扩展）

**验收标准**：
1. HealthCheck.Run 超时重试逻辑正确，PodReady 支持 minReady 部分就绪
2. checkEndpointReady 正确检查 Service Endpoint 是否有就绪端点
3. checkCustom 正确执行命令，退出码 0=通过
4. YamlInstaller 能正确加载 Bundle 文件 + 内联 Resources 并应用
5. ServerSideApply 支持幂等更新，Replace 支持删除重建，CreateOnly 仅创建不更新
6. DeleteComponent 按 GVK 逆序删除（CRD 最后），IsNotFound 视为成功
7. PruneResources 仅裁剪 `PruneableGVKs` 安全范围内的资源，不误删 CRD/Namespace
8. YamlComponentExecutor 根据 VersionContext 正确跳过无需升级的组件
9. CRD 扩展后旧 ComponentVersion 反序列化不受影响
10. 集成测试：YAML 组件全新安装/增量更新/Prune 裁剪通过
11. 单元测试覆盖率 ≥85%

#### Story 4：Feature Gate + 兼容层 + 完整流程集成 + 迁移 + 测试

**对应 Milestone/Enhancement**：M4 + E1 + E2 + E3

**规格**：
- **Feature Gate**：
  - 新增 `HelmComponentAnnotationKey = "cvo.openfuyao.cn/helm-component"`
  - 实现 `HelmComponentEnabled(obj)`：全局 `config.HelmComponentSupport` 为 true 或对象带注解 `"true"` 时启用
  - 在 `utils/capbke/config` 注册全局 flag `--helm-component-support`
- **兼容层**：
  - `executeComponent` 中，Feature Gate OFF 或类型未注册时回退 `executeComponentLegacy`（原有 Inline/Manifest 二路分发）
  - Feature Gate ON 时走新路径：注册表获取执行器 → 填充 TemplateContext 版本信息 → 执行
  - 旧集群保持原有硬编码 Phase 路径，新集群走 DAG 路径
- **完整安装流程集成**：
  - BKEClusterReconciler 检测新集群 → 解析 ReleaseImage.GetInstallComponents → 构建安装 DAG → Scheduler.ExecuteDAG
  - Helm/YAML/Inline 三种组件通过各自 Executor 并行执行（批次内并发，批次间串行）
  - 安装完成后健康检查确认，更新 ClusterStatus = Ready
- **完整升级流程集成**：
  - ClusterVersionReconciler 检测版本变更 → 对比当前/目标 ReleaseImage 确定升级范围 → 构建升级 DAG → Scheduler.ExecuteDAG
  - Helm 组件使用 `helm upgrade --atomic` 确保原子性
  - 失败策略：FailFast（立即终止）/Continue（记录警告继续）/Rollback（回滚后继续）正确执行
- **Hooks 执行引擎（增强）**：
  - PreInstallHooks/PreUninstallHooks 执行引擎：Job 类型（创建→等待完成→清理），支持模板变量渲染
  - PreInstallHooks 在 Helm 操作前执行，Job 失败则中断流程
  - PreUninstallHooks 在 Helm Uninstall 前执行
- **高级错误处理与恢复（增强）**：
  - 错误分类：可重试（网络超时）/不可重试（配置错误）/部分失败
  - 可重试错误支持指数退避重试
  - 失败时记录详细错误信息至 ComponentLifecycleStatus.Message
- **迁移策略**：
  - 阶段 1：部署新 CRD 与控制器（Feature Gate 关闭），保持原有逻辑
  - 阶段 2：开启 Feature Gate，新集群走 DAG 路径，旧集群保持旧路径
  - 阶段 3：灰度发布，测试环境验证后推广到生产
- **测试**：
  - 集成测试：Helm 全新安装/升级/回滚、YAML 全新安装/增量更新/Prune、离线环境
  - E2E 测试：小规模安装（1M+2W）、中规模安装（3M+10W）、跨版本升级（v2.5.0→v2.6.0）、升级失败恢复
  - 迁移验证：旧集群 → 部署新 CRD → 开启 Feature Gate → 新流程接管，集群状态一致

**验收标准**：
1. Feature Gate 关闭时，Helm/YAML 执行器未注册，`registry.Get()` 返回错误并回退旧路径
2. Feature Gate 开启时，新路径接管，Helm/YAML 组件通过 DAG 调度执行
3. 注解 `cvo.openfuyao.cn/helm-component: "true"` 可覆盖全局 Flag
4. 集成测试：Helm 组件全新安装、升级、回滚均通过
5. 集成测试：YAML 组件全新安装、ServerSideApply 增量更新、Prune 裁剪均通过
6. 集成测试：离线环境使用本地缓存可正常安装/升级
7. E2E 测试：小规模安装全流程通过，所有组件健康检查通过
8. E2E 测试：中规模安装并行执行无资源竞争
9. E2E 测试：跨版本升级通过，Helm 组件使用 `--atomic` 升级
10. E2E 测试：升级失败后 Continue/Rollback 策略正确执行
11. PreInstallHooks 在 Helm 操作前执行，Job 失败则中断流程并返回错误
12. PreUninstallHooks 在 Helm Uninstall 前执行
13. 迁移验证：旧集群部署新 CRD → 开启 Feature Gate → 新流程接管，集群状态一致
14. E2E 通过率 ≥95%

### 16.3 最小集（4 个 Milestone）

#### Milestone 1：基础框架 + Scheduler 重构 + TemplateContext 扩展 + Inline 路径打通

**目标**：建立执行器框架骨架，phaseframe 解耦，状态模型就位，Scheduler 走新框架，TemplateContext 完整扩展就位，Inline 组件通过注册表分发，兼容层保留旧路径。

| # | 任务 | 说明 |
|---|------|------|
| 1.1 | `ExecutionContext` 定义 | phaseframe-free，携带 OldCluster/Cluster/VersionContext/TemplateContext/TargetClient/Log |
| 1.2 | `InlineRunner` 接口 | `Execute(ctx, oldCluster, newCluster, handler, version) error` |
| 1.3 | `ComponentExecutor` 接口 | `ExecuteComponent` + `GetComponentType` |
| 1.4 | `ExecutorRegistry` | `Register`/`Get`/`Has`，替代硬编码 if-else |
| 1.5 | `ComponentVersionStore` 接口 + `BundleStore` 实现 | `GetComponentVersion(ctx, name, version)` |
| 1.6 | `ComponentLifecycleStatus` 状态模型 | `ClusterComponentStatuses` map、`LifecyclePhase` 枚举、`ComponentType` 枚举、`ComponentStatusUpdater` 接口 |
| 1.7 | `BKEComponentStatusUpdater` 实现 | client.Patch 更新，幂等 |
| 1.8 | `InlineComponentExecutor` 实现 | 委托 `InlineRunner.Execute` |
| 1.9 | `InlinePhaseRunnerAdapter` | controllers 层桥接，phaseframe 仅限 controllers 包 |
| 1.10 | `Scheduler.ExecuteDAG` 签名重构 | 接收 `*ExecutionContext` 替代 `*PhaseContext` |
| 1.11 | `executeComponent` 三路分发 | 注册表获取 → 未注册回退 `executeComponentLegacy` |
| 1.12 | `executeComponentLegacy` 保留 | 原有 Inline/Manifest 二路分发，Feature Gate OFF 时走此路径 |
| 1.13 | Scheduler Config 扩展 | InlineRunner/CVStore/ComponentStatusUpdater 等字段 |
| 1.14 | `NewScheduler` 按需注册 | 依赖非 nil 时注册，Feature Gate OFF 时不注册 Helm/YAML |
| 1.15 | TemplateContext 完整扩展 | 新增 Config/APIServer/ServiceCIDR/PodCIDR/DNSDomain/ComponentVersion/ComponentPreviousVersion/ImageRegistry/ImagePullSecret |
| 1.16 | `NewExecutionContext` 填充扩展字段 | 处理 nil ClusterConfig 边界条件，不 panic |
| 1.17 | Scheduler 执行前填充版本信息 | ComponentVersion/ComponentPreviousVersion 注入 TemplateContext |

**验收标准**：
1. `pkg/dagexec` 不 import `pkg/phaseframe`，可独立编译
2. `ExecutorRegistry.Get()` 未注册类型返回错误，触发 `executeComponentLegacy` 回退旧路径
3. `InlineComponentExecutor` 通过 Adapter 调用原有 Phase 逻辑
4. `ComponentStatusUpdater` Patch 操作幂等
5. `BundleStore.GetComponentVersion` 正确返回 CV 对象，key 不存在时返回明确错误
6. Scheduler 初始化时，依赖为 nil 时不注册对应执行器
7. TemplateContext 扩展字段在 `NewExecutionContext` 中正确填充
8. nil ClusterConfig 时不 panic，基础字段留空
9. Inline 组件通过新注册表路径执行，结果与旧路径一致
10. Feature Gate OFF 时回退旧路径，行为不变
11. 集成测试：Inline 组件（kubernetes-master/worker）执行通过
12. 单元测试覆盖率 ≥85%

#### Milestone 2：HelmInstaller + HelmComponentExecutor + CRD Helm 扩展

**目标**：Helm 组件端到端跑通（Chart 拉取 → 渲染 → 安装/升级 → 健康检查）。

| # | 任务 | 说明 |
|---|------|------|
| 2.1 | CRD Helm 扩展 | `HelmSpec`/`ChartSpec`/`OCIChartSpec`/`HelmStrategySpec` 等类型，v1alpha1 直接扩展 |
| 2.2 | `HelmInstaller` 结构体 | client/restConfig/cacheDir/httpClient/chartAuth/logger |
| 2.3 | Chart 获取 | 复用 `FetchChartUniversal`/`FetchChartOCI`/`FetchChartTraditional`，支持 OCI/HTTP/本地 |
| 2.4 | `renderValues` + `renderTemplateString` | 递归渲染模板变量，支持嵌套 map |
| 2.5 | `renderChartSpec` | OCI.Repository/Tag/URL 支持模板变量 |
| 2.6 | `loadValuesFile` + `mergeValues` | 文件加载 + 递归合并 |
| 2.7 | Helm Action 执行 | install/upgrade/uninstall/rollback，支持 --wait/--atomic/--timeout |
| 2.8 | 健康检查（委托共享包） | 直接委托 `healthcheck.Run`（共享包在 M3 实现，此处先定义接口） |
| 2.9 | `HelmComponentExecutor` | VersionContext 判定 Install/Upgrade，ComponentStatusUpdater 标记状态 |
| 2.10 | FailurePolicy=Rollback | atomic=false 时升级失败触发 helm rollback |

**验收标准**：
1. OCI/HTTP/本地三种 Chart 来源正确拉取
2. Values 模板变量渲染正确（含嵌套 map）
3. ChartSpec 来源字段模板变量渲染正确
4. Install/Upgrade 执行正确，参数映射到 Helm SDK 正确
5. `atomic=true` 时 Helm SDK 自动回滚
6. HelmComponentExecutor 根据 VersionContext 自动判定 Install/Upgrade
7. FailurePolicy=Rollback + atomic=false 时正确触发 helm rollback
8. CRD 扩展后旧 ComponentVersion 反序列化不受影响
9. 集成测试：coredns Helm 组件全新安装 + 升级通过
10. 单元测试覆盖率 ≥85%

#### Milestone 3：HealthCheck 共享包 + YamlInstaller + YamlComponentExecutor + CRD YAML 扩展

**目标**：YAML 组件端到端跑通（清单加载 → Apply → Prune → Delete → 健康检查），HealthCheck 共享包就位。

| # | 任务 | 说明 |
|---|------|------|
| 3.1 | 抽取 `pkg/healthcheck` 包 | 类型定义迁移至 `pkg/healthcheck/types.go` |
| 3.2 | 类型定义 | `HealthCheckSpec`/`HealthCheckItemSpec`/`PodReadyCheckSpec`/`EndpointReadyCheckSpec`/`CustomCheckSpec` |
| 3.3 | `Run(ctx, client, hc)` 实现 | 超时重试循环，遍历 Checks，全部通过或超时返回 |
| 3.4 | `checkPodReady` 实现 | 支持 minReady 部分就绪 |
| 3.5 | `checkEndpointReady` 实现 | 检查 Service Endpoint 就绪端点 |
| 3.6 | `checkCustom` 实现 | 执行命令，退出码 0=通过 |
| 3.7 | HelmInstaller 委托共享包 | 替换 M2 中的临时内联实现 |
| 3.8 | CRD YAML 扩展 | `YAMLSpec` 类型（Namespace/ApplyStrategy/Prune/PruneLabelSelector/HealthCheck），v1alpha1 扩展 |
| 3.9 | `YamlInstaller` 结构体 | store + applier + logger |
| 3.10 | `Apply` 方法 | store 加载清单 → applier.ApplyComponent → 健康检查 |
| 3.11 | `DeleteComponent` 实现 | GVK 逆序删除，DeletePropagationBackground，IsNotFound 幂等 |
| 3.12 | `PruneResources` 实现 | label selector + wantSet 比对，PruneableGVKs 安全范围 |
| 3.13 | 扩展 `Applier` 接口 | 新增 `DeleteComponent` + `PruneResources` 方法签名 |
| 3.14 | `YamlComponentExecutor` | VersionContext 判断 + 委托 Apply + 状态标记 |
| 3.15 | ApplyStrategy 引擎 | ServerSideApply / Replace / CreateOnly 三种策略 |

**验收标准**：
1. HealthCheck.Run 超时重试逻辑正确
2. PodReady 支持 minReady 部分就绪
3. checkEndpointReady 正确检查就绪端点
4. checkCustom 正确执行命令，退出码 0=通过
5. YamlInstaller 正确加载 Bundle 文件 + 内联 Resources 并应用
6. ServerSideApply 幂等更新，Replace 删除重建，CreateOnly 仅创建
7. DeleteComponent 按 GVK 逆序删除，IsNotFound 视为成功
8. PruneResources 仅裁剪 PruneableGVKs 安全范围内的资源
9. YamlComponentExecutor 根据 VersionContext 正确跳过无需升级的组件
10. CRD 扩展后旧 ComponentVersion 反序列化不受影响
11. 集成测试：YAML 组件全新安装/增量更新/Prune 裁剪通过
12. 单元测试覆盖率 ≥85%

#### Milestone 4：Feature Gate + 完整流程集成 + 基础测试

**目标**：灰度发布机制就位，完整安装/升级流程跑通，基础测试覆盖。

| # | 任务 | 说明 |
|---|------|------|
| 4.1 | Feature Gate 实现 | 注解 `cvo.openfuyao.cn/helm-component` + 全局 flag `--helm-component-support` |
| 4.2 | `HelmComponentEnabled(obj)` | 全局 config OR 对象注解，注解优先 |
| 4.3 | 兼容层完善 | Feature Gate OFF → 旧路径；ON → 新路径 |
| 4.4 | 完整安装流程集成 | BKEClusterReconciler → ReleaseImage → DAG → Scheduler.ExecuteDAG |
| 4.5 | 完整升级流程集成 | ClusterVersionReconciler → 版本对比 → 升级 DAG |
| 4.6 | 基础错误处理 | FailFast/Continue 策略，错误信息记录到 ComponentLifecycleStatus.Message |
| 4.7 | 集成测试 | Helm/YAML/Inline 全新安装/升级、离线环境 |
| 4.8 | E2E 测试（小规模） | 1M+2W 完整安装流程 |

**验收标准**：
1. Feature Gate 关闭时 Helm/YAML 执行器未注册，回退旧路径
2. Feature Gate 开启时新路径接管
3. 注解可覆盖全局 Flag
4. 集成测试：Helm/YAML/Inline 全新安装/升级通过
5. 集成测试：离线环境正常安装
6. E2E：小规模安装全流程通过，所有组件健康检查通过
7. 迁移阶段 1 验证：部署新 CRD + 控制器，Feature Gate 关闭，旧逻辑不受影响

### 16.4 增强集（最小集完成后迭代）

#### Enhancement 1：Hooks 执行引擎

| # | 任务 | 说明 |
|---|------|------|
| E1.1 | `HookExecutor` 接口 | 统一 PreInstall/PreUninstall 钩子执行 |
| E1.2 | Job 类型实现 | 创建 Job → 等待完成 → 清理 |
| E1.3 | Hook Manifest 模板渲染 | 支持 `{{clusterName}}`/`{{namespace}}` 等变量 |
| E1.4 | PreInstallHooks 集成 | Helm 操作前执行，失败中断流程 |
| E1.5 | PreUninstallHooks 集成 | Helm Uninstall 前执行 |

#### Enhancement 2：高级错误处理与恢复

| # | 任务 | 说明 |
|---|------|------|
| E2.1 | 错误分类 | 可重试（网络超时）/不可重试（配置错误）/部分失败 |
| E2.2 | 指数退避重试 | 可重试错误自动重试 |
| E2.3 | Rollback 策略完善 | 各组件类型的回滚策略 |
| E2.4 | 部分失败处理 | 记录失败节点，支持手动重试 |

#### Enhancement 3：迁移与大规模测试

| # | 任务 | 说明 |
|---|------|------|
| E3.1 | 迁移阶段 2 | 开启 Feature Gate，新集群走 DAG 路径 |
| E3.2 | 迁移阶段 3 | 灰度发布，生产验证 |
| E3.3 | 迁移验证 | 旧集群 → 新 CRD → Feature Gate → 状态一致 |
| E3.4 | E2E 中规模测试 | 3M+10W 并行安装 |
| E3.5 | 跨版本升级 E2E | v2.5.0→v2.6.0 |
| E3.6 | 升级失败恢复 E2E | Continue/Rollback 策略验证 |

### 16.5 对比总览

| 维度 | 最小集 (M1-M4) | 增强集 (E1-E3) |
|------|---------------|---------------|
| **组件类型** | Inline + Helm + YAML | — |
| **安装/升级** | 三种类型端到端 | — |
| **健康检查** | 共享 pkg/healthcheck（PodReady/EndpointReady/Custom） | — |
| **TemplateContext** | 完整扩展 | — |
| **Hooks** | 无 | + PreInstall/PreUninstall |
| **错误处理** | FailFast/Continue | + 重试/Rollback/部分失败 |
| **Feature Gate** | ✅ 完整 | — |
| **迁移** | 阶段 1（部署新 CRD） | + 阶段 2/3（灰度发布） |
| **测试** | 集成 + 小规模 E2E (1M+2W) | + 中规模 (3M+10W) + 跨版本 + 恢复 |
| **预估工时** | ~25 人日 | ~8 人日 |

### 16.6 Milestone 依赖关系

```
M1 (基础框架 + Scheduler + TemplateContext + Inline)
 │
 ├──→ M2 (HelmInstaller + HelmComponentExecutor)
 │     └──→ M4 (Feature Gate + 集成 + 测试)
 │
 └──→ M3 (HealthCheck + YamlInstaller + YamlComponentExecutor)
       └──→ M4 (Feature Gate + 集成 + 测试)

M2 和 M3 可并行开发，均在 M4 前完成
```

**关键路径**：M1 → M2/M3（并行） → M4
