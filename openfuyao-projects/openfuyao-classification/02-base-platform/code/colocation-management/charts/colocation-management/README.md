# UPGRADE Guide: Colocation Management

本文档指导您如何从旧的 `overquota-pod-management` 和 `overquota-resource-management` Chart 升级到新的统一 Chart `colocation-management`。本次升级涉及重大变更，请仔细阅读并遵循所有步骤。

## 从 `overquota-pod-management` / `overquota-resource-management`  升级到 `colocation-management` 

本次发布是一个**主要版本**，旨在将两个独立的chart包整合为一个统一的包协同管理。因此，配置结构发生了**破坏性变更（Breaking Changes）**，旧的 Values 配置将无法直接兼容。

### 概述
*   **新 Chart 名称:** `colocation-management`
*   **旧 Chart 名称（已废弃）:** `overquota-pod-management`, `overquota-resource-management`
*   **主要变更:** 配置结构重构与合并，移除 `oversubscription-config`configmap，使用全新的crd资源。

---

### 重大变更 (Breaking Changes)

#### 1. 配置结构重构
旧的、分散的配置结构已被一个新的、统一的层级结构所取代，所有配置现在都集中在 `.Values.management` 下。

**旧配置 (必须修改):**

```yaml
# 旧 values.yaml 示例 (来自 overquota-pod-management)
configMap:
  enabled: true
  logDriver: "stdio"
  logDir: "/var/log/oversubscription"
  logSize: "1024"
  logLevel: "info"
  cgroupRoot: "/sys/fs/cgroup"
  cgroupDriver: "systemd"
  numaAffinityEnabled: "false"
  name: "oversubscription-config"


# 来自 overquota-resource-management 相关的配置
master:
  pprof:
    port: 8090
  logLevel: 3
agent:
	pprof:
    port: 8090
  logLevel: 4
  
```

**新配置（必须使用）:**
```yaml
# values.yaml
management:
	manager:
    config:
      pprof:
        port: 8090
      logLevel: 3
  agent:
    config:
      pprof:
        port: 8090
      logLevel: 4
      cgroupRoot: "/sys/fs/cgroup"
      cgroupDriver: "systemd" 
```

#### 2. 移除oversubscription-config configmap
oversubscription-config 这个配置项已被完全移除。其功能已被整合到新的 .Values.management部分中。您必须将相关配置迁移到新的键下。

####  3. 使用新的crd对象

旧的crd对象已经被全新的crd对象取代

**旧自定义资源**

```yaml
Kind: ContainerCheckpoint  
Group: oversubscription.openfuyao.com
```

**新自定义资源**

```yaml
Kind: ContainerCheckpoint  
Group: colocation.openfuyao.com
```



### 升级前准备工作 (必须执行)

#### 1. 备份当前配置和状态：

备份您当前使用的 values.yaml 文件。

考虑对相关的 Kubernetes 资源（如 ConfigMaps）进行备份。

（重要） 如果组件涉及自定义编辑的内容，请确保您有可用的备份和回滚计划。

#### 2.审查新 Chart 的 Values：
使用 helm show values 命令查看新 Chart 的全部可配置选项，熟悉新的结构
```
helm show values colocation-management-0.0.0-latest.tgz
# 或者，如果已添加到仓库
helm show values colocation(您配置的仓库名)/colocation-management --version 0.0.0-latest
```

### 故障排除与回滚
升级过程中，如果出现任何问题，请务必检查错误信息并尝试修复。如果升级失败，请使用 Helm rollback 命令回滚到上一个版本。

### 附加说明
Hooks 和 CRDs： 旧 Chart 使用了overquota-webhook的MutatingWebhookConfiguration 对象，请确认新 Chart 保留了这些钩子 更名为colocation-manager-webhook，并注意是否需要替换。同时，检查 Custom Resource Definitions (CRDs) 的变更。

依赖变更： 新 Chart 的依赖项已发生变化，请一并检查。

---
如果您遇到任何本指南未涵盖的问题，请查阅 Chart 的 README.md 或联系维护团队。
