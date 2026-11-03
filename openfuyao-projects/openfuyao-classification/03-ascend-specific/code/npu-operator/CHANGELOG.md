# Changelog

> ⚠️ **注意**：历史版本记录系由 git 历史（tag 与 release 分支）尽力还原，未经逐一核实，欢迎在评审中补充与修正。(2026/09/23)

本文件记录 npu-operator 各版本的用户可感知变更。维护规则见 [docs/README.md](docs/README.md) 第 2、3 节：行为变更在合入时写入 Unreleased 段，发布（打 tag）时将 Unreleased 段定稿为对应版本号。

## 格式约定

- 每个版本一个二级标题：`## [版本号] - YYYY-MM-DD`，新版本在前；
- 版本内按变更类型分组：`Features` / `Bug Fixes` / `Docs` / `Chore`；
- 每条变更一行，注明涉及的组件或 CRD 字段，已知 MR 编号的以 `!N` 标注；
- 仅记录用户可感知的变更；纯内部重构、CI 调整等可归入 `Chore` 或不记录。

---

## [Unreleased]

> 面向 26.9.0，已发布预Release版本 26.9.0-rc.1（2026-08-14）与 26.9.0-rc.2（2026-09-03）。

### Features

- 新增 vNPU 与 DRA 组件及 `NPUClusterPolicy` 对应配置段（`vnpu` / `dra`），含 soft/hard vNPU 算力切分、DRA webhook 与 kubelet-plugin 清单
- 新增组件冲突检测（ComponentConflict），检测到冲突时回写 Error 并停止调和

### Bug Fixes

- 支持 vNPU 与 MindCluster 之间 Volcano 调度器的切换
- 适配 Volcano scheduler plugin 新架构（!117）
- 修复 DRA `DeviceClass` 与 soft-dra 资源状态未清理的问题
- operator 自身 `nodeSelector` 可配置
- 修复 `getInstance` / `updateStatus` 中错误链丢失的问题

## [26.6.0] - 2026-06-23

> 版本号方案由 1.x 切换为 CalVer（YY.MM）。预发布：26.6.0-rc.2（2026-06-05）、26.6.0-rc.3（2026-06-16）、26.6.0-cve（2026-06-18）。

### Features

- 新增英文版 README（README-en.md）

### Bug Fixes

- 修复 NodeD enable 标签未清理的问题
- 修复 `/var/log/mindx-dl/devicePlugin`、`/var/log/mindx-dl/clusterd` 报 "not a directory" 错误的问题
- 修复文档失效链接

### 备注

- tag 之后 release-26.6.0 分支追加了 operator Deployment 模板缩进与滚动升级策略修复（!107），待随下个补丁版本发布
- release-1.2.x 分支在 1.2.0 tag 后同样追加了日志目录修复与 chart 版本更新（!102）

## [1.2.0] - 2026-03-05

### Features

- 支持组件版本 7.3.0，values.yaml 默认值同步更新

### Bug Fixes

- 修复环境指标（Environmental Indicators）未去重的问题

### Chore

- 更新 Chart description 与 keywords

## [1.1.1] - 2026-01-07

### Bug Fixes

- 修复集群中不含 NPU 卡的节点的安装与判断逻辑问题（!94）

### Chore

- npu-feature-discovery 镜像 tag 更新为 1.1.1

## [1.1.0] - 2025-12-24

### Features

- 新增多项 CRD 配置项（含 device-plugin volumes 等），并同步组件 YAML
- 支持 910C 芯片识别
- 驱动/固件支持自定义读取路径；新增 mindio API 配置

### Bug Fixes

- 修复 initContainer 无法更新 env 的问题

### Chore

- 适配 mindcluster 7.2.RC1（镜像与 values.yaml）

## [1.0.1] - 2025-12-02

> 版本号方案由 openFuyao-vYY.MM 切换为 1.x，首个 1.x 版本

### Features

- 组件版本升级至 7.1.RC1，新增可配置参数
- npu-exporter 更新为重新构建的补丁版本，修复高版本 containerd 接口不可用的问题
- 更新第三方镜像仓库

### Chore

- codecheck 与单元测试修复

## [v25.09] - 2025-12-02

### Features

- 镜像与 Chart 仓库迁移至 openfuyao.cn
- device-plugin 暴露 command / args 配置入口

### Bug Fixes

- 修复 NFD 镜像引用错误
- 更新存在安全漏洞的第三方库版本

### Docs

- README 更新（含许可证说明）

## [v25.06] - 2025-06-30

### Features

- 支持离线安装

### Bug Fixes

- 修复 SCA（软件成分分析）扫描问题

### Chore

- busybox 镜像 tag 更新为 latest

## [v25.03] - 2025-04-01

### Features

- CRD group/名称确定为 `npu.openfuyao.com`，新增与调整多项配置项
- 适配多架构（ARM / x86）组件安装
- 支持屏蔽 tpl 文件配置与离线部署
- volcano 组件支持资源配置（resource requirements）
- 调整 `NPUClusterPolicy` 状态管理逻辑
- 增加 driver 日志

### Bug Fixes

- 修复 Ubuntu 下 runtime endpoint 软链接到绝对路径导致文件找不到的问题
- 补充容器 volume mount 以保证 nsenter 可用
- 修复 npu-feature-discovery Helm chart
- 修复 npu-exporter 容器模式问题

### Chore

- 组件镜像 tag 更新为 v25.03；补充大量单元测试

## [v24.12] - 2025-01-10

### Features

- 组件增加 `openfuyao-component` 标签

### Bug Fixes

- 修复驱动重安装问题
- 修复第三方依赖问题

## [v24.09] - 2024-11-01

> 首个发行版本

### Features

- 初始发布：基于 `NPUClusterPolicy` CR 的 Ascend NPU 全栈组件声明式部署（驱动、device-plugin 等）
- 新增 NFD（Node Feature Discovery）Helm 子图表与节点特征规则
- 新增 driver 组件及对应 CRD 字段
- 支持 nodeSelector 配置

### Bug Fixes

- 组件镜像地址与拉取策略、RBAC、日志目录创建、标签更新、驱动组件安装顺序等多项修复

### Chore

- 建立单元测试；启用生产配置并将清单打入镜像
