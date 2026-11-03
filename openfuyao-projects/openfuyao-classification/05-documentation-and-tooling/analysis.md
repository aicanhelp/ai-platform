# 05 文档、社区、治理与工程工具详细分析

## 1. 归类原则

openFuyao 作为开源/企业项目，仓库中必然包含大量非运行时组件：

- 社区治理与贡献指南
- 设计提案（Enhancement Proposals）
- 文档与示例
- 自动化测试与质量工具
- 开发者体验和发布管理脚本

这些内容与 01~04 的功能代码不在同一抽象层，因此单独归为一类，避免混淆对 openFuyao 运行时的分析。

## 2. 各子目录说明

| 目录 | 类型 | 主要内容 |
|---|---|---|
| `community` | 社区治理 | 贡献指南、会议记录、组织流程。 |
| `docs` | 项目文档 | 面向开发者和用户的总览文档。 |
| `ofep` | 增强提案 | openFuyao Enhancement Proposals，记录新功能设计。 |
| `security-committee` | 安全治理 | 漏洞报告、安全流程、委员会职责。 |
| `tc` | 技术治理 | 技术指导委员会章程等。 |
| `sig-ai-inference` | SIG | AI 推理专项兴趣组。 |
| `sig-colocation` | SIG | 混部/协同调度兴趣组。 |
| `sig-container-platform` | SIG | 容器平台兴趣组。 |
| `sig-distributed-framework` | SIG | 分布式框架兴趣组。 |
| `sig-installation` | SIG | 安装器兴趣组。 |
| `sig-large-scale-cluster` | SIG | 大规模集群兴趣组。 |
| `sig-numa-affinity` | SIG | NUMA 亲和兴趣组。 |
| `sig-orchestration-engine` | SIG | 编排引擎兴趣组。 |
| `sig-qa` | SIG | 质量保障兴趣组。 |
| `sig-ub-enable` | SIG | UB 能力兴趣组。 |
| `e2e-auto-test` | 测试 | 端到端自动化测试用例与框架。 |
| `qa-tool` | 测试/工具 | 辅助质量保障的脚本/工具。 |
| `developer-experience` | 开发者工具 | 开发环境、示例、脚手架。 |
| `dependency-database` | 工程工具 | 依赖版本扫描/数据库。 |
| `release-management` | 发布管理 | 版本规划、发布规则、报告。 |

## 3. 与功能分类的关系

- **01 安装部署类**：`sig-installation`、`release-management` 与安装器流程存在关联；`dependency-database` 影响安装器依赖版本管理。
- **02 通用基础平台类**：`sig-numa-affinity`、`sig-colocation`、`sig-orchestration-engine`、`sig-distributed-framework`、`sig-ai-inference` 与 Volcano、Ray、NUMA 等通用能力相关。
- **03 昇腾专有类**：各 SIG 中可能会有昇腾相关的讨论或示例，但 SIG 本身保持厂商中性。
- **04 上游依赖类**：`docs`、`community` 中可能引用上游项目说明。

## 4. 复制范围

本次完整复制了这些目录（`.git` 除外），保留了原目录结构。
