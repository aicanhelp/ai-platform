---
title: 【安全】基于SBOM的三方件治理报表与汇总产物
ofep-number: 0005
authors:
  - "@openfuyao-security"
owning-sig: security-committee
status: provisional
creation-date: 2026-07-16
reviewers:
approvers:
see-also:
replaces:
---
#
# oFEP-0005: 【安全】基于SBOM的三方件治理报表与汇总产物
#
<!-- toc -->
- [发布签核清单](#发布签核清单)
- [摘要](#摘要)
- [动机](#动机)
  - [目标](#目标)
  - [非目标](#非目标)
- [提案](#提案)
  - [用户故事（可选）](#用户故事可选)
  - [注释/约束/警告（可选）](#注释约束警告可选)
  - [风险与缓解措施](#风险与缓解措施)
- [设计细节](#设计细节)
  - [总体结构](#总体结构)
  - [输入（Trivy SBOM）](#输入trivy-sbom)
  - [关键概念与判定规则](#关键概念与判定规则)
  - [输出产物目录](#输出产物目录)
  - [JSON/CSV 字段定义](#jsoncsv-字段定义)
    - [1) index.json（产物索引）](#1-indexjson产物索引)
    - [2) packages.json（全量三方件清单）](#2-packagesjson全量三方件清单)
    - [3) relations.ndjson.gz（引用关系明细）](#3-relationsndjsongz引用关系明细)
    - [4) package-versions.json（同包多版本汇总）](#4-package-versionsjson同包多版本汇总)
    - [5) report.csv（人工治理报表）](#5-reportcsv人工治理报表)
  - [社区地址获取策略](#社区地址获取策略)
  - [生成与发布流程](#生成与发布流程)
  - [数据质量与一致性约束](#数据质量与一致性约束)
  - [隐私与脱敏](#隐私与脱敏)
  - [测试计划](#测试计划)
      - [先决条件测试更新](#先决条件测试更新)
      - [单元测试](#单元测试)
      - [集成测试](#集成测试)
      - [e2e 测试](#e2e-测试)
  - [毕业标准](#毕业标准)
  - [升级/降级策略](#升级降级策略)
  - [版本倾斜策略](#版本倾斜策略)
- [生产准备情况评审问卷](#生产准备情况评审问卷)
  - [功能启用和回滚](#功能启用和回滚)
  - [推出、升级和回滚规划](#推出升级和回滚规划)
  - [监控要求](#监控要求)
  - [依赖项](#依赖项)
  - [可扩展性](#可扩展性)
  - [故障排除](#故障排除)
- [实施历史](#实施历史)
- [缺点](#缺点)
- [替代方案](#替代方案)
- [所需基础设施（可选）](#所需基础设施可选)
<!-- /toc -->
#
## 发布签核清单
#
标记有（R）的项目在达到里程碑/发布之前是必需的。
- [ ]（R）发布里程碑中的增强问题，链接到 openfuyao/ofep 中的 oFEP 目录
- [ ] (R) oFEP 审批者已批准 oFEP 状态为“可实施”
- [ ] (R) 设计细节已适当记录
- [ ]（R）测试计划已到位，并考虑了 SIG 架构和 SIG 测试的输入
- [ ] (R) 毕业标准已设定
- [ ] (R) 生产准备情况审查完成
- [ ] (R) 生产准备情况审查已获批准
- [ ] “实施历史”部分已更新里程碑
#
## 摘要
#
openFuyao 社区已能使用 Trivy 为各组件/制品生成 SBOM。随着社区规模增长，三方件数量快速增加，包含开发时通过依赖声明引入的库（源码依赖）与运行时/镜像/OS 包维度的依赖（架构依赖）。
#
本提案新增一套“基于 SBOM 的三方件治理汇总信息表”，输出结构化 JSON 与一张 CSV 报表，供人工进行三方件治理分析，满足：

- 3.1 输出三方件全量列表
- 3.2 区分 direct/transitive（直接/间接）引用
- 3.3 汇总同一三方件被引用的多个版本
- 3.4 输出三方件的社区地址（如果有直接数据源可用）
- 3.5 生成一张报表（CSV）供人工分析
#
本提案不包含自动化治理策略与阻断，仅产出事实数据与报表。
#
## 动机
#
### 目标
#
- 将现有 SBOM 能力沉淀为“可复用的治理数据底座”，让人工能快速完成三方件现状盘点与治理优先级排序。
- 输出最关键的治理视角字段：包身份（PURL）、引用关系、直接/间接、版本信息、社区地址、License（如可得）。
- 输出既可人读也可机器消费的产物，后续可扩展联动漏洞（CVE/OSV）与 License 风险。
#
### 非目标
#
- 不在本提案中输出“治理建议”或“自动收敛版本”的结论。
- 不实现门禁、阻断、自动 PR 等自动化治理动作。
- 不要求一次性覆盖所有制品类型；以 Trivy 能稳定产出 SBOM 的范围为准。
#
## 提案
#
定义一个“三方件治理目录”，每次生成一版，包含：

- `packages.json`：三方件全量清单（包维度去重）

  `report.csv`：人工治理报表（单表）
  #

### 用户故事（可选）
#
- 故事 1：安全委员会周期性拉取报表，优先处理“同包多版本”与“扩散面广的间接依赖”，推动版本收敛与替换计划。
- 故事 2：组件维护者发现某个三方件被引入多个版本，查询其出现场景（哪些组件、direct 还是 transitive），评估治理影响面。
#
### 注释/约束/警告（可选）
#
- direct/transitive 的判定优先基于 SBOM 依赖图；当 SBOM 缺失依赖图时，允许输出 `unknown` 并标注。
- 三方件社区地址可能缺失或多值；输出时提供 registry 兜底链接，并在质量标志中体现缺失。
#
### 风险与缓解措施
#



- 风险：输出泄露内部信息（内网仓库、私有镜像仓库地址等）。
  - 缓解：本提案只内部使用，不对外发布
- 经分析，本提案的内容都是在内部使用，不对外发布，不涉及《openFuyao安全设计规范》中要求的检查项。

## 设计细节
#
### 总体结构
#
总体流程（概念）：
- 输入：每个社区组件/制品（subject）的 Trivy SBOM

- 解析：抽取 packages/components 与 dependencies（依赖图）

- 归一化：统一 PURL、生态类型、包身份、版本

- 判定：direct/transitive、依赖深度（如可得）

- 汇总：生成全量清单、关系明细、多版本汇总、CSV 报表

  
### 输入（Trivy SBOM）

输入建议：

- 格式优先：CycloneDX JSON
- 必要信息优先级：
  - 包身份：PURL（若 SBOM 缺失，输出中标注 `purlDerived=true`）
  - 依赖图：dependency graph（用于 direct/transitive、depth）
  - external references：homepage/repository（用于社区地址）

### 关键概念与判定规则

- subject
  - 定义：参与统计的“社区组件/制品”，如某 repo、某镜像、某二进制制品
  
    
  
- third-party package
  - 定义：subject 的 SBOM 中出现的非社区自研包
  - 主键：`packageKey = purl without version`（例如 `pkg:cargo/memchr`）
  
- relation
  - 定义：`subjectId` 引用了某 `packageKey@version`
  
- directness
  - 优先：若 SBOM 有依赖图，root 的 1-hop 为 `direct`，其余可达为 `transitive`
  - 降级：若无依赖图，则输出 `unknown`
  
- multi-version
  - 规则：同一 `packageKey` 出现多个不同 `version` 即为多版本引用

### 输出列表

#### 1)  report.csv（人工治理报表）
#
用途：
- 输出单表，便于人工筛选、排序、透视分析
  #
  列定义：

- 三方件名称

- 三方件版本

- 三方件社区

- 三方件licese

- openFuyao组件

- 引用方式（直接、间接、未知）

- 其他说明

  
  
### 社区地址获取策略
#
优先级：
- 优先从 SBOM external references 获取 `homepage`/`repository`
- 缺失时按生态提供 registry 兜底：
  - cargo：`https://crates.io/crates/<name>`
  - npm：`https://www.npmjs.com/package/<name>`（scope 按实际拼接）
  - pypi：`https://pypi.org/project/<name>/`
  - go：`https://pkg.go.dev/<module>`（name 为 module path 时）
  #
### 生成与发布流程
#
- 触发：建议以“周期性生成（周/月）或里程碑发布”方式生成一版报表，供人工治理会议使用

- 步骤：
  - 收集 SBOM → 解析 → 归一化 → 判定 directness/depth → 汇总 → 校验 → 输出文件
  
  
- 
### 隐私与脱敏
### 测试计划
#
[ ] 我/我们理解，相关组件所有者可能会要求在实现前补充或重构测试。
#
##### 先决条件测试更新
#
- 明确 subject 清单与 SBOM 生成参数，确保重复执行输出稳定。
- 仅内部使用
##### 单元测试


##### 集成测试

##### e2e 测试
能获取社区的软件成分清单报表，信息有效。内部用于治理决策
### 毕业标准
#
- 能稳定生成并交付：`index.json + packages.json + relations.ndjson.gz + package-versions.json + report.csv`

  
### 升级/降级策略
不涉及



### 版本倾斜策略
内部工具，不涉及



- 
## 实施历史
#
- 2026-07-16：形成方案初稿（provisional）。
#
## 缺点
- 
## 替代方案
- 
- 
