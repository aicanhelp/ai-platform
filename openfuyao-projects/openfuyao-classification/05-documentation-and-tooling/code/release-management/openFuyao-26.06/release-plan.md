# Version Info

openFuyao 26.06 是26年Q2的社区发布版本，本版本的内核是基于k8s 1.34.1增强定制版构建的，保持和v25.12/v26.03版本内核一致，本版本主要聚焦于内核增强和特性孵化，同时支持内核bugfix，具体如下：

## Release Plan

| Stage Name | Deadline for PR | Begin Time | End Time | Days | Note |
| --- | --- | --- | --- | --- | --- |
| Collect key features | - | 2026/3/01 | 2026/4/30 | 61 | 版本需求收集 |
| Change Review | - | 2026/4/1 | 2026/5/20 | 50 | Review 组件变更（升级/退役/淘汰） |
| Feature Develop&Test | - | 2026/3/25 | 2026/5/25 | 60 | 特性合入和特性验证期（从前一个版本让出main或master分支后开始） |
| 明确开源三方件配套版本 | - | 2026/4/15 | 2026/4/25 | 10 | 明确&切换开源三方件版本 |
| Alpha Version | 2026/5/9 | 2026/5/10 | 2026/5/12 | 3 | Alpha版本发布(rc.1) |
| Beta Version | 2026/5/26 | 2026/5/27 | 2026/5/29 | 03 | Beta版本发布(rc.2), 基本上所有新特性上车 |
| 全量测试期 | 2026/6/1 | 2026/6/1 | 2026/6/15 | 15 | 全量验证(全量SIT) |
| RC版本 | 2026/6/15 | 2026/6/17 | 2026/6/19 | 03 | RC版本验证与发布(rc.3) |
| bugfix刷新版(预留) | 2026/6/23 | 2026/6/24 | 2026/6/25 | 02 | 回归测试 |
| Branch | - | 2026/6/20 | 2026/6/21 | 02 | 拉取发布分支 |
| Release Review | - | 2026/6/25 | 2026/6/25 | 01 | 版本发布评审 |
| Release preparation | - | 2026/6/26 | 2026/6/28 | 03 | 发布前准备阶段，发布件系统梳理 |
| Release | - | 2026/6/29 | 2026/6/30 | 02 | 社区Release评审通过正式发布 |

## 代码合入说明

项目采用oneTrack方式开发，主程序基于Main分支构建，个别组件有分支诉求的，可在具体特性需求中标明构建分支

## Feature list

状态说明/status：Discussion(方案讨论中，尚未拿到TC或SIG组maintainer接纳结论)、Developing(开发中)、Testing(测试中)、Accepted(已验收)

发布方式：IMG、OBS等，成熟度目标是指该特性在本版本的交付目标，参考《特性发布成熟度管理规范》中定义，请注意特性在本版本从孵化期转成熟期也需要登记，测试需要以目标成熟度来验收。

| ofep link | feature | status | sig | owner | branch | 发布方式 | 涉及软件包列表 | 成熟度目标 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|样例： [ofep-0033](https://gitcode.com/openFuyao/ofep/pull/60) | 基于时间因素的潮汐调度决策算法 | Developing | sig-ai-inference | 陈锌鸿 | / | IMG | elastic-scaler | |
|  |  |  |  |  |  |  |  | |

## 需求/特性反馈基本流程

1、开发者/sig在本文件表格中填写要合入本版本的需求/特性，并同时注意在release Management仓中新建需求issue（请在收集截止时间前提交）

2、申请在TC例会上评审需求（owner或者SIG maintainer参会）
