
# Version Info

openFuyao 25.12 是首个LTS版本，本版本的内核是基于k8s 1.33.1增强定制版构建的

## Release Plan

| Stage Name                    | Deadline for PR | Begin Time | End Time   | Days | Note                                     |
| ----------------------------- | --------------- | ---------- | ---------  | ---- | ---------------------------------------- |
| Collect key features          |        -        | 2025/9/01 | 2025/10/31 | 61 | 版本需求收集                              |
| Change Review                 |        -        | 2025/10/09 | 2025/11/30 | 51 | Review 组件变更（升级/退役/淘汰）  |
| Feature Develop&Test          |        -        | 2025/9/19 | 2025/11/28 | 70 | 特性合入窗口期（从前一个版本让出master分支后开始） |
| 明确开源三方件配套版本          |        -        | 2025/10/15 | 2025/10/25 | 10 | 明确开源三方件版本|
| 三方件切换 & Alpha版本         |    2025/10/27   | 2025/10/27 | 2025/11/15 | 20 | 组件切换，Alpha版本发布(rc.1)    |
| Beta Version                  |    2025/11/23   | 2025/11/24 | 2025/11/30 | 05 | Beta版本发布(rc.2)    |
| 全量测试期                     |    2025/11/28   | 2025/11/29 | 2025/12/19 | 21 | 全量验证(全量SIT)  |
| RC版本                        |    2025/12/12   | 2025/12/13 | 2025/12/19 | 07 | RC版本验证与发布(rc.3)          |
| bugfix刷新版(预留)             |    2025/12/19   | 2025/12/20 | 2025/12/24 | 05 | 回归测试                 |
| Branch release-25.12          |    -            | 2025/12/19 | 2025/12/20 | 01 | Master 拉取 25.12 发布分支|
| Release Review                |        -        | 2025/12/22 | 2025/12/26 | 05 | 版本发布评审        |
| Release preparation           |        -        | 2025/12/24 | 2025/12/26 | 03 | 发布前准备阶段，发布件系统梳理    |
| Release                       |        -        | 2025/12/28 | 2025/12/30 | 03 | 社区Release评审通过正式布       |
|||||||

## 代码合入说明

项目采用oneTrack方式开发，主程序基于Main分支构建， 个别组件有分支诉求的，可在具体特性需求中标明构建分支

// 新特性代码请及时合入版本&自验证，跟随整体计划赶在第一轮转测试

## Feature list

状态说明/status：Discussion(方案讨论中，尚未拿到TC提案接纳结论)、 Developing(开发中)、 Testing(测试中)、 Accepted(已验收)

发布方式：IMG、OBS等

| ofep link | feature | status | sig | owner | branch | 发布方式 | 涉及软件包列表    |
|-----------| ------- | ------ | ----|-------| ------ |------|------------|
| ofep-0018 |基于Mooncake实现共享内存语义池化改造|Discussion|sig-ai-inference|王雨欣|/| IMG | mooncake |
| ofep-0023 |大规模集群KCM快速倒换、pod批量创建和绑定、access-log能力合入社区主干分支|Discussion|sig-large-scale-cluster| 边琛    |of-v1.33.1| IMG  | kubernetes |
| / |k8s支持1.5s内UB的容器热迁移，构建装箱优化、pod重调度、动态超卖率等调度能力|Discussion|sig-orchestration-engine| 边琛 |master| IMG  | pod-live-migration， mce-k8s |
| ofep-0024 |KVCache存储Mooncake Store优化|Discussion|sig-ai-inference|陈文潇、张仕畅|client优化: v0.3.7.dev; cache tier昇腾适配: CaceTier | client优化: IMG; cache tier昇腾适配: SourceCode| mooncake |
| ofep-0025 |AI推理智能路由增强|Discussion|sig-ai-inference|郑钦文、李乐奇|master|IMG| hermes-router|
|ofep-0020|openFuyao支持大版本升级能力 |Discussion|sig-installation|徐尤红|master|OBS,IMG|bkeadm, bkeagent, provider|
|oFEP-0019|基于Go语言的社区发行版自动化测试体系|Discussion|sig-QA|朱管瑜、杨煦庭|master|code|e2e-auto-test|
| ofep-0026 | AI推理可观测                                      |Discussion|sig-ai-inference| 朱雪仪     |master|IMG| eagle-eye    |

## 需求/特性反馈基本流程

1、开发者/sig在本文件表格中填写要合入本版本的需求/特性，并同时注意在release Management仓中新建需求issue （请在收集截止时间前提交）

2、申请在TC例会上评审需求 （owner或者SIG maintainer参会）

3、owner参加RM Sig例会，按照版本计划按时完成合入和验证
