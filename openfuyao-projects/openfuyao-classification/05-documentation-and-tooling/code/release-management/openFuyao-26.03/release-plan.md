# Version Info

openFuyao 26.03 是26年Q1的社区发布版本，本版本的内核是基于k8s 1.34.1增强定制版构建的

## Release Plan

| Stage Name | Deadline for PR | Begin Time | End Time | Days | Note |
| --- | --- | --- | --- | --- | --- |
| Collect key features | - | 2025/12/01 | 2026/1/31 | 62 | 版本需求收集 |
| Change Review | - | 2026/1/1 | 2026/2/28 | 59 | Review 组件变更（升级/退役/淘汰） |
| Feature Develop&Test | - | 2025/12/25 | 2026/2/25 | 60 | 特性合入和特性验证期（从前一个版本让出main或master分支后开始） |
| 明确开源三方件配套版本 | - | 2026/1/15 | 2026/1/25 | 10 | 明确&切换开源三方件版本 |
| Alpha Version | 2026/2/9 | 2026/2/10 | 2026/2/12 | 3 | Alpha版本发布(rc.1) |
| Beta Version | 2026/2/25 | 2026/2/26 | 2026/2/28 | 03 | Beta版本发布(rc.2) |
| 全量测试期 | 2026/2/25 | 2026/2/26 | 2026/3/13 | 16 | 全量验证(全量SIT) |
| RC版本 | 2026/3/10 | 2026/3/11 | 2026/3/13 | 03 | RC版本验证与发布(rc.3) |
| bugfix刷新版(预留) | 2026/3/18 | 2026/3/19 | 2026/3/20 | 02 | 回归测试 |
| Branch | - | 2026/3/19 | 2026/3/20 | 02 | 拉取发布分支 |
| Release Review | - | 2026/3/23 | 2026/3/25 | 03 | 版本发布评审 |
| Release preparation | - | 2026/3/26 | 2026/3/29 | 04 | 发布前准备阶段，发布件系统梳理 |
| Release | - | 2026/3/30 | 2026/3/31 | 02 | 社区Release评审通过正式发布 |

## 代码合入说明

项目采用oneTrack方式开发，主程序基于Main分支构建，个别组件有分支诉求的，可在具体特性需求中标明构建分支

## Feature list

状态说明/status：Discussion(方案讨论中，尚未拿到TC或SIG组maintainer接纳结论)、Developing(开发中)、Testing(测试中)、Accepted(已验收)

发布方式：IMG、OBS等，成熟度目标是指该特性在本版本的交付目标，参考《特性发布成熟度管理规范》中定义，请注意特性在本版本从孵化期转成熟期也需要登记，测试需要以目标成熟度来验收。

| ofep link                                                           | feature                                         | status | sig | owner   | branch | 发布方式 | 涉及软件包列表 | 成熟度目标 |
|---------------------------------------------------------------------|-------------------------------------------------| --- | --- |---------| --- | --- | --- | --- |
| [ofep-0033](https://gitcode.com/openFuyao/ofep/pull/60)             | 基于时间因素的潮汐调度决策算法                                 | Accepted | sig-ai-inference | 陈锌鸿     | / | IMG | elastic-scaler | |
| [ofep-0030](https://gitcode.com/openFuyao/ofep/pull/53)             | 通用扩缩容决策框架设计提案                                   | Accepted | sig-ai-inference | 王雨欣     | / | IMG | elastic-scaler | |
| [ofep-0028](https://gitcode.com/openFuyao/ofep/pull/51)             | Mooncake Store AscendCacheTier实现                | Accepted | sig-ai-inference | 陈文潇、陈小燕 | / | 回合上游PR | mooncake | |
| [ofep-0034](https://gitcode.com/openFuyao/ofep/pull/61)             | Mooncake Store V3 跨机RPC读写                       | Accepted | sig-ai-inference | 陈小燕、陈文潇 | / | 回合上游PR | mooncake | |
| [ofep-0025](https://gitcode.com/openFuyao/ofep/pull/47)             | AI推理智能路由增强-新增容灾设计                               | Accepted | sig-ai-inference | 李乐奇、郑钦文 | / | IMG | hermes-router | |
| [ofep-0036](https://gitcode.com/openFuyao/ofep/pull/63)             | AI推理服务化框架组件端到端helm包一键部署                         | Accepted | sig-ai-inference | 张仕畅     | / | helm chart | infernex | |
| [ofep-0048](https://gitcode.com/openFuyao/ofep/pull/74)             | 安装部署前置检查工具                                      | Accepted | sig-installation | 罗龙      | / | OBS | envCheck | |
| [ofep-0029](https://gitcode.com/openFuyao/ofep/pull/52)             | 面向PD分离场景的动态扩缩调度方案                               | Accepted | sig-ai-inference | 许立      | / | IMG | elastic-scaler | |
| [ofep-0026](https://gitcode.com/openFuyao/ofep/pull/49)             | AI推理可观测                                         | Accepted | sig-ai-inference | 朱雪仪     | / | IMG | eagle-eye | |
| [ofep-0040](https://gitcode.com/openFuyao/ofep/pull/67)             | 提供组件健康检测接口 引导节点镜像仓库自依赖                          | Accepted | sig-installation | 徐尤红     | / | IMG | cluste-api-propider-bke等 | |
| [ofep-0046](https://gitcode.com/openFuyao/ofep/pull/75)             | 安装部署支持节点前置后置操作                                  | Accepted | sig-installation | 张庆伟     | / | IMG | cluste-api-propider-bke等 | |
| [ofep-0047](https://gitcode.com/openFuyao/ofep/pull/73)             | 安装部署BKECluster CR 拆分，支持各节点CR单独运维                | Accepted | sig-installation | 马鑫、刘峻麟  | / | IMG | cluste-api-propider-bke等 | |
| [ofep-0035](https://gitcode.com/openFuyao/ofep/pull/62)             | 安装部署支持多仓库镜像以及兼容k8s的1.28.8版本             | Accepted | sig-installation | 徐尤红     | / | IMG | cluste-api-propider-bke等 | |
| [ofep-0045](https://gitcode.com/openFuyao/ofep/pull/72)             | 构建openFuyao统一日志框架                               | Accepted | sig-container-platform | 戚士伯     | / | IMG | common-modules | |
| [ofep-0038](https://gitcode.com/openFuyao/ofep/pull/65)             | 内存池化借用能力                                        | Accepted | sig-orchestration-engine | 边琛      | / | IMG | 独立发布 | |
| [ofep-0041](https://gitcode.com/openFuyao/ofep/pull/69)             | 基于DRA动态加载NPU                                    | Accepted | sig-orchestration-engine | 张卫涛     | / | IMG | 独立发布 | |
| [ofep-0042](https://gitcode.com/openFuyao/ofep/pull/69)             | 提供deviceplugin纳管URMA网络设备                        | Accepted | sig-orchestration-engine | 王力      | / | IMG | 独立发布 | |
| [ofep-0031](https://gitcode.com/openFuyao/ofep/pull/59)             | 鲲鹏机密容器operator化安装部署                             | Accepted | sig-orchestration-engine | 霍然      | / | SRC | 独立发布 | |
| [ofep-0032](https://gitcode.com/openFuyao/ofep/pull/58)             | calico typha、node-local-dns、victoriametrics集成部署 | Accepted | sig-installation | 程于飞、史子龙 | / | IMG | cluste-api-propider-bke等 | |
| [ofep-0049](https://gitcode.com/openFuyao/ofep/pull/76) | bke支持部署chart形态插件                                | Accepted | sig-installation | 史子龙     | / | IMG | cluste-api-propider-bke等 | |
| 待定                                                                  | bke支持ipv4、ipv6双栈，支持配置去中心化负载均衡器                  | Postpone | sig-installation | 朱桂华、马奥  | / | IMG | cluste-api-propider-bke等 | |

## 需求/特性反馈基本流程

1、开发者/sig在本文件表格中填写要合入本版本的需求/特性，并同时注意在release Management仓中新建需求issue（请在收集截止时间前提交）

2、申请在TC例会上评审需求（owner或者SIG maintainer参会）
