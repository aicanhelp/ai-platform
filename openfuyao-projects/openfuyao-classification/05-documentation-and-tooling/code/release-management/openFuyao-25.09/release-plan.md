
# Version Info

openFuyao 25.09 是首个全面开源版本

## Release Plan

| Stage Name                    | Deadline for PR | Begin Time | End Time   | Days | Note                                     |
| ----------------------------- | --------------- | ---------- | ---------  | ---- | ---------------------------------------- |
| Collect key features          |        -        | 2025/07/01 | 2025/08/27 | 58 | 版本需求收集                              |
| Change Review                 |        -        | 2025/07/01 | 2025/08/27 | 58 | Review 软件包变更（升级/退役/淘汰）  |
| Develop                       |        -        | 2025/07/01 | 2025/08/27 | 65 | 新特性开发，合入Master |
| 切换K8s配套版本                |        -        | 2025/07/20 | 2025/07/31 | 12 | 审视并适配新版本K8s源码 |
| Build & Alpha                 |    2025/07/30   | 2025/08/01 | 2025/08/07 | 07 | 新特性合入，Alpha版本发布    |
| Test round 1                  |    2025/08/06   | 2025/08/08 | 2025/08/14 | 07 | 模块测试           |
| Test round 2 (Beta Version)   |    2025/08/13   | 2025/08/15 | 2025/08/21 | 07 | Beta版本发布    |
| Test round 3                  |    2025/08/20   | 2025/08/22 | 2025/08/28 | 07 | 模块测试       |
| Test round 4                  |    2025/08/27   | 2025/08/29 | 2025/09/04 | 07 | 全量验证(全量SIT)  |
| Test round 5                  |    2025/09/03   | 2025/09/05 | 2025/09/11 | 07 | 回归测试, RC版本发布          |
| Test round 6                  |    2025/09/10   | 2025/09/12 | 2025/09/18 | 07 | 回归测试                         |
| Branch release-25.09          |    -            | 2025/09/17 | 2025/09/18 | 01 | Master 拉取 25.09 发布分支|
| Test round 7 (预留)           |    2025/09/17   | 2025/09/19 | 2025/09/24 | 06 | 回归测试                         |
| Release Review                |        -        | 2025/09/22 | 2025/09/26 | 05 | 版本发布评审        |
| Release preparation           |        -        | 2025/09/24 | 2025/09/26 | 03 | 发布前准备阶段，发布件系统梳理    |
| Release                       |        -        | 2025/09/28 | 2025/09/30 | 03 | 社区Release评审通过正式布       |
|||||||

## 代码合入说明

项目采用oneTrack方式开发，主程序基于Main分支构建， 个别组件有分支诉求的，可在具体特性需求中标明构建分支

// 新特性代码请及时合入版本&自验证，跟随整体计划赶在第一轮转测试

## Feature list

状态说明/status：Accepted(方案讨论中，尚未拿到TC提案接纳结论)、 Accepted(开发中)、 Testing(测试中)、 Accepted(已验收)

发布方式：IMG、OBS等

| ofep link | feature | status | sig | owner | branch |  发布方式 | 涉及软件包列表 |
| -------------| ------- | ------ | ----| ----- | ------ | -------- | ------------- |
|ofep-0004|kubelet支持服务启动时CPU垂直扩容加速java程序启动|Accepted|sig-orchestration-engine|张卫涛|master|IMG|kubernetes|
|ofep-0007|statefulset的pvc模板支持扩容|Accepted|sig-orchestration-engine|张卫涛|master|IMG|kubernetes|
|ofep-0005|k8s证书管理增强-热加载|Accepted|sig-orchestration-engine|张卫涛|master|IMG|kubernetes,etcd,coredns|
|opef-0016|安装部署完全支持离线环境|Accepted|sig-installation|徐尤红|master|OBS,IMG|bkeadm,bkeagent|
|opef-0010|在离线混部组件集成rubik的cpu/内存相关功能|Accepted|sig-colocation|李航宇、刘峻麟|master|IMG|colocation-website,colocation-service,colocation-agent,colocation-rubik-agent|
|opef-0012|合并在离线混部仓库，优化在离线混部组件部署结构|Accepted|sig-colocation|郭宇翔|master|IMG|colocation-overquota-agent,colocation-manager|
|opef-0014|fuyao-log-runner支持日志轮转和可靠性增强|Accepted|sig-orchestration-engine|张卫涛|master|IMG|kubernetes|
|opef-0006|容器高密部署|Accepted|sig-orchestration-engine|张卫涛|master|IMG|kubernetes,containerd|
|ofep-0009|提供OS一致性检查工具|Accepted|sig-orchestration-engine|张卫涛|master|IMG|node-checker|
|opef-0013|AI推理软件套件|Accepted|sig-ai-inference|马子凡|master|IMG|ai-all-in-one|
|opef-0011|smart routerAI推理智能路由|Accepted|sig-ai-inference|郑钦文|master|IMG|hermes-router|
|||||||||

## 需求/特性反馈基本流程

1、开发者/sig在本文件表格中填写要合入本版本的需求/特性，并同时注意在release Management仓中新建需求issue （请在收集截止时间前提交）

2、申请在TC例会上评审需求 （owner或者SIG maintainer参会）

3、owner参加RM Sig例会，按照版本计划按时完成合入和验证
