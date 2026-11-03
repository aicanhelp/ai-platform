---
sidebar_position: 6
---

# 已修复问题

| 序号 | ISSUE标题 | ISSUE优先级 | 所属项目 | ISSUE URL |
| ---- | ---- | ---- | ---- | ---- |
| 1 | 【bke】ubuntu环境引导节点bke-controller不断重启 | 次要 | openFuyao/cluster-api-provider-bke | https://gitcode.com/openFuyao/cluster-api-provider-bke/issues/168 |
| 2 | 【installation】创建集群超时失败后未清理环境，测试集群残留 | 主要 | openFuyao/e2e-auto-test | https://gitcode.com/openFuyao/e2e-auto-test/issues/41 |
| 3 | 【npu-operator】使用默认配置部署时volcano组件异常 | 主要 | openFuyao/npu-operator | https://gitcode.com/openFuyao/npu-operator/issues/40 |
| 4 | 【KubeOS测试提单】bkeagent-launcher在只读rootfs上覆盖/usr/local/bin/bkeagent导致BC卡EnsureClusterManage | 主要 | openFuyao/cluster-api-provider-bke | https://gitcode.com/openFuyao/cluster-api-provider-bke/issues/170 |
| 5 | 【性能测试】200并发性能测试中curl压测结果未满足1s要求 | 主要 | openFuyao/serverlessdb-operator | https://gitcode.com/openFuyao/serverlessdb-operator/issues/1 |
| 6 | 【npu-dra-driver】资源上报错误 | 次要 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/32 |
| 7 | 【npu-dra-driver】硬切分失败 | 不重要 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/33 |
| 8 | 【npu-dra-driver】文档中前置条件npu驱动描述较简陋 | 次要 | openFuyao/sig-orchestration-engine | https://gitcode.com/openFuyao/sig-orchestration-engine/issues/19 |
| 9 | 【infernex】部署1p1d后发送推理请求失败 | 主要 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/146 |
| 10 | 上下文超长场景下，未能正确返回错误码，proxy server返回200 | 无优先级 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/143 |
| 11 | 【test-cases-generator】在输入示例中补充文件名所属sig | 次要 | openFuyao/openfuyao-powers | https://gitcode.com/openFuyao/openfuyao-powers/issues/17 |
| 12 | 【缺陷】mooncake te与zmq端口冲突 | 主要 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/119 |
| 13 | 【文档提单】NUMA亲和插件KubeOS升级镜像构建手册存在冗余内容 | 次要 | openFuyao/kubeos-support | https://gitcode.com/openFuyao/kubeos-support/issues/1 |
| 14 | 【证书生成】EnsureCerts阶段lookupRegularCert缺少重试机制，Informer cache同步延迟导致 "certs need generate again"错误 | 不重要 | openFuyao/cluster-api-provider-bke | https://gitcode.com/openFuyao/cluster-api-provider-bke/issues/185 |
| 15 | 【npu-dra-plugin】镜像地址错误 | 次要 | openFuyao/sig-orchestration-engine | https://gitcode.com/openFuyao/sig-orchestration-engine/issues/20 |
| 16 | tokenizer data-producer初始化失败(如模型目录无chat_template)会让EPP永不就绪，经InferencePool的全部请求挂死/503 —— 期望init阶段与运行期一致fail-open | 无优先级 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/118 |
| 17 | DRA插件软切分劫持库路径可配置 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/30 |
| 18 | dra插件添加Helm chart包 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/11 |
| 19 | draConfig节点配置改为结构化可配置 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/12 |
| 20 | dra resourceslice属性ID命名标准化 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/13 |
| 21 | resourceslice新增type字段 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/14 |
| 22 | dra硬切分模板可配置 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/15 |
| 23 | DRA硬切分支持910C | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/16 |
| 24 | DRA硬切分支持310P | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/17 |
| 25 | DRA软切分支持910C | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/18 |
| 26 | DRA软切分支持310P | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/19 |
| 27 | 软切分支持设置最大切分数 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/20 |
| 28 | DRA插件软切分支持三种调度策略 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/22 |
| 29 | DRA插件硬切分支持根据AiCore， AiCPU进行容量申请 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/23 |
| 30 | DRA插件支持卡级软切分 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/21 |
| 31 | DRA插件硬切分模板匹配强校验 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/24 |
| 32 | DRA插件废弃文档清理 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/25 |
| 33 | DRA插件部署yaml优化 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/26 |
| 34 | DRA插件测试Demo优化 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/27 |
| 35 | DRA插件硬切分支持容器化部署 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/28 |
| 36 | DRA插件软切分支持劫持库选择安装 | 无优先级 | openFuyao/npu-dra-plugin | https://gitcode.com/openFuyao/npu-dra-plugin/issues/29 |
| 37 | 【管理集群创建时存在ERROR日志】 | 次要 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/281 |
| 38 | 【安装64节点依然存在api server限流】 | 主要 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/282 |
| 39 | 【开发自提单】修复H-06子网校验与H-07连通性检查结果不一致问题 | 无优先级 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/147 |
| 40 | 【开发自提单】优化慢卡检测结论解析逻辑，提升压测负载强度 | 无优先级 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/148 |
| 41 | 【开发自提单】优化K-04节点剩余资源计算逻辑 | 无优先级 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/149 |
| 42 | 【开发自提单】优化跨节点快慢卡指标展示能力 | 无优先级 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/150 |
| 43 | 【开发自提单】修正跳过检查范围输出错误 | 无优先级 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/151 |
| 44 | 【开发自提单】优化H-08检查建议，提供可执行指导 | 无优先级 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/152 |
| 45 | 【需求】/【Requirement】Hermes Router tokenizer对齐KServe llm-d scheduler契约 | 无优先级 | openFuyao/hermes-router | https://gitcode.com/openFuyao/hermes-router/issues/44 |
| 46 | 【需求】Calico安装与校验流程优化 | 不重要 | openFuyao/cluster-api-provider-bke | https://gitcode.com/openFuyao/cluster-api-provider-bke/issues/199 |
| 47 | 【需求】健康检查配置支持从管理集群同步到业务集群 | 不重要 | openFuyao/cluster-api-provider-bke | https://gitcode.com/openFuyao/cluster-api-provider-bke/issues/198 |
| 48 | 【健康检查】错误包装后组件优先级识别丢失导致重试间隔错误 | 不重要 | openFuyao/cluster-api-provider-bke | https://gitcode.com/openFuyao/cluster-api-provider-bke/issues/197 |
| 49 | 【需求】健康检查失败重试间隔按组件优先级生效 | 不重要 | openFuyao/cluster-api-provider-bke | https://gitcode.com/openFuyao/cluster-api-provider-bke/issues/196 |
| 50 | 【需求】健康检查支持核心组件清单配置化 | 不重要 | openFuyao/cluster-api-provider-bke | https://gitcode.com/openFuyao/cluster-api-provider-bke/issues/195 |
| 51 | 【需求】bkeadm优化Calico安装校验与组件部署策略 | 不重要 | openFuyao/bkeadm | https://gitcode.com/openFuyao/bkeadm/issues/124 |
| 52 | 【需求】bkeadm支持健康检查配置导入与管理集群同步 | 不重要 | openFuyao/bkeadm | https://gitcode.com/openFuyao/bkeadm/issues/123 |
| 53 | 【引导节点apiserver的QPS默认配置为5】 | 次要 | openFuyao/docs | https://gitcode.com/openFuyao/docs/issues/2056 |
| 54 | 【缺陷】拉取obs index.yaml异常导致无法获取fuyao版本，进而无法拉起新集群 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/285 |
| 55 | 【cluster-provider配置参数资料问题】bc文件中containerd默认描述存在问题 | 不重要 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/286 |
| 56 | 【cluster-provider资料问题】 | 不重要 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/287 |
| 57 | 【cluster-api-provider-bke资料问题】 | 不重要 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/288 |
| 58 | 【UT】hermes-router单元测试覆盖率提升至80% 以上 | 无优先级 | openFuyao/hermes-router | https://gitcode.com/openFuyao/hermes-router/issues/48 |
| 59 | 【UT】monitoring-dashboard-service单元测试覆盖率提升至80% 以上 | 无优先级 | openFuyao/monitoring-dashboard-service | https://gitcode.com/openFuyao/monitoring-dashboard-service/issues/15 |
| 60 | 【开发自提单】当enabled=false时，eagle-eye namespace仍被无条件创建 | 无优先级 | openFuyao/eagle-eye | https://gitcode.com/openFuyao/eagle-eye/issues/20 |
| 61 | 【开发自提单】Jaeger Query服务未通过NodePort对外暴露 | 无优先级 | openFuyao/eagle-eye | https://gitcode.com/openFuyao/eagle-eye/issues/21 |
| 62 | 【开发自提单】Dockerfile路径分散未统一 | 无优先级 | openFuyao/eagle-eye | https://gitcode.com/openFuyao/eagle-eye/issues/22 |
| 63 | 【开发自提单】 tracing chart中引入了未被使用的OpenTelemetry Collector helm chart，造成冗余依赖 | 无优先级 | openFuyao/eagle-eye | https://gitcode.com/openFuyao/eagle-eye/issues/23 |
| 64 | 【容器】IPv6网络平面下节点/Pod/Service页面IP地址解析显示有误 | 次要 | openFuyao/console-website | https://gitcode.com/openFuyao/console-website/issues/714 |
| 65 | 【容器】IPv6网络平面下console-service存在多处双栈适配缺陷 | 主要 | openFuyao/console-service | https://gitcode.com/openFuyao/console-service/issues/118 |
| 66 | docs: InferNex-Bridge链接错误、GIE版本及文字错误 | 无优先级 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/156 |
| 67 | Tokenizer冷启动Initialize被短超时重试打穿，sidecar叠线程触发transformers ImportError | 无优先级 | openFuyao/hermes-router | https://gitcode.com/openFuyao/hermes-router/issues/49 |
| 68 | helm配置文件优化 | 次要 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/142 |
| 69 | 【test-cases-generator】FIT与DFX测试用例边界不清晰，导致两类用例相互混入且跨CSV重复 | 次要 | openFuyao/openfuyao-powers | https://gitcode.com/openFuyao/openfuyao-powers/issues/25 |
| 70 | 【npu-operator】部署dra前需要提供前置条件 | 主要 | openFuyao/docs | https://gitcode.com/openFuyao/docs/issues/2058 |
| 71 | 【npu-operator】dra软切分配置挂载项和npui-dra-plugin不统一 | 次要 | openFuyao/docs | https://gitcode.com/openFuyao/docs/issues/2059 |
| 72 | 【infernex】推理请求失败，mooncake kv传输失败 | 无优先级 | openFuyao/sig-ai-inference | https://gitcode.com/openFuyao/sig-ai-inference/issues/49 |
| 73 | 【安全加固】仓库缺失入口脚本bin/k8s-hardening.sh | 次要 | openFuyao/compliance-operator | https://gitcode.com/openFuyao/compliance-operator/issues/3 |
| 74 | 【安全加固】changes.json变更记录缺失rule_id字段，无法按规则追溯变更 | 次要 | openFuyao/compliance-operator | https://gitcode.com/openFuyao/compliance-operator/issues/4 |
| 75 | 【创建3master3worker集群失败】 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/293 |
| 76 | 【资料问题-离线安装版本文件下载路径修改】 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/295 |
| 77 | 【构建离线包报错】 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/296 |
| 78 | 【资料问题-DRA用户手册】安装指令中dra版本不存在 | 无优先级 | openFuyao/sig-orchestration-engine | https://gitcode.com/openFuyao/sig-orchestration-engine/issues/21 |
| 79 | 【需求】适配openFuyao v26.09-rc.2安装与升级链路 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/297 |
| 80 | 【serverlessdb文档】DBResourcePool示例的compute/init镜像404，资源池永远无法WarmReady | 次要 | openFuyao/sig-orchestration-engine | https://gitcode.com/openFuyao/sig-orchestration-engine/issues/22 |
| 81 | 【serverlessdb文档】helm install示例缺 -n参数，release落到default触发CRD所有权校验失败 | 次要 | openFuyao/sig-orchestration-engine | https://gitcode.com/openFuyao/sig-orchestration-engine/issues/23 |
| 82 | 【serverlessdb文档】RBAC示例YAML缩进错误，kubectl apply无法解析 | 次要 | openFuyao/sig-orchestration-engine | https://gitcode.com/openFuyao/sig-orchestration-engine/issues/24 |
| 83 | 【serverlessdb文档】缺少创建调用方ServiceAccount与获取Bearer Token的步骤 | 次要 | openFuyao/sig-orchestration-engine | https://gitcode.com/openFuyao/sig-orchestration-engine/issues/25 |
| 84 | 【serverlessdb文档/代码】Gin仅绑定PodIP，port-forward调试必然失败且文档未说明 | 次要 | openFuyao/sig-orchestration-engine | https://gitcode.com/openFuyao/sig-orchestration-engine/issues/26 |
| 85 | 【集群创建】bke cluster create自动加bke- 前缀导致BN与BC命名空间不一致被webhook拦截 | 无优先级 | openFuyao/bkeadm | https://gitcode.com/openFuyao/bkeadm/issues/145 |
| 86 | 【安装文档】自定义配置与集群生命周期相关说明不完整 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/305 |
| 87 | 【安装文档】集群卸载指南分散且缺少删除范围/顺序/成功示意 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/306 |
| 88 | 【需求】补充v25.12集群回滚操作指南（中英文） | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/308 |
| 89 | 【需求】升级指导补充bkeagent二进制准备说明 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/310 |
| 90 | 【26.09rc2离线包制作失败】 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/313 |
| 91 | 【CodeCheck】v26.09版本codecheck问题清除 | 次要 | openFuyao/console-service | https://gitcode.com/openFuyao/console-service/issues/119 |
| 92 | 【日志】配置LogFile后文件日志静默丢失且未持久化 | 次要 | openFuyao/serverlessdb-operator | https://gitcode.com/openFuyao/serverlessdb-operator/issues/2 |
| 93 | 【需求】补充v26.06到v26.09升级指导（Secret预创建与Webhook变更） | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/317 |
| 94 | [Compatibility] L1 ingest silently skips map-encoded `BlockStored` observed with vLLM 0.28.0 | 无优先级 | openFuyao/cache-indexer | https://gitcode.com/openFuyao/cache-indexer/issues/17 |
| 95 | 【沙箱Watcher】containerd运行时部署失败：cri-socket挂载的 /run/cri-multiplex.sock不存在 | 不重要 | openFuyao/sig-agent-sandbox | https://gitcode.com/openFuyao/sig-agent-sandbox/issues/1 |
| 96 | 【沙箱Agent】agent镜像缺少 /tmp目录，创建沙箱报stat /tmp失败 | 次要 | openFuyao/sig-agent-sandbox | https://gitcode.com/openFuyao/sig-agent-sandbox/issues/2 |
| 97 | 【沙箱Chart】Chart.yaml的kubeVersion约束不兼容openFuyao自定义版本号v1.34.3-of.1，安装失败 | 次要 | openFuyao/sig-agent-sandbox | https://gitcode.com/openFuyao/sig-agent-sandbox/issues/3 |
| 98 | 【沙箱管理】Agent Pod删除后沙箱phase未进入Lost阶段 | 主要 | openFuyao/sig-agent-sandbox | https://gitcode.com/openFuyao/sig-agent-sandbox/issues/4 |
| 99 | 【部署配置】Controller和Scheduler的imagePullPolicy: Never导致多节点集群部署失败 | 主要 | openFuyao/sig-agent-sandbox | https://gitcode.com/openFuyao/sig-agent-sandbox/issues/5 |
| 100 | 【gRPC安全】Controller gRPC参数仅做非空校验，存在注入攻击风险 | 次要 | openFuyao/sig-agent-sandbox | https://gitcode.com/openFuyao/sig-agent-sandbox/issues/6 |
| 101 | 【日志可诊断性】多个核心组件错误路径缺乏ERROR日志或上下文 | 次要 | openFuyao/sig-agent-sandbox | https://gitcode.com/openFuyao/sig-agent-sandbox/issues/7 |
| 102 | 【资料测试】性能测试报告（v26.06）缺少数据集生成方法与测试脚本，无法复现 | 无优先级 | openFuyao/sig-ai-inference | https://gitcode.com/openFuyao/sig-ai-inference/issues/52 |
| 103 | 【UT】ray-service单元测试覆盖率提升至80% 以上 | 无优先级 | openFuyao/ray-service | https://gitcode.com/openFuyao/ray-service/issues/18 |
| 104 | 【集群生命周期】管理集群自纳管后前端误显示为业务集群 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/319 |
| 105 | 【文档】业务集群安装指南混入管理集群内容且缺管理面补装说明 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/322 |
| 106 | 【文档】k8s_security_hardening_user_guide.md存在路径缺失、内容重复问题 | 次要 | openFuyao/compliance-operator | https://gitcode.com/openFuyao/compliance-operator/issues/8 |
| 107 | 【CodeCheck】v26.09版本codecheck问题清除 | 无优先级 | openFuyao/InferNex | https://gitcode.com/openFuyao/InferNex/issues/169 |
| 108 | 【安装部署文档】补充bke init交互确认说明与集群规格配置建议 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/304 |
| 109 | 【安装部署文档】补充安装成功回显示例与集群信息滚动完成判定标准 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/298 |
| 110 | 【chartRepo】管理集群BKECluster资源未适配自定义chartRepo，离线/私有化场景下chart addon安装失败 | 主要 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/291 |
| 111 | 【自提】创建集群"一键校验"后点击"创建"按钮3-4秒无反馈，易导致用户重复点击误触 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/283 |
| 112 | 【自提】提升pkg/phaseframe/phases包单测覆盖率至88.9% | 无优先级 | openFuyao/cluster-api-provider-bke | https://gitcode.com/openFuyao/cluster-api-provider-bke/issues/165 |
| 113 | 【bkeadm】将k3s引导集群容器日志挂载到主机 | 无优先级 | openFuyao/bkeadm | https://gitcode.com/openFuyao/bkeadm/issues/144 |
| 114 | 【集群管理】集群安装日志弹窗滚动到底部后应随内容生成持续自动滚动 | 无优先级 | openFuyao/installer-website | https://gitcode.com/openFuyao/installer-website/issues/57 |
| 115 | 【doc】管理集群安装文档包含业务集群，前端安装缺少负载均衡且IP说明 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/321 |
| 116 | 【文档整改】补齐在线增量安装openFuyao平台扩展组件文档（online_incremental_component_installation） | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/307 |
| 117 | 【安装部署文档】快速入门补充仓库同步后应用才可见的说明 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/303 |
| 118 | 【文档整改】 fix comments in installation docs (Chinese and English) | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/309 |
| 119 | 【文档结构调整】新增集群扩缩容文档 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/311 |
| 120 | 【用户手册】客户端示例两处错误：流式回调用同步lambda、缺use_server_proxy | 主要 | openFuyao/sig-agent-sandbox | https://gitcode.com/openFuyao/sig-agent-sandbox/issues/8 |
| 121 | 【bkeagent/reset】SourceClean中yum makecache过慢导致缩容Reset超时 | 无优先级 | openFuyao/cluster-api-provider-bke | https://gitcode.com/openFuyao/cluster-api-provider-bke/issues/231 |
| 122 | 【UB提升集群利用率20%指标未体现】 | 无优先级 | openFuyao/ubs-k8s-enable | https://gitcode.com/openFuyao/ubs-k8s-enable/issues/42 |
| 123 | 【安装部署性能报告部分指标劣化分析】 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/326 |
| 124 | 【静态pod配置】 | 无优先级 | openFuyao/sig-installation | https://gitcode.com/openFuyao/sig-installation/issues/329 |
| 125 | 【codecheck】PD打分函数参数过多，需收敛签名 | 无优先级 | openFuyao/hermes-router | https://gitcode.com/openFuyao/hermes-router/issues/59 |