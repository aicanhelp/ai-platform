package utils

// ElasticScalerE2ENamespace e2e 全部资源（controller/webhook、测试工作负载、CR）统一部署的命名空间，
// 不使用 chart 默认的 scaling-system，原因有二：
//  1. 共享集群上常已有其它 pd-orchestrator 安装（例如 infernex 套件的 release 装在 zxyyy 命名空间、
//     资源却落在 scaling-system），会占用 chart 里名字固定的对象，使 helm 归属校验直接失败：
//     "Unable to continue with install: ... invalid ownership metadata; ... must equal ..."；
//  2. e2e 的清理逻辑会删除整个命名空间（tidal 容器 AfterAll），落在默认 ns 会误伤别人的资源。
//
// 并行跑多套时用 ELASTIC_SCALER_E2E_NAMESPACE 覆盖，例如 ELASTIC_SCALER_E2E_NAMESPACE=scaling-e2e-run2。
var ElasticScalerE2ENamespace = envOrDefault("ELASTIC_SCALER_E2E_NAMESPACE", "scaling-system-e2e")

// ElasticScalerHelm 安装 elastic-scaler 的 helm 配置
const (
	ElasticScalerHelmReleaseName = "elastic-scaler"
	ElasticScalerHelmChart       = "oci://cr.openfuyao.cn/charts/pd-orchestrator"
	ElasticScalerHelmVersion     = "0.0.0-latest"
	ElasticScalerPodPrefix       = "elastic-scaler-controller-manager"

	// Helm 安装可通过环境变量覆盖（本地开发 elastic-scaler 仓库时使用）
	EnvElasticScalerHelmChart   = "ELASTIC_SCALER_HELM_CHART"   // 本地 Chart 路径，如 .../elastic-scaler_last/charts/pd-orchestrator
	EnvElasticScalerHelmVersion = "ELASTIC_SCALER_HELM_VERSION" // 默认 0.0.0-latest；本地路径时不传 --version
	EnvElasticScalerImageRepo   = "ELASTIC_SCALER_IMAGE_REPOSITORY"
	EnvElasticScalerImageTag    = "ELASTIC_SCALER_IMAGE_TAG"

	// helm --timeout：helm 默认仅 5m，且它同时是"等待 hook 完成"的上限。chart 内 pre-install（生成 webhook 证书）
	// 与 post-install（注册 webhook，activeDeadlineSeconds=600s）hook 都要拉取 hub.oepkgs.net/openfuyao/alpine/openssl、
	// cr.openfuyao.cn/openfuyao/kubectl-openfuyao 镜像，冷启动时 5m 会先超时，
	// 表现为 "failed pre-install: ... timed out waiting for the condition"。
	ElasticScalerHelmTimeout = "10m"

	// helm 安装总尝试次数（首次 + 重试）；失败重试前会清理 failed release 与残留 hook Job。
	ElasticScalerHelmInstallAttempts = 2
)

// ElasticScalerTest 测试用
var (
	// ElasticScalerHelmNamespace 是 helm release 的命名空间，ElasticScalerTestNamespace 是测试工作负载（targetRef 目标）的命名空间；
	// 两者历史上都取 chart 默认值 scaling-system，现统一指向 e2e 专属命名空间。
	ElasticScalerHelmNamespace = ElasticScalerE2ENamespace
	ElasticScalerTestNamespace = ElasticScalerE2ENamespace
)

// ES 指标驱动测试用（参考 exam_design test/test_3 等）
const (
	ESTestDeploymentName            = "elastic-scaler-test3"              // 指标驱动主用例 Deployment
	ESTestDeploymentNameMultiES     = "elastic-scaler-test3-multi"        // 多 ES 并存第二个 Deployment
	ESTestDeploymentNameExistingHPA = "elastic-scaler-test3-existing-hpa" // 预先存在 HPA 用例的 Deployment
	ESTestDeploymentNameCRUD        = "elastic-scaler-crud-deploy"        // CRUD 用例专用 Deployment
	ESTestDeploymentNameBigMax      = "elastic-scaler-bigmax-deploy"      // 极端 MaxReplicas 用例专用（与主 ES target 隔离，满足一 target 一 ES）

	ESTestStatefulSetName = "elastic-scaler-test4" // StatefulSet 兼容性用例

	ESTestCRName             = "elastic-scaler-metrics"      // 主测 ES CR
	ESTestCRNameMultiMetrics = "es-multi-metrics"            // 多指标用例
	ESTestCRNameUpdate       = "es-update-test"              // 更新 MaxReplicas 用例
	ESTestCRNameBigMax       = "elastic-scaler-maxreplicas"  // 极端 MaxReplicas 用例
	ESTestCRNameExistingHPA  = "elastic-scaler-existing-hpa" // 预先存在 HPA 的用例
	ESTestCRNameStatefulSet  = "elastic-scaler-statefulset"  // StatefulSet 目标类型用例
	ESTestCRNameMultiES1     = "elastic-scaler-multi-1"      // 多 ES 并存 - ES1
	ESTestCRNameMultiES2     = "elastic-scaler-multi-2"      // 多 ES 并存 - ES2

	// 输入/校验相关 ES 名称
	ESTestCRNameMetricsEmpty       = "elastic-scaler-metrics-empty"        // metrics=[]
	ESTestCRNameUnknownTriggerType = "elastic-scaler-unknown-trigger"      // 未知 Trigger.Type
	ESTestCRNameNoScalingAlgorithm = "elastic-scaler-no-scaling-algorithm" // MetricsTrigger 缺失 scalingAlgorithm
	ESTestCRNameSignalNoEndpoint   = "elastic-scaler-signal-no-endpoint"   // SignalTrigger 缺失 endpoint
	ESTestCRNameMetricNoResource   = "elastic-scaler-metric-no-resource"   // Resource 类型缺失 resource 字段

	// 未实现路径/跨命名空间/StateTrigger 相关 ES 名称
	ESTestCRNameExternalMetric     = "elastic-scaler-external-metric"      // External 指标
	ESTestCRNameExternalServer     = "elastic-scaler-external-server"      // ExternalServer 指标
	ESTestCRNameCustomAlgorithm    = "elastic-scaler-custom-algo"          // 自定义 scalingAlgorithm
	ESTestCRNameSignal             = "elastic-scaler-signal"               // SignalTrigger 正常配置
	ESTestCRNameCrossNamespace     = "elastic-scaler-cross-namespace"      // 跨命名空间 TargetRef
	ESTestCRNameStateMissingTarget = "elastic-scaler-state-missing-target" // StateTrigger 目标资源不存在
	ESTestCRNameStateBasic         = "elastic-scaler-state-basic"          // StateTrigger 基本功能用例

	// StateTrigger 功能测试专用（与 tidal 用例隔离，同在 e2e 专属命名空间 ElasticScalerE2ENamespace）
	ESTestStateTriggerDeploymentName = "elastic-scaler-statetrigger-deploy" // StateTrigger 用例目标 Deployment
	ESTestStateTriggerTidalName      = "elastic-scaler-statetrigger-tidal"  // StateTrigger 用例用 Tidal CR

	// 跨命名空间用的 namespace / 目标 Deployment 名称
	ESTestNamespaceA           = "elastic-scaler-a"
	ESTestNamespaceB           = "elastic-scaler-b"
	ESTestCrossNamespaceDeploy = "elastic-scaler-cross-deploy"

	// 竞态测试用 ES 名称（参考 test/test_24）
	ESTestCRNameConcurrentUpdate = "elastic-scaler-concurrent-update" // 同一 ES Spec 并发更新
	ESTestCRNameConcurrentDelete = "elastic-scaler-concurrent-delete" // 删除与并发更新竞态
)
