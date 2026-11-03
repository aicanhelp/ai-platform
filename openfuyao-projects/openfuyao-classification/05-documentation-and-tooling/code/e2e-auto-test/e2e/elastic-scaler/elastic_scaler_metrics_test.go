package elastic_scaler

import (
	"time"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// 通用扩缩容决策框架 E2E（oFEP-0030 / oFEP-0044）
// 用例编号与名称见 通用扩缩容决策框架测试用例.tsv
//
// 用例间复用（Ordered 顺序执行）：
//   - BeforeAll：metrics-server、elastic-scaler helm；AfterAll：CleanupMetricsFrameworkTest 卸载 CR + rsg/es helm
//   - 002/003：复用 001 的 metrics-es-001 与 Deployment；003 DeferCleanup → CleanupMetricsFrameworkAfter001To003
//   - 005 DeferCleanup → CleanupMetricsFrameworkAfter005
//   - 006→007：复用 ExternalServer 双端点；007 DeferCleanup → CleanupMetricsFrameworkAfter006To007
//   - 008→009：复用推理服务；009 DeferCleanup → CleanupMetricsFrameworkAfter008To009
//   - 010 入口兜底 CleanupMetricsFrameworkPhase001To009Resources；010~015 各用例 DeferCleanup 及时释放
//   - 016 入口兜底 CleanupMetricsFrameworkPhase010To015Resources；019 DeferCleanup → CleanupMetricsFrameworkAfter019
//   - 016→017/018/020/021：共用 metrics-es-016、metrics-rsg-016（017 推理 deploy-017 在 018 停止压测后保留至 AfterAll）
//   - Prometheus：SkipUnlessClusterPrometheusAvailable 首次探测后缓存，后续用例不再等待
var _ = Describe("通用扩缩容决策框架", Ordered, Label("elastic-scaler", "metrics-framework", "with-workload-cluster"), func() {
	ns := MetricsTestNamespace

	BeforeAll(func() {
		E2EProgressLog("[metrics-framework BeforeAll] 等待测试命名空间就绪...")
		Expect(EnsureNamespaceReady(clientset, ctx, ns, 5*time.Minute)).NotTo(HaveOccurred())
		E2EProgressLog("[metrics-framework BeforeAll] 清理遗留 CR/Deployment...")
		PrepareMetricsFrameworkTest(dynamicClient, clientset, ctx)
		E2EProgressLog("[metrics-framework BeforeAll] 安装/升级 elastic-scaler（helm，可能拉取 OCI chart）...")
		Expect(EnsureElasticScalerForMetrics()).NotTo(HaveOccurred())
		E2EProgressLog("[metrics-framework BeforeAll] 等待 controller Pod Ready（最长 5 分钟）...")
		VerifyElasticScalerPodsRunning(clientset, ctx, 5*time.Minute, 10*time.Second)
		VerifyElasticScalerCRDAvailable(clientset, ctx)
		E2EProgressLog("[metrics-framework BeforeAll] 确认 metrics-server 可用...")
		EnsureMetricsServerAvailable(clientset, ctx)
		E2EProgressLog("[metrics-framework BeforeAll] 完成，开始执行用例")
	})

	BeforeEach(func() {
		Expect(EnsureNamespaceReady(clientset, ctx, ns, 2*time.Minute)).NotTo(HaveOccurred())
	})

	// FlakeAttempts 只挂在首个用例上：BeforeAll 里的 helm 安装/拉镜像属于网络抖动高发点，
	// 而 Ordered 容器内重试用例会重跑"失败过的" BeforeAll（BeforeAll 本身不能挂 FlakeAttempts）。
	// 首个用例若两次都失败，后续用例会被 Ginkgo 跳过，套件快速失败；挂在这里也不会让每个用例都跑两遍。
	It("安装部署-控制器部署与健康验证", FlakeAttempts(2), func() {
		VerifyHelmReleaseStatus(ElasticScalerHelmReleaseName, ElasticScalerHelmNamespace)
		VerifyElasticScalerPodsRunning(clientset, ctx, 5*time.Minute, 10*time.Second)
		VerifyElasticScalerCRDAvailable(clientset, ctx)
	})

	It("ES-E2E-001-BuildMetricsContext-默认策略与基础上下文", func() {
		targetRef := ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       MetricsTestDeployName,
			Namespace:  ns,
		}
		singleMetric := []ESApaMetricEntry{DefaultResourceCPUMetric()}

		By("准备 targetRef 指向的 Deployment（Running Pod，含 CPU request；幂等，供 002/003 复用 ES）")
		EnsureESMetricsTestDeploymentReady(clientset, ctx, MetricsTestDeployName, ns, 1, false, true)

		By("创建 apa MetricsTrigger ES（Resource/cpu，无 metrics 策略注解；min=max=1 避免扩缩容干扰 metrics 采集；供 ES-E2E-002 复用）")
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName001, ns, targetRef, 1, 1, singleMetric)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName001, ns)
		VerifyESHasNoMetricsPolicyAnnotations(dynamicClient, ctx, MetricsESName001, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName001, ns)
		resourceRecovery := &ResourceMetricsWaitRecovery{Clientset: clientset, Namespace: ns, DeployName: MetricsTestDeployName}
		VerifyESMetricsCollectionFullSuccess(dynamicClient, ctx, MetricsESName001, ns, 1, true, resourceRecovery)
		VerifyHPANotExists(clientset, ctx, MetricsESName001+"-hpa", ns)
	})

	It("ES-E2E-002-ParseMetricsPolicyConfig-合法注解与归一化不破坏采集", func() {
		By("前置：复用 ES-E2E-001 已成功的 ElasticScaler")
		VerifyESCRExists(dynamicClient, ctx, MetricsESName001, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName001, ns)

		By("Patch metadata.annotations：aggregationType=Max（大小写混合）、missingDataPolicy=ignore、windowSeconds=120")
		Expect(PatchESAnnotations(dynamicClient, ctx, MetricsESName001, ns, ValidMetricsPolicyAnnotationsE2E002())).NotTo(HaveOccurred())

		By("等待下一次 Reconcile（generation 对齐后），验证 MetricsPipeline 仍为全量采集成功")
		VerifyESMetricsCollectionFullSuccess(dynamicClient, ctx, MetricsESName001, ns, 1, true)
	})

	It("ES-E2E-003-ParseMetricsPolicyConfig-非法注解回退默认且可继续采集", func() {
		DeferCleanup(func() {
			CleanupMetricsFrameworkAfter001To003(dynamicClient, clientset, ctx)
		})

		By("前置：复用 ES-E2E-002 已 Patch 合法注解的 ElasticScaler")
		VerifyESCRExists(dynamicClient, ctx, MetricsESName001, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName001, ns)

		By("Patch 非法注解：windowSeconds=-1、aggregationType=UnknownType、enableOutlierFilter=not-a-bool")
		Expect(PatchESAnnotations(dynamicClient, ctx, MetricsESName001, ns, InvalidMetricsPolicyAnnotationsE2E003())).NotTo(HaveOccurred())

		By("确认 targetRef Deployment Pod metrics 仍可用（避免 Reconcile 与 metrics-server  scrape 竞态）")
		WaitForDeploymentPodMetricsReady(clientset, ctx, ns, MetricsTestDeployName, 3*time.Minute)

		By("等待下一次 Reconcile（generation 对齐后），验证非法项被忽略且 MetricsPipeline 仍为全量采集成功")
		VerifyESMetricsCollectionFullSuccess(dynamicClient, ctx, MetricsESName001, ns, 1, true)
	})

	// It("ES-E2E-004-BuildMetricsContext-非法/缺失 MetricsTrigger 的 CR 行为", func() {
	// 	Skip("P2 用例，暂时跳过")
	// })

	It("ES-E2E-005-GetMetrics-全量成功（detail 无部分失败）", func() {
		DeferCleanup(func() {
			CleanupMetricsFrameworkAfter005(dynamicClient, clientset, ctx)
		})

		targetRef := ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       MetricsTestDeployName005,
			Namespace:  ns,
		}
		singleMetric := []ESApaMetricEntry{DefaultResourceCPUMetric()}

		By("准备独立 targetRef Deployment（1 条 Resource/cpu 可采集指标）")
		EnsureESMetricsTestDeploymentReady(clientset, ctx, MetricsTestDeployName005, ns, 1, true, true)

		By("创建 apa MetricsTrigger ES（Metrics 仅 1 条 Resource/cpu）")
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName005, ns, targetRef, 1, 3, singleMetric)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName005, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName005, ns)

		By("轮询 MetricsPipeline，验证 GetMetrics 全量成功且无 PartialSuccess/CollectFailures")
		VerifyESGetMetricsFullSuccess(dynamicClient, ctx, MetricsESName005, ns, 1, true)
		VerifyHPANotExists(clientset, ctx, MetricsESName005+"-hpa", ns)
	})

	It("ES-E2E-006-GetMetrics-部分成功（CollectFailures→detail→condition）", func() {
		targetRef := ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       MetricsTestDeployName006,
			Namespace:  ns,
		}
		metrics := E2E006PartialSuccessExternalServerMetrics(ns)

		By("部署 ExternalServer 双端点（protocol=http）：Prometheus 标准成功响应 + 非 Prometheus 失败响应")
		EnsureOKPrometheusExternalServerReady(clientset, ctx, ns)
		EnsureBrokenExternalServerReady(clientset, ctx, ns)

		By("准备 targetRef Deployment")
		EnsureESMetricsTestDeploymentReady(clientset, ctx, MetricsTestDeployName006, ns, 1, false, false)

		By("创建 apa MetricsTrigger ES（infer_qps 成功端点 + broken_latency 失败端点）")
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName006, ns, targetRef, 1, 3, metrics)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName006, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName006, ns)

		By("轮询 MetricsPipeline，验证部分成功且 message 含 broken_latency (Error):")
		VerifyESGetMetricsPartialSuccess(dynamicClient, ctx, MetricsESName006, ns, MetricsExternalServerBrokenLatency)
		VerifyHPANotExists(clientset, ctx, MetricsESName006+"-hpa", ns)
	})

	It("ES-E2E-007-GetMetrics-全部指标条目采集失败（仍为 PartialSuccess 语义）", func() {
		DeferCleanup(func() {
			CleanupMetricsFrameworkAfter006To007(dynamicClient, clientset, ctx)
		})

		targetRef := ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       MetricsTestDeployName007,
			Namespace:  ns,
		}
		metrics := E2E007AllCollectFailureExternalServerMetric(ns)

		By("确保 ExternalServer 失败端点可用（复用 ES-E2E-006 已部署的 broken 端点）")
		EnsureBrokenExternalServerReady(clientset, ctx, ns)

		By("准备 targetRef Deployment")
		EnsureESMetricsTestDeploymentReady(clientset, ctx, MetricsTestDeployName007, ns, 1, false, false)

		By("创建 apa MetricsTrigger ES（Metrics 仅 1 条，指向失败端点）")
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName007, ns, targetRef, 1, 3, metrics)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName007, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName007, ns)

		By("轮询 MetricsPipeline，验证全失败仍为 PartialSuccess 且 message 含 broken_latency (Error):")
		VerifyESGetMetricsPartialSuccess(dynamicClient, ctx, MetricsESName007, ns, MetricsExternalServerBrokenLatency)
		VerifyHPANotExists(clientset, ctx, MetricsESName007+"-hpa", ns)
	})

	It("ES-E2E-008-部署推理服务后基于 ExternalServer 推理指标扩容", func() {
		SkipUnlessClusterPrometheusAvailable(clientset, ctx)

		targetRef := ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       MetricsInferenceDeployName,
			Namespace:  ns,
		}
		metrics := []ESApaMetricEntry{E2E008InferQPSExternalServerMetric()}
		const minReplicas int32 = 1
		const maxReplicas int32 = 3

		By("部署推理服务（replicas=1）并等待 Pod Ready（幂等，单跑 009 时复用已扩容 Deployment）")
		EnsureMetricsInferenceServiceReady(clientset, ctx, ns)

		By("发送推理探针请求，验证 HTTP 200")
		RunMetricsInferenceHTTPProbe(clientset, ctx, ns)

		By("启动持续推理压测")
		StartMetricsInferenceLoadTest(clientset, ctx, ns)

		By("创建 apa MetricsTrigger ES（ExternalServer/infer_qps → 集群 Prometheus；endpoint 默认 http://，HTTPS 集群可设 METRICS_PROMETHEUS_PROTOCOL=https 或 METRICS_PROMETHEUS_ENDPOINT_URL）")
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName008, ns, targetRef, minReplicas, maxReplicas, metrics)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName008, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName008, ns)

		By("轮询 MetricsPipeline / ScalingCalculated / desiredReplicas，验证 apa 扩容")
		WaitForESApaScaleUp(dynamicClient, clientset, ctx, MetricsESName008, ns, MetricsInferenceDeployName, minReplicas, maxReplicas, 5*time.Minute)
		VerifyHPANotExists(clientset, ctx, MetricsESName008+"-hpa", ns)

		StopMetricsInferenceLoadTest(clientset, ctx, ns)
	})

	It("ES-E2E-009-推理低流量场景缩容到 minReplicas", func() {
		DeferCleanup(func() {
			CleanupMetricsFrameworkAfter008To009(dynamicClient, clientset, ctx)
		})

		_, err := GetElasticScalerCR(dynamicClient, ctx, MetricsESName008, ns)
		if err != nil {
			Skip("依赖 ES-E2E-008 已成功创建 metrics-es-008，跳过 ES-E2E-009")
		}

		const minReplicas int32 = 1

		By("前置：确认 ES-E2E-008 扩容态仍成立（有序套件紧接 008，单次校验即可）")
		scaledDesired := GetESDesiredReplicas(dynamicClient, ctx, MetricsESName008, ns)
		Expect(scaledDesired).To(BeNumerically(">", minReplicas), "ES-E2E-008 应已扩容至 minReplicas 之上")
		VerifyESApaDesiredReplicasImmediate(dynamicClient, clientset, ctx, MetricsESName008, ns, MetricsInferenceDeployName, scaledDesired)

		By("停止推理压测，降低 infer_qps（Patch PromQL 为低流量查询）")
		StopMetricsInferenceLoadTest(clientset, ctx, ns)
		Expect(PatchESApaExternalServerPromQL(dynamicClient, ctx, MetricsESName008, ns, PrometheusInferQPSLowQuery())).NotTo(HaveOccurred())

		By("等待 Patch PromQL 后 Reconcile（generation 对齐）")
		WaitForESApaPromQLPatchObserved(dynamicClient, ctx, MetricsESName008, ns)

		By("发送少量推理探针，确认服务仍返回 HTTP 200")
		RunMetricsInferenceHTTPProbe(clientset, ctx, ns)

		By("等待 apa 缩容至 minReplicas（controller 默认 scaleDown 稳定窗口 180s、maxPodsPerStep=1，分步 3→2→1；默认超时 10m）")
		WaitForESApaScaleDown(dynamicClient, clientset, ctx, MetricsESName008, ns, MetricsInferenceDeployName, minReplicas, MetricsApaScaleDownWaitTimeout())
		VerifyESScalingCalculatedCondition(dynamicClient, ctx, MetricsESName008, ns, "True", MetricsReasonScalingSucceeded)
	})

	It("ES-E2E-010-Resource 指标依赖 ContextBuilder 注入 PodList", func() {
		DeferCleanup(func() {
			CleanupMetricsFrameworkAfter010(dynamicClient, clientset, ctx)
		})

		targetRef := ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       MetricsInferenceDeployName010,
			Namespace:  ns,
		}
		singleMetric := []ESApaMetricEntry{DefaultResourceCPUMetric()}
		const minReplicas int32 = 1
		const maxReplicas int32 = 3

		By("清理 ES-E2E-001~009 遗留 CR/workload，释放调度与 metrics-server 资源（010 不依赖前述 targetRef）")
		CleanupMetricsFrameworkPhase001To009Resources(dynamicClient, clientset, ctx)

		By("部署推理服务（轻量 CPU request）并等待 Pod metrics 稳定可采集（与 ES-E2E-001 一致，避免创建 ES 后长期 PartialSuccess）")
		EnsureMetricsInferenceService010Ready(clientset, ctx, ns)

		By("启动推理压测，制造 CPU 使用")
		StartMetricsInferenceLoadTest010(clientset, ctx, ns)

		By("创建 ES 前再次确认 Pod metrics（压测启动后 scrape 可能短暂抖动）")
		WaitForDeploymentPodMetricsReady(clientset, ctx, ns, MetricsInferenceDeployName010, 3*time.Minute)

		By("创建 apa MetricsTrigger ES（Resource/cpu，targetRef 指向推理 Deployment）")
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName010, ns, targetRef, minReplicas, maxReplicas, singleMetric)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName010, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName010, ns)

		By("发送推理探针，验证 HTTP 200")
		RunMetricsInferenceHTTPProbe010(clientset, ctx, ns)

		By("等待 Reconcile，验证 MetricsPipeline/ScalingCalculated 与 Deployment 副本对齐")
		resourceRecovery := &ResourceMetricsWaitRecovery{
			Clientset: clientset, Namespace: ns, DeployName: MetricsInferenceDeployName010,
		}
		WaitForESApaResourceMetricScaling(dynamicClient, clientset, ctx, MetricsESName010, ns, MetricsInferenceDeployName010, minReplicas, maxReplicas, resourceRecovery, MetricsPodMetricsWaitTimeout())
		VerifyHPANotExists(clientset, ctx, MetricsESName010+"-hpa", ns)

		StopMetricsInferenceLoadTest010(clientset, ctx, ns)
	})

	It("ES-E2E-011-ExternalServer.query 被正确拼接并参与推理指标查询", func() {
		DeferCleanup(func() {
			CleanupMetricsFrameworkAfter011(dynamicClient, clientset, ctx)
		})

		targetRef := ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       MetricsInferenceDeployName011,
			Namespace:  ns,
		}
		metrics := []ESApaMetricEntry{AuditPrometheusExternalServerMetric(ns)}
		const minReplicas int32 = 1
		const maxReplicas int32 = 3

		By("部署 Prometheus 审计端点（endpoint 含 tenant/namespace 固定查询参数；access log 输出 stdout）")
		EnsureAuditPrometheusExternalServerReady(clientset, ctx, ns)

		By("部署推理服务并向 Service 发送推理请求产生流量")
		EnsureMetricsInferenceService011Ready(clientset, ctx, ns)
		RunMetricsInferenceHTTPProbe011(clientset, ctx, ns)

		By("创建 apa MetricsTrigger ES（ExternalServer endpoint 固定参数 + query PromQL/step 分离；protocol=http）")
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName011, ns, targetRef, minReplicas, maxReplicas, metrics)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName011, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName011, ns)

		By("等待 Reconcile，验证 MetricsPipeline 全量采集成功且 ScalingCalculated 成功")
		VerifyESGetMetricsFullSuccess(dynamicClient, ctx, MetricsESName011, ns, 1, true)
		VerifyESScalingCalculatedCondition(dynamicClient, ctx, MetricsESName011, ns, "True", MetricsReasonScalingSucceeded)
		VerifyHPANotExists(clientset, ctx, MetricsESName011+"-hpa", ns)

		By("从审计端点 access log 确认 controller GET /api/v1/query URL 同时包含 endpoint 固定参数与 spec.query")
		VerifyExternalServerQueryAppendInAuditLogs(clientset, ctx, ns)
	})

	It("ES-E2E-012-Resource + ExternalServer 多指标全量成功", func() {
		SkipUnlessClusterPrometheusAvailable(clientset, ctx)
		DeferCleanup(func() {
			CleanupMetricsFrameworkAfter012(dynamicClient, clientset, ctx)
		})

		targetRef := ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       MetricsInferenceDeployName012,
			Namespace:  ns,
		}
		metrics := ApaResourceAndInferQPSMetrics()
		const minReplicas int32 = 1
		const maxReplicas int32 = 3

		By("部署推理服务（含 CPU request）并等待 Pod metrics 稳定可采集")
		EnsureMetricsInferenceService012Ready(clientset, ctx, ns)

		By("启动推理压测并发送探针，验证 HTTP 200")
		StartMetricsInferenceLoadTest012(clientset, ctx, ns)
		RunMetricsInferenceHTTPProbe012(clientset, ctx, ns)

		By("创建 ES 前再次确认 Pod metrics")
		WaitForDeploymentPodMetricsReady(clientset, ctx, ns, MetricsInferenceDeployName012, 3*time.Minute)

		By("创建 apa MetricsTrigger ES（Resource/cpu + ExternalServer/infer_qps）")
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName012, ns, targetRef, minReplicas, maxReplicas, metrics)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName012, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName012, ns)

		By("等待 Reconcile，验证双指标全量采集成功（processed 2 metric(s)）且 ScalingCalculated 成功")
		resourceRecovery := &ResourceMetricsWaitRecovery{
			Clientset: clientset, Namespace: ns, DeployName: MetricsInferenceDeployName012,
		}
		WaitForESApaMultiMetricsFullSuccess(dynamicClient, clientset, ctx, MetricsESName012, ns, MetricsInferenceDeployName012, ApaDualMetricsProcessedCount, minReplicas, maxReplicas, resourceRecovery, MetricsPodMetricsWaitTimeout())
		VerifyHPANotExists(clientset, ctx, MetricsESName012+"-hpa", ns)

		StopMetricsInferenceLoadTest012(clientset, ctx, ns)
	})

	It("ES-E2E-013-多指标部分失败时成功推理指标仍进入 apa 算法", func() {
		SkipUnlessClusterPrometheusAvailable(clientset, ctx)
		DeferCleanup(func() {
			CleanupMetricsFrameworkAfter013(dynamicClient, clientset, ctx)
		})

		targetRef := ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       MetricsInferenceDeployName013,
			Namespace:  ns,
		}
		metrics := PartialSuccessInferenceExternalServerMetrics(ns)
		const minReplicas int32 = 1
		const maxReplicas int32 = 3

		By("确保 broken_latency 失败 ExternalServer 端点可用（复用 ES-E2E-006）")
		EnsureBrokenExternalServerReady(clientset, ctx, ns)

		By("部署推理服务、启动压测并发送探针，验证 HTTP 200")
		EnsureMetricsInferenceService013Ready(clientset, ctx, ns)
		StartMetricsInferenceLoadTest013(clientset, ctx, ns)
		RunMetricsInferenceHTTPProbe013(clientset, ctx, ns)

		By("创建 apa MetricsTrigger ES（infer_qps→集群 Prometheus + broken_latency→失败端点）")
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName013, ns, targetRef, minReplicas, maxReplicas, metrics)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName013, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName013, ns)

		By("等待 Reconcile，验证 MetricsPartialSuccess 且 apa 仍输出 ScalingCalculated")
		WaitForESApaPartialSuccessWithScaling(dynamicClient, clientset, ctx, MetricsESName013, ns, MetricsInferenceDeployName013, []string{MetricsExternalServerBrokenLatency}, minReplicas, maxReplicas, 5*time.Minute)
		VerifyHPANotExists(clientset, ctx, MetricsESName013+"-hpa", ns)

		StopMetricsInferenceLoadTest013(clientset, ctx, ns)
	})

	It("ES-E2E-014-指标策略注解影响真实推理流量下的窗口聚合", func() {
		SkipUnlessClusterPrometheusAvailable(clientset, ctx)
		DeferCleanup(func() {
			CleanupMetricsFrameworkAfter014(dynamicClient, clientset, ctx)
		})

		targetRef := ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       MetricsInferenceDeployName014,
			Namespace:  ns,
		}
		const minReplicas int32 = 1
		const maxReplicas int32 = 3

		By("部署推理服务")
		EnsureMetricsInferenceService014Ready(clientset, ctx, ns)

		By("阶段一（低流量）：创建带 windowSeconds=120、aggregationType=Max 注解的 ES，PromQL 低流量")
		lowMetric := []ESApaMetricEntry{ExternalServerMetricWithPromQL(PrometheusInferQPSLowQuery())}
		Expect(CreateApaElasticScalerCRWithAnnotations(dynamicClient, ctx, MetricsESName014, ns, targetRef, minReplicas, maxReplicas, lowMetric, WindowMaxAggregationPolicyAnnotations())).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName014, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName014, ns)
		RunMetricsInferenceHTTPProbe014(clientset, ctx, ns)
		VerifyESPhaseMetricsCollectionSuccess(dynamicClient, ctx, MetricsESName014, ns, "低流量")
		WaitForESApaDesiredReplicas(dynamicClient, clientset, ctx, MetricsESName014, ns, MetricsInferenceDeployName014, minReplicas, true, 5*time.Minute)

		By("阶段二（高流量）：Patch PromQL 为高 infer_qps，启动压测并等待扩容")
		Expect(PatchESApaExternalServerPromQL(dynamicClient, ctx, MetricsESName014, ns, PrometheusInferQPSQuery())).NotTo(HaveOccurred())
		StartMetricsInferenceLoadTest014(clientset, ctx, ns)
		RunMetricsInferenceHTTPProbe014(clientset, ctx, ns)
		VerifyESPhaseMetricsCollectionSuccess(dynamicClient, ctx, MetricsESName014, ns, "高流量")
		WaitForESApaScaleUp(dynamicClient, clientset, ctx, MetricsESName014, ns, MetricsInferenceDeployName014, minReplicas, maxReplicas, 5*time.Minute)
		highPhaseDesired := GetESDesiredReplicas(dynamicClient, ctx, MetricsESName014, ns)
		Expect(highPhaseDesired).To(BeNumerically(">", minReplicas), "高流量阶段应触发 apa 扩容")

		By("阶段三（低流量）：停止压测并 Patch PromQL 为低流量，验证窗口 Max 仍保留高流量决策")
		StopMetricsInferenceLoadTest014(clientset, ctx, ns)
		Expect(PatchESApaExternalServerPromQL(dynamicClient, ctx, MetricsESName014, ns, PrometheusInferQPSLowQuery())).NotTo(HaveOccurred())
		RunMetricsInferenceHTTPProbe014(clientset, ctx, ns)
		WaitForESApaWindowMaxRetainedAfterLowTraffic(dynamicClient, clientset, ctx, MetricsESName014, ns, MetricsInferenceDeployName014, minReplicas, highPhaseDesired, 3*time.Minute)
		VerifyHPANotExists(clientset, ctx, MetricsESName014+"-hpa", ns)
	})

	It("ES-E2E-015-Custom 指标基于推理请求完成 apa 算法扩缩", func() {
		DeferCleanup(func() {
			CleanupMetricsFrameworkAfter015(dynamicClient, clientset, ctx)
		})

		targetRef := ESMetricsTargetRef{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       MetricsInferenceDeployName015,
			Namespace:  ns,
		}
		const minReplicas int32 = 1
		const maxReplicas int32 = 3

		By("部署推理服务并确认 Pod Ready")
		EnsureMetricsInferenceService015Ready(clientset, ctx, ns)

		By("启动推理压测，为 Custom 指标产生可观测流量")
		StartMetricsInferenceLoadTest015(clientset, ctx, ns)
		RunMetricsInferenceHTTPProbe015(clientset, ctx, ns)

		By("确认 custom.metrics.k8s.io 可查询该 Deployment 的 Custom 指标（无 adapter 时部署 mock APIService）")
		customMetricName := EnsureCustomMetricsReadyForDeployment(clientset, dynamicClient, ctx, ns, MetricsInferenceDeployName015)

		By("创建 apa MetricsTrigger ES（Custom 指标）")
		metrics := []ESApaMetricEntry{ApaCustomMetricEntry(customMetricName)}
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName015, ns, targetRef, minReplicas, maxReplicas, metrics)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName015, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName015, ns)

		By("发送推理请求并等待 Reconcile，验证 MetricsPipeline 与 ScalingCalculated")
		RunMetricsInferenceHTTPProbe015(clientset, ctx, ns)
		WaitForESApaCustomMetricScaling(dynamicClient, clientset, ctx, MetricsESName015, ns, MetricsInferenceDeployName015, minReplicas, maxReplicas, 5*time.Minute)
		VerifyHPANotExists(clientset, ctx, MetricsESName015+"-hpa", ns)

		StopMetricsInferenceLoadTest015(clientset, ctx, ns)
	})

	It("ES-E2E-016-ES targetRef 对接 RSG GroupReplication 的 CR 校验与指标采集基线", func() {
		SkipUnlessClusterPrometheusAvailable(clientset, ctx)

		By("清理 ES-E2E-010~015 遗留 workload，为 RSG 部署释放资源（各用例 DeferCleanup 已及时清理时可幂等）")
		CleanupMetricsFrameworkPhase010To015Resources(dynamicClient, clientset, ctx)

		const minReplicas int32 = 1
		const maxReplicas int32 = 3

		By("确认 ResourceScalingGroup 控制器可用（幂等，重跑时跳过 helm upgrade）")
		EnsureResourceScalingGroupControllerIfNeeded(clientset, ctx)

		By("部署 prefill/decode 目标 workload 并创建 GroupReplication RSG（groupConfig.replicas=1）")
		rsgName, prefillName, decodeName, err := DeployMetricsFrameworkRSGEnvironment(clientset, dynamicClient, ctx, ns)
		Expect(err).NotTo(HaveOccurred())

		By("确认 RSG Ready=True，group-0 目标 Deployment 已打标且 Pod Ready")
		WaitForMetricsFrameworkRSGReady(dynamicClient, clientset, ctx, rsgName, prefillName, decodeName, ns)

		targetRef := ESMetricsRSGTargetRef(rsgName, ns)
		metrics := []ESApaMetricEntry{RSGBaselineLowQPSExternalServerMetric()}

		By("创建 apa MetricsTrigger ES（targetRef=ResourceScalingGroup，ExternalServer/infer_qps）")
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName016, ns, targetRef, minReplicas, maxReplicas, metrics)).NotTo(HaveOccurred())

		By("等待 Reconcile，校验 Validated / MetricsPipeline / scale 子资源与 ES status")
		WaitForMetricsESOnRSGBaseline(dynamicClient, clientset, ctx, MetricsESName016, rsgName, ns, prefillName, decodeName, minReplicas, maxReplicas)
		VerifyHPANotExists(clientset, ctx, MetricsESName016+"-hpa", ns)
	})

	It("ES-E2E-017-apa 算法经 ES scale 子资源驱动 RSG 组数扩容并克隆新组", func() {
		SkipUnlessClusterPrometheusAvailable(clientset, ctx)

		_, err := GetElasticScalerCR(dynamicClient, ctx, MetricsESName016, ns)
		if err != nil {
			Skip("依赖 ES-E2E-016 已成功创建 metrics-es-016 与 metrics-rsg-016，跳过 ES-E2E-017")
		}

		const minReplicas int32 = 1
		const maxReplicas int32 = 3
		rsgName := MetricsFrameworkRSGName
		groupName := MetricsFrameworkRSGGroupName
		prefillName, decodeName, _, _, _ := MetricsFrameworkRSGDeploymentNames(ns)

		By("前置：复用 ES-E2E-016 基线（即时校验，避免重复 5 分钟 Eventually）")
		VerifyMetricsESOnRSGBaselineBeforeScaleUp(dynamicClient, clientset, ctx, MetricsESName016, rsgName, ns, prefillName, decodeName, minReplicas, maxReplicas)

		By("部署推理服务并启动持续压测，使 infer_qps 高于 apa 扩容阈值")
		TriggerRSGScaleUpLoadTest(dynamicClient, clientset, ctx, MetricsESName016, ns)
		defer StopMetricsInferenceLoadTest017(clientset, ctx, ns)

		By("轮询 ES status 与 RSG scale 子资源，验证组数扩至 >1 且 group 克隆就绪")
		scaledGroups := WaitForMetricsRSGScaleUp(
			dynamicClient, clientset, ctx,
			MetricsESName016, rsgName, ns, groupName, prefillName, decodeName,
			minReplicas, maxReplicas, 5*time.Minute,
		)
		Expect(scaledGroups).To(BeNumerically(">", int64(minReplicas)), "apa 扩容后 RSG 组数应大于 minReplicas")
		VerifyHPANotExists(clientset, ctx, MetricsESName016+"-hpa", ns)
	})

	It("ES-E2E-018-apa 算法驱动 RSG 组数缩容至 minReplicas", func() {
		_, err := GetElasticScalerCR(dynamicClient, ctx, MetricsESName016, ns)
		if err != nil {
			Skip("依赖 ES-E2E-016 已成功创建 metrics-es-016，跳过 ES-E2E-018")
		}

		const minReplicas int32 = 1
		rsgName := MetricsFrameworkRSGName
		groupName := MetricsFrameworkRSGGroupName
		prefillName, decodeName, prefillSvc, decodeSvc, _ := MetricsFrameworkRSGDeploymentNames(ns)

		priorScaledGroups := GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, ns)
		if priorScaledGroups <= int64(minReplicas) {
			Skip("依赖 ES-E2E-017 已成功将 RSG 扩容至 >1 组，跳过 ES-E2E-018")
		}
		priorScaledGroups = VerifyMetricsRSGScaledStateImmediate(dynamicClient, clientset, ctx, rsgName, ns, minReplicas)

		By("停止推理压测，降低 infer_qps（Patch PromQL 为低流量查询）")
		StopMetricsInferenceLoadTest017(clientset, ctx, ns)
		Expect(PatchESApaExternalServerPromQL(dynamicClient, ctx, MetricsESName016, ns, PrometheusInferQPSLowQuery())).NotTo(HaveOccurred())

		By("等待 Patch PromQL 后 Reconcile（generation 对齐）")
		WaitForESApaPromQLPatchObserved(dynamicClient, ctx, MetricsESName016, ns)

		By("发送少量推理探针，确认服务仍返回 HTTP 200")
		EnsureMetricsInferenceService017Ready(clientset, ctx, ns)
		RunMetricsInferenceHTTPProbe017(clientset, ctx, ns)

		By("等待 apa 缩容至 minReplicas（controller 默认 scaleDown 稳定窗口 180s、maxPodsPerStep=1，分步组数回落；默认超时 10m）")
		WaitForMetricsRSGScaleDown(
			dynamicClient, clientset, ctx,
			MetricsESName016, rsgName, ns, groupName, prefillName, decodeName, prefillSvc, decodeSvc,
			minReplicas, priorScaledGroups, MetricsApaScaleDownWaitTimeout(),
		)
		VerifyESLastScaleTimePresent(dynamicClient, ctx, MetricsESName016, ns)
		VerifyHPANotExists(clientset, ctx, MetricsESName016+"-hpa", ns)
	})

	It("ES-E2E-019-targetRef 为 RSG 时 Resource 指标聚合多组 Pod", func() {
		DeferCleanup(func() {
			CleanupMetricsFrameworkAfter019(dynamicClient, clientset, ctx)
		})

		const minReplicas int32 = 1
		const maxReplicas int32 = 1

		By("确认 ResourceScalingGroup 控制器可用（复用 ES-E2E-016 安装）")
		EnsureResourceScalingGroupControllerIfNeeded(clientset, ctx)

		By("部署带 CPU request 的 prefill/decode 并创建 GroupReplication RSG")
		rsgName, prefillName, decodeName, prefillSvc, decodeSvc, appName, err := DeployMetricsRSGResourceCPUTestEnvironment(clientset, dynamicClient, ctx, ns)
		Expect(err).NotTo(HaveOccurred())
		WaitForMetricsRSGResourceReady(dynamicClient, clientset, ctx, rsgName, prefillName, decodeName, ns)

		By("确认 RSG status.selector 非空且可解析，Pod 数≥1")
		selector := VerifyRSGStatusSelectorParsable(dynamicClient, ctx, rsgName, ns, appName)
		Expect(CountPodsMatchingSelector(clientset, ctx, ns, selector)).To(BeNumerically(">=", 2),
			"RSG selector 应至少覆盖 prefill/decode 各 1 个 Running Pod")

		By("向 prefill/decode Service 发 HTTP 请求，制造 CPU 使用")
		StartRSGPrefillDecodeHTTPLoad(clientset, ctx, ns, prefillSvc, decodeSvc)
		defer StopRSGPrefillDecodeHTTPLoad(clientset, ctx, ns)

		targetRef := ESMetricsRSGTargetRef(rsgName, ns)
		singleMetric := []ESApaMetricEntry{DefaultResourceCPUMetric()}

		By("创建 apa MetricsTrigger ES（targetRef=ResourceScalingGroup，metrics 仅 Resource/cpu）")
		Expect(CreateApaElasticScalerCR(dynamicClient, ctx, MetricsESName019, ns, targetRef, minReplicas, maxReplicas, singleMetric)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, MetricsESName019, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName019, ns)

		By("等待 Reconcile，验证 MetricsPipeline/ScalingCalculated 与 RSG 组数对齐")
		resourceRecovery := &ResourceMetricsWaitRecovery{
			Clientset:        clientset,
			Namespace:        ns,
			DeployName:       prefillName,
			ExtraDeployNames: []string{decodeName},
		}
		WaitForMetricsRSGResourceCPUMetricScaling(
			dynamicClient, clientset, ctx, MetricsESName019, rsgName, ns,
			minReplicas, maxReplicas, resourceRecovery, 5*time.Minute,
		)
		VerifyHPANotExists(clientset, ctx, MetricsESName019+"-hpa", ns)
	})

	It("ES-E2E-020-ES 写入 RSG scale 后双控制器状态收敛", func() {
		SkipUnlessClusterPrometheusAvailable(clientset, ctx)

		_, err := GetElasticScalerCR(dynamicClient, ctx, MetricsESName016, ns)
		if err != nil {
			Skip("依赖 ES-E2E-016 已成功创建 metrics-es-016 与 metrics-rsg-016，跳过 ES-E2E-020")
		}

		const minReplicas int32 = 1
		const maxReplicas int32 = 3
		rsgName := MetricsFrameworkRSGName
		groupName := MetricsFrameworkRSGGroupName
		prefillName, decodeName, _, _, _ := MetricsFrameworkRSGDeploymentNames(ns)

		By("记录扩缩前 RSG spec/status 与 scale 子资源基线")
		baseline := RecordRSGScaleConvergenceBaseline(dynamicClient, ctx, rsgName, ns)
		GinkgoWriter.Printf("扩缩前 baseline: groupConfig=%d current=%d scale.status=%d\n",
			baseline.GroupConfigReplicas, baseline.CurrentReplicas, baseline.ScaleStatusReplicas)
		Expect(baseline.GroupConfigReplicas).To(Equal(baseline.CurrentReplicas))
		Expect(baseline.GroupConfigReplicas).To(Equal(baseline.ScaleStatusReplicas))

		By("触发 ES 扩容：Patch 高 PromQL + 推理压测，使 desiredReplicas > 基线组数")
		TriggerRSGScaleUpLoadTest(dynamicClient, clientset, ctx, MetricsESName016, ns)
		defer StopMetricsInferenceLoadTest017(clientset, ctx, ns)

		By("5 分钟内轮询 ES/RSG，验证 UpdateReplicas 后 groupConfig/currentReplicas/scale 与 ES status 收敛")
		scaledGroups := WaitForESRSGScaleConvergence(
			dynamicClient, clientset, ctx,
			MetricsESName016, rsgName, ns, groupName, prefillName, decodeName,
			minReplicas, maxReplicas, baseline.GroupConfigReplicas, 5*time.Minute,
		)
		Expect(scaledGroups).To(BeNumerically(">", baseline.GroupConfigReplicas),
			"扩容后 RSG 组数应大于扩缩前基线 %d", baseline.GroupConfigReplicas)
		VerifyHPANotExists(clientset, ctx, MetricsESName016+"-hpa", ns)
	})

	It("ES-E2E-021-禁止两个 ElasticScaler 同时管理同一 RSG targetRef", func() {
		_, err := GetElasticScalerCR(dynamicClient, ctx, MetricsESName016, ns)
		if err != nil {
			Skip("依赖 ES-E2E-016 已成功创建 metrics-es-016（targetRef=RSG），跳过 ES-E2E-021")
		}

		const minReplicas int32 = 1
		const maxReplicas int32 = 3
		rsgName := MetricsFrameworkRSGName
		targetRef := ESMetricsRSGTargetRef(rsgName, ns)

		By("前置：确认 owner ES metrics-es-016 已 Validated 且 targetRef 指向 RSG")
		VerifyESValidatedCondition(dynamicClient, ctx, MetricsESName016, ns)

		By("尝试创建第二条 ES（相同 targetRef，metrics 改用 Resource/cpu）")
		createErr := AttemptCreateDuplicateRSGTargetRefES(
			dynamicClient, ctx, MetricsESName021, ns, targetRef, minReplicas, maxReplicas,
		)

		By("校验第二条 ES 因 targetRef 一一对应约束被拒绝")
		VerifyDuplicateRSGTargetRefESRejected(dynamicClient, ctx, MetricsESName021, MetricsESName016, ns, createErr)

		By("确认 owner ES 继续 Reconcile 正常，MetricsPipeline/ScalingCalculated 不受破坏")
		VerifyOwnerESRSGTargetRefUnaffected(dynamicClient, ctx, MetricsESName016, ns)
		VerifyHPANotExists(clientset, ctx, MetricsESName016+"-hpa", ns)
	})

	AfterAll(func() {
		E2EProgressLog("[metrics-framework AfterAll] 清理测试 CR/Deployment 并卸载 helm controller...")
		CleanupMetricsFrameworkTest(dynamicClient, clientset, ctx)
	})
})
