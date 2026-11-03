package elastic_scaler

import (
	"fmt"
	"sync"
	"time"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// 指标驱动 ES CR 功能测试（参考 exam_design test/test_3）
// 前置：ElasticScaler controller 正常运行；应用 test/test_3 内 yaml 思路
var _ = Describe("Elastic Scaler 指标驱动 ES CR 功能测试", Ordered, Label("elastic-scaler", "with-workload-cluster"), func() {
	ns := ElasticScalerTestNamespace
	singleMetric := []ESMetricSpec{
		{Type: "Resource", ResourceName: "cpu", TargetType: "Utilization", AverageUtilization: 1},
	}
	multiMetrics := []ESMetricSpec{
		{Type: "Resource", ResourceName: "cpu", TargetType: "Utilization", AverageUtilization: 1},
		{Type: "Resource", ResourceName: "memory", TargetType: "Utilization", AverageUtilization: 80},
	}
	crudESName := "elastic-scaler-crud"
	targetRef := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: ns}

	BeforeAll(func() {
		Expect(EnsureNamespace(clientset, ctx, ns)).NotTo(HaveOccurred())
	})

	It("1-安装部署-控制器部署与健康验证", func() {
		Expect(WaitForNamespaceDeletionIfTerminating(clientset, ctx, ElasticScalerHelmNamespace, 3*time.Minute, 5*time.Second)).NotTo(HaveOccurred())
		Expect(InstallElasticScaler()).NotTo(HaveOccurred())
		VerifyHelmReleaseStatus(ElasticScalerHelmReleaseName, ElasticScalerHelmNamespace)
		VerifyElasticScalerPodsRunning(clientset, ctx, 5*time.Minute, 10*time.Second)
	})

	It("2-功能验证-基于指标驱动的ES CR创建验证", func() {
		Expect(CreateESMetricsTestDeployment(clientset, ctx, ESTestDeploymentName, ns, 1, true)).NotTo(HaveOccurred())
		time.Sleep(15 * time.Second)
		// 初始设置 maxReplicas 为 2，后续用例中再扩展到 3，避免产生过多副本
		Expect(CreateElasticScalerCRMetricsTrigger(dynamicClient, ctx, ESTestCRName, ns, targetRef, 1, 2, singleMetric)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, ESTestCRName, ns)
	})

	It("3-兼容性测试-指标驱动支持 Deployment 目标类型", func() {
		VerifyHPATargetRefMatchesES(clientset, dynamicClient, ctx, ESTestCRName, ns)
	})

	It("4-功能测试-CR创建后状态校验更新", func() {
		VerifyESValidatedCondition(dynamicClient, ctx, ESTestCRName, ns)
	})

	It("5-功能测试-HPA资源创建校验(集群中无对应HPA)", func() {
		VerifyHPACreated(clientset, ctx, ESTestCRName+"-hpa", ns)
	})

	It("6-功能测试-HPA目标资源指向校验", func() {
		VerifyHPATargetRefMatchesES(clientset, dynamicClient, ctx, ESTestCRName, ns)
	})

	It("7-功能测试-多指标配置校验", func() {
		Expect(UpdateESMetrics(dynamicClient, ctx, ESTestCRName, ns, multiMetrics)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, ESTestCRName, ns)
	})

	It("8-功能测试-校验HPA目标副本数与ES状态同步", func() {
		VerifyESStatusReplicasMatchHPA(dynamicClient, clientset, ctx, ESTestCRName, ns)
	})

	It("9-功能测试-校验HPA Conditions与ES Conditions 同步", func() {
		VerifyESConditionsMatchHPA(dynamicClient, clientset, ctx, ESTestCRName, ns)
	})

	It("10-功能测试-Resource类型指标映射到HPA", func() {
		VerifyHPAResourceMetricCPU(dynamicClient, clientset, ctx, ESTestCRName, ns)
		VerifyHPAResourceMetricMemory(dynamicClient, clientset, ctx, ESTestCRName, ns)
	})

	It("11-功能测试-更新ES后HPA同步更新", func() {
		// 将 maxReplicas 从 2 更新为 3，验证 HPA 能同步更新配置
		Expect(UpdateESMaxReplicas(dynamicClient, ctx, ESTestCRName, ns, 3)).NotTo(HaveOccurred())
		time.Sleep(30 * time.Second)
		VerifyHPAMinMaxReplicas(dynamicClient, clientset, ctx, ESTestCRName, ns)
	})

	It("12-功能测试-指标驱动资源扩容", func() {
		// Deployment 已为高 CPU 容器，等待 HPA 扩容到最大值（当前 maxReplicas=3）
		WaitForDeploymentReplicas(clientset, ctx, ESTestDeploymentName, ns, 3, 5*time.Minute)
	})

	It("13-功能测试-HPA LastScaleTime 同步到 ES.Status", func() {
		VerifyESLastScaleTimeMatchHPA(dynamicClient, clientset, ctx, ESTestCRName, ns)
	})

	It("14-功能测试-Controller 重启后的恢复能力", func() {
		// 前置：已有 ES 与 HPA（前序用例已创建）
		Expect(RestartElasticScalerController(clientset, ctx, 5*time.Minute)).NotTo(HaveOccurred())
		// 重启后再次更新 ES，验证 HPA 仍能同步更新
		Expect(UpdateESMaxReplicas(dynamicClient, ctx, ESTestCRName, ns, 5)).NotTo(HaveOccurred())
		// 等待 controller 完成一次 Reconcile，将 ES 变更同步到 HPA
		time.Sleep(30 * time.Second)
		VerifyHPAMinMaxReplicas(dynamicClient, clientset, ctx, ESTestCRName, ns)
	})

	It("15-功能测试-极端副本数（大 MaxReplicas）", func() {
		// 独立 Deployment + targetRef：主用例 ESTestCRName 已占用 ESTestDeploymentName，Validator 要求同一 target 仅允许一个 ES
		Expect(CreateESMetricsTestDeployment(clientset, ctx, ESTestDeploymentNameBigMax, ns, 1, true)).NotTo(HaveOccurred())
		bigMaxTarget := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentNameBigMax, Namespace: ns}
		Expect(CreateElasticScalerCRMetricsTrigger(dynamicClient, ctx, ESTestCRNameBigMax, ns, bigMaxTarget, 1, 50, singleMetric)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameBigMax, ns)
		time.Sleep(30 * time.Second)
		VerifyHPAMinMaxReplicas(dynamicClient, clientset, ctx, ESTestCRNameBigMax, ns)
		// 用例验证完成后及时删除该 ES，避免长时间保留大 MaxReplicas 的 HPA 浪费资源
		Expect(DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameBigMax, ns)).NotTo(HaveOccurred())
		// 等 HPA GC 完成后再缩回副本，避免与 HPA 并发 Update 触发 409
		VerifyHPANotExists(clientset, ctx, ESTestCRNameBigMax+"-hpa", ns)
		Expect(UpdateDeploymentReplicas(clientset, ctx, ESTestDeploymentNameBigMax, ns, 1)).NotTo(HaveOccurred())
		// 主 Deployment 由前序 Ordered 用例创建；--focus 单跑本用例时可能不存在
		Expect(UpdateDeploymentReplicasIfExists(clientset, ctx, ESTestDeploymentName, ns, 1)).NotTo(HaveOccurred())
	})

	It("16-功能测试-集群已有HPA资源, HPA资源更新", func() {
		// 1. 为新的 Deployment 创建预先存在的同名 HPA（maxReplicas 设为 3，避免浪费资源）
		Expect(CreateESMetricsTestDeployment(clientset, ctx, ESTestDeploymentNameExistingHPA, ns, 1, false)).NotTo(HaveOccurred())
		Expect(CreateOrUpdatePlainHPA(clientset, ctx, ESTestCRNameExistingHPA, ns, "Deployment", ESTestDeploymentNameExistingHPA, 1, 3, 1)).NotTo(HaveOccurred())

		// 2. 创建同名 ES（MetricsTrigger/HPA），maxReplicas 与 HPA 对齐为 3
		existingHPATarget := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentNameExistingHPA, Namespace: ns}
		Expect(CreateElasticScalerCRMetricsTrigger(dynamicClient, ctx, ESTestCRNameExistingHPA, ns, existingHPATarget, 1, 3, singleMetric)).NotTo(HaveOccurred())

		// 3. 检查 HPA 被 Controller 使用并与 ES 配置对齐（不产生重复 HPA）
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameExistingHPA, ns)
		time.Sleep(30 * time.Second)
		VerifyHPAMinMaxReplicas(dynamicClient, clientset, ctx, ESTestCRNameExistingHPA, ns)
		VerifyHPAResourceMetricCPU(dynamicClient, clientset, ctx, ESTestCRNameExistingHPA, ns)
	})

	It("17-兼容性测试-指标驱动支持 StatefulSet 目标类型", func() {
		// 1. 创建 StatefulSet 目标
		Expect(CreateStatefulSetForES(clientset, ctx, ESTestStatefulSetName, ns, 1)).NotTo(HaveOccurred())

		// 2. 创建 ES，TargetRef 指向 StatefulSet，maxReplicas 设为 3 即可验证行为
		stsTarget := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "StatefulSet", Name: ESTestStatefulSetName, Namespace: ns}
		Expect(CreateElasticScalerCRMetricsTrigger(dynamicClient, ctx, ESTestCRNameStatefulSet, ns, stsTarget, 1, 3, singleMetric)).NotTo(HaveOccurred())

		// 3. 验证生成的 HPA targetRef 正确指向 StatefulSet
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameStatefulSet, ns)
		VerifyHPATargetRefMatchesES(clientset, dynamicClient, ctx, ESTestCRNameStatefulSet, ns)
	})

	It("18-功能测试-多 ElasticScaler 并存互不影响", func() {
		// 1. 验证每个 ES 仅影响自己的 HPA/TargetRef
		VerifyESCRExists(dynamicClient, ctx, ESTestCRName, ns)
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameStatefulSet, ns)
		VerifyHPATargetRefMatchesES(clientset, dynamicClient, ctx, ESTestCRName, ns)
		VerifyHPATargetRefMatchesES(clientset, dynamicClient, ctx, ESTestCRNameStatefulSet, ns)
	})

	It("19-基本能力-ElasticScaler CR CRUD 能力", func() {
		// 1. 为 CRUD 用例创建单独的目标 Deployment，避免与主用 ES 共享同一 targetRef
		Expect(CreateESMetricsTestDeployment(clientset, ctx, ESTestDeploymentNameCRUD, ns, 1, false)).NotTo(HaveOccurred())
		crudTarget := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentNameCRUD, Namespace: ns}
		Expect(CreateElasticScalerCRMetricsTrigger(dynamicClient, ctx, crudESName, ns, crudTarget, 1, 2, singleMetric)).NotTo(HaveOccurred())

		// 2. 列表查询应包含该 ES
		names, err := ListElasticScalerCRNames(dynamicClient, ctx, ns)
		Expect(err).NotTo(HaveOccurred())
		Expect(names).To(ContainElement(crudESName))

		// 3. 单个查询
		_, err = GetElasticScalerCR(dynamicClient, ctx, crudESName, ns)
		Expect(err).NotTo(HaveOccurred())

		// 4. 删除 ES
		Expect(DeleteElasticScalerCR(dynamicClient, ctx, crudESName, ns)).NotTo(HaveOccurred())
	})

	It("20-功能测试-已删除 ES 的 Reconcile 处理", func() {
		// 1. 再次创建同名 ES，等待至少一次 Reconcile 完成（存在即视为成功）
		crudTarget := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentNameCRUD, Namespace: ns}
		Expect(CreateElasticScalerCRMetricsTrigger(dynamicClient, ctx, crudESName, ns, crudTarget, 1, 2, singleMetric)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, crudESName, ns)

		// 2. 删除该 ES
		Expect(DeleteElasticScalerCR(dynamicClient, ctx, crudESName, ns)).NotTo(HaveOccurred())

		// 3. 确认 ES 最终被删除（controller 对已删除对象的 Reconcile 不会报错/Crash）
		Eventually(func() bool {
			_, err := GetElasticScalerCR(dynamicClient, ctx, crudESName, ns)
			return err != nil
		}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "ElasticScaler %s/%s 应被删除", ns, crudESName)
	})

	It("21-功能测试-MetricsTrigger 删除 ES 后 HPA 自动清理", func() {
		// 1. 使用主用 ES（ESTestCRName），保证 ES 与 HPA 均已存在
		VerifyESCRExists(dynamicClient, ctx, ESTestCRName, ns)
		VerifyHPACreated(clientset, ctx, ESTestCRName+"-hpa", ns)

		// 2. 删除 ES
		Expect(DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRName, ns)).NotTo(HaveOccurred())

		// 3. 校验对应 HPA 被自动清理
		VerifyHPANotExists(clientset, ctx, ESTestCRName+"-hpa", ns)
	})

	It("22-输入测试-Metrics 列表为空时的拦截", func() {
		// 双路径：部分集群已挂 Validating Webhook（创建即失败）；未挂 Webhook 时 CR 可写入，由 Reconcile 侧 Validator 拒绝并避免创建 HPA
		emptyMetrics := []ESMetricSpec{}
		err := CreateElasticScalerCRMetricsTrigger(dynamicClient, ctx, ESTestCRNameMetricsEmpty, ns, targetRef, 1, 2, emptyMetrics)
		if err != nil {
			GinkgoWriter.Printf("metrics 为空已在准入阶段被拒绝: %v\n", err)
			return
		}
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameMetricsEmpty, ns)
		VerifyESValidationFailedCondition(dynamicClient, ctx, ESTestCRNameMetricsEmpty, ns)
		VerifyHPANotExists(clientset, ctx, ESTestCRNameMetricsEmpty+"-hpa", ns)
		Expect(DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameMetricsEmpty, ns)).NotTo(HaveOccurred())
	})

	It("23-输入验证-未知 Trigger.Type 处理", func() {
		// 1. 配置 triggerType 为 unknownTriggerType，期望 CRD/Admission 拒绝创建
		target := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: ns}
		err := CreateElasticScalerCRWithUnknownTriggerType(dynamicClient, ctx, ESTestCRNameUnknownTriggerType, ns, target)
		Expect(err).To(HaveOccurred())
	})

	It("24-输入验证-MetricsTrigger 类型但 ScalingAlgorithm 缺失", func() {
		// 1. 创建 Trigger.Type=MetricsTrigger 但缺失 scalingAlgorithm 的 ES，期望被 CRD 拒绝
		target := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: ns}
		err := CreateElasticScalerCRMetricsTriggerNoScalingAlgorithm(dynamicClient, ctx, ESTestCRNameNoScalingAlgorithm, ns, target)
		Expect(err).To(HaveOccurred())
	})

	It("25-输入验证-SignalTrigger 类型但 endpoint 缺失时的处理", func() {
		// 1. 创建 Trigger.Type=SignalTrigger 且缺失 endpoint 的 ES，期望被 CRD 拒绝
		target := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: ns}
		err := CreateSignalTriggerESWithoutEndpoint(dynamicClient, ctx, ESTestCRNameSignalNoEndpoint, ns, target)
		Expect(err).To(HaveOccurred())
	})

	It("26-输入验证-Resource 类型但未填 resource 指标字段时的拦截", func() {
		target := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: ns}
		err := CreateElasticScalerCRMetricMissingResource(dynamicClient, ctx, ESTestCRNameMetricNoResource, ns, target)
		if err != nil {
			GinkgoWriter.Printf("非法 Resource 指标已在准入阶段被拒绝: %v\n", err)
			return
		}
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameMetricNoResource, ns)
		VerifyESValidationFailedCondition(dynamicClient, ctx, ESTestCRNameMetricNoResource, ns)
		VerifyHPANotExists(clientset, ctx, ESTestCRNameMetricNoResource+"-hpa", ns)
		Expect(DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameMetricNoResource, ns)).NotTo(HaveOccurred())
	})

	It("27-功能测试-External 类型指标映射到 HPA", func() {
		// External 指标使用 K8s MetricTarget（type/value）；HPA 路径已支持 External 指标源
		// 单独 focus 本用例时 test 2 的主 ES 可能仍存在，先清理避免 targetRef 一一对应冲突
		_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRName, ns)

		target := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: ns}
		Expect(CreateElasticScalerCRWithExternalMetric(dynamicClient, ctx, ESTestCRNameExternalMetric, ns, target)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameExternalMetric, ns)
		VerifyESValidatedCondition(dynamicClient, ctx, ESTestCRNameExternalMetric, ns)
		VerifyHPACreated(clientset, ctx, ESTestCRNameExternalMetric+"-hpa", ns)
		Expect(DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameExternalMetric, ns)).NotTo(HaveOccurred())
	})

	It("28-未实现路径拦截-MetricsTrigger 使用自定义算法名", func() {
		// 1. 创建 scalingAlgorithm=MyAlgo 的 MetricsTrigger ES
		target := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: ns}
		Expect(CreateElasticScalerCRMetricsTriggerCustomAlgorithm(dynamicClient, ctx, ESTestCRNameCustomAlgorithm, ns, target)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameCustomAlgorithm, ns)

		Expect(DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameCustomAlgorithm, ns)).NotTo(HaveOccurred())
	})

	It("29-未实现路径拦截-SignalTrigger 未实现的异常路径", func() {
		// 1. 创建正常配置的 SignalTrigger ES
		target := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: ns}
		Expect(CreateSignalTriggerES(dynamicClient, ctx, ESTestCRNameSignal, ns, target)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameSignal, ns)

		Expect(DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameSignal, ns)).NotTo(HaveOccurred())
	})

	It("30-功能测试-HPA 不支持 ExternalServer 类型", func() {
		target := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: ns}
		err := CreateElasticScalerCRWithExternalServerMetric(dynamicClient, ctx, ESTestCRNameExternalServer, ns, target)
		if err != nil {
			GinkgoWriter.Printf("HPA+ExternalServer 已在准入阶段被拒绝: %v\n", err)
			return
		}
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameExternalServer, ns)
		VerifyESValidationFailedCondition(dynamicClient, ctx, ESTestCRNameExternalServer, ns)
		VerifyHPANotExists(clientset, ctx, ESTestCRNameExternalServer+"-hpa", ns)
		Expect(DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameExternalServer, ns)).NotTo(HaveOccurred())
	})

	It("31-功能测试-目标资源与ES跨命名空间管理", func() {
		// 1. 创建 namespace A、B
		Expect(EnsureNamespace(clientset, ctx, ESTestNamespaceA)).NotTo(HaveOccurred())
		Expect(EnsureNamespace(clientset, ctx, ESTestNamespaceB)).NotTo(HaveOccurred())

		// 2. 在 namespace B 中创建目标 Deployment
		Expect(CreateESMetricsTestDeployment(clientset, ctx, ESTestCrossNamespaceDeploy, ESTestNamespaceB, 1, false)).NotTo(HaveOccurred())

		// 3. 在 namespace A 中创建 ES，TargetRef 指向 B 中的 Deployment
		target := ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestCrossNamespaceDeploy, Namespace: ESTestNamespaceB}
		Expect(CreateElasticScalerCRMetricsTrigger(dynamicClient, ctx, ESTestCRNameCrossNamespace, ESTestNamespaceA, target, 1, 2, singleMetric)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameCrossNamespace, ESTestNamespaceA)

		// 4. 等待 Reconcile，期望 Controller 明确拒绝跨命名空间 HPA，同名 HPA 在 B 命名空间中不存在
		time.Sleep(30 * time.Second)
		VerifyHPANotExists(clientset, ctx, ESTestCRNameCrossNamespace+"-hpa", ESTestNamespaceB)

		// 简单清理跨命名空间用例生成的对象
		Expect(DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameCrossNamespace, ESTestNamespaceA)).NotTo(HaveOccurred())
	})

	It("32-功能测试-StateTrigger 类型 ES 创建", func() {
		// 1. 使用 ES 测试专用配置：创建 Deployment 与 Tidal（与 tidal 用例隔离）
		Expect(CreateTestDeployment(clientset, ctx, ESTestStateTriggerDeploymentName, ns, 1)).NotTo(HaveOccurred())
		tidalTarget := TidalTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestStateTriggerDeploymentName, Namespace: ns}
		// 配置一个不会影响当前副本数的占位规则：在“当前时间 + 10s”触发，replicas 与初始副本数一致（1）
		now := time.Now().Add(10 * time.Second)
		cronExpr := fmt.Sprintf("CRON_TZ=Asia/Shanghai %d %d %d * * *", now.Second(), now.Minute(), now.Hour())
		placeholderRules := []TidalRule{
			{Name: "state-placeholder", Cron: cronExpr, Replicas: 1, Description: "placeholder rule for StateTrigger e2e"},
		}
		Expect(CreateTidalCRWithTargetAndRules(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, tidalTarget, placeholderRules)).NotTo(HaveOccurred())
		Expect(UpdateTidalLabels(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, map[string]string{"elasticscaler.io/name": ESTestCRNameStateBasic})).NotTo(HaveOccurred())
		time.Sleep(10 * time.Second)
		VerifyTidalCRExists(dynamicClient, ctx, ESTestStateTriggerTidalName, ns)

		// 2. 创建 StateTrigger 类型 ES，TargetRef 指向上述 Tidal
		Expect(CreateStateTriggerESForTidal(dynamicClient, ctx, ESTestCRNameStateBasic, ns, ESTestStateTriggerTidalName, ns)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameStateBasic, ns)
	})

	It("33-功能测试-StateTrigger 放大场景", func() {
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameStateBasic, ns)
		VerifyTidalCRExists(dynamicClient, ctx, ESTestStateTriggerTidalName, ns)

		Expect(UpdateTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 3)).NotTo(HaveOccurred())
		VerifyTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 3)
		WaitForDeploymentReplicas(clientset, ctx, ESTestStateTriggerDeploymentName, ns, 3, 3*time.Minute)
	})

	It("34-功能测试-StateTrigger 缩容场景", func() {
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameStateBasic, ns)
		VerifyTidalCRExists(dynamicClient, ctx, ESTestStateTriggerTidalName, ns)

		WaitForDeploymentReplicas(clientset, ctx, ESTestStateTriggerDeploymentName, ns, 3, 3*time.Minute)
		Expect(UpdateTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 1)).NotTo(HaveOccurred())
		VerifyTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 1)
		WaitForDeploymentReplicas(clientset, ctx, ESTestStateTriggerDeploymentName, ns, 1, 3*time.Minute)
	})

	It("35-功能测试-StateTrigger 等值场景不重复缩放", func() {
		WaitForDeploymentReplicas(clientset, ctx, ESTestStateTriggerDeploymentName, ns, 1, 3*time.Minute)
		Expect(UpdateTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 1)).NotTo(HaveOccurred())
		VerifyTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 1)
		WaitForDeploymentReplicas(clientset, ctx, ESTestStateTriggerDeploymentName, ns, 1, 3*time.Minute)
	})

	It("36-功能测试-StateTrigger 缩容到零", func() {
		Expect(UpdateTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 0)).NotTo(HaveOccurred())
		VerifyTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 0)
		WaitForDeploymentReplicas(clientset, ctx, ESTestStateTriggerDeploymentName, ns, 0, 3*time.Minute)
	})

	It("37-功能测试-StateTrigger 从零扩容", func() {
		WaitForDeploymentReplicas(clientset, ctx, ESTestStateTriggerDeploymentName, ns, 0, 3*time.Minute)
		Expect(UpdateTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 2)).NotTo(HaveOccurred())
		VerifyTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 2)
		WaitForDeploymentReplicas(clientset, ctx, ESTestStateTriggerDeploymentName, ns, 2, 3*time.Minute)
	})

	It("38-功能测试-StateTriggerWatcher 监听状态变化", func() {
		Expect(UpdateTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 3)).NotTo(HaveOccurred())
		VerifyTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 3)
		WaitForDeploymentReplicas(clientset, ctx, ESTestStateTriggerDeploymentName, ns, 3, 3*time.Minute)
		Expect(UpdateTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 1)).NotTo(HaveOccurred())
		VerifyTidalDesiredReplicas(dynamicClient, ctx, ESTestStateTriggerTidalName, ns, 1)
		WaitForDeploymentReplicas(clientset, ctx, ESTestStateTriggerDeploymentName, ns, 1, 3*time.Minute)
	})

	It("39-功能测试-StateTrigger 目标资源不存在", func() {
		// Webhook 不做集群态校验，Create 可成功；Reconcile 中 Validator 在校验 targetRef 存在性时失败，写入 Validated=False（非 StateTrigger 路径）
		Expect(CreateStateTriggerESMissingTarget(dynamicClient, ctx, ESTestCRNameStateMissingTarget, ns)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameStateMissingTarget, ns)
		VerifyESValidationFailedCondition(dynamicClient, ctx, ESTestCRNameStateMissingTarget, ns)
	})

	It("40-多功能测试-StateTrigger 执行扩缩资源不存在", func() {
		// 1. 前置：已有 StateTrigger ES 及其 Tidal，且首次状态已更新
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameStateBasic, ns)
		VerifyTidalCRExists(dynamicClient, ctx, ESTestStateTriggerTidalName, ns)
		VerifyTidalCRStatusUpdated(dynamicClient, ctx, ESTestStateTriggerTidalName, ns)

		// 2. 将 Tidal 的 targetRef 指向不存在的目标，模拟执行扩缩过程中目标被删除
		Expect(UpdateTidalTargetToNonExist(dynamicClient, ctx, ESTestStateTriggerTidalName, ns)).NotTo(HaveOccurred())

		// 3. 等待 Reconcile，期望 ES 在 StateTrigger 路径上标记失败（GetCurrentReplicas 等内部错误可通过条件观察）
		VerifyESStateTriggerFailed(dynamicClient, ctx, ESTestCRNameStateBasic, ns)
	})

	It("41-多功能测试-StateTrigger Watcher 并发入队压力（高频目标变更）", func() {
		// 1. 前置：确保 StateTrigger ES 与 Tidal 存在
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameStateBasic, ns)
		VerifyTidalCRExists(dynamicClient, ctx, ESTestStateTriggerTidalName, ns)

		// 2. 对目标 Deployment 持续执行 annotation patch（每秒一次，持续约 1 分钟），模拟高频入队压力
		const workers = 2
		const rounds = 60
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				for r := 0; r < rounds; r++ {
					key := fmt.Sprintf("e2e-statetrigger-race-%d", id)
					_ = PatchDeploymentAnnotation(clientset, ctx, ESTestStateTriggerDeploymentName, ns, key, fmt.Sprintf("v-%d", r))
					time.Sleep(1 * time.Second)
				}
			}(w)
		}
		wg.Wait()

		// 3. 用例关注 Controller 稳定性：不崩溃、不 OOM（若发生崩溃，Suite 会在后续用例中暴露）
	})

	It("42-竞态测试-同一 ES Spec 并发更新（最后写入收敛）", func() {
		// 1. 复用已有 Deployment，创建专用于竞态测试的 ES（参考 test_24）
		Expect(CreateElasticScalerCRMetricsTrigger(dynamicClient, ctx, ESTestCRNameConcurrentUpdate, ns, targetRef, 1, 4, singleMetric)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameConcurrentUpdate, ns)
		time.Sleep(15 * time.Second) // 等待 HPA 创建

		// 2. 并发更新：多 goroutine 交替写配置 A（maxReplicas=5）与 B（maxReplicas=8）
		const workers = 4
		const rounds = 8
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				for r := 0; r < rounds; r++ {
					if id%2 == 0 {
						_ = UpdateESMaxReplicas(dynamicClient, ctx, ESTestCRNameConcurrentUpdate, ns, 5)
					} else {
						_ = UpdateESMaxReplicas(dynamicClient, ctx, ESTestCRNameConcurrentUpdate, ns, 8)
					}
					time.Sleep(50 * time.Millisecond)
				}
			}(w)
		}
		wg.Wait()

		// 3. 最后执行一次明确的“最终配置”
		Expect(UpdateESMaxReplicas(dynamicClient, ctx, ESTestCRNameConcurrentUpdate, ns, 3)).NotTo(HaveOccurred())

		// 4. 等待 1-2 个 Reconcile 周期，检查 HPA spec 收敛到最后一次写入
		time.Sleep(45 * time.Second)
		VerifyHPAMinMaxReplicas(dynamicClient, clientset, ctx, ESTestCRNameConcurrentUpdate, ns)
		Expect(DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameConcurrentUpdate, ns)).NotTo(HaveOccurred())
	})

	It("43-竞态测试-删除 ES 与并发更新/轮询入队（删除竞态）", func() {
		// 1. 创建专用于删除竞态测试的 ES，复用已有 Deployment
		Expect(CreateElasticScalerCRMetricsTrigger(dynamicClient, ctx, ESTestCRNameConcurrentDelete, ns, targetRef, 1, 2, singleMetric)).NotTo(HaveOccurred())
		VerifyESCRExists(dynamicClient, ctx, ESTestCRNameConcurrentDelete, ns)
		VerifyHPACreated(clientset, ctx, ESTestCRNameConcurrentDelete+"-hpa", ns)

		// 2. 并发：一边循环轻量 patch（annotation），一边循环 delete ES
		const updateWorkers = 3
		const deleteWorkers = 2
		const rounds = 12
		var wg sync.WaitGroup
		for w := 0; w < updateWorkers; w++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				for r := 0; r < rounds; r++ {
					_ = PatchESAnnotation(dynamicClient, ctx, ESTestCRNameConcurrentDelete, ns, "e2e-race", "v")
					time.Sleep(30 * time.Millisecond)
				}
			}(w)
		}
		for w := 0; w < deleteWorkers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for r := 0; r < rounds; r++ {
					_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameConcurrentDelete, ns)
					time.Sleep(40 * time.Millisecond)
				}
			}()
		}
		wg.Wait()

		// 3. 确保 ES 已删除（若某次 delete 未生效则再删一次）
		_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameConcurrentDelete, ns)

		// 4. 校验：ES 不存在，HPA 被 GC 或不再被创建
		Eventually(func() bool {
			_, err := GetElasticScalerCR(dynamicClient, ctx, ESTestCRNameConcurrentDelete, ns)
			return err != nil
		}, 1*time.Minute, 5*time.Second).Should(BeTrue(), "ES 应在删除竞态后不存在")
		VerifyHPANotExists(clientset, ctx, ESTestCRNameConcurrentDelete+"-hpa", ns)
		Expect(DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameConcurrentDelete, ns)).NotTo(HaveOccurred())
	})

	AfterAll(func() {
		CleanupElasticScalerTest(dynamicClient, clientset, ctx)
	})
})
