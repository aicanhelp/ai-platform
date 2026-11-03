package infernex_elastic_scaler

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	esUtils "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	iesutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/elastic-scaler/utils"
	infernexUtils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

var _ = Describe("InferNex 聚合模式指标扩缩容", Ordered, Label("infernex-elastic-scaler", "npu"), func() {
	BeforeAll(func() {
		Expect(infernexUtils.EnsureNamespace(clientset, ctx, infernexUtils.ESInfernexHelmNamespace)).NotTo(HaveOccurred())
		Expect(infernexUtils.EnsureLeaderWorkerSetAPI(clientset)).NotTo(HaveOccurred())
		Expect(iesutils.InstallInfernex(clientset, ctx, infernexUtils.ESInfernexHelmReleaseName, infernexUtils.ESInfernexHelmNamespace, iesutils.InfernexBackendAggregated)).NotTo(HaveOccurred())
		infernexUtils.WaitForInfernexAllDeploymentsReady(clientset, dynamicClient, ctx, 10*time.Minute)
	})

	Describe("使用 HPA 算法扩缩容", Ordered, func() {
		BeforeAll(func() {
			By("创建 ElasticScaler CR，使用 HPA 算法管理推理引擎，设置 maxReplicas 为 2")
			Expect(iesutils.CreateHpaElasticScalerCRForAggregated(dynamicClient, ctx, infernexUtils.ESInfernexHelmNamespace, infernexUtils.ESInfernexESCRName, 1, 2, 50)).NotTo(HaveOccurred())
			time.Sleep(15 * time.Second)
			esUtils.VerifyESCRExists(dynamicClient, ctx, infernexUtils.ESInfernexESCRName, infernexUtils.ESInfernexHelmNamespace)
			esUtils.VerifyHPACreated(clientset, ctx, infernexUtils.ESInfernexESCRName+"-hpa", infernexUtils.ESInfernexHelmNamespace)
		})

		AfterAll(func() {
			By("删除 HPA ElasticScaler CR")
			Expect(infernexUtils.DeleteElasticScalerCRForInfernex(dynamicClient, ctx)).NotTo(HaveOccurred())
		})

		It("InferNex_FIT_86-聚合模式指标扩容测试", Label("P1", "InferNex_FIT_86"), func() {
			By("开始对推理引擎进行压测，触发 HPA 扩容")
			Expect(infernexUtils.StartLoadTest(clientset, ctx, 10*time.Minute)).NotTo(HaveOccurred())

			By("等待推理引擎 HPA 扩容到 2 个副本")
			infernexUtils.WaitForInfernexDeploymentReplicas(dynamicClient, ctx, 2, 10*time.Minute)
		})

		It("InferNex_FIT_87-聚合模式指标缩容测试", Label("P1", "InferNex_FIT_87"), func() {
			By("停止压测，等待 HPA 缩容")
			infernexUtils.StopLoadTest(clientset, ctx)
			time.Sleep(5 * time.Second) // 等待压测完全停止

			By("等待推理引擎 HPA 缩容回 1 个副本")
			infernexUtils.WaitForInfernexDeploymentReplicas(dynamicClient, ctx, 1, 10*time.Minute)
		})
	})

	Describe("使用 APA 算法扩缩容", Ordered, func() {
		BeforeAll(func() {
			By("创建 APA 算法 ElasticScaler CR，设置 maxReplicas 为 2")
			Expect(iesutils.CreateApaElasticScalerCRForAggregated(dynamicClient, ctx, infernexUtils.ESInfernexHelmNamespace, infernexUtils.ESInfernexESCRName, 1, 2, 50)).NotTo(HaveOccurred())
			time.Sleep(15 * time.Second)
			esUtils.VerifyESCRExists(dynamicClient, ctx, infernexUtils.ESInfernexESCRName, infernexUtils.ESInfernexHelmNamespace)
			esUtils.VerifyESValidatedCondition(dynamicClient, ctx, infernexUtils.ESInfernexESCRName, infernexUtils.ESInfernexHelmNamespace)
		})

		AfterAll(func() {
			By("删除 APA ElasticScaler CR")
			Expect(infernexUtils.DeleteElasticScalerCRForInfernex(dynamicClient, ctx)).NotTo(HaveOccurred())
		})

		It("InferNex_FIT_79-聚合模式指标扩容测试", Label("P1", "InferNex_FIT_79"), func() {
			By("开始对推理引擎进行压测，触发 APA 扩容")
			Expect(infernexUtils.StartLoadTest(clientset, ctx, 10*time.Minute)).NotTo(HaveOccurred())

			By("等待推理引擎 APA 扩容到 2 个副本")
			infernexUtils.WaitForInfernexDeploymentReplicas(dynamicClient, ctx, 2, 10*time.Minute)
		})

		It("InferNex_FIT_80-聚合模式指标缩容测试", Label("P1", "InferNex_FIT_80"), func() {
			By("停止压测，等待 APA 缩容")
			infernexUtils.StopLoadTest(clientset, ctx)
			time.Sleep(5 * time.Second)

			By("等待推理引擎 APA 缩容回 1 个副本")
			infernexUtils.WaitForInfernexDeploymentReplicas(dynamicClient, ctx, 1, 10*time.Minute)
		})
	})

	AfterAll(func() {
		// 清理：停止压测（如果还在运行）
		infernexUtils.StopLoadTest(clientset, ctx)

		Expect(iesutils.UninstallInfernex(infernexUtils.ESInfernexHelmReleaseName, infernexUtils.ESInfernexHelmNamespace)).NotTo(HaveOccurred())
	})
})
