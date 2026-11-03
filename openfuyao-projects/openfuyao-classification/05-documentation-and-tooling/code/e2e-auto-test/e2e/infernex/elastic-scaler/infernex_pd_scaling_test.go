package infernex_elastic_scaler

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	esUtils "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	iesutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/elastic-scaler/utils"
	infernexUtils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

var _ = Describe("InferNex PD 分离模式扩缩容", Ordered, Label("infernex-rsg", "infernex-tidal", "npu"), func() {
	ns := infernexUtils.RSGInfernexHelmNamespace
	rsg := infernexUtils.RSGInfernexDefaultRSGName

	BeforeAll(func() {
		Expect(infernexUtils.EnsureNamespace(clientset, ctx, ns)).NotTo(HaveOccurred())
		Expect(iesutils.InstallInfernex(clientset, ctx, infernexUtils.RSGInfernexHelmReleaseName, ns, iesutils.InfernexBackendPD)).NotTo(HaveOccurred())
		Expect(iesutils.CreatePDResourceScalingGroup(dynamicClient, ctx, ns, rsg)).NotTo(HaveOccurred())

		infernexUtils.WaitForInfernexRSGBaseWorkloadsReady(clientset, dynamicClient, ctx, 10*time.Minute)
		infernexUtils.WaitForInfernexRSGScalingSystemControllersReady(clientset, ctx, 10*time.Minute)
		infernexUtils.WaitForRSGGroupReplicas(dynamicClient, ctx, 1, 5*time.Minute)
		infernexUtils.WaitForRSGLeaderWorkerSetGroups(clientset, dynamicClient, ctx, ns, 1, 5*time.Minute)
	})

	Describe("使用 APA 算法扩缩容", Ordered, func() {

		BeforeAll(func() {
			By("创建 ElasticScaler CR，使用 APA 算法管理推理引擎，设置 maxReplicas 为 2")
			Expect(iesutils.CreateApaElasticScalerCRForRSG(dynamicClient, ctx, ns, infernexUtils.RSGInfernexESCRName, rsg, 1, 2, infernexUtils.RSGInfernexScaleUpCPUUtil)).NotTo(HaveOccurred())
			esUtils.VerifyESCRExists(dynamicClient, ctx, infernexUtils.RSGInfernexESCRName, ns)
		})

		AfterAll(func() {
			By("删除 APA ElasticScaler CR")
			Expect(infernexUtils.DeleteElasticScalerCRForRSG(dynamicClient, ctx)).NotTo(HaveOccurred())
		})

		It("InferNex_FIT_81-PD分离模式指标扩容测试", Label("P1"), func() {
			By("开始通过 PD proxy service 持续发送推理请求，等待 APA 触发 RSG 扩容...")
			Expect(infernexUtils.StartRSGPDLoadTest(clientset, ctx, 3*time.Minute)).NotTo(HaveOccurred())

			infernexUtils.WaitForRSGGroupReplicas(dynamicClient, ctx, 2, 10*time.Minute)
			infernexUtils.WaitForRSGLeaderWorkerSetGroups(clientset, dynamicClient, ctx, infernexUtils.RSGInfernexHelmNamespace, 2, 10*time.Minute)

			By("校验扩容到 2 个 PD 组后推理服务仍可用，轻量探针请求成功返回")
			Expect(infernexUtils.RunRSGPDInferenceProbe(clientset, ctx, 10*time.Minute)).To(Succeed())
		})

		It("InferNex_FIT_82-PD分离模式指标缩容测试", Label("P1"), func() {
			By("停止 PD 推理压测，等待 APA 将 RSG 缩容到 minReplicas=1...")
			infernexUtils.StopRSGPDLoadTest(clientset, ctx)
			Expect(infernexUtils.ConfigureDefaultElasticScalerCRForRSG(dynamicClient, ctx, 1, 2, infernexUtils.RSGInfernexScaleDownCPUUtil)).NotTo(HaveOccurred())

			infernexUtils.WaitForRSGGroupReplicas(dynamicClient, ctx, 1, 10*time.Minute)
			infernexUtils.WaitForRSGLeaderWorkerSetGroups(clientset, dynamicClient, ctx, ns, 1, 10*time.Minute)

			By("校验缩容回 1 个 PD 组后推理服务仍可用，轻量探针请求成功返回")
			Expect(infernexUtils.RunRSGPDInferenceProbe(clientset, ctx, 10*time.Minute)).To(Succeed())
		})
	})

	Describe("使用 Tidal 算法扩缩容", Ordered, func() {

		BeforeAll(func() {
			By("创建 Tidal CR")
			Expect(infernexUtils.CreateInfernexTidalAndESForDefaultRSG(dynamicClient, ctx, ns)).NotTo(HaveOccurred())
		})

		AfterAll(func() {
			By("删除 Tidal CR")
			Expect(esUtils.DeleteTidalCR(dynamicClient, ctx, infernexUtils.InfernexTidalName, infernexUtils.RSGInfernexHelmNamespace)).NotTo(HaveOccurred())
		})

		It("InferNex_FIT_84-PD分离模式潮汐扩容测试", Label("P1"), func() {
			infernexUtils.WaitForInfernexBackendReplicas(clientset, dynamicClient, ctx, ns, 2, 2, 8*time.Minute)
			GinkgoWriter.Printf("已扩容到 2 个 PD 组")
		})

		It("InferNex_FIT_83-PD分离模式潮汐缩容测试", Label("P1"), func() {
			Expect(infernexUtils.SwitchInfernexTidalToScaleDownRule(dynamicClient, ctx, ns)).NotTo(HaveOccurred())
			infernexUtils.WaitForInfernexBackendReplicas(clientset, dynamicClient, ctx, ns, 1, 1, 8*time.Minute)
			GinkgoWriter.Printf("已缩容到 1 个 PD 组")
		})
	})

	It("InferNex_FIT_85-PD分离模式手动扩容测试", Label("P2"), func() {
		Expect(infernexUtils.UpdateRSGGroupReplicasInNamespace(dynamicClient, ctx, ns, rsg, 2)).NotTo(HaveOccurred())
		infernexUtils.WaitForInfernexBackendReplicas(clientset, dynamicClient, ctx, ns, 2, 2, 5*time.Minute)
		GinkgoWriter.Printf("已手动扩容到 2 个 PD 组")
	})

	AfterAll(func() {
		infernexUtils.CleanupInfernexTidalAndES(dynamicClient, ctx, ns)
		infernexUtils.CleanupInfernexPDPodsDeployments(clientset, ctx, ns)
		infernexUtils.CleanupRSGInfernexResources(clientset, dynamicClient, ctx)

		Expect(iesutils.UninstallInfernex(infernexUtils.RSGInfernexHelmReleaseName, ns)).NotTo(HaveOccurred())
	})
})
