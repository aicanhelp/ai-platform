package hermesrouter

import (
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

// 组件选装测试：验证关闭部分可选组件后 InferNex 仍可正常部署并完成推理。
// 运行时请使用 --procs=1，避免与同套件其他 Describe 争用同一 Helm release（infernex）。
var _ = Describe("智能路由组件选装（hermes-router/inferenceGateway 均禁用）",
	Label("hermes-router-component-optional", "npu"),
	Ordered,
	func() {
		var (
			env        *HermesTestEnv
			valuesFile string
			cfg        *ReleaseConfig
		)

		BeforeAll(func() {
			By("初始化测试环境")
			var err error
			env, err = SetupHermesRouterTestWithoutGateway(45 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			By("构建 aggregate-random values")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			valuesFile, err = BuildValuesFromLocalAndOverlay(
				filepath.Join(hermesDir, AggregateBaseValuesPath),
				filepath.Join(configDir, "profiles/aggregate-random.yaml"),
			)
			Expect(err).NotTo(HaveOccurred(), "构建 aggregate-random values 失败")
			DeferCleanup(CleanupTempValuesFile, valuesFile)

			By("Helm 安装（仅禁用 hermes-router 与 inferenceGateway，保留 eagle-eye/cache-indexer）")
			cfg = env.HelmConfig["infernex"]
			Expect(cfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfg.ValuesFile = valuesFile
			cfg.SetValues = []string{
				"--set hermes-router.enabled=false",
				"--set inferenceGateway.enabled=false",
			}
			err = HelmInstallWithLatestFallback(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装失败")

			time.Sleep(60 * time.Second)
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载失败")
			}
		})

		It("HR-PRED-IT-034-开源网关组件选装;智能路由组件选装", Label("P0"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "容器未全部 Ready")

			By("绕过 gateway 直连 inference-backend 校验推理请求")
			Eventually(func() error {
				return doInferenceRequestViaBackend(env)
			}, 5*time.Minute, 15*time.Second).Should(Succeed(), "推理请求校验失败")

			By("检查 helm 安装资源清单非空")
			count, err := helmReleaseResourceCount(cfg.ReleaseName, cfg.Namespace)
			Expect(err).NotTo(HaveOccurred())
			Expect(count).To(BeNumerically(">", 0), "helm 应该安装了至少一个资源")
			GinkgoWriter.Printf("\nhelm 安装了 %d 个资源\n", count)
		})
	})
