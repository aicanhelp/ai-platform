package hermesrouter

import (
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

var _ = Describe("aggregate-random routing E2E",
	Label("hermes-router-routing-e2e", "npu"),
	Ordered,
	func() {
		var (
			env *HermesTestEnv
			cfg *ReleaseConfig
		)

		BeforeAll(func() {
			By("初始化测试环境")
			var err error
			env, err = SetupHermesRouterTest(45 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			By("构建 aggregate-random values")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			valuesFile, err := BuildValuesFromLocalAndOverlay(
				filepath.Join(hermesDir, AggregateBaseValuesPath),
				filepath.Join(configDir, "profiles/aggregate-random.yaml"),
			)
			Expect(err).NotTo(HaveOccurred(), "构建 aggregate-random values 失败")
			DeferCleanup(CleanupTempValuesFile, valuesFile)

			By("Helm 安装 aggregate-random")
			cfg = env.HelmConfig["infernex"]
			Expect(cfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfg.ValuesFile = valuesFile
			err = HelmInstallWithLatestFallback(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 aggregate-random 失败")

			time.Sleep(60 * time.Second)
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 aggregate-random 失败")
			}
		})

		It("HR-PRED-IT-012-验证 aggregate-random 路由链在运行时可用且请求能够成功到达后端", Label("P2"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "aggregate-random 容器未全部 Ready")

			By("顺序发送 3 次推理请求")
			Eventually(func() error {
				for i := 0; i < 3; i++ {
					reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceMessage)
					statusCode, body, err := DoInferenceRequestFromEnv(env, reqBody)
					if err != nil {
						return err
					}
					if err := ValidateResponse(statusCode, body, DefaultInferenceStream); err != nil {
						return err
					}
				}
				return nil
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "aggregate-random 顺序推理请求失败")

			// random 策略不渲染 request-lifecycle-tracker（见 _routing.tpl 的 random 分支），
			// EPP 不写 completed.jsonl，因此无法断言 selectedRoute，仅验证 3 次推理请求均成功。
		})
	})

var _ = Describe("pd-random routing E2E",
	Label("hermes-router-routing-e2e", "npu"),
	Ordered,
	func() {
		var (
			env *HermesTestEnv
			cfg *ReleaseConfig
		)

		BeforeAll(func() {
			By("初始化测试环境")
			var err error
			env, err = SetupHermesRouterTest(45 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			By("构建 pd-random values")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			valuesFile, err := BuildValuesFromLocalAndOverlay(
				filepath.Join(hermesDir, PDBaseValuesPath),
				filepath.Join(configDir, "profiles/pd-random.yaml"),
			)
			Expect(err).NotTo(HaveOccurred(), "构建 pd-random values 失败")
			DeferCleanup(CleanupTempValuesFile, valuesFile)

			By("Helm 安装 pd-random")
			cfg = env.HelmConfig["infernex"]
			Expect(cfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfg.ValuesFile = valuesFile
			err = HelmInstallWithLatestFallback(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 pd-random 失败")

			time.Sleep(60 * time.Second)
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 pd-random 失败")
			}
		})

		It("HR-PRED-IT-015-验证 pd-random 路由链在运行时可用且请求能够成功到达后端", Label("P2"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "pd-random 容器未全部 Ready")

			By("顺序发送 3 次推理请求")
			Eventually(func() error {
				for i := 0; i < 3; i++ {
					reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceMessage)
					statusCode, body, err := DoInferenceRequestFromEnv(env, reqBody)
					if err != nil {
						return err
					}
					if err := ValidateResponse(statusCode, body, DefaultInferenceStream); err != nil {
						return err
					}
				}
				return nil
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "pd-random 顺序推理请求失败")

			// random 策略不渲染 request-lifecycle-tracker（见 _routing.tpl 的 random 分支），
			// EPP 不写 completed.jsonl，因此无法断言 selectedRoute，仅验证 3 次推理请求均成功。
		})
	})
