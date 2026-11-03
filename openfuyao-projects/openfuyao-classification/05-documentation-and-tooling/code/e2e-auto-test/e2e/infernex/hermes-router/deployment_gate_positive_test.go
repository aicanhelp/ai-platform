package hermesrouter

import (
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

var _ = Describe("aggregate-prediction 部署门禁正向测试",
	Label("hermes-router-prediction-deployment", "npu"),
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
			env, err = SetupHermesRouterTest(45 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			By("构建 aggregate-prediction values")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			valuesFile, err = BuildValuesFromLocalAndOverlay(
				filepath.Join(hermesDir, AggregateBaseValuesPath),
				filepath.Join(configDir, "profiles/aggregate-prediction.yaml"),
			)
			Expect(err).NotTo(HaveOccurred(), "构建 aggregate-prediction values 失败")
			DeferCleanup(CleanupTempValuesFile, valuesFile)

			By("注入预测模型 volume（如果已配置）")
			if env.HelmConfig["infernex"] != nil && env.HelmConfig["infernex"].PredictionModelPath != "" {
				valuesFile, err = InjectPredictionModelVolume(valuesFile, env.HelmConfig["infernex"].PredictionModelPath)
				Expect(err).NotTo(HaveOccurred(), "注入预测模型 volume 失败")
				DeferCleanup(CleanupTempValuesFile, valuesFile)
			}

			By("Helm 安装 aggregate-prediction")
			cfg = env.HelmConfig["infernex"]
			Expect(cfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfg.ValuesFile = valuesFile
			err = HelmInstallWithLatestFallback(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 aggregate-prediction 失败")

			time.Sleep(60 * time.Second)
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 aggregate-prediction 失败")
			}
		})

		It("HR-PRED-IT-003-验证 aggregate-prediction profile 替换后可成功部署且 sidecar 与 socket 就绪并完成一次推理请求", Label("P0"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "aggregate-prediction 容器未全部 Ready")

			By("验证 tokenizer.sock 可用")
			Eventually(func() bool {
				ok, err := CheckSocketExists(env.Client, env.Ctx, env.Namespace, "/var/run/tokenizer/tokenizer.sock")
				if err != nil {
					return false
				}
				return ok
			}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "tokenizer.sock 不可用")

			By("验证 prediction.sock 可用")
			Eventually(func() bool {
				ok, err := CheckSocketExists(env.Client, env.Ctx, env.Namespace, "/var/run/hermes/prediction.sock")
				if err != nil {
					return false
				}
				return ok
			}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "prediction.sock 不可用")

			By("发送推理请求验证返回 200")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceMessage)
				statusCode, body, err := DoInferenceRequestFromEnv(env, reqBody)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, DefaultInferenceStream)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "aggregate-prediction 推理请求失败")
		})
	})

var _ = Describe("pd-prediction 部署门禁正向测试",
	Label("hermes-router-prediction-deployment", "npu"),
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
			env, err = SetupHermesRouterTest(45 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			By("构建 pd-prediction values")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			valuesFile, err = BuildValuesFromLocalAndOverlay(
				filepath.Join(hermesDir, PDBaseValuesPath),
				filepath.Join(configDir, "profiles/pd-prediction.yaml"),
			)
			Expect(err).NotTo(HaveOccurred(), "构建 pd-prediction values 失败")
			DeferCleanup(CleanupTempValuesFile, valuesFile)

			By("注入预测模型 volume（如果已配置）")
			if env.HelmConfig["infernex"] != nil && env.HelmConfig["infernex"].PredictionModelPath != "" {
				valuesFile, err = InjectPredictionModelVolume(valuesFile, env.HelmConfig["infernex"].PredictionModelPath)
				Expect(err).NotTo(HaveOccurred(), "注入预测模型 volume 失败")
				DeferCleanup(CleanupTempValuesFile, valuesFile)
			}

			By("Helm 安装 pd-prediction")
			cfg = env.HelmConfig["infernex"]
			Expect(cfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfg.ValuesFile = valuesFile
			err = HelmInstallWithLatestFallback(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 pd-prediction 失败")

			time.Sleep(60 * time.Second)
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 pd-prediction 失败")
			}
		})

		// 暂时跳过：pd-prediction 路径尚未稳定，运行 ginkgo 时用 -ginkgo.skip='skip-temporarily' 即可跳过。
		It("HR-PRED-IT-007-验证 pd-prediction profile 替换后可成功部署且 sidecar 与 socket 就绪并完成一次推理请求", Label("P0", "skip-temporarily"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "pd-prediction 容器未全部 Ready")

			By("验证 tokenizer.sock 可用")
			Eventually(func() bool {
				ok, err := CheckSocketExists(env.Client, env.Ctx, env.Namespace, "/var/run/tokenizer/tokenizer.sock")
				if err != nil {
					return false
				}
				return ok
			}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "tokenizer.sock 不可用")

			By("验证 prediction.sock 可用")
			Eventually(func() bool {
				ok, err := CheckSocketExists(env.Client, env.Ctx, env.Namespace, "/var/run/hermes/prediction.sock")
				if err != nil {
					return false
				}
				return ok
			}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "prediction.sock 不可用")

			By("发送推理请求验证返回 200")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceMessage)
				statusCode, body, err := DoInferenceRequestFromEnv(env, reqBody)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, DefaultInferenceStream)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "pd-prediction 推理请求失败")
		})
	})

var _ = Describe("aggregate-random 部署门禁正向测试",
	Label("hermes-router-non-prediction-deployment", "npu"),
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
			env, err = SetupHermesRouterTest(45 * time.Minute)
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

		It("HR-PRED-IT-001-验证 aggregate-random profile 替换后可成功部署并完成一次推理请求", Label("P2"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "aggregate-random 容器未全部 Ready")

			By("发送推理请求验证返回 200")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceMessage)
				statusCode, body, err := DoInferenceRequestFromEnv(env, reqBody)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, DefaultInferenceStream)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "aggregate-random 推理请求失败")
		})
	})

var _ = Describe("aggregate-kv-cache-aware 部署门禁正向测试",
	Label("hermes-router-non-prediction-deployment", "npu"),
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
			env, err = SetupHermesRouterTest(45 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			By("构建 aggregate-kv-cache-aware values")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			valuesFile, err = BuildValuesFromLocalAndOverlay(
				filepath.Join(hermesDir, AggregateBaseValuesPath),
				filepath.Join(configDir, "profiles/aggregate-kv-cache-aware.yaml"),
			)
			Expect(err).NotTo(HaveOccurred(), "构建 aggregate-kv-cache-aware values 失败")
			DeferCleanup(CleanupTempValuesFile, valuesFile)

			By("Helm 安装 aggregate-kv-cache-aware")
			cfg = env.HelmConfig["infernex"]
			Expect(cfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfg.ValuesFile = valuesFile
			err = HelmInstallWithLatestFallback(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 aggregate-kv-cache-aware 失败")

			time.Sleep(60 * time.Second)
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 aggregate-kv-cache-aware 失败")
			}
		})

		It("HR-PRED-IT-002-验证 aggregate-kv-cache-aware profile 替换后可成功部署并完成一次推理请求", Label("P2"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "aggregate-kv-cache-aware 容器未全部 Ready")

			By("验证 tokenizer.sock 可用")
			Eventually(func() bool {
				ok, err := CheckSocketExists(env.Client, env.Ctx, env.Namespace, "/var/run/tokenizer/tokenizer.sock")
				if err != nil {
					return false
				}
				return ok
			}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "tokenizer.sock 不可用")

			By("发送推理请求验证返回 200")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceMessage)
				statusCode, body, err := DoInferenceRequestFromEnv(env, reqBody)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, DefaultInferenceStream)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "aggregate-kv-cache-aware 推理请求失败")
		})
	})

var _ = Describe("pd-random 部署门禁正向测试",
	Label("hermes-router-non-prediction-deployment", "npu"),
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
			env, err = SetupHermesRouterTest(45 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			By("构建 pd-random values")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			valuesFile, err = BuildValuesFromLocalAndOverlay(
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

		It("HR-PRED-IT-004-验证 pd-random profile 替换后可成功部署并完成一次推理请求", Label("P2"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "pd-random 容器未全部 Ready")

			By("发送推理请求验证返回 200")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceMessage)
				statusCode, body, err := DoInferenceRequestFromEnv(env, reqBody)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, DefaultInferenceStream)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "pd-random 推理请求失败")
		})
	})

var _ = Describe("pd-kv-cache-aware 部署门禁正向测试",
	Label("hermes-router-non-prediction-deployment", "npu"),
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
			env, err = SetupHermesRouterTest(45 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			By("构建 pd-kv-cache-aware values")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			valuesFile, err = BuildValuesFromLocalAndOverlay(
				filepath.Join(hermesDir, PDBaseValuesPath),
				filepath.Join(configDir, "profiles/pd-kv-cache-aware.yaml"),
			)
			Expect(err).NotTo(HaveOccurred(), "构建 pd-kv-cache-aware values 失败")
			DeferCleanup(CleanupTempValuesFile, valuesFile)

			By("Helm 安装 pd-kv-cache-aware")
			cfg = env.HelmConfig["infernex"]
			Expect(cfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfg.ValuesFile = valuesFile
			err = HelmInstallWithLatestFallback(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 pd-kv-cache-aware 失败")

			time.Sleep(60 * time.Second)
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 pd-kv-cache-aware 失败")
			}
		})

		It("HR-PRED-IT-005-验证 pd-kv-cache-aware profile 替换后可成功部署并完成一次推理请求", Label("P2"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "pd-kv-cache-aware 容器未全部 Ready")

			By("验证 tokenizer.sock 可用")
			Eventually(func() bool {
				ok, err := CheckSocketExists(env.Client, env.Ctx, env.Namespace, "/var/run/tokenizer/tokenizer.sock")
				if err != nil {
					return false
				}
				return ok
			}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "tokenizer.sock 不可用")

			By("发送推理请求验证返回 200")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceMessage)
				statusCode, body, err := DoInferenceRequestFromEnv(env, reqBody)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, DefaultInferenceStream)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "pd-kv-cache-aware 推理请求失败")
		})
	})

var _ = Describe("pd-bucket 部署门禁正向测试",
	Label("hermes-router-non-prediction-deployment", "npu"),
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
			env, err = SetupHermesRouterTest(45 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			By("构建 pd-bucket values")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			valuesFile, err = BuildValuesFromLocalAndOverlay(
				filepath.Join(hermesDir, PDBaseValuesPath),
				filepath.Join(configDir, "profiles/pd-bucket.yaml"),
			)
			Expect(err).NotTo(HaveOccurred(), "构建 pd-bucket values 失败")
			DeferCleanup(CleanupTempValuesFile, valuesFile)

			By("Helm 安装 pd-bucket")
			cfg = env.HelmConfig["infernex"]
			Expect(cfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfg.ValuesFile = valuesFile
			err = HelmInstallWithLatestFallback(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 pd-bucket 失败")

			time.Sleep(60 * time.Second)
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 pd-bucket 失败")
			}
		})

		It("HR-PRED-IT-006-验证 pd-bucket profile 替换后可成功部署并完成一次推理请求", Label("P2"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "pd-bucket 容器未全部 Ready")

			By("发送推理请求验证返回 200")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceMessage)
				statusCode, body, err := DoInferenceRequestFromEnv(env, reqBody)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, DefaultInferenceStream)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "pd-bucket 推理请求失败")
		})
	})
