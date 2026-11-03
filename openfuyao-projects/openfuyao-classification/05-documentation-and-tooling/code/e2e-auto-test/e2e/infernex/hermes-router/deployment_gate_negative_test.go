package hermesrouter

import (
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

// 运行 hermes-router 部署门禁负向测试时请使用 --procs=1，避免多个 Describe 争用同一 Helm release。
var _ = Describe("prediction.enabled=false 部署门禁负向测试",
	Label("hermes-router-prediction-deployment", "npu"),
	Ordered,
	func() {
		var (
			env            *HermesTestEnv
			variantOverlay string
			valuesFile     string
			cfg            *ReleaseConfig
			stderr         string
			err            error
		)

		BeforeAll(func() {
			By("初始化测试环境")
			env, err = SetupHermesRouterTestWithoutGateway(5 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			By("构建缺失 prediction.enabled 的 overlay 变体")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			variantOverlay, err = BuildOverlayVariant(
				filepath.Join(configDir, "profiles/aggregate-prediction.yaml"),
				[]string{"hermes-router.inferenceExtension.prediction.enabled"},
			)
			Expect(err).NotTo(HaveOccurred(), "构建 overlay 变体失败")
			DeferCleanup(CleanupTempValuesFile, variantOverlay)

			By("构建完整 values（base + 缺失版 overlay）")
			valuesFile, err = BuildValuesFromLocalAndOverlay(
				filepath.Join(hermesDir, AggregateBaseValuesPath),
				variantOverlay,
			)
			Expect(err).NotTo(HaveOccurred(), "构建完整 values 失败")
			DeferCleanup(CleanupTempValuesFile, valuesFile)

			By("注入预测模型 volume（如果已配置）")
			if env.HelmConfig["infernex"] != nil && env.HelmConfig["infernex"].PredictionModelPath != "" {
				valuesFile, err = InjectPredictionModelVolume(valuesFile, env.HelmConfig["infernex"].PredictionModelPath)
				Expect(err).NotTo(HaveOccurred(), "注入预测模型 volume 失败")
				DeferCleanup(CleanupTempValuesFile, valuesFile)
			}

			By("执行 Helm 安装（预期失败）")
			cfg = env.HelmConfig["infernex"]
			Expect(cfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfg.ValuesFile = valuesFile
			stderr, err = HelmInstallExpectFailure(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装应失败但返回了成功")
			Expect(stderr).NotTo(BeEmpty(), "stderr 应包含错误信息")
		})

		AfterAll(func() {
		})

		It("HR-PRED-IT-008-验证 prediction profile 开启但 inferenceExtension.prediction.enabled=false 时部署失败", Label("P0"), func() {
			By("断言 stderr 包含关键错误短语")
			Expect(stderr).To(ContainSubstring("prediction.enabled"), "stderr 应包含 prediction.enabled 相关错误")
		})
	})

var _ = Describe("缺少 tokenizer 参数部署门禁负向测试",
	Label("hermes-router-prediction-deployment", "npu"),
	Ordered,
	func() {
		var (
			env        *HermesTestEnv
			valuesFile string
			cfg        *ReleaseConfig
			stderr     string
			err        error
		)

		BeforeAll(func() {
			By("初始化测试环境")
			env, err = SetupHermesRouterTestWithoutGateway(5 * time.Minute)
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

			By("执行 Helm 安装（预期失败）")
			baseCfg := env.HelmConfig["infernex"]
			Expect(baseCfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfgCopy := *baseCfg
			cfg = &cfgCopy
			cfg.ValuesFile = valuesFile
			cfg.SetValues = []string{
				"--set-string hermes-router.inferenceExtension.routing.tokenizer.model=",
				"--set-string hermes-router.inferenceExtension.routing.tokenizer.socketPath=",
			}
			stderr, err = HelmInstallExpectFailure(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装应失败但返回了成功")
			Expect(stderr).NotTo(BeEmpty(), "stderr 应包含错误信息")
		})

		AfterAll(func() {
		})

		It("HR-PRED-IT-009-验证 prediction profile 缺少 tokenizer 核心参数时部署失败", Label("P0"), func() {
			By("断言 stderr 包含关键错误短语")
			Expect(stderr).To(ContainSubstring("tokenizer.model"), "stderr 应包含 tokenizer.model 相关错误")
			Expect(stderr).To(ContainSubstring("tokenizer.socketPath"), "stderr 应包含 tokenizer.socketPath 相关错误")
		})
	})

var _ = Describe("缺少 cacheIndexer.address 部署门禁负向测试",
	Label("hermes-router-prediction-deployment", "npu"),
	Ordered,
	func() {
		var (
			env        *HermesTestEnv
			valuesFile string
			cfg        *ReleaseConfig
			stderr     string
			err        error
		)

		BeforeAll(func() {
			By("初始化测试环境")
			env, err = SetupHermesRouterTestWithoutGateway(5 * time.Minute)
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

			By("执行 Helm 安装（预期失败）")
			baseCfg := env.HelmConfig["infernex"]
			Expect(baseCfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfgCopy := *baseCfg
			cfg = &cfgCopy
			cfg.ValuesFile = valuesFile
			cfg.SetValues = []string{
				"--set-string hermes-router.inferenceExtension.routing.cacheIndexer.address=",
			}
			stderr, err = HelmInstallExpectFailure(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装应失败但返回了成功")
			Expect(stderr).NotTo(BeEmpty(), "stderr 应包含错误信息")
		})

		AfterAll(func() {
		})

		It("HR-PRED-IT-010-验证 prediction profile 缺少 routing.cacheIndexer.address 时部署失败", Label("P1"), func() {
			By("断言 stderr 包含关键错误短语")
			Expect(stderr).To(ContainSubstring("cacheIndexer.address"), "stderr 应包含 cacheIndexer.address 相关错误")
		})
	})

var _ = Describe("缺少 npuExporter 配置部署门禁负向测试",
	Label("hermes-router-prediction-deployment", "npu"),
	Ordered,
	func() {
		var (
			env            *HermesTestEnv
			variantOverlay string
			valuesFile     string
			cfg            *ReleaseConfig
			stderr         string
			err            error
		)

		BeforeAll(func() {
			By("初始化测试环境")
			env, err = SetupHermesRouterTestWithoutGateway(5 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			By("构建缺失 npuExporter 配置的 overlay 变体")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			variantOverlay, err = BuildOverlayVariant(
				filepath.Join(configDir, "profiles/aggregate-prediction.yaml"),
				[]string{
					"hermes-router.inferenceExtension.routing.npuExporter.exporterNamespace",
					"hermes-router.inferenceExtension.routing.npuExporter.exporterPodLabelSelector",
					"hermes-router.inferenceExtension.routing.npuExporter.exporterPort",
					"hermes-router.inferenceExtension.routing.npuExporter.path",
					"hermes-router.inferenceExtension.routing.npuExporter.cacheTTL",
					"hermes-router.inferenceExtension.routing.npuExporter.scrapeTimeout",
					"hermes-router.inferenceExtension.routing.npuExporter.staleAfter",
				},
			)
			Expect(err).NotTo(HaveOccurred(), "构建 overlay 变体失败")
			DeferCleanup(CleanupTempValuesFile, variantOverlay)

			By("构建完整 values（base + 缺失版 overlay）")
			valuesFile, err = BuildValuesFromLocalAndOverlay(
				filepath.Join(hermesDir, AggregateBaseValuesPath),
				variantOverlay,
			)
			Expect(err).NotTo(HaveOccurred(), "构建完整 values 失败")
			DeferCleanup(CleanupTempValuesFile, valuesFile)

			By("注入预测模型 volume（如果已配置）")
			if env.HelmConfig["infernex"] != nil && env.HelmConfig["infernex"].PredictionModelPath != "" {
				valuesFile, err = InjectPredictionModelVolume(valuesFile, env.HelmConfig["infernex"].PredictionModelPath)
				Expect(err).NotTo(HaveOccurred(), "注入预测模型 volume 失败")
				DeferCleanup(CleanupTempValuesFile, valuesFile)
			}

			By("执行 Helm 安装（预期失败）")
			cfg = env.HelmConfig["infernex"]
			Expect(cfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
			cfg.ValuesFile = valuesFile
			stderr, err = HelmInstallExpectFailure(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装应失败但返回了成功")
			Expect(stderr).NotTo(BeEmpty(), "stderr 应包含错误信息")
		})

		AfterAll(func() {
		})

		It("HR-PRED-IT-011-验证 prediction profile 缺少 routing.npuExporter 必填配置时部署失败", Label("P1"), func() {
			By("断言 stderr 包含关键错误短语")
			Expect(stderr).To(ContainSubstring("npuExporter"), "stderr 应包含 npuExporter 相关错误")
		})
	})

var _ = Describe("predictorAddress 非 unix:// 前缀时 EPP 启动失败",
	Label("hermes-router-prediction-deployment", "npu"),
	Ordered,
	func() {
		var (
			env       *HermesTestEnv
			tempFiles []string
			cfg       *ReleaseConfig
		)

		BeforeAll(func() {
			By("初始化测试环境")
			var err error
			env, err = SetupHermesRouterTestWithoutGateway(10 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			var valuesFile string
			valuesFile, cfg, tempFiles, err = installAggregatePredictionValues(env)
			Expect(err).NotTo(HaveOccurred(), "构建 aggregate-prediction values 失败")

			By("注入非 unix:// predictorAddress 的 EPP 插件配置")
			valuesFile, err = injectInvalidPredictorAddress(valuesFile)
			Expect(err).NotTo(HaveOccurred(), "注入无效 predictorAddress 失败")
			tempFiles = append(tempFiles, valuesFile)
			cfg.ValuesFile = valuesFile
			DeferCleanup(func() { deferCleanupTempValuesFiles(tempFiles) })

			By("Helm 安装 aggregate-prediction（预期 EPP 启动失败）")
			err = HelmInstallWithLatestFallback(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 aggregate-prediction 失败")
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 aggregate-prediction 失败")
			}
		})

		It("HR-PRED-IT-031-验证 Prediction 参数 predictorAddress 使用非 unix:// 前缀时 EPP 启动失败", Label("P3"), func() {
			waitForEPPContainerCrashLoop(env, eppContainerName, 8*time.Minute)

			assertEPPCrashLogContains(env, errPredictorAddressMustUnix)
		})
	})

var _ = Describe("ttftWeight 与 tpotWeight 均为 0 时 EPP 启动失败",
	Label("hermes-router-prediction-deployment", "npu"),
	Ordered,
	func() {
		var (
			env       *HermesTestEnv
			tempFiles []string
			cfg       *ReleaseConfig
		)

		BeforeAll(func() {
			By("初始化测试环境")
			var err error
			env, err = SetupHermesRouterTestWithoutGateway(10 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			_, cfg, tempFiles, err = installAggregatePredictionValues(env,
				"--set hermes-router.inferenceExtension.prediction.ttftWeight=0",
				"--set hermes-router.inferenceExtension.prediction.tpotWeight=0",
			)
			Expect(err).NotTo(HaveOccurred(), "构建 aggregate-prediction values 失败")
			DeferCleanup(func() { deferCleanupTempValuesFiles(tempFiles) })

			By("Helm 安装 aggregate-prediction（预期 EPP 启动失败）")
			err = HelmInstallWithLatestFallback(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 aggregate-prediction 失败")
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 aggregate-prediction 失败")
			}
		})

		It("HR-PRED-IT-032-验证 Prediction 参数 ttftWeight 与 tpotWeight 均为 0 时 EPP 启动失败", Label("P3"), func() {
			waitForEPPContainerCrashLoop(env, eppContainerName, 8*time.Minute)

			assertEPPCrashLogContains(env, errScoreWeightsSumGT0)
		})
	})

var _ = Describe("所有 fallback 权重均为 0 时 EPP 启动失败",
	Label("hermes-router-prediction-deployment", "npu"),
	Ordered,
	func() {
		var (
			env       *HermesTestEnv
			tempFiles []string
			cfg       *ReleaseConfig
		)

		BeforeAll(func() {
			By("初始化测试环境")
			var err error
			env, err = SetupHermesRouterTestWithoutGateway(10 * time.Minute)
			Expect(err).NotTo(HaveOccurred(), "初始化测试环境失败")
			DeferCleanup(env.Cancel)

			_, cfg, tempFiles, err = installAggregatePredictionValues(env,
				"--set hermes-router.inferenceExtension.prediction.kvWeight=0",
				"--set hermes-router.inferenceExtension.prediction.queueWeight=0",
				"--set hermes-router.inferenceExtension.prediction.prefixWeight=0",
				"--set hermes-router.inferenceExtension.prediction.inflightWeight=0",
			)
			Expect(err).NotTo(HaveOccurred(), "构建 aggregate-prediction values 失败")
			DeferCleanup(func() { deferCleanupTempValuesFiles(tempFiles) })

			By("Helm 安装 aggregate-prediction（预期 EPP 启动失败）")
			err = HelmInstallWithLatestFallback(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 aggregate-prediction 失败")
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 aggregate-prediction 失败")
			}
		})

		It("HR-PRED-IT-033-验证 Prediction 参数所有 fallback 权重均为 0 时 EPP 启动失败", Label("P3"), func() {
			waitForEPPContainerCrashLoop(env, eppContainerName, 8*time.Minute)

			assertEPPCrashLogContains(env, errFallbackWeightsSumGT0)
		})
	})
