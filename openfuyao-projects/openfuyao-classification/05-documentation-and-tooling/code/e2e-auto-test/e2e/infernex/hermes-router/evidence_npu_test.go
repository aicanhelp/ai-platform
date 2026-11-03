package hermesrouter

import (
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

const multiTPNPURequestCount = 3

var _ = Describe("NPU 数据存在性验证",
	Label("hermes-router-prediction-evidence", "npu"),
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
			err = HelmInstall(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 aggregate-prediction 失败")

			time.Sleep(60 * time.Second)
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 aggregate-prediction 失败")
			}
		})

		It("HR-PRED-IT-018-验证 NPU Exporter 正常时 prediction-routing 完成记录写出 dataStatus=Present 的 NPU 快照", Label("P1"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "aggregate-prediction 容器未全部 Ready")

			By("发送推理请求")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceMessage)
				statusCode, body, err := DoInferenceRequestFromEnv(env, reqBody)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, DefaultInferenceStream)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "aggregate-prediction 推理请求失败")

			By("读取 completed.jsonl")
			records, err := ReadCompletedJSONL(env.Client, env.Ctx, env.Namespace, 1, 2*time.Minute)
			Expect(err).NotTo(HaveOccurred(), "读取 completed.jsonl 失败")
			Expect(len(records)).To(Equal(1), "应有 1 条 completed 记录")

			By("断言 NPU 数据存在且有效")
			Expect(ValidateNPUPresent(records[0])).To(Succeed(), "NPU 数据应存在且 deviceCount >= 1")
		})
	})

var _ = Describe("多 TP NPU 设备聚合验证",
	Label("hermes-router-prediction-evidence", "npu"),
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
			err = HelmInstall(cfg)
			Expect(err).NotTo(HaveOccurred(), "Helm 安装 aggregate-prediction 失败")

			time.Sleep(60 * time.Second)
		})

		AfterAll(func() {
			if cfg != nil {
				err := HelmUninstall(cfg)
				Expect(err).NotTo(HaveOccurred(), "Helm 卸载 aggregate-prediction 失败")
			}
		})

		It("HR-PRED-IT-019-验证多 TP 场景下单 Pod 多卡 NPU 指标会按 Pod 或 endpoint 级正确聚合", Label("P1"), func() {
			waitForPredictionRuntimeServingReady(env, "aggregate-prediction")

			By("连续发送多条推理请求")
			sendPredictionRuntimeRequests(env, "aggregate-prediction", multiTPNPURequestCount, 1*time.Second)

			By("读取 completed.jsonl")
			records, err := ReadCompletedJSONL(env.Client, env.Ctx, env.Namespace, multiTPNPURequestCount, 2*time.Minute)
			Expect(err).NotTo(HaveOccurred(), "读取 completed.jsonl 失败")
			Expect(len(records)).To(Equal(multiTPNPURequestCount), "应有多条 completed 记录")
			latestRecord := records[len(records)-1]

			By("断言 NPU 数据存在且利用率有效")
			Expect(ValidateNPUPresent(latestRecord)).To(Succeed(), "NPU 数据应存在且有效")
			Expect(ValidateNPUMetricsInRange(latestRecord)).To(Succeed(), "NPU 利用率指标应在 [0, 100] 范围内")

			By("断言多 TP 设备聚合")
			Expect(latestRecord.SelectedRoute).NotTo(BeNil(), "selectedRoute 应存在")
			Expect(latestRecord.SelectedRoute.Leader.NPU).NotTo(BeNil(), "leader NPU 数据应存在")
			Expect(latestRecord.SelectedRoute.Leader.NPU.DeviceCount).To(BeNumerically(">", 1), "leader NPU deviceCount 应大于 1（多 TP 聚合）")
			for i, device := range latestRecord.SelectedRoute.Leader.NPU.Devices {
				Expect(device.DeviceID).NotTo(BeEmpty(), "devices[%d].deviceID 应非空", i)
			}

			By("断言 predictionInputs.leader 资源指标存在")
			Expect(latestRecord.PredictionInputs).NotTo(BeNil(), "predictionInputs 应存在")
			Expect(latestRecord.PredictionInputs.Leader).NotTo(BeNil(), "predictionInputs.leader 应存在")
			Expect(latestRecord.PredictionInputs.Leader.HBMUtilization).To(BeNumerically(">", 0), "predictionInputs.leader.hbmUtilization 应大于 0")
			Expect(latestRecord.PredictionInputs.Leader.AICoreUtilization).To(BeNumerically(">", 0), "predictionInputs.leader.aiCoreUtilization 应大于 0")
			Expect(latestRecord.PredictionInputs.Leader.BandwidthUtilization).To(BeNumerically(">", 0), "predictionInputs.leader.bandwidthUtilization 应大于 0")
		})
	})
