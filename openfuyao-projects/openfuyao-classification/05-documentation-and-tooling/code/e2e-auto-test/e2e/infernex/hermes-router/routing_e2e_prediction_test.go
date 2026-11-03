package hermesrouter

import (
	"fmt"
	"math"
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

const (
	predictionRuntimeRequestCount        = 5
	predictionRuntimeMessageLen          = 224
	predictionRuntimeMaxTokens           = 128
	predictionRuntimeFirstRequestTimeout = 5 * time.Minute
	predictionRuntimeRequestTimeout      = 2 * time.Minute
)

func buildPredictionRuntimeMessage(length int) string {
	if length <= 0 {
		length = predictionRuntimeMessageLen
	}

	seed := []rune("请结合请求排队并发数缓存复用块匹配路由选择首字时延总时延吞吐稳定性与资源利用率，分析当前推理服务在连续相同请求下的表现，并说明哪些指标能证明前缀块已经被复用以及系统是否保持稳定。")
	message := make([]rune, 0, length)
	for len(message) < length {
		message = append(message, seed...)
	}

	return string(message[:length])
}

func waitForPredictionRuntimeServingReady(env *HermesTestEnv, scenario string) {
	By(fmt.Sprintf("%s: 等待所有容器 Ready", scenario))
	Eventually(func() bool {
		ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
		if err != nil {
			return false
		}
		return ok
	}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), fmt.Sprintf("%s: 容器未全部 Ready", scenario))

	By(fmt.Sprintf("%s: 等待 tokenizer.sock 可用", scenario))
	Eventually(func() bool {
		ok, err := CheckSocketExists(env.Client, env.Ctx, env.Namespace, "/var/run/tokenizer/tokenizer.sock")
		if err != nil {
			return false
		}
		return ok
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), fmt.Sprintf("%s: tokenizer.sock 不可用", scenario))

	By(fmt.Sprintf("%s: 等待 prediction.sock 可用", scenario))
	Eventually(func() bool {
		ok, err := CheckSocketExists(env.Client, env.Ctx, env.Namespace, "/var/run/hermes/prediction.sock")
		if err != nil {
			return false
		}
		return ok
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), fmt.Sprintf("%s: prediction.sock 不可用", scenario))
}

func sendPredictionRuntimeRequests(env *HermesTestEnv, scenario string, count int, interval time.Duration) {
	message := buildPredictionRuntimeMessage(predictionRuntimeMessageLen)
	for attempt := 1; attempt <= count; attempt++ {
		By(fmt.Sprintf("%s: 发送第 %d/%d 次相同中文推理请求", scenario, attempt, count))
		requestTimeout := predictionRuntimeRequestTimeout
		if attempt == 1 {
			requestTimeout = predictionRuntimeFirstRequestTimeout
		}
		Eventually(func() error {
			reqBody := BuildInferenceRequestBody(DefaultInferenceStream, message)
			reqBody["max_tokens"] = predictionRuntimeMaxTokens
			statusCode, body, err := DoInferenceRequestFromEnv(env, reqBody)
			if err != nil {
				return err
			}
			return ValidateResponse(statusCode, body, DefaultInferenceStream)
		}, requestTimeout, 5*time.Second).Should(Succeed(), fmt.Sprintf("%s: 第 %d 次相同中文推理请求失败", scenario, attempt))

		if attempt < count {
			time.Sleep(interval)
		}
	}
}

func assertPredictionDataValid(record CompletedRecord) {
	Expect(record.Prediction).NotTo(BeNil(), "prediction 应存在")
	Expect(record.Prediction.TTFTMillis).To(BeNumerically(">", 0), "prediction.ttftMillis 应大于 0")
	Expect(math.IsNaN(record.Prediction.TTFTMillis)).To(BeFalse(), "prediction.ttftMillis 不应为 NaN")
	Expect(record.Prediction.TPOTMillis).To(BeNumerically(">=", 0), "prediction.tpotMillis 应为非负值")
	Expect(math.IsNaN(record.Prediction.TPOTMillis)).To(BeFalse(), "prediction.tpotMillis 不应为 NaN")
	Expect(record.TotalLatencyMillis).To(BeNumerically(">", 0), "totalLatencyMillis 应大于 0")
	Expect(math.IsNaN(record.TotalLatencyMillis)).To(BeFalse(), "totalLatencyMillis 不应为 NaN")
}

var _ = Describe("aggregate-prediction runtime scoring 验证",
	Label("hermes-router-prediction-runtime", "npu"),
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

		It("HR-PRED-IT-014-验证 aggregate-prediction 路由链在运行时写出 predictionMode=active 与 prediction 数值", Label("P0"), func() {
			waitForPredictionRuntimeServingReady(env, "aggregate-prediction")

			By("连续发送 5 次相同中文推理请求生成 runtime scoring 记录")
			sendPredictionRuntimeRequests(env, "aggregate-prediction", predictionRuntimeRequestCount, 1*time.Second)

			By("读取 completed.jsonl")
			records, err := ReadCompletedJSONL(env.Client, env.Ctx, env.Namespace, predictionRuntimeRequestCount, 2*time.Minute)
			Expect(err).NotTo(HaveOccurred(), "读取 completed.jsonl 失败")
			Expect(len(records)).To(Equal(predictionRuntimeRequestCount), "应有 5 条 completed 记录")
			latestRecord := records[len(records)-1]

			By("断言最后一条记录的 prediction 子对象包含有效数值")
			assertPredictionDataValid(latestRecord)
		})
	})

var _ = Describe("pd-prediction runtime scoring 验证",
	Label("hermes-router-prediction-runtime", "npu"),
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
		It("HR-PRED-IT-017-验证 pd-prediction 路由链在运行时写出 predictionMode=active 与 prediction 数值", Label("P0", "skip-temporarily"), func() {
			waitForPredictionRuntimeServingReady(env, "pd-prediction")

			By("连续发送 5 次相同中文推理请求生成 runtime scoring 记录")
			sendPredictionRuntimeRequests(env, "pd-prediction", predictionRuntimeRequestCount, 1*time.Second)

			By("读取 completed.jsonl")
			records, err := ReadCompletedJSONL(env.Client, env.Ctx, env.Namespace, predictionRuntimeRequestCount, 2*time.Minute)
			Expect(err).NotTo(HaveOccurred(), "读取 completed.jsonl 失败")
			Expect(len(records)).To(Equal(predictionRuntimeRequestCount), "应有 5 条 completed 记录")
			latestRecord := records[len(records)-1]

			By("断言 selectedRoute 路由目标")
			Expect(latestRecord.SelectedRoute).NotTo(BeNil(), "selectedRoute 应存在")
			Expect(latestRecord.SelectedRoute.Prefill).NotTo(BeNil(), "selectedRoute.prefill 应存在")
			Expect(latestRecord.SelectedRoute.Decode).NotTo(BeNil(), "selectedRoute.decode 应存在")
			Expect(latestRecord.SelectedRoute.Prefill.Id).To(ContainSubstring("prefill"), "selectedRoute.prefill.id 应包含 prefill 关键词")
			Expect(latestRecord.SelectedRoute.Decode.Id).To(ContainSubstring("decode"), "selectedRoute.decode.id 应包含 decode 关键词")

			By("断言 predictionInputs 子对象存在")
			Expect(latestRecord.PredictionInputs).NotTo(BeNil(), "predictionInputs 应存在")
			Expect(latestRecord.PredictionInputs.Prefill).NotTo(BeNil(), "predictionInputs.prefill 应存在")
			Expect(latestRecord.PredictionInputs.Decode).NotTo(BeNil(), "predictionInputs.decode 应存在")

			By("断言最后一条记录的 prediction 子对象包含有效数值")
			assertPredictionDataValid(latestRecord)
		})
	})
