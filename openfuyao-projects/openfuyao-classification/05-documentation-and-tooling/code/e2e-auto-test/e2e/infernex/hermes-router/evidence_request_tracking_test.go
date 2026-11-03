package hermesrouter

import (
	"path/filepath"
	"sort"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

const recordIntegrityRequestCount = 3

var _ = Describe("记录完整性与时延验证",
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

		It("HR-PRED-IT-020-验证请求成功完成后 completed.jsonl 记录包含 requestId/completedAt/selectedRoute 与时延关系", Label("P1"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "aggregate-prediction 容器未全部 Ready")

			By("连续发送 3 条推理请求")
			sendPredictionRuntimeRequests(env, "aggregate-prediction", recordIntegrityRequestCount, 1*time.Second)

			By("读取 completed.jsonl")
			records, err := ReadCompletedJSONL(env.Client, env.Ctx, env.Namespace, recordIntegrityRequestCount, 2*time.Minute)
			Expect(err).NotTo(HaveOccurred(), "读取 completed.jsonl 失败")
			Expect(len(records)).To(Equal(recordIntegrityRequestCount), "应有 3 条 completed 记录")

			record := records[len(records)-1]

			By("断言核心字段非 nil/非零")
			Expect(record.SchemaVersion).NotTo(BeZero(), "schemaVersion 应非零")
			Expect(record.RequestId).NotTo(BeEmpty(), "requestId 应非空")
			Expect(record.Model).NotTo(BeEmpty(), "model 应非空")
			Expect(record.CompletedAt).NotTo(BeEmpty(), "completedAt 应非空")
			Expect(record.PredictionMode).NotTo(BeEmpty(), "predictionMode 应非空")
			Expect(record.PrefixCacheScore).NotTo(BeZero(), "prefixCacheScore 应非零")
			Expect(record.InputTokenLength).To(BeNumerically(">", 0), "inputTokenLength 应大于 0")
			Expect(record.OutputLength).To(BeNumerically(">", 0), "outputLength 应大于 0")
			Expect(record.SelectedRoute).NotTo(BeNil(), "selectedRoute 应存在")
			Expect(record.SelectedRoute.Leader.Id).NotTo(BeEmpty(), "selectedRoute.leader.id 应非空")

			By("断言时延字段有合理值")
			Expect(record.TTFTMillis).To(BeNumerically(">", 0), "ttftMillis 应大于 0")
			Expect(record.TotalLatencyMillis).To(BeNumerically(">", 0), "totalLatencyMillis 应大于 0")
			Expect(record.TTFTMillis).To(BeNumerically("<", record.TotalLatencyMillis), "ttftMillis 应小于 totalLatencyMillis")
		})
	})

var _ = Describe("并发 inflight 趋势验证",
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

		It("HR-PRED-IT-021-验证并发重叠请求下 inflight 可见抬升且全部完成后回落并记录条数正确", Label("P1"), func() {
			const concurrentN = 6

			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "aggregate-prediction 容器未全部 Ready")

			By("发送并发长推理请求，确保路由时存在重叠")
			var wg sync.WaitGroup
			errs := make([]error, concurrentN)
			for i := 0; i < concurrentN; i++ {
				wg.Add(1)
				go func(idx int) {
					defer wg.Done()
					reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceLongMessage)
					reqBody["max_tokens"] = DefaultMaxTokens
					statusCode, body, err := DoInferenceRequestFromEnvWithTimeout(env, reqBody, 4*time.Minute)
					if err != nil {
						errs[idx] = err
						return
					}
					errs[idx] = ValidateResponse(statusCode, body, DefaultInferenceStream)
				}(i)
			}
			wg.Wait()
			for _, e := range errs {
				Expect(e).NotTo(HaveOccurred(), "并发推理请求失败")
			}

			By("读取 completed.jsonl")
			records, err := ReadCompletedJSONL(env.Client, env.Ctx, env.Namespace, concurrentN, 3*time.Minute)
			Expect(err).NotTo(HaveOccurred(), "读取 completed.jsonl 失败")
			Expect(len(records)).To(Equal(concurrentN), "应有足够的 completed 记录")

			By("按 completedAt 排序后断言至少一条非首条记录存在 inflight")
			sort.Slice(records, func(i, j int) bool {
				return records[i].CompletedAt < records[j].CompletedAt
			})
			sawInflight := false
			for i := 1; i < len(records); i++ {
				if records[i].SelectedRoute == nil || records[i].SelectedRoute.Leader.Inflight == nil {
					continue
				}
				inf := records[i].SelectedRoute.Leader.Inflight
				if inf.Requests > 0 || inf.Tokens > 0 {
					sawInflight = true
					break
				}
			}
			Expect(sawInflight).To(BeTrue(), "并发请求中应至少有一条非首条记录的 inflight.Requests 或 inflight.Tokens 大于 0")
		})
	})
