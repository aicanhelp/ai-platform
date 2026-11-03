package hermesrouter

import (
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

// longPromptInferenceTimeout：长 prompt（DefaultInferenceLongMessage）在 NPU 上首 token
// 较慢，30s 默认 HTTP 超时不足，曾导致 IT-013/016 预热请求 "Client.Timeout exceeded"。
const longPromptInferenceTimeout = 4 * time.Minute

var _ = Describe("aggregate-kv-cache-aware routing E2E",
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

			By("构建 aggregate-kv-cache-aware values")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			valuesFile, err := BuildValuesFromLocalAndOverlay(
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

		It("HR-PRED-IT-013-验证 aggregate-kv-cache-aware 路由链预热后 completed.jsonl 记录 prefixCacheScore>0", Label("P2"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "aggregate-kv-cache-aware 容器未全部 Ready")

			By("发送预热请求（第 1 次）")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(false, DefaultInferenceLongMessage)
				statusCode, body, err := DoInferenceRequestFromEnvWithTimeout(env, reqBody, longPromptInferenceTimeout)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, false)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "aggregate-kv-cache-aware 预热请求失败")

			By("发送相同消息触发 prefix cache（第 2 次）")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(false, DefaultInferenceLongMessage)
				statusCode, body, err := DoInferenceRequestFromEnvWithTimeout(env, reqBody, longPromptInferenceTimeout)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, false)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "aggregate-kv-cache-aware 第 2 次请求失败")

			By("读取 completed.jsonl 断言 prefixCacheScore")
			records, err := ReadCompletedJSONL(env.Client, env.Ctx, env.Namespace, 2, 2*time.Minute)
			Expect(err).NotTo(HaveOccurred(), "读取 completed.jsonl 失败")
			Expect(len(records)).To(Equal(2), "应有 2 条 completed 记录")
			for i, rec := range records {
				GinkgoWriter.Printf("record[%d] requestId=%s inputTokenLength=%d prefixCacheScore=%f\n", i, rec.RequestId, rec.InputTokenLength, rec.PrefixCacheScore)
			}
			Expect(records[1].PrefixCacheScore).To(BeNumerically(">", 0), "第 2 次请求的 prefixCacheScore 应大于 0")
		})
	})

var _ = Describe("pd-kv-cache-aware routing E2E",
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

			By("构建 pd-kv-cache-aware values")
			hermesDir, err := GetHermesRouterDir()
			Expect(err).NotTo(HaveOccurred(), "获取 hermes-router 目录失败")
			configDir, err := GetHermesRouterConfigDir()
			Expect(err).NotTo(HaveOccurred(), "获取配置目录失败")
			valuesFile, err := BuildValuesFromLocalAndOverlay(
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

		It("HR-PRED-IT-016-验证 pd-kv-cache-aware 路由链预热后 completed.jsonl 记录 prefixCacheScore>0", Label("P2"), func() {
			By("验证所有容器 Ready")
			Eventually(func() bool {
				ok, err := CheckAllContainersReady(env.Client, env.Ctx, env.Namespace)
				if err != nil {
					return false
				}
				return ok
			}, containerReadyTimeout, 10*time.Second).Should(BeTrue(), "pd-kv-cache-aware 容器未全部 Ready")

			By("发送预热请求（第 1 次）")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(false, DefaultInferenceLongMessage)
				statusCode, body, err := DoInferenceRequestFromEnvWithTimeout(env, reqBody, longPromptInferenceTimeout)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, false)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "pd-kv-cache-aware 预热请求失败")

			By("发送相同消息触发 prefix cache（第 2 次）")
			Eventually(func() error {
				reqBody := BuildInferenceRequestBody(false, DefaultInferenceLongMessage)
				statusCode, body, err := DoInferenceRequestFromEnvWithTimeout(env, reqBody, longPromptInferenceTimeout)
				if err != nil {
					return err
				}
				return ValidateResponse(statusCode, body, false)
			}, 2*time.Minute, 5*time.Second).Should(Succeed(), "pd-kv-cache-aware 第 2 次请求失败")

			By("读取 completed.jsonl 断言 prefixCacheScore")
			records, err := ReadCompletedJSONL(env.Client, env.Ctx, env.Namespace, 2, 2*time.Minute)
			Expect(err).NotTo(HaveOccurred(), "读取 completed.jsonl 失败")
			Expect(len(records)).To(Equal(2), "应有 2 条 completed 记录")
			for i, rec := range records {
				GinkgoWriter.Printf("record[%d] requestId=%s inputTokenLength=%d prefixCacheScore=%f\n", i, rec.RequestId, rec.InputTokenLength, rec.PrefixCacheScore)
			}
			Expect(records[1].PrefixCacheScore).To(BeNumerically(">", 0), "第 2 次请求的 prefixCacheScore 应大于 0")
		})
	})
