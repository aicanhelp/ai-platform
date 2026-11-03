package cacheindexer

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Cache Indexer PD Single Group", Ordered, Label("cache-indexer", "npu", "26.06"), func() {
	var (
		ctx                context.Context
		cancel             context.CancelFunc
		infernexValuesFile string
		infernexCfg        *utils.ReleaseConfig
		pdRuntime          utils.PDBackendRuntimeConfig
		cacheRuntime       utils.CacheIndexerRuntimeConfig
	)

	BeforeAll(func() {
		ctx, cancel = context.WithTimeout(context.Background(), 60*time.Minute)
		DeferCleanup(cancel)

		infernexCfg = mustHelmRelease("infernex")
		cleanupLeftoverInfernexAndCacheIndexer(ctx)

		By("使用 infernex 默认 values + pd-single topology values 安装单 PD 组场景（1P1D，tp1）")
		var err error
		infernexValuesFile, pdRuntime, cacheRuntime, err = preparePDSingleGroupValues()
		Expect(err).NotTo(HaveOccurred(), "构建 pd-single values 失败")
		DeferCleanup(func() {
			_ = utils.CleanupTempValuesFile(infernexValuesFile)
		})

		cfg := *infernexCfg
		cfg.ValuesFile = infernexValuesFile
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("等待 Prefill、Decode 与 Mooncake Master 就绪")
		Eventually(func() error {
			_, _, err := utils.GetPDSingleGroupPodIdentity(ctx, k8sClient, infernexCfg.Namespace, pdRuntime.ServiceName, pdSingleGroupPrefillRole)
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			_, _, err := utils.GetPDSingleGroupPodIdentity(ctx, k8sClient, infernexCfg.Namespace, pdRuntime.ServiceName, pdSingleGroupDecodeRole)
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			return utils.WaitForMooncakeMasterPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 60*time.Second, 5*time.Second).Should(Succeed())
	})

	AfterAll(func() {
		uninstallReleaseIfPresent(infernexCfg, "infernex uninstall success")
	})

	It("cache-indexer_FIT_003-PD单组命中Prefill实例", Label("P0"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("等待 cache-indexer Pod Running 且 Ready=1/1")
		Eventually(func() error {
			return utils.AssertCacheIndexerPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		By("等待 Prefill Pod Running 且 Ready=1/1，并获取 Prefill Pod IP")
		var prefillPodName string
		var prefillPodIP string
		Eventually(func() error {
			var err error
			prefillPodName, prefillPodIP, err = utils.GetPDSingleGroupPodIdentity(ctx, k8sClient, infernexCfg.Namespace, pdRuntime.ServiceName, pdSingleGroupPrefillRole)
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Expect(prefillPodName).NotTo(BeEmpty(), "Prefill Pod Name 不应为空")
		Expect(prefillPodIP).NotTo(BeEmpty(), "Prefill Pod IP 不应为空")

		By("等待 discovery 日志中出现 Prefill Pod 的 vLLM pod added 记录")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "vLLM pod added", prefillPodIP)
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("通过 Prefill Pod 的 /tokenize 对 PromptA 分词，并截取前 2 个 block 作为查询 token_ids")
		promptA := buildAggregatePromptA()
		var queryTokens []int32
		Eventually(func() error {
			tokens, err := utils.TokenizePromptViaPDBackend(ctx, infernexCfg.Namespace, prefillPodName, pdRuntime, promptA)
			if err != nil {
				return err
			}
			queryTokenCount := pdSingleGroupQueryBlocks * pdRuntime.PrefillBlockSize
			if len(tokens) < queryTokenCount {
				return fmt.Errorf("expected PromptA tokens >= %d, got %d", queryTokenCount, len(tokens))
			}
			queryTokens = append([]int32(nil), tokens[:queryTokenCount]...)
			return nil
		}, 2*time.Minute, 5*time.Second).Should(Succeed())

		By("通过单 PD 组发送一次 PromptA 推理请求并确认成功")
		Eventually(func() error {
			return utils.DoPDBackendInference(ctx, infernexCfg.Namespace, pdRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())

		By("在 60 秒内轮询 /kv-cache/hit-rate，校验 Prefill Pod L1 达到 1.0")
		var latestHitRateBody []byte
		Eventually(func() error {
			scores, body, err := utils.QueryCacheIndexerHitRateForServerIPs(
				ctx,
				infernexCfg.Namespace,
				[]string{prefillPodIP},
				queryTokens,
				cacheRuntime,
				pdRuntime.PrefillBlockSize,
			)
			if err != nil {
				return err
			}
			latestHitRateBody = body
			if len(scores) != 1 {
				return fmt.Errorf("expected 1 score, got %d body=%s", len(scores), string(body))
			}
			if scores[0].L1HitRatio != 1.0 {
				return fmt.Errorf("expected L1=1.0, got l1=%v l3=%v body=%s", scores[0].L1HitRatio, scores[0].L3HitRatio, string(body))
			}
			return nil
		}, 60*time.Second, 5*time.Second).Should(Succeed())

		GinkgoWriter.Printf("FIT_003 latest hit-rate response: %s\n", string(latestHitRateBody))
	})

	It("cache-indexer_FIT_010-PD中Decode被发现且L1命中为零", Label("P1"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("等待 cache-indexer Pod Running 且 Ready=1/1")
		Eventually(func() error {
			return utils.AssertCacheIndexerPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		By("获取 Prefill/Decode Pod 身份，并确认两者均被 discovery 发现")
		var prefillPodName string
		var prefillPodIP string
		var decodePodIP string
		Eventually(func() error {
			name, ip, err := utils.GetPDSingleGroupPodIdentity(ctx, k8sClient, infernexCfg.Namespace, pdRuntime.ServiceName, pdSingleGroupPrefillRole)
			if err != nil {
				return err
			}
			prefillPodName = name
			prefillPodIP = ip
			return nil
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			_, ip, err := utils.GetPDSingleGroupPodIdentity(ctx, k8sClient, infernexCfg.Namespace, pdRuntime.ServiceName, pdSingleGroupDecodeRole)
			if err != nil {
				return err
			}
			decodePodIP = ip
			return nil
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "vLLM pod added", prefillPodIP)
		}, 90*time.Second, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "vLLM pod added", decodePodIP)
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("通过 Prefill Pod 的 /tokenize 对 PromptA 分词，并截取前 2 个 block 作为查询 token_ids")
		promptA := buildAggregatePromptA()
		var queryTokens []int32
		Eventually(func() error {
			tokens, err := utils.TokenizePromptViaPDBackend(ctx, infernexCfg.Namespace, prefillPodName, pdRuntime, promptA)
			if err != nil {
				return err
			}
			queryTokenCount := pdSingleGroupQueryBlocks * pdRuntime.PrefillBlockSize
			if len(tokens) < queryTokenCount {
				return fmt.Errorf("expected PromptA tokens >= %d, got %d", queryTokenCount, len(tokens))
			}
			queryTokens = append([]int32(nil), tokens[:queryTokenCount]...)
			return nil
		}, 2*time.Minute, 5*time.Second).Should(Succeed())

		By("通过单 PD 组发送一次 PromptA 推理请求并确认成功")
		Eventually(func() error {
			return utils.DoPDBackendInference(ctx, infernexCfg.Namespace, pdRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())

		By("在 20 秒内持续查询 Decode Pod IP，校验其 L1 命中率始终为 0.0")
		var latestHitRateBody []byte
		Consistently(func() error {
			score, body, err := utils.QuerySingleBackendHitRate(ctx, infernexCfg.Namespace, decodePodIP, queryTokens, cacheRuntime, pdRuntime.PrefillBlockSize)
			if err != nil {
				return err
			}
			latestHitRateBody = body
			if score.L1HitRatio != 0.0 {
				return fmt.Errorf("expected l1=0.0, got l1=%v l3=%v body=%s", score.L1HitRatio, score.L3HitRatio, string(body))
			}
			return nil
		}, 20*time.Second, 5*time.Second).Should(Succeed())

		GinkgoWriter.Printf("FIT_010 latest hit-rate response: %s\n", string(latestHitRateBody))
	})
})

func preparePDSingleGroupValues() (string, utils.PDBackendRuntimeConfig, utils.CacheIndexerRuntimeConfig, error) {
	configDir, err := utils.GetCacheIndexerConfigDir()
	if err != nil {
		return "", utils.PDBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}

	infernexBasePath := filepath.Join(configDir, "values-infernex.yaml")
	infernexTopologyPath := filepath.Join(configDir, "topology", "pd-single.yaml")

	valuesFile, err := utils.MergeValuesFiles(infernexBasePath, infernexTopologyPath)
	if err != nil {
		return "", utils.PDBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}

	pdRuntime, err := utils.LoadPDBackendRuntimeConfig(valuesFile)
	if err != nil {
		_ = utils.CleanupTempValuesFile(valuesFile)
		return "", utils.PDBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}
	cacheRuntime, err := utils.LoadEmbeddedCacheIndexerRuntimeConfig(valuesFile)
	if err != nil {
		_ = utils.CleanupTempValuesFile(valuesFile)
		return "", utils.PDBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}

	return valuesFile, pdRuntime, cacheRuntime, nil
}
