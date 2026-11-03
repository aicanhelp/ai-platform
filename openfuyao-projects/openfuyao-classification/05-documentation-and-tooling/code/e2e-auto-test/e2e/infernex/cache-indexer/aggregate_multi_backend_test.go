package cacheindexer

import (
	"context"
	"fmt"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Cache Indexer Aggregate Multi Backend", Ordered, Label("cache-indexer", "npu", "26.06"), func() {
	var (
		ctx                context.Context
		cancel             context.CancelFunc
		infernexValuesFile string
		infernexCfg        *utils.ReleaseConfig
		aggregateRuntime   utils.AggregatedBackendRuntimeConfig
		cacheRuntime       utils.CacheIndexerRuntimeConfig
	)

	BeforeAll(func() {
		ctx, cancel = context.WithTimeout(context.Background(), 60*time.Minute)
		DeferCleanup(cancel)

		infernexCfg = mustHelmRelease("infernex")
		cleanupLeftoverInfernexAndCacheIndexer(ctx)

		By("使用 infernex 默认 values + aggregate-single topology values，并将聚合后端 replicas 调整为 2")
		var err error
		infernexValuesFile, aggregateRuntime, cacheRuntime, err = prepareAggregateMultiBackendValues()
		Expect(err).NotTo(HaveOccurred(), "构建 aggregate-multi values 失败")
		DeferCleanup(func() {
			_ = utils.CleanupTempValuesFile(infernexValuesFile)
		})

		cfg := *infernexCfg
		cfg.ValuesFile = infernexValuesFile
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("等待 2 个聚合后端与 Mooncake Master 就绪")
		Eventually(func() error {
			backendPods, err := utils.GetAggregatedBackendPodIdentities(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err != nil {
				return err
			}
			if len(backendPods) != aggregatedMultiBackendReplicaCount {
				return fmt.Errorf("expected %d aggregated backend pods, got %d", aggregatedMultiBackendReplicaCount, len(backendPods))
			}
			return nil
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			return utils.WaitForMooncakeMasterPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 60*time.Second, 5*time.Second).Should(Succeed())
	})

	AfterAll(func() {
		uninstallReleaseIfPresent(infernexCfg, "infernex uninstall success")
	})

	// 聚合双后端保持请求顺序，且仅预热的后端命中。
	It("cache-indexer_FIT_008-聚合双后端保持请求顺序", Label("P1"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("等待 cache-indexer Pod Running 且 Ready=1/1")
		Eventually(func() error {
			return utils.AssertCacheIndexerPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		By("等待 2 个聚合后端 Pod Running 且 Ready=1/1，并按 Pod 名稳定排序")
		var backendPods []utils.AggregatedBackendPodIdentity
		Eventually(func() error {
			var err error
			backendPods, err = utils.GetAggregatedBackendPodIdentities(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err != nil {
				return err
			}
			if len(backendPods) != aggregatedMultiBackendReplicaCount {
				return fmt.Errorf("expected %d aggregated backend pods, got %d", aggregatedMultiBackendReplicaCount, len(backendPods))
			}
			return nil
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		backendA := backendPods[0]
		backendB := backendPods[1]

		By("通过聚合 service 对 PromptA 分词，并截取前 2 个 block 作为查询 token_ids")
		promptA := buildAggregatePromptA()
		var queryTokens []int32
		Eventually(func() error {
			promptATokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
			if err != nil {
				return err
			}
			queryTokenCount := aggregatedMultiBackendQueryBlocks * aggregateRuntime.BlockSize
			if len(promptATokens) < queryTokenCount {
				return fmt.Errorf("expected prompt tokens >= %d, got %d", queryTokenCount, len(promptATokens))
			}
			queryTokens = append([]int32(nil), promptATokens[:queryTokenCount]...)
			return nil
		}, 2*time.Minute, 5*time.Second).Should(Succeed())

		By("仅向后端 A 发送 PromptA 并确认推理成功")
		Eventually(func() error {
			return utils.DoAggregatedBackendInferenceViaPod(ctx, infernexCfg.Namespace, backendA.Name, aggregateRuntime, promptA)
		}, 2*time.Minute, 5*time.Second).Should(Succeed())

		By("在 60s 内轮询 hit-rate，请求 server_ip 顺序为 [B Pod IP, A Pod IP]，直到 B L1=0 且 A L1=1")
		var latestHitRateBody []byte
		Eventually(func() error {
			scores, body, err := utils.QueryCacheIndexerHitRateForServerIPs(
				ctx,
				infernexCfg.Namespace,
				[]string{backendB.IP, backendA.IP},
				queryTokens,
				cacheRuntime,
				aggregateRuntime.BlockSize,
			)
			if err != nil {
				return err
			}
			latestHitRateBody = body
			if scores[0].L1HitRatio != 0.0 {
				return fmt.Errorf("expected backend B L1=0, got l1=%v l3=%v body=%s", scores[0].L1HitRatio, scores[0].L3HitRatio, string(body))
			}
			if scores[1].L1HitRatio != 1.0 {
				return fmt.Errorf("expected backend A L1=1, got l1=%v l3=%v body=%s", scores[1].L1HitRatio, scores[1].L3HitRatio, string(body))
			}
			return nil
		}, 60*time.Second, 5*time.Second).Should(Succeed())

		GinkgoWriter.Printf("FIT_008 latest hit-rate response: %s\n", string(latestHitRateBody))
	})
})

func prepareAggregateMultiBackendValues() (string, utils.AggregatedBackendRuntimeConfig, utils.CacheIndexerRuntimeConfig, error) {
	baseValuesFile, _, _, err := prepareAggregateSingleBackendValues(false)
	if err != nil {
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}

	valuesFile, err := utils.SetAggregatedReplicasInValues(baseValuesFile, aggregatedMultiBackendReplicaCount)
	_ = utils.CleanupTempValuesFile(baseValuesFile)
	if err != nil {
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}

	aggregateRuntime, err := utils.LoadAggregatedBackendRuntimeConfig(valuesFile)
	if err != nil {
		_ = utils.CleanupTempValuesFile(valuesFile)
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}
	cacheRuntime, err := utils.LoadEmbeddedCacheIndexerRuntimeConfig(valuesFile)
	if err != nil {
		_ = utils.CleanupTempValuesFile(valuesFile)
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}

	return valuesFile, aggregateRuntime, cacheRuntime, nil
}
