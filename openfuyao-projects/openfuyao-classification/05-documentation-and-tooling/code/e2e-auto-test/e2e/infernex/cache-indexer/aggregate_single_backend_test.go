package cacheindexer

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	frameworkk8s "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("Cache Indexer Aggregate Single Backend", Ordered, Label("cache-indexer", "npu", "26.06"), func() {
	var (
		ctx                context.Context
		cancel             context.CancelFunc
		baseValuesFile     string
		infernexValuesFile string
		generatedValues    []string
		infernexCfg        *utils.ReleaseConfig
		aggregateRuntime   utils.AggregatedBackendRuntimeConfig
		cacheRuntime       utils.CacheIndexerRuntimeConfig
		latestHitRate      []byte
	)

	BeforeAll(func() {
		ctx, cancel = context.WithTimeout(context.Background(), 180*time.Minute)
		DeferCleanup(cancel)

		infernexCfg = mustHelmRelease("infernex")
		cleanupLeftoverInfernexAndCacheIndexer(ctx)

		By("使用 infernex 默认 values + aggregate-single topology values 安装单聚合后端场景（含内置 cache-indexer）")
		var err error
		infernexValuesFile, aggregateRuntime, cacheRuntime, err = prepareAggregateSingleBackendValues(false)
		Expect(err).NotTo(HaveOccurred(), "构建 aggregate-single values 失败")
		baseValuesFile = infernexValuesFile
		DeferCleanup(func() {
			_ = utils.CleanupTempValuesFile(baseValuesFile)
			for _, generatedFile := range generatedValues {
				_ = utils.CleanupTempValuesFile(generatedFile)
			}
		})

		cfg := *infernexCfg
		cfg.ValuesFile = infernexValuesFile
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("等待聚合后端与 Mooncake Master 就绪")
		Eventually(func() error {
			_, _, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			return utils.WaitForMooncakeMasterPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 60*time.Second, 5*time.Second).Should(Succeed())
	})

	AfterAll(func() {
		uninstallReleaseIfPresent(infernexCfg, "infernex uninstall success")
	})

	// 聚合单后端同 Prompt 全命中。
	It("cache-indexer_FIT_002-聚合单后端同Prompt全命中;cache-indexer_FIT_012-TE映射仅来自MasterSegments", Label("P0"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("等待 cache-indexer Pod Running 且 Ready=1/1")
		Eventually(func() error {
			return utils.AssertCacheIndexerPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		By("等待聚合后端 Pod Running 且 Ready=1/1，并获取聚合 Pod IP")
		var backendPodName string
		var backendPodIP string
		Eventually(func() error {
			var err error
			backendPodName, backendPodIP, err = utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Expect(backendPodName).NotTo(BeEmpty(), "聚合 Pod Name 不应为空")
		Expect(backendPodIP).NotTo(BeEmpty(), "聚合 Pod IP 不应为空")

		By("显式确认当前环境不存在手动 TE 来源")
		Expect(utils.AssertAggregatedBackendNoManualTESource(ctx, k8sClient, infernexCfg.Namespace, backendPodName, aggregateRuntime.ServiceName, infernexValuesFile)).To(Succeed())

		By("等待 cache-indexer 完成后端发现与 L1 订阅建链")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "L1 ZMQ SUB connected")
		}, 60*time.Second, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			logText, err := utils.GetCacheIndexerLogText(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer)
			if err != nil {
				return err
			}
			if !strings.Contains(logText, backendPodIP) {
				return fmt.Errorf("cache-indexer log does not contain backend pod ip %s", backendPodIP)
			}
			if !strings.Contains(logText, "\"mooncakeResolved\":true") {
				return fmt.Errorf("cache-indexer log does not contain mooncakeResolved=true")
			}
			return nil
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		promptA := buildAggregatePromptA()

		By("通过聚合后端 /tokenize 获取与 PromptA 一致的 token_ids")
		tokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 失败")
		Expect(len(tokens)).To(BeNumerically(">=", aggregatedPromptMinTokens), "PromptA token_ids 长度应当至少达到 256")
		GinkgoWriter.Printf("PromptA token count=%d fullBlocks=%d backendPodIP=%s\n", len(tokens), len(tokens)/aggregateRuntime.BlockSize, backendPodIP)

		By("向聚合后端发送一次 PromptA 推理请求并确认成功")
		Eventually(func() error {
			return utils.DoAggregatedBackendInference(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())

		By("在 120 秒内轮询 /kv-cache/hit-rate，校验聚合后端 L1/L3 均达到 1.0")
		Eventually(func() error {
			var pollErr error
			latestHitRate, pollErr = utils.QueryAggregateSingleBackendFullHit(ctx, infernexCfg.Namespace, backendPodIP, tokens, cacheRuntime, aggregateRuntime.BlockSize)
			return pollErr
		}, 120*time.Second, 5*time.Second).Should(Succeed())

		By("校验 responseBody 含该 Pod IP 与命中率字段")
		Expect(string(latestHitRate)).To(ContainSubstring(backendPodIP))
		Expect(string(latestHitRate)).To(ContainSubstring("l1_hit_ratio"))
		Expect(string(latestHitRate)).To(ContainSubstring("l3_hit_ratio"))

		By("采集 cache-indexer Pod info 日志并校验 hit-rate request completed")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "hit-rate request completed")
		}, 30*time.Second, 5*time.Second).Should(Succeed())

		By("显式确认 discovery 日志中不存在 missing transport_endpoint，说明 L3 TE 映射并非来自手动来源")
		Expect(utils.AssertCacheIndexerLogNotContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "master /get_all_segments missing vLLM transport_endpoint")).To(Succeed())

		utils.DumpPodsAndServices(ctx, k8sClient, infernexCfg.Namespace)
	})

	// 聚合冷 Prompt 命中率为零。
	It("cache-indexer_FIT_005-聚合冷Prompt命中率为零", Label("P1"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("等待聚合后端 Pod Running 且 Ready=1/1，并获取聚合 Pod IP")
		var backendPodName string
		var backendPodIP string
		Eventually(func() error {
			var err error
			backendPodName, backendPodIP, err = utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Expect(backendPodName).NotTo(BeEmpty(), "聚合 Pod Name 不应为空")
		Expect(backendPodIP).NotTo(BeEmpty(), "聚合 Pod IP 不应为空")

		By("构造未推理过的 PromptCold，并通过 /tokenize 获取 token_ids")
		promptCold := buildAggregatePromptCold()
		tokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptCold)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 失败")
		Expect(len(tokens)).To(BeNumerically(">=", aggregatedPromptMinTokens), "PromptCold token_ids 长度应当至少达到 256")
		GinkgoWriter.Printf("PromptCold token count=%d fullBlocks=%d backendPodIP=%s\n", len(tokens), len(tokens)/aggregateRuntime.BlockSize, backendPodIP)

		By("在 20 秒内多次轮询 /kv-cache/hit-rate，校验冷 Prompt 始终为 0.0/0.0")
		Consistently(func() error {
			var pollErr error
			latestHitRate, pollErr = utils.QueryAggregateSingleBackendZeroHit(ctx, infernexCfg.Namespace, backendPodIP, tokens, cacheRuntime, aggregateRuntime.BlockSize)
			return pollErr
		}, 20*time.Second, 5*time.Second).Should(Succeed())

		By("校验 responseBody 含该 Pod IP 与命中率字段")
		Expect(string(latestHitRate)).To(ContainSubstring(backendPodIP))
		Expect(string(latestHitRate)).To(ContainSubstring("l1_hit_ratio"))
		Expect(string(latestHitRate)).To(ContainSubstring("l3_hit_ratio"))

		By("采集 cache-indexer 日志，确认未出现 score evaluation failed")
		Expect(utils.AssertCacheIndexerLogNotContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "score evaluation failed")).To(Succeed())
	})

	// 聚合缓存前缀返回部分命中。
	It("cache-indexer_FIT_006-聚合缓存前缀返回部分命中", Label("P1"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		Expect(aggregateRuntime.BlockSize).To(Equal(cacheIndexerRequestBlockSize), "聚合 block_size 应保持 128")

		By("等待聚合后端 Pod Running 且 Ready=1/1，并获取聚合 Pod IP")
		var backendPodName string
		var backendPodIP string
		Eventually(func() error {
			var err error
			backendPodName, backendPodIP, err = utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Expect(backendPodName).NotTo(BeEmpty(), "聚合 Pod Name 不应为空")
		Expect(backendPodIP).NotTo(BeEmpty(), "聚合 Pod IP 不应为空")

		By("构造 PromptA 与 PromptCold，并通过 /tokenize 获取 token_ids")
		promptA := buildAggregatePromptA()
		promptCold := buildAggregatePromptCold()
		promptATokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 获取 PromptA token_ids 失败")
		promptColdTokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptCold)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 获取 PromptCold token_ids 失败")

		prefixTokenCount := aggregatedPromptPrefixBlocks * aggregateRuntime.BlockSize
		totalTokenCount := aggregatedPromptTotalBlocks * aggregateRuntime.BlockSize
		Expect(len(promptATokens)).To(BeNumerically(">=", totalTokenCount), "PromptA token_ids 长度应当至少达到 3 个 block")
		Expect(len(promptColdTokens)).To(BeNumerically(">=", aggregateRuntime.BlockSize), "PromptCold token_ids 长度应当至少达到 1 个 block")
		Expect(promptColdTokens[:aggregateRuntime.BlockSize]).NotTo(Equal(promptATokens[prefixTokenCount:totalTokenCount]), "PromptCold 首 block 不应与 PromptA 第 3 个 block 完全相同")

		promptABTokens := buildAggregatePromptABTokens(promptATokens, promptColdTokens, aggregateRuntime.BlockSize)
		Expect(len(promptABTokens)).To(Equal(totalTokenCount), "PromptAB token_ids 长度应当是 3 个 block")
		GinkgoWriter.Printf("PromptAB token count=%d prefixBlocks=%d backendPodIP=%s\n", len(promptABTokens), aggregatedPromptPrefixBlocks, backendPodIP)

		By("向聚合后端发送一次 PromptA 推理请求并确认成功")
		Eventually(func() error {
			return utils.DoAggregatedBackendInference(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())

		By("在 60 秒内轮询 /kv-cache/hit-rate，校验 PromptAB 至少一次命中 2/3")
		Eventually(func() error {
			var pollErr error
			latestHitRate, pollErr = utils.QueryAggregateSingleBackendExpectedHit(
				ctx,
				infernexCfg.Namespace,
				backendPodIP,
				promptABTokens,
				cacheRuntime,
				aggregateRuntime.BlockSize,
				aggregatedPartialHitRatio,
				aggregatedPartialHitRatio,
				aggregatedHitRatioTolerance,
			)
			return pollErr
		}, 60*time.Second, 5*time.Second).Should(Succeed())

		By("校验 responseBody 含该 Pod IP 与命中率字段")
		Expect(string(latestHitRate)).To(ContainSubstring(backendPodIP))
		Expect(string(latestHitRate)).To(ContainSubstring("l1_hit_ratio"))
		Expect(string(latestHitRate)).To(ContainSubstring("l3_hit_ratio"))

		By("采集 cache-indexer Pod info 日志并校验 hit-rate request completed")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "hit-rate request completed")
		}, 30*time.Second, 5*time.Second).Should(Succeed())
	})

	// 聚合首 Block 变化命中归零。
	It("cache-indexer_FIT_007-聚合首Block变化命中归零", Label("P1"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		Expect(aggregateRuntime.BlockSize).To(Equal(cacheIndexerRequestBlockSize), "聚合 block_size 应保持 128")

		By("等待聚合后端 Pod Running 且 Ready=1/1，并获取聚合 Pod IP")
		var backendPodName string
		var backendPodIP string
		Eventually(func() error {
			var err error
			backendPodName, backendPodIP, err = utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Expect(backendPodName).NotTo(BeEmpty(), "聚合 Pod Name 不应为空")
		Expect(backendPodIP).NotTo(BeEmpty(), "聚合 Pod IP 不应为空")

		By("构造 PromptA，并通过 /tokenize 获取 token_ids")
		promptA := buildAggregatePromptA()
		promptATokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 获取 PromptA token_ids 失败")

		prefixTokenCount := aggregatedPromptPrefixBlocks * aggregateRuntime.BlockSize
		Expect(len(promptATokens)).To(BeNumerically(">=", prefixTokenCount), "PromptA token_ids 长度应当至少达到 2 个 block")

		promptAMutatedTokens := buildAggregatePromptAMutatedTokens(promptATokens, aggregateRuntime.BlockSize)
		Expect(len(promptAMutatedTokens)).To(Equal(prefixTokenCount), "PromptA_mut1 token_ids 长度应当是 2 个 block")
		Expect(promptAMutatedTokens[:aggregateRuntime.BlockSize]).NotTo(Equal(promptATokens[:aggregateRuntime.BlockSize]), "PromptA_mut1 第 1 个 block 必须不同")
		Expect(promptAMutatedTokens[aggregateRuntime.BlockSize:prefixTokenCount]).To(Equal(promptATokens[aggregateRuntime.BlockSize:prefixTokenCount]), "PromptA_mut1 仅应修改首 block")
		GinkgoWriter.Printf("PromptA_mut1 token count=%d mutatedIndex=%d backendPodIP=%s\n", len(promptAMutatedTokens), aggregatedMutatedTokenIndex, backendPodIP)

		By("向聚合后端发送一次 PromptA 推理请求并确认成功")
		Eventually(func() error {
			return utils.DoAggregatedBackendInference(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())

		By("在 60 秒内轮询 /kv-cache/hit-rate，校验 PromptA_mut1 至少一次回到 0.0/0.0")
		Eventually(func() error {
			var pollErr error
			latestHitRate, pollErr = utils.QueryAggregateSingleBackendZeroHit(ctx, infernexCfg.Namespace, backendPodIP, promptAMutatedTokens, cacheRuntime, aggregateRuntime.BlockSize)
			return pollErr
		}, 60*time.Second, 5*time.Second).Should(Succeed())

		By("校验 responseBody 含该 Pod IP 与命中率字段")
		Expect(string(latestHitRate)).To(ContainSubstring(backendPodIP))
		Expect(string(latestHitRate)).To(ContainSubstring("l1_hit_ratio"))
		Expect(string(latestHitRate)).To(ContainSubstring("l3_hit_ratio"))

		By("采集 cache-indexer 日志，确认未出现 score evaluation failed")
		Expect(utils.AssertCacheIndexerLogNotContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "score evaluation failed")).To(Succeed())
	})

	// logLevel 为 debug 输出调试日志。
	It("cache-indexer_FIT_015-logLevel为debug输出调试日志", Label("P2"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("将 cache-indexer log.level 调整为 debug 并升级 release")
		aggregateRuntime, cacheRuntime, infernexValuesFile = upgradeAggregateSingleWithVariant(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
			"debug-log",
			func(values map[string]any) error {
				cacheIndexerValues := utils.EnsureNestedMap(values, "cache-indexer")
				logValues := utils.EnsureNestedMap(cacheIndexerValues, "log")
				logValues["level"] = "debug"
				return nil
			},
		)

		By("等待聚合后端和 cache-indexer Ready")
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)

		By("等待 cache-indexer 完成发现并获取后端 Pod IP")
		var backendPodIP string
		Eventually(func() error {
			_, ip, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err == nil {
				backendPodIP = ip
			}
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "L1 ZMQ SUB connected")
		}, 60*time.Second, 5*time.Second).Should(Succeed())

		By("发送一次命中率请求触发 debug 日志")
		promptA := buildAggregatePromptA()
		tokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 失败")
		Expect(len(tokens)).To(BeNumerically(">=", aggregatedPromptMinTokens), "PromptA token_ids 长度应当至少达到 256")
		Eventually(func() error {
			return utils.DoAggregatedBackendInference(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			_, _, err := utils.QueryAggregateSingleBackendHitRate(ctx, infernexCfg.Namespace, backendPodIP, tokens, cacheRuntime, aggregateRuntime.BlockSize)
			return err
		}, 60*time.Second, 5*time.Second).Should(Succeed())

		By("校验日志包含 debug 级请求日志")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "hit-rate request received")
		}, 20*time.Second, 2*time.Second).Should(Succeed())
	})

	// blockKeySeed 不一致命中为零。
	It("cache-indexer_FIT_017-blockKeySeed不一致命中为零", Label("P2"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("将 cache-indexer blockKey.pythonHashSeed 调整为 1 并升级 release")
		aggregateRuntime, cacheRuntime, infernexValuesFile = upgradeAggregateSingleWithVariant(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
			"python-hash-seed",
			func(values map[string]any) error {
				cacheIndexerValues := utils.EnsureNestedMap(values, "cache-indexer")
				blockKeyValues := utils.EnsureNestedMap(cacheIndexerValues, "blockKey")
				blockKeyValues["pythonHashSeed"] = "1"
				return nil
			},
		)

		By("等待聚合后端和 cache-indexer Ready")
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)

		By("获取聚合后端 Pod IP，并用 PromptA 预热真实缓存")
		var backendPodIP string
		Eventually(func() error {
			_, ip, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err == nil {
				backendPodIP = ip
			}
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		promptA := buildAggregatePromptA()
		tokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 失败")
		Expect(len(tokens)).To(BeNumerically(">=", aggregatedPromptMinTokens), "PromptA token_ids 长度应当至少达到 256")
		Eventually(func() error {
			return utils.DoAggregatedBackendInference(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())

		By("在窗口内持续查询，确认 L1 命中始终为 0.0，且请求可稳定返回")
		Consistently(func() error {
			_, body, err := utils.QueryAggregateSingleBackendHitRate(ctx, infernexCfg.Namespace, backendPodIP, tokens, cacheRuntime, aggregateRuntime.BlockSize)
			if err != nil {
				return err
			}
			scoreResp, err := utils.ParseCacheIndexerHitRateResponse(body)
			if err != nil {
				return err
			}
			if len(scoreResp.ServerScoreList) != 1 {
				return fmt.Errorf("expected 1 server score, got %d body=%s", len(scoreResp.ServerScoreList), string(body))
			}
			score := scoreResp.ServerScoreList[0]
			if score.L1HitRatio != 0.0 || score.L3HitRatio != 0.0 {
				return fmt.Errorf("expected zero hit with mismatched pythonHashSeed, got l1=%v l3=%v body=%s", score.L1HitRatio, score.L3HitRatio, string(body))
			}
			return nil
		}, 20*time.Second, 5*time.Second).Should(Succeed())
	})

	// useIntBlockHashes 不一致命中为零。
	It("cache-indexer_FIT_018-useIntBlockHashes不一致命中为零", Label("P2"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("将 cache-indexer blockKey.useIntBlockHashes 调整为 false 并升级 release")
		aggregateRuntime, cacheRuntime, infernexValuesFile = upgradeAggregateSingleWithVariant(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
			"use-int-block-hashes",
			func(values map[string]any) error {
				cacheIndexerValues := utils.EnsureNestedMap(values, "cache-indexer")
				blockKeyValues := utils.EnsureNestedMap(cacheIndexerValues, "blockKey")
				blockKeyValues["useIntBlockHashes"] = false
				return nil
			},
		)

		By("等待聚合后端和 cache-indexer Ready")
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)

		By("获取聚合后端 Pod IP，并用 PromptA 预热真实缓存")
		var backendPodIP string
		Eventually(func() error {
			_, ip, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err == nil {
				backendPodIP = ip
			}
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		promptA := buildAggregatePromptA()
		tokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 失败")
		Expect(len(tokens)).To(BeNumerically(">=", aggregatedPromptMinTokens), "PromptA token_ids 长度应当至少达到 256")
		Eventually(func() error {
			return utils.DoAggregatedBackendInference(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())

		By("在窗口内持续查询，确认命中率始终为 0.0/0.0")
		Consistently(func() error {
			_, body, err := utils.QueryAggregateSingleBackendHitRate(ctx, infernexCfg.Namespace, backendPodIP, tokens, cacheRuntime, aggregateRuntime.BlockSize)
			if err != nil {
				return err
			}
			scoreResp, err := utils.ParseCacheIndexerHitRateResponse(body)
			if err != nil {
				return err
			}
			if len(scoreResp.ServerScoreList) != 1 {
				return fmt.Errorf("expected 1 server score, got %d body=%s", len(scoreResp.ServerScoreList), string(body))
			}
			score := scoreResp.ServerScoreList[0]
			if score.L1HitRatio != 0.0 {
				return fmt.Errorf("expected l1 zero hit with mismatched useIntBlockHashes, got l1=%v l3=%v body=%s", score.L1HitRatio, score.L3HitRatio, string(body))
			}
			return nil
		}, 20*time.Second, 5*time.Second).Should(Succeed())

		By("校验日志中未出现 score evaluation failed")
		Expect(utils.AssertCacheIndexerLogNotContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "score evaluation failed")).To(Succeed())
	})

	// 发现周期配置影响收敛时间。
	It("cache-indexer_FIT_019-发现周期配置影响收敛时间", Label("P2"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("将 cache-indexer discovery.refreshInterval 调整为 2s 并升级 release")
		aggregateRuntime, cacheRuntime, infernexValuesFile = upgradeAggregateSingleWithVariant(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
			"refresh-interval",
			func(values map[string]any) error {
				cacheIndexerValues := utils.EnsureNestedMap(values, "cache-indexer")
				discoveryValues := utils.EnsureNestedMap(cacheIndexerValues, "discovery")
				discoveryValues["refreshInterval"] = "2s"
				return nil
			},
		)

		By("等待聚合后端和 cache-indexer Ready")
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)

		By("在短窗口内等待 discovery started 和 vLLM pod added 日志")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(
				ctx,
				k8sClient,
				infernexCfg.Namespace,
				cacheIndexerLogContainer,
				"discovery started",
				"refreshInterval",
				"2s",
			)
		}, 20*time.Second, 2*time.Second).Should(Succeed())

		var backendPodIP string
		Eventually(func() error {
			_, ip, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err == nil {
				backendPodIP = ip
			}
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(
				ctx,
				k8sClient,
				infernexCfg.Namespace,
				cacheIndexerLogContainer,
				"vLLM pod added",
				backendPodIP,
			)
		}, 10*time.Second, 2*time.Second).Should(Succeed())
	})

	// tokenIDs 不足一个 Block 返回零。
	It("cache-indexer_FIT_040-tokenIDs不足一个Block返回零", Label("P3"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("恢复 aggregate-single 基线配置")
		aggregateRuntime, cacheRuntime, infernexValuesFile = restoreAggregateSingleBaseline(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
		)

		By("等待聚合后端和 cache-indexer Ready，并获取聚合后端 Pod IP")
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)
		var backendPodIP string
		Eventually(func() error {
			_, ip, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err == nil {
				backendPodIP = ip
			}
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		By("请求 token_ids 长度为 127，确认返回 0.0/0.0")
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			[]string{backendPodIP},
			utils.BuildSequentialTokenIDs(cacheIndexerRequestBlockSize-1),
			cacheIndexerRequestBlockSize,
			"",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			infernexCfg.Namespace,
			cacheRuntime.ServiceName,
			cacheRuntime.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")
		Expect(utils.AssertSingleZeroHitResponse(statusCode, body, backendPodIP)).To(Succeed())
	})

	// tokenIDs 为空返回零。
	It("cache-indexer_FIT_041-tokenIDs为空返回零", Label("P3"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("等待聚合后端 Pod Ready，并获取聚合后端 Pod IP")
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)
		var backendPodIP string
		Eventually(func() error {
			_, ip, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err == nil {
				backendPodIP = ip
			}
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		By("请求空 token_ids，确认返回 0.0/0.0")
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			[]string{backendPodIP},
			[]int32{},
			cacheIndexerRequestBlockSize,
			"",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			infernexCfg.Namespace,
			cacheRuntime.ServiceName,
			cacheRuntime.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")
		Expect(utils.AssertSingleZeroHitResponse(statusCode, body, backendPodIP)).To(Succeed())
	})

	// Mooncake 端口名缺失使用兜底端口。
	It("cache-indexer_FIT_021-Mooncake端口名缺失使用兜底端口", Label("P2"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("将 cache-indexer Mooncake HTTP 端口名置空，仅保留兜底端口 9003")
		aggregateRuntime, cacheRuntime, infernexValuesFile = upgradeAggregateSingleWithVariant(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
			"mooncake-http-fallback",
			func(values map[string]any) error {
				cacheIndexerValues := utils.EnsureNestedMap(values, "cache-indexer")
				discoveryValues := utils.EnsureNestedMap(cacheIndexerValues, "discovery")
				mooncakeValues := utils.EnsureNestedMap(discoveryValues, "mooncakeMaster")
				mooncakeValues["httpPortName"] = ""
				mooncakeValues["httpPort"] = 9003
				return nil
			},
		)

		By("等待聚合后端和 cache-indexer Ready，并获取聚合后端 Pod IP")
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)
		var backendPodIP string
		Eventually(func() error {
			_, ip, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err == nil {
				backendPodIP = ip
			}
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		By("等待 L3 目标更新日志回显 9003，并验证 PromptA 最终可通过 fallback 端口实现 L3 命中")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(
				ctx,
				k8sClient,
				infernexCfg.Namespace,
				cacheIndexerLogContainer,
				"L3 mooncake target updated",
				":9003",
			)
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		promptA := buildAggregatePromptA()
		tokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 失败")
		Expect(len(tokens)).To(BeNumerically(">=", aggregatedPromptMinTokens), "PromptA token_ids 长度应当至少达到 256")
		Eventually(func() error {
			return utils.DoAggregatedBackendInference(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			var pollErr error
			latestHitRate, pollErr = utils.QueryAggregateSingleBackendExpectedHit(
				ctx,
				infernexCfg.Namespace,
				backendPodIP,
				tokens,
				cacheRuntime,
				aggregateRuntime.BlockSize,
				0.0,
				1.0,
				0.0,
			)
			return pollErr
		}, 120*time.Second, 5*time.Second).Should(Succeed())
	})

	// 自定义 pdRoleValue 发现 aggregate。
	It("cache-indexer_FIT_027-自定义pdRoleValue发现aggregate", Label("P2"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("将 cache-indexer discovery.labels.pdRoleValue 显式收敛到 aggregate")
		aggregateRuntime, cacheRuntime, infernexValuesFile = upgradeAggregateSingleWithVariant(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
			"pd-role-aggregate-only",
			func(values map[string]any) error {
				cacheIndexerValues := utils.EnsureNestedMap(values, "cache-indexer")
				discoveryValues := utils.EnsureNestedMap(cacheIndexerValues, "discovery")
				labelValues := utils.EnsureNestedMap(discoveryValues, "labels")
				labelValues["pdRoleValue"] = []string{"aggregate"}
				return nil
			},
		)

		By("等待聚合后端和 cache-indexer Ready，并确认聚合 Pod 被发现")
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)
		var backendPodIP string
		Eventually(func() error {
			_, ip, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err == nil {
				backendPodIP = ip
			}
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(
				ctx,
				k8sClient,
				infernexCfg.Namespace,
				cacheIndexerLogContainer,
				"vLLM pod added",
				backendPodIP,
			)
		}, 60*time.Second, 5*time.Second).Should(Succeed())

		promptA := buildAggregatePromptA()
		tokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 失败")
		Eventually(func() error {
			_, body, err := utils.QueryAggregateSingleBackendHitRate(ctx, infernexCfg.Namespace, backendPodIP, tokens, cacheRuntime, aggregateRuntime.BlockSize)
			if err != nil {
				return err
			}
			if !strings.Contains(string(body), backendPodIP) {
				return fmt.Errorf("response does not contain backend ip %s: %s", backendPodIP, string(body))
			}
			return nil
		}, 30*time.Second, 5*time.Second).Should(Succeed())
	})

	// 单 Master 发现并更新 L3 目标。
	It("cache-indexer_FIT_029-单Master发现并更新L3目标", Label("P2"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("恢复 aggregate-single 基线配置")
		aggregateRuntime, cacheRuntime, infernexValuesFile = restoreAggregateSingleBaseline(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
		)

		By("等待聚合后端和 cache-indexer Ready，并确认只有 1 个 Mooncake Master")
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)
		Eventually(func() error {
			pods, err := k8sClient.ListPods(ctx, infernexCfg.Namespace, metav1.ListOptions{})
			if err != nil {
				return err
			}
			readyMasters := 0
			for _, pod := range pods.Items {
				if pod.Labels["openfuyao.com/kvmanager"] != "mooncake" || pod.Status.Phase != corev1.PodRunning {
					continue
				}
				for _, cond := range pod.Status.Conditions {
					if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
						readyMasters++
						break
					}
				}
			}
			if readyMasters != 1 {
				return fmt.Errorf("expected exactly 1 ready mooncake master, got %d", readyMasters)
			}
			return nil
		}, 30*time.Second, 3*time.Second).Should(Succeed())

		By("等待日志出现 Mooncake master changed 和 L3 目标更新")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "Mooncake master changed")
		}, 60*time.Second, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "L3 mooncake target updated")
		}, 60*time.Second, 5*time.Second).Should(Succeed())

		var backendPodIP string
		Eventually(func() error {
			_, ip, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err == nil {
				backendPodIP = ip
			}
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		promptA := buildAggregatePromptA()
		tokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 失败")
		Eventually(func() error {
			_, body, err := utils.QueryAggregateSingleBackendHitRate(ctx, infernexCfg.Namespace, backendPodIP, tokens, cacheRuntime, aggregateRuntime.BlockSize)
			if err != nil {
				return err
			}
			if !strings.Contains(string(body), backendPodIP) {
				return fmt.Errorf("response does not contain backend ip %s: %s", backendPodIP, string(body))
			}
			return nil
		}, 30*time.Second, 5*time.Second).Should(Succeed())
	})

	// L3 轮询间隔配置生效。
	It("cache-indexer_FIT_032-L3轮询间隔配置生效", Label("P2"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("将 cache-indexer log.level 调整为 debug，ingest.l3.pollInterval 调整为 2s")
		aggregateRuntime, cacheRuntime, infernexValuesFile = upgradeAggregateSingleWithVariant(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
			"l3-poll-interval",
			func(values map[string]any) error {
				cacheIndexerValues := utils.EnsureNestedMap(values, "cache-indexer")
				logValues := utils.EnsureNestedMap(cacheIndexerValues, "log")
				logValues["level"] = "debug"
				ingestValues := utils.EnsureNestedMap(cacheIndexerValues, "ingest")
				l3Values := utils.EnsureNestedMap(ingestValues, "l3")
				l3Values["pollInterval"] = "2s"
				return nil
			},
		)

		By("等待聚合后端和 cache-indexer Ready")
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)

		By("等待日志回显 interval=2，并在窗口内观察到多次 L3 poller refresh")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(
				ctx,
				k8sClient,
				infernexCfg.Namespace,
				cacheIndexerLogContainer,
				"L3 mooncake target updated",
				"\"interval\":2",
			)
		}, 60*time.Second, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			logText, err := utils.GetCacheIndexerLogText(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer)
			if err != nil {
				return err
			}
			if strings.Count(logText, "L3 poller refresh") < 2 {
				return fmt.Errorf("expected at least 2 L3 poller refresh logs")
			}
			return nil
		}, 20*time.Second, 2*time.Second).Should(Succeed())
	})

	// cacheSalt 不同命中隔离。
	It("cache-indexer_FIT_044-cacheSalt不同命中隔离", Label("P3"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("恢复 aggregate-single 基线配置")
		aggregateRuntime, cacheRuntime, infernexValuesFile = restoreAggregateSingleBaseline(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
		)

		By("等待聚合后端和 cache-indexer Ready，并获取聚合后端 Pod IP")
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)
		var backendPodIP string
		Eventually(func() error {
			_, ip, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err == nil {
				backendPodIP = ip
			}
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		promptA := buildAggregatePromptA()
		tokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 失败")
		Expect(len(tokens)).To(BeNumerically(">=", aggregatedPromptMinTokens), "PromptA token_ids 长度应当至少达到 256")
		Eventually(func() error {
			return utils.DoAggregatedBackendInference(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())

		By("使用相同 token_ids 但附带 cache_salt 请求，确认 L1 命中归零")
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			[]string{backendPodIP},
			tokens,
			int64(aggregateRuntime.BlockSize),
			"case-salt",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			infernexCfg.Namespace,
			cacheRuntime.ServiceName,
			cacheRuntime.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")
		Expect(statusCode).To(Equal(http.StatusOK))
		resp, err := utils.ParseCacheIndexerHitRateResponse(body)
		Expect(err).NotTo(HaveOccurred(), "解析 /kv-cache/hit-rate 响应失败")
		Expect(resp.ServerScoreList).To(HaveLen(1))
		Expect(resp.ServerScoreList[0].ServerIP).To(Equal(backendPodIP))
		Expect(resp.ServerScoreList[0].L1HitRatio).To(Equal(0.0))
	})

	// 错误 pdRole 标签不被发现。
	It("cache-indexer_FIT_045-错误pdRole标签不被发现", Label("P3"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("恢复 aggregate-single 基线配置")
		aggregateRuntime, cacheRuntime, infernexValuesFile = restoreAggregateSingleBaseline(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
		)
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)

		By("创建带错误 pdRole=worker 的 Ready 临时 Pod")
		podName := utils.UniqueTemporaryPodName("fake-vllm-worker")
		_, err := utils.CreateTemporaryVLLMPod(
			ctx,
			k8sClient,
			infernexCfg.Namespace,
			podName,
			map[string]string{
				"openfuyao.com/engine": "vllm",
				"openfuyao.com/pdRole": "worker",
			},
			true,
		)
		Expect(err).NotTo(HaveOccurred(), "创建错误 pdRole 临时 Pod 失败")
		defer func() {
			_ = utils.DeletePodIfExists(ctx, k8sClient, infernexCfg.Namespace, podName)
			Expect(utils.WaitForPodGone(ctx, k8sClient, infernexCfg.Namespace, podName, 60*time.Second)).To(Succeed())
		}()

		By("等待一个 refresh 窗口，确认日志中没有该 Pod 的 vLLM pod added，并且查询返回 0.0/0.0")
		time.Sleep(12 * time.Second)
		logText, err := utils.GetCacheIndexerLogText(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer)
		Expect(err).NotTo(HaveOccurred(), "读取 cache-indexer 日志失败")
		Expect(logText).NotTo(ContainSubstring(podName))

		pod, err := k8sClient.Clientset.CoreV1().Pods(infernexCfg.Namespace).Get(ctx, podName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "读取临时 Pod 失败")
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			[]string{pod.Status.PodIP},
			utils.BuildSequentialTokenIDs(cacheIndexerRequestBlockSize),
			cacheIndexerRequestBlockSize,
			"",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			infernexCfg.Namespace,
			cacheRuntime.ServiceName,
			cacheRuntime.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")
		Expect(utils.AssertSingleZeroHitResponse(statusCode, body, pod.Status.PodIP)).To(Succeed())
	})

	// 未 Ready 后端不被发现。
	It("cache-indexer_FIT_046-未Ready后端不被发现", Label("P3"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")

		By("恢复 aggregate-single 基线配置")
		aggregateRuntime, cacheRuntime, infernexValuesFile = restoreAggregateSingleBaseline(
			ctx,
			k8sClient,
			infernexCfg,
			baseValuesFile,
			&generatedValues,
		)
		waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)

		By("创建标签正确但未 Ready 的临时 Pod")
		podName := utils.UniqueTemporaryPodName("fake-vllm-not-ready")
		_, err := utils.CreateTemporaryVLLMPod(
			ctx,
			k8sClient,
			infernexCfg.Namespace,
			podName,
			map[string]string{
				"openfuyao.com/engine": "vllm",
				"openfuyao.com/pdRole": "aggregate",
			},
			false,
		)
		Expect(err).NotTo(HaveOccurred(), "创建未 Ready 临时 Pod 失败")
		defer func() {
			_ = utils.DeletePodIfExists(ctx, k8sClient, infernexCfg.Namespace, podName)
			Expect(utils.WaitForPodGone(ctx, k8sClient, infernexCfg.Namespace, podName, 60*time.Second)).To(Succeed())
		}()

		By("等待一个 refresh 窗口，确认日志中没有该 Pod 的 vLLM pod added，并且查询返回 0.0/0.0")
		time.Sleep(12 * time.Second)
		logText, err := utils.GetCacheIndexerLogText(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer)
		Expect(err).NotTo(HaveOccurred(), "读取 cache-indexer 日志失败")
		Expect(logText).NotTo(ContainSubstring(podName))

		pod, err := k8sClient.Clientset.CoreV1().Pods(infernexCfg.Namespace).Get(ctx, podName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "读取临时 Pod 失败")
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			[]string{pod.Status.PodIP},
			utils.BuildSequentialTokenIDs(cacheIndexerRequestBlockSize),
			cacheIndexerRequestBlockSize,
			"",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			infernexCfg.Namespace,
			cacheRuntime.ServiceName,
			cacheRuntime.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")
		Expect(utils.AssertSingleZeroHitResponse(statusCode, body, pod.Status.PodIP)).To(Succeed())
	})

	// 聚合无 Mooncake 仅 L1 命中。
	It("cache-indexer_FIT_004-聚合无Mooncake仅L1命中", Label("P1"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		Expect(aggregateRuntime.ServiceName).NotTo(BeEmpty(), "聚合后端 runtime 配置为空")

		By("记录当前聚合后端 Pod 身份，确保断开 Mooncake 过程中不触发 vLLM Pod 重启")
		var backendPodName string
		var backendPodIP string
		Eventually(func() error {
			var err error
			backendPodName, backendPodIP, err = utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Expect(backendPodName).NotTo(BeEmpty(), "聚合 Pod Name 不应为空")
		Expect(backendPodIP).NotTo(BeEmpty(), "聚合 Pod IP 不应为空")

		By("等待 cache-indexer Pod Running 且 Ready=1/1")
		Eventually(func() error {
			return utils.AssertCacheIndexerPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		By("等待 cache-indexer 完成后端发现与 L1 订阅建链")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "L1 ZMQ SUB connected")
		}, 60*time.Second, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			logText, err := utils.GetCacheIndexerLogText(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer)
			if err != nil {
				return err
			}
			if !strings.Contains(logText, backendPodIP) {
				return fmt.Errorf("cache-indexer log does not contain backend pod ip %s", backendPodIP)
			}
			if !strings.Contains(logText, "\"mooncakeResolved\":true") {
				return fmt.Errorf("cache-indexer log does not contain mooncakeResolved=true")
			}
			return nil
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("在断开 Mooncake Master 前，先为全新 PromptA-FIT004 执行一轮冷推理，确保不会复用前序用例的缓存状态")
		promptA := buildAggregatePromptAFIT004()
		tokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 失败")
		Expect(len(tokens)).To(BeNumerically(">=", aggregatedPromptMinTokens), "PromptA-FIT004 token_ids 长度应当至少达到 256")
		GinkgoWriter.Printf("PromptA-FIT004 token count=%d fullBlocks=%d backendPodIP=%s\n", len(tokens), len(tokens)/aggregateRuntime.BlockSize, backendPodIP)
		Eventually(func() error {
			return utils.DoAggregatedBackendInference(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())

		By("在断开 Mooncake Master 前轮询 /kv-cache/hit-rate，确认新的 PromptA-FIT004 已达到 L1/L3 全命中")
		Eventually(func() error {
			var pollErr error
			latestHitRate, pollErr = utils.QueryAggregateSingleBackendFullHit(ctx, infernexCfg.Namespace, backendPodIP, tokens, cacheRuntime, aggregateRuntime.BlockSize)
			return pollErr
		}, 120*time.Second, 5*time.Second).Should(Succeed())

		By("缩容 mooncake-master-deployment 到 0，断开 Mooncake Master 与 vLLM 后端连接，但不触发 vLLM Pod 重启")
		err = utils.ScaleMooncakeMasterDeployment(ctx, k8sClient, infernexCfg.Namespace, 0)
		Expect(err).NotTo(HaveOccurred(), "缩容 Mooncake Master Deployment 失败")

		By("等待 Mooncake Master 资源消失")
		Eventually(func() error {
			return utils.AssertMooncakeMasterPodAbsent(ctx, k8sClient, infernexCfg.Namespace)
		}, 2*time.Minute, 5*time.Second).Should(Succeed())

		By("确认聚合后端 Pod 未重建，名称与 IP 保持不变")
		Eventually(func() error {
			currentPodName, currentPodIP, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			if err != nil {
				return err
			}
			if currentPodName != backendPodName {
				return fmt.Errorf("aggregated backend pod restarted: before=%s after=%s", backendPodName, currentPodName)
			}
			if currentPodIP != backendPodIP {
				return fmt.Errorf("aggregated backend pod ip changed: before=%s after=%s", backendPodIP, currentPodIP)
			}
			return nil
		}, 2*time.Minute, 5*time.Second).Should(Succeed())

		By("等待 cache-indexer Pod Running 且 Ready=1/1")
		Eventually(func() error {
			return utils.AssertCacheIndexerPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		By("等待 cache-indexer 记录 Mooncake Master 断开后的 L3 目标变化日志")
		Eventually(func() error {
			logText, err := utils.GetCacheIndexerLogText(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer)
			if err != nil {
				return err
			}
			if strings.Contains(logText, "L3 mooncake target cleared") {
				return nil
			}
			if strings.Contains(logText, "\"next\":\"\"") &&
				(strings.Contains(logText, "Mooncake master changed") || strings.Contains(logText, "Mooncake Master 端点变更")) {
				return nil
			}
			return fmt.Errorf("cache-indexer log does not contain mooncake disconnect evidence")
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("在 120 秒内轮询 /kv-cache/hit-rate，校验已存在的 PromptA 前缀缓存退化为 L1=1.0 且 L3=0.0")
		Eventually(func() error {
			var pollErr error
			latestHitRate, pollErr = utils.QueryAggregateSingleBackendL1OnlyHit(ctx, infernexCfg.Namespace, backendPodIP, tokens, cacheRuntime, aggregateRuntime.BlockSize)
			return pollErr
		}, 120*time.Second, 5*time.Second).Should(Succeed())

		By("校验 responseBody 含该 Pod IP 与命中率字段")
		Expect(string(latestHitRate)).To(ContainSubstring(backendPodIP))
		Expect(string(latestHitRate)).To(ContainSubstring("l1_hit_ratio"))
		Expect(string(latestHitRate)).To(ContainSubstring("l3_hit_ratio"))

		By("采集 cache-indexer Pod info 日志并校验 hit-rate request completed")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "hit-rate request completed")
		}, 30*time.Second, 5*time.Second).Should(Succeed())

		utils.DumpPodsAndServices(ctx, k8sClient, infernexCfg.Namespace)
	})

	// 后端删除后旧 IP 命中清零。
	It("cache-indexer_FIT_009-后端删除后旧IP命中清零", Label("P1"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		Expect(aggregateRuntime.ServiceName).NotTo(BeEmpty(), "聚合后端 runtime 配置为空")

		By("先检查命名空间内是否已有 Mooncake Master，如不存在则先恢复")
		if err := utils.WaitForMooncakeMasterPodReady(ctx, k8sClient, infernexCfg.Namespace); err != nil {
			if absentErr := utils.AssertMooncakeMasterPodAbsent(ctx, k8sClient, infernexCfg.Namespace); absentErr == nil {
				err = utils.ScaleMooncakeMasterDeployment(ctx, k8sClient, infernexCfg.Namespace, 1)
				Expect(err).NotTo(HaveOccurred(), "恢复 Mooncake Master Deployment 失败")
			}
			Eventually(func() error {
				return utils.WaitForMooncakeMasterPodReady(ctx, k8sClient, infernexCfg.Namespace)
			}, 5*time.Minute, 5*time.Second).Should(Succeed())
		}

		By("等待 cache-indexer Pod Running 且 Ready=1/1")
		Eventually(func() error {
			return utils.AssertCacheIndexerPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		By("获取当前聚合后端 Pod 身份，作为待删除的旧后端")
		var backendPodName string
		var backendPodIP string
		Eventually(func() error {
			var err error
			backendPodName, backendPodIP, err = utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
			return err
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Expect(backendPodName).NotTo(BeEmpty(), "聚合 Pod Name 不应为空")
		Expect(backendPodIP).NotTo(BeEmpty(), "聚合 Pod IP 不应为空")

		By("恢复后先等待 cache-indexer 完成后端发现、Mooncake 反查和 L1 订阅建链")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "L1 ZMQ SUB connected")
		}, 60*time.Second, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			logText, err := utils.GetCacheIndexerLogText(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer)
			if err != nil {
				return err
			}
			if !strings.Contains(logText, backendPodIP) {
				return fmt.Errorf("cache-indexer log does not contain backend pod ip %s", backendPodIP)
			}
			if !strings.Contains(logText, "\"mooncakeResolved\":true") {
				return fmt.Errorf("cache-indexer log does not contain mooncakeResolved=true")
			}
			return nil
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("恢复后先对全新 PromptA-FIT009 执行一轮冷推理，并确认当前后端命中率收敛到 L1/L3 都为 1.0")
		promptA := buildAggregatePromptAFIT009()
		tokens, err := utils.TokenizePromptViaAggregatedBackend(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		Expect(err).NotTo(HaveOccurred(), "调用聚合后端 /tokenize 失败")
		Expect(len(tokens)).To(BeNumerically(">=", aggregatedPromptMinTokens), "PromptA-FIT009 token_ids 长度应当至少达到 256")
		GinkgoWriter.Printf("FIT_009 isolated token count=%d fullBlocks=%d backendPod=%s backendPodIP=%s\n", len(tokens), len(tokens)/aggregateRuntime.BlockSize, backendPodName, backendPodIP)
		Eventually(func() error {
			return utils.DoAggregatedBackendInference(ctx, infernexCfg.Namespace, aggregateRuntime, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			var pollErr error
			latestHitRate, pollErr = utils.QueryAggregateSingleBackendFullHit(ctx, infernexCfg.Namespace, backendPodIP, tokens, cacheRuntime, aggregateRuntime.BlockSize)
			return pollErr
		}, 120*time.Second, 5*time.Second).Should(Succeed())

		By("将当前聚合后端工作负载缩容到 0，确保旧后端实例被真正移除")
		err = utils.ScaleAggregatedBackendWorkload(ctx, k8sClient, infernexCfg.Namespace, backendPodName, 0)
		Expect(err).NotTo(HaveOccurred(), "缩容聚合后端工作负载失败")

		By("等待旧聚合后端 Pod 从命名空间中消失")
		Eventually(func() error {
			return utils.AssertAggregatedBackendPodsAbsent(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime.ServiceName)
		}, 2*time.Minute, 5*time.Second).Should(Succeed())

		By("等待 cache-indexer 记录旧后端下线与 L1 退订日志")
		Eventually(func() error {
			logText, err := utils.GetCacheIndexerLogText(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer)
			if err != nil {
				return err
			}
			if !strings.Contains(logText, "vLLM pod removed") || !strings.Contains(logText, backendPodIP) {
				return fmt.Errorf("cache-indexer log does not contain removed backend pod ip %s", backendPodIP)
			}
			if !strings.Contains(logText, "L1 unsubscribe completed") {
				return fmt.Errorf("cache-indexer log does not contain L1 unsubscribe completed")
			}
			return nil
		}, 120*time.Second, 5*time.Second).Should(Succeed())

		By("在 60 秒内轮询 /kv-cache/hit-rate，只要旧后端 IP 至少一次收敛到 0.0/0.0 就认为成功")
		Eventually(func() error {
			var pollErr error
			latestHitRate, pollErr = utils.QueryAggregateSingleBackendZeroHit(ctx, infernexCfg.Namespace, backendPodIP, tokens, cacheRuntime, aggregateRuntime.BlockSize)
			return pollErr
		}, 60*time.Second, 5*time.Second).Should(Succeed())

		By("校验 responseBody 回显旧后端 Pod IP 与命中率字段")
		Expect(string(latestHitRate)).To(ContainSubstring(backendPodIP))
		Expect(string(latestHitRate)).To(ContainSubstring("l1_hit_ratio"))
		Expect(string(latestHitRate)).To(ContainSubstring("l3_hit_ratio"))

		utils.DumpPodsAndServices(ctx, k8sClient, infernexCfg.Namespace)
	})

})

func prepareAggregateSingleBackendValues(disableMooncake bool) (string, utils.AggregatedBackendRuntimeConfig, utils.CacheIndexerRuntimeConfig, error) {
	configDir, err := utils.GetCacheIndexerConfigDir()
	if err != nil {
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}

	infernexBasePath := filepath.Join(configDir, "values-infernex.yaml")
	infernexTopologyPath := filepath.Join(configDir, "topology", "aggregate-single.yaml")

	valuesFile, err := utils.MergeValuesFiles(infernexBasePath, infernexTopologyPath)
	if err != nil {
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}

	if disableMooncake {
		noMooncakeValuesFile, err := utils.RemoveFirstInferenceBackendServiceKVTransferConfig(valuesFile)
		if err != nil {
			_ = utils.CleanupTempValuesFile(valuesFile)
			return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
		}
		_ = utils.CleanupTempValuesFile(valuesFile)
		valuesFile = noMooncakeValuesFile
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

func waitForAggregateSingleReady(
	ctx context.Context,
	k8sClient *frameworkk8s.K8SClient,
	namespace string,
	aggregateRuntime utils.AggregatedBackendRuntimeConfig,
	cacheRuntime utils.CacheIndexerRuntimeConfig,
) {
	Eventually(func() error {
		return utils.AssertCacheIndexerPodReady(ctx, k8sClient, namespace)
	}, 10*time.Minute, 5*time.Second).Should(Succeed())
	Eventually(func() error {
		_, _, err := utils.GetAggregatedBackendPodIdentity(ctx, k8sClient, namespace, aggregateRuntime.ServiceName)
		return err
	}, 10*time.Minute, 5*time.Second).Should(Succeed())
	Eventually(func() error {
		return utils.AssertCacheIndexerHealthEndpoints(ctx, namespace, cacheRuntime)
	}, 30*time.Second, 3*time.Second).Should(Succeed())
}

func upgradeAggregateSingleWithVariant(
	ctx context.Context,
	k8sClient *frameworkk8s.K8SClient,
	infernexCfg *utils.ReleaseConfig,
	baseValuesFile string,
	generatedValues *[]string,
	variantName string,
	mutate func(map[string]any) error,
) (utils.AggregatedBackendRuntimeConfig, utils.CacheIndexerRuntimeConfig, string) {
	valuesFile, aggregateRuntime, cacheRuntime, err := buildAggregateSingleVariantValues(baseValuesFile, variantName, mutate)
	Expect(err).NotTo(HaveOccurred(), "构建 %s values 失败", variantName)
	*generatedValues = append(*generatedValues, valuesFile)

	cfg := *infernexCfg
	cfg.ValuesFile = valuesFile
	err = utils.HelmUpgradeWithLatestFallback(&cfg)
	Expect(err).NotTo(HaveOccurred(), "升级 %s infernex release 失败", variantName)

	waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)
	return aggregateRuntime, cacheRuntime, valuesFile
}

func restoreAggregateSingleBaseline(
	ctx context.Context,
	k8sClient *frameworkk8s.K8SClient,
	infernexCfg *utils.ReleaseConfig,
	baseValuesFile string,
	generatedValues *[]string,
) (utils.AggregatedBackendRuntimeConfig, utils.CacheIndexerRuntimeConfig, string) {
	aggregateRuntime, err := utils.LoadAggregatedBackendRuntimeConfig(baseValuesFile)
	Expect(err).NotTo(HaveOccurred(), "解析 aggregate-single 基线后端 runtime 失败")
	cacheRuntime, err := utils.LoadEmbeddedCacheIndexerRuntimeConfig(baseValuesFile)
	Expect(err).NotTo(HaveOccurred(), "解析 aggregate-single 基线 cache-indexer runtime 失败")

	cfg := *infernexCfg
	cfg.ValuesFile = baseValuesFile
	err = utils.HelmUpgradeWithLatestFallback(&cfg)
	Expect(err).NotTo(HaveOccurred(), "恢复 aggregate-single 基线 release 失败")

	waitForAggregateSingleReady(ctx, k8sClient, infernexCfg.Namespace, aggregateRuntime, cacheRuntime)
	return aggregateRuntime, cacheRuntime, baseValuesFile
}

func buildAggregateSingleVariantValues(
	baseValuesFile, variantName string,
	mutate func(map[string]any) error,
) (string, utils.AggregatedBackendRuntimeConfig, utils.CacheIndexerRuntimeConfig, error) {
	data, err := os.ReadFile(baseValuesFile)
	if err != nil {
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, fmt.Errorf("read aggregate-single values failed: %w", err)
	}

	var values map[string]any
	if err := yaml.Unmarshal(data, &values); err != nil {
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, fmt.Errorf("unmarshal aggregate-single values failed: %w", err)
	}

	if mutate != nil {
		if err := mutate(values); err != nil {
			return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
		}
	}

	yamlData, err := yaml.Marshal(values)
	if err != nil {
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, fmt.Errorf("marshal aggregate-single variant values failed: %w", err)
	}

	tempFile := filepath.Join(filepath.Dir(baseValuesFile), fmt.Sprintf("aggregate-single-%s-%d.yaml", variantName, time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, fmt.Errorf("write aggregate-single variant values failed: %w", err)
	}

	aggregateRuntime, err := utils.LoadAggregatedBackendRuntimeConfig(tempFile)
	if err != nil {
		_ = utils.CleanupTempValuesFile(tempFile)
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}
	cacheRuntime, err := utils.LoadEmbeddedCacheIndexerRuntimeConfig(tempFile)
	if err != nil {
		_ = utils.CleanupTempValuesFile(tempFile)
		return "", utils.AggregatedBackendRuntimeConfig{}, utils.CacheIndexerRuntimeConfig{}, err
	}

	return tempFile, aggregateRuntime, cacheRuntime, nil
}

func buildAggregatePromptA() string {
	return strings.Repeat(aggregatedPromptBaseTextA, aggregatedPromptRepeatCount)
}

func buildAggregatePromptAFIT004() string {
	return strings.Repeat(aggregatedPromptBaseTextA+"[fit004-isolated] ", aggregatedPromptRepeatCount)
}

func buildAggregatePromptAFIT009() string {
	return strings.Repeat(aggregatedPromptBaseTextA+"[fit009-isolated] ", aggregatedPromptRepeatCount)
}

func buildAggregatePromptCold() string {
	return strings.Repeat(aggregatedPromptBaseTextCold, aggregatedPromptRepeatCount)
}

func buildAggregatePromptABTokens(promptATokens, promptColdTokens []int32, blockSize int) []int32 {
	prefixTokenCount := aggregatedPromptPrefixBlocks * blockSize
	tailTokenCount := (aggregatedPromptTotalBlocks - aggregatedPromptPrefixBlocks) * blockSize

	promptABTokens := make([]int32, 0, prefixTokenCount+tailTokenCount)
	promptABTokens = append(promptABTokens, promptATokens[:prefixTokenCount]...)
	promptABTokens = append(promptABTokens, promptColdTokens[:tailTokenCount]...)
	return promptABTokens
}

func buildAggregatePromptAMutatedTokens(promptATokens []int32, blockSize int) []int32 {
	prefixTokenCount := aggregatedPromptPrefixBlocks * blockSize
	mutatedTokens := append([]int32(nil), promptATokens[:prefixTokenCount]...)
	mutatedTokens[aggregatedMutatedTokenIndex] = mutatedTokens[aggregatedMutatedTokenIndex] ^ 1
	return mutatedTokens
}
