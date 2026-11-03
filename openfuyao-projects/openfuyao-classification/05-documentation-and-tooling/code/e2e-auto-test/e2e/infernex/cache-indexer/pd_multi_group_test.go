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

var _ = Describe("Cache Indexer PD Multi Group", Ordered, Label("cache-indexer", "npu", "26.06"), func() {
	var (
		ctx                context.Context
		cancel             context.CancelFunc
		infernexValuesFile string
		infernexCfg        *utils.ReleaseConfig
		pdRuntimes         []utils.PDBackendRuntimeConfig
		cacheRuntime       utils.CacheIndexerRuntimeConfig
	)

	BeforeAll(func() {
		ctx, cancel = context.WithTimeout(context.Background(), 60*time.Minute)
		DeferCleanup(cancel)

		infernexCfg = mustHelmRelease("infernex")
		cleanupLeftoverInfernexAndCacheIndexer(ctx)

		By("使用 infernex 默认 values + pd-multi topology values 安装双 PD 组场景（每组 1P1D，tp1）")
		var err error
		infernexValuesFile, pdRuntimes, cacheRuntime, err = preparePDMultiGroupValues()
		Expect(err).NotTo(HaveOccurred(), "构建 pd-multi values 失败")
		DeferCleanup(func() {
			_ = utils.CleanupTempValuesFile(infernexValuesFile)
		})

		cfg := *infernexCfg
		cfg.ValuesFile = infernexValuesFile
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("等待两组 Prefill/Decode 与 Mooncake Master 就绪")
		Expect(pdRuntimes).To(HaveLen(pdMultiGroupBackendCount), "应当加载到 2 组 PD runtime")
		for _, pdRuntime := range pdRuntimes {
			runtime := pdRuntime
			Eventually(func() error {
				_, _, err := utils.GetPDBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, runtime.ServiceName, pdSingleGroupPrefillRole)
				return err
			}, 10*time.Minute, 5*time.Second).Should(Succeed())
			Eventually(func() error {
				_, _, err := utils.GetPDBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, runtime.ServiceName, pdSingleGroupDecodeRole)
				return err
			}, 10*time.Minute, 5*time.Second).Should(Succeed())
		}
		Eventually(func() error {
			return utils.WaitForMooncakeMasterPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 60*time.Second, 5*time.Second).Should(Succeed())
	})

	AfterAll(func() {
		uninstallReleaseIfPresent(infernexCfg, "infernex uninstall success")
	})

	It("cache-indexer_FIT_011-PD双组识别高命中Prefill", Label("P1"), func() {
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		Expect(pdRuntimes).To(HaveLen(pdMultiGroupBackendCount), "应当存在 2 组 PD runtime")

		groupG1 := pdRuntimes[0]
		groupG2 := pdRuntimes[1]

		By("等待 cache-indexer Pod Running 且 Ready=1/1")
		Eventually(func() error {
			return utils.AssertCacheIndexerPodReady(ctx, k8sClient, infernexCfg.Namespace)
		}, 10*time.Minute, 5*time.Second).Should(Succeed())

		By("获取 G1/G2 Prefill Pod 身份，并确认二者都已被 discovery 发现")
		var g1PrefillPodName string
		var g1PrefillPodIP string
		var g2PrefillPodIP string
		Eventually(func() error {
			name, ip, err := utils.GetPDBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, groupG1.ServiceName, pdSingleGroupPrefillRole)
			if err != nil {
				return err
			}
			g1PrefillPodName = name
			g1PrefillPodIP = ip
			return nil
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			_, ip, err := utils.GetPDBackendPodIdentity(ctx, k8sClient, infernexCfg.Namespace, groupG2.ServiceName, pdSingleGroupPrefillRole)
			if err != nil {
				return err
			}
			g2PrefillPodIP = ip
			return nil
		}, 10*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "vLLM pod added", g1PrefillPodIP)
		}, 90*time.Second, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(ctx, k8sClient, infernexCfg.Namespace, cacheIndexerLogContainer, "vLLM pod added", g2PrefillPodIP)
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("通过 G1 Prefill Pod 的 /tokenize 对 PromptA 分词，并截取前 2 个 block 作为查询 token_ids")
		promptA := buildAggregatePromptA()
		var queryTokens []int32
		Eventually(func() error {
			tokens, err := utils.TokenizePromptViaPDBackend(ctx, infernexCfg.Namespace, g1PrefillPodName, groupG1, promptA)
			if err != nil {
				return err
			}
			queryTokenCount := pdSingleGroupQueryBlocks * groupG1.PrefillBlockSize
			if len(tokens) < queryTokenCount {
				return fmt.Errorf("expected PromptA tokens >= %d, got %d", queryTokenCount, len(tokens))
			}
			queryTokens = append([]int32(nil), tokens[:queryTokenCount]...)
			return nil
		}, 2*time.Minute, 5*time.Second).Should(Succeed())

		By("仅通过 G1 PD 组发送一次 PromptA 推理请求并确认成功")
		Eventually(func() error {
			return utils.DoPDBackendInference(ctx, infernexCfg.Namespace, groupG1, promptA)
		}, 3*time.Minute, 5*time.Second).Should(Succeed())

		By("在 60 秒内轮询 /kv-cache/hit-rate，请求顺序为 [G2 Prefill IP, G1 Prefill IP]，直到 G2 L1=0 且 G1 L1=1")
		var latestHitRateBody []byte
		Eventually(func() error {
			scores, body, err := utils.QueryCacheIndexerHitRateForServerIPs(
				ctx,
				infernexCfg.Namespace,
				[]string{g2PrefillPodIP, g1PrefillPodIP},
				queryTokens,
				cacheRuntime,
				groupG1.PrefillBlockSize,
			)
			if err != nil {
				return err
			}
			latestHitRateBody = body
			if scores[0].L1HitRatio != 0.0 {
				return fmt.Errorf("expected G2 L1=0, got l1=%v l3=%v body=%s", scores[0].L1HitRatio, scores[0].L3HitRatio, string(body))
			}
			if scores[1].L1HitRatio != 1.0 {
				return fmt.Errorf("expected G1 L1=1, got l1=%v l3=%v body=%s", scores[1].L1HitRatio, scores[1].L3HitRatio, string(body))
			}
			return nil
		}, 60*time.Second, 5*time.Second).Should(Succeed())

		GinkgoWriter.Printf("FIT_011 latest hit-rate response: %s\n", string(latestHitRateBody))
	})
})

func preparePDMultiGroupValues() (string, []utils.PDBackendRuntimeConfig, utils.CacheIndexerRuntimeConfig, error) {
	configDir, err := utils.GetCacheIndexerConfigDir()
	if err != nil {
		return "", nil, utils.CacheIndexerRuntimeConfig{}, err
	}

	infernexBasePath := filepath.Join(configDir, "values-infernex.yaml")
	infernexTopologyPath := filepath.Join(configDir, "topology", "pd-multi.yaml")

	valuesFile, err := utils.MergeValuesFiles(infernexBasePath, infernexTopologyPath)
	if err != nil {
		return "", nil, utils.CacheIndexerRuntimeConfig{}, err
	}

	pdRuntimes, err := utils.LoadPDBackendRuntimeConfigs(valuesFile)
	if err != nil {
		_ = utils.CleanupTempValuesFile(valuesFile)
		return "", nil, utils.CacheIndexerRuntimeConfig{}, err
	}
	cacheRuntime, err := utils.LoadEmbeddedCacheIndexerRuntimeConfig(valuesFile)
	if err != nil {
		_ = utils.CleanupTempValuesFile(valuesFile)
		return "", nil, utils.CacheIndexerRuntimeConfig{}, err
	}

	return valuesFile, pdRuntimes, cacheRuntime, nil
}
