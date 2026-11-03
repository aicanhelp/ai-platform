package cacheindexer

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	frameworkk8s "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("Cache Indexer Standalone", Ordered, Label("cache-indexer", "npu", "26.06"), func() {
	var (
		ctx                 context.Context
		cancel              context.CancelFunc
		basePath            string
		topologyPath        string
		defaultValuesFile   string
		generatedValuesFile []string
		runtimeCfg          utils.CacheIndexerRuntimeConfig
		cacheIndexerCfg     *utils.ReleaseConfig
	)

	BeforeAll(func() {
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		DeferCleanup(cancel)

		cacheIndexerCfg = mustHelmRelease("cache-indexer")
		cleanupLeftoverInfernexAndCacheIndexer(ctx)

		configDir, err := utils.GetCacheIndexerConfigDir()
		Expect(err).NotTo(HaveOccurred())
		basePath = filepath.Join(configDir, "values-cache-indexer.yaml")
		topologyPath = filepath.Join(configDir, "topology", "standalone.yaml")

		By("使用 cache-indexer 默认 values + standalone topology values 安装独立 cache-indexer chart")
		defaultValuesFile, err = utils.MergeValuesFiles(basePath, topologyPath)
		Expect(err).NotTo(HaveOccurred(), "构建 standalone values 失败")
		DeferCleanup(func() {
			_ = utils.CleanupTempValuesFile(defaultValuesFile)
			for _, generatedFile := range generatedValuesFile {
				_ = utils.CleanupTempValuesFile(generatedFile)
			}
		})

		cfg := *cacheIndexerCfg
		cfg.ValuesFile = defaultValuesFile
		runtimeCfg, err = utils.LoadStandaloneCacheIndexerRuntimeConfig(defaultValuesFile)
		Expect(err).NotTo(HaveOccurred(), "解析 cache-indexer runtime 配置失败")
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install cache-indexer 执行失败")
	})

	AfterAll(func() {
		uninstallReleaseIfPresent(cacheIndexerCfg, "cache-indexer uninstall success")
	})

	// 独立部署默认配置启动成功。
	// @date 2024-06-26 @status: 已验证，后续可根据需要增加功能性测试用例。
	It("cache-indexer_FIT_001-独立部署默认配置启动成功", Label("P0"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("在 60 秒内校验 cache-indexer Pod Running 且 Ready=1/1")
		Eventually(func() error {
			return utils.AssertCacheIndexerPodReady(ctx, k8sClient, cacheIndexerCfg.Namespace)
		}, 60*time.Second, 5*time.Second).Should(Succeed())

		By("通过 Service 端口转发访问 /healthz 和 /readyz")
		Eventually(func() error {
			return utils.AssertCacheIndexerHealthEndpoints(ctx, cacheIndexerCfg.Namespace, runtimeCfg)
		}, 30*time.Second, 3*time.Second).Should(Succeed())

		By("采集 cache-indexer Pod info 日志并校验关键字")
		Eventually(func() error {
			return utils.AssertCacheIndexerStartupLogs(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer)
		}, 30*time.Second, 5*time.Second).Should(Succeed())

		utils.DumpPodsAndServices(ctx, k8sClient, cacheIndexerCfg.Namespace)
	})

	// server_ip 为空返回 400。
	It("cache-indexer_FIT_013-server_ip为空返回400", Label("P1"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("调用 POST /kv-cache/hit-rate，请求体 server_ip 为空数组")
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			[]string{},
			utils.BuildSequentialTokenIDs(cacheIndexerRequestBlockSize),
			cacheIndexerRequestBlockSize,
			"",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg.ServiceName,
			runtimeCfg.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")

		By("校验返回 HTTP 400 且 message=server_ip is required")
		Expect(statusCode).To(Equal(http.StatusBadRequest))
		resp, err := utils.ParseCacheIndexerHitRateResponse(body)
		Expect(err).NotTo(HaveOccurred(), "解析 /kv-cache/hit-rate 响应失败")
		Expect(resp.Status).To(Equal(http.StatusBadRequest))
		Expect(resp.Message).To(Equal("server_ip is required"))

		By("校验日志包含 hit-rate request rejected: server_ip is required")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer, "hit-rate request rejected: server_ip is required")
		}, 15*time.Second, 2*time.Second).Should(Succeed())
	})

	// 命中率响应日志可回溯。
	It("cache-indexer_FIT_033-命中率响应日志可回溯", Label("P2"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("记录当前 cache-indexer Pod 与日志基线")
		podSnapshot, err := utils.SnapshotCacheIndexerPod(ctx, k8sClient, cacheIndexerCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "采集 cache-indexer Pod 快照失败")
		logBefore, err := utils.GetCacheIndexerLogText(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer)
		Expect(err).NotTo(HaveOccurred(), "采集 cache-indexer 日志失败")

		By("发送一个合法且可预测的零命中请求")
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			[]string{standaloneUnknownServerIP},
			utils.BuildSequentialTokenIDs(cacheIndexerRequestBlockSize),
			cacheIndexerRequestBlockSize,
			"",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg.ServiceName,
			runtimeCfg.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")

		resp, err := utils.ParseCacheIndexerHitRateResponse(body)
		Expect(err).NotTo(HaveOccurred(), "解析 /kv-cache/hit-rate 响应失败")
		Expect(resp.Status).To(Equal(0))
		Expect(resp.ServerScoreList).To(HaveLen(1))
		Expect(resp.ServerScoreList[0].ServerIP).To(Equal(standaloneUnknownServerIP))
		Expect(resp.ServerScoreList[0].L1HitRatio).To(Equal(0.0))
		Expect(resp.ServerScoreList[0].L3HitRatio).To(Equal(0.0))
		Expect(statusCode).To(Equal(http.StatusOK))

		By("校验最新 hit-rate request completed 日志中的 responseBody 与接口响应一致")
		Eventually(func() error {
			logAfter, err := utils.GetCacheIndexerLogText(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer)
			if err != nil {
				return err
			}
			loggedResp, err := utils.ParseLatestLoggedHitRateResponse(utils.NewLogSegment(logBefore, logAfter))
			if err != nil {
				return err
			}
			return utils.CompareHitRateResponses(resp, loggedResp)
		}, 15*time.Second, 2*time.Second).Should(Succeed())

		By("校验 cache-indexer Pod 未重建或重启")
		Expect(utils.AssertCacheIndexerPodUnchanged(ctx, k8sClient, cacheIndexerCfg.Namespace, podSnapshot)).To(Succeed())
	})

	// GET 命中率接口返回 405。
	It("cache-indexer_FIT_034-GET命中率接口返回405", Label("P3"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("记录当前 cache-indexer Pod 快照")
		podSnapshot, err := utils.SnapshotCacheIndexerPod(ctx, k8sClient, cacheIndexerCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "采集 cache-indexer Pod 快照失败")

		By("使用 GET 访问 /kv-cache/hit-rate")
		statusCode, body, err := doRawCacheIndexerRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg,
			http.MethodGet,
			"",
			nil,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 GET /kv-cache/hit-rate 失败")
		Expect(assertHitRateErrorResponse(statusCode, body, http.StatusMethodNotAllowed, "method not allowed")).To(Succeed())

		By("校验 cache-indexer Pod 未重建或重启")
		Expect(utils.AssertCacheIndexerPodUnchanged(ctx, k8sClient, cacheIndexerCfg.Namespace, podSnapshot)).To(Succeed())
	})

	// 空请求体返回 400。
	It("cache-indexer_FIT_035-空请求体返回400", Label("P3"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("记录当前 cache-indexer Pod 快照")
		podSnapshot, err := utils.SnapshotCacheIndexerPod(ctx, k8sClient, cacheIndexerCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "采集 cache-indexer Pod 快照失败")

		By("POST /kv-cache/hit-rate 且请求体为空")
		statusCode, body, err := doRawCacheIndexerRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg,
			http.MethodPost,
			"application/json",
			nil,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 POST /kv-cache/hit-rate 失败")
		Expect(assertHitRateErrorResponse(statusCode, body, http.StatusBadRequest, "empty request body")).To(Succeed())

		By("校验日志包含 empty body")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer, "hit-rate request rejected: empty body")
		}, 15*time.Second, 2*time.Second).Should(Succeed())

		By("校验 cache-indexer Pod 未重建或重启")
		Expect(utils.AssertCacheIndexerPodUnchanged(ctx, k8sClient, cacheIndexerCfg.Namespace, podSnapshot)).To(Succeed())
	})

	// 非法 JSON 返回 400。
	It("cache-indexer_FIT_036-非法JSON返回400", Label("P3"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("记录当前 cache-indexer Pod 快照")
		podSnapshot, err := utils.SnapshotCacheIndexerPod(ctx, k8sClient, cacheIndexerCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "采集 cache-indexer Pod 快照失败")

		By("POST /kv-cache/hit-rate 发送非法 JSON")
		statusCode, body, err := doRawCacheIndexerRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg,
			http.MethodPost,
			"application/json",
			[]byte("{"),
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 POST /kv-cache/hit-rate 失败")
		Expect(assertHitRateErrorResponseContains(statusCode, body, http.StatusBadRequest, "invalid JSON:")).To(Succeed())

		By("校验日志包含 invalid JSON")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer, "hit-rate request rejected: invalid JSON")
		}, 15*time.Second, 2*time.Second).Should(Succeed())

		By("校验 cache-indexer Pod 未重建或重启")
		Expect(utils.AssertCacheIndexerPodUnchanged(ctx, k8sClient, cacheIndexerCfg.Namespace, podSnapshot)).To(Succeed())
	})

	// 未知字段返回 400。
	It("cache-indexer_FIT_037-未知字段返回400", Label("P3"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("记录当前 cache-indexer Pod 快照")
		podSnapshot, err := utils.SnapshotCacheIndexerPod(ctx, k8sClient, cacheIndexerCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "采集 cache-indexer Pod 快照失败")

		By("POST /kv-cache/hit-rate 发送带 unknown_field 的请求体")
		statusCode, body, err := doRawCacheIndexerRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg,
			http.MethodPost,
			"application/json",
			[]byte(`{"server_ip":["192.0.2.1"],"body":{"token_ids":[1,2,3,4],"block_size":128},"unknown_field":"unexpected"}`),
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 POST /kv-cache/hit-rate 失败")
		Expect(assertHitRateErrorResponseContains(statusCode, body, http.StatusBadRequest, `unknown field "unknown_field"`)).To(Succeed())

		By("校验日志包含 invalid JSON")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer, "hit-rate request rejected: invalid JSON")
		}, 15*time.Second, 2*time.Second).Should(Succeed())

		By("校验 cache-indexer Pod 未重建或重启")
		Expect(utils.AssertCacheIndexerPodUnchanged(ctx, k8sClient, cacheIndexerCfg.Namespace, podSnapshot)).To(Succeed())
	})

	// blockSize 为 0 返回 400。
	It("cache-indexer_FIT_038-blockSize为0返回400", Label("P3"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("记录当前 cache-indexer Pod 快照")
		podSnapshot, err := utils.SnapshotCacheIndexerPod(ctx, k8sClient, cacheIndexerCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "采集 cache-indexer Pod 快照失败")

		By("POST /kv-cache/hit-rate 且 body.block_size=0")
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			[]string{standaloneUnknownServerIP},
			utils.BuildSequentialTokenIDs(cacheIndexerRequestBlockSize),
			0,
			"",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg.ServiceName,
			runtimeCfg.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")
		Expect(assertHitRateErrorResponse(statusCode, body, http.StatusBadRequest, "body.block_size must be > 0")).To(Succeed())

		By("校验日志包含 body.block_size must be > 0")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer, "hit-rate request rejected: body.block_size must be > 0")
		}, 15*time.Second, 2*time.Second).Should(Succeed())

		By("校验 cache-indexer Pod 未重建或重启")
		Expect(utils.AssertCacheIndexerPodUnchanged(ctx, k8sClient, cacheIndexerCfg.Namespace, podSnapshot)).To(Succeed())
	})

	// blockSize 为负数返回 400。
	It("cache-indexer_FIT_039-blockSize为负数返回400", Label("P3"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("记录当前 cache-indexer Pod 快照")
		podSnapshot, err := utils.SnapshotCacheIndexerPod(ctx, k8sClient, cacheIndexerCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "采集 cache-indexer Pod 快照失败")

		By("POST /kv-cache/hit-rate 且 body.block_size=-1")
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			[]string{standaloneUnknownServerIP},
			utils.BuildSequentialTokenIDs(cacheIndexerRequestBlockSize),
			-1,
			"",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg.ServiceName,
			runtimeCfg.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")
		Expect(assertHitRateErrorResponse(statusCode, body, http.StatusBadRequest, "body.block_size must be > 0")).To(Succeed())

		By("校验日志包含 body.block_size must be > 0")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer, "hit-rate request rejected: body.block_size must be > 0")
		}, 15*time.Second, 2*time.Second).Should(Succeed())

		By("校验 cache-indexer Pod 未重建或重启")
		Expect(utils.AssertCacheIndexerPodUnchanged(ctx, k8sClient, cacheIndexerCfg.Namespace, podSnapshot)).To(Succeed())
	})

	// 重复 serverIP 保持重复响应。
	It("cache-indexer_FIT_042-重复serverIP保持重复响应", Label("P3"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("记录当前 cache-indexer Pod 快照")
		podSnapshot, err := utils.SnapshotCacheIndexerPod(ctx, k8sClient, cacheIndexerCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "采集 cache-indexer Pod 快照失败")

		By("调用 POST /kv-cache/hit-rate，请求 server_ip 中包含重复 IP")
		repeatedIPs := []string{standaloneUnknownServerIP, standaloneUnknownServerIP}
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			repeatedIPs,
			utils.BuildSequentialTokenIDs(cacheIndexerRequestBlockSize),
			cacheIndexerRequestBlockSize,
			"",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg.ServiceName,
			runtimeCfg.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")
		Expect(utils.AssertRepeatedServerIPsZeroHitResponse(statusCode, body, repeatedIPs)).To(Succeed())

		By("校验 cache-indexer Pod 未重建或重启")
		Expect(utils.AssertCacheIndexerPodUnchanged(ctx, k8sClient, cacheIndexerCfg.Namespace, podSnapshot)).To(Succeed())
	})

	// 不存在 serverIP 返回零。
	It("cache-indexer_FIT_043-不存在serverIP返回零", Label("P3"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("记录当前 cache-indexer Pod 快照")
		podSnapshot, err := utils.SnapshotCacheIndexerPod(ctx, k8sClient, cacheIndexerCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "采集 cache-indexer Pod 快照失败")

		By("调用 POST /kv-cache/hit-rate，请求集群中不存在的 Pod IP")
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			[]string{standaloneAnotherUnknownServerIP},
			utils.BuildSequentialTokenIDs(cacheIndexerRequestBlockSize),
			cacheIndexerRequestBlockSize,
			"",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg.ServiceName,
			runtimeCfg.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")
		Expect(utils.AssertSingleZeroHitResponse(statusCode, body, standaloneAnotherUnknownServerIP)).To(Succeed())

		By("校验 cache-indexer Pod 未重建或重启")
		Expect(utils.AssertCacheIndexerPodUnchanged(ctx, k8sClient, cacheIndexerCfg.Namespace, podSnapshot)).To(Succeed())
	})

	// RBAC 缺少 listPods 日志可诊断。
	It("cache-indexer_FIT_047-RBAC缺少listPods日志可诊断", Label("P3"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("读取当前 Role 作为恢复基线")
		roleName := cacheIndexerCfg.ReleaseName
		originalRole, err := k8sClient.Clientset.RbacV1().Roles(cacheIndexerCfg.Namespace).Get(ctx, roleName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "读取 cache-indexer Role 失败")
		DeferCleanup(func() {
			_ = restoreRoleRules(ctx, k8sClient, cacheIndexerCfg.Namespace, roleName, originalRole.Rules)
		})

		By("去掉 pods 资源的 list 权限")
		err = removePodsListVerb(ctx, k8sClient, cacheIndexerCfg.Namespace, roleName)
		Expect(err).NotTo(HaveOccurred(), "移除 pods list 权限失败")

		logBefore, err := utils.GetCacheIndexerLogText(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer)
		Expect(err).NotTo(HaveOccurred(), "采集 cache-indexer 日志失败")

		By("等待 discovery list pods failed 日志出现")
		Eventually(func() error {
			logAfter, err := utils.GetCacheIndexerLogText(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer)
			if err != nil {
				return err
			}
			if !strings.Contains(utils.NewLogSegment(logBefore, logAfter), "discovery list pods failed") {
				return fmt.Errorf("新增日志中未出现 discovery list pods failed")
			}
			return nil
		}, 35*time.Second, 2*time.Second).Should(Succeed())

		By("恢复 Role 规则")
		err = restoreRoleRules(ctx, k8sClient, cacheIndexerCfg.Namespace, roleName, originalRole.Rules)
		Expect(err).NotTo(HaveOccurred(), "恢复 cache-indexer Role 失败")

		restoreLogBase, err := utils.GetCacheIndexerLogText(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer)
		Expect(err).NotTo(HaveOccurred(), "采集恢复后的日志基线失败")

		By("等待一个 discovery 周期，确认恢复后不再追加 list pods failed")
		Consistently(func() error {
			logAfter, err := utils.GetCacheIndexerLogText(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer)
			if err != nil {
				return err
			}
			if strings.Contains(utils.NewLogSegment(restoreLogBase, logAfter), "discovery list pods failed") {
				return fmt.Errorf("恢复 RBAC 后仍持续出现 discovery list pods failed")
			}
			return nil
		}, 12*time.Second, 3*time.Second).Should(Succeed())

		By("恢复后再次访问健康检查，确认服务仍可用")
		Expect(utils.AssertCacheIndexerHealthEndpoints(ctx, cacheIndexerCfg.Namespace, runtimeCfg)).To(Succeed())
	})

	// 自定义 targetPort 健康检查成功。
	It("cache-indexer_FIT_014-自定义targetPort健康检查成功", Label("P2"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("使用 service.targetPort=28081 重装 cache-indexer")
		runtimeCfg = reinstallStandaloneWithVariant(
			ctx,
			k8sClient,
			cacheIndexerCfg,
			basePath,
			topologyPath,
			&generatedValuesFile,
			"target-port",
			func(values map[string]any) error {
				serviceValues, err := utils.NestedMap(values, "service")
				if err != nil {
					return err
				}
				serviceValues["targetPort"] = standaloneCustomTargetPort
				return nil
			},
		)

		By("校验 Pod Ready 且通过 Service 健康检查")
		waitForStandaloneReady(ctx, k8sClient, cacheIndexerCfg.Namespace, runtimeCfg)

		By("校验日志中的 server started addr 为 :28081")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer, "server started", ":28081")
		}, 20*time.Second, 2*time.Second).Should(Succeed())
	})

	// maxHitRateBodyBytes 限制大请求。
	It("cache-indexer_FIT_016-maxHitRateBodyBytes限制大请求", Label("P2"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("使用较小的 maxHitRateBodyBytes 重装 cache-indexer")
		runtimeCfg = reinstallStandaloneWithVariant(
			ctx,
			k8sClient,
			cacheIndexerCfg,
			basePath,
			topologyPath,
			&generatedValuesFile,
			"body-limit",
			func(values map[string]any) error {
				httpValues, err := utils.NestedMap(values, "http")
				if err != nil {
					return err
				}
				httpValues["maxHitRateBodyBytes"] = standaloneTinyHitRateBodyBytes
				return nil
			},
		)

		By("校验 Pod Ready 且通过 Service 健康检查")
		waitForStandaloneReady(ctx, k8sClient, cacheIndexerCfg.Namespace, runtimeCfg)

		By("发送超限 POST 请求体")
		oversizedReq := utils.BuildCacheIndexerHitRateRequest(
			[]string{standaloneUnknownServerIP},
			utils.BuildSequentialTokenIDs(cacheIndexerRequestBlockSize*32),
			cacheIndexerRequestBlockSize,
			"",
		)
		oversizedBody, err := json.Marshal(oversizedReq)
		Expect(err).NotTo(HaveOccurred(), "序列化超限请求体失败")
		Expect(len(oversizedBody)).To(BeNumerically(">", standaloneTinyHitRateBodyBytes))

		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg.ServiceName,
			runtimeCfg.ServicePort,
			oversizedReq,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")
		Expect(assertHitRateErrorResponse(statusCode, body, http.StatusRequestEntityTooLarge, "request body too large")).To(Succeed())

		By("校验日志包含 body too large")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContains(ctx, k8sClient, cacheIndexerCfg.Namespace, cacheIndexerLogContainer, "hit-rate request rejected: body too large")
		}, 15*time.Second, 2*time.Second).Should(Succeed())
	})

	// 非法 hashAlgo 日志可诊断。
	It("cache-indexer_FIT_048-非法hashAlgo日志可诊断", Label("P3"), func() {
		Expect(cacheIndexerCfg).NotTo(BeNil(), "Helm 配置中不包含 cache-indexer")

		By("使用非法 prefixCachingHashAlgo 重装 cache-indexer")
		runtimeCfg = reinstallStandaloneWithVariant(
			ctx,
			k8sClient,
			cacheIndexerCfg,
			basePath,
			topologyPath,
			&generatedValuesFile,
			"invalid-hash-algo",
			func(values map[string]any) error {
				blockKeyValues, err := utils.NestedMap(values, "blockKey")
				if err != nil {
					return err
				}
				blockKeyValues["prefixCachingHashAlgo"] = standaloneUnsupportedHashAlgo
				return nil
			},
		)

		By("校验 Pod Ready 且通过 Service 健康检查")
		waitForStandaloneReady(ctx, k8sClient, cacheIndexerCfg.Namespace, runtimeCfg)

		By("校验日志包含 builder disabled 与非法 hash algo")
		Eventually(func() error {
			return utils.AssertCacheIndexerLogContainsLineWithAll(
				ctx,
				k8sClient,
				cacheIndexerCfg.Namespace,
				cacheIndexerLogContainer,
				"builder disabled; L1 hit rate stays 0",
				"unsupported prefix caching hash algo",
			)
		}, 20*time.Second, 2*time.Second).Should(Succeed())

		By("调用 /kv-cache/hit-rate，确认查询仍成功且命中率为 0.0/0.0")
		reqBody := utils.BuildCacheIndexerHitRateRequest(
			[]string{standaloneUnknownServerIP},
			utils.BuildSequentialTokenIDs(cacheIndexerRequestBlockSize),
			cacheIndexerRequestBlockSize,
			"",
		)
		statusCode, body, err := utils.DoCacheIndexerHitRateRequest(
			ctx,
			cacheIndexerCfg.Namespace,
			runtimeCfg.ServiceName,
			runtimeCfg.ServicePort,
			reqBody,
			10*time.Second,
		)
		Expect(err).NotTo(HaveOccurred(), "调用 /kv-cache/hit-rate 失败")
		Expect(utils.AssertSingleZeroHitResponse(statusCode, body, standaloneUnknownServerIP)).To(Succeed())
	})
})

const (
	standaloneUnknownServerIP        = "192.0.2.1"
	standaloneAnotherUnknownServerIP = "192.0.2.2"
	standaloneCustomTargetPort       = 28081
	standaloneTinyHitRateBodyBytes   = 128
	standaloneUnsupportedHashAlgo    = "unsupported_algo"
)

type cacheIndexerPodSnapshot struct {
	PodName      string
	RestartCount int32
}

func waitForStandaloneReady(
	ctx context.Context,
	k8sClient *frameworkk8s.K8SClient,
	namespace string,
	runtimeCfg utils.CacheIndexerRuntimeConfig,
) {
	Eventually(func() error {
		return utils.AssertCacheIndexerPodReady(ctx, k8sClient, namespace)
	}, 90*time.Second, 5*time.Second).Should(Succeed())
	Eventually(func() error {
		return utils.AssertCacheIndexerHealthEndpoints(ctx, namespace, runtimeCfg)
	}, 30*time.Second, 3*time.Second).Should(Succeed())
}

func reinstallStandaloneWithVariant(
	ctx context.Context,
	k8sClient *frameworkk8s.K8SClient,
	cacheIndexerCfg *utils.ReleaseConfig,
	basePath, topologyPath string,
	generatedFiles *[]string,
	variantName string,
	mutate func(map[string]any) error,
) utils.CacheIndexerRuntimeConfig {
	valuesFile, err := buildStandaloneValuesVariant(basePath, topologyPath, variantName, mutate)
	Expect(err).NotTo(HaveOccurred(), "构建 %s values 失败", variantName)
	*generatedFiles = append(*generatedFiles, valuesFile)

	runtimeCfg, err := reinstallStandaloneRelease(cacheIndexerCfg, valuesFile)
	Expect(err).NotTo(HaveOccurred(), "重装 %s cache-indexer 失败", variantName)

	waitForStandaloneReady(ctx, k8sClient, cacheIndexerCfg.Namespace, runtimeCfg)
	return runtimeCfg
}

func buildStandaloneValuesVariant(
	basePath, topologyPath, variantName string,
	mutate func(map[string]any) error,
) (string, error) {
	mergedFile, err := utils.MergeValuesFiles(basePath, topologyPath)
	if err != nil {
		return "", err
	}
	defer func() {
		_ = utils.CleanupTempValuesFile(mergedFile)
	}()
	return utils.BuildValuesVariantFromFile(mergedFile, "standalone-"+variantName, mutate)
}

func reinstallStandaloneRelease(
	cacheIndexerCfg *utils.ReleaseConfig,
	valuesFile string,
) (utils.CacheIndexerRuntimeConfig, error) {
	runtimeCfg, err := utils.LoadStandaloneCacheIndexerRuntimeConfig(valuesFile)
	if err != nil {
		return utils.CacheIndexerRuntimeConfig{}, err
	}

	cfg := *cacheIndexerCfg
	cfg.ValuesFile = valuesFile
	if err := utils.HelmUninstallIfExists(cfg.ReleaseName, cfg.Namespace); err != nil {
		return utils.CacheIndexerRuntimeConfig{}, err
	}
	if err := utils.HelmInstallWithLatestFallback(&cfg); err != nil {
		return utils.CacheIndexerRuntimeConfig{}, err
	}
	return runtimeCfg, nil
}

func doRawCacheIndexerRequest(
	ctx context.Context,
	namespace string,
	runtimeCfg utils.CacheIndexerRuntimeConfig,
	method, contentType string,
	body []byte,
	timeout time.Duration,
) (int, []byte, error) {
	return utils.DoServiceRequestViaPortForward(
		ctx,
		namespace,
		runtimeCfg.ServiceName,
		runtimeCfg.ServicePort,
		method,
		utils.CacheIndexerHitRatePath,
		contentType,
		body,
		timeout,
	)
}

func assertHitRateErrorResponse(statusCode int, body []byte, expectedStatus int, expectedMessage string) error {
	if statusCode != expectedStatus {
		return fmt.Errorf("unexpected http status=%d body=%s", statusCode, string(body))
	}
	resp, err := utils.ParseCacheIndexerHitRateResponse(body)
	if err != nil {
		return err
	}
	if resp.Status != expectedStatus {
		return fmt.Errorf("unexpected response status=%d body=%s", resp.Status, string(body))
	}
	if resp.Message != expectedMessage {
		return fmt.Errorf("unexpected response message=%q want=%q body=%s", resp.Message, expectedMessage, string(body))
	}
	return nil
}

func assertHitRateErrorResponseContains(statusCode int, body []byte, expectedStatus int, expectedSubstring string) error {
	if statusCode != expectedStatus {
		return fmt.Errorf("unexpected http status=%d body=%s", statusCode, string(body))
	}
	resp, err := utils.ParseCacheIndexerHitRateResponse(body)
	if err != nil {
		return err
	}
	if resp.Status != expectedStatus {
		return fmt.Errorf("unexpected response status=%d body=%s", resp.Status, string(body))
	}
	if !strings.Contains(resp.Message, expectedSubstring) {
		return fmt.Errorf("response message=%q does not contain %q", resp.Message, expectedSubstring)
	}
	return nil
}

func removePodsListVerb(
	ctx context.Context,
	k8sClient *frameworkk8s.K8SClient,
	namespace, roleName string,
) error {
	role, err := k8sClient.Clientset.RbacV1().Roles(namespace).Get(ctx, roleName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get role %s/%s failed: %w", namespace, roleName, err)
	}

	removed := false
	updated := role.DeepCopy()
	for idx := range updated.Rules {
		rule := &updated.Rules[idx]
		if !containsString(rule.Resources, "pods") {
			continue
		}
		nextVerbs := make([]string, 0, len(rule.Verbs))
		for _, verb := range rule.Verbs {
			if verb == "list" {
				removed = true
				continue
			}
			nextVerbs = append(nextVerbs, verb)
		}
		rule.Verbs = nextVerbs
	}
	if !removed {
		return fmt.Errorf("role %s/%s does not contain pods list verb", namespace, roleName)
	}

	_, err = k8sClient.Clientset.RbacV1().Roles(namespace).Update(ctx, updated, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("update role %s/%s failed: %w", namespace, roleName, err)
	}
	return nil
}

func restoreRoleRules(
	ctx context.Context,
	k8sClient *frameworkk8s.K8SClient,
	namespace, roleName string,
	rules []rbacv1.PolicyRule,
) error {
	role, err := k8sClient.Clientset.RbacV1().Roles(namespace).Get(ctx, roleName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get role %s/%s failed: %w", namespace, roleName, err)
	}
	role = role.DeepCopy()
	role.Rules = append([]rbacv1.PolicyRule(nil), rules...)
	_, err = k8sClient.Clientset.RbacV1().Roles(namespace).Update(ctx, role, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("restore role %s/%s failed: %w", namespace, roleName, err)
	}
	return nil
}

func containsString(items []string, target string) bool {
	for _, item := range items {
		if item == target {
			return true
		}
	}
	return false
}
