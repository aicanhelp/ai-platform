package hermesrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

var _ = Describe("Hermes Router disaster aggregate test", Label("hermes-router-disaster-aggregate", "npu", "skip-temporarily"), Ordered, func() {
	var (
		k8sClient  *k8s.K8SClient
		ctx        context.Context
		cancel     context.CancelFunc
		helmConfig map[string]*utils.ReleaseConfig
	)

	BeforeAll(func() {
		ctx, cancel = context.WithTimeout(context.Background(), 60*time.Minute)
		DeferCleanup(cancel)

		var err error
		k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
		Expect(err).NotTo(HaveOccurred(), "K8s 客户端创建失败")

		helmConfig, err = utils.LoadHelmConfig()
		Expect(err).NotTo(HaveOccurred(), "Helm 配置加载失败")

		infernexCfg := helmConfig["infernex"]
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中应当包含 infernex")
		err = k8sClient.CreateNamespaceIfNotExists(ctx, infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "命名空间创建失败")
	})

	AfterAll(func() {
		infernexCfg := helmConfig["infernex"]
		if infernexCfg == nil {
			return
		}
		needUninstall, err := utils.CheckHelmRelease(infernexCfg)
		if err != nil {
			GinkgoWriter.Println(err)
			return
		}
		if needUninstall {
			err = utils.HelmUninstall(infernexCfg)
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "Helm uninstall infernex 执行失败")
		}
	})

	waitNamespacePodsReady := func(ns string, timeout time.Duration) {
		Eventually(func() bool {
			pods, err := k8sClient.ListPods(ctx, ns, metav1.ListOptions{})
			if err != nil || len(pods.Items) == 0 {
				return false
			}
			for _, pod := range pods.Items {
				if pod.Status.Phase != "Running" {
					return false
				}
				for _, cs := range pod.Status.ContainerStatuses {
					if !cs.Ready {
						return false
					}
				}
			}
			return true
		}, timeout, 10*time.Second).Should(BeTrue(), fmt.Sprintf("命名空间 %s 的 Pod 未全部 Running/Ready", ns))
	}

	runShell := func(cmd string) (string, error) {
		c := exec.Command("bash", "-lc", cmd)
		out, err := c.CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("执行命令失败: %s, err: %w, output: %s", cmd, err, string(out))
		}
		return strings.TrimSpace(string(out)), nil
	}

	getEPPPodName := func(ns string) (string, error) {
		pods, err := k8sClient.ListPods(ctx, ns, metav1.ListOptions{})
		if err != nil {
			return "", err
		}
		for _, p := range pods.Items {
			if p.DeletionTimestamp != nil {
				continue
			}
			if strings.Contains(p.Name, "epp") && p.Status.Phase == "Running" {
				return p.Name, nil
			}
		}
		return "", fmt.Errorf("未找到 epp pod, namespace=%s", ns)
	}

	validatePodCountFromEPPLogs := func(ns string, expect int, startLine int) (int, error) {
		eppPod, err := getEPPPodName(ns)
		if err != nil {
			return 0, err
		}
		logs, err := runShell(fmt.Sprintf("kubectl -n %s logs %s -c epp", ns, eppPod))
		if err != nil {
			return 0, err
		}
		lines := strings.Split(logs, "\n")
		totalLines := len(lines)
		if startLine < 1 {
			startLine = 1
		}
		if startLine > totalLines {
			return totalLines, fmt.Errorf("日志总行数=%d，尚无从第 %d 行开始的新日志", totalLines, startLine)
		}

		for i := startLine - 1; i < totalLines; i++ {
			line := strings.TrimSpace(lines[i])
			if line == "" || !strings.Contains(line, "podCount") {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(line), &obj); err != nil {
				continue
			}
			raw, ok := obj["podCount"]
			if !ok {
				continue
			}
			switch v := raw.(type) {
			case float64:
				if int(v) == expect {
					return totalLines, nil
				}
			case int:
				if v == expect {
					return totalLines, nil
				}
			case string:
				n, convErr := strconv.Atoi(v)
				if convErr == nil && n == expect {
					return totalLines, nil
				}
			}
		}
		return totalLines, fmt.Errorf("在日志第 %d-%d 行中未匹配到 podCount=%d", startLine, totalLines, expect)
	}

	findSelectedNameInJSON := func(data []byte) (string, bool) {
		var root any
		if err := json.Unmarshal(data, &root); err != nil {
			return "", false
		}
		var walk func(any) (string, bool)
		walk = func(v any) (string, bool) {
			switch x := v.(type) {
			case map[string]any:
				for k, val := range x {
					if strings.EqualFold(k, "SelectedName") {
						if s, ok := val.(string); ok && strings.TrimSpace(s) != "" {
							return strings.TrimSpace(s), true
						}
					}
					if s, ok := walk(val); ok {
						return s, true
					}
				}
			case []any:
				for _, it := range x {
					if s, ok := walk(it); ok {
						return s, true
					}
				}
			}
			return "", false
		}
		return walk(root)
	}

	// extractSelectedBackendFromEPPLogs 从 EPP Pod 日志中解析指派后端名称（字段 SelectedName / selectedName，支持行首前缀与嵌套 JSON）
	extractSelectedBackendFromEPPLogs := func(ns string) (string, error) {
		eppPod, err := getEPPPodName(ns)
		if err != nil {
			return "", err
		}
		logs, err := runShell(fmt.Sprintf("kubectl -n %s logs %s -c epp --tail=3000", ns, eppPod))
		if err != nil {
			return "", err
		}
		lines := strings.Split(logs, "\n")
		for i := len(lines) - 1; i >= 0; i-- {
			line := strings.TrimSpace(lines[i])
			if line == "" || !strings.Contains(line, "{") {
				continue
			}
			payload := line
			if j := strings.Index(line, "{"); j > 0 {
				payload = line[j:]
			}
			if s, ok := findSelectedNameInJSON([]byte(payload)); ok {
				return s, nil
			}
		}
		return "", fmt.Errorf("未在 EPP 日志中找到 SelectedName 字段（已尝试嵌套与大小写不敏感匹配）")
	}

	doInferenceWithTimeout := func(gatewayURL string, reqBody map[string]any, timeout time.Duration) (int, []byte, error) {
		bodyBytes, err := json.Marshal(reqBody)
		if err != nil {
			return 0, nil, fmt.Errorf("marshal request body: %w", err)
		}
		client := &http.Client{Timeout: timeout}
		req, err := http.NewRequest(http.MethodPost, gatewayURL, bytes.NewReader(bodyBytes))
		if err != nil {
			return 0, nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			return 0, nil, err
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return resp.StatusCode, nil, err
		}
		return resp.StatusCode, body, nil
	}

	getBackendEnginePods := func(ns string) ([]string, error) {
		pods, err := k8sClient.ListPods(ctx, ns, metav1.ListOptions{LabelSelector: "openfuyao.com/engine=vllm"})
		if err != nil {
			return nil, err
		}
		var names []string
		for _, p := range pods.Items {
			if p.DeletionTimestamp != nil {
				continue
			}
			names = append(names, p.Name)
		}
		sort.Strings(names)
		return names, nil
	}

	getEPPLogLineCount := func(ns string) (int, error) {
		eppPod, err := getEPPPodName(ns)
		if err != nil {
			return 0, err
		}
		logs, err := runShell(fmt.Sprintf("kubectl -n %s logs %s -c epp", ns, eppPod))
		if err != nil {
			return 0, err
		}
		return len(strings.Split(logs, "\n")), nil
	}

	It("EPP智能路由-实例状态自动发现-PD聚合-随机策略-新增后端", func() {
		infernexCfg := helmConfig["infernex"]
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		backendNamespace := infernexCfg.Namespace

		By("使用 values.yaml + profiles/aggregate-random.yaml 构建部署 values")
		configDir, err := utils.GetHermesRouterConfigDir()
		Expect(err).NotTo(HaveOccurred())
		basePath := filepath.Join(configDir, "values.yaml")
		overlayPath := filepath.Join(configDir, "profiles/aggregate-random.yaml")
		valuesFile, err := utils.BuildValuesFromLocalAndOverlay(basePath, overlayPath)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = utils.CleanupTempValuesFile(valuesFile) })

		By("安装前清理潜在冲突并通过 Helm chart 部署 InferNex")
		err = utils.CleanupLegacyReleaseConflictsForInfernex(ctx, k8sClient, infernexCfg.ReleaseName, infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "安装前冲突清理失败")
		cfg := *infernexCfg
		cfg.ValuesFile = valuesFile
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("等待命名空间相关 Pod Running/Ready")
		waitNamespacePodsReady(infernexCfg.Namespace, 8*time.Minute)
		waitNamespacePodsReady(backendNamespace, 10*time.Minute)

		By("发送推理请求并验证响应成功")
		baseURL, err := utils.ResolveGatewayURL(ctx, k8sClient, []string{infernexCfg.Namespace})
		Expect(err).NotTo(HaveOccurred(), "解析网关地址失败")
		gatewayURL := utils.GatewayURLForPath(baseURL, "/v1/chat/completions")
		Eventually(func() error {
			reqBody := utils.BuildInferenceRequestBody(false, DefaultInferenceMessage)
			statusCode, body, reqErr := utils.DoInferenceRequest(gatewayURL, reqBody)
			if reqErr != nil {
				return reqErr
			}
			return utils.ValidateResponse(statusCode, body, false)
		}, 2*time.Minute, 5*time.Second).Should(Succeed())

		var scannedLines int

		By("第一次全量扫描 epp 日志，校验 podCount=1，并记录总行数 n")
		Eventually(func() error {
			nextLine, vErr := validatePodCountFromEPPLogs(infernexCfg.Namespace, 1, 1)
			if vErr == nil {
				scannedLines = nextLine
			}
			return vErr
		}, 90*time.Second, 5*time.Second).Should(Succeed())
		GinkgoWriter.Printf("第一次日志扫描总行数 n=%d\n", scannedLines)

		By("将推理后端 LeaderWorkerSet 扩容到 2 副本")
		lwsName, err := runShell(fmt.Sprintf("kubectl -n %s get leaderworkersets.leaderworkerset.x-k8s.io -l openfuyao.com/engine=vllm -o jsonpath='{.items[0].metadata.name}'", backendNamespace))
		Expect(err).NotTo(HaveOccurred(), "查询推理后端 LeaderWorkerSet 失败")
		Expect(lwsName).NotTo(BeEmpty(), "未找到推理后端 LeaderWorkerSet")
		_, err = runShell(fmt.Sprintf("kubectl -n %s patch leaderworkersets.leaderworkerset.x-k8s.io %s --type merge -p '{\"spec\":{\"replicas\":2}}'", backendNamespace, lwsName))
		Expect(err).NotTo(HaveOccurred(), "扩容推理后端 LeaderWorkerSet 失败")
		Eventually(func() string {
			out, getErr := runShell(fmt.Sprintf("kubectl -n %s get leaderworkersets.leaderworkerset.x-k8s.io %s -o jsonpath='{.spec.replicas}'", backendNamespace, lwsName))
			if getErr != nil {
				return ""
			}
			return strings.Trim(out, "'")
		}, 60*time.Second, 3*time.Second).Should(Equal("2"), "LeaderWorkerSet 目标副本数未生效")

		By("等待扩容后相关 Pod Running/Ready")
		waitNamespacePodsReady(backendNamespace, 10*time.Minute)

		By("再次发送推理请求并验证响应成功")
		Eventually(func() error {
			reqBody := utils.BuildInferenceRequestBody(false, DefaultInferenceMessage)
			statusCode, body, reqErr := utils.DoInferenceRequest(gatewayURL, reqBody)
			if reqErr != nil {
				return reqErr
			}
			return utils.ValidateResponse(statusCode, body, false)
		}, 2*time.Minute, 5*time.Second).Should(Succeed())

		By("第二次从第 n+1 行开始扫描 epp 日志，校验 podCount=2")
		secondScanStartLine := scannedLines + 1
		Eventually(func() error {
			nextLine, vErr := validatePodCountFromEPPLogs(infernexCfg.Namespace, 2, secondScanStartLine)
			if vErr == nil {
				scannedLines = nextLine
			}
			return vErr
		}, 90*time.Second, 5*time.Second).Should(Succeed())
		GinkgoWriter.Printf("第二次日志扫描后总行数=%d\n", scannedLines)
	})

	It("EPP智能路由-实例状态自动发现-PD聚合-随机策略-删除忙碌后端", func() {
		infernexCfg := helmConfig["infernex"]
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		backendNamespace := infernexCfg.Namespace

		By("使用 values.yaml + profiles/aggregate-random.yaml，并将聚合后端副本数设为 2")
		configDir, err := utils.GetHermesRouterConfigDir()
		Expect(err).NotTo(HaveOccurred())
		basePath := filepath.Join(configDir, "values.yaml")
		overlayPath := filepath.Join(configDir, "profiles/aggregate-random.yaml")
		baseValuesFile, err := utils.BuildValuesFromLocalAndOverlay(basePath, overlayPath)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = utils.CleanupTempValuesFile(baseValuesFile) })
		valuesFile, err := utils.SetAggregatedReplicasInValues(baseValuesFile, 2)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = utils.CleanupTempValuesFile(valuesFile) })

		By("安装前清理并通过 Helm 部署 InferNex")
		err = utils.CleanupLegacyReleaseConflictsForInfernex(ctx, k8sClient, infernexCfg.ReleaseName, infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "安装前冲突清理失败")
		cfg := *infernexCfg
		cfg.ValuesFile = valuesFile
		// 删除后端时上游会 reset，需开启 mesh 重试，单次非流式请求才有机会返回 200
		cfg.SetValues = []string{
			"--set", "hermes-router.provider.istio.retryConfig.enabled=true",
			"--set", "hermes-router.provider.istio.retryConfig.numRetries=5",
		}
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("确认命名空间中所有 Pod 均为 Running 且 Ready（超时则失败）")
		waitNamespacePodsReady(infernexCfg.Namespace, 8*time.Minute)
		waitNamespacePodsReady(backendNamespace, 10*time.Minute)

		baseURL, err := utils.ResolveGatewayURL(ctx, k8sClient, []string{infernexCfg.Namespace})
		Expect(err).NotTo(HaveOccurred(), "解析网关地址失败")
		gatewayURL := utils.GatewayURLForPath(baseURL, "/v1/chat/completions")

		reqBody := utils.BuildInferenceRequestBody(false, DefaultInferenceLongMessage)
		Expect(reqBody["stream"]).To(BeFalse(), "本用例必须为非流式请求")
		reqBody["max_tokens"] = DefaultMaxTokens

		done := make(chan struct{})
		var statusCode int
		var respBody []byte
		var reqErr error
		go func() {
			statusCode, respBody, reqErr = doInferenceWithTimeout(gatewayURL, reqBody, 4*time.Minute)
			close(done)
		}()

		time.Sleep(2 * time.Second)

		var backendSelected string
		By("读取 EPP 日志，解析 SelectedName 得到被指派的后端")
		Eventually(func() error {
			var parseErr error
			backendSelected, parseErr = extractSelectedBackendFromEPPLogs(infernexCfg.Namespace)
			return parseErr
		}, 45*time.Second, 2*time.Second).Should(Succeed())
		Expect(backendSelected).NotTo(BeEmpty())
		GinkgoWriter.Printf("backendSelected=%s\n", backendSelected)

		time.Sleep(2 * time.Second)

		select {
		case <-done:
			Expect(false).To(BeTrue(), "请求在删除忙碌后端前已结束，请拉长 DefaultInferenceLongMessage 或增大 DefaultMaxTokens")
		default:
		}

		By("删除被指派的忙碌后端 Pod")
		_, err = runShell(fmt.Sprintf("kubectl -n %s delete pod %s --wait=false", backendNamespace, backendSelected))
		Expect(err).NotTo(HaveOccurred(), "删除后端 Pod 失败")

		By("等待同一推理请求结束并校验响应成功（网关重试到健康后端）")
		Eventually(func() bool {
			select {
			case <-done:
				return true
			default:
				return false
			}
		}, 4*time.Minute, 1*time.Second).Should(BeTrue(), "推理请求在超时内未结束")
		Expect(reqErr).NotTo(HaveOccurred())
		Expect(utils.ValidateResponse(statusCode, respBody, false)).To(Succeed())
	})

	It("EPP智能路由-实例状态自动发现-PD聚合-随机策略-删除空闲后端", func() {
		infernexCfg := helmConfig["infernex"]
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		backendNamespace := infernexCfg.Namespace

		By("使用 values.yaml + profiles/aggregate-random.yaml，并将聚合后端副本数设为 2（非流式）")
		configDir, err := utils.GetHermesRouterConfigDir()
		Expect(err).NotTo(HaveOccurred())
		basePath := filepath.Join(configDir, "values.yaml")
		overlayPath := filepath.Join(configDir, "profiles/aggregate-random.yaml")
		baseValuesFile, err := utils.BuildValuesFromLocalAndOverlay(basePath, overlayPath)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = utils.CleanupTempValuesFile(baseValuesFile) })
		valuesFile, err := utils.SetAggregatedReplicasInValues(baseValuesFile, 2)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = utils.CleanupTempValuesFile(valuesFile) })

		By("安装前清理并通过 Helm 部署 InferNex")
		err = utils.CleanupLegacyReleaseConflictsForInfernex(ctx, k8sClient, infernexCfg.ReleaseName, infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "安装前冲突清理失败")
		cfg := *infernexCfg
		cfg.ValuesFile = valuesFile
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("确认命名空间中所有 Pod 均为 Running 且 Ready（超时则失败）")
		waitNamespacePodsReady(infernexCfg.Namespace, 8*time.Minute)
		waitNamespacePodsReady(backendNamespace, 10*time.Minute)

		By("获取推理后端 Pod 列表（openfuyao.com/engine=vllm），删除第一个 Pod 并立刻发送长请求")
		backendPods, err := getBackendEnginePods(backendNamespace)
		Expect(err).NotTo(HaveOccurred(), "获取后端 Pod 列表失败")
		Expect(len(backendPods)).To(BeNumerically(">=", 2), "后端 Pod 数量不足，无法执行删除空闲后端场景")
		targetPod := backendPods[0]
		_, err = runShell(fmt.Sprintf("kubectl -n %s delete pod %s --wait=false", backendNamespace, targetPod))
		Expect(err).NotTo(HaveOccurred(), "删除空闲后端 Pod 失败")

		baseURL, err := utils.ResolveGatewayURL(ctx, k8sClient, []string{infernexCfg.Namespace})
		Expect(err).NotTo(HaveOccurred(), "解析网关地址失败")
		gatewayURL := utils.GatewayURLForPath(baseURL, "/v1/chat/completions")

		reqBody := utils.BuildInferenceRequestBody(false, DefaultInferenceLongMessage)
		Expect(reqBody["stream"]).To(BeFalse(), "本用例必须为非流式请求")
		reqBody["max_tokens"] = DefaultMaxTokens

		startLine, err := getEPPLogLineCount(infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "获取 EPP 日志基线行数失败")

		done := make(chan struct{})
		var statusCode int
		var respBody []byte
		var reqErr error
		go func() {
			statusCode, respBody, reqErr = doInferenceWithTimeout(gatewayURL, reqBody, 4*time.Minute)
			close(done)
		}()

		By("等待 2 秒后校验 EPP 日志中 podCount=1")
		time.Sleep(2 * time.Second)
		Eventually(func() error {
			_, vErr := validatePodCountFromEPPLogs(infernexCfg.Namespace, 1, startLine+1)
			return vErr
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("验证请求响应成功")
		Eventually(func() bool {
			select {
			case <-done:
				return true
			default:
				return false
			}
		}, 4*time.Minute, 1*time.Second).Should(BeTrue(), "推理请求在超时内未结束")
		Expect(reqErr).NotTo(HaveOccurred())
		Expect(utils.ValidateResponse(statusCode, respBody, false)).To(Succeed())
	})

})
