package hermesrouter

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

var _ = Describe("Hermes Router Disaster Single PD", Ordered, Label("hermes-router-disaster-single-pd", "npu", "skip-temporarily"), func() {
	var (
		k8sClient   *k8s.K8SClient
		ctx         context.Context
		cancel      context.CancelFunc
		helmConfig  map[string]*utils.ReleaseConfig
		pdGroupName = "vllm-pd-1p1d-1"
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

	validatePDGroupSummaryFromEPPLogs := func(ns string, startLine int, expectGroupCount, expectPrefillCount, expectDecodeCount int) (int, error) {
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
			// EPP Pod 重启或日志被截断后行数变少，按快照起始行会越界；改为扫描当前日志全文
			startLine = 1
		}

		for i := startLine - 1; i < totalLines; i++ {
			line := strings.TrimSpace(lines[i])
			if line == "" || !strings.Contains(line, "pdGroupCount") || !strings.Contains(line, "pdGroups") {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(line), &obj); err != nil {
				continue
			}
			rawGroupCount, ok := obj["pdGroupCount"]
			if !ok {
				continue
			}
			groupCount := -1
			switch v := rawGroupCount.(type) {
			case float64:
				groupCount = int(v)
			case int:
				groupCount = v
			case string:
				n, convErr := strconv.Atoi(v)
				if convErr == nil {
					groupCount = n
				}
			}
			if groupCount != expectGroupCount {
				continue
			}
			if expectGroupCount == 0 {
				return totalLines, nil
			}

			pdGroups, ok := obj["pdGroups"].([]any)
			if !ok || len(pdGroups) == 0 {
				continue
			}
			first, ok := pdGroups[0].(map[string]any)
			if !ok {
				continue
			}

			parseInt := func(v any) int {
				switch x := v.(type) {
				case float64:
					return int(x)
				case int:
					return x
				case string:
					n, err := strconv.Atoi(x)
					if err == nil {
						return n
					}
				}
				return -1
			}
			prefill := parseInt(first["prefillCount"])
			decode := parseInt(first["decodeCount"])
			if prefill == expectPrefillCount && decode == expectDecodeCount {
				return totalLines, nil
			}
		}

		return totalLines, fmt.Errorf("在日志第 %d-%d 行中未匹配到 pdGroupCount=%d 且 prefillCount=%d、decodeCount=%d",
			startLine, totalLines, expectGroupCount, expectPrefillCount, expectDecodeCount)
	}

	getHermesRouterTempDir := func() (string, error) {
		configDir, err := utils.GetHermesRouterConfigDir()
		if err != nil {
			return "", err
		}
		tempDir := filepath.Join(filepath.Dir(configDir), "temp")
		if err := os.MkdirAll(tempDir, 0o755); err != nil {
			return "", err
		}
		return tempDir, nil
	}

	startEPPLogCapture := func(ns, fileName string) (string, func(), error) {
		tempDir, err := getHermesRouterTempDir()
		if err != nil {
			return "", nil, err
		}
		eppPod, err := getEPPPodName(ns)
		if err != nil {
			return "", nil, err
		}

		logPath := filepath.Join(tempDir, fileName)
		pidPath := logPath + ".pid"
		if err := os.WriteFile(logPath, []byte(""), 0o644); err != nil {
			return "", nil, err
		}

		_, err = runShell(fmt.Sprintf("kubectl -n %s logs -f %s -c epp >> %q 2>&1 & echo $! > %q", ns, eppPod, logPath, pidPath))
		if err != nil {
			return "", nil, err
		}

		stopFn := func() {
			_, _ = runShell(fmt.Sprintf("if [ -f %q ]; then kill $(cat %q) 2>/dev/null || true; rm -f %q; fi", pidPath, pidPath, pidPath))
		}
		return logPath, stopFn, nil
	}

	assertLogContains := func(logPath, keyword string) error {
		data, err := os.ReadFile(logPath)
		if err != nil {
			return err
		}
		if !strings.Contains(string(data), keyword) {
			return fmt.Errorf("日志 %s 未包含关键词 %q", logPath, keyword)
		}
		return nil
	}

	buildSinglePDRandomValues := func(basePath, overlayPath, pdGroupName string) (string, error) {
		mergedPath, err := utils.BuildValuesFromLocalAndOverlay(basePath, overlayPath)
		if err != nil {
			return "", err
		}
		defer func() { _ = os.Remove(mergedPath) }()

		mergedData, err := os.ReadFile(mergedPath)
		if err != nil {
			return "", fmt.Errorf("读取 overlay 合并 values 失败: %w", err)
		}
		var base map[string]any
		if err := yaml.Unmarshal(mergedData, &base); err != nil {
			return "", fmt.Errorf("解析 overlay 合并 values 失败: %w", err)
		}

		ib, _ := base["inference-backend"].(map[string]any)
		if ib == nil {
			return "", fmt.Errorf("values 缺少 inference-backend")
		}
		svcs, _ := ib["services"].([]any)
		if len(svcs) == 0 {
			return "", fmt.Errorf("values inference-backend.services 为空")
		}
		firstSvc, _ := svcs[0].(map[string]any)
		if firstSvc == nil {
			return "", fmt.Errorf("第一个 service 结构非法")
		}
		firstSvc["name"] = pdGroupName
		firstSvc["pdGroupID"] = pdGroupName
		pd, _ := firstSvc["pd"].(map[string]any)
		if pd == nil {
			return "", fmt.Errorf("service.pd 缺失")
		}
		prefill, _ := pd["prefill"].(map[string]any)
		if prefill == nil {
			return "", fmt.Errorf("service.pd.prefill 缺失")
		}
		prefill["tensorParallelSize"] = 1
		prefill["replicas"] = 1
		if decode, ok := pd["decode"].(map[string]any); ok {
			decode["replicas"] = 1
		}
		ib["services"] = []any{firstSvc}

		out, err := yaml.Marshal(base)
		if err != nil {
			return "", fmt.Errorf("序列化 values 失败: %w", err)
		}
		tempFile, err := os.CreateTemp("", "values-single-pd-random-*.yaml")
		if err != nil {
			return "", fmt.Errorf("创建临时 values 失败: %w", err)
		}
		if _, err := tempFile.Write(out); err != nil {
			_ = tempFile.Close()
			return "", fmt.Errorf("写入临时 values 失败: %w", err)
		}
		if err := tempFile.Close(); err != nil {
			return "", err
		}
		return tempFile.Name(), nil
	}

	getLWSNameByKeywords := func(ns string, include1, include2 string) (string, error) {
		lws, err := runShell(fmt.Sprintf("kubectl -n %s get leaderworkersets.leaderworkerset.x-k8s.io -o name", ns))
		if err != nil {
			return "", err
		}
		for _, line := range strings.Split(lws, "\n") {
			name := strings.TrimSpace(line)
			name = strings.TrimPrefix(name, "leaderworkerset.leaderworkerset.x-k8s.io/")
			name = strings.TrimPrefix(name, "leaderworkerset.x-k8s.io/")
			if name == "" {
				continue
			}
			if strings.Contains(name, include1) && strings.Contains(name, include2) {
				return name, nil
			}
		}
		return "", fmt.Errorf("未找到包含关键字 %q 和 %q 的 LeaderWorkerSet", include1, include2)
	}

	scaleLWSReplicas := func(ns, name string, replicas int) error {
		patch := fmt.Sprintf(`{"spec":{"replicas":%d}}`, replicas)
		_, err := runShell(fmt.Sprintf("kubectl -n %s patch leaderworkersets.leaderworkerset.x-k8s.io %s --type merge -p %q", ns, name, patch))
		return err
	}

	It("EPP智能路由-实例状态自动发现-PD分离-随机策略-新增PD实例", func() {
		infernexCfg := helmConfig["infernex"]
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		backendNamespace := infernexCfg.Namespace

		By(fmt.Sprintf("生成初始临时 values（PD random + 非流式 + 单 PD 组 %s，prefill.tp=1）", pdGroupName))
		configDir, err := utils.GetHermesRouterConfigDir()
		Expect(err).NotTo(HaveOccurred())
		basePath := filepath.Join(configDir, "values-pd-base.yaml")
		overlayPath := filepath.Join(configDir, "profiles/pd-random.yaml")
		valuesFile, err := buildSinglePDRandomValues(basePath, overlayPath, pdGroupName)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = os.Remove(valuesFile) })

		By("通过 helm chart 部署 InferNex")
		err = utils.CleanupLegacyReleaseConflictsForInfernex(ctx, k8sClient, infernexCfg.ReleaseName, infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "安装前冲突清理失败")
		cfg := *infernexCfg
		cfg.ValuesFile = valuesFile
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("确认命名空间中所有pod都running")
		waitNamespacePodsReady(backendNamespace, 10*time.Minute)

		baseURL, err := utils.ResolveGatewayURL(ctx, k8sClient, []string{infernexCfg.Namespace})
		Expect(err).NotTo(HaveOccurred(), "解析网关地址失败")
		gatewayURL := utils.GatewayURLForPath(baseURL, "/v1/chat/completions")
		shortMsg := DefaultInferenceMessage

		By("发送一次短请求并校验响应成功，增量校验 pdGroupCount=1 且 prefillCount/decodeCount=1")
		startLine1, err := getEPPLogLineCount(infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred())
		Eventually(func() error {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequest(gatewayURL, reqBody)
			if reqErr != nil {
				return reqErr
			}
			return utils.ValidateResponse(statusCode, body, false)
		}, 2*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			_, vErr := validatePDGroupSummaryFromEPPLogs(infernexCfg.Namespace, startLine1+1, 1, 1, 1)
			return vErr
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("将 prefill/decode LeaderWorkerSet 的 replicas 都改为2")
		prefillLWS, err := getLWSNameByKeywords(backendNamespace, pdGroupName, "prefill")
		Expect(err).NotTo(HaveOccurred())
		decodeLWS, err := getLWSNameByKeywords(backendNamespace, pdGroupName, "decode")
		Expect(err).NotTo(HaveOccurred())
		err = scaleLWSReplicas(backendNamespace, prefillLWS, 2)
		Expect(err).NotTo(HaveOccurred(), "scale prefill LeaderWorkerSet 失败")
		err = scaleLWSReplicas(backendNamespace, decodeLWS, 2)
		Expect(err).NotTo(HaveOccurred(), "scale decode LeaderWorkerSet 失败")

		By("确认命名空间中所有pod都running")
		waitNamespacePodsReady(backendNamespace, 12*time.Minute)

		By("再次发送一次短请求并校验响应成功，增量校验 pdGroupCount=1 且 prefillCount/decodeCount=2")
		startLine2, err := getEPPLogLineCount(infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred())
		Eventually(func() error {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequest(gatewayURL, reqBody)
			if reqErr != nil {
				return reqErr
			}
			return utils.ValidateResponse(statusCode, body, false)
		}, 2*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			_, vErr := validatePDGroupSummaryFromEPPLogs(infernexCfg.Namespace, startLine2+1, 1, 2, 2)
			return vErr
		}, 90*time.Second, 5*time.Second).Should(Succeed())
	})

	It("EPP智能路由-实例状态自动发现-PD分离-随机策略-删除唯一P实例", func() {
		infernexCfg := helmConfig["infernex"]
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		backendNamespace := infernexCfg.Namespace

		By(fmt.Sprintf("生成初始临时 values（PD random + 非流式 + 单 PD 组 %s，prefill.tp=1）", pdGroupName))
		configDir, err := utils.GetHermesRouterConfigDir()
		Expect(err).NotTo(HaveOccurred())
		basePath := filepath.Join(configDir, "values-pd-base.yaml")
		overlayPath := filepath.Join(configDir, "profiles/pd-random.yaml")
		valuesFile, err := buildSinglePDRandomValues(basePath, overlayPath, pdGroupName)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = os.Remove(valuesFile) })

		By("通过 helm chart 部署 InferNex")
		err = utils.CleanupLegacyReleaseConflictsForInfernex(ctx, k8sClient, infernexCfg.ReleaseName, infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "安装前冲突清理失败")
		cfg := *infernexCfg
		cfg.ValuesFile = valuesFile
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("确认命名空间中所有pod都running")
		waitNamespacePodsReady(backendNamespace, 10*time.Minute)

		baseURL, err := utils.ResolveGatewayURL(ctx, k8sClient, []string{infernexCfg.Namespace})
		Expect(err).NotTo(HaveOccurred(), "解析网关地址失败")
		gatewayURL := utils.GatewayURLForPath(baseURL, "/v1/chat/completions")
		shortMsg := DefaultInferenceMessage

		By("先发送一次短请求验证成功，并增量校验 pdGroupCount=1 且 prefillCount/decodeCount=1")
		startLine1, err := getEPPLogLineCount(infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred())
		Eventually(func() error {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequest(gatewayURL, reqBody)
			if reqErr != nil {
				return reqErr
			}
			return utils.ValidateResponse(statusCode, body, false)
		}, 2*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			_, vErr := validatePDGroupSummaryFromEPPLogs(infernexCfg.Namespace, startLine1+1, 1, 1, 1)
			return vErr
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("删除唯一 prefill 节点 Pod")
		eppLogPath, stopCapture, err := startEPPLogCapture(infernexCfg.Namespace, "epp-delete-prefill.log")
		Expect(err).NotTo(HaveOccurred(), "启动 EPP 日志持续采集失败")
		DeferCleanup(stopCapture)
		pods, err := k8sClient.ListPods(ctx, backendNamespace, metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred(), "列出后端 Pod 失败")
		prefillPods := make([]string, 0)
		for _, p := range pods.Items {
			if strings.Contains(p.Name, pdGroupName) && strings.Contains(p.Name, "prefill") {
				prefillPods = append(prefillPods, p.Name)
			}
		}
		Expect(len(prefillPods)).To(Equal(1), "应仅存在一个 prefill Pod")
		_, err = runShell(fmt.Sprintf("kubectl -n %s delete pod %s --wait=false", backendNamespace, prefillPods[0]))
		Expect(err).NotTo(HaveOccurred(), "删除 prefill Pod 失败")

		By("发送一次短请求并验证失败，等待5秒后校验日志包含 PrefillPods cannot be empty")
		Eventually(func() bool {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequest(gatewayURL, reqBody)
			if reqErr != nil {
				return true
			}
			return utils.ValidateResponse(statusCode, body, false) != nil
		}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "删除唯一 prefill 后请求应失败")
		time.Sleep(5 * time.Second)
		Eventually(func() error {
			return assertLogContains(eppLogPath, "PrefillPods cannot be empty")
		}, 90*time.Second, 5*time.Second).Should(Succeed())
	})

	It("EPP智能路由-实例状态自动发现-PD分离-随机策略-删除唯一D实例", func() {
		infernexCfg := helmConfig["infernex"]
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		backendNamespace := infernexCfg.Namespace

		By(fmt.Sprintf("生成初始临时 values（PD random + 非流式 + 单 PD 组 %s，prefill.tp=1）", pdGroupName))
		configDir, err := utils.GetHermesRouterConfigDir()
		Expect(err).NotTo(HaveOccurred())
		basePath := filepath.Join(configDir, "values-pd-base.yaml")
		overlayPath := filepath.Join(configDir, "profiles/pd-random.yaml")
		valuesFile, err := buildSinglePDRandomValues(basePath, overlayPath, pdGroupName)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = os.Remove(valuesFile) })

		By("通过 helm chart 部署 InferNex")
		err = utils.CleanupLegacyReleaseConflictsForInfernex(ctx, k8sClient, infernexCfg.ReleaseName, infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "安装前冲突清理失败")
		cfg := *infernexCfg
		cfg.ValuesFile = valuesFile
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("确认命名空间中所有pod都running")
		waitNamespacePodsReady(backendNamespace, 10*time.Minute)

		baseURL, err := utils.ResolveGatewayURL(ctx, k8sClient, []string{infernexCfg.Namespace})
		Expect(err).NotTo(HaveOccurred(), "解析网关地址失败")
		gatewayURL := utils.GatewayURLForPath(baseURL, "/v1/chat/completions")
		shortMsg := DefaultInferenceMessage

		By("先发送一次短请求验证成功，并增量校验 pdGroupCount=1 且 prefillCount/decodeCount=1")
		startLine1, err := getEPPLogLineCount(infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred())
		Eventually(func() error {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequest(gatewayURL, reqBody)
			if reqErr != nil {
				return reqErr
			}
			return utils.ValidateResponse(statusCode, body, false)
		}, 2*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			_, vErr := validatePDGroupSummaryFromEPPLogs(infernexCfg.Namespace, startLine1+1, 1, 1, 1)
			return vErr
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("删除唯一 decode 节点 Pod")
		eppLogPath, stopCapture, err := startEPPLogCapture(infernexCfg.Namespace, "epp-delete-decode.log")
		Expect(err).NotTo(HaveOccurred(), "启动 EPP 日志持续采集失败")
		DeferCleanup(stopCapture)
		pods, err := k8sClient.ListPods(ctx, backendNamespace, metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred(), "列出后端 Pod 失败")
		decodePods := make([]string, 0)
		for _, p := range pods.Items {
			if strings.Contains(p.Name, pdGroupName) && strings.Contains(p.Name, "decode") {
				decodePods = append(decodePods, p.Name)
			}
		}
		Expect(len(decodePods)).To(Equal(1), "应仅存在一个 decode Pod")
		_, err = runShell(fmt.Sprintf("kubectl -n %s delete pod %s --wait=false", backendNamespace, decodePods[0]))
		Expect(err).NotTo(HaveOccurred(), "删除 decode Pod 失败")

		By("发送一次短请求并验证失败，等待5秒后校验日志包含 DecodePods cannot be empty")
		Eventually(func() bool {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequest(gatewayURL, reqBody)
			if reqErr != nil {
				return true
			}
			return utils.ValidateResponse(statusCode, body, false) != nil
		}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "删除唯一 decode 后请求应失败")
		time.Sleep(5 * time.Second)
		Eventually(func() error {
			return assertLogContains(eppLogPath, "DecodePods cannot be empty")
		}, 90*time.Second, 5*time.Second).Should(Succeed())
	})

	It("EPP智能路由-实例状态自动发现-PD分离-随机策略-删除Leader实例", func() {
		infernexCfg := helmConfig["infernex"]
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		backendNamespace := infernexCfg.Namespace

		By(fmt.Sprintf("生成初始临时 values（PD random + 非流式 + 单 PD 组 %s，prefill.tp=1）", pdGroupName))
		configDir, err := utils.GetHermesRouterConfigDir()
		Expect(err).NotTo(HaveOccurred())
		basePath := filepath.Join(configDir, "values-pd-base.yaml")
		overlayPath := filepath.Join(configDir, "profiles/pd-random.yaml")
		valuesFile, err := buildSinglePDRandomValues(basePath, overlayPath, pdGroupName)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = os.Remove(valuesFile) })

		By("通过 helm chart部署 InferNex")
		err = utils.CleanupLegacyReleaseConflictsForInfernex(ctx, k8sClient, infernexCfg.ReleaseName, infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "安装前冲突清理失败")
		cfg := *infernexCfg
		cfg.ValuesFile = valuesFile
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("确认命名空间中所有pod都running")
		waitNamespacePodsReady(backendNamespace, 10*time.Minute)

		baseURL, err := utils.ResolveGatewayURL(ctx, k8sClient, []string{infernexCfg.Namespace})
		Expect(err).NotTo(HaveOccurred(), "解析网关地址失败")
		gatewayURL := utils.GatewayURLForPath(baseURL, "/v1/chat/completions")
		shortMsg := DefaultInferenceMessage

		By("发送一次短请求验证响应成功，并增量校验 pdGroupCount=1")
		startLine1, err := getEPPLogLineCount(infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred())
		Eventually(func() error {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequest(gatewayURL, reqBody)
			if reqErr != nil {
				return reqErr
			}
			return utils.ValidateResponse(statusCode, body, false)
		}, 2*time.Minute, 5*time.Second).Should(Succeed())
		Eventually(func() error {
			_, vErr := validatePDGroupSummaryFromEPPLogs(infernexCfg.Namespace, startLine1+1, 1, 1, 1)
			return vErr
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("删除唯一 leader 节点（proxy pod）")
		eppLogPath, stopCapture, err := startEPPLogCapture(infernexCfg.Namespace, "epp-delete-leader.log")
		Expect(err).NotTo(HaveOccurred(), "启动 EPP 日志持续采集失败")
		DeferCleanup(stopCapture)
		pods, err := k8sClient.ListPods(ctx, backendNamespace, metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred(), "列出后端 Pod 失败")
		leaderPods := make([]string, 0)
		for _, p := range pods.Items {
			if strings.Contains(p.Name, pdGroupName+"-proxy-") {
				leaderPods = append(leaderPods, p.Name)
			}
		}
		Expect(len(leaderPods)).To(Equal(1), "应仅存在一个 leader(proxy) Pod")
		_, err = runShell(fmt.Sprintf("kubectl -n %s delete pod %s --wait=false", backendNamespace, leaderPods[0]))
		Expect(err).NotTo(HaveOccurred(), "删除 leader Pod 失败")

		By("发送一次短请求并验证失败，等待5秒后校验日志包含 LeaderPod cannot be empty")
		Eventually(func() bool {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequest(gatewayURL, reqBody)
			if reqErr != nil {
				return true
			}
			return utils.ValidateResponse(statusCode, body, false) != nil
		}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "删除唯一 leader 后请求应失败")
		time.Sleep(5 * time.Second)
		Eventually(func() error {
			return assertLogContains(eppLogPath, "LeaderPod cannot be empty")
		}, 90*time.Second, 5*time.Second).Should(Succeed())
	})
})
