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

var _ = Describe("Hermes Router disaster pd group test", Label("hermes-router-disaster-pdgroup", "npu", "skip-temporarily"), Ordered, func() {
	var (
		k8sClient  *k8s.K8SClient
		ctx        context.Context
		cancel     context.CancelFunc
		helmConfig map[string]*utils.ReleaseConfig
	)

	// PD + vLLM 在 install/upgrade 后首包推理常超过默认 30s HTTP 超时
	const inferSlowHTTPTimeout = 4 * time.Minute

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

	validatePDGroupCountFromEPPLogs := func(ns string, expect int, startLine int) (int, error) {
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
			if line == "" || !strings.Contains(line, "pdGroupCount") {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(line), &obj); err != nil {
				continue
			}
			raw, ok := obj["pdGroupCount"]
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
		return totalLines, fmt.Errorf("在日志第 %d-%d 行中未匹配到 pdGroupCount=%d", startLine, totalLines, expect)
	}

	buildPDRandomValuesWithServiceCount := func(basePath, overlayPath string, serviceCount int) (string, error) {
		if serviceCount < 1 {
			return "", fmt.Errorf("serviceCount must be >= 1")
		}
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
		firstName := "vllm-pd-1p1d-1"
		firstSvc["name"] = firstName
		firstSvc["pdGroupID"] = firstName
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

		newSvcs := make([]any, 0, serviceCount)
		newSvcs = append(newSvcs, firstSvc)
		if serviceCount >= 2 {
			dupBytes, _ := yaml.Marshal(firstSvc)
			var second map[string]any
			if err := yaml.Unmarshal(dupBytes, &second); err != nil {
				return "", fmt.Errorf("复制第二个 pd 组失败: %w", err)
			}
			secondName := "vllm-pd-1p1d-2"
			second["name"] = secondName
			second["pdGroupID"] = secondName
			newSvcs = append(newSvcs, second)
		}
		ib["services"] = newSvcs
		out, err := yaml.Marshal(base)
		if err != nil {
			return "", fmt.Errorf("序列化 values 失败: %w", err)
		}
		tempFile, err := os.CreateTemp("", fmt.Sprintf("values-pd-random-%d-*.yaml", serviceCount))
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

	It("EPP智能路由-实例状态自动发现-PD分离-随机策略-新增PD组", func() {
		infernexCfg := helmConfig["infernex"]
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		backendNamespace := infernexCfg.Namespace

		configDir, err := utils.GetHermesRouterConfigDir()
		Expect(err).NotTo(HaveOccurred())
		basePath := filepath.Join(configDir, "values-pd-base.yaml")
		overlayPath := filepath.Join(configDir, "profiles/pd-random.yaml")

		By("生成初始临时 values（PD random + 非流式 + 单PD组 vllm-pd-1p1d-1，prefill.tp=1）")
		valuesV1, err := buildPDRandomValuesWithServiceCount(basePath, overlayPath, 1)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = os.Remove(valuesV1) })

		By("通过 helm chart 部署 InferNex")
		err = utils.CleanupLegacyReleaseConflictsForInfernex(ctx, k8sClient, infernexCfg.ReleaseName, infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "安装前冲突清理失败")
		cfg := *infernexCfg
		cfg.ValuesFile = valuesV1
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("确认命名空间中所有 Pod 均 Running/Ready")
		waitNamespacePodsReady(backendNamespace, 10*time.Minute)

		baseURL, err := utils.ResolveGatewayURL(ctx, k8sClient, []string{infernexCfg.Namespace})
		Expect(err).NotTo(HaveOccurred(), "解析网关地址失败")
		gatewayURL := utils.GatewayURLForPath(baseURL, "/v1/chat/completions")
		shortMsg := DefaultInferenceMessage

		By("发送短请求并增量扫描 EPP 日志验证 pdGroupCount=1（每轮长超时 + 校验日志）")
		startLine1, err := getEPPLogLineCount(infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred())
		Eventually(func() error {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequestWithTimeout(gatewayURL, reqBody, inferSlowHTTPTimeout)
			if reqErr != nil {
				return reqErr
			}
			if err := utils.ValidateResponse(statusCode, body, false); err != nil {
				return err
			}
			_, vErr := validatePDGroupCountFromEPPLogs(infernexCfg.Namespace, 1, startLine1+1)
			return vErr
		}, 8*time.Minute, 8*time.Second).Should(Succeed())

		By("生成升级临时 values：新增 PD 组 vllm-pd-1p1d-2，且每组 pdGroupID 与 name 一致")
		valuesV2, err := buildPDRandomValuesWithServiceCount(basePath, overlayPath, 2)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = os.Remove(valuesV2) })

		By("执行 helm upgrade 应用新的 values")
		upCfg := *infernexCfg
		upCfg.ValuesFile = valuesV2
		err = utils.HelmUpgradeWithLatestFallback(&upCfg)
		Expect(err).NotTo(HaveOccurred(), "helm upgrade infernex 执行失败")

		By("确认升级后命名空间中所有 Pod 均 Running/Ready")
		waitNamespacePodsReady(backendNamespace, 12*time.Minute)

		By("升级后每轮长超时推理并增量扫描 EPP 日志验证 pdGroupCount=2")
		startLine2, err := getEPPLogLineCount(infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred())
		Eventually(func() error {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequestWithTimeout(gatewayURL, reqBody, inferSlowHTTPTimeout)
			if reqErr != nil {
				return reqErr
			}
			if err := utils.ValidateResponse(statusCode, body, false); err != nil {
				return err
			}
			_, vErr := validatePDGroupCountFromEPPLogs(infernexCfg.Namespace, 2, startLine2+1)
			return vErr
		}, 12*time.Minute, 8*time.Second).Should(Succeed())
	})

	It("EPP智能路由-实例状态自动发现-PD分离-随机策略-删除PD组", func() {
		infernexCfg := helmConfig["infernex"]
		Expect(infernexCfg).NotTo(BeNil(), "Helm 配置中不包含 infernex")
		backendNamespace := infernexCfg.Namespace

		configDir, err := utils.GetHermesRouterConfigDir()
		Expect(err).NotTo(HaveOccurred())
		basePath := filepath.Join(configDir, "values-pd-base.yaml")
		overlayPath := filepath.Join(configDir, "profiles/pd-random.yaml")

		By("生成初始临时 values（PD random + 非流式 + 两个PD组：vllm-pd-1p1d-1 和 vllm-pd-1p1d-2，prefill.tp=1）")
		valuesV1, err := buildPDRandomValuesWithServiceCount(basePath, overlayPath, 2)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = os.Remove(valuesV1) })

		By("通过 helm chart 部署 InferNex")
		err = utils.CleanupLegacyReleaseConflictsForInfernex(ctx, k8sClient, infernexCfg.ReleaseName, infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "安装前冲突清理失败")
		cfg := *infernexCfg
		cfg.ValuesFile = valuesV1
		err = utils.HelmInstallWithLatestFallback(&cfg)
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 执行失败")

		By("确认命名空间中所有 Pod 均 Running/Ready")
		waitNamespacePodsReady(backendNamespace, 10*time.Minute)

		baseURL, err := utils.ResolveGatewayURL(ctx, k8sClient, []string{infernexCfg.Namespace})
		Expect(err).NotTo(HaveOccurred(), "解析网关地址失败")
		gatewayURL := utils.GatewayURLForPath(baseURL, "/v1/chat/completions")
		shortMsg := DefaultInferenceMessage

		By("发送短请求并增量扫描 EPP 日志验证 pdGroupCount=2（Eventually 最长 90s）")
		startLine1, err := getEPPLogLineCount(infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred())
		Eventually(func() error {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequestWithTimeout(gatewayURL, reqBody, 60*time.Second)
			if reqErr != nil {
				return reqErr
			}
			if err := utils.ValidateResponse(statusCode, body, false); err != nil {
				return err
			}
			_, vErr := validatePDGroupCountFromEPPLogs(infernexCfg.Namespace, 2, startLine1+1)
			return vErr
		}, 90*time.Second, 5*time.Second).Should(Succeed())

		By("生成升级临时 values：删除 services 下 pd 组 vllm-pd-1p1d-2")
		valuesV2, err := buildPDRandomValuesWithServiceCount(basePath, overlayPath, 1)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func() { _ = os.Remove(valuesV2) })

		By("执行 helm upgrade 应用新的 values")
		upCfg := *infernexCfg
		upCfg.ValuesFile = valuesV2
		err = utils.HelmUpgradeWithLatestFallback(&upCfg)
		Expect(err).NotTo(HaveOccurred(), "helm upgrade infernex 执行失败")

		By("确认升级后命名空间中所有 Pod 均 Running/Ready")
		waitNamespacePodsReady(backendNamespace, 12*time.Minute)

		By("删除 PD 组并升级后，每轮长超时推理并增量扫描 EPP 日志验证 pdGroupCount=1")
		startLine2, err := getEPPLogLineCount(infernexCfg.Namespace)
		Expect(err).NotTo(HaveOccurred())
		Eventually(func() error {
			reqBody := utils.BuildInferenceRequestBody(false, shortMsg)
			statusCode, body, reqErr := utils.DoInferenceRequestWithTimeout(gatewayURL, reqBody, inferSlowHTTPTimeout)
			if reqErr != nil {
				return reqErr
			}
			if err := utils.ValidateResponse(statusCode, body, false); err != nil {
				return err
			}
			_, vErr := validatePDGroupCountFromEPPLogs(infernexCfg.Namespace, 1, startLine2+1)
			return vErr
		}, 12*time.Minute, 8*time.Second).Should(Succeed())
	})

})
