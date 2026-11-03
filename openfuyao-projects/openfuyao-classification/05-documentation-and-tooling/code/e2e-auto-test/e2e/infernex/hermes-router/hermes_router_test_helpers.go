package hermesrouter

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
	"gopkg.in/yaml.v3"
)

const (
	eppContainerName        = "epp"
	predictionContainerName = "prediction"
)

func runShell(cmd string) (string, error) {
	out, err := exec.Command("bash", "-c", cmd).CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("%w: %s", err, strings.TrimSpace(string(out)))
	}
	return string(out), nil
}

func findEPPPodName(env *HermesTestEnv) (string, error) {
	pods, err := env.Client.ListPods(env.Ctx, env.Namespace, metav1.ListOptions{})
	if err != nil {
		return "", fmt.Errorf("list pods failed: %w", err)
	}
	for _, pod := range pods.Items {
		// 跳过正在删除（Terminating）的旧 release 残留 pod，避免误选无 jsonl 文件的残留实例。
		if pod.DeletionTimestamp != nil {
			continue
		}
		if strings.Contains(pod.Name, "epp") && pod.Status.Phase == corev1.PodRunning {
			return pod.Name, nil
		}
	}
	return "", fmt.Errorf("no running epp pod found in namespace %s", env.Namespace)
}

func getEPPContainerLogs(env *HermesTestEnv, container string, tail int, previous bool) (string, error) {
	podName, err := findEPPPodName(env)
	if err != nil {
		return "", err
	}
	prevFlag := ""
	if previous {
		prevFlag = " --previous"
	}
	cmd := fmt.Sprintf("kubectl -n %s logs %s -c %s%s --tail=%d", env.Namespace, podName, container, prevFlag, tail)
	return runShell(cmd)
}

// getEPPContainerLogsWithFallback reads current container logs and falls back to --previous
// when the current log stream is empty or missing the expected substring.
func getEPPContainerLogsWithFallback(env *HermesTestEnv, container string, tail int, mustContain string) (string, error) {
	logs, err := getEPPContainerLogs(env, container, tail, false)
	if err == nil && strings.Contains(logs, mustContain) {
		return logs, nil
	}
	prevLogs, prevErr := getEPPContainerLogs(env, container, tail, true)
	if prevErr != nil {
		if err != nil {
			return "", err
		}
		return logs, nil
	}
	if strings.TrimSpace(prevLogs) != "" {
		return prevLogs, nil
	}
	if err != nil {
		return "", err
	}
	return logs, nil
}

// assertEPPCrashLogContains 轮询读取 EPP 容器日志（含 --previous 回退）直到出现 mustContain。
// CrashLoopBackOff 时错误日志会在 current/previous 间漂移且首读时机不稳，一次性读取可能漏判
// （IT-031/032/033 曾因此 flaky），故用 Eventually 轮询确保稳定捕获门禁报错。
func assertEPPCrashLogContains(env *HermesTestEnv, mustContain string) {
	Eventually(func(g Gomega) {
		logs, err := getEPPContainerLogsWithFallback(env, eppContainerName, 200, mustContain)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(logs).To(ContainSubstring(mustContain))
	}, 2*time.Minute, 5*time.Second).Should(Succeed())
}

func killPredictionSidecar(env *HermesTestEnv) error {
	podName, err := findEPPPodName(env)
	if err != nil {
		return err
	}

	killCmd := fmt.Sprintf(
		"kubectl -n %s exec %s -c %s -- sh -c 'kill -TERM 1'",
		env.Namespace, podName, predictionContainerName,
	)
	if _, err = runShell(killCmd); err != nil {
		return fmt.Errorf("kill prediction sidecar failed: %w", err)
	}

	// Remove shared socket so EPP cannot reconnect during the kubelet restart window.
	socketCmd := fmt.Sprintf(
		"kubectl -n %s exec %s -c %s -- sh -c 'rm -f /var/run/hermes/prediction.sock'",
		env.Namespace, podName, eppContainerName,
	)
	if _, socketErr := runShell(socketCmd); socketErr != nil {
		GinkgoWriter.Printf("warning: remove prediction.sock failed: %v\n", socketErr)
	}

	Eventually(func() bool {
		ready, checkErr := CheckSidecarsReady(env.Client, env.Ctx, env.Namespace, []string{predictionContainerName})
		if checkErr != nil {
			return false
		}
		socketOK, socketErr := CheckSocketExists(env.Client, env.Ctx, env.Namespace, "/var/run/hermes/prediction.sock")
		return !ready || socketErr != nil || !socketOK
	}, 30*time.Second, 500*time.Millisecond).Should(BeTrue(), "prediction sidecar 应进入不可用状态")

	return nil
}

func predictionSidecarUnavailable(env *HermesTestEnv) bool {
	ready, err := CheckSidecarsReady(env.Client, env.Ctx, env.Namespace, []string{predictionContainerName})
	if err != nil || ready {
		return false
	}
	socketOK, err := CheckSocketExists(env.Client, env.Ctx, env.Namespace, "/var/run/hermes/prediction.sock")
	return err != nil || !socketOK
}

func waitForPredictionSidecarReady(env *HermesTestEnv, timeout time.Duration) {
	Eventually(func() bool {
		ok, err := CheckSidecarsReady(env.Client, env.Ctx, env.Namespace, []string{predictionContainerName})
		if err != nil {
			return false
		}
		if !ok {
			return false
		}
		socketOK, err := CheckSocketExists(env.Client, env.Ctx, env.Namespace, "/var/run/hermes/prediction.sock")
		return err == nil && socketOK
	}, timeout, 5*time.Second).Should(BeTrue(), "prediction sidecar 未恢复 Ready")
}

func waitForEPPContainerCrashLoop(env *HermesTestEnv, container string, timeout time.Duration) {
	Eventually(func() bool {
		pods, err := env.Client.ListPods(env.Ctx, env.Namespace, metav1.ListOptions{})
		if err != nil {
			return false
		}
		for _, pod := range pods.Items {
			if !strings.Contains(pod.Name, "epp") {
				continue
			}
			for _, cs := range pod.Status.ContainerStatuses {
				if cs.Name != container {
					continue
				}
				if cs.State.Waiting == nil {
					continue
				}
				reason := cs.State.Waiting.Reason
				return reason == "CrashLoopBackOff" || reason == "Error"
			}
		}
		return false
	}, timeout, 5*time.Second).Should(BeTrue(), fmt.Sprintf("%s 容器未进入 CrashLoopBackOff/Error 状态", container))
}

func buildEmptyUserMessageRequestBody() map[string]any {
	return map[string]any{
		"model": DefaultInferenceModel,
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": "",
			},
		},
		"stream":     DefaultInferenceStream,
		"max_tokens": DefaultMaxTokens,
	}
}

func buildWhitespaceUserMessageRequestBody() map[string]any {
	return map[string]any{
		"model": DefaultInferenceModel,
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": "   \t  ",
			},
		},
		"stream":     DefaultInferenceStream,
		"max_tokens": DefaultMaxTokens,
	}
}

// predictionEmpty 判断一条记录是否「无有效预测」：prediction 为 nil，或为全零占位对象。
// 降级路径下 store 写入的是零值 LatencyPrediction{}（Go 的 omitempty 对 struct 不生效，JSON 里
// 会出现 prediction: {}），测试侧反序列化为非 nil 的全零 *PredictionData，故不能只判 nil。
func predictionEmpty(record CompletedRecord) bool {
	if record.Prediction == nil {
		return true
	}
	return record.Prediction.TTFTMillis == 0 &&
		record.Prediction.TPOTMillis == 0 &&
		record.Prediction.TotalLatencyMillis == 0
}

func assertPredictionEmpty(record CompletedRecord) {
	Expect(predictionEmpty(record)).To(BeTrue(), "prediction 应为空值（nil 或全零占位）")
	Expect(record.PredictionMode).NotTo(Equal("active"), "predictionMode 不应为 active")
}

func assertPredictionActive(record CompletedRecord) {
	Expect(record.PredictionMode).To(Equal("active"), "predictionMode 应为 active")
	Expect(record.Prediction).NotTo(BeNil(), "prediction 应存在")
	Expect(record.Prediction.TTFTMillis).To(BeNumerically(">", 0), "prediction.ttftMillis 应大于 0")
}

// assertRequestRecorded 验证请求成功路由并产生了 completed 记录（fail-open 不中断）。
// 空/空白 prompt 经 chat template 后仍会产生若干 token，prediction 未必被跳过；本断言只确认
// 请求被正常处理并记录，不强制 prediction 为空。
func assertRequestRecorded(record CompletedRecord) {
	Expect(record.RequestId).NotTo(BeEmpty(), "completed 记录应有 requestId")
	Expect(record.Model).NotTo(BeEmpty(), "completed 记录应有 model")
}

func assertNewCompletedRecordsPredictionEmpty(env *HermesTestEnv, baselineCount, expectedNew int, timeout time.Duration) {
	var records []CompletedRecord
	Eventually(func() error {
		currentCount, countErr := countCompletedRecords(env)
		if countErr != nil {
			return countErr
		}
		if currentCount < baselineCount+expectedNew {
			return fmt.Errorf("waiting for %d new records: have %d want >= %d", expectedNew, currentCount-baselineCount, expectedNew)
		}
		newRecords, readErr := ReadCompletedJSONL(env.Client, env.Ctx, env.Namespace, expectedNew, 30*time.Second)
		if readErr != nil {
			return readErr
		}
		if len(newRecords) < expectedNew {
			return fmt.Errorf("read %d records, want %d", len(newRecords), expectedNew)
		}
		records = newRecords
		return nil
	}, timeout, 2*time.Second).Should(Succeed(), "等待新增 completed 记录失败")
	for _, record := range records {
		assertPredictionEmpty(record)
	}
}

// getAggregateBackendName 返回聚合推理后端的 LeaderWorkerSet 名称。
// aggregate 模式下 chart 渲染的是 aggregated-engine-lws.yaml（LeaderWorkerSet），
// 而非 Deployment，故用 leaderworkerset 资源类型按 openfuyao.com/engine=vllm 标签查找。
// 用 jsonpath 直接取 metadata.name，避免 -o name 的 "<plural>.<group>/<name>" 格式解析。
func getAggregateBackendName(env *HermesTestEnv) (string, error) {
	cmd := fmt.Sprintf(
		"kubectl -n %s get leaderworkerset -l openfuyao.com/engine=vllm -o jsonpath='{.items[*].metadata.name}'",
		env.Namespace,
	)
	name, err := runShell(cmd)
	if err != nil {
		return "", fmt.Errorf("find aggregate backend leaderworkerset failed: %w", err)
	}
	name = strings.TrimSpace(name)
	// jsonpath 无匹配时返回空串且退出码 0，需显式判空；多匹配时取第一个。
	if first := strings.Fields(name); len(first) > 0 {
		name = first[0]
	} else {
		return "", fmt.Errorf("aggregate backend leaderworkerset not found")
	}
	return name, nil
}

// scaleAggregateBackend 将聚合推理后端（LeaderWorkerSet）扩缩容到 replicas。
// LWS CRD 暴露了 scale 子资源（specReplicasPath=.spec.replicas），故 kubectl scale 可直接生效。
func scaleAggregateBackend(env *HermesTestEnv, replicas int) error {
	name, err := getAggregateBackendName(env)
	if err != nil {
		return err
	}
	_, err = runShell(fmt.Sprintf("kubectl -n %s scale leaderworkerset %s --replicas=%d", env.Namespace, name, replicas))
	if err != nil {
		return fmt.Errorf("scale leaderworkerset %s failed: %w", name, err)
	}
	GinkgoWriter.Printf("scaled leaderworkerset %s to %d replicas\n", name, replicas)
	return nil
}

func countAggregateBackendPods(env *HermesTestEnv) (int, error) {
	cmd := fmt.Sprintf(
		`kubectl -n %s get pods -l openfuyao.com/engine=vllm --field-selector=status.phase=Running -o name`,
		env.Namespace,
	)
	out, err := runShell(cmd)
	if err != nil {
		return 0, err
	}
	out = strings.TrimSpace(out)
	if out == "" {
		return 0, nil
	}
	return len(strings.Split(out, "\n")), nil
}

func countCompletedRecords(env *HermesTestEnv) (int, error) {
	podName, err := findEPPPodName(env)
	if err != nil {
		return 0, err
	}
	cmd := fmt.Sprintf("kubectl exec %s -n %s -c epp -- cat /tmp/hermes-inflight/completed.jsonl", podName, env.Namespace)
	stdout, err := runShell(cmd)
	if err != nil {
		if strings.Contains(err.Error(), "No such file") {
			return 0, nil
		}
		return 0, err
	}
	count := 0
	scanner := bufio.NewScanner(strings.NewReader(stdout))
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) != "" {
			count++
		}
	}
	return count, nil
}

func waitForNewCompletedRecord(env *HermesTestEnv, baselineCount int, timeout time.Duration) CompletedRecord {
	var record CompletedRecord
	Eventually(func() error {
		currentCount, countErr := countCompletedRecords(env)
		if countErr != nil {
			return countErr
		}
		if currentCount <= baselineCount {
			return fmt.Errorf("waiting for completed record: have %d want > %d", currentCount, baselineCount)
		}
		records, readErr := ReadCompletedJSONL(env.Client, env.Ctx, env.Namespace, currentCount-baselineCount, 30*time.Second)
		if readErr != nil {
			return readErr
		}
		if len(records) == 0 {
			return fmt.Errorf("no new completed records")
		}
		record = records[len(records)-1]
		return nil
	}, timeout, 2*time.Second).Should(Succeed())
	return record
}

// injectInvalidPredictorAddress overrides default-plugins.yaml via pluginsCustomConfig so
// prediction-scorer receives a non-unix:// predictorAddress (HR-PRED-IT-031).
func injectInvalidPredictorAddress(valuesFilePath string) (string, error) {
	configDir, err := GetHermesRouterConfigDir()
	if err != nil {
		return "", err
	}
	fixturePath := filepath.Join(configDir, "fixtures/it-031-invalid-predictor-plugins.yaml")
	fixture, err := os.ReadFile(fixturePath)
	if err != nil {
		return "", fmt.Errorf("read IT-031 fixture %s: %w", fixturePath, err)
	}

	data, err := os.ReadFile(valuesFilePath)
	if err != nil {
		return "", fmt.Errorf("read values file %s: %w", valuesFilePath, err)
	}
	var values map[string]any
	if err := yaml.Unmarshal(data, &values); err != nil {
		return "", fmt.Errorf("parse values yaml: %w", err)
	}

	hr, ok := values["hermes-router"].(map[string]any)
	if !ok {
		return "", fmt.Errorf("values 中缺少 hermes-router 键")
	}
	infExt, ok := hr["inferenceExtension"].(map[string]any)
	if !ok {
		return "", fmt.Errorf("hermes-router 中缺少 inferenceExtension 键")
	}
	infExt["pluginsCustomConfig"] = map[string]any{
		"default-plugins.yaml": string(fixture),
	}

	yamlData, err := yaml.Marshal(values)
	if err != nil {
		return "", fmt.Errorf("marshal patched values yaml: %w", err)
	}

	hermesDir, err := GetHermesRouterDir()
	if err != nil {
		return "", err
	}
	tempDir := filepath.Join(hermesDir, "tmp")
	if err := os.MkdirAll(tempDir, 0o755); err != nil {
		return "", fmt.Errorf("create temp dir: %w", err)
	}
	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-it-031-%d.yaml", time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlData, 0o644); err != nil {
		return "", fmt.Errorf("write temp values file: %w", err)
	}
	return tempFile, nil
}

func installAggregatePredictionValues(env *HermesTestEnv, extraSetValues ...string) (valuesFile string, cfg *ReleaseConfig, tempFiles []string, err error) {
	hermesDir, err := GetHermesRouterDir()
	if err != nil {
		return "", nil, nil, err
	}
	configDir, err := GetHermesRouterConfigDir()
	if err != nil {
		return "", nil, nil, err
	}
	valuesFile, err = BuildValuesFromLocalAndOverlay(
		filepath.Join(hermesDir, AggregateBaseValuesPath),
		filepath.Join(configDir, "profiles/aggregate-prediction.yaml"),
	)
	if err != nil {
		return "", nil, nil, err
	}
	tempFiles = append(tempFiles, valuesFile)

	if env.HelmConfig["infernex"] != nil && env.HelmConfig["infernex"].PredictionModelPath != "" {
		injectedFile, injectErr := InjectPredictionModelVolume(valuesFile, env.HelmConfig["infernex"].PredictionModelPath)
		if injectErr != nil {
			return "", nil, tempFiles, injectErr
		}
		valuesFile = injectedFile
		tempFiles = append(tempFiles, injectedFile)
	}

	cfgEntry := env.HelmConfig["infernex"]
	if cfgEntry == nil {
		return "", nil, tempFiles, fmt.Errorf("helm config missing infernex entry")
	}
	cfgCopy := *cfgEntry
	cfgCopy.ValuesFile = valuesFile
	cfgCopy.SetValues = append([]string(nil), extraSetValues...)
	// 将 EPP 钉到持有 hostPath 模型/缓存数据的节点（LoadHelmConfig 按本机对照集群 Node 识别）。
	// 经 --set 注入：-f 里的 affinity 会被 chart 的空 affinity:{} 默认值 coalesce 掉，只有 --set 生效。
	cfgCopy.SetValues = append(cfgCopy.SetValues, eppNodeAffinitySetValues(env)...)
	return valuesFile, &cfgCopy, tempFiles, nil
}

// eppNodeAffinitySetValues 生成把 EPP（经 inferenceExtension.affinity nodeAffinity）钉到指定节点的
// --set 参数；节点取自 LoadHelmConfig 运行时识别的 EppNode。
func eppNodeAffinitySetValues(env *HermesTestEnv) []string {
	cfg := env.HelmConfig["infernex"]
	if cfg == nil || cfg.EppNode == "" {
		return nil
	}
	const prefix = "hermes-router.inferenceExtension.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0]"
	return []string{
		"--set " + prefix + ".key=kubernetes.io/hostname",
		"--set " + prefix + ".operator=In",
		"--set " + prefix + ".values[0]=" + cfg.EppNode,
	}
}

// bootstrapAggregatePrediction 安装共享的 aggregate-prediction release，供 prediction-fallback
// 各 Describe 在 BeforeAll 中调用：临时 values 文件通过 DeferCleanup 清理，返回的 cfg 需由调用方
// 在 AfterAll 中 HelmUninstall 卸载。
func bootstrapAggregatePrediction(env *HermesTestEnv) (cfg *ReleaseConfig) {
	_, cfg, tempFiles, err := installAggregatePredictionValues(env)
	Expect(err).NotTo(HaveOccurred(), "构建 aggregate-prediction values 失败")
	DeferCleanup(func() { deferCleanupTempValuesFiles(tempFiles) })

	By("Helm 安装 aggregate-prediction")
	err = HelmInstallWithLatestFallback(cfg)
	Expect(err).NotTo(HaveOccurred(), "Helm 安装 aggregate-prediction 失败")

	time.Sleep(60 * time.Second)
	return cfg
}

func deferCleanupTempValuesFiles(files []string) {
	for _, file := range files {
		if strings.TrimSpace(file) == "" {
			continue
		}
		_ = CleanupTempValuesFile(file)
	}
}

// resolveBackendServiceURL 在指定命名空间中查找 inference-backend Service 并构建直连 URL。
// 当 inferenceGateway / hermes-router 被关闭（选装场景）时，gateway Service 不存在，
// 需绕过 gateway 直接访问 inference-backend 的 ClusterIP Service 完成推理校验。
// 逻辑对齐 infernex 套件 resolveInferenceURL 的 backend 回退分支。
func resolveBackendServiceURL(env *HermesTestEnv) (string, error) {
	svcs, err := env.Client.ListServices(env.Ctx, env.Namespace, metav1.ListOptions{})
	if err != nil {
		return "", fmt.Errorf("list services in %s: %w", env.Namespace, err)
	}
	for i := range svcs.Items {
		svc := &svcs.Items[i]
		if svc.Labels["app.kubernetes.io/name"] != "inference-backend" {
			continue
		}
		// 跳过 status-port/health 等非业务端口，取第一个业务端口
		skipPortNames := map[string]bool{"status-port": true, "health": true, "healthz": true, "readiness": true, "metrics": true, "monitor": true}
		for _, sp := range svc.Spec.Ports {
			if skipPortNames[strings.ToLower(sp.Name)] {
				continue
			}
			if svc.Spec.ClusterIP != "" && svc.Spec.ClusterIP != "None" {
				return fmt.Sprintf("http://%s:%d", svc.Spec.ClusterIP, sp.Port), nil
			}
		}
	}
	return "", fmt.Errorf("在命名空间 %s 中未找到可用的 inference-backend Service", env.Namespace)
}

// doInferenceRequestViaBackend 绕过 gateway，直接向 inference-backend Service 发送推理请求并校验响应。
// 用于 inferenceGateway 关闭的选装场景（gateway Service 不存在，无法走 DoInferenceRequestFromEnv）。
func doInferenceRequestViaBackend(env *HermesTestEnv) error {
	baseURL, err := resolveBackendServiceURL(env)
	if err != nil {
		return err
	}
	fullURL := strings.TrimSuffix(baseURL, "/") + "/v1/chat/completions"
	reqBody := BuildInferenceRequestBody(DefaultInferenceStream, DefaultInferenceMessage)
	statusCode, body, err := DoInferenceRequest(fullURL, reqBody)
	if err != nil {
		return fmt.Errorf("backend 直连推理请求失败: %w", err)
	}
	return ValidateResponse(statusCode, body, DefaultInferenceStream)
}

// helmReleaseResourceCount 获取 helm release 部署的资源数量（经默认 driver，与 HelmInstallWithLatestFallback 一致）。
// hermes-router 套件的 HelmInstallWithLatestFallback/HelmUninstall 用 execLocal（默认 secret driver），
// 而共享的 HelmGetManifest 强制 HELM_DRIVER=configmap，两者 driver 不一致会导致 "release: not found"。
// 故此用例内用 execLocal 直跑 helm get manifest，避免 driver 冲突。
func helmReleaseResourceCount(releaseName, namespace string) (int, error) {
	cmd := fmt.Sprintf("helm get manifest %s -n %s", releaseName, namespace)
	out, err := runShell(cmd)
	if err != nil {
		return 0, fmt.Errorf("helm get manifest %s 失败: %w", releaseName, err)
	}
	resources, err := ParseHelmManifestResources(out)
	if err != nil {
		return 0, fmt.Errorf("解析 helm manifest 失败: %w", err)
	}
	return len(resources), nil
}
