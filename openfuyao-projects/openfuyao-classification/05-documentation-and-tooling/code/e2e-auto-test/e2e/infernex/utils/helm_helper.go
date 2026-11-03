package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// execLocal 在本机执行命令并返回 stdout/stderr
func execLocal(cmd string) (string, string, error) {
	c := exec.Command("bash", "-lc", cmd)
	var stdout, stderr bytes.Buffer
	c.Stdout = &stdout
	c.Stderr = &stderr
	err := c.Run()
	return stdout.String(), stderr.String(), err
}

// getPackageDir 返回当前包所在目录（即 utils 所在目录的父目录：infernex）
func getPackageDir() (string, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("runtime.Caller failed")
	}
	return filepath.Dir(filepath.Dir(file)), nil
}

// GetPackageDir returns the absolute path to the infernex package directory,
// which is the parent of the utils directory. Useful for resolving project-level
// resource paths for profile overlays (now in hermes-router-config/profiles/).
func GetPackageDir() (string, error) {
	return getPackageDir()
}

// GetHermesRouterDir 返回 hermes-router 包目录的绝对路径，
// hermes-router 目录位于 infernex 目录下：e2e/infernex/hermes-router
func GetHermesRouterDir() (string, error) {
	dir, err := getPackageDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "hermes-router"), nil
}

// GetHermesRouterConfigDir 返回 hermes-router 配置目录的绝对路径（hermes-router/hermes-router-config），
// 该目录下存放 values.yaml、epp-*.yaml、inference-backend-*.yaml、helm_config.yaml 等配置
func GetHermesRouterConfigDir() (string, error) {
	dir, err := GetHermesRouterDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "hermes-router-config"), nil
}

// GetCacheIndexerDir 返回 cache-indexer 包目录的绝对路径，
// cache-indexer 目录位于 infernex 目录下：e2e/infernex/cache-indexer
func GetCacheIndexerDir() (string, error) {
	dir, err := getPackageDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "cache-indexer"), nil
}

// GetCacheIndexerConfigDir 返回 cache-indexer 配置目录的绝对路径（cache-indexer/cache-indexer-config）。
func GetCacheIndexerConfigDir() (string, error) {
	dir, err := GetCacheIndexerDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "cache-indexer-config"), nil
}

// LoadHelmConfig 从 hermes-router-config 目录加载 helm_config.yaml（不依赖当前工作目录）
func LoadHelmConfig() (map[string]*ReleaseConfig, error) {
	configDir, err := GetHermesRouterConfigDir()
	if err != nil {
		return nil, err
	}
	configPath := filepath.Join(configDir, "helm_config.yaml")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("read helm config failed: %v", err)
	}
	var cfg map[string]*ReleaseConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse helm config yaml failed: %v", err)
	}
	for _, entry := range cfg {
		if entry == nil {
			continue
		}
		resolved, resolveErr := resolveHelmURL(configDir, entry.HelmURL)
		if resolveErr != nil {
			return nil, resolveErr
		}
		entry.HelmURL = resolved
	}
	if infernexCfg := cfg["infernex"]; infernexCfg != nil && infernexCfg.PredictionModelPath == "" {
		if hermesCfg := cfg["hermes-router"]; hermesCfg != nil && hermesCfg.PredictionModelPath != "" {
			infernexCfg.PredictionModelPath = hermesCfg.PredictionModelPath
		}
	}
	for _, entry := range cfg {
		if entry == nil || entry.PredictionModelPath == "" {
			continue
		}
		resolved, resolveErr := resolvePredictionModelPath(configDir, entry.PredictionModelPath)
		if resolveErr != nil {
			return nil, resolveErr
		}
		entry.PredictionModelPath = resolved
	}
	eppNode, eppErr := detectTestExecutorNode()
	if eppErr != nil {
		return nil, eppErr
	}
	for _, key := range []string{"infernex", "hermes-router"} {
		if entry := cfg[key]; entry != nil {
			entry.EppNode = eppNode
		}
	}
	return cfg, nil
}

// resolveHelmURL 将 helm_config 中的 helmUrl 解析为绝对路径。
// OCI URL (oci://...) 或绝对路径原样返回；相对路径以 config 目录为基准解析。
func resolveHelmURL(configDir, helmURL string) (string, error) {
	if helmURL == "" || strings.HasPrefix(helmURL, "oci://") || filepath.IsAbs(helmURL) {
		return helmURL, nil
	}
	abs, err := filepath.Abs(filepath.Join(configDir, helmURL))
	if err != nil {
		return "", fmt.Errorf("resolve helmUrl %q failed: %w", helmURL, err)
	}
	return abs, nil
}

// resolvePredictionModelPath 将 helm_config 中的 predictionModelPath 解析为绝对路径。
// 相对路径以 hermes-router-config 目录为基准；绝对路径原样返回（供 hostPath 挂载使用）。
func resolvePredictionModelPath(configDir, modelPath string) (string, error) {
	if filepath.IsAbs(modelPath) {
		return filepath.Clean(modelPath), nil
	}
	abs, err := filepath.Abs(filepath.Join(configDir, modelPath))
	if err != nil {
		return "", fmt.Errorf("resolve predictionModelPath %q failed: %w", modelPath, err)
	}
	return abs, nil
}

// LoadCacheIndexerHelmConfig 从 cache-indexer-config 目录加载 helm_config.yaml（不依赖当前工作目录）。
func LoadCacheIndexerHelmConfig() (map[string]*ReleaseConfig, error) {
	configDir, err := GetCacheIndexerConfigDir()
	if err != nil {
		return nil, err
	}
	configPath := filepath.Join(configDir, "helm_config.yaml")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("read cache-indexer helm config failed: %v", err)
	}
	var cfg map[string]*ReleaseConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse cache-indexer helm config yaml failed: %v", err)
	}
	for _, entry := range cfg {
		if entry == nil {
			continue
		}
		resolved, resolveErr := resolveHelmURL(configDir, entry.HelmURL)
		if resolveErr != nil {
			return nil, resolveErr
		}
		entry.HelmURL = resolved
	}
	return cfg, nil
}

// HelmInstall 安装 Helm Release，支持 --set 参数和 -f values 文件
// isLocalChartPath 判断 helmURL 是否为本地路径（非 OCI URL）。
func isLocalChartPath(helmURL string) bool {
	return helmURL != "" && !strings.HasPrefix(helmURL, "oci://")
}

// HelmInstall 安装 Helm Release，支持 --set 参数和 -f values 文件。
// 支持 OCI URL (oci://...) 和本地 chart 路径。
func HelmInstall(cfg *ReleaseConfig) error {
	cmd := fmt.Sprintf("helm install %s %s -n %s",
		cfg.ReleaseName, cfg.HelmURL, cfg.Namespace)

	// 仅 OCI URL 时传 --version，本地路径忽略 version
	if !isLocalChartPath(cfg.HelmURL) && strings.TrimSpace(cfg.Version) != "" && strings.TrimSpace(cfg.Version) != "latest" {
		cmd += fmt.Sprintf(" --version %s", cfg.Version)
	}

	// 如果指定了 values 文件，使用 -f 参数（优先级高于 --set）
	if cfg.ValuesFile != "" {
		cmd += fmt.Sprintf(" -f %s", cfg.ValuesFile)
	}

	// 追加 --set 参数
	for _, setValue := range cfg.SetValues {
		cmd += " " + setValue
	}

	// 打印将要执行的完整命令
	fmt.Printf("执行 Helm 安装命令: %s\n", cmd)

	_, stderr, err := execLocal(cmd)
	if err != nil {
		return fmt.Errorf("helm install %s failed: %w, stderr: %s", cfg.ReleaseName, err, stderr)
	}
	return nil
}

func shouldInstallFromPulledChart(cfg *ReleaseConfig) bool {
	if cfg == nil {
		return false
	}
	if !strings.HasPrefix(strings.TrimSpace(cfg.HelmURL), "oci://") {
		return false
	}
	version := strings.TrimSpace(cfg.Version)
	return version != "" && version != "latest"
}

func pullInstallableChart(cfg *ReleaseConfig) (string, func(), error) {
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("helm-chart-%s-", cfg.ReleaseName))
	if err != nil {
		return "", nil, fmt.Errorf("create temp helm chart dir failed: %w", err)
	}
	cleanup := func() {
		_ = os.RemoveAll(tempDir)
	}

	chartDir, err := PullHelmChart(cfg.HelmURL, cfg.Version, tempDir)
	if err != nil {
		cleanup()
		return "", nil, err
	}
	return chartDir, cleanup, nil
}

// HelmInstallWithLatestFallback 安装 Helm Release。
// 支持 OCI URL (oci://...) 和本地 chart 路径。
// 当 version=latest 或为空时不传 --version，避免 "improper constraint: latest"。
func HelmInstallWithLatestFallback(cfg *ReleaseConfig) error {
	if cfg == nil {
		return fmt.Errorf("release config is nil")
	}
	// 本地路径直接走 HelmInstall 逻辑（不走 --version 判断）
	if isLocalChartPath(cfg.HelmURL) {
		return HelmInstall(cfg)
	}
	if strings.TrimSpace(cfg.Version) != "" && strings.TrimSpace(cfg.Version) != "latest" {
		return HelmInstall(cfg)
	}

	cmd := fmt.Sprintf("helm install %s %s -n %s",
		cfg.ReleaseName, cfg.HelmURL, cfg.Namespace)
	if cfg.ValuesFile != "" {
		cmd += fmt.Sprintf(" -f %s", cfg.ValuesFile)
	}
	for _, setValue := range cfg.SetValues {
		cmd += " " + setValue
	}
	_, stderr, err := execLocal(cmd)
	if err != nil {
		return fmt.Errorf("helm install %s failed: %w, stderr: %s", cfg.ReleaseName, err, stderr)
	}
	return nil
}

// HelmUpgradeWithLatestFallback 升级 Helm Release。
// 支持 OCI URL (oci://...) 和本地 chart 路径。
// 当 version=latest 或为空时不传 --version。
func HelmUpgradeWithLatestFallback(cfg *ReleaseConfig) error {
	if cfg == nil {
		return fmt.Errorf("release config is nil")
	}
	if shouldInstallFromPulledChart(cfg) {
		chartDir, cleanup, err := pullInstallableChart(cfg)
		if err != nil {
			return fmt.Errorf("prepare pulled chart for %s failed: %w", cfg.ReleaseName, err)
		}
		defer cleanup()

		cmd := fmt.Sprintf("helm upgrade %s %s -n %s",
			cfg.ReleaseName, chartDir, cfg.Namespace)
		if cfg.ValuesFile != "" {
			cmd += fmt.Sprintf(" -f %s", cfg.ValuesFile)
		}
		for _, setValue := range cfg.SetValues {
			cmd += " " + setValue
		}
		_, stderr, err := execLocal(cmd)
		if err != nil {
			return fmt.Errorf("helm upgrade %s from pulled chart failed: %w, stderr: %s", cfg.ReleaseName, err, stderr)
		}
		return nil
	}
	cmd := fmt.Sprintf("helm upgrade %s %s -n %s",
		cfg.ReleaseName, cfg.HelmURL, cfg.Namespace)
	// 仅 OCI URL 时传 --version，本地路径忽略 version
	if !isLocalChartPath(cfg.HelmURL) && strings.TrimSpace(cfg.Version) != "" && strings.TrimSpace(cfg.Version) != "latest" {
		cmd += fmt.Sprintf(" --version %s", cfg.Version)
	}
	if cfg.ValuesFile != "" {
		cmd += fmt.Sprintf(" -f %s", cfg.ValuesFile)
	}
	for _, setValue := range cfg.SetValues {
		cmd += " " + setValue
	}
	_, stderr, err := execLocal(cmd)
	if err != nil {
		return fmt.Errorf("helm upgrade %s failed: %w, stderr: %s", cfg.ReleaseName, err, stderr)
	}
	return nil
}

// CleanupLegacyReleaseConflictsForInfernex 统一清理 Infernex 安装前已知冲突（best effort）。
// 先执行通用的 Helm ownership 冲突处理，再清理历史遗留资源。
func CleanupLegacyReleaseConflictsForInfernex(ctx context.Context, k8sClient K8SClientInterface, releaseName, targetNamespace string) error {
	if err := CleanupConflictsForInfernex(ctx, k8sClient, releaseName, targetNamespace); err != nil {
		return err
	}
	_, _, _ = execLocal(fmt.Sprintf("helm uninstall %s -n ai-inference || true", releaseName))
	_, _, _ = execLocal(fmt.Sprintf("helm uninstall %s -n ai-hms || true", releaseName))
	_, _, _ = execLocal("helm uninstall elastic-scaler -n scaling-system || true")
	_, _, _ = execLocal("kubectl -n istio-system delete sa istio-reader-service-account --ignore-not-found=true")
	_, _, _ = execLocal("kubectl -n scaling-system delete sa tidal-manager --ignore-not-found=true")
	_, _, _ = execLocal("kubectl delete clusterrole tidal-controller-manager-manager-role --ignore-not-found=true")
	return nil
}

// HelmUninstall 卸载 Helm Release
func HelmUninstall(cfg *ReleaseConfig) error {
	cmd := fmt.Sprintf("helm uninstall %s -n %s", cfg.ReleaseName, cfg.Namespace)
	_, stderr, err := execLocal(cmd)
	if err != nil {
		return fmt.Errorf("helm uninstall %s failed: %w, stderr: %s", cfg.ReleaseName, err, stderr)
	}
	return nil
}

// HelmUninstallIfExists 若 release 存在则卸载（不存在不报错）
func HelmUninstallIfExists(releaseName, namespace string) error {
	cmd := fmt.Sprintf("helm list -n %s -o json", namespace)
	stdout, stderr, err := execLocal(cmd)
	if err != nil {
		return fmt.Errorf("helm list failed: %w, stderr: %s", err, stderr)
	}

	var helmList []map[string]any
	if err := json.Unmarshal([]byte(stdout), &helmList); err != nil {
		return fmt.Errorf("parse helm list json failed: %v", err)
	}

	found := false
	for _, item := range helmList {
		if name, ok := item["name"].(string); ok && name == releaseName {
			found = true
			break
		}
	}
	if !found {
		return nil
	}

	cmd = fmt.Sprintf("helm uninstall %s -n %s", releaseName, namespace)
	_, stderr, err = execLocal(cmd)
	if err != nil {
		return fmt.Errorf("helm uninstall %s failed: %w, stderr: %s", releaseName, err, stderr)
	}
	return nil
}

// HelmResource 记录 helm manifest 中的一个 Kubernetes 资源。
type HelmResource struct {
	Kind      string
	Name      string
	Namespace string
}

func helmDriverEnv() []string {
	return append(os.Environ(), "HELM_DRIVER=configmap")
}

// ListHelmReleasesInNamespace 列出命名空间中所有 helm release 名称。
func ListHelmReleasesInNamespace(namespace string) ([]string, error) {
	cmd := exec.Command("helm", "list", "-n", namespace, "-o", "json")
	cmd.Env = helmDriverEnv()
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("helm list -n %s: %w", namespace, err)
	}
	var list []map[string]any
	if err := json.Unmarshal(out, &list); err != nil {
		return nil, fmt.Errorf("parse helm list: %w", err)
	}
	var names []string
	for _, item := range list {
		if name, ok := item["name"].(string); ok && name != "" {
			names = append(names, name)
		}
	}
	return names, nil
}

// ListHelmReleasesInNamespaceIfExists 命名空间存在时列出 helm release；不存在时返回 nil, nil。
func ListHelmReleasesInNamespaceIfExists(ctx context.Context, k8sClient K8SClientInterface, namespace string) ([]string, error) {
	if !NamespaceExists(ctx, k8sClient, namespace) {
		return nil, nil
	}
	return ListHelmReleasesInNamespace(namespace)
}

// HelmReleaseExists 检查 helm release 是否存在。
func HelmReleaseExists(releaseName, namespace string) bool {
	releases, err := ListHelmReleasesInNamespace(namespace)
	if err != nil {
		return false
	}
	for _, name := range releases {
		if name == releaseName {
			return true
		}
	}
	return false
}

// HelmGetManifest 获取 helm release 部署的资源清单。
func HelmGetManifest(releaseName, namespace string) ([]HelmResource, error) {
	cmd := exec.Command("helm", "get", "manifest", releaseName, "-n", namespace)
	cmd.Env = helmDriverEnv()
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("helm get manifest failed: %w", err)
	}
	return ParseHelmManifestResources(string(out))
}

// ParseHelmManifestResources 解析 helm manifest YAML，提取每个资源的 Kind/Name/Namespace。
func ParseHelmManifestResources(manifest string) ([]HelmResource, error) {
	var resources []HelmResource

	docs := strings.Split(manifest, "---\n")
	for _, doc := range docs {
		doc = strings.TrimSpace(doc)
		if doc == "" {
			continue
		}

		var parsed struct {
			Kind     string `yaml:"kind"`
			Metadata struct {
				Name      string `yaml:"name"`
				Namespace string `yaml:"namespace"`
			} `yaml:"metadata"`
		}
		if err := yaml.Unmarshal([]byte(doc), &parsed); err != nil {
			continue
		}
		if parsed.Kind == "" || parsed.Metadata.Name == "" {
			continue
		}

		resources = append(resources, HelmResource{
			Kind:      parsed.Kind,
			Name:      parsed.Metadata.Name,
			Namespace: parsed.Metadata.Namespace,
		})
	}

	return resources, nil
}

// ResourceBelongsToHelmRelease 判断资源是否由指定 helm release 创建（用于共享命名空间卸载校验）。
func ResourceBelongsToHelmRelease(labels, annotations map[string]string, releaseName string) bool {
	if releaseName == "" {
		return false
	}
	if labels != nil {
		if labels["meta.helm.sh/release-name"] == releaseName {
			return true
		}
		if labels["app.kubernetes.io/instance"] == releaseName {
			return true
		}
	}
	if annotations != nil && annotations["meta.helm.sh/release-name"] == releaseName {
		return true
	}
	return false
}

// CheckHelmRelease 判断 release 是否存在
func CheckHelmRelease(cfg *ReleaseConfig) (bool, error) {
	cmd := fmt.Sprintf("helm list -n %s -o json", cfg.Namespace)
	stdout, stderr, err := execLocal(cmd)
	if err != nil {
		return false, fmt.Errorf("helm list failed: %w, stderr: %s", err, stderr)
	}

	var helmList []map[string]any
	if err := json.Unmarshal([]byte(stdout), &helmList); err != nil {
		return false, fmt.Errorf("parse helm list json failed: %v", err)
	}

	for _, item := range helmList {
		if name, ok := item["name"].(string); ok && name == cfg.ReleaseName {
			return true, nil
		}
	}
	return false, nil
}

// CleanupTempValuesFile 清理临时 values 文件
func CleanupTempValuesFile(valuesFile string) error {
	if valuesFile == "" {
		return nil
	}
	if err := os.Remove(valuesFile); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("删除临时文件失败: %w", err)
	}
	return nil
}

// BuildValuesFromLocalAndEPP 以本地 base values 文件为底，用 EPP 片段文件中的 pluginsConfigFile 与 pluginsCustomConfig
// 覆盖 hermes-router.inferenceExtension（与 hermes-router-config/values.yaml 结构一致），返回合并后的临时 values 文件路径。
func BuildValuesFromLocalAndEPP(localBasePath, eppFilePath string) (string, error) {
	data, err := os.ReadFile(eppFilePath)
	if err != nil {
		return "", fmt.Errorf("读取 EPP 文件失败: %w", err)
	}
	return BuildValuesFromLocalAndEPPContent(localBasePath, string(data))
}

// BuildValuesFromLocalAndEPPContent 与 BuildValuesFromLocalAndEPP 相同，但 EPP 配置由字符串 eppYAMLContent 提供，
// 便于测试中传入内存修改后的内容（如删除 plugins、修改权重等）。
func BuildValuesFromLocalAndEPPContent(localBasePath, eppYAMLContent string) (string, error) {
	baseData, err := os.ReadFile(localBasePath)
	if err != nil {
		return "", fmt.Errorf("读取本地 base values 失败: %w", err)
	}
	var baseValues map[string]any
	if err := yaml.Unmarshal(baseData, &baseValues); err != nil {
		return "", fmt.Errorf("解析 base values yaml 失败: %v", err)
	}
	var eppExt map[string]any
	if err := yaml.Unmarshal([]byte(eppYAMLContent), &eppExt); err != nil {
		return "", fmt.Errorf("解析 EPP 片段 yaml 失败: %v", err)
	}
	// 只保留 EPP 中的 pluginsConfigFile 与 pluginsCustomConfig，作为 hermes-router.inferenceExtension 的覆盖
	overrideExt := make(map[string]any)
	if v, ok := eppExt["pluginsConfigFile"]; ok {
		overrideExt["pluginsConfigFile"] = v
	}
	if v, ok := eppExt["pluginsCustomConfig"]; ok {
		overrideExt["pluginsCustomConfig"] = deepCopyValue(v)
	}
	overrideValues := map[string]any{
		"hermes-router": map[string]any{
			"inferenceExtension": overrideExt,
		},
	}
	merged := deepMergeMaps(baseValues, overrideValues)
	yamlData, err := yaml.Marshal(merged)
	if err != nil {
		return "", fmt.Errorf("marshal 合并后的 values yaml 失败: %v", err)
	}
	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}
	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}
	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-local-epp-%d.yaml", time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}
	return tempFile, nil
}

// BuildValuesFromLocalEPPAndInferenceBackend 以本地 base + EPP 片段生成「base + aggregate EPP」的 values，
// 再用 inferenceBackendFilePath 中的 services 列表整体替换 inference-backend.services，返回合并后的临时 values 文件路径。
// 用于后端副本数测试：aggregate 路由 + aggregate 推理后端。
func BuildValuesFromLocalEPPAndInferenceBackend(localBasePath, eppFilePath, inferenceBackendFilePath string) (string, error) {
	baseWithEPP, err := BuildValuesFromLocalAndEPP(localBasePath, eppFilePath)
	if err != nil {
		return "", err
	}
	defer func() { _ = CleanupTempValuesFile(baseWithEPP) }()

	data, err := os.ReadFile(baseWithEPP)
	if err != nil {
		return "", fmt.Errorf("读取合并后的 values 失败: %w", err)
	}
	var merged map[string]any
	if err := yaml.Unmarshal(data, &merged); err != nil {
		return "", fmt.Errorf("解析合并后的 values yaml 失败: %v", err)
	}

	backendData, err := os.ReadFile(inferenceBackendFilePath)
	if err != nil {
		return "", fmt.Errorf("读取 inference-backend 片段失败: %w", err)
	}
	var backendExt map[string]any
	if err := yaml.Unmarshal(backendData, &backendExt); err != nil {
		return "", fmt.Errorf("解析 inference-backend 片段 yaml 失败: %v", err)
	}
	services, ok := backendExt["services"]
	if !ok {
		return "", fmt.Errorf("inference-backend 片段中缺少 services 键")
	}
	servicesList, ok := services.([]any)
	if !ok {
		return "", fmt.Errorf("inference-backend 片段中 services 不是列表")
	}

	ib, _ := merged["inference-backend"].(map[string]any)
	if ib == nil {
		ib = make(map[string]any)
		merged["inference-backend"] = ib
	}
	ib["services"] = deepCopyValue(servicesList)

	yamlData, err := yaml.Marshal(merged)
	if err != nil {
		return "", fmt.Errorf("marshal 合并后的 values yaml 失败: %v", err)
	}
	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}
	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}
	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-nbackend-base-%d.yaml", time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}
	return tempFile, nil
}

// SetAggregatedReplicasInValues 读取 values 文件，将 inference-backend.services[0].aggregated.replicas 设为 replicas，写入新临时文件并返回路径。
func SetAggregatedReplicasInValues(valuesFilePath string, replicas int) (string, error) {
	data, err := os.ReadFile(valuesFilePath)
	if err != nil {
		return "", fmt.Errorf("读取 values 文件失败: %w", err)
	}
	var values map[string]any
	if err := yaml.Unmarshal(data, &values); err != nil {
		return "", fmt.Errorf("解析 values yaml 失败: %v", err)
	}
	ib, ok := values["inference-backend"].(map[string]any)
	if !ok || ib == nil {
		return "", fmt.Errorf("values 中缺少 inference-backend 或类型不是 map")
	}
	servicesRaw, ok := ib["services"]
	if !ok {
		return "", fmt.Errorf("inference-backend 中缺少 services")
	}
	services, ok := servicesRaw.([]any)
	if !ok || len(services) == 0 {
		return "", fmt.Errorf("inference-backend.services 为空或不是列表")
	}
	first, ok := services[0].(map[string]any)
	if !ok {
		return "", fmt.Errorf("inference-backend.services[0] 不是 map")
	}
	aggregated, _ := first["aggregated"].(map[string]any)
	if aggregated == nil {
		aggregated = make(map[string]any)
		first["aggregated"] = aggregated
	}
	aggregated["replicas"] = replicas

	yamlData, err := yaml.Marshal(values)
	if err != nil {
		return "", fmt.Errorf("marshal 修改后的 values yaml 失败: %v", err)
	}
	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}
	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}
	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-nbackend-replicas-%d-%d.yaml", replicas, time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}
	return tempFile, nil
}

// deepCopyValue 深拷贝一个值（支持 map、slice 和基本类型）
func deepCopyValue(v any) any {
	switch val := v.(type) {
	case map[string]any:
		result := make(map[string]any)
		for k, v := range val {
			result[k] = deepCopyValue(v)
		}
		return result
	case []any:
		result := make([]any, len(val))
		for i, v := range val {
			result[i] = deepCopyValue(v)
		}
		return result
	default:
		return v
	}
}

// setNestedField 根据点号分隔的路径设置嵌套字段的值
// 如果路径不存在，会创建中间结构
func setNestedField(item map[string]any, fieldPath string, value any) {
	parts := strings.Split(fieldPath, ".")
	if len(parts) == 1 {
		// 简单字段，直接设置
		item[fieldPath] = value
		return
	}

	// 嵌套字段，需要遍历路径
	current := item
	for i := 0; i < len(parts)-1; i++ {
		part := parts[i]
		if next, ok := current[part]; ok {
			if nextMap, ok := next.(map[string]any); ok {
				current = nextMap
			} else {
				// 如果类型不匹配，创建新的 map
				current[part] = make(map[string]any)
				current = current[part].(map[string]any)
			}
		} else {
			// 路径不存在，创建新的 map
			current[part] = make(map[string]any)
			current = current[part].(map[string]any)
		}
	}
	// 设置最后一个字段
	current[parts[len(parts)-1]] = value
}

// BuildValuesFileForMultipleFields 构建 values 文件来修改列表中的多个字段（包括嵌套字段）
// fieldUpdates: 字段路径到新值的映射，例如：
//   - "enabled" -> false
//   - "name" -> "vllm-pd-1p1d"
//   - "pd.prefill.tensorParallelSize" -> 1
//   - "pd.decode.tensorParallelSize" -> 1
//   - "openfuyao.pdGroupID" -> "qwen3-8b-pd-01"
func BuildValuesFileForMultipleFields(helmURL, version, listPath, nameField, nameValue string, fieldUpdates map[string]any) (string, error) {
	// 1. 执行 helm show values 命令获取默认值
	cmd := fmt.Sprintf("helm show values %s --version %s", helmURL, version)
	stdout, stderr, err := execLocal(cmd)
	if err != nil {
		return "", fmt.Errorf("helm show values failed: %w, stderr: %s", err, stderr)
	}

	// 2. 解析 YAML
	var values map[string]any
	if err := yaml.Unmarshal([]byte(stdout), &values); err != nil {
		return "", fmt.Errorf("parse values yaml failed: %v", err)
	}

	// 3. 查找列表路径并匹配 name
	pathParts := strings.Split(listPath, ".")
	var current any = values
	for _, part := range pathParts {
		if m, ok := current.(map[string]any); ok {
			current = m[part]
		} else {
			return "", fmt.Errorf("path %s is not a map at %s", listPath, part)
		}
	}

	// 4. 检查是否为列表
	list, ok := current.([]any)
	if !ok {
		return "", fmt.Errorf("path %s is not a list", listPath)
	}

	// 5. 查找匹配 name 的项
	index := -1
	for i, item := range list {
		if im, ok := item.(map[string]any); ok {
			if fieldValue, exists := im[nameField]; exists {
				if fmt.Sprintf("%v", fieldValue) == nameValue {
					index = i
					break
				}
			}
		}
	}

	if index == -1 {
		return "", fmt.Errorf("no item found with %s=%s in list %s", nameField, nameValue, listPath)
	}

	// 6. 构建 values 文件：复制整个列表，只修改目标项的指定字段
	servicesList := make([]map[string]any, len(list))
	for i, item := range list {
		if im, ok := item.(map[string]any); ok {
			// 深拷贝整个 item
			newItem := deepCopyValue(im).(map[string]any)
			// 如果是目标项，修改指定字段
			if i == index {
				// 应用所有字段更新
				for fieldPath, newValue := range fieldUpdates {
					// 类型转换
					typedValue := newValue
					// 对于数字字符串，尝试转换为整数
					if strVal, ok := newValue.(string); ok {
						var intVal int
						if _, err := fmt.Sscanf(strVal, "%d", &intVal); err == nil {
							typedValue = intVal
						} else if strVal == "true" || strVal == "false" {
							// 布尔值字符串
							typedValue = (strVal == "true")
						}
					}
					setNestedField(newItem, fieldPath, typedValue)
				}
			}
			servicesList[i] = newItem
		}
	}

	// 7. 构建最终的 values 结构（只包含需要修改的路径部分）
	overrideValues := make(map[string]any)

	// 构建嵌套结构：inference-backend.services
	parts := strings.Split(listPath, ".")
	currentMap := overrideValues
	for i, part := range parts {
		if i == len(parts)-1 {
			// 最后一个部分是列表名
			currentMap[part] = servicesList
		} else {
			// 中间部分创建嵌套 map
			newMap := make(map[string]any)
			currentMap[part] = newMap
			currentMap = newMap
		}
	}

	// 8. 将 values 转换为 YAML
	yamlData, err := yaml.Marshal(overrideValues)
	if err != nil {
		return "", fmt.Errorf("marshal values yaml failed: %v", err)
	}

	// 9. 创建临时文件
	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}

	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}

	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-%s-multi-%d.yaml", nameValue, index))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}

	return tempFile, nil
}

// BuildValuesFileForMultipleServices 构建 values 文件来同时修改多个 service
// serviceUpdates: map[string]map[string]any，key 是 service 的 name，value 是该 service 的字段更新
func BuildValuesFileForMultipleServices(helmURL, version, listPath, nameField string, serviceUpdates map[string]map[string]any) (string, error) {
	// 1. 执行 helm show values 命令获取默认值
	cmd := fmt.Sprintf("helm show values %s --version %s", helmURL, version)
	stdout, stderr, err := execLocal(cmd)
	if err != nil {
		return "", fmt.Errorf("helm show values failed: %w, stderr: %s", err, stderr)
	}

	// 2. 解析 YAML
	var values map[string]any
	if err := yaml.Unmarshal([]byte(stdout), &values); err != nil {
		return "", fmt.Errorf("parse values yaml failed: %v", err)
	}

	// 3. 查找列表路径
	pathParts := strings.Split(listPath, ".")
	var current any = values
	for _, part := range pathParts {
		if m, ok := current.(map[string]any); ok {
			current = m[part]
		} else {
			return "", fmt.Errorf("path %s is not a map at %s", listPath, part)
		}
	}

	// 4. 检查是否为列表
	list, ok := current.([]any)
	if !ok {
		return "", fmt.Errorf("path %s is not a list", listPath)
	}

	// 5. 构建 values 文件：复制整个列表，修改所有目标项的指定字段
	servicesList := make([]map[string]any, len(list))
	for i, item := range list {
		if im, ok := item.(map[string]any); ok {
			// 深拷贝整个 item
			newItem := deepCopyValue(im).(map[string]any)
			// 检查是否是目标 service
			if name, exists := newItem[nameField]; exists {
				nameStr := fmt.Sprintf("%v", name)
				if fieldUpdates, found := serviceUpdates[nameStr]; found {
					// 应用所有字段更新
					for fieldPath, newValue := range fieldUpdates {
						// 类型转换
						typedValue := newValue
						// 对于数字字符串，尝试转换为整数
						if strVal, ok := newValue.(string); ok {
							var intVal int
							if _, err := fmt.Sscanf(strVal, "%d", &intVal); err == nil {
								typedValue = intVal
							} else if strVal == "true" || strVal == "false" {
								// 布尔值字符串
								typedValue = (strVal == "true")
							}
						}
						setNestedField(newItem, fieldPath, typedValue)
					}
				}
			}
			servicesList[i] = newItem
		}
	}

	// 6. 构建最终的 values 结构（只包含需要修改的路径部分）
	overrideValues := make(map[string]any)

	// 构建嵌套结构：inference-backend.services
	parts := strings.Split(listPath, ".")
	currentMap := overrideValues
	for i, part := range parts {
		if i == len(parts)-1 {
			// 最后一个部分是列表名
			currentMap[part] = servicesList
		} else {
			// 中间部分创建嵌套 map
			newMap := make(map[string]any)
			currentMap[part] = newMap
			currentMap = newMap
		}
	}

	// 7. 将 values 转换为 YAML
	yamlData, err := yaml.Marshal(overrideValues)
	if err != nil {
		return "", fmt.Errorf("marshal values yaml failed: %v", err)
	}

	// 8. 创建临时文件
	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}

	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}

	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-multi-services-%d.yaml", len(serviceUpdates)))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}

	return tempFile, nil
}

// BuildValuesFileWithServiceCopy 构建 values 文件，复制指定的 service 配置并修改
// 用于创建多个相同配置但不同 name 和 pdGroupID 的 service
// sourceName: 源 service 的 name
// targetServices: 目标 service 配置列表，每个包含 name 和 pdGroupID
func BuildValuesFileWithServiceCopy(helmURL, version, listPath, nameField, sourceName string, targetServices []map[string]any) (string, error) {
	// 1. 执行 helm show values 命令获取默认值
	cmd := fmt.Sprintf("helm show values %s --version %s", helmURL, version)
	stdout, stderr, err := execLocal(cmd)
	if err != nil {
		return "", fmt.Errorf("helm show values failed: %w, stderr: %s", err, stderr)
	}

	// 2. 解析 YAML
	var values map[string]any
	if err := yaml.Unmarshal([]byte(stdout), &values); err != nil {
		return "", fmt.Errorf("parse values yaml failed: %v", err)
	}

	// 3. 查找列表路径并匹配 name
	pathParts := strings.Split(listPath, ".")
	var current any = values
	for _, part := range pathParts {
		if m, ok := current.(map[string]any); ok {
			current = m[part]
		} else {
			return "", fmt.Errorf("path %s is not a map at %s", listPath, part)
		}
	}

	// 4. 检查是否为列表
	list, ok := current.([]any)
	if !ok {
		return "", fmt.Errorf("path %s is not a list", listPath)
	}

	// 5. 查找匹配 name 的源 service
	sourceIndex := -1
	var sourceItemMap map[string]any
	for i, item := range list {
		if im, ok := item.(map[string]any); ok {
			if fieldValue, exists := im[nameField]; exists {
				if fmt.Sprintf("%v", fieldValue) == sourceName {
					sourceIndex = i
					sourceItemMap = im
					break
				}
			}
		}
	}

	if sourceIndex == -1 {
		return "", fmt.Errorf("no item found with %s=%s in list %s", nameField, sourceName, listPath)
	}

	// 6. 构建新的 services 列表
	// 首先保留所有非源 service 的项（但禁用它们，除非是目标 service）
	servicesList := make([]map[string]any, 0, len(list)+len(targetServices))

	// 6.1 处理原有列表中的项
	for i, item := range list {
		if im, ok := item.(map[string]any); ok {
			// 如果是源 service，跳过（后面会用复制的版本替换）
			if i == sourceIndex {
				continue
			}
			// 深拷贝并禁用（除非是目标 service 之一）
			newItem := deepCopyValue(im).(map[string]any)
			// 检查是否是目标 service（通过 name 匹配）
			isTarget := false
			if name, exists := newItem["name"]; exists {
				for _, target := range targetServices {
					if targetName, ok := target["name"].(string); ok && fmt.Sprintf("%v", name) == targetName {
						isTarget = true
						break
					}
				}
			}
			// 如果不是目标 service，禁用它
			if !isTarget {
				newItem["enabled"] = false
			}
			servicesList = append(servicesList, newItem)
		}
	}

	// 6.2 为每个目标 service 创建复制的配置
	for _, target := range targetServices {
		// 深拷贝源 service 配置
		newService := deepCopyValue(sourceItemMap).(map[string]any)

		// 应用目标配置中的字段更新
		for fieldPath, value := range target {
			// 类型转换
			typedValue := value
			if strVal, ok := value.(string); ok {
				var intVal int
				if _, err := fmt.Sscanf(strVal, "%d", &intVal); err == nil {
					typedValue = intVal
				} else if strVal == "true" || strVal == "false" {
					typedValue = (strVal == "true")
				}
			}
			setNestedField(newService, fieldPath, typedValue)
		}

		// 确保 enabled 为 true
		newService["enabled"] = true

		servicesList = append(servicesList, newService)
	}

	// 7. 构建最终的 values 结构（只包含需要修改的路径部分）
	overrideValues := make(map[string]any)

	// 构建嵌套结构：inference-backend.services
	parts := strings.Split(listPath, ".")
	currentMap := overrideValues
	for i, part := range parts {
		if i == len(parts)-1 {
			// 最后一个部分是列表名
			currentMap[part] = servicesList
		} else {
			// 中间部分创建嵌套 map
			newMap := make(map[string]any)
			currentMap[part] = newMap
			currentMap = newMap
		}
	}

	// 8. 将 values 转换为 YAML
	yamlData, err := yaml.Marshal(overrideValues)
	if err != nil {
		return "", fmt.Errorf("marshal values yaml failed: %v", err)
	}

	// 9. 创建临时文件
	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}

	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}

	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-%s-copy-%d.yaml", sourceName, len(targetServices)))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}

	return tempFile, nil
}

// deleteNestedField 根据点号分隔的路径删除嵌套字段
// 如果路径不存在，则不执行任何操作
func deleteNestedField(item map[string]any, fieldPath string) {
	parts := strings.Split(fieldPath, ".")
	if len(parts) == 1 {
		// 简单字段，直接删除
		delete(item, fieldPath)
		return
	}

	// 嵌套字段，需要遍历路径
	current := item
	for i := 0; i < len(parts)-1; i++ {
		part := parts[i]
		if next, ok := current[part]; ok {
			if nextMap, ok := next.(map[string]any); ok {
				current = nextMap
			} else {
				// 路径不存在，直接返回
				return
			}
		} else {
			// 路径不存在，直接返回
			return
		}
	}
	// 删除最后一个字段
	delete(current, parts[len(parts)-1])
}

// BuildValuesFileForNestedMap 构建 values 文件来修改嵌套 map 结构中的字段
// 用于修改非列表类型的嵌套配置，如 hermes-router.inferencepool.inferencePool.modelServers.matchLabels
// fieldPath: 点号分隔的字段路径，如 "inferencepool.inferencePool.modelServers.matchLabels"
// fieldKey: map 中的键名，如 "openfuyao.com/model"（可能包含点号，所以需要单独指定）
// fieldValue: 要设置的值
func BuildValuesFileForNestedMap(helmURL, version, rootPath, fieldPath, fieldKey string, fieldValue any) (string, error) {
	// 1. 执行 helm show values 命令获取默认值
	cmd := fmt.Sprintf("helm show values %s --version %s", helmURL, version)
	stdout, stderr, err := execLocal(cmd)
	if err != nil {
		return "", fmt.Errorf("helm show values failed: %w, stderr: %s", err, stderr)
	}

	// 2. 解析 YAML
	var values map[string]any
	if err := yaml.Unmarshal([]byte(stdout), &values); err != nil {
		return "", fmt.Errorf("parse values yaml failed: %v", err)
	}

	// 3. 查找根路径
	rootParts := strings.Split(rootPath, ".")
	var rootCurrent any = values
	for _, part := range rootParts {
		if m, ok := rootCurrent.(map[string]any); ok {
			rootCurrent = m[part]
		} else {
			return "", fmt.Errorf("path %s is not a map at %s", rootPath, part)
		}
	}

	// 4. 确保 rootCurrent 是 map
	rootMap, ok := rootCurrent.(map[string]any)
	if !ok {
		return "", fmt.Errorf("path %s is not a map", rootPath)
	}

	// 5. 深拷贝 rootMap
	newRootMap := deepCopyValue(rootMap).(map[string]any)

	// 6. 查找并修改目标字段
	// 先找到 fieldPath 指向的 map
	fieldParts := strings.Split(fieldPath, ".")
	current := newRootMap
	for i := 0; i < len(fieldParts); i++ {
		part := fieldParts[i]
		if next, ok := current[part]; ok {
			if nextMap, ok := next.(map[string]any); ok {
				current = nextMap
			} else {
				// 如果类型不匹配，创建新的 map
				current[part] = make(map[string]any)
				current = current[part].(map[string]any)
			}
		} else {
			// 路径不存在，创建新的 map
			current[part] = make(map[string]any)
			current = current[part].(map[string]any)
		}
	}

	// 7. 设置目标键的值
	current[fieldKey] = fieldValue

	// 8. 构建最终的 values 结构（只包含需要修改的路径部分）
	overrideValues := make(map[string]any)

	// 构建嵌套结构
	parts := strings.Split(rootPath, ".")
	currentMap := overrideValues
	for i, part := range parts {
		if i == len(parts)-1 {
			// 最后一个部分是根路径名
			currentMap[part] = newRootMap
		} else {
			// 中间部分创建嵌套 map
			newMap := make(map[string]any)
			currentMap[part] = newMap
			currentMap = newMap
		}
	}

	// 9. 将 values 转换为 YAML
	yamlData, err := yaml.Marshal(overrideValues)
	if err != nil {
		return "", fmt.Errorf("marshal values yaml failed: %v", err)
	}

	// 10. 创建临时文件
	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}

	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}

	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-nested-%s.yaml", strings.ReplaceAll(fieldPath, ".", "-")))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}

	return tempFile, nil
}

const eppPluginsConfigKey = "epp-aggregate-kv-cache-aware.yaml"
const pickerMinRandomPlugin = "picker-min-random"

// BuildValuesFileForHermesRouterEPPPluginsWithoutPicker 获取 hermes-router chart 的 values，保留完整
// inferencepool 段，仅将 inferencepool.inferenceExtension.pluginsCustomConfig 中 epp 文件内容里的
// plugins 列表中的 picker-min-random 删除（不删 schedulingProfiles[].plugins），生成 override 临时文件并返回路径。
func BuildValuesFileForHermesRouterEPPPluginsWithoutPicker(helmURL, version string) (string, error) {
	cmd := fmt.Sprintf("helm show values %s --version %s", helmURL, version)
	stdout, stderr, err := execLocal(cmd)
	if err != nil {
		return "", fmt.Errorf("helm show values failed: %w, stderr: %s", err, stderr)
	}

	var rootValues map[string]any
	if err := yaml.Unmarshal([]byte(stdout), &rootValues); err != nil {
		return "", fmt.Errorf("parse values yaml failed: %v", err)
	}

	values := rootValues
	for _, seg := range []string{"inferencepool", "inferenceExtension", "pluginsCustomConfig"} {
		next, ok := values[seg]
		if !ok {
			return "", fmt.Errorf("values path missing key %s", seg)
		}
		var okMap bool
		values, okMap = next.(map[string]any)
		if !okMap {
			return "", fmt.Errorf("values path %s is not a map", seg)
		}
	}
	pluginsCustomConfigMap := values

	configKey := eppPluginsConfigKey
	rawStr, ok := pluginsCustomConfigMap[configKey]
	if !ok {
		for k, v := range pluginsCustomConfigMap {
			if strings.Contains(k, "epp-aggregate") {
				configKey = k
				rawStr = v
				break
			}
		}
	}
	if rawStr == nil {
		return "", fmt.Errorf("pluginsCustomConfig missing key %s", eppPluginsConfigKey)
	}

	contentStr, ok := rawStr.(string)
	if !ok {
		return "", fmt.Errorf("pluginsCustomConfig[%s] is not a string", configKey)
	}

	var inner map[string]any
	if err := yaml.Unmarshal([]byte(contentStr), &inner); err != nil {
		return "", fmt.Errorf("parse epp plugin config yaml failed: %v", err)
	}

	plugins, _ := inner["plugins"]
	pluginsList, ok := plugins.([]any)
	if !ok {
		return "", fmt.Errorf("epp config plugins is not a list")
	}

	newPlugins := make([]any, 0, len(pluginsList))
	for _, p := range pluginsList {
		if m, ok := p.(map[string]any); ok {
			if t, _ := m["type"].(string); t == pickerMinRandomPlugin {
				continue
			}
		}
		newPlugins = append(newPlugins, p)
	}
	inner["plugins"] = newPlugins

	newContent, err := yaml.Marshal(inner)
	if err != nil {
		return "", fmt.Errorf("marshal epp plugin config yaml failed: %v", err)
	}

	// 使用完整 inferencepool（与 chart 一致），仅替换 pluginsCustomConfig 中对应键
	inferencepool := deepCopyValue(rootValues["inferencepool"]).(map[string]any)
	infExt, _ := inferencepool["inferenceExtension"].(map[string]any)
	pluginsCustomConfigNew := make(map[string]any)
	for k, v := range pluginsCustomConfigMap {
		if k == configKey {
			pluginsCustomConfigNew[k] = string(newContent)
		} else {
			pluginsCustomConfigNew[k] = v
		}
	}
	infExt["pluginsCustomConfig"] = pluginsCustomConfigNew
	inferencepool["inferenceExtension"] = infExt
	overrideValues := map[string]any{"inferencepool": inferencepool}

	yamlData, err := yaml.Marshal(overrideValues)
	if err != nil {
		return "", fmt.Errorf("marshal override values yaml failed: %v", err)
	}

	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}
	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}
	tempFile := filepath.Join(tempDir, "values-hermes-router-epp-without-picker.yaml")
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}
	return tempFile, nil
}

// eppPDKVCacheAwareYAML 为 hermes-router PD 模式 EPP 配置（epp-pd-kv-cache-aware.yaml）的默认内容
const eppPDKVCacheAwareYAML = `apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: filter-by-pd-label
  parameters:
    pdLabelName: openfuyao.com/pdRole
    pdGroupLabelName: openfuyao.com/pdGroupID
    prefillValue: prefill
    decodeValue: decode
    leaderValue: leader
- type: scorer-pd-kv-cache-aware
  parameters:
    kvCacheHitNotRateWeight: 1.0
    xpuCacheUsageWeight: 1.0
    prefillWaitingRequestWeight: 1.0
    decodeWaitingRequestWeight: 1.0
    prefillPodScoreWeight: 1.0
    decodePodScoreWeight: 1.0
    kvCacheManagerIP: cache-indexer-service
    kvCacheManagerPort: 8080
    kvCacheManagerPath: /match_sort
    kvCacheManagerTimeout: 10
- type: picker-pd-kv-cache-aware
- type: pd-header-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: filter-by-pd-label
  - pluginRef: scorer-pd-kv-cache-aware
  - pluginRef: picker-pd-kv-cache-aware
  - pluginRef: pd-header-handler
`

// eppPDKVCacheAwareWithoutPickerYAML 为 PD 模式 EPP 配置去掉 picker-pd-kv-cache-aware 后的内容（plugins 与 schedulingProfiles.plugins 均不包含 picker），缩进与键序与 eppPDKVCacheAwareYAML 一致。
const eppPDKVCacheAwareWithoutPickerYAML = `apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: filter-by-pd-label
  parameters:
    pdLabelName: openfuyao.com/pdRole
    pdGroupLabelName: openfuyao.com/pdGroupID
    prefillValue: prefill
    decodeValue: decode
    leaderValue: leader
- type: scorer-pd-kv-cache-aware
  parameters:
    kvCacheHitNotRateWeight: 1.0
    xpuCacheUsageWeight: 1.0
    prefillWaitingRequestWeight: 1.0
    decodeWaitingRequestWeight: 1.0
    prefillPodScoreWeight: 1.0
    decodePodScoreWeight: 1.0
    kvCacheManagerIP: cache-indexer-service
    kvCacheManagerPort: 8080
    kvCacheManagerPath: /match_sort
    kvCacheManagerTimeout: 10
- type: pd-header-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: filter-by-pd-label
  - pluginRef: scorer-pd-kv-cache-aware
  - pluginRef: picker-pd-kv-cache-aware
  - pluginRef: pd-header-handler
`

// BuildValuesFileForHermesRouterEPPPDConfig 获取 hermes-router chart 的 values，保留完整 inferencepool 段，
// 仅将 inferencepool.inferenceExtension 的 pluginsConfigFile 与 pluginsCustomConfig 改为 PD 模式
// （epp-pd-kv-cache-aware.yaml），生成 override 临时文件并返回路径。
func BuildValuesFileForHermesRouterEPPPDConfig(helmURL, version string) (string, error) {
	cmd := fmt.Sprintf("helm show values %s --version %s", helmURL, version)
	stdout, stderr, err := execLocal(cmd)
	if err != nil {
		return "", fmt.Errorf("helm show values failed: %w, stderr: %s", err, stderr)
	}

	var rootValues map[string]any
	if err := yaml.Unmarshal([]byte(stdout), &rootValues); err != nil {
		return "", fmt.Errorf("parse values yaml failed: %v", err)
	}

	inferencepoolRaw, ok := rootValues["inferencepool"]
	if !ok {
		return "", fmt.Errorf("values missing key inferencepool")
	}
	inferencepool := deepCopyValue(inferencepoolRaw).(map[string]any)

	infExt, ok := inferencepool["inferenceExtension"]
	if !ok {
		return "", fmt.Errorf("inferencepool missing key inferenceExtension")
	}
	infExtMap, ok := infExt.(map[string]any)
	if !ok {
		return "", fmt.Errorf("inferencepool.inferenceExtension is not a map")
	}

	infExtMap["pluginsConfigFile"] = "epp-pd-kv-cache-aware.yaml"
	infExtMap["pluginsCustomConfig"] = map[string]any{
		"epp-pd-kv-cache-aware.yaml": eppPDKVCacheAwareYAML,
	}
	inferencepool["inferenceExtension"] = infExtMap

	overrideValues := map[string]any{"inferencepool": inferencepool}
	yamlData, err := yaml.Marshal(overrideValues)
	if err != nil {
		return "", fmt.Errorf("marshal override values yaml failed: %v", err)
	}

	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}
	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}
	tempFile := filepath.Join(tempDir, "values-hermes-router-epp-pd-config.yaml")
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}
	return tempFile, nil
}

// BuildValuesFileForHermesRouterEPPPDConfigWithoutPicker 与 BuildValuesFileForHermesRouterEPPPDConfig 相同，
// 但使用 eppPDKVCacheAwareWithoutPickerYAML（plugins 与 schedulingProfiles.plugins 均不包含 picker-pd-kv-cache-aware），用于负向测试（预期 Pod 未成功运行）。
func BuildValuesFileForHermesRouterEPPPDConfigWithoutPicker(helmURL, version string) (string, error) {
	cmd := fmt.Sprintf("helm show values %s --version %s", helmURL, version)
	stdout, stderr, err := execLocal(cmd)
	if err != nil {
		return "", fmt.Errorf("helm show values failed: %w, stderr: %s", err, stderr)
	}
	var rootValues map[string]any
	if err := yaml.Unmarshal([]byte(stdout), &rootValues); err != nil {
		return "", fmt.Errorf("parse values yaml failed: %v", err)
	}
	inferencepoolRaw, ok := rootValues["inferencepool"]
	if !ok {
		return "", fmt.Errorf("values missing key inferencepool")
	}
	inferencepool := deepCopyValue(inferencepoolRaw).(map[string]any)
	infExt, ok := inferencepool["inferenceExtension"]
	if !ok {
		return "", fmt.Errorf("inferencepool missing key inferenceExtension")
	}
	infExtMap, ok := infExt.(map[string]any)
	if !ok {
		return "", fmt.Errorf("inferencepool.inferenceExtension is not a map")
	}
	infExtMap["pluginsConfigFile"] = "epp-pd-kv-cache-aware.yaml"
	infExtMap["pluginsCustomConfig"] = map[string]any{
		"epp-pd-kv-cache-aware.yaml": eppPDKVCacheAwareWithoutPickerYAML,
	}
	inferencepool["inferenceExtension"] = infExtMap

	overrideValues := map[string]any{"inferencepool": inferencepool}
	yamlData, err := yaml.Marshal(overrideValues)
	if err != nil {
		return "", fmt.Errorf("marshal override values yaml failed: %v", err)
	}
	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}
	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}
	tempFile := filepath.Join(tempDir, "values-hermes-router-epp-pd-without-picker.yaml")
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}
	return tempFile, nil
}

// BuildValuesFileForMultipleFieldsWithDeletion 构建 values 文件来修改列表中的多个字段，并支持删除字段
// fieldUpdates: 字段路径到新值的映射，如果值为 nil，则删除该字段
// deletedFields: 要删除的字段路径列表
func BuildValuesFileForMultipleFieldsWithDeletion(helmURL, version, listPath, nameField, nameValue string, fieldUpdates map[string]any, deletedFields []string) (string, error) {
	// 1. 执行 helm show values 命令获取默认值
	cmd := fmt.Sprintf("helm show values %s --version %s", helmURL, version)
	stdout, stderr, err := execLocal(cmd)
	if err != nil {
		return "", fmt.Errorf("helm show values failed: %w, stderr: %s", err, stderr)
	}

	// 2. 解析 YAML
	var values map[string]any
	if err := yaml.Unmarshal([]byte(stdout), &values); err != nil {
		return "", fmt.Errorf("parse values yaml failed: %v", err)
	}

	// 3. 查找列表路径并匹配 name
	pathParts := strings.Split(listPath, ".")
	var current any = values
	for _, part := range pathParts {
		if m, ok := current.(map[string]any); ok {
			current = m[part]
		} else {
			return "", fmt.Errorf("path %s is not a map at %s", listPath, part)
		}
	}

	// 4. 检查是否为列表
	list, ok := current.([]any)
	if !ok {
		return "", fmt.Errorf("path %s is not a list", listPath)
	}

	// 5. 查找匹配 name 的项
	index := -1
	for i, item := range list {
		if im, ok := item.(map[string]any); ok {
			if fieldValue, exists := im[nameField]; exists {
				if fmt.Sprintf("%v", fieldValue) == nameValue {
					index = i
					break
				}
			}
		}
	}

	if index == -1 {
		return "", fmt.Errorf("no item found with %s=%s in list %s", nameField, nameValue, listPath)
	}

	// 6. 构建 values 文件：复制整个列表，只修改目标项的指定字段
	servicesList := make([]map[string]any, len(list))
	for i, item := range list {
		if im, ok := item.(map[string]any); ok {
			// 深拷贝整个 item
			newItem := deepCopyValue(im).(map[string]any)
			// 如果是目标项，修改指定字段
			if i == index {
				// 应用所有字段更新
				for fieldPath, newValue := range fieldUpdates {
					if newValue == nil {
						// 如果值为 nil，删除该字段
						deleteNestedField(newItem, fieldPath)
					} else {
						// 类型转换
						typedValue := newValue
						// 对于数字字符串，尝试转换为整数
						if strVal, ok := newValue.(string); ok {
							var intVal int
							if _, err := fmt.Sscanf(strVal, "%d", &intVal); err == nil {
								typedValue = intVal
							} else if strVal == "true" || strVal == "false" {
								// 布尔值字符串
								typedValue = (strVal == "true")
							}
						}
						setNestedField(newItem, fieldPath, typedValue)
					}
				}
				// 删除指定的字段
				for _, fieldPath := range deletedFields {
					deleteNestedField(newItem, fieldPath)
				}
			}
			servicesList[i] = newItem
		}
	}

	// 7. 构建最终的 values 结构（只包含需要修改的路径部分）
	overrideValues := make(map[string]any)

	// 构建嵌套结构：inference-backend.services
	parts := strings.Split(listPath, ".")
	currentMap := overrideValues
	for i, part := range parts {
		if i == len(parts)-1 {
			// 最后一个部分是列表名
			currentMap[part] = servicesList
		} else {
			// 中间部分创建嵌套 map
			newMap := make(map[string]any)
			currentMap[part] = newMap
			currentMap = newMap
		}
	}

	// 8. 将 values 转换为 YAML
	yamlData, err := yaml.Marshal(overrideValues)
	if err != nil {
		return "", fmt.Errorf("marshal values yaml failed: %v", err)
	}

	// 9. 创建临时文件
	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}

	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}

	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-%s-delete-%d.yaml", nameValue, index))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}

	return tempFile, nil
}

// MergeValuesFiles 合并两个 values YAML 文件，返回合并后的临时文件路径
// 第二个文件的值会覆盖第一个文件中相同的键
func MergeValuesFiles(file1, file2 string) (string, error) {
	// 读取第一个文件
	data1, err := os.ReadFile(file1)
	if err != nil {
		return "", fmt.Errorf("读取第一个 values 文件失败: %w", err)
	}

	// 读取第二个文件
	data2, err := os.ReadFile(file2)
	if err != nil {
		return "", fmt.Errorf("读取第二个 values 文件失败: %w", err)
	}

	// 解析两个 YAML 文件
	var values1, values2 map[string]any
	if err := yaml.Unmarshal(data1, &values1); err != nil {
		return "", fmt.Errorf("解析第一个 values 文件失败: %v", err)
	}
	if err := yaml.Unmarshal(data2, &values2); err != nil {
		return "", fmt.Errorf("解析第二个 values 文件失败: %v", err)
	}

	// 合并两个 map（values2 覆盖 values1）
	merged := deepMergeMaps(values1, values2)

	// 将合并后的值转换为 YAML
	yamlData, err := yaml.Marshal(merged)
	if err != nil {
		return "", fmt.Errorf("marshal 合并后的 values yaml 失败: %v", err)
	}

	// 创建临时文件
	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}

	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}

	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-merged-%d.yaml", time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}

	return tempFile, nil
}

// InjectPredictionModelVolume reads a merged values YAML file, sets prediction.modelVolume to a
// hostPath source pointing to modelPath, and writes the result to a new temp file. The chart's
// prediction sidecar template mounts this volume (named prediction-model) at artifactRoot, so only
// the volume SOURCE is supplied here — never a mountPath, and never a duplicate volumeMounts/volumes
// entry (those would collide with the template's native prediction-model rendering).
func InjectPredictionModelVolume(valuesFilePath, modelPath string) (string, error) {
	data, err := os.ReadFile(valuesFilePath)
	if err != nil {
		return "", fmt.Errorf("读取 values 文件失败: %w", err)
	}

	var values map[string]any
	if err := yaml.Unmarshal(data, &values); err != nil {
		return "", fmt.Errorf("解析 values yaml 失败: %v", err)
	}

	// 注入 prediction.modelVolume（hostPath 源）。chart 的 prediction sidecar 模板据此渲染
	// prediction-model 卷并挂载到 artifactRoot，故这里只设 volume 源；切勿再改 volumeMounts/
	// volumes，否则与模板原生渲染的 prediction-model 冲突产生重复卷。
	hr, ok := values["hermes-router"].(map[string]any)
	if !ok {
		return "", fmt.Errorf("values 中缺少 hermes-router 键")
	}
	infExt, ok := hr["inferenceExtension"].(map[string]any)
	if !ok {
		return "", fmt.Errorf("hermes-router 中缺少 inferenceExtension 键")
	}
	prediction, ok := infExt["prediction"].(map[string]any)
	if !ok {
		return "", fmt.Errorf("inferenceExtension 中缺少 prediction 键")
	}
	prediction["modelVolume"] = map[string]any{
		"hostPath": map[string]any{
			"path": modelPath,
			"type": "Directory",
		},
	}

	yamlData, err := yaml.Marshal(values)
	if err != nil {
		return "", fmt.Errorf("marshal 修改后的 values yaml 失败: %v", err)
	}

	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}
	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}

	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-prediction-vol-%d.yaml", time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}
	return tempFile, nil
}

// BuildValuesFromLocalAndOverlay deep-merges a base values file with a profile overlay file,
// returning the path to a temporary merged values file. Callers should use
// DeferCleanup(CleanupTempValuesFile, valuesFile) to remove the temp file after the test.
func BuildValuesFromLocalAndOverlay(baseValuesPath, overlayPath string) (string, error) {
	baseAbs, err := filepath.Abs(baseValuesPath)
	if err != nil {
		return "", fmt.Errorf("resolve base values path failed: %w", err)
	}
	overlayAbs, err := filepath.Abs(overlayPath)
	if err != nil {
		return "", fmt.Errorf("resolve overlay path failed: %w", err)
	}

	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}
	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}

	data1, err := os.ReadFile(baseAbs)
	if err != nil {
		return "", fmt.Errorf("读取 base values 文件失败: %w", err)
	}
	data2, err := os.ReadFile(overlayAbs)
	if err != nil {
		return "", fmt.Errorf("读取 overlay values 文件失败: %w", err)
	}

	var values1, values2 map[string]any
	if err := yaml.Unmarshal(data1, &values1); err != nil {
		return "", fmt.Errorf("解析 base values yaml 失败: %v", err)
	}
	if err := yaml.Unmarshal(data2, &values2); err != nil {
		return "", fmt.Errorf("解析 overlay values yaml 失败: %v", err)
	}

	merged := deepMergeMaps(values1, values2)
	yamlData, err := yaml.Marshal(merged)
	if err != nil {
		return "", fmt.Errorf("marshal 合并后的 values yaml 失败: %v", err)
	}

	tempFile := filepath.Join(tempDir, fmt.Sprintf("values-overlay-%d.yaml", time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}
	return tempFile, nil
}

// BuildOverlayVariant reads a profile overlay YAML file, deletes the specified nested
// field paths, and writes the modified overlay to a temp file. Callers should use
// DeferCleanup(CleanupTempValuesFile, tempPath) to remove the temp file after the test.
func BuildOverlayVariant(overlayPath string, deleteFields []string) (string, error) {
	overlayAbs, err := filepath.Abs(overlayPath)
	if err != nil {
		return "", fmt.Errorf("resolve overlay path failed: %w", err)
	}

	data, err := os.ReadFile(overlayAbs)
	if err != nil {
		return "", fmt.Errorf("读取 overlay 文件失败: %w", err)
	}

	var overlayMap map[string]any
	if err := yaml.Unmarshal(data, &overlayMap); err != nil {
		return "", fmt.Errorf("解析 overlay yaml 失败: %v", err)
	}

	for _, fieldPath := range deleteFields {
		deleteNestedField(overlayMap, fieldPath)
	}

	yamlData, err := yaml.Marshal(overlayMap)
	if err != nil {
		return "", fmt.Errorf("marshal 修改后的 overlay yaml 失败: %v", err)
	}

	dir, err := getPackageDir()
	if err != nil {
		return "", fmt.Errorf("获取包目录失败: %w", err)
	}
	tempDir := filepath.Join(dir, "tmp")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", fmt.Errorf("创建临时目录失败: %w", err)
	}

	tempFile := filepath.Join(tempDir, fmt.Sprintf("overlay-variant-%d.yaml", time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlData, 0644); err != nil {
		return "", fmt.Errorf("写入临时文件失败: %w", err)
	}
	return tempFile, nil
}

// deepMergeMaps 深度合并两个 map，第二个 map 的值会覆盖第一个 map 中相同的键
func deepMergeMaps(map1, map2 map[string]any) map[string]any {
	result := make(map[string]any)

	// 先复制 map1 的所有值
	for k, v := range map1 {
		result[k] = deepCopyValue(v)
	}

	// 然后合并 map2 的值
	for k, v2 := range map2 {
		if v1, exists := result[k]; exists {
			// 如果两个值都是 map，递归合并
			if m1, ok1 := v1.(map[string]any); ok1 {
				if m2, ok2 := v2.(map[string]any); ok2 {
					result[k] = deepMergeMaps(m1, m2)
					continue
				}
			}
			// 如果两个值都是 slice，合并 slice（对于 services 列表，需要特殊处理）
			if s1, ok1 := v1.([]any); ok1 {
				if s2, ok2 := v2.([]any); ok2 {
					// 检查是否是 service 列表（元素有 name 字段）
					// 如果是 service 列表，按 name 匹配并合并
					// 否则，直接使用第二个值覆盖（如 targetPorts）
					if isServiceList(s1) || isServiceList(s2) {
						result[k] = mergeServiceLists(s1, s2)
					} else {
						// 非 service 列表，直接使用第二个值覆盖
						result[k] = deepCopyValue(v2)
					}
					continue
				}
			}
		}
		// 如果键不存在或类型不匹配，直接使用 map2 的值
		result[k] = deepCopyValue(v2)
	}

	return result
}

// isServiceList 检查 slice 是否是 service 列表（元素有 name 字段）
func isServiceList(list []any) bool {
	if len(list) == 0 {
		return false
	}
	// 检查第一个元素是否有 name 字段
	if item, ok := list[0].(map[string]any); ok {
		_, hasName := item["name"]
		return hasName
	}
	return false
}

// mergeServiceLists 合并两个 service 列表，按 name 字段匹配并合并
func mergeServiceLists(list1, list2 []any) []any {
	// 创建一个 map 来存储按 name 索引的 service
	serviceMap := make(map[string]map[string]any)

	// 先处理 list1
	for _, item := range list1 {
		if itemMap, ok := item.(map[string]any); ok {
			if name, exists := itemMap["name"]; exists {
				nameStr := fmt.Sprintf("%v", name)
				serviceMap[nameStr] = deepCopyValue(itemMap).(map[string]any)
			}
		}
	}

	// 然后处理 list2，合并或覆盖
	for _, item := range list2 {
		if itemMap, ok := item.(map[string]any); ok {
			if name, exists := itemMap["name"]; exists {
				nameStr := fmt.Sprintf("%v", name)
				if existing, found := serviceMap[nameStr]; found {
					// 如果已存在，深度合并
					serviceMap[nameStr] = deepMergeMaps(existing, itemMap)
				} else {
					// 如果不存在，直接添加
					serviceMap[nameStr] = deepCopyValue(itemMap).(map[string]any)
				}
			}
		}
	}

	// 将 map 转换回 slice，保持 list1 的顺序，但用合并后的值替换
	result := make([]any, 0, len(serviceMap))
	// 先添加 list1 中的项（已合并）
	for _, item := range list1 {
		if itemMap, ok := item.(map[string]any); ok {
			if name, exists := itemMap["name"]; exists {
				nameStr := fmt.Sprintf("%v", name)
				if merged, found := serviceMap[nameStr]; found {
					result = append(result, merged)
					delete(serviceMap, nameStr)
				}
			}
		}
	}
	// 然后添加 list2 中新增的项
	for _, merged := range serviceMap {
		result = append(result, merged)
	}

	return result
}

func chartNameFromHelmURL(helmURL string) (string, error) {
	trimmed := strings.TrimRight(strings.TrimSpace(helmURL), "/")
	if trimmed == "" {
		return "", fmt.Errorf("helm url is empty")
	}
	parts := strings.Split(trimmed, "/")
	chartName := parts[len(parts)-1]
	if chartName == "" {
		return "", fmt.Errorf("chart name is empty in helm url %s", helmURL)
	}
	return chartName, nil
}

// PullHelmChart 拉取 Helm Chart 到本地目录，返回本地 chart 路径
// targetDir: 目标目录，如果为空则使用当前包目录下的 charts 子目录
func PullHelmChart(helmURL, version, targetDir string) (string, error) {
	// 如果没有指定目标目录，使用当前包目录下的 charts 子目录
	if targetDir == "" {
		dir, err := getPackageDir()
		if err != nil {
			return "", fmt.Errorf("获取包目录失败: %w", err)
		}
		targetDir = filepath.Join(dir, "charts")
	}

	// 确保目标目录存在
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return "", fmt.Errorf("创建目标目录失败: %w", err)
	}

	// 执行 helm pull 命令
	// helm pull 会下载 .tgz 文件到指定目录
	chartName, err := chartNameFromHelmURL(helmURL)
	if err != nil {
		return "", err
	}
	cmd := fmt.Sprintf("helm pull %s --version %s -d %s", helmURL, version, targetDir)
	_, stderr, err := execLocal(cmd)
	if err != nil {
		return "", fmt.Errorf("helm pull 失败: %w, stderr: %s", err, stderr)
	}

	// helm pull 下载的文件名可能包含版本号，查找实际下载的文件
	// 通常格式是: <chartName>-<version>.tgz
	files, err := os.ReadDir(targetDir)
	if err != nil {
		return "", fmt.Errorf("读取目标目录失败: %w", err)
	}

	var chartFile string
	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(file.Name(), ".tgz") && strings.HasPrefix(file.Name(), chartName) {
			chartFile = filepath.Join(targetDir, file.Name())
			break
		}
	}

	if chartFile == "" {
		return "", fmt.Errorf("未找到下载的 chart 文件")
	}

	fmt.Printf("Chart 已下载到: %s\n", chartFile)
	fmt.Printf("解压命令: tar -xzf %s -C %s\n", chartFile, targetDir)

	// 解压 chart
	cmd = fmt.Sprintf("tar -xzf %s -C %s", chartFile, targetDir)
	_, stderr, err = execLocal(cmd)
	if err != nil {
		return "", fmt.Errorf("解压 chart 失败: %w, stderr: %s", err, stderr)
	}

	// 查找解压后的目录（可能包含版本号）
	files, err = os.ReadDir(targetDir)
	if err != nil {
		return "", fmt.Errorf("读取解压目录失败: %w", err)
	}

	var chartDir string
	for _, file := range files {
		if file.IsDir() && strings.HasPrefix(file.Name(), chartName) {
			chartDir = filepath.Join(targetDir, file.Name())
			break
		}
	}

	if chartDir == "" {
		return "", fmt.Errorf("未找到解压后的 chart 目录")
	}

	fmt.Printf("Chart 已解压到: %s\n", chartDir)
	fmt.Printf("\n现在可以使用以下命令手动测试:\n")
	fmt.Printf("  cd %s\n", chartDir)
	fmt.Printf("  helm template test-release . -n istio-system --set inference-backend.services[1].replicas=0\n")
	fmt.Printf("或者:\n")
	fmt.Printf("  helm template test-release %s -n istio-system --set inference-backend.services[1].replicas=0\n", chartDir)

	return chartDir, nil
}

// HelmInstallExpectFailure runs helm install and expects it to fail.
// On failure, it returns the stderr content and nil error (this is the expected outcome).
// On unexpected success, it returns an empty string and an error.
func HelmInstallExpectFailure(cfg *ReleaseConfig) (string, error) {
	cmd := fmt.Sprintf("helm install %s %s -n %s",
		cfg.ReleaseName, cfg.HelmURL, cfg.Namespace)
	if strings.TrimSpace(cfg.Version) != "" && strings.TrimSpace(cfg.Version) != "latest" {
		cmd += fmt.Sprintf(" --version %s", cfg.Version)
	}

	if cfg.ValuesFile != "" {
		cmd += fmt.Sprintf(" -f %s", cfg.ValuesFile)
	}

	for _, setValue := range cfg.SetValues {
		cmd += " " + setValue
	}

	_, stderr, err := execLocal(cmd)
	if err != nil {
		return stderr, nil
	}
	return "", fmt.Errorf("helm install %s succeeded but was expected to fail", cfg.ReleaseName)
}
