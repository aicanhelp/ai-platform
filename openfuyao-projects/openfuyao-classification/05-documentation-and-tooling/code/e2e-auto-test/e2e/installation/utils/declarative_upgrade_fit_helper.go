package utils

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

const (
	// HelmComponentAnnotation 开启 YamlComponentExecutor 路径（覆盖全局 Gate）。
	HelmComponentAnnotation = "cvo.openfuyao.cn/helm-component"

	// DefaultFITSourceVersion FIT 升级源版本（与 DeclarativeUpgradeOpenFuyaoVersion 一致）。
	DefaultFITSourceVersion = DeclarativeUpgradeOpenFuyaoVersion
	// DefaultFITTargetVersion FIT 升级目标版本（与 DeclarativeUpgradeTargetOpenFuyaoVersion 一致）。
	DefaultFITTargetVersion = DeclarativeUpgradeTargetOpenFuyaoVersion
	// DefaultFITTargetVersionNegPrune 缺少 pruneLabelSelector 的负向目标版本。
	DefaultFITTargetVersionNegPrune = "vfit-cur-neg-prune"
	// DefaultFITTargetVersionNegHC 健康检查失败的负向目标版本。
	DefaultFITTargetVersionNegHC = "vfit-cur-neg-hc"
	// DefaultFITTargetVersionNegExec 执行阶段失败的负向目标版本（DAG_EXEC_FIT_4）。
	DefaultFITTargetVersionNegExec = "vfit-cur-neg-exec"
	// DefaultFITTargetVersionNegFailFast FailFast 批次隔离负向目标版本（DAG_EXEC_DFR_1）。
	DefaultFITTargetVersionNegFailFast = "vfit-cur-neg-failfast"
	// DefaultFITTargetVersionNegHelmChart Helm chart version not found negative target version.
	DefaultFITTargetVersionNegHelmChart = "vfit-cur-neg-helm-chart"
	// DefaultFITTargetVersionNegHelmHC Helm health check failed negative target version.
	DefaultFITTargetVersionNegHelmHC = "vfit-cur-neg-helm-hc"
	// DefaultFITTargetVersionNotExist is used to verify upgrade version not found handling.
	DefaultFITTargetVersionNotExist = "version-not-exist"
	// DefaultFITPostTargetVersion 组件移除场景目标版本（与 DeclarativeUpgradeSecondTargetOpenFuyaoVersion 一致）。
	DefaultFITPostTargetVersion = DeclarativeUpgradeSecondTargetOpenFuyaoVersion

	// FITInlineDAGProbeComponent DAG_EXEC_FIT_1 观测 inline 组件。
	FITInlineDAGProbeComponent = "bkeagent"
	// FITYamlGateComponent DAG_EXEC_FIT_2/3 Gate 路径观测 yaml 组件。
	FITYamlGateComponent = "fit-yaml-gate"
	// FITYamlNewComponent YAML_ENGINE_FIT_1 首次安装观测组件。
	FITYamlNewComponent = "fit-yaml-new"
	// FITYamlSkipComponent YAML_ENGINE_FIT_10 Skip 组件（不写 clusterComponentStatuses）。
	FITYamlSkipComponent = "fit-yaml-skip"
	// FITYamlRemoveComponent YAML_ENGINE_FIT_8 高版本 RI 不再包含、但不自动卸载的组件。
	FITYamlRemoveComponent = "fit-yaml-removed"
	// FITYamlHCOKComponent DAG_EXEC_FIT_5 健康检查成功观测组件。
	FITYamlHCOKComponent = "fit-yaml-hc-ok"
	// FITYamlPruneComponent YAML_ENGINE_FIT_5 prune观测组件。
	FITYamlPruneComponent = "fit-yaml-prune"
	// FITYamlUpgradeComponent YAML_ENGINE_FIT_* upgrade观测组件。
	FITYamlUpgradeComponent = "fit-yaml-upgrade-b"
	// FITFailFastFailedComponent DAG_EXEC_DFR_1 前置失败批次组件。
	FITFailFastFailedComponent = "fit-yaml-failfast-f"
	// FITFailFastLaterComponent DAG_EXEC_DFR_1 后置依赖组件。
	FITFailFastLaterComponent = "fit-yaml-failfast-s"
	// FITPruneLegacyConfigMap YAML_ENGINE_FIT_5 应被 Prune 删除的 legacy ConfigMap。
	FITPruneLegacyConfigMap = "fit-yaml-prune-legacy"
	// FITPruneKeepConfigMap YAML_ENGINE_FIT_5 应保留的 keep ConfigMap。
	FITPruneKeepConfigMap = "fit-yaml-prune-keep"

	// ClusterComponentPhaseInstalled 组件安装/升级成功终态。
	ClusterComponentPhaseInstalled = "Installed"
	// ClusterComponentPhaseFailed 组件失败终态。
	ClusterComponentPhaseFailed = "Failed"
	// ClusterComponentPhaseRemoved 组件已移除标记。
	ClusterComponentPhaseRemoved = "Removed"
)

// FIT 预制 yaml 组件名（release-image FIT-TEST-NOTES）。
var (
	FITYamlBaselineComponents = []string{
		FITYamlCreateOnlyConfigMap,
		FITYamlGateComponent,
		FITYamlHCOKComponent,
		FITYamlNewComponent,
		FITYamlPruneComponent,
		FITYamlReplaceConfigMap,
		FITYamlSSAConfigMap,
		FITYamlUpgradeComponent,
	}
)

// ClusterComponentStatusInfo BKECluster.status.clusterComponentStatuses 单组件快照。
type ClusterComponentStatusInfo struct {
	Name               string
	Phase              string
	CurrentVersion     string
	LastTransitionTime string
	Message            string
}

// DeclarativeUpgradeCompletedInfo BKECluster.status.declarativeUpgrade.completed 单条记录。
// Gate 关闭时 yaml 走 Legacy：不写 clusterComponentStatuses Installed，但会写入本列表。
type DeclarativeUpgradeCompletedInfo struct {
	Name        string
	Version     string
	CompletedAt string
}

// DeclarativeUpgradeFITSourceVersion 返回 FIT 源版本，支持环境变量覆盖。
func DeclarativeUpgradeFITSourceVersion() string {
	if v := strings.TrimSpace(os.Getenv("DECLARATIVE_UPGRADE_FIT_SOURCE_VERSION")); v != "" {
		return v
	}
	return DefaultFITSourceVersion
}

// DeclarativeUpgradeFITTargetVersion 返回 FIT 目标版本，支持环境变量覆盖。
func DeclarativeUpgradeFITTargetVersion() string {
	if v := strings.TrimSpace(os.Getenv("DECLARATIVE_UPGRADE_FIT_TARGET_VERSION")); v != "" {
		return v
	}
	return DefaultFITTargetVersion
}

// GetClusterComponentStatus 读取 BC.status.clusterComponentStatuses[componentName]。
func (h *DeclarativeUpgradeHelper) GetClusterComponentStatus(clusterName, componentName, kubeconfigPath string) (*ClusterComponentStatusInfo, error) {
	bcName := BKEClusterResourceName(clusterName)
	ns := BKEClusterNamespace(clusterName)
	componentName = strings.TrimSpace(componentName)
	if componentName == "" {
		return nil, fmt.Errorf("componentName 不能为空")
	}

	cmd := fmt.Sprintf("%skubectl get bc %s -n %s -o json",
		h.kubeconfigPrefix(kubeconfigPath), bcName, ns)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return nil, fmt.Errorf("kubectl get bc %s/%s: %w", ns, bcName, err)
	}
	if result.ExitCode != 0 {
		return nil, fmt.Errorf("kubectl get bc %s/%s failed: %s", ns, bcName, result.Stderr)
	}

	var bc struct {
		Status struct {
			ClusterComponentStatuses map[string]struct {
				Name               string `json:"name"`
				Phase              string `json:"phase"`
				CurrentVersion     string `json:"currentVersion"`
				LastTransitionTime string `json:"lastTransitionTime"`
				Message            string `json:"message"`
			} `json:"clusterComponentStatuses"`
		} `json:"status"`
	}
	if err := json.Unmarshal([]byte(result.Stdout), &bc); err != nil {
		return nil, fmt.Errorf("parse bc %s/%s json failed: %w", ns, bcName, err)
	}

	item, ok := bc.Status.ClusterComponentStatuses[componentName]
	if !ok {
		return nil, fmt.Errorf("clusterComponentStatuses[%s] 不存在", componentName)
	}
	return &ClusterComponentStatusInfo{
		Name:               componentName,
		Phase:              strings.TrimSpace(item.Phase),
		CurrentVersion:     strings.TrimSpace(item.CurrentVersion),
		LastTransitionTime: strings.TrimSpace(item.LastTransitionTime),
		Message:            strings.TrimSpace(item.Message),
	}, nil
}

// ListClusterComponentStatusNames 列出 clusterComponentStatuses 中的组件名。
func (h *DeclarativeUpgradeHelper) ListClusterComponentStatusNames(clusterName, kubeconfigPath string) ([]string, error) {
	bcName := BKEClusterResourceName(clusterName)
	ns := BKEClusterNamespace(clusterName)

	cmd := fmt.Sprintf("%skubectl get bc %s -n %s -o json",
		h.kubeconfigPrefix(kubeconfigPath), bcName, ns)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return nil, fmt.Errorf("kubectl get bc %s/%s: %w", ns, bcName, err)
	}
	if result.ExitCode != 0 {
		return nil, fmt.Errorf("kubectl get bc %s/%s failed: %s", ns, bcName, result.Stderr)
	}

	var bc struct {
		Status struct {
			ClusterComponentStatuses map[string]json.RawMessage `json:"clusterComponentStatuses"`
		} `json:"status"`
	}
	if err := json.Unmarshal([]byte(result.Stdout), &bc); err != nil {
		return nil, fmt.Errorf("parse bc %s/%s json failed: %w", ns, bcName, err)
	}

	names := make([]string, 0, len(bc.Status.ClusterComponentStatuses))
	for k := range bc.Status.ClusterComponentStatuses {
		if strings.TrimSpace(k) == "" {
			continue
		}
		names = append(names, k)
	}
	return names, nil
}

// ListDeclarativeUpgradeCompleted 列出 BC.status.declarativeUpgrade.completed。
func (h *DeclarativeUpgradeHelper) ListDeclarativeUpgradeCompleted(
	clusterName, kubeconfigPath string,
) ([]DeclarativeUpgradeCompletedInfo, error) {
	bcName := BKEClusterResourceName(clusterName)
	ns := BKEClusterNamespace(clusterName)

	cmd := fmt.Sprintf("%skubectl get bc %s -n %s -o json",
		h.kubeconfigPrefix(kubeconfigPath), bcName, ns)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return nil, fmt.Errorf("kubectl get bc %s/%s: %w", ns, bcName, err)
	}
	if result.ExitCode != 0 {
		return nil, fmt.Errorf("kubectl get bc %s/%s failed: %s", ns, bcName, result.Stderr)
	}

	var bc struct {
		Status struct {
			DeclarativeUpgrade struct {
				Completed []struct {
					Name        string `json:"name"`
					Version     string `json:"version"`
					CompletedAt string `json:"completedAt"`
				} `json:"completed"`
			} `json:"declarativeUpgrade"`
		} `json:"status"`
	}
	if err := json.Unmarshal([]byte(result.Stdout), &bc); err != nil {
		return nil, fmt.Errorf("parse bc %s/%s json failed: %w", ns, bcName, err)
	}

	out := make([]DeclarativeUpgradeCompletedInfo, 0, len(bc.Status.DeclarativeUpgrade.Completed))
	for _, item := range bc.Status.DeclarativeUpgrade.Completed {
		name := strings.TrimSpace(item.Name)
		if name == "" {
			continue
		}
		out = append(out, DeclarativeUpgradeCompletedInfo{
			Name:        name,
			Version:     strings.TrimSpace(item.Version),
			CompletedAt: strings.TrimSpace(item.CompletedAt),
		})
	}
	return out, nil
}

// FindDeclarativeUpgradeCompleted 在 completed 中查找组件（同名多条取最后一条）。
func (h *DeclarativeUpgradeHelper) FindDeclarativeUpgradeCompleted(
	clusterName, kubeconfigPath, componentName string,
) (*DeclarativeUpgradeCompletedInfo, error) {
	componentName = strings.TrimSpace(componentName)
	if componentName == "" {
		return nil, fmt.Errorf("componentName 不能为空")
	}
	items, err := h.ListDeclarativeUpgradeCompleted(clusterName, kubeconfigPath)
	if err != nil {
		return nil, err
	}
	var found *DeclarativeUpgradeCompletedInfo
	for i := range items {
		if items[i].Name == componentName {
			item := items[i]
			found = &item
		}
	}
	if found == nil {
		return nil, fmt.Errorf("declarativeUpgrade.completed 不含 %s", componentName)
	}
	return found, nil
}

// PatchBCAnnotation 为 BKECluster 设置/覆盖注解。
func (h *DeclarativeUpgradeHelper) PatchBCAnnotation(clusterName, kubeconfigPath, key, value string) error {
	bcName := BKEClusterResourceName(clusterName)
	ns := BKEClusterNamespace(clusterName)
	cmd := fmt.Sprintf(`%skubectl annotate bc %s -n %s %s=%s --overwrite`,
		h.kubeconfigPrefix(kubeconfigPath), bcName, ns, key, value)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("annotate bc %s: %w", key, err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("annotate bc %s failed: %s", key, result.Stderr)
	}
	return nil
}

// RetriggerUpgradeViaUpgradeReady 通过 upgrade-ready 注解二次触发同目标版本升级（YAML_ENGINE_FIT_2）。
func (h *DeclarativeUpgradeHelper) RetriggerUpgradeViaUpgradeReady(clusterName, kubeconfigPath, targetVersion string) error {
	return h.PatchBCAnnotation(clusterName, kubeconfigPath, UpgradeReadyAnnotation, targetVersion)
}

// WaitClusterComponentPhase 等待指定组件达到期望 phase。
func (h *DeclarativeUpgradeHelper) WaitClusterComponentPhase(
	clusterName, componentName, kubeconfigPath, expectedPhase string,
	timeout, interval time.Duration,
) *ClusterComponentStatusInfo {
	var snap *ClusterComponentStatusInfo
	Eventually(func() string {
		status, err := h.GetClusterComponentStatus(clusterName, componentName, kubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("[clusterComponentStatuses] read %s failed: %v\n", componentName, err)
			return ""
		}
		snap = status
		GinkgoWriter.Printf("[clusterComponentStatuses] %s phase=%s current=%s lastTransition=%s message=%q\n",
			componentName, status.Phase, status.CurrentVersion, status.LastTransitionTime, status.Message)
		return status.Phase
	}, timeout, interval).Should(Equal(expectedPhase), "组件 %s phase 应为 %s", componentName, expectedPhase)
	return snap
}

// ExpectClusterComponentInstalled 断言组件 Installed 且 currentVersion 匹配。
func ExpectClusterComponentInstalled(status *ClusterComponentStatusInfo, targetVersion string) {
	Expect(status).NotTo(BeNil())
	Expect(status.Phase).To(Equal(ClusterComponentPhaseInstalled),
		"组件 %s phase 应为 Installed，实际 %s message=%q", status.Name, status.Phase, status.Message)
	Expect(strings.TrimSpace(status.CurrentVersion)).NotTo(BeEmpty(),
		"组件 %s currentVersion 不应为空", status.Name)
	Expect(status.LastTransitionTime).NotTo(BeEmpty(),
		"组件 %s lastTransitionTime 不应为空", status.Name)
}

// ExpectClusterComponentFailed 断言组件 Failed 且 message 含关键字。
func ExpectClusterComponentFailed(status *ClusterComponentStatusInfo, messageContains ...string) {
	Expect(status).NotTo(BeNil())
	Expect(status.Phase).To(Equal(ClusterComponentPhaseFailed),
		"组件 %s phase 应为 Failed，实际 %s", status.Name, status.Phase)
	Expect(status.Message).NotTo(BeEmpty(), "组件 %s 失败 message 不应为空", status.Name)
	for _, kw := range messageContains {
		if kw == "" {
			continue
		}
		Expect(strings.ToLower(status.Message)).To(ContainSubstring(strings.ToLower(kw)),
			"组件 %s message 应包含 %q，实际 %q", status.Name, kw, status.Message)
	}
	Expect(status.LastTransitionTime).NotTo(BeEmpty())
}

// LoadDeclarativeUpgrade1M1WWorkloadNodesOnMgmt 管理集群占 nodes[0:2]，业务集群使用 nodes[2:4]。
func LoadDeclarativeUpgrade1M1WWorkloadNodesOnMgmt() ([]config.NodeInfo, error) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 4 {
		return nil, fmt.Errorf("1M1W 管理集群 + 1M1W 业务集群需要至少 4 个测试节点")
	}
	clusterNodes := make([]config.NodeInfo, 2)
	copy(clusterNodes, nodes[2:4])
	clusterNodes[0].Role = []string{"master/node", "etcd"}
	clusterNodes[1].Role = []string{"node"}
	return clusterNodes, nil
}

// LoadDeclarativeUpgrade3M1WWorkloadNodesOnMgmt 管理集群占 nodes[0:4]，业务集群使用 nodes[4:8]。
func LoadDeclarativeUpgrade3M1WWorkloadNodesOnMgmt() ([]config.NodeInfo, error) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 8 {
		return nil, fmt.Errorf("3M1W 管理集群 + 3M1W 业务集群需要至少 8 个测试节点")
	}

	clusterNodes := make([]config.NodeInfo, 4)
	copy(clusterNodes, nodes[4:8])
	for i := 0; i < 3; i++ {
		clusterNodes[i].Role = []string{"master/node", "etcd"}
	}
	clusterNodes[3].Role = []string{"node"}
	return clusterNodes, nil
}

// NewDeclarativeUpgradeFITClusterConfig 创建 FIT 场景 BC 配置（默认 vfit-pre，可开 helm-component 注解）。
func NewDeclarativeUpgradeFITClusterConfig(clusterName string, nodes []config.NodeInfo, openFuyaoVersion string, enableHelmComponent bool) *config.BKEClusterConfig {
	cfg := NewDeclarativeUpgradeClusterConfig(clusterName, nodes)
	cfg.OpenFuyaoVersion = openFuyaoVersion
	if enableHelmComponent {
		if cfg.Annotations == nil {
			cfg.Annotations = map[string]string{}
		}
		cfg.Annotations[HelmComponentAnnotation] = "true"
	}
	return cfg
}
