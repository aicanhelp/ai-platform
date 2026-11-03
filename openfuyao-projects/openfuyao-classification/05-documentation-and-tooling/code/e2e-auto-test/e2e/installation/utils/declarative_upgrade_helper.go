package utils

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

const (
	DeclarativeUpgradeOpenFuyaoVersion             = "vfit-pre"
	DeclarativeUpgradeTargetOpenFuyaoVersion       = "vfit-cur"
	DeclarativeUpgradeSecondTargetOpenFuyaoVersion = "vfit-post"
	DeclarativeUpgradeK8sVersion                   = "v1.33.1-of.2"
	DeclarativeUpgradeEtcdVersion                  = "v3.6.7-of.0"

	DeclarativeUpgradeMgmtAbnormalSourceVersion = "v26.06-rc.2"
	DeclarativeUpgradeMgmtAbnormalRc3Version    = "v26.06-rc.3"
	DeclarativeUpgradeMgmtAbnormalStableVersion = DeclarativeUpgradeTargetOpenFuyaoVersion
	DeclarativeUpgradeMgmtAbnormalK8sVersion    = "v1.34.3-of.1"
	DeclarativeUpgradeMgmtAbnormalEtcdVersion   = "v3.6.7-of.1"

	// UpgradeReadyAnnotation BKECluster 升级就绪注解（升级过程中由 BC 调谐器写入，完成后删除）。
	UpgradeReadyAnnotation = "cvo.openfuyao.cn/upgrade-ready"

	// ClusterVersionUpgradeStatusSucceeded upgradeHistory 成功状态。
	ClusterVersionUpgradeStatusSucceeded = "Succeeded"
)

// ManagementClusterDeclarativeResourceClusterName 返回管理集群自管 BC 的逻辑集群名（用于 CV/RI/扩缩容）。
// bootstrapClusterName 为引导侧管理集群逻辑名；合入 sig-installation#224 后与自管 BC 一致，不再固定为 "cluster"。
func ManagementClusterDeclarativeResourceClusterName(bootstrapClusterName string) string {
	return strings.TrimSpace(bootstrapClusterName)
}

// DeclarativeUpgradeHelper 声明式升级框架（ClusterVersion / ReleaseImage）校验辅助。
type DeclarativeUpgradeHelper struct {
	executor *executor.SSHExecutor
}

// NewDeclarativeUpgradeHelper 创建声明式升级校验辅助实例。
func NewDeclarativeUpgradeHelper(exec *executor.SSHExecutor) *DeclarativeUpgradeHelper {
	return &DeclarativeUpgradeHelper{executor: exec}
}

// BKEClusterResourceName 返回 BC 资源名（bke-{clusterName}）。
func BKEClusterResourceName(clusterName string) string {
	return "bke-" + clusterName
}

// BKEClusterNamespace 返回 BC 所在命名空间（与资源名一致）。
func BKEClusterNamespace(clusterName string) string {
	return BKEClusterResourceName(clusterName)
}

// ReleaseImageNameForVersion 根据 openFuyao 版本推导 ReleaseImage 资源名（如 openfuyao-vfit-pre）。
func ReleaseImageNameForVersion(version string) string {
	version = strings.ToLower(strings.TrimSpace(version))
	return "openfuyao-" + version
}

// ClusterVersionInfo ClusterVersion 关键字段快照。
type ClusterVersionInfo struct {
	Name           string
	Namespace      string
	DesiredVersion string
	CurrentVersion string
	Phase          string
	OwnerBCName    string
}

// ClusterVersionUpgradeHop ClusterVersion.status.upgradeHistory 单条记录。
type ClusterVersionUpgradeHop struct {
	From   string
	To     string
	Status string
}

// UpgradePathInfo UpgradePath 关键字段快照（Cluster 级资源）。
type UpgradePathInfo struct {
	Name      string
	Phase     string
	PathCount int
}

// ReleaseImageInfo ReleaseImage 关键字段快照。
type ReleaseImageInfo struct {
	Name           string
	Namespace      string
	Version        string
	Phase          string
	Digest         string
	ComponentCount int
}

func (h *DeclarativeUpgradeHelper) kubeconfigPrefix(kubeconfigPath string) string {
	if kubeconfigPath == "" {
		return ""
	}
	return fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
}

func (h *DeclarativeUpgradeHelper) kubectlClusterJSONPath(kubeconfigPath, resource, jsonPath string) (string, error) {
	cmd := fmt.Sprintf("%skubectl get %s -o jsonpath=%q",
		h.kubeconfigPrefix(kubeconfigPath), resource, jsonPath)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return "", fmt.Errorf("kubectl get %s: %w", resource, err)
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("kubectl get %s failed: %s", resource, result.Stderr)
	}
	// kubectl jsonpath 输出有时会用单/双引号包裹字符串值，统一裁剪两侧引号避免比较失败
	return strings.TrimSpace(strings.Trim(result.Stdout, `'"`)), nil
}

func (h *DeclarativeUpgradeHelper) kubectlJSONPath(kubeconfigPath, resource, name, namespace, jsonPath string) (string, error) {
	cmd := fmt.Sprintf("%skubectl get %s %s -n %s -o jsonpath=%q",
		h.kubeconfigPrefix(kubeconfigPath), resource, name, namespace, jsonPath)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return "", fmt.Errorf("kubectl get %s/%s: %w", namespace, name, err)
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("kubectl get %s/%s failed: %s", namespace, name, result.Stderr)
	}
	// kubectl jsonpath 输出有时会用单/双引号包裹字符串值，统一裁剪两侧引号避免比较失败
	return strings.TrimSpace(strings.Trim(result.Stdout, `'"`)), nil
}

// PatchClusterVersionDesiredVersion 将 ClusterVersion.spec.desiredVersion 补丁为目标版本以触发升级。
func (h *DeclarativeUpgradeHelper) PatchClusterVersionDesiredVersion(clusterName, kubeconfigPath, version string) error {
	bcName := BKEClusterResourceName(clusterName)
	ns := BKEClusterNamespace(clusterName)
	cmd := fmt.Sprintf(`%skubectl patch clusterversion %s -n %s --type=merge -p '{"spec":{"desiredVersion":"%s"}}'`,
		h.kubeconfigPrefix(kubeconfigPath), bcName, ns, version)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("patch clusterversion desiredVersion: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("patch clusterversion desiredVersion failed: %s", result.Stderr)
	}
	return nil
}

// GetBCAnnotation 读取 BKECluster 指定注解值（不存在时返回空字符串）。
func (h *DeclarativeUpgradeHelper) GetBCAnnotation(clusterName, kubeconfigPath, annotationKey string) (string, error) {
	bcName := BKEClusterResourceName(clusterName)
	ns := BKEClusterNamespace(clusterName)
	cmd := fmt.Sprintf("%skubectl get bc %s -n %s -o json",
		h.kubeconfigPrefix(kubeconfigPath), bcName, ns)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return "", fmt.Errorf("kubectl get bc %s/%s: %w", ns, bcName, err)
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("kubectl get bc %s/%s failed: %s", ns, bcName, result.Stderr)
	}

	var bc struct {
		Metadata struct {
			Annotations map[string]string `json:"annotations"`
		} `json:"metadata"`
	}
	if err := json.Unmarshal([]byte(result.Stdout), &bc); err != nil {
		return "", fmt.Errorf("parse bc %s/%s json failed: %w", ns, bcName, err)
	}
	return strings.TrimSpace(bc.Metadata.Annotations[annotationKey]), nil
}

// GetOpenFuyaoVersionFromBC 从 BKECluster 读取 openFuyaoVersion。
func (h *DeclarativeUpgradeHelper) GetOpenFuyaoVersionFromBC(clusterName, kubeconfigPath string) (string, error) {
	bcName := BKEClusterResourceName(clusterName)
	ns := BKEClusterNamespace(clusterName)
	return h.kubectlJSONPath(kubeconfigPath, "bc", bcName, ns, "{.spec.clusterConfig.cluster.openFuyaoVersion}")
}

// GetClusterVersion 获取 ClusterVersion 关键字段（CV 与 BC 同名同命名空间）。
func (h *DeclarativeUpgradeHelper) GetClusterVersion(clusterName, kubeconfigPath string) (*ClusterVersionInfo, error) {
	bcName := BKEClusterResourceName(clusterName)
	ns := BKEClusterNamespace(clusterName)

	desired, err := h.kubectlJSONPath(kubeconfigPath, "clusterversion", bcName, ns, "{.spec.desiredVersion}")
	if err != nil {
		return nil, err
	}
	current, err := h.kubectlJSONPath(kubeconfigPath, "clusterversion", bcName, ns, "{.status.currentVersion}")
	if err != nil {
		return nil, err
	}
	phase, err := h.kubectlJSONPath(kubeconfigPath, "clusterversion", bcName, ns, "{.status.phase}")
	if err != nil {
		return nil, err
	}
	ownerBC, err := h.kubectlJSONPath(kubeconfigPath, "clusterversion", bcName, ns,
		"{.metadata.ownerReferences[?(@.kind==\"BKECluster\")].name}")
	if err != nil {
		return nil, err
	}
	return &ClusterVersionInfo{
		Name:           bcName,
		Namespace:      ns,
		DesiredVersion: desired,
		CurrentVersion: current,
		Phase:          phase,
		OwnerBCName:    ownerBC,
	}, nil
}

// GetClusterVersionUpgradeHistory 读取 ClusterVersion.status.upgradeHistory。
func (h *DeclarativeUpgradeHelper) GetClusterVersionUpgradeHistory(clusterName, kubeconfigPath string) ([]ClusterVersionUpgradeHop, error) {
	bcName := BKEClusterResourceName(clusterName)
	ns := BKEClusterNamespace(clusterName)
	raw, err := h.kubectlJSONPath(kubeconfigPath, "clusterversion", bcName, ns,
		`{range .status.upgradeHistory[*]}{.from},{.to},{.status};{end}`)
	if err != nil {
		return nil, err
	}
	raw = strings.TrimSpace(strings.TrimSuffix(raw, ";"))
	if raw == "" {
		return nil, nil
	}

	var hops []ClusterVersionUpgradeHop
	for _, entry := range strings.Split(raw, ";") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		parts := strings.Split(entry, ",")
		if len(parts) < 3 {
			continue
		}
		hops = append(hops, ClusterVersionUpgradeHop{
			From:   strings.TrimSpace(parts[0]),
			To:     strings.TrimSpace(parts[1]),
			Status: strings.TrimSpace(parts[2]),
		})
	}
	return hops, nil
}

// ClusterVersionExists 检查 ClusterVersion 是否存在。
func (h *DeclarativeUpgradeHelper) ClusterVersionExists(clusterName, kubeconfigPath string) (bool, error) {
	bcName := BKEClusterResourceName(clusterName)
	ns := BKEClusterNamespace(clusterName)
	cmd := fmt.Sprintf("%skubectl get clusterversion %s -n %s --no-headers 2>/dev/null",
		h.kubeconfigPrefix(kubeconfigPath), bcName, ns)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return false, err
	}
	return result.ExitCode == 0 && strings.TrimSpace(result.Stdout) != "", nil
}

// GetReleaseImage 按名称获取 ReleaseImage 关键字段。
func (h *DeclarativeUpgradeHelper) GetReleaseImage(clusterName, releaseImageName, kubeconfigPath string) (*ReleaseImageInfo, error) {
	ns := BKEClusterNamespace(clusterName)

	version, err := h.kubectlJSONPath(kubeconfigPath, "releaseimage", releaseImageName, ns, "{.spec.version}")
	if err != nil {
		return nil, err
	}
	phase, err := h.kubectlJSONPath(kubeconfigPath, "releaseimage", releaseImageName, ns, "{.status.phase}")
	if err != nil {
		return nil, err
	}
	digest, err := h.kubectlJSONPath(kubeconfigPath, "releaseimage", releaseImageName, ns, "{.status.digest}")
	if err != nil {
		return nil, err
	}
	countStr, err := h.kubectlJSONPath(kubeconfigPath, "releaseimage", releaseImageName, ns, "{.status.componentCount}")
	if err != nil {
		return nil, err
	}
	componentCount, _ := strconv.Atoi(countStr)

	return &ReleaseImageInfo{
		Name:           releaseImageName,
		Namespace:      ns,
		Version:        version,
		Phase:          phase,
		Digest:         digest,
		ComponentCount: componentCount,
	}, nil
}

// ListReleaseImageNames 列出命名空间内所有 ReleaseImage 名称。
func (h *DeclarativeUpgradeHelper) ListReleaseImageNames(clusterName, kubeconfigPath string) ([]string, error) {
	ns := BKEClusterNamespace(clusterName)
	cmd := fmt.Sprintf("%skubectl get releaseimage -n %s -o jsonpath='{range .items[*]}{.metadata.name}{\"\\n\"}{end}'",
		h.kubeconfigPrefix(kubeconfigPath), ns)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return nil, err
	}
	if result.ExitCode != 0 {
		return nil, fmt.Errorf("list releaseimage failed: %s", result.Stderr)
	}
	names := []string{}
	for _, line := range strings.Split(strings.TrimSpace(strings.Trim(result.Stdout, "'")), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			names = append(names, line)
		}
	}
	return names, nil
}

// FindReleaseImageByVersion 按 spec.version 查找 ReleaseImage 名称。
func (h *DeclarativeUpgradeHelper) FindReleaseImageByVersion(clusterName, version, kubeconfigPath string) (string, error) {
	names, err := h.ListReleaseImageNames(clusterName, kubeconfigPath)
	if err != nil {
		return "", err
	}
	for _, name := range names {
		ri, err := h.GetReleaseImage(clusterName, name, kubeconfigPath)
		if err != nil {
			continue
		}
		if strings.TrimSpace(ri.Version) == strings.TrimSpace(version) {
			return name, nil
		}
	}
	return "", fmt.Errorf("未找到 version=%s 的 ReleaseImage", version)
}

// UpgradePathExists 检查 UpgradePath CR 是否存在。
func (h *DeclarativeUpgradeHelper) UpgradePathExists(kubeconfigPath string) (bool, error) {
	cmd := fmt.Sprintf("%skubectl get upgradepath --no-headers 2>/dev/null", h.kubeconfigPrefix(kubeconfigPath))
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return false, err
	}
	return result.ExitCode == 0 && strings.TrimSpace(result.Stdout) != "", nil
}

// HasUpgradePathBetween 检查 UpgradePath.spec.paths 中是否存在 from→to 且未 blocked 的升级边。
func (h *DeclarativeUpgradeHelper) HasUpgradePathBetween(kubeconfigPath, fromVersion, toVersion string) (bool, error) {
	raw, err := h.kubectlClusterJSONPath(kubeconfigPath, "upgradepath",
		`{range .items[0].spec.paths[*]}{.from},{.to},{.blocked};{end}`)
	if err != nil {
		return false, err
	}
	fromVersion = strings.TrimSpace(fromVersion)
	toVersion = strings.TrimSpace(toVersion)
	for _, entry := range strings.Split(strings.TrimSuffix(raw, ";"), ";") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		parts := strings.Split(entry, ",")
		if len(parts) < 2 {
			continue
		}
		from := strings.TrimSpace(parts[0])
		to := strings.TrimSpace(parts[1])
		blocked := len(parts) >= 3 && strings.EqualFold(strings.TrimSpace(parts[2]), "true")
		if from == fromVersion && to == toVersion && !blocked {
			return true, nil
		}
	}
	return false, nil
}

// GetUpgradePath 获取 UpgradePath 关键字段。
func (h *DeclarativeUpgradeHelper) GetUpgradePath(kubeconfigPath string) (*UpgradePathInfo, error) {
	name, err := h.kubectlClusterJSONPath(kubeconfigPath, "upgradepath", "{.items[0].metadata.name}")
	if err != nil {
		return nil, err
	}
	phase, err := h.kubectlClusterJSONPath(kubeconfigPath, "upgradepath", "{.items[0].status.phase}")
	if err != nil {
		return nil, err
	}
	countStr, err := h.kubectlClusterJSONPath(kubeconfigPath, "upgradepath", "{.items[0].status.pathCount}")
	if err != nil {
		return nil, err
	}
	pathCount, _ := strconv.Atoi(countStr)
	return &UpgradePathInfo{
		Name:      name,
		Phase:     phase,
		PathCount: pathCount,
	}, nil
}

// LoadDeclarativeUpgrade1M1WNodes 加载 1Master1Worker 业务集群节点（nodes[0] master，nodes[1] worker）。
func LoadDeclarativeUpgrade1M1WNodes() ([]config.NodeInfo, error) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 2 {
		return nil, fmt.Errorf("1Master1Worker 业务集群需要至少 2 个节点")
	}

	clusterNodes := make([]config.NodeInfo, 2)
	copy(clusterNodes, nodes[:2])
	clusterNodes[0].Role = []string{"master/node", "etcd"}
	clusterNodes[1].Role = []string{"node"}
	return clusterNodes, nil
}

// LoadDeclarativeUpgradeDual1M1WNodes 加载两套互斥 1M1W 节点组（nodes[0:2] 与 nodes[2:4]），用于 FIT_7 双集群并存。
func LoadDeclarativeUpgradeDual1M1WNodes() (clusterA, clusterB []config.NodeInfo, err error) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 4 {
		return nil, nil, fmt.Errorf("双 1Master1Worker 业务集群需要至少 4 个节点")
	}
	clusterA = make([]config.NodeInfo, 2)
	clusterB = make([]config.NodeInfo, 2)
	copy(clusterA, nodes[0:2])
	copy(clusterB, nodes[2:4])
	clusterA[0].Role = []string{"master/node", "etcd"}
	clusterA[1].Role = []string{"node"}
	clusterB[0].Role = []string{"master/node", "etcd"}
	clusterB[1].Role = []string{"node"}
	return clusterA, clusterB, nil
}

// LoadDeclarativeUpgrade3M1WNodes 加载 3Master1Worker 业务集群节点（nodes[0:3] master，nodes[3] worker）。
func LoadDeclarativeUpgrade3M1WNodes() ([]config.NodeInfo, error) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 4 {
		return nil, fmt.Errorf("3Master1Worker 业务集群需要至少 4 个节点")
	}

	clusterNodes := make([]config.NodeInfo, 4)
	copy(clusterNodes, nodes[:4])
	for i := 0; i < 3; i++ {
		clusterNodes[i].Role = []string{"master/node", "etcd"}
	}
	clusterNodes[3].Role = []string{"node"}
	return clusterNodes, nil
}

// PatchClusterConfigEtcdVersion 在引导节点 BC 配置中写入声明式升级 etcd 版本。
func (h *DeclarativeUpgradeHelper) PatchClusterConfigEtcdVersion(clusterConfigPath string) error {
	cmd := fmt.Sprintf(
		`yq -i '.spec.clusterConfig.cluster.etcdVersion = "%s"' %s`,
		DeclarativeUpgradeEtcdVersion,
		clusterConfigPath,
	)
	result, err := h.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("设置 etcdVersion 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("设置 etcdVersion 失败: %s", result.Stderr)
	}
	return nil
}

// NewDeclarativeUpgradeClusterConfig 创建声明式升级安装场景的 BC 配置（业务集群，不含 cluster-api）。
func NewDeclarativeUpgradeClusterConfig(clusterName string, nodes []config.NodeInfo) *config.BKEClusterConfig {
	cfg := config.NewDefaultBKEClusterConfig(clusterName, nodes)
	cfg.OpenFuyaoVersion = DeclarativeUpgradeOpenFuyaoVersion
	cfg.KubernetesVersion = DeclarativeUpgradeK8sVersion
	cfg.AddonVersionOverrides = map[string]string{
		"kubeproxy": DeclarativeUpgradeK8sVersion,
	}
	return cfg
}

// NewDeclarativeUpgradeMgmtClusterConfig 创建声明式升级管理集群 BC 配置（含 cluster-api）。
func NewDeclarativeUpgradeMgmtClusterConfig(clusterName string, nodes []config.NodeInfo) *config.BKEClusterConfig {
	cfg := NewDeclarativeUpgradeClusterConfig(clusterName, nodes)
	cfg.ExcludeAddons = []string{"openfuyao-system-controller"}
	cfg.AddonParamOverrides = map[string]map[string]string{
		"cluster-api": {
			"openFuyaoVersion": DeclarativeUpgradeOpenFuyaoVersion,
		},
	}
	return cfg
}

// NewDeclarativeUpgradeAbnormalPodClusterConfig 创建业务集群异常场景专用配置。
func NewDeclarativeUpgradeAbnormalPodClusterConfig(clusterName string, nodes []config.NodeInfo) *config.BKEClusterConfig {
	cfg := NewDeclarativeUpgradeClusterConfig(clusterName, nodes)
	cfg.OpenFuyaoVersion = DeclarativeUpgradeMgmtAbnormalSourceVersion
	cfg.KubernetesVersion = DeclarativeUpgradeMgmtAbnormalK8sVersion
	cfg.ExcludeAddons = []string{"cluster-api"}
	cfg.AddonVersionOverrides = map[string]string{
		"kubeproxy": DeclarativeUpgradeMgmtAbnormalK8sVersion,
	}
	return cfg
}

// LoadDeclarativeUpgrade1M1WMgmtNodes 加载 1Master1Worker 管理集群节点（nodes[0:2]）。
func LoadDeclarativeUpgrade1M1WMgmtNodes() ([]config.NodeInfo, error) {
	nodes, err := LoadDeclarativeUpgrade1M1WNodes()
	if err != nil {
		return nil, err
	}
	nodes[0].Hostname = "mm1"
	nodes[1].Hostname = "mw1"
	return nodes, nil
}

// LoadDeclarativeUpgrade3M1WMgmtNodes 加载 3Master1Worker 管理集群节点（nodes[0:3] master，nodes[3] worker）。
func LoadDeclarativeUpgrade3M1WMgmtNodes() ([]config.NodeInfo, error) {
	nodes, err := LoadDeclarativeUpgrade3M1WNodes()
	if err != nil {
		return nil, err
	}
	nodes[0].Hostname = "mm1"
	nodes[1].Hostname = "mm2"
	nodes[2].Hostname = "mm3"
	nodes[3].Hostname = "mw1"
	return nodes, nil
}
