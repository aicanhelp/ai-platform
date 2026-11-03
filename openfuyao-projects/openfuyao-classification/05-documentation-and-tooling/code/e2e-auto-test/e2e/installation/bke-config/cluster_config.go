package config

import (
	"fmt"
	"os"
	"strconv"
)

// LoadGuideNodeFromEnv 从环境变量加载引导节点配置
func LoadGuideNodeFromEnv() *GuideNodeConfig {
	port, _ := strconv.Atoi(getEnvOrDefault("GUIDE_NODE_PORT", "22"))
	return &GuideNodeConfig{
		Host:     os.Getenv("GUIDE_NODE_HOST"),
		Port:     port,
		Username: getEnvOrDefault("GUIDE_NODE_USER", "root"),
		Password: os.Getenv("GUIDE_NODE_PASSWORD"),
	}
}

// OpenFuyaoChartRepoConfig 获取openFuyao在线chart仓库配置
func OpenFuyaoChartRepoConfig() *ChartRepoConfig {
	return &ChartRepoConfig{
		Domain: "cr.openfuyao.cn",
		IP:     "119.3.216.97",
		Port:   "443",
		Prefix: "charts",
	}
}

// GrafanaChartRepoConfig 获取 grafana chart 仓库配置
func GrafanaChartRepoConfig() *ChartRepoConfig {
	return &ChartRepoConfig{
		Domain: "grafana.github.io",
		IP:     "185.199.110.153",
		Port:   "443",
		Prefix: "helm-charts",
	}
}

// GuideNodeChartRepoConfig 获取引导节点 chart 仓库配置
func GuideNodeChartRepoConfig() *ChartRepoConfig {
	nodeConfig := LoadGuideNodeFromEnv()
	return &ChartRepoConfig{
		Domain: nodeConfig.Host,
		IP:     nodeConfig.Host,
		Port:   "38080",
		Prefix: "",
	}
}

// LoadTestNodesFromEnv 从环境变量加载测试节点信息
// 支持加载 TEST_NODE1 到 TEST_NODE8，用于管理集群和业务集群
func LoadTestNodesFromEnv() []NodeInfo {
	nodes := []NodeInfo{}

	// Helper function to load a node
	loadNode := func(nodeNum int, defaultHostname string, defaultRole []string) {
		ipKey := fmt.Sprintf("TEST_NODE%d_IP", nodeNum)
		if ip := os.Getenv(ipKey); ip != "" {
			nodes = append(nodes, NodeInfo{
				Hostname: getEnvOrDefault(fmt.Sprintf("TEST_NODE%d_HOSTNAME", nodeNum), defaultHostname),
				IP:       ip,
				Port:     getEnvOrDefault(fmt.Sprintf("TEST_NODE%d_PORT", nodeNum), "22"),
				Username: getEnvOrDefault(fmt.Sprintf("TEST_NODE%d_USER", nodeNum), "root"),
				Password: os.Getenv(fmt.Sprintf("TEST_NODE%d_PASSWORD", nodeNum)),
				Role:     defaultRole,
			})
		}
	}

	// Node1-8: Load all available nodes
	// Note: Role will be set in test code based on cluster type
	loadNode(2, "m1", []string{"master/node", "etcd"})
	loadNode(3, "m2", []string{"master/node", "etcd"})
	loadNode(4, "m3", []string{"master/node", "etcd"})
	loadNode(5, "n1", []string{"node"})
	loadNode(6, "m4", []string{"master/node", "etcd"})
	loadNode(7, "m5", []string{"master/node", "etcd"})
	loadNode(8, "m6", []string{"master/node", "etcd"})
	loadNode(9, "n2", []string{"node"})

	return nodes
}

// NewDefaultBKEClusterConfig 创建默认的 BKE 集群配置
// addon 版本从引导节点模板继承，默认排除 cluster-api 和 openfuyao-system-controller 以加快测试速度
func NewDefaultBKEClusterConfig(clusterName string, nodes []NodeInfo) *BKEClusterConfig {
	return &BKEClusterConfig{
		ClusterName:   clusterName,
		Namespace:     clusterName,
		Nodes:         nodes,
		ExcludeAddons: []string{"cluster-api", "openfuyao-system-controller"}, // 默认不安装耗时组件
	}
}

// NewBKEClusterConfigForMgmt 创建管理集群配置（保留所有 addon，仅排除 openfuyao-system-controller）
func NewBKEClusterConfigForMgmt(clusterName string, nodes []NodeInfo) *BKEClusterConfig {
	return &BKEClusterConfig{
		ClusterName:   clusterName,
		Namespace:     clusterName,
		Nodes:         nodes,
		ExcludeAddons: []string{"openfuyao-system-controller"},
	}
}

// NewBKEClusterConfigForCalicoTest 创建用于 Calico 测试的集群配置（排除 cluster-api、openfuyao-system-controller 和 calico）
func NewBKEClusterConfigForCalicoTest(clusterName string, nodes []NodeInfo) *BKEClusterConfig {
	return &BKEClusterConfig{
		ClusterName:   clusterName,
		Namespace:     clusterName,
		Nodes:         nodes,
		ExcludeAddons: []string{"cluster-api", "openfuyao-system-controller", "calico"},
	}
}

// NewBKEClusterConfigWithAllAddons 创建包含所有模板 addon 的集群配置（不排除任何 addon）
func NewBKEClusterConfigWithAllAddons(clusterName string, nodes []NodeInfo) *BKEClusterConfig {
	return &BKEClusterConfig{
		ClusterName: clusterName,
		Namespace:   clusterName,
		Nodes:       nodes,
	}
}

// NewBKEClusterConfigForHA 创建高可用集群配置（带 VIP）
func NewBKEClusterConfigForHA(clusterName string, nodes []NodeInfo, vipHost, vipPort string) *BKEClusterConfig {
	cfg := NewDefaultBKEClusterConfig(clusterName, nodes)
	cfg.ControlPlaneEndpoint = &ControlPlaneEndpoint{
		Host: vipHost,
		Port: vipPort,
	}
	return cfg
}

// NewBKEClusterConfigOffline 创建离线模式集群配置（自定义镜像仓库）
func NewBKEClusterConfigOffline(clusterName string, nodes []NodeInfo, imageRepoDomain, imageRepoIP string) *BKEClusterConfig {
	cfg := NewDefaultBKEClusterConfig(clusterName, nodes)
	cfg.ImageRepo = &ImageRepoConfig{
		Domain: imageRepoDomain,
		IP:     imageRepoIP,
	}
	return cfg
}

// KubeOSAddonName KubeOS addon 名称
const KubeOSAddonName = "kubeos-addon"

// NewKubeOSClusterConfig 创建 KubeOS 不可变 OS 模式集群配置
// 在默认配置基础上：
//  1. 设置 immutable-os annotation（bke.bocloud.com/immutable-os=kubeos）
//  2. 添加 kubeos-addon 作为 chart 类型 addon（安装 os-operator/os-proxy/CRD）
//
// chartRepo 不为空时使用指定的 chart 仓库；为空时使用 openFuyao OCI chart 仓库
// （oci://cr.openfuyao.cn/charts/kubeos-addon --version=0.0.0-latest）
func NewKubeOSClusterConfig(clusterName string, nodes []NodeInfo, chartRepo *ChartRepoConfig) *BKEClusterConfig {
	cfg := NewDefaultBKEClusterConfig(clusterName, nodes)
	cfg.Annotations = map[string]string{
		"openfuyao.io/immutable-os": "kubeos",
	}
	cfg.ExtraAddons = append(cfg.ExtraAddons, AddonConfig{
		Name:        KubeOSAddonName,
		Version:     "0.0.0-latest",
		Type:        "chart",
		ReleaseName: "kubeos-addon",
		Namespace:   "upgrade-system",
		Block:       true,
	})
	if chartRepo != nil {
		cfg.ChartRepo = chartRepo
	} else {
		// 默认使用 openFuyao OCI chart 仓库（oci://cr.openfuyao.cn/charts/）
		cfg.ChartRepo = OpenFuyaoChartRepoConfig()
	}
	return cfg
}

// LoadKubeOSChartRepoFromEnv 从环境变量加载 KubeOS addon chart 仓库配置
// 不配置时返回 nil（使用引导节点默认仓库）
func LoadKubeOSChartRepoFromEnv() *ChartRepoConfig {
	domain := os.Getenv("KUBEOS_CHART_REPO_DOMAIN")
	if domain == "" {
		return nil
	}
	return &ChartRepoConfig{
		Domain: domain,
		IP:     getEnvOrDefault("KUBEOS_CHART_REPO_IP", domain),
		Port:   getEnvOrDefault("KUBEOS_CHART_REPO_PORT", "443"),
		Prefix: getEnvOrDefault("KUBEOS_CHART_REPO_PREFIX", "charts"),
	}
}

// LoadManagementClusterVIPFromEnv 从环境变量加载管理集群的 VIP 配置
func LoadManagementClusterVIPFromEnv() (host, port string) {
	host = os.Getenv("MGMT_CLUSTER_VIP_HOST")
	port = getEnvOrDefault("MGMT_CLUSTER_VIP_PORT", "36443")
	return host, port
}

// LoadWorkloadClusterVIPFromEnv 从环境变量加载业务集群的 VIP 配置
func LoadWorkloadClusterVIPFromEnv() (host, port string) {
	host = os.Getenv("WORKLOAD_CLUSTER_VIP_HOST")
	port = getEnvOrDefault("WORKLOAD_CLUSTER_VIP_PORT", "36443")
	return host, port
}

// GetBKETemplatePath 获取 BKE 模板文件路径
func GetBKETemplatePath() *BKEClusterTemplatePath {
	return &BKEClusterTemplatePath{
		OneMasterClusterTemplate:        getEnvOrDefault("BKE_1MASTER_CLUSTER_TEMPLATE", "/bke/cluster/1master-cluster.yaml"),
		OneMasterOneNodeClusterTemplate: getEnvOrDefault("BKE_1MASTER1NODE_CLUSTER_TEMPLATE", "/bke/cluster/1master1node-cluster.yaml"),
		ThreeMasterClusterTemplate:      getEnvOrDefault("BKE_3MASTER_CLUSTER_TEMPLATE", "/bke/cluster/3master-cluster.yaml"),
	}
}

func getEnvOrDefault(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}
