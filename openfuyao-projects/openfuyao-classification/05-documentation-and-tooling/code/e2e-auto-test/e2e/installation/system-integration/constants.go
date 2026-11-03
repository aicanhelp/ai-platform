package system_integration

type NodeInfo = struct {
	Host string
	Port int
	User string
	Pass string
}

// BootstrapNodeInfo 引导节点信息
var BootstrapNodeInfo = NodeInfo{
	Host: "192.168.201.40",
	Port: 22,
	User: "root",
	Pass: "",
}

// ManagementClusterNodeInfo 管理集群节点登录信息
var ManagementClusterNodeInfo = NodeInfo{
	Host: "192.168.201.42",
	Port: 22,
	User: "root",
	Pass: "",
}

// WorkloadClusterNodeInfo 业务集群节点登录信息
var WorkloadClusterNodeInfo = NodeInfo{
	Host: "192.168.201.44",
	Port: 22,
	User: "root",
	Pass: "",
}

// FailMasterNodeInfo 故障节点登录信息
var FailMasterNodeInfo = NodeInfo{
	Host: "192.168.201.42",
	Port: 22,
	User: "root",
	Pass: "",
}

// 故障测试管理集群其他节点登录信息
var FailMasterNode2Info = NodeInfo{
	Host: "192.168.201.43",
	Port: 22,
	User: "root",
	Pass: "",
}

// 故障测试管理集群其他节点登录信息
var FailMasterNode3Info = NodeInfo{
	Host: "192.168.201.41",
	Port: 22,
	User: "root",
	Pass: "",
}

// 故障测试业务集群故障节点登录信息
var FailWorkLoadNodeInfo = NodeInfo{
	Host: "192.168.201.45",
	Port: 22,
	User: "root",
	Pass: "",
}

// 管理集群VIP地址（根据实际情况填写）
var ManagementClusterVIP = "192.168.201.254"

// VIP节点名称
var OriginalVIPHolder = "master-1"

const (
	ManagementClusterName = "cluster"
	WorkloadClusterName   = "cluster-2n"
)

// ComponentExpectation Kubernetes组件期望配置
type ComponentExpectation struct {
	Name           string   // 组件名称
	Namespace      string   // 命名空间
	ExpectedCount  int      // 期望数量
	ExpectedImages []string // 期望镜像列表
}

// 基础组件配置
var (
	// cluster-system 命名空间
	BKEControllerBase = ComponentExpectation{
		Name:          "bke-controller-manager",
		Namespace:     "cluster-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/cluster-api-provider-bke:1.0.3",
			"cr.openfuyao.cn/openfuyao/bke-manifests:1.0.3",
		},
	}

	BKEAgentDeployerBase = ComponentExpectation{
		Name:          "bkeagent-deployer",
		Namespace:     "cluster-system",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/bkeagent-deployer:1.0.3",
		},
	}

	CapiControllerBase = ComponentExpectation{
		Name:          "capi-controller-manager",
		Namespace:     "cluster-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/cluster-api/cluster-api-controller:v1.4.3",
			"cr.openfuyao.cn/openfuyao/cluster-api/kubeadm-bootstrap-controller:v1.4.3",
			"cr.openfuyao.cn/openfuyao/cluster-api/kubeadm-control-plane-controller:v1.4.3",
		},
	}

	IngressControllerBase = ComponentExpectation{
		Name:          "ingress-nginx-controller",
		Namespace:     "ingress-nginx",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/ingress-nginx/controller:v1.9.4",
		},
	}

	// kube-system 命名空间
	CalicoControllerBase = ComponentExpectation{
		Name:          "calico-kube-controllers",
		Namespace:     "kube-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/calico/kube-controllers:v3.27.3",
		},
	}

	CalicoNodeBase = ComponentExpectation{
		Name:          "calico-node",
		Namespace:     "kube-system",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/calico/node:v3.27.3",
			"cr.openfuyao.cn/openfuyao/calico/cni:v3.27.3",
		},
	}

	CoreDNSBase = ComponentExpectation{
		Name:          "coredns",
		Namespace:     "kube-system",
		ExpectedCount: 2,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/kubernetes/coredns:v1.10.1",
		},
	}

	ETCDBase = ComponentExpectation{
		Name:          "etcd",
		Namespace:     "kube-system",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/etcd:3.5.21-of.1",
		},
	}

	HaproxyBase = ComponentExpectation{
		Name:          "haproxy",
		Namespace:     "kube-system",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/haproxy/haproxy:2.1.4",
		},
	}

	KeepalivedBase = ComponentExpectation{
		Name:          "keepalived",
		Namespace:     "kube-system",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/keepalived/keepalived:1.3.5",
		},
	}

	KubeAPIServerBase = ComponentExpectation{
		Name:          "kube-apiserver",
		Namespace:     "kube-system",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/kube-apiserver:1.33.1-of.2",
		},
	}

	KubeControllerBase = ComponentExpectation{
		Name:          "kube-controller-manager",
		Namespace:     "kube-system",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/kube-controller-manager:1.33.1-of.2",
		},
	}

	KubeProxyBase = ComponentExpectation{
		Name:          "kube-proxy",
		Namespace:     "kube-system",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/kube-proxy:1.33.1-of.2",
		},
	}

	KubeSchedulerBase = ComponentExpectation{
		Name:          "kube-scheduler",
		Namespace:     "kube-system",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/kube-scheduler:1.33.1-of.2",
		},
	}

	MetricsServerBase = ComponentExpectation{
		Name:          "metrics-server",
		Namespace:     "kube-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/metrics-server/metrics-server:v0.7.1",
		},
	}

	// monitoring 命名空间
	AlertManagerBase = ComponentExpectation{
		Name:          "alertmanager-main",
		Namespace:     "monitoring",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/prometheus-operator/prometheus-config-reloader:v0.74.0",
			"cr.openfuyao.cn/openfuyao/prometheus/alertmanager:v0.27.0",
		},
	}

	BlackboxExporterBase = ComponentExpectation{
		Name:          "blackbox-exporter",
		Namespace:     "monitoring",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/prometheus/blackbox-exporter:v0.25.0",
			"cr.openfuyao.cn/openfuyao/jimmidyson/configmap-reload:v0.13.0",
			"cr.openfuyao.cn/openfuyao/brancz/kube-rbac-proxy:v0.18.0",
		},
	}

	KubeStateMetricsBase = ComponentExpectation{
		Name:          "kube-state-metrics",
		Namespace:     "monitoring",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/brancz/kube-rbac-proxy:v0.18.0",
			"cr.openfuyao.cn/openfuyao/kube-state-metrics/kube-state-metrics:v2.12.0",
		},
	}

	NodeExporterBase = ComponentExpectation{
		Name:          "node-exporter",
		Namespace:     "monitoring",
		ExpectedCount: 3,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/brancz/kube-rbac-proxy:v0.18.0",
			"cr.openfuyao.cn/openfuyao/prometheus/node-exporter:v1.8.1",
		},
	}

	PrometheusBase = ComponentExpectation{
		Name:          "prometheus-k8s",
		Namespace:     "monitoring",
		ExpectedCount: 2,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/prometheus-operator/prometheus-config-reloader:v0.74.0",
			"cr.openfuyao.cn/openfuyao/prometheus/prometheus:v2.52.0",
		},
	}

	PrometheusOperatorBase = ComponentExpectation{
		Name:          "prometheus-operator",
		Namespace:     "monitoring",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/brancz/kube-rbac-proxy:v0.18.0",
			"cr.openfuyao.cn/openfuyao/prometheus-operator/prometheus-operator:v0.74.0",
		},
	}

	OpenfuyaoControllerBase = ComponentExpectation{
		Name:          "openfuyao-system-controller",
		Namespace:     "openfuyao-system-controller",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/openfuyao-system-controller:1.0.2",
		},
	}

	// openfuyao-system 命名空间
	ApplicationManagementBase = ComponentExpectation{
		Name:          "application-management-service",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/application-management-service:1.0.2",
		},
	}

	ConsoleServiceBase = ComponentExpectation{
		Name:          "console-service",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/console-service:1.0.2",
		},
	}

	ConsoleWebsiteBase = ComponentExpectation{
		Name:          "console-website",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/console-website:1.0.2",
		},
	}

	InstallerServiceBase = ComponentExpectation{
		Name:          "installer-service",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/installer-service:1.0.3",
		},
	}

	InstallerWebsiteBase = ComponentExpectation{
		Name:          "installer-website",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/installer-website:1.0.2",
		},
	}

	HarborChartmuseumBase = ComponentExpectation{
		Name:          "local-harbor-chartmuseum",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/harbor/chartmuseum-photon:v2.7.0",
		},
	}

	HarborCoreBase = ComponentExpectation{
		Name:          "local-harbor-core",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/harbor/harbor-core:v2.7.0",
		},
	}

	HarborDatabaseBase = ComponentExpectation{
		Name:          "local-harbor-database",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/harbor/harbor-db:v2.7.0",
		},
	}

	HarborJobServiceBase = ComponentExpectation{
		Name:          "local-harbor-jobservice",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/harbor/harbor-jobservice:v2.7.0",
		},
	}

	HarborNginxBase = ComponentExpectation{
		Name:          "local-harbor-nginx",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/harbor/nginx-photon:v2.7.0",
		},
	}

	HarborPortalBase = ComponentExpectation{
		Name:          "local-harbor-portal",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/harbor/harbor-portal:v2.7.0",
		},
	}

	HarborRedisBase = ComponentExpectation{
		Name:          "local-harbor-redis",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/harbor/redis-photon:v2.7.0",
		},
	}

	HarborRegistryBase = ComponentExpectation{
		Name:          "local-harbor-registry",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/harbor/harbor-registryctl:v2.7.0",
			"cr.openfuyao.cn/openfuyao/harbor/registry-photon:v2.7.0",
		},
	}

	MarketplaceServiceBase = ComponentExpectation{
		Name:          "marketplace-service",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/marketplace-service:1.0.2",
		},
	}

	MonitoringServiceBase = ComponentExpectation{
		Name:          "monitoring-service",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/monitoring-service:1.0.2",
		},
	}

	OauthServerBase = ComponentExpectation{
		Name:          "oauth-server",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/oauth-server:1.0.2",
		},
	}

	OauthWebhookBase = ComponentExpectation{
		Name:          "oauth-webhook",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/oauth-webhook:1.0.2",
		},
	}

	PluginManagementBase = ComponentExpectation{
		Name:          "plugin-management-service",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/plugin-management-service:1.0.2",
		},
	}

	UserManagementBase = ComponentExpectation{
		Name:          "user-management-operator",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/user-management-operator:1.0.2",
		},
	}

	WebTerminalBase = ComponentExpectation{
		Name:          "web-terminal-service",
		Namespace:     "openfuyao-system",
		ExpectedCount: 1,
		ExpectedImages: []string{
			"cr.openfuyao.cn/openfuyao/web-terminal-service:1.0.2",
		},
	}
)

// 节点组件期望配置
var (
	// 引导节点 - BKE 引导集群运行的组件
	BootstrapNodeExpectations = map[string]ComponentExpectation{
		// openfuyao-system - openFuyao 系统组件
		"installer-service":         InstallerServiceBase,
		"console-website":           ConsoleWebsiteBase,
		"console-service":           ConsoleServiceBase,
		"plugin-management-service": PluginManagementBase,
		"user-management-operator":  UserManagementBase,
		"oauth-server":              OauthServerBase,
		"oauth-webhook":             OauthWebhookBase,
		"installer-website":         InstallerWebsiteBase,

		// kube-system - Kubernetes 系统组件
		"coredns": CreateComponentWithCount(CoreDNSBase, 1),

		// cluster-system - CAPI 相关组件
		"capi-controller-manager": CapiControllerBase,
		"bke-controller-manager":  BKEControllerBase,

		// ingress-nginx - Ingress 控制器
		"ingress-nginx-controller": CreateComponentWithCount(IngressControllerBase, 1),
	}

	// 管理节点
	ManagerNodeExpectations = map[string]ComponentExpectation{
		// cluster-system
		"bke-controller-manager":  BKEControllerBase,
		"bkeagent-deployer":       BKEAgentDeployerBase,
		"capi-controller-manager": CapiControllerBase,

		// ingress-nginx
		"ingress-nginx-controller": IngressControllerBase,

		// kube-system
		"calico-kube-controllers": CalicoControllerBase,
		"calico-node":             CalicoNodeBase,
		"coredns":                 CoreDNSBase,
		"etcd":                    ETCDBase,
		"haproxy":                 HaproxyBase,
		"keepalived":              KeepalivedBase,
		"kube-apiserver":          KubeAPIServerBase,
		"kube-controller-manager": KubeControllerBase,
		"kube-proxy":              KubeProxyBase,
		"kube-scheduler":          KubeSchedulerBase,
		"metrics-server":          MetricsServerBase,

		// monitoring
		"alertmanager-main":   AlertManagerBase,
		"blackbox-exporter":   BlackboxExporterBase,
		"kube-state-metrics":  KubeStateMetricsBase,
		"node-exporter":       NodeExporterBase,
		"prometheus-k8s":      PrometheusBase,
		"prometheus-operator": PrometheusOperatorBase,

		// openfuyao-system-controller
		"openfuyao-system-controller": OpenfuyaoControllerBase,

		// openfuyao-system
		"application-management-service": ApplicationManagementBase,
		"console-service":                ConsoleServiceBase,
		"console-website":                ConsoleWebsiteBase,
		"installer-service":              InstallerServiceBase,
		"installer-website":              InstallerWebsiteBase,
		"local-harbor-chartmuseum":       HarborChartmuseumBase,
		"local-harbor-core":              HarborCoreBase,
		"local-harbor-database":          HarborDatabaseBase,
		"local-harbor-jobservice":        HarborJobServiceBase,
		"local-harbor-nginx":             HarborNginxBase,
		"local-harbor-portal":            HarborPortalBase,
		"local-harbor-redis":             HarborRedisBase,
		"local-harbor-registry":          HarborRegistryBase,
		"marketplace-service":            MarketplaceServiceBase,
		"monitoring-service":             MonitoringServiceBase,
		"oauth-server":                   OauthServerBase,
		"oauth-webhook":                  OauthWebhookBase,
		"plugin-management-service":      PluginManagementBase,
		"user-management-operator":       UserManagementBase,
		"web-terminal-service":           WebTerminalBase,
	}

	// 业务节点
	BusinessNodeExpectations = map[string]ComponentExpectation{
		// cluster-system（业务节点没有bke-controller-manager和capi-controller-manager）
		"bkeagent-deployer": CreateComponentWithCount(BKEAgentDeployerBase, 2),

		// ingress-nginx
		"ingress-nginx-controller": CreateComponentWithCount(IngressControllerBase, 2),

		// kube-system（业务节点没有控制平面组件）
		"calico-kube-controllers": CalicoControllerBase,
		"calico-node":             CreateComponentWithCount(CalicoNodeBase, 2), // 2个节点
		"coredns":                 CoreDNSBase,
		"etcd":                    CreateComponentWithCount(ETCDBase, 1),
		"kube-apiserver":          CreateComponentWithCount(KubeAPIServerBase, 1),
		"kube-controller-manager": CreateComponentWithCount(KubeControllerBase, 1),
		"kube-proxy":              CreateComponentWithCount(KubeProxyBase, 2), // 2个节点
		"kube-scheduler":          CreateComponentWithCount(KubeSchedulerBase, 1),
		"metrics-server":          MetricsServerBase,

		// monitoring
		"alertmanager-main":   AlertManagerBase,
		"blackbox-exporter":   BlackboxExporterBase,
		"kube-state-metrics":  KubeStateMetricsBase,
		"node-exporter":       CreateComponentWithCount(NodeExporterBase, 2), // 2个节点
		"prometheus-k8s":      PrometheusBase,
		"prometheus-operator": PrometheusOperatorBase,

		// openfuyao-system-controller
		"openfuyao-system-controller": OpenfuyaoControllerBase,

		// openfuyao-system
		"application-management-service": ApplicationManagementBase,
		"console-service":                ConsoleServiceBase,
		"console-website":                ConsoleWebsiteBase,
		"local-harbor-chartmuseum":       HarborChartmuseumBase,
		"local-harbor-core":              HarborCoreBase,
		"local-harbor-database":          HarborDatabaseBase,
		"local-harbor-jobservice":        HarborJobServiceBase,
		"local-harbor-nginx":             HarborNginxBase,
		"local-harbor-portal":            HarborPortalBase,
		"local-harbor-redis":             HarborRedisBase,
		"local-harbor-registry":          HarborRegistryBase,
		"marketplace-service":            MarketplaceServiceBase,
		"monitoring-service":             MonitoringServiceBase,
		"oauth-server":                   OauthServerBase,
		"oauth-webhook":                  OauthWebhookBase,
		"plugin-management-service":      PluginManagementBase,
		"user-management-operator":       UserManagementBase,
		"web-terminal-service":           WebTerminalBase,
	}
)

// ConfigVersions 定义所有配置文件中需要验证的版本信息
type ConfigVersions struct {
	Addons struct {
		KubeProxy               string
		Calico                  string
		CoreDNS                 string
		BKEAgentDeployer        string
		BKEAgentDeployerTag     string
		ClusterAPI              string
		ClusterAPIContainerd    string
		ClusterAPIManifests     string
		ClusterAPIOpenFuyao     string
		ClusterAPIProvider      string
		OpenFuyaoController     string
		OpenFuyaoControllerTag  string
		OpenFuyaoControllerRepo string
	}
	Cluster struct {
		ContainerdVersion string
		EtcdVersion       string
		KubernetesVersion string
		OpenFuyaoVersion  string
	}
	CustomExtra struct {
		ClusterAPI string
		Containerd string
	}
}

// bootstrapNodeVersions 提供默认的版本配置
var BootstrapNodeVersions = &ConfigVersions{
	Addons: struct {
		KubeProxy               string
		Calico                  string
		CoreDNS                 string
		BKEAgentDeployer        string
		BKEAgentDeployerTag     string
		ClusterAPI              string
		ClusterAPIContainerd    string
		ClusterAPIManifests     string
		ClusterAPIOpenFuyao     string
		ClusterAPIProvider      string
		OpenFuyaoController     string
		OpenFuyaoControllerTag  string
		OpenFuyaoControllerRepo string
	}{
		KubeProxy:               "v1.33.1-of.2",
		Calico:                  "v3.27.3",
		CoreDNS:                 "v1.10.1",
		BKEAgentDeployer:        "v1.0.3",
		BKEAgentDeployerTag:     "1.0.3",
		ClusterAPI:              "v1.4.3",
		ClusterAPIContainerd:    "v2.1.1",
		ClusterAPIManifests:     "1.0.3",
		ClusterAPIOpenFuyao:     "v25.12",
		ClusterAPIProvider:      "1.0.3",
		OpenFuyaoController:     "v1.0.2",
		OpenFuyaoControllerTag:  "1.0.2",
		OpenFuyaoControllerRepo: "https://helm.openfuyao.cn/_core",
	},
	Cluster: struct {
		ContainerdVersion string
		EtcdVersion       string
		KubernetesVersion string
		OpenFuyaoVersion  string
	}{
		ContainerdVersion: "v2.1.1",
		EtcdVersion:       "v3.5.21-of.1",
		KubernetesVersion: "v1.33.1-of.2",
		OpenFuyaoVersion:  "v25.12",
	},
	CustomExtra: struct {
		ClusterAPI string
		Containerd string
	}{
		ClusterAPI: "1.0.3",
		Containerd: "containerd-v2.1.1-linux-{.arch}.tar.gz",
	},
}

// BKEClusterConfig 用于解析BKECluster YAML文件
type BKEClusterConfig struct {
	APIVersion string `yaml:"apiVersion"`
	Kind       string `yaml:"kind"`
	Metadata   struct {
		Name      string `yaml:"name"`
		Namespace string `yaml:"namespace"`
	} `yaml:"metadata"`
	Spec struct {
		ClusterConfig struct {
			Addons []struct {
				Name    string                 `yaml:"name"`
				Param   map[string]interface{} `yaml:"param,omitempty"`
				Version string                 `yaml:"version,omitempty"`
				Block   bool                   `yaml:"block,omitempty"`
			} `yaml:"addons"`
			Cluster struct {
				ContainerdVersion string `yaml:"containerdVersion"`
				EtcdVersion       string `yaml:"etcdVersion"`
				KubernetesVersion string `yaml:"kubernetesVersion"`
				OpenFuyaoVersion  string `yaml:"openFuyaoVersion"`
			} `yaml:"cluster"`
			CustomExtra struct {
				ClusterAPI string `yaml:"clusterapi"`
				Containerd string `yaml:"containerd"`
			} `yaml:"customExtra"`
		} `yaml:"clusterConfig"`
	} `yaml:"spec"`
}

// Prometheus API响应结构体
type PrometheusQueryResponse struct {
	Status string `json:"status"`
	Data   struct {
		ResultType string `json:"resultType"`
		Result     []struct {
			Metric map[string]string `json:"metric"`
			Value  []interface{}     `json:"value"`
		} `json:"result"`
	} `json:"data"`
	Error     string `json:"error,omitempty"`
	ErrorType string `json:"errorType,omitempty"`
}

// ServiceExpectation 服务期望配置
type ServiceExpectation struct {
	Name      string // 服务名称
	Namespace string // 命名空间
	Type      string // 服务类型: ClusterIP, NodePort, LoadBalancer, ExternalName
}

// 所有服务的期望配置
var (
	// default 命名空间
	KubernetesDefaultService = ServiceExpectation{
		Name:      "kubernetes",
		Namespace: "default",
		Type:      "ClusterIP",
	}

	// ingress-nginx 命名空间
	IngressNginxControllerService = ServiceExpectation{
		Name:      "ingress-nginx-controller",
		Namespace: "ingress-nginx",
		Type:      "NodePort",
	}

	IngressNginxControllerAdmissionService = ServiceExpectation{
		Name:      "ingress-nginx-controller-admission",
		Namespace: "ingress-nginx",
		Type:      "ClusterIP",
	}

	// kube-system 命名空间
	EtcdService = ServiceExpectation{
		Name:      "etcd-service",
		Namespace: "kube-system",
		Type:      "ClusterIP",
	}

	KubeAPIServerService = ServiceExpectation{
		Name:      "kube-apiserver-service",
		Namespace: "kube-system",
		Type:      "ClusterIP",
	}

	KubeControllerManagerService = ServiceExpectation{
		Name:      "kube-controller-manager-service",
		Namespace: "kube-system",
		Type:      "ClusterIP",
	}

	KubeDNSService = ServiceExpectation{
		Name:      "kube-dns",
		Namespace: "kube-system",
		Type:      "ClusterIP",
	}

	KubeProxyService = ServiceExpectation{
		Name:      "kube-proxy-service",
		Namespace: "kube-system",
		Type:      "ClusterIP",
	}

	KubeSchedulerService = ServiceExpectation{
		Name:      "kube-scheduler-service",
		Namespace: "kube-system",
		Type:      "ClusterIP",
	}

	KubeletService = ServiceExpectation{
		Name:      "kubelet",
		Namespace: "kube-system",
		Type:      "ClusterIP",
	}

	MetricsServerService = ServiceExpectation{
		Name:      "metrics-server",
		Namespace: "kube-system",
		Type:      "ClusterIP",
	}

	// monitoring 命名空间
	AlertManagerMainService = ServiceExpectation{
		Name:      "alertmanager-main",
		Namespace: "monitoring",
		Type:      "ClusterIP",
	}

	AlertManagerOperatedService = ServiceExpectation{
		Name:      "alertmanager-operated",
		Namespace: "monitoring",
		Type:      "ClusterIP",
	}

	BlackboxExporterService = ServiceExpectation{
		Name:      "blackbox-exporter",
		Namespace: "monitoring",
		Type:      "ClusterIP",
	}

	KubeStateMetricsService = ServiceExpectation{
		Name:      "kube-state-metrics",
		Namespace: "monitoring",
		Type:      "ClusterIP",
	}

	NodeExporterService = ServiceExpectation{
		Name:      "node-exporter",
		Namespace: "monitoring",
		Type:      "ClusterIP",
	}

	PrometheusK8sService = ServiceExpectation{
		Name:      "prometheus-k8s",
		Namespace: "monitoring",
		Type:      "ClusterIP",
	}

	PrometheusOperatedService = ServiceExpectation{
		Name:      "prometheus-operated",
		Namespace: "monitoring",
		Type:      "ClusterIP",
	}

	PrometheusOperatorService = ServiceExpectation{
		Name:      "prometheus-operator",
		Namespace: "monitoring",
		Type:      "ClusterIP",
	}

	// openfuyao-system 命名空间
	ApplicationManagementService = ServiceExpectation{
		Name:      "application-management-service",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	ConsoleService = ServiceExpectation{
		Name:      "console-service",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	ConsoleWebsiteService = ServiceExpectation{
		Name:      "console-website",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	LocalHarborService = ServiceExpectation{
		Name:      "local-harbor",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	LocalHarborChartmuseumService = ServiceExpectation{
		Name:      "local-harbor-chartmuseum",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	LocalHarborCoreService = ServiceExpectation{
		Name:      "local-harbor-core",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	LocalHarborDatabaseService = ServiceExpectation{
		Name:      "local-harbor-database",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	LocalHarborJobserviceService = ServiceExpectation{
		Name:      "local-harbor-jobservice",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	LocalHarborPortalService = ServiceExpectation{
		Name:      "local-harbor-portal",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	LocalHarborRedisService = ServiceExpectation{
		Name:      "local-harbor-redis",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	LocalHarborRegistryService = ServiceExpectation{
		Name:      "local-harbor-registry",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	MarketplaceService = ServiceExpectation{
		Name:      "marketplace-service",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	MonitoringService = ServiceExpectation{
		Name:      "monitoring-service",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	OauthServerService = ServiceExpectation{
		Name:      "oauth-server",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	OauthWebhookService = ServiceExpectation{
		Name:      "oauth-webhook",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	PluginManagementService = ServiceExpectation{
		Name:      "plugin-management-service",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	UserManagementOperatorService = ServiceExpectation{
		Name:      "user-management-operator",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}

	WebTerminalService = ServiceExpectation{
		Name:      "web-terminal-service",
		Namespace: "openfuyao-system",
		Type:      "ClusterIP",
	}
)

// ServiceExpectations 服务期望配置映射
var ServiceExpectations = map[string]ServiceExpectation{
	// default 命名空间
	"kubernetes": KubernetesDefaultService,

	// ingress-nginx 命名空间
	"ingress-nginx-controller":           IngressNginxControllerService,
	"ingress-nginx-controller-admission": IngressNginxControllerAdmissionService,

	// kube-system 命名空间
	"etcd-service":                    EtcdService,
	"kube-apiserver-service":          KubeAPIServerService,
	"kube-controller-manager-service": KubeControllerManagerService,
	"kube-dns":                        KubeDNSService,
	"kube-proxy-service":              KubeProxyService,
	"kube-scheduler-service":          KubeSchedulerService,
	"kubelet":                         KubeletService,
	"metrics-server":                  MetricsServerService,

	// monitoring 命名空间
	"alertmanager-main":     AlertManagerMainService,
	"alertmanager-operated": AlertManagerOperatedService,
	"blackbox-exporter":     BlackboxExporterService,
	"kube-state-metrics":    KubeStateMetricsService,
	"node-exporter":         NodeExporterService,
	"prometheus-k8s":        PrometheusK8sService,
	"prometheus-operated":   PrometheusOperatedService,
	"prometheus-operator":   PrometheusOperatorService,

	// openfuyao-system 命名空间
	"application-management-service": ApplicationManagementService,
	"console-service":                ConsoleService,
	"console-website":                ConsoleWebsiteService,
	"local-harbor":                   LocalHarborService,
	"local-harbor-chartmuseum":       LocalHarborChartmuseumService,
	"local-harbor-core":              LocalHarborCoreService,
	"local-harbor-database":          LocalHarborDatabaseService,
	"local-harbor-jobservice":        LocalHarborJobserviceService,
	"local-harbor-portal":            LocalHarborPortalService,
	"local-harbor-redis":             LocalHarborRedisService,
	"local-harbor-registry":          LocalHarborRegistryService,
	"marketplace-service":            MarketplaceService,
	"monitoring-service":             MonitoringService,
	"oauth-server":                   OauthServerService,
	"oauth-webhook":                  OauthWebhookService,
	"plugin-management-service":      PluginManagementService,
	"user-management-operator":       UserManagementOperatorService,
	"web-terminal-service":           WebTerminalService,
}

// 资源信息结构体
type ResourceInfo struct {
	Type      string
	Name      string
	Namespace string
}
