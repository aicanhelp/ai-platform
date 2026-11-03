/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package config

// NodeLabel 节点标签
type NodeLabel struct {
	Key   string `yaml:"key" json:"key"`
	Value string `yaml:"value" json:"value"`
}

// NodeInfo 节点信息
type NodeInfo struct {
	Hostname string      `yaml:"hostname" json:"hostname"`
	IP       string      `yaml:"ip" json:"ip"`
	Port     string      `yaml:"port" json:"port"`
	Username string      `yaml:"username" json:"username"`
	Password string      `yaml:"password" json:"password"`
	Role     []string    `yaml:"role" json:"role"`
	Labels   []NodeLabel `yaml:"labels,omitempty" json:"labels,omitempty"`
}

// AddonConfig 附加组件配置
type AddonConfig struct {
	Name               string              `yaml:"name" json:"name"`
	Version            string              `yaml:"version" json:"version"`
	Type               string              `json:"type,omitempty"`        // addon 类型
	ReleaseName        string              `json:"releaseName,omitempty"` // chart addon release name
	Namespace          string              `json:"namespace,omitempty"`   // chart 安装的命名空间
	Block              bool                `yaml:"block,omitempty" json:"block,omitempty"`
	Param              map[string]string   `yaml:"param,omitempty" json:"param,omitempty"`
	NodeSelector       map[string]string   `yaml:"nodeSelector,omitempty" json:"nodeSelector,omitempty"` // 节点亲和性选择器
	ValuesConfigMapRef *ValuesConfigMapRef `json:"valuesConfigMapRef,omitempty"`                         // 引用的 values.yaml
}

// ValuesConfigMapRef is a reference to a ConfigMap containing values.yaml for a chart.
type ValuesConfigMapRef struct {
	// Name of the ValuesConfigMapRef resource
	Name string `json:"name"`
	// Namespace of the ValuesConfigMapRef resource
	// If empty, defaults to the same namespace as the Cluster resource
	Namespace string `json:"namespace,omitempty"`
	// ValuesKey is the key name that stores the values.yaml in the ConfigMap
	// If empty, defaults to "values.yaml"
	ValuesKey string `json:"valuesKey,omitempty"`
}

// ClusterCustomExtra 写入 BKECluster spec.customExtra（与模板字段对齐）
type ClusterCustomExtra struct {
	// ExtraLoadBalanceIP 外部负载均衡（如 haproxy/F5）所在节点 IP，供集群侧识别 LB 流量来源等
	ExtraLoadBalanceIP string `yaml:"extraLoadBalanceIP,omitempty" json:"extraLoadBalanceIP,omitempty"`
}

// BKEClusterConfig BKE 集群配置
// 只包含需要修改的字段，其他配置从模板继承
type BKEClusterConfig struct {
	ClusterName          string                `yaml:"clusterName" json:"clusterName"`
	Namespace            string                `yaml:"namespace" json:"namespace"`
	KubernetesVersion    string                `yaml:"kubernetesVersion,omitempty" json:"kubernetesVersion,omitempty"`       // 可选，默认从模板继承
	OpenFuyaoVersion     string                `yaml:"openFuyaoVersion,omitempty" json:"openFuyaoVersion,omitempty"`         // 可选，默认从模板继承
	AddonVersionOverrides map[string]string            `yaml:"addonVersionOverrides,omitempty" json:"addonVersionOverrides,omitempty"` // 可选，按 addon 名称覆盖 version
	AddonParamOverrides   map[string]map[string]string `yaml:"addonParamOverrides,omitempty" json:"addonParamOverrides,omitempty"`     // 可选，按 addon 名称覆盖 param 字段
	Nodes                []NodeInfo            `yaml:"nodes" json:"nodes"`                                                   // 必填，节点列表
	Addons               []AddonConfig         `yaml:"addons,omitempty" json:"addons,omitempty"`                             // 可选，完全替换模板中的 addon 列表（一般不使用）
	ExcludeAddons        []string              `yaml:"excludeAddons,omitempty" json:"excludeAddons,omitempty"`               // 可选，从模板中排除指定 addon（按名称）
	ExtraAddons          []AddonConfig         `yaml:"extraAddons,omitempty" json:"extraAddons,omitempty"`                   // 可选，追加到模板 addon 列表中的额外 addon
	ImageRepo            *ImageRepoConfig      `yaml:"imageRepo,omitempty" json:"imageRepo,omitempty"`                       // 可选，离线模式修改镜像仓库
	ChartRepo            *ChartRepoConfig      `yaml:"chartRepo,omitempty" json:"chartRepo,omitempty"`                       // 可选，chart拉取地址
	ControlPlaneEndpoint *ControlPlaneEndpoint `yaml:"controlPlaneEndpoint,omitempty" json:"controlPlaneEndpoint,omitempty"` // 可选，高可用 VIP
	CustomExtra          *ClusterCustomExtra   `yaml:"customExtra,omitempty" json:"customExtra,omitempty"`                   // 可选，spec.customExtra
	LoadBalanceType      string                `yaml:"loadBalanceType,omitempty" json:"loadBalanceType,omitempty"`           // 可选，负载均衡模式 ha/nginx；空则继承模板默认 ha
	Annotations          map[string]string     `yaml:"annotations,omitempty" json:"annotations,omitempty"`                   // 可选，BKECluster metadata.annotations
}

// ImageRepoConfig 镜像仓库配置（仅在离线模式需要修改时使用）
type ImageRepoConfig struct {
	Domain string `yaml:"domain,omitempty" json:"domain,omitempty"`
	IP     string `yaml:"ip,omitempty" json:"ip,omitempty"`
	Port   string `yaml:"port,omitempty" json:"port,omitempty"`
	Prefix string `yaml:"prefix,omitempty" json:"prefix,omitempty"`
}

// ChartRepoConfig chart仓库配置
type ChartRepoConfig struct {
	// Domain defines the Repo domain name
	Domain string `json:"domain,omitempty"`
	// Ip defines the Repo ip
	IP string `json:"ip,omitempty"`
	// Port defines the number of port to connect to the Repo
	Port string `json:"port,omitempty"`
	// Prefix defines the chart address
	Prefix string `json:"prefix"`

	// AuthSecretRef defines the secret name, namespace and other information for authentication
	AuthSecretRef *AuthSecretRef `json:"authSecretRef,omitempty"`
	// TlsSecretRef defines the secret name, namespace and other information for TLS certificates
	TlsSecretRef *TlsSecretRef `json:"tlsSecretRef,omitempty"`
	// InsecureSkipTLSVerify defines whether to skip TLS verification when connecting to the repo
	// If empty, defaults to false
	InsecureSkipTLSVerify bool `json:"insecureSkipTLSVerify,omitempty"`
}

// AuthSecretRef is a reference to a secret containing authentication credentials for a repo.
type AuthSecretRef struct {
	// Name of the AuthSecretRef resource
	Name string `json:"name"`
	// Namespace of the AuthSecretRef resource
	// If empty, defaults to the same namespace as the Cluster resource
	Namespace string `json:"namespace,omitempty"`
	// UsernameKey is the key name that stores the username in the secret
	// If empty, defaults to "username"
	UsernameKey string `json:"usernameKey,omitempty"`
	// PasswordKey is the key name that stores the password in the secret
	// If empty, defaults to "password"
	PasswordKey string `json:"passwordKey,omitempty"`
}

// TlsSecretRef is a reference to a secret containing TLS certificates for a repo.
type TlsSecretRef struct {
	// Name of the TlsSecretRef resource
	Name string `json:"name"`
	// Namespace of the TlsSecretRef resource
	// If empty, defaults to the same namespace as the Cluster resource
	Namespace string `json:"namespace,omitempty"`
	// CaKey is the key name that stores the ca.crt in the secret
	// If empty, defaults to "ca.crt"
	CaKey string `json:"caKey,omitempty"`
	// CertKey is the key name that stores the cert.crt in the secret
	// If empty, defaults to "cert.crt"
	CertKey string `json:"certKey,omitempty"`
	// KeyKey is the key name that stores the key.key in the secret
	// If empty, defaults to "key.key"
	KeyKey string `json:"keyKey,omitempty"`
}

// ControlPlaneEndpoint 控制平面端点配置（高可用 VIP）
type ControlPlaneEndpoint struct {
	Host string `yaml:"host,omitempty" json:"host,omitempty"`
	Port string `yaml:"port,omitempty" json:"port,omitempty"`
}

// GuideNodeConfig 引导节点配置（用于 SSH 执行命令）
type GuideNodeConfig struct {
	Host     string
	Port     int
	Username string
	Password string
}

// BKEClusterTemplatePath BKE 集群模板配置文件路径
type BKEClusterTemplatePath struct {
	// 集群配置模板文件路径（拆分后只包含集群配置，不含节点信息）
	OneMasterClusterTemplate        string
	OneMasterOneNodeClusterTemplate string
	ThreeMasterClusterTemplate      string
}

// DefaultTemplatePaths 默认模板路径（拆分后集群和节点分开）
func DefaultTemplatePaths() *BKEClusterTemplatePath {
	return &BKEClusterTemplatePath{
		OneMasterClusterTemplate:        "/bke/cluster/1master-cluster.yaml",
		OneMasterOneNodeClusterTemplate: "/bke/cluster/1master1node-cluster.yaml",
		ThreeMasterClusterTemplate:      "/bke/cluster/3master-cluster.yaml",
	}
}

// ClusterResourceStatus 集群资源状态
type ClusterResourceStatus struct {
	NodeCount       int
	ReadyNodeCount  int
	PodCount        int
	RunningPodCount int
	NamespaceCount  int
}
