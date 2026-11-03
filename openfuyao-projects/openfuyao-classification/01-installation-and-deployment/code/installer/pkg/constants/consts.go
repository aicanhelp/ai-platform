/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package constants

const (
	// ARM64 aarch64 arch
	ARM64 = "arm64"
	// AMD64 x86_64 arch
	AMD64 = "amd64"
	// X8664 x86_64
	X8664 = "x86_64"
	// AARCH64 aarch64
	AARCH64 = "aarch64"

	// Kubernetes k8s component
	Kubernetes = "kubernetes"
	// FuyaoSystem is fuyao manager
	FuyaoSystem = "openfuyao-system"
	// AscendDeployer is ascend deployer
	AscendDeployer = "ascend-deployer"
	// Master is one master node
	Master = "master"
	// Worker is worker node
	Worker = "master"
	// All is all node
	All = "all"
	// Masters is all master node
	Masters = "masters"
	// Registry is registry node
	Registry = "registry"

	// PreCheck 执行前置检查
	PreCheck = "preinstall"
	// Install 执行安装
	Install = "install"
	// Uninstall 执行卸载
	Uninstall = "uninstall"

	// Fuyao for file naming
	Fuyao = "fuyao"
	// OpenFuyao for file naming
	OpenFuyao = "openFuyao"

	// OptPath opt path
	OptPath = "/opt"
	// EtcPath etc path
	EtcPath = "/etc"
	// WebHookPath is webhook path
	WebHookPath = "webhook"
	// Slash is /
	Slash = "/"
	// Asterisk is *
	Asterisk = "*"
	// HubCaPath 镜像仓证书存放路径
	HubCaPath = "/etc/ssl/certs/openfuyao-harbor-ca.crt"
	// HubCaBkPath 备份的镜像仓证书路径
	HubCaBkPath = "/etc/ssl/certs/openfuyao-harbor-ca-bk.crt"

	// Asset asset 目录
	Asset = "asset"
	// Tmp 临时目录
	Tmp = "tmp"
	// OfflineArtifactPath 离线制品文件名
	OfflineArtifactPath = "fuyao-offline-artifact"
	// Utils utils目录
	Utils = "utils"
	// CommonScripts is common scripts folder
	CommonScripts = "common-scripts"
	// Manifests is manifests folder
	Manifests = "manifests"
	// Config is config folder
	Config = "config"
	// Targets is targets folder
	Targets = "targets"
	// Image is image save path
	Image = "image"
	// Chart is helm chart save path
	Chart = "chart"
	// Yaml is yaml file save path
	Yaml = "yaml"
	// AuthDir is auth path
	AuthDir = "auth"

	// DefaultProject is hub default project
	DefaultProject = "library"

	// PreCheckShellFile 基础前置检查shell脚本
	PreCheckShellFile = "precheck.sh"
	// OnlineInstallFile 在线安装脚本
	OnlineInstallFile = "online-install.sh"
	// PreInstallFile 安装前置检查脚本
	PreInstallFile = "preinstall.sh"
	// InstallFile 安装脚本
	InstallFile = "install.sh"
	// UnInstallFile 卸载脚本
	UnInstallFile = "uninstall.sh"
	// DownloadFile 下载制品脚本
	DownloadFile = "download.sh"
	// UploadFile 上传镜像脚本
	UploadFile = "upload.sh"
	// ConstsFile 常量定义脚本
	ConstsFile = "consts.sh"
	// InitKubernetesFile 初始化k8s脚本
	InitKubernetesFile = "init-kubernetes.sh"
	// SetK8sManifests is modifying k8s status pod yaml
	SetK8sManifests = "set-k8s-manifests.sh"
	// ArtifactPreCheck is artifact pre-check file
	ArtifactPreCheck = "artifact-precheck.sh"
	// ComponentManifest 组件清单文件名
	ComponentManifest = "fuyao-component-config.yaml"
	// RegistryConfig 镜像仓配置文件
	RegistryConfig = "registry-config.yaml"
	// DefaultValues 默认参数
	DefaultValues = "default-values.yaml"
	// AdvancedParamsFile is advanced param file
	AdvancedParamsFile = "advanced-params.yaml"
	// InstallComponentManifestFile 安装的组件清单
	InstallComponentManifestFile = "install-component-manifest.yaml"
	// MetaDataFile 组件元数据文件
	MetaDataFile = "meta-data.json"
	// ArtifactPreCheckRes is artifact pre-check result file
	ArtifactPreCheckRes = "artifact-precheck-res.json"
	// KubeConfig is kube config file
	KubeConfig = "kubeconfig"

	// KubeConfigSavePath is kubeconfig file save path
	KubeConfigSavePath = ".kube/config"

	// InternalIP is node ip
	InternalIP = "InternalIP"

	// TarGz tar.gz压缩文件后缀
	TarGz = ".tar.gz"
	// SuffixYaml .yaml文件后缀
	SuffixYaml = ".yaml"
	// Minus is -
	Minus = "-"

	// RwxRwxRwx is 777 permission
	RwxRwxRwx = 0777
	// RwxRxRx is 755 permission
	RwxRxRx = 0755
	// RwRR is 644
	RwRR = 0644
	// RwxXX is 711
	RwxXX = 0711
	// Rw is 600
	Rw = 0600
	// Rwx is 700
	Rwx = 0700

	// AddNodePostCheckTimeOut 添加节点后检查pod状态等待时间
	AddNodePostCheckTimeOut = 30
	// CreateClusterPostCheckTimeOut 创建集群后检查pod状态等待时间
	CreateClusterPostCheckTimeOut = 60
	// MemoryConversionNum is use memory conversion
	MemoryConversionNum = 1000
	// MemoryRatio is memory loss conversion ratio
	MemoryRatio = 0.9

	// FloatBitSizeMax is float bit size
	FloatBitSizeMax = 64

	// MasterCount 高可用部署至少需要3个master节点
	MasterCount = 2

	// Ha is ha component group
	Ha = "ha"
	// AI is ai component group
	AI = "ai"
	// PkgManager is cluster package manager component group
	PkgManager = "pkg-manager"
	// Cni is cni component group
	Cni = "cni"
	// Core is core component group
	Core = "core"
	// Cri is cri component group
	Cri = "cri"
	// Oci is oci component group
	Oci = "oci"
	// Tools is tools component group
	Tools = "tools"
	// Hub is hub component group
	Hub = "hub"

	// Tar is tar tool
	Tar = "tar"
	// Socat is socat component
	Socat = "socat"
	// Conntrack is conntrack component
	Conntrack = "conntrack"
	// Iptables is iptables component
	Iptables = "iptables"
	// OpenSSL is openssl component
	OpenSSL = "openssl"
	// IpSet is ipset component
	IpSet = "ipset"
	// IpvsAdm is ipvsadm component
	IpvsAdm = "ipvsadm"
	// Harbor is harbor component
	Harbor = "harbor"
	// Nerdctl is nerdctl component
	Nerdctl = "nerdctl"

	// HubCfg is hub config kind
	HubCfg = "hubCfg"
	// Build is build kind
	Build = "build"
	// Cluster is cluster kind
	Cluster = "cluster"
	// ApiVersion is now api version
	ApiVersion = "openfuyao.io/v1beta1"
	// FuyaoZone is fuyao time zone
	FuyaoZone = "fuyao_zone"

	// Installed used to determine whether a component is installed
	Installed = "installed"
	// HasInstall used to determine whether a component is installed
	HasInstall = "1"

	// TCP is tcp link
	TCP = "tcp"
	// Xterm is RequestPty
	Xterm = "xterm"

	// LogLevelDebug is debug level
	LogLevelDebug = "debug"
	// LogLevelInfo is info level
	LogLevelInfo = "info"
	// LogLevelWarn is warn level
	LogLevelWarn = "warn"
	// LogLevelError is error level
	LogLevelError = "error"
	// LogOutModOne is log out console
	LogOutModOne = "console"
	// LogOutModTwo is log out file
	LogOutModTwo = "file"
	// LogOutModThree is log out file and console
	LogOutModThree = "both"

	// AptGet is base cmd
	AptGet = "apt-get"
	// Yum is base cmd
	Yum = "yum"

	// DefaultMasterLabelName is control-plane default label name
	DefaultMasterLabelName = "node-role.kubernetes.io/control-plane"
	// DefaultMasterLabelValue is control-plane default label value
	DefaultMasterLabelValue = "control-plane"
	// DefaultWorkerLabelName is worker default label name
	DefaultWorkerLabelName = "node-role.kubernetes.io/worker"
	// DefaultWorkerLabelValue is worker default label value
	DefaultWorkerLabelValue = "worker"
	// NoSchedule is not schedule label value
	NoSchedule = "NoSchedule"

	// DefaultMasterLabelKey is control-plane label key
	DefaultMasterLabelKey = "node-role.kubernetes.io/master"

	// DefaultHarborUsername is harbor login default username
	DefaultHarborUsername = "admin"

	// InstallCluster is install cluster
	InstallCluster = "installCluster"
	// AddNodes is add nodes
	AddNodes = "addNode"

	// Zone is cli zone
	Zone = "zone"
	// RegistryDomain is harbor domain
	RegistryDomain = "registryDomain"
	// RegistryCertExpiryDaysMin is harbor cert expiry days min
	RegistryCertExpiryDaysMin = 365
	// RegistryCertExpiryDaysMax is harbor cert expiry days max
	RegistryCertExpiryDaysMax = 36500

	// PasswordMinLen 密码最小长度
	PasswordMinLen = 8
	// PasswordMaxLen 密码最大长度
	PasswordMaxLen = 32

	// FuyaoSystemPort fuyao 管理面开放端口号
	FuyaoSystemPort = 31616
	// VIP 高可用部署所用浮动ip
	VIP = "vip"

	// Https is https
	Https = "https://"
	// Http is http
	Http = "http://"

	// LatestVersion is latest version
	LatestVersion = "latest"

	// CentOS is centos
	CentOS = "CentOS"
	// EulerOS is EulerOS
	EulerOS = "EulerOS"
	// OpenEuler is openEuler
	OpenEuler = "openEuler"
	// CULinux is CULinux
	CULinux = "CULinux Enterprise Edition"
	// KylinOS is Kylin Linux
	KylinOS = "Kylin Linux Advanced Server"

	// DefaultRegistryHttpsPort is default registry port
	DefaultRegistryHttpsPort = "30443"
	// DefaultRegistryHttpPort is default registry port
	DefaultRegistryHttpPort = "30080"
)
