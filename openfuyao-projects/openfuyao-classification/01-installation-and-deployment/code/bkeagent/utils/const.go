package utils

import (
	"fmt"
	corev1 "k8s.io/api/core/v1"
	"path/filepath"
)

var SupportPlatforms = []string{"centos", "kylin", "ubuntu"}

const (
	DefaultImageRepo     = "deploy.bocloud.k8s"
	DefaultImageRepoPort = "40443"
	DefaultImageSpace    = "kubernetes"
	DefaultYumRepo       = "deploy.bocloud.k8s"
	DefaultYumRepoPort   = "40080"
	Workspace            = "/etc/bkeagent"
	AgentScripts         = "/etc/bkeagent/scripts"
	AgentBin             = "/etc/bkeagent/bin"
	ClusterNameLabelKey  = "cluster"
)

func GetDefaultImageRepo() string {
	return fmt.Sprintf("%s:%s/%s/", DefaultImageRepo, DefaultImageRepoPort, DefaultImageSpace)
}

// cpu and memory min
const (
	MinControlPlaneNumCPU = 2
	MinControlPlaneMem    = 2
)

// bootstrap phases
const (
	InitControlPlane    = "InitControlPlane"
	JoinControlPlane    = "JoinControlPlane"
	JoinWorker          = "JoinWorker"
	UpgradeControlPlane = "UpgradeControlPlane"
	UpgradeWorker       = "UpgradeWorker"
	UpgradeEtcd         = "UpgradeEtcd"
)

// NTPServerEnvKey is the environment variable key for NTP server
const (
	NTPServerEnvKey = "NTP_SERVER"
	NTPServerPort   = "123"
)

// kubelet constants
const (
	// BKESecretType defines the type of secret created by core components.
	BKESecretType corev1.SecretType = "bke.bocloud.com/secret"

	// KubeletConfigMapNamePrefix is the prefix of the ConfigMap name where the kubelet stores the configuration data.
	KubeletConfigMapNamePrefix = "kubelet-config-"

	// KubeletConfigPath is the path where the kubelet stores the configuration data.
	KubeletConfigPath = "/var/lib/kubelet"

	// KubeletConfigFileName is the name of the configuration file for the kubelet.
	KubeletConfigFileName = "config.yaml"

	// KubernetesDir is the path where the kubernetes stores the configuration data.
	KubernetesDir = "/etc/kubernetes"

	//KubeletScriptName kubelet sh name
	KubeletScriptName = "kubelet.sh"

	// SystemdDir systemd dir
	SystemdDir = "/etc/systemd/system"
	// KubeletServiceUnitName kubelet service unit name
	KubeletServiceUnitName = "kubelet.service"
	// KubeletSavePath kubelet save path
	KubeletSavePath = "/usr/bin/kubelet"
)

const (
	// GlobalCANamespace defines the secret namespace for global ca
	GlobalCANamespace = "kube-system"
	// GlobalCASecretName defines the secret name for global ca
	GlobalCASecretName = "global-ca"
)

const (
	// RwxRxRx is the permission of the directory
	RwxRxRx = 0755
	// RwRR is the permission of the file
	RwRR = 0644
)

// GetKubeletConfPath get kubelet conf path
func GetKubeletConfPath() string {
	return filepath.Join(KubeletConfigPath, KubeletConfigFileName)
}

// GetKubeletScriptPath get kubelet script path
func GetKubeletScriptPath() string {
	return filepath.Join(KubernetesDir, KubeletScriptName)
}

// GetKubeletServicePath get kubelet service path
func GetKubeletServicePath() string {
	return filepath.Join(SystemdDir, KubeletServiceUnitName)
}

var (
	// RunKubeletPreCreateDirs is the name of the pre create dir name
	RunKubeletPreCreateDirs = []string{
		"/var/lib/calico",
		"/var/lib/lxc",
		"/opt/fabric",
		"/opt/cni",
		"/usr/libexec/kubernetes",
		"/var/lib/cni",
		"/var/lib/etcd",
		"/var/lib/kubelet",
		"/var/log/pods",
		"/var/log/containers",
		"/etc/cni",
		"/etc/kubernetes",
		"/etc/sysconfig/network-scripts",
	}
)

const (
	// UbuntuOS is the name of the ubuntu os
	UbuntuOS = "ubuntu"
	// OpenEulerOS is the name of the openeuler os
	OpenEulerOS = "openeuler"
)

// OpenFuyaoSystemController is OpenFuyaoSystemController component name
const OpenFuyaoSystemController = "openfuyao-system-controller"
