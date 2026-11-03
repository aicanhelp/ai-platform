/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package utils

import (
	"fmt"
	"path/filepath"

	corev1 "k8s.io/api/core/v1"
)

// GetSupportPlatforms define supported os platform.
func GetSupportPlatforms() []string {
	return []string{"centos", "kylin", "ubuntu"}
}

const (
	DefaultImageRepo     = "deploy.bocloud.k8s"
	DefaultImageRepoPort = "40443"
	DefaultImageSpace    = "kubernetes"
	DefaultYumRepo       = "deploy.bocloud.k8s"
	DefaultYumRepoPort   = "40080"
	Workspace            = "/etc/openFuyao/bkeagent"
	AgentScripts         = "/etc/openFuyao/bkeagent/scripts"
	AgentBin             = "/etc/openFuyao/bkeagent/bin"
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
	InitControlPlane          = "InitControlPlane"
	JoinControlPlane          = "JoinControlPlane"
	JoinWorker                = "JoinWorker"
	UpgradeControlPlane       = "UpgradeControlPlane"
	UpgradeWorker             = "UpgradeWorker"
	UpgradeEtcd               = "UpgradeEtcd"
	UpgradeControlPlaneConfig = "UpgradeControlPlaneConfig"
	UpgradeEtcdConfig         = "UpgradeEtcdConfig"
	ReconfigureNginx          = "ReconfigureNginx"
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
	// KubeConfigFilePerm is the permission of bkeagent kubeconfig (/etc/openFuyao/bkeagent/config).
	KubeConfigFilePerm = 0600
)

const (
	// OneMonthHour defines hour for one month
	OneMonthHour = 24 * 30
	// OneYearHour defines hour for one year
	OneYearHour = 365 * 24
)

const (
	// FirstBitLocalHost defines localhost first bit
	FirstBitLocalHost = 127
)

const (
	// NamespaceAndNameLen defines len of namespace and name for k8s resource
	NamespaceAndNameLen = 2
	// MinimumClusterNameLength defines min cluster name len
	MinimumClusterNameLength = 2
	// IPByteLength defines Standardized 16-byte representation for both IPv4 and IPv6 addresses.
	IPByteLength = 16
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

// GetRunKubeletPreCreateDirs defines kubelet pre create dirs.
func GetRunKubeletPreCreateDirs() []string {
	return []string{
		"/var/lib/calico",
		"/var/lib/lxc",
		"/opt/fabric",
		"/opt/cni",
		"/usr/libexec/kubernetes",
		"/var/lib/cni",
		"/var/lib/openFuyao/etcd",
		"/var/lib/kubelet",
		"/var/log/pods",
		"/var/log/containers",
		"/etc/cni",
		"/etc/kubernetes",
		"/etc/sysconfig/network-scripts",
	}
}

const (
	// UbuntuOS is the name of the ubuntu os
	UbuntuOS = "ubuntu"
	// OpenEulerOS is the name of the openeuler os
	OpenEulerOS = "openeuler"
)

// OpenFuyaoSystemController is OpenFuyaoSystemController component name
const OpenFuyaoSystemController = "openfuyao-system-controller"
