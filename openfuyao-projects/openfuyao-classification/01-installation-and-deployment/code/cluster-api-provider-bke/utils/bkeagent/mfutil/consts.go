/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package mfutil

import (
	"path"
	"path/filepath"
)

const (
	// KubernetesDir - main configuration directory
	KubernetesDir = "/etc/kubernetes"
	// ManifestsSubDirName - manifests directory name
	ManifestsSubDirName = "manifests"
	// Etcd - component identifier (internal use)
	Etcd = "etcd"
	// KubeAPIServer - API server component name
	KubeAPIServer = "kube-apiserver"
	// KubeControllerManager - controller manager component name
	KubeControllerManager = "kube-controller-manager"
	// KubeScheduler - scheduler component name
	KubeScheduler = "kube-scheduler"
	// Kubelet - node agent component name
	Kubelet = "kubelet"
	// AuditPolicyFileName - audit policy file name
	AuditPolicyFileName = "audit-policy.yaml"
	// EtcdDataDir - etcd data storage directory
	EtcdDataDir = "/var/lib/openFuyao/etcd"
)

const (
	// HAProxy defines variable used internally when referring to the HAProxy component
	HAProxy = "haproxy"
	//Keepalived defines variable used internally when referring to the keepalived component
	Keepalived = "keepalived"
	// IngressKeepalived
	IngressKeepalived = "ingress-Keepalived"

	HAProxyConfPath = "/etc/openFuyao/haproxy"
	HAProxyConfName = "haproxy.cfg"

	KeepAlivedConfPath = "/etc/openFuyao/keepalived"
	KeepAlivedConfName = "keepalived.conf"

	MasterKeepalivedInstanceReg  = `(?ms)^\s*#master-instance-start.*#master-instance-end\s*$`
	IngressKeepalivedInstanceReg = `(?ms)^\s*#ingress-instance-start.*#ingress-instance-end\s*$`
)

const (
	// Minimum number of parts after CIDR splitting (network address/mask)
	cidrPartsCount = 2

	// Index of mask part after CIDR string splitting
	cidrMaskIndex = 1

	// Subnet mask threshold, values less than or equal to this will use 24-bit mask
	subnetMaskThreshold = 24

	// Minimum subnet mask bits
	minSubnetMaskBits = 24

	// Character set for random string generation (lowercase letters + digits)
	randomStringLetters = "abcdefghijklmnopqrstuvwxyz1234567890"

	// Default priority value
	defaultPriority = 100

	// Priority decrement step for each node position
	priorityDecrementStep = 10

	// Default weight multiplier for computeWeight function
	weightMultiplier = 10

	// Template paths
	tmplCheckMaster    = "tmpl/keepalived/check-master.sh.tmpl"
	tmplCheckIngress   = "tmpl/keepalived/check-ingress.sh.tmpl"
	tmplKeepalivedBase = "tmpl/keepalived/keepalived.base.conf.tmpl"
	tmplKeepalivedYaml = "tmpl/keepalived/keepalived.yaml.tmpl"

	// Script names
	scriptCheckMaster  = "check-master.sh"
	scriptCheckIngress = "check-ingress.sh"

	// Log messages
	logRenderScript     = "render keepalived check script: %s"
	logRenderMasterVIP  = "render keepalived master VIP instance"
	logRenderIngressVIP = "render keepalived ingress VIP instance"
	logRenderConfFile   = "render keepalived conf file"
)

func GetAuditPolicyFilePath() string {
	return path.Join(KubernetesDir, AuditPolicyFileName)
}

func GetDefaultManifestsPath() string {
	return filepath.Join(KubernetesDir, ManifestsSubDirName)
}

func GetSchedulerPolicyFilePath() string {
	return filepath.Join(KubernetesDir, "scheduler-policy-config.json")
}

func GetSchedulerAdmissionConfigFilePath() string {
	return filepath.Join(KubernetesDir, "gpu-admission.config")
}

func GetControlPlaneComponents() []string {
	return []string{
		KubeAPIServer,
		KubeControllerManager,
		KubeScheduler,
	}
}
