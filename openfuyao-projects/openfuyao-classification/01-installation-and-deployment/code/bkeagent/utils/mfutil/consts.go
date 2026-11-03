package mfutil

import (
	"path"
	"path/filepath"
)

var (
	ControlPlaneComponents = []string{
		KubeAPIServer,
		KubeControllerManager,
		KubeScheduler,
	}
)

const (
	// KubernetesDir is the directory Kubernetes owns for storing various configuration files
	KubernetesDir = "/etc/kubernetes"
	// ManifestsSubDirName defines directory name to store manifests
	ManifestsSubDirName = "manifests"
	// Etcd defines variable used internally when referring to etcd component
	Etcd = "etcd"
	// KubeAPIServer defines variable used internally when referring to kube-apiserver component
	KubeAPIServer = "kube-apiserver"
	// KubeControllerManager defines variable used internally when referring to kube-controller-manager component
	KubeControllerManager = "kube-controller-manager"
	// KubeScheduler defines variable used internally when referring to kube-scheduler component
	KubeScheduler = "kube-scheduler"
	// Kubelet defines variable used internally when referring to the Kubelet
	Kubelet             = "kubelet"
	AuditPolicyFileName = "audit-policy.yaml"
	// EtcdDataDir defines variable used internally when referring to etcd data directory
	EtcdDataDir = "/var/lib/etcd"
)

const (
	// HAProxy defines variable used internally when referring to the HAProxy component
	HAProxy = "haproxy"
	//Keepalived defines variable used internally when referring to the keepalived component
	Keepalived = "keepalived"
	// IngressKeepalived
	IngressKeepalived = "ingress-Keepalived"

	HAProxyConfPath = "/etc/haproxy"
	HAProxyConfName = "haproxy.cfg"

	KeepAlivedConfPath = "/etc/keepalived"
	KeepAlivedConfName = "keepalived.conf"

	MasterKeepalivedInstanceReg  = `(?ms)^\s*#master-instance-start.*#master-instance-end\s*$`
	IngressKeepalivedInstanceReg = `(?ms)^\s*#ingress-instance-start.*#ingress-instance-end\s*$`
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
