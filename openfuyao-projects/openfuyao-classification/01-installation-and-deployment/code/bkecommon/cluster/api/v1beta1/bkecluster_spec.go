/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// +k8s:deepcopy-gen=package

package v1beta1

import (
	"fmt"
	"net"
)

// BKEClusterSpec defines the desired state of BKECluster
type BKEClusterSpec struct {
	// ControlPlaneEndpoint represents the endpoint used to communicate with the control plane.
	// +optional
	ControlPlaneEndpoint APIEndpoint `json:"controlPlaneEndpoint"`

	// ClusterConfig defines the cluster config
	// +kubebuilder:object:generate:=true
	// +optional
	ClusterConfig *BKEConfig `json:"clusterConfig"`

	// KubeletConfigRef references a KubeletConfig to use for kubelet configuration
	// +optional
	KubeletConfigRef *KubeletConfigRef `json:"KubeletConfigRef,omitempty"`

	// ClusterFrom defines the manager cluster type, enum: bocloud, other
	// bocloud:  manage bocloud cluster ("bocloud" cluster type is a cluster deployed by ansible)
	// other:   manage other cluster ("other" cluster type is a cluster deployed by such as kubeadm etc)
	// 3.when ClusterFrom is "other", BKE only collect the cluster information, and does not manage the cluster,
	// so such as pause, dryRun, reset is not supported
	// 4.when ClusterFrom is "bocloud", BKE will collect the cluster information, and manage the cluster.
	// if bocloud cluster k8s version less than 1.21.1, need to use ansible to upgrade to 1.21.1, and then use BKE to upgrade to 1.25.6(plan version is 1.27)
	// after this upgrade, BKE will take over the life cycle management of the cluster, including upgrading, scaling, etc

	// Pause is used to pause reconciliation of the BKECluster, it also pauses the BKECluster's machines.
	// +optional
	Pause bool `json:"pause"`

	//DryRun is used to dry run the BKECluster, it also dries run the BKECluster's machines.
	// +optional
	DryRun bool `json:"dryRun,omitempty"`

	// Reset is used to reset the BKECluster, it also resets the BKECluster's machines, include cluster-api Cluster Machine etc.
	Reset bool `json:"reset,omitempty"`
}

type KubeletConfigRef struct {
	// +optional
	Name string `json:"name"`

	// +optional
	Namespace string `json:"namespace,omitempty"`
}

// +kubebuilder:object:generate:=true
type BKEConfig struct {
	// Cluster defines the configuration of the target cluster
	// +optional
	Cluster Cluster `json:"cluster,omitempty"`

	// Addons defines the addons that the target cluster will install after deployment
	// +optional
	Addons []Product `json:"addons,omitempty"`

	// CustomArgs defines the custom args
	// +optional
	CustomExtra map[string]string `json:"customExtra,omitempty"`
}

type APIEndpoint struct {
	// Host sets the Host for the API server to advertise.
	// +optional
	Host string `json:"host,omitempty"`

	// Port sets the secure port for the API Server to bind to. Defaults to 6443.
	// +optional
	Port int32 `json:"port,omitempty"`
}

// Label represents a key-value pair used for setting labels on Kubernetes nodes
type Label struct {
	// +optional
	Key string `json:"key,omitempty"`
	// +optional
	Value string `json:"value,omitempty"`
}

type Cluster struct {
	// ControlPlane defines the configuration of all the control plane nodes in the target cluster
	// will be overwritten by the configuration in node
	// +optional
	ControlPlane `json:",omitempty"`

	// Kubelet define kubelet configuration for all nodes in the target cluster
	// +optional
	Kubelet *Kubelet `json:"kubelet,omitempty"`

	// Networking defines the configuration of target cluster network
	// +optional
	Networking Networking `json:"networking,omitempty"`

	// HTTPRepo defines the HTTP repository to use when deploying
	// rpm / deb / http server
	// +optional
	HTTPRepo Repo `json:"httpRepo,omitempty"`

	// ImageRepo defines the global image repository of the deployment target cluster
	// +optional
	ImageRepo Repo `json:"imageRepo,omitempty"`

	// ChartRepo defines the global chart repository of the deployment target cluster
	// +optional
	ChartRepo Repo `json:"chartRepo,omitempty"`

	// ContainerRuntime defines the container runtime of the target cluster
	// +optional
	ContainerRuntime ContainerRuntime `json:"containerRuntime,omitempty"`

	// ContainerdConfigRef references a ContainerdConfig custom resource for advanced containerd configuration
	// If specified, this will override the default containerd configuration
	// +optional
	ContainerdConfigRef *ContainerdConfigRef `json:"containerdConfigRef,omitempty"`

	// +optional
	OpenFuyaoVersion string `json:"openFuyaoVersion,omitempty"`

	// +optional
	ContainerdVersion string `json:"containerdVersion,omitempty"`

	// KubernetesVersion defines the Kubernetes version of the target cluster
	// support up to v1.25.6 in bke
	// +optional
	KubernetesVersion string `json:"kubernetesVersion,omitempty"`

	// EtcdVersion defines the Etcd version of the target cluster
	// +optional
	EtcdVersion string `json:"etcdVersion,omitempty"`

	// CertificatesDir defines the directory path for storing or locating all required certificates.
	// +optional
	CertificatesDir string `json:"certificatesDir"`

	// NTPServer defines the ntp server information used for time synchronization
	// +required
	NTPServer string `json:"ntpServer,omitempty"`

	// AgentHealthPort defines the agent health port
	// +optional
	AgentHealthPort string `json:"agentHealthPort,omitempty"`

	// Global node labels
	// +optional
	Labels []Label `json:"labels,omitempty"`
}

type ContainerRuntime struct {
	// CRI defines the name of the runtime
	// +optional
	// +kubebuilder:validation:Enum=docker;containerd
	CRI string `json:"cri,omitempty"`

	// Runtime defines the lower runtime of the runtime
	// +kubebuilder:validation:Enum=runc;richrunc;kata
	Runtime string `json:"runtime,omitempty"`

	// Param defines the param of the runtime
	Param map[string]string `json:"param,omitempty"`
}

// ContainerdConfigRef references a ContainerdConfig custom resource
type ContainerdConfigRef struct {
	// Name of the ContainerdConfig resource
	// +required
	Name string `json:"name"`

	// Namespace of the ContainerdConfig resource
	// If empty, defaults to the same namespace as the Cluster resource
	// +optional
	Namespace string `json:"namespace,omitempty"`
}

type Repo struct {
	// Domain defines the Repo domain name
	// +optional
	Domain string `json:"domain,omitempty"`
	// Ip defines the Repo ip
	// +optional
	Ip string `json:"ip,omitempty"`
	// Port defines the number of port to connect to the Repo
	// +optional
	Port string `json:"port,omitempty"`
	// Prefix defines the kubernetes image address
	// +optional
	Prefix string `json:"prefix"`

	// AuthSecretRef defines the secret name, namespace and other information for authentication
	// +optional
	AuthSecretRef *AuthSecretRef `json:"authSecretRef,omitempty"`
	// TlsSecretRef defines the secret name, namespace and other information for TLS certificates
	// +optional
	TlsSecretRef *TlsSecretRef `json:"tlsSecretRef,omitempty"`
	// InsecureSkipTLSVerify defines whether to skip TLS verification when connecting to the repo
	// If empty, defaults to false
	// +optional
	InsecureSkipTLSVerify bool `json:"insecureSkipTLSVerify,omitempty"`
}

// AuthSecretRef is a reference to a secret containing authentication credentials for a repo.
type AuthSecretRef struct {
	// Name of the AuthSecretRef resource
	// +required
	Name string `json:"name"`
	// Namespace of the AuthSecretRef resource
	// If empty, defaults to the same namespace as the Cluster resource
	// +optional
	Namespace string `json:"namespace,omitempty"`
	// UsernameKey is the key name that stores the username in the secret
	// If empty, defaults to "username"
	// +optional
	UsernameKey string `json:"usernameKey,omitempty"`
	// PasswordKey is the key name that stores the password in the secret
	// If empty, defaults to "password"
	// +optional
	PasswordKey string `json:"passwordKey,omitempty"`
}

// TlsSecretRef is a reference to a secret containing TLS certificates for a repo.
type TlsSecretRef struct {
	// Name of the TlsSecretRef resource
	// +required
	Name string `json:"name"`
	// Namespace of the TlsSecretRef resource
	// If empty, defaults to the same namespace as the Cluster resource
	// +optional
	Namespace string `json:"namespace,omitempty"`
	// CaKey is the key name that stores the ca.crt in the secret
	// If empty, defaults to "ca.crt"
	// +optional
	CaKey string `json:"caKey,omitempty"`
	// CertKey is the key name that stores the cert.crt in the secret
	// If empty, defaults to "cert.crt"
	// +optional
	CertKey string `json:"certKey,omitempty"`
	// KeyKey is the key name that stores the key.key in the secret
	// If empty, defaults to "key.key"
	// +optional
	KeyKey string `json:"keyKey,omitempty"`
}

// Networking defines the network configuration settings for the cluster.
type Networking struct {
	// ServiceSubnet specifies the CIDR block for Kubernetes services.
	// If not specified, defaults to "10.96.0.0/12".
	// +optional
	ServiceSubnet string `json:"serviceSubnet,omitempty"`
	// PodSubnet specifies the CIDR block for Pod IP addresses.
	// +optional
	PodSubnet string `json:"podSubnet,omitempty"`
	// DNSDomain specifies the DNS domain suffix for Kubernetes services.
	// If not specified, defaults to "cluster.local".
	// +optional
	DNSDomain string `json:"dnsDomain,omitempty"`
}

// Node is an alias for BKENodeSpec, kept for backward compatibility.
// The actual node configuration is now defined in BKENode CRD.
type Node = BKENodeSpec

type Product struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
	// Type defines the product type, such as "chart", "yaml"
	// If empty, defaults to "yaml"
	// +optional
	// +kubebuilder:validation:Enum=yaml;chart
	Type string `json:"type,omitempty"`
	// ReleaseName defines the release name of the chart
	// If empty, defaults to the product name
	// +optional
	ReleaseName string `json:"releaseName,omitempty"`
	// Namespace defines the namespace of the chart
	// If empty, use the default configuration of chart
	// +optional
	Namespace string `json:"namespace,omitempty"`
	// ValuesConfigMapRef references a ConfigMap containing the values.yaml for the chart
	// If empty, use the default configuration of chart
	// +optional
	ValuesConfigMapRef *ValuesConfigMapRef `json:"valuesConfigMapRef,omitempty"`
	// Timeout defines the timeout for the Product to be deployed\removed\upgraded successfully
	// If empty, defaults to 300 seconds
	// +optional
	Timeout int `json:"timeout,omitempty"`

	// +optional
	Param map[string]string `json:"param,omitempty"`
	// Block defines fully wait for the Product to be deployed successfully
	// +optional
	// +kubebuilder:default=false
	Block bool `json:"block,omitempty"`
}

type ValuesConfigMapRef struct {
	// Name of the ValuesConfigMapRef resource
	// +required
	Name string `json:"name"`
	// Namespace of the ValuesConfigMapRef resource
	// If empty, defaults to the same namespace as the Cluster resource
	// +optional
	Namespace string `json:"namespace,omitempty"`
	// ValuesKey is the key name that stores the values.yaml in the ConfigMap
	// If empty, defaults to "values.yaml"
	// +optional
	ValuesKey string `json:"valuesKey,omitempty"`
}

type ControlPlane struct {
	// ControllerManager contains additional settings for the controller manager component
	// +optional
	ControllerManager *ControlPlaneComponent `json:"controllerManager,omitempty"`

	// Scheduler contains additional settings for the scheduler component
	// +optional
	Scheduler *ControlPlaneComponent `json:"scheduler,omitempty"`

	// APIServer contains additional settings for the API server component
	// +optional
	APIServer *APIServer `json:"apiServer,omitempty"`

	// Etcd contains configuration for etcd
	// +optional
	Etcd *Etcd `json:"etcd,omitempty"`
}

// ControlPlaneComponent defines common settings for the control plane components of the cluster
type ControlPlaneComponent struct {
	// ExtraArgs specifies additional command line flags to pass to the control plane component
	// +optional
	ExtraArgs map[string]string `json:"extraArgs,omitempty"`

	// ExtraVolumes specifies additional host volumes to mount to the control plane component
	// +optional
	ExtraVolumes []HostPathMount `json:"extraVolumes,omitempty"`
}

// HostPathMount describes volumes that are mounted from the host into pods
type HostPathMount struct {
	// Name specifies the name of the volume within the pod template
	Name string `json:"name,omitempty"`
	// HostPath specifies the path on the host that will be mounted into the pod
	HostPath string `json:"hostPath,omitempty"`
	// MountPath specifies the path inside the pod where the hostPath will be mounted
	MountPath string `json:"mountPath,omitempty"`
	// ReadOnly specifies whether the volume should be mounted as read-only
	ReadOnly bool `json:"readOnly,omitempty"`
	// PathType specifies the type of the HostPath
	PathType string `json:"pathType,omitempty"`
}

// APIServer contains configuration settings for the API server deployments in the cluster
type APIServer struct {
	// +optional
	APIEndpoint `json:",omitempty"`
	// +optional
	ControlPlaneComponent `json:",omitempty"`
	// CertSANs sets extra Subject Alternative Names for the API Server signing certificate
	// +optional
	CertSANs []string `json:"certSANs,omitempty"`
}

// Etcd defines the configuration settings for the etcd distributed key-value store.
type Etcd struct {
	// +optional
	ControlPlaneComponent `json:",omitempty"`
	// DataDir specifies the directory path where etcd will store its data.
	// If not specified, defaults to "/var/lib/etcd".
	// +optional
	DataDir string `json:"dataDir,omitempty"`
	// ServerCertSANs defines additional Subject Alternative Names (SANs) for the etcd server certificate.
	// +optional
	ServerCertSANs []string `json:"serverCertSANs,omitempty"`
	// PeerCertSANs defines additional Subject Alternative Names (SANs) for the etcd peer-to-peer communication certificate.
	// +optional
	PeerCertSANs []string `json:"peerCertSANs,omitempty"`
}

// Kubelet contains elements describing kubelet configuration.
type Kubelet struct {
	// +optional
	ControlPlaneComponent `json:",omitempty"`
	// ManifestsDir is the directory where kubelet will store manifests
	// +optional
	ManifestsDir string `json:"manifestsDir,omitempty"`
}

// IsZero returns true if both host and port are zero values.
func (v APIEndpoint) IsZero() bool {
	return v.Host == "" && v.Port == 0
}

// IsValid returns true if both host and port are non-zero values.
func (v APIEndpoint) IsValid() bool {
	return v.Host != "" && v.Port != 0
}

// String returns a formatted version HOST:PORT of this APIEndpoint.
func (v APIEndpoint) String() string {
	return net.JoinHostPort(v.Host, fmt.Sprintf("%d", v.Port))
}
