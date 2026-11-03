/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original file: https://github.com/kubernetes/kubernetes/blob/release-1.21/cmd/kubeadm/app/constants/constants.go
*/

package pkiutil

import (
	"time"
)

const (
	// KubernetesDir is the directory Kubernetes owns for storing various configuration files
	KubernetesDir = "/etc/kubernetes"
	pemDir        = "/etc/kubernetes/pki"
	// CACertAndKeyBaseName defines certificate authority base name
	CACertAndKeyBaseName = "ca"
	// CACertName defines certificate name
	CACertName = "ca.crt"
	// CAKeyName defines certificate name
	CAKeyName = "ca.key"

	// ClusterCACertAndKeyBaseName defines cluster certificate authority base name
	ClusterCACertAndKeyBaseName = "cluster-ca"
	// ClusterCACertName defines certificate name
	ClusterCACertName = "cluster-ca.crt"
	// ClusterCAKeyName defines certificate name
	ClusterCAKeyName = "cluster-ca.key"

	// GlobalCACertAndKeyBaseName defines cluster certificate authority base name
	GlobalCACertAndKeyBaseName = "global-ca"
	// GlobalCACertName defines certificate name
	GlobalCACertName = "global-ca.crt"
	// GlobalCAKeyName defines certificate name
	GlobalCAKeyName = "global-ca.key"

	// CertConfigMapName is the ConfigMap name for certificate configuration
	CertConfigMapName = "cluster-cert-config"
	// CertConfigMapNamespace is the ConfigMap namespace for certificate configuration
	CertConfigMapNamespace = "kube-system"
	// CertConfigDir is the directory for saving certification config json files
	CertConfigDir = "/etc/kubernetes/certs/config"

	// APIServerCertAndKeyBaseName defines API's server certificate and key base name
	APIServerCertAndKeyBaseName = "apiserver"
	// APIServerCertName defines API's server certificate name
	APIServerCertName = "apiserver.crt"
	// APIServerKeyName defines API's server key name
	APIServerKeyName = "apiserver.key"
	// APIServerCertCommonName defines API's server certificate common name (CN)
	APIServerCertCommonName = "kube-apiserver"

	// APIServerKubeletClientCertAndKeyBaseName defines kubelet client certificate and key base name
	APIServerKubeletClientCertAndKeyBaseName = "apiserver-kubelet-client"
	// APIServerKubeletClientCertName defines kubelet client certificate name
	APIServerKubeletClientCertName = "apiserver-kubelet-client.crt"
	// APIServerKubeletClientKeyName defines kubelet client key name
	APIServerKubeletClientKeyName = "apiserver-kubelet-client.key"
	// APIServerKubeletClientCertCommonName defines kubelet client certificate common name (CN)
	APIServerKubeletClientCertCommonName = "kube-apiserver-kubelet-client"

	// EtcdCACertAndKeyBaseName defines etcd's CA certificate and key base name
	EtcdCACertAndKeyBaseName = "etcd/ca"
	// EtcdCACertName defines etcd's CA certificate name
	EtcdCACertName = "etcd/ca.crt"
	// EtcdCAKeyName defines etcd's CA key name
	EtcdCAKeyName = "etcd/ca.key"

	// EtcdServerCertAndKeyBaseName defines etcd's server certificate and key base name
	EtcdServerCertAndKeyBaseName = "etcd/server"
	// EtcdServerCertName defines etcd's server certificate name
	EtcdServerCertName = "etcd/server.crt"
	// EtcdServerKeyName defines etcd's server key name
	EtcdServerKeyName = "etcd/server.key"

	// EtcdListenClientPort defines the port etcd listen on for client traffic
	EtcdListenClientPort = 2379
	// EtcdMetricsPort is the port at which to obtain etcd metrics and health status
	EtcdMetricsPort = 2381

	// KubeletBootstrapKubeConfigFileName defines the file name for the kubeconfig that the kubelet will use to do
	// the TLS bootstrap to get itself an unique credential
	KubeletBootstrapKubeConfigFileName = "bootstrap-kubelet.conf"

	// EtcdPeerCertAndKeyBaseName defines etcd's peer certificate and key base name
	EtcdPeerCertAndKeyBaseName = "etcd/peer"
	// EtcdPeerCertName defines etcd's peer certificate name
	EtcdPeerCertName = "etcd/peer.crt"
	// EtcdPeerKeyName defines etcd's peer key name
	EtcdPeerKeyName = "etcd/peer.key"

	// EtcdListenPeerPort defines the port etcd listen on for peer traffic
	EtcdListenPeerPort = 2380

	// EtcdHealthcheckClientCertAndKeyBaseName defines etcd's healthcheck client certificate and key base name
	EtcdHealthcheckClientCertAndKeyBaseName = "etcd/healthcheck-client"
	// EtcdHealthcheckClientCertName defines etcd's healthcheck client certificate name
	EtcdHealthcheckClientCertName = "etcd/healthcheck-client.crt"
	// EtcdHealthcheckClientKeyName defines etcd's healthcheck client key name
	EtcdHealthcheckClientKeyName = "etcd/healthcheck-client.key"
	// EtcdHealthcheckClientCertCommonName defines etcd's healthcheck client certificate common name (CN)
	EtcdHealthcheckClientCertCommonName = "kube-etcd-healthcheck-client"

	// APIServerEtcdClientCertAndKeyBaseName defines apiserver's etcd client certificate and key base name
	APIServerEtcdClientCertAndKeyBaseName = "apiserver-etcd-client"
	// APIServerEtcdClientCertName defines apiserver's etcd client certificate name
	APIServerEtcdClientCertName = "apiserver-etcd-client.crt"
	// APIServerEtcdClientKeyName defines apiserver's etcd client key name
	APIServerEtcdClientKeyName = "apiserver-etcd-client.key"
	// APIServerEtcdClientCertCommonName defines apiserver's etcd client certificate common name (CN)
	APIServerEtcdClientCertCommonName = "kube-apiserver-etcd-client"

	// ServiceAccountKeyBaseName defines SA key base name
	ServiceAccountKeyBaseName = "sa"
	// ServiceAccountPublicKeyName defines SA public key base name
	ServiceAccountPublicKeyName = "sa.pub"
	// ServiceAccountPrivateKeyName defines SA private key base name
	ServiceAccountPrivateKeyName = "sa.key"

	// FrontProxyCACertAndKeyBaseName defines front proxy CA certificate and key base name
	FrontProxyCACertAndKeyBaseName = "front-proxy-ca"
	// FrontProxyCACertName defines front proxy CA certificate name
	FrontProxyCACertName = "front-proxy-ca.crt"
	// FrontProxyCAKeyName defines front proxy CA key name
	FrontProxyCAKeyName = "front-proxy-ca.key"

	// FrontProxyClientCertAndKeyBaseName defines front proxy certificate and key base name
	FrontProxyClientCertAndKeyBaseName = "front-proxy-client"
	// FrontProxyClientCertName defines front proxy certificate name
	FrontProxyClientCertName = "front-proxy-client.crt"
	// FrontProxyClientKeyName defines front proxy key name
	FrontProxyClientKeyName = "front-proxy-client.key"
	// FrontProxyClientCertCommonName defines front proxy certificate common name
	FrontProxyClientCertCommonName = "front-proxy-client" // used as subject.commonname attribute (CN)

	// ControllerManagerUser defines the well-known user the controller-manager should be authenticated as
	ControllerManagerUser = "system:kube-controller-manager"

	// ControllerManagerCertAndKeyBaseName defines the controller manager base name
	ControllerManagerCertAndKeyBaseName = "controller-manager"

	// SchedulerUser defines the well-known user the scheduler should be authenticated as
	SchedulerUser = "system:kube-scheduler"

	// SchedulerCertAndKeyBaseName defines scheduler certificate and key base name
	SchedulerCertAndKeyBaseName = "scheduler"

	// AggregatorCommonName defines aggregator certificate and key base name
	AggregatorCommonName = "system:aggregator"
	// AggregatorCertAndKeyBaseName defines aggregator certificate and key base name
	AggregatorCertAndKeyBaseName = "aggregator"

	// SystemPrivilegedGroup defines the well-known group for the apiservers. This group is also superuser by default
	// (i.e. bound to the cluster-admin ClusterRole)
	SystemPrivilegedGroup = "system:masters"
	// NodesGroup defines the well-known group for all nodes.
	NodesGroup = "system:nodes"
	// NodesUserPrefix defines the user name prefix as requested by the Node authorizer.
	NodesUserPrefix = "system:node:"
	// NodesClusterRoleBinding defines the well-known ClusterRoleBinding which binds the too permissive system:node
	// ClusterRole to the system:nodes group. Since kubeadm is using the Node Authorizer, this ClusterRoleBinding's
	// system:nodes group subject is removed if present.
	NodesClusterRoleBinding = "system:node"

	// APICallRetryInterval defines how long kubeadm should wait before retrying a failed API operation
	APICallRetryInterval = 500 * time.Millisecond
	// DiscoveryRetryInterval specifies how long kubeadm should wait before retrying to connect to the control-plane
	// when doing discovery
	DiscoveryRetryInterval = 5 * time.Second
	// PatchNodeTimeout specifies how long kubeadm should wait for applying the label and taint on the control-plane
	// before timing out
	PatchNodeTimeout = 2 * time.Minute
	// TLSBootstrapTimeout specifies how long kubeadm should wait for the kubelet to perform the TLS Bootstrap
	TLSBootstrapTimeout = 5 * time.Minute
	// TLSBootstrapRetryInterval specifies how long kubeadm should wait before retrying the TLS Bootstrap check
	TLSBootstrapRetryInterval = 5 * time.Second
	// APICallWithWriteTimeout specifies how long kubeadm should wait for api calls with at least one write
	APICallWithWriteTimeout = 40 * time.Second
	// APICallWithReadTimeout specifies how long kubeadm should wait for api calls with only reads
	APICallWithReadTimeout = 15 * time.Second
	// PullImageRetry specifies how many times ContainerRuntime retries when pulling image failed
	PullImageRetry = 5

	// DefaultControlPlaneTimeout specifies the default control plane (actually API Server) timeout for use by kubeadm
	DefaultControlPlaneTimeout = 4 * time.Minute

	// MinimumAddressesInServiceSubnet defines minimum amount of nodes the Service subnet should allow.
	// We need at least ten, because the DNS service is always at the tenth cluster clusterIP
	MinimumAddressesInServiceSubnet = 10

	// MaximumBitsForServiceSubnet defines maximum possible size of the service subnet in terms of bits.
	// For example, if the value is 20, then the largest supported service subnet is /12 for IPv4 and /108 for IPv6.
	// Note however that anything in between /108 and /112 will be clamped to /112 due to the limitations of the
	// underlying allocation logic.
	MaximumBitsForServiceSubnet = 20

	// MinimumAddressesInPodSubnet defines minimum amount of pods in the cluster.
	// We need at least more than services, an IPv4 /28 or IPv6 /128 subnet means 14 util addresses
	MinimumAddressesInPodSubnet = 14

	// PodSubnetNodeMaskMaxDiff is limited to 16 due to an issue with uncompressed IP bitmap in core:
	// xref: #44918
	// The node subnet mask size must be no more than the pod subnet mask size + 16
	PodSubnetNodeMaskMaxDiff = 16

	// DefaultTokenDuration specifies the default amount of time that a bootstrap token will be valid
	// Default behaviour is 24 hours
	DefaultTokenDuration = 24 * time.Hour

	// DefaultCertTokenDuration specifies the default amount of time that the token used by upload certs will be valid
	// Default behaviour is 2 hours
	DefaultCertTokenDuration = 2 * time.Hour

	// CertificateKeySize specifies the size of the key used to encrypt certificates on uploadcerts phase
	CertificateKeySize = 32

	// LabelNodeRoleOldControlPlane specifies that a node hosts control-plane components
	// DEPRECATED: https://github.com/kubernetes/kubeadm/issues/2200
	LabelNodeRoleOldControlPlane = "node-role.kubernetes.io/master"

	// LabelNodeRoleControlPlane specifies that a node hosts control-plane components
	LabelNodeRoleControlPlane = "node-role.kubernetes.io/control-plane"

	// LabelExcludeFromExternalLB can be set on a node to exclude it from external load balancers.
	// This is added to control plane nodes to preserve backwards compatibility with a legacy behavior.
	LabelExcludeFromExternalLB = "node.kubernetes.io/exclude-from-external-load-balancers"

	// AnnotationKubeadmCRISocket specifies the annotation kubeadm uses to preserve the crisocket information given to
	// kubeadm at init/join time for use later. kubeadm annotates the node object with this information
	AnnotationKubeadmCRISocket = "kubeadm.alpha.kubernetes.io/cri-socket"

	// UnknownCRISocket defines the undetected or unknown CRI socket
	UnknownCRISocket = "/var/run/unknown.sock"

	// KubeadmConfigConfigMap specifies in what ConfigMap in the kube-system namespace the `kubeadm init` configuration
	// should be stored
	KubeadmConfigConfigMap = "kubeadm-config"

	// ClusterConfigurationConfigMapKey specifies in what ConfigMap key the cluster configuration should be stored
	ClusterConfigurationConfigMapKey = "ClusterConfiguration"

	// ClusterStatusConfigMapKey specifies in what ConfigMap key the cluster status should be stored
	ClusterStatusConfigMapKey = "ClusterStatus"

	// KubeProxyConfigMap specifies in what ConfigMap in the kube-system namespace the kube-proxy configuration should
	// be stored
	KubeProxyConfigMap = "kube-proxy"

	// KubeProxyConfigMapKey specifies in what ConfigMap key the component config of kube-proxy should be stored
	KubeProxyConfigMapKey = "config.conf"

	// KubeletBaseConfigurationConfigMapPrefix specifies in what ConfigMap in the kube-system namespace the initial
	// remote configuration of kubelet should be stored
	KubeletBaseConfigurationConfigMapPrefix = "kubelet-config-"

	// KubeletBaseConfigurationConfigMapKey specifies in what ConfigMap key the initial remote configuration of kubelet
	// should be stored
	KubeletBaseConfigurationConfigMapKey = "kubelet"

	// KubeletBaseConfigMapRolePrefix defines the base kubelet configuration ConfigMap.
	KubeletBaseConfigMapRolePrefix = "kubeadm:kubelet-config-"

	// KubeletRunDirectory specifies the directory where the kubelet runtime information is stored.
	KubeletRunDirectory = "/var/lib/kubelet"

	// KubeletConfigurationFileName specifies the file name on the node which stores initial remote
	// configuration of kubelet
	// This file should exist under KubeletRunDirectory
	KubeletConfigurationFileName = "config.yaml"

	// KubeletEnvFileName is a file "kubeadm init" writes at runtime. Using that interface, kubeadm can customize
	// certain kubelet flags conditionally based on the environment at runtime. Also, parameters given to the
	// configuration file might be passed through this file. "kubeadm init" writes one variable, with the name
	// ${KubeletEnvFileVariableName}. This file should exist under KubeletRunDirectory
	KubeletEnvFileName = "kubeadm-flags.env"

	// KubeletEnvFileVariableName specifies the shell script variable name "kubeadm init" should write a value to in
	// KubeletEnvFile
	KubeletEnvFileVariableName = "KUBELET_KUBEADM_ARGS"

	// KubeletHealthzPort is the port of the kubelet healthz endpoint
	KubeletHealthzPort = 10248

	// MinExternalEtcdVersion indicates minimum external etcd version which kubeadm supports
	MinExternalEtcdVersion = "3.2.18"

	// KubeCertificatesVolumeName specifies the name for the Volume that is used for injecting certificates to
	// control plane components (can be both a hostPath volume or a projected, all-in-one volume)
	KubeCertificatesVolumeName = "k8s-certs"

	// KubeConfigVolumeName specifies the name for the Volume that is used for injecting the kubeconfig to talk
	// securely to the api server for a control plane component if applicable
	KubeConfigVolumeName = "kubeconfig"

	// NodeBootstrapTokenAuthGroup specifies which group a Node Bootstrap Token should be authenticated in
	NodeBootstrapTokenAuthGroup = "system:bootstrappers:kubeadm:default-node-token"

	// DefaultCIImageRepository points to image registry where CI uploads images from ci-cross build job
	DefaultCIImageRepository = "gcr.io/k8s-staging-ci-images"

	// CoreDNSConfigMap specifies in what ConfigMap in the kube-system namespace the CoreDNS config should be stored
	CoreDNSConfigMap = "coredns"

	// CoreDNSDeploymentName specifies the name of the Deployment for CoreDNS add-on
	CoreDNSDeploymentName = "coredns"

	// CoreDNSVersion is the version of CoreDNS to be deployed if it is used
	CoreDNSVersion = "v1.8.0"

	// ClusterConfigurationKind is the string kind value for the ClusterConfiguration struct
	ClusterConfigurationKind = "ClusterConfiguration"

	// InitConfigurationKind is the string kind value for the InitConfiguration struct
	InitConfigurationKind = "InitConfiguration"

	// JoinConfigurationKind is the string kind value for the JoinConfiguration struct
	JoinConfigurationKind = "JoinConfiguration"

	// YAMLDocumentSeparator is the separator for YAML documents
	YAMLDocumentSeparator = "---\n"

	// DefaultAPIServerBindAddress is the default bind address for the API Server
	DefaultAPIServerBindAddress = "0.0.0.0"

	// ControlPlaneNumCPU is the number of CPUs required on control-plane
	ControlPlaneNumCPU = 2

	// ControlPlaneMem is the number of megabytes of memory required on the control-plane
	// Below that amount of RAM running a stable control plane would be difficult.
	ControlPlaneMem = 1700

	// KubeadmCertsSecret specifies in what Secret in the kube-system namespace the certificates should be stored
	KubeadmCertsSecret = "kubeadm-certs"

	// KubeletPort is the default port for the kubelet server on each host machine.
	// May be overridden by a flag at startup.
	KubeletPort = 10250
	// KubeSchedulerPort is the default port for the scheduler status server.
	// May be overridden by a flag at startup.
	KubeSchedulerPort = 10259
	// KubeControllerManagerPort is the default port for the controller manager status server.
	// May be overridden by a flag at startup.
	KubeControllerManagerPort = 10257

	// EtcdAdvertiseClientUrlsAnnotationKey is the annotation key on every etcd pod, describing the
	// advertise client URLs
	EtcdAdvertiseClientUrlsAnnotationKey = "kubeadm.kubernetes.io/etcd.advertise-client-urls"
	// KubeAPIServerAdvertiseAddressEndpointAnnotationKey is the annotation key on every apiserver pod,
	// describing the API endpoint (advertise address and bind port of the api server)
	KubeAPIServerAdvertiseAddressEndpointAnnotationKey = "kubeadm.kubernetes.io/kube-apiserver.advertise-address." +
		"endpoint"
	// ComponentConfigHashAnnotationKey holds the config map annotation key that kubeadm uses to store
	// a SHA256 sum to check for user changes
	ComponentConfigHashAnnotationKey = "kubeadm.kubernetes.io/component-config.hash"

	// ControlPlaneTier is the value used in the tier label to identify control plane components
	ControlPlaneTier = "control-plane"

	// DirDefaultPermission is the default permission for creating directories (rwxr-xr-x)
	DirDefaultPermission = 0755
	// FileDefaultPermission is the default permission for creating files (rw-r--r--)
	FileDefaultPermission = 0644
)
