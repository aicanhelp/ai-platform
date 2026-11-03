/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
 */

// Package cluster provides a suite og tools adn utilities for managing cluster objects.
package cluster

import (
	"encoding/json"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/clientcmd/api"
)

const (
	karmadaConfigPath = "/etc/karmada/karmada-apiserver.config"
)

// InfoOfCluster defines the essential details required to connect and manage a Kubernetes cluster.
// This includes general information about cluster, access configurations, and the path to the Kubeconfig file.
type InfoOfCluster struct {
	// Name is the cluster name
	Name string `json:"name"`

	// Kubeconfig contains the kubeconfig data which is necessary to connect to the kubernetes cluster.
	// Kubeconfig files provide the nesessary details to access the cluster, such as cluster API server URL,
	// credentials, and context.
	// Kubeconfig api.Config `json:"kubeconfig"`
	// Kubeconfig map[string]interface{} `json:"kubeconfig"`
	Kubeconfig json.RawMessage `json:"kubeconfig"`

	// KubeconfigPath specifies the file path to the kubeconfig file assocaited with this cluster.
	KubeconfigPath string `json:"path"`

	// ClusterLabels define labels to the specified cluster.
	ClusterLabels map[string]string `json:"clusterlabels"`
}

// OptionsOfCluster defines the option ifor cluster registrations.
type OptionsOfCluster struct {
	// ClusterName is the cluster's name.
	ClusterName string

	// ClusterAPIEndpoint is the cluster's server host.
	ClusterAPIEndpoint string

	// ClusterID is the unqiue UID for the cluster.
	ClusterID string

	// ClusterLabels are the labels for the clsuter.
	ClusterLabels map[string]string

	// ClusterTLSVerify defines if the cluster needs TLS.
	ClusterTLSVerify bool

	// ClusterKubeconfig is the cluster's kubeconfig.
	ClusterKubeconfig *api.Config

	// ClusterKubeconfigData is the cluster's kubeconfig data.
	ClusterKubeconfigData []byte

	// ClusterKubeconfigPath is the cluster's kubeconfig path.
	ClusterKubeconfigPath string

	// ClusterSecret records credentials for Karmada APIServer to the Cluster APIServer.
	ClusterSecret corev1.Secret

	// ImpersonatorSecret records credentials for Karmada APIServer to teh Cluster APIServer.
	ImpersonatorSecret corev1.Secret

	// OutClusterConfig is the cluster's outcluster kubeconfig data.
	OutClusterConfig []byte
}

// JoinOptions contains settings and identifiers required for the join command.
// It includes unique identifiers like cluster name and necessary configuration details.
type JoinOptions struct {
	// ClusterName is the cluster name.
	ClusterName string `json:"clustername"`

	// Kubeconfig contains the kubeconfig data which is necessary to connect to the kubernetes cluster.
	Kubeconfig json.RawMessage `json:"kubeconfig"`

	// KubeconfigPath specifies the file path to the kubeconfig file assocaited with this cluster.
	KubeconfigPath string `json:"path"`

	// ClusterLabels define labels to the specified cluster.
	ClusterLabels map[string]string `json:"clusterlabels,omitempty"`
}

// UnjoinOptions includes identifiers necessary for teh removing command.
type UnjoinOptions struct {
	// ClusterName is the cluster name.
	ClusterName string `json:"clustername"`
}

// EditOptions holds settings for modifying the configuration or properties of a cluster.
type EditOptions struct {
	// ClusterName is the cluster name.
	ClusterName string `json:"clustername"`

	// ClusterLabels define labels to the specified cluster.
	ClusterLabels map[string]string `json:"clusterlabels,omitempty"`
}

// InspectOptions provides the necessary parameters for fetching detailed information about cluster status.
type InspectOptions struct {
	// ClusterName is the cluster name.
	ClusterName string `json:"clustername"`
}

// AuthorizeOptions carries user credentials and permissions required to authenticate an authorize user access
// to cluster resources.
type AuthorizeOptions struct {
	// users contain all users' information.
	Users []*IdentityDescriptor `json:"users,omitempty"`
}

// TagOfCluster defines labels added to the cluster object
type TagOfCluster struct {
	// Labels define labels added to the cluster object.
	ClusterLabels map[string]string `json:"clusterlabels,omitempty"`
}

// ResourcesInCluster defines the summary of resources in the member cluster.
type ResourcesInCluster struct {
	// Allocatable represents the resources of a cluster that are available for scheduling.
	// Total amount of allocatable resources on all nodes.
	Allocatable corev1.ResourceList `json:"allocatable,omitempty"`

	// Allocated represents the resources of a cluster that have been scheduled.
	// Total amount of required resources of all Pods that have been scheduled to nodes.
	Allocated corev1.ResourceList `json:"allocated,omitempty"`
}

// ListOfCluster defines a structure for holding information about cluster objects.
type ListOfCluster struct {
	// Info holds cluster information.
	Info map[string]*InformationOfCluster `json:"info,omitempty"`
}

// InformationOfCluster contains detailed information about a specified cluster,
// including name, labels and resource usage details.
type InformationOfCluster struct {
	// ClusterName defines name.
	ClusterName string `json:"clustername,omitempty"`
	// ClusterLabels defines labels.
	ClusterLabels []string `json:"clusterlabels,omitempty"`
	// ClusterResourcesUsage holds a resourceusage structure.
	ClusterResourcesUsage *ResourceUsage `json:"clusterresources,omitempty"`
	// ClusterConfig provides the outcluster kubeconfig.
	ClusterConfig json.RawMessage `json:"kubeconfig,omitempty"`
}

// ResourceUsage structure.
type ResourceUsage struct {
	// AllocatableCPU represents the CPU resources that are available for allocation.
	AllocatableCPU string `json:"allocatablecpu,omitempty"`

	// AllocatableMemory represents the memory reources that are available ofr allocation.
	AllocatableMemory string `json:"allocatablemem,omitempty"`

	// AllocatedCPU represents the CPU resources that have been allocated.
	AllocatedCPU string `json:"allocatedcpu,omitempty"`

	// AllocatedMemory represents the memory resources that have been allocated.
	AllocatedMemory string `json:"allocatedmem,omitempty"`
}

// IdentityCollection holds a map of user identity informations.
type IdentityCollection struct {
	UserInfo map[string]*IdentityDescriptor `json:"userinfo,omitempty"`
}

// IdentityDescriptor describes user identity.
// It includes the user's name, the user API Group, the user platform Role and a list of clusters they are members of.
type IdentityDescriptor struct {
	// Unique name identifying the user.
	IdentityName string `json:"identityname,omitempty"`
	// APIGroup that user belongs to.
	ApiGroup string `json:"apigroup,omitempty"`
	// Flag indicating whether the user has platform admin role.
	PlatformAdmin bool `json:"platformadmin"`
	// List of clusters the user if part of.
	MemberClusters []string `json:"memberclusters"`
}
