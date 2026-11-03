/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// TenantClusterSpec defines the desired state of TenantCluster.
type TenantClusterSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// +kubebuilder:validation:Enum=managed;unmanaged
	Mode     string   `json:"mode"`
	NodeList []string `json:"nodeList,omitempty"`

	ControlPlane    ControlPlane    `json:"controlPlane,omitempty"`
	ManagedClusters ManagedClusters `json:"managedClusters,omitempty"`
}

type ControlPlane struct {
	Karmada       KarmadaDeployment `json:"karmada"`
	SuperClusterd ManagerDeployment `json:"superClusterd,omitempty"`
	Clusterd      ManagerDeployment `json:"clusterd,omitempty"`
}

type ManagerDeployment struct {
	Image    string   `json:"image"`
	Command  []string `json:"command,omitempty"`
	Replicas int32    `json:"replicas,omitempty"`
}

type KarmadaDeployment struct {
	Name           string `json:"name,omitempty"`
	EtcdVersion    string `json:"etcdVersion"`
	KarmadaVersion string `json:"karmadaVersion"`
	KubeVersion    string `json:"kubeVersion"`
	CrdUrl         string `json:"crdUrl"`
}

type ManagedClusters struct {
	Installer Installer `json:"installer,omitempty"`
	Replicas  int32     `json:"replicas,omitempty"`
}

type Installer struct {
	FuyaoInstaller  FuyaoInstaller  `json:"fuyaoInstaller"`
	AscendInstaller AscendInstaller `json:"ascendInstaller"`
	Image           string          `json:"image"`
}

type FuyaoInstaller struct {
	Command []string `json:"command,omitempty"`
}

type AscendInstaller struct {
	Command []string `json:"command,omitempty"`
}

// TenantClusterStatus defines the observed state of TenantCluster.
type TenantClusterStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Conditions         []metav1.Condition `json:"conditions,omitempty"`
	MemberClusterState MemberClusterState `json:"memberClusterState,omitempty"`
}

type MemberClusterState struct {
	ReadyClusterCount    int      `json:"readyClusterCount,omitempty"`
	ReadyClusterList     []string `json:"readyClusterList,omitempty"`
	NotReadyClusterCount int      `json:"notReadyClusterCount,omitempty"`
	NotReadyClusterList  []string `json:"notReadyClusterList,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// TenantCluster is the Schema for the tenantclusters API.
type TenantCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TenantClusterSpec   `json:"spec,omitempty"`
	Status TenantClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// TenantClusterList contains a list of TenantCluster.
type TenantClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TenantCluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TenantCluster{}, &TenantClusterList{})
}
