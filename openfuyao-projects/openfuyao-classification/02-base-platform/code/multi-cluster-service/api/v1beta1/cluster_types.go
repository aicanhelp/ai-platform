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

package v1beta1

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// ClusterSpec defines configuration parameters
type ClusterSpec struct {
	UID              string               `json:"uid,omitempty"`
	APIEndpoint      string               `json:"apiEndpoint,omitempty"`
	Kubeconfigpath   string               `json:"kubeconfigpath,omitempty"`
	KubeConfig       runtime.RawExtension `json:"kubeconfig,omitempty"`
	OutClusterConfig runtime.RawExtension `json:"outclusterconfig,omitempty"`
}

// ClusterStatus reflects operational state
type ClusterStatus struct {
	// Conditions shows the current cluster conditions.
	Condition []v1.Condition `json:"conditions,omitempty"`
	// ResourceList represents the summary of resources usage of the cluster.
	ResourceList ResourcesCondition `json:"resourceList,omitempty"`
}

// ResourcesCondition contains capacity measurements
type ResourcesCondition struct {
	// AllocatableCPU represents the CPU resources that are available for allocation.
	AllocatableCPU string `json:"allocatablecpu,omitempty"`

	// AllocatableMemory represents the memory reources that are available ofr allocation.
	AllocatableMemory string `json:"allocatablemem,omitempty"`

	// AllocatedCPU represents the CPU resources that have been allocated.
	AllocatedCPU string `json:"allocatedcpu,omitempty"`

	// AllocatedMemory represents the memory resources that have been allocated.
	AllocatedMemory string `json:"allocatedmem,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// Cluster represents the core cluster resource schema
type Cluster struct {
	v1.TypeMeta   `json:",inline"`
	v1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ClusterSpec   `json:"spec,omitempty"`
	Status ClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
// ClusterList contains collections of Cluster resources
type ClusterList struct {
	v1.TypeMeta `json:",inline"`
	v1.ListMeta `json:"metadata,omitempty"`
	Items       []Cluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Cluster{}, &ClusterList{})
}
