/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// InferNexServiceConfigSpec defines a reusable template spec for InferNexService.
type InferNexServiceConfigSpec struct {
	// Keep Config and Service in the same schema model for merge compatibility.
	InferNexServiceSpec `json:",inline"`
}

// +kubebuilder:object:root=true
// +kubebuilder:resource:shortName=insvccfg

// InferNexServiceConfig is the Schema for the infernexserviceconfigs API
type InferNexServiceConfig struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is a standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec defines the desired state of InferNexServiceConfig
	// +required
	Spec InferNexServiceConfigSpec `json:"spec"`
}

// +kubebuilder:object:root=true

// InferNexServiceConfigList contains a list of InferNexServiceConfig
type InferNexServiceConfigList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferNexServiceConfig `json:"items"`
}

func init() {
	SchemeBuilder.Register(&InferNexServiceConfig{}, &InferNexServiceConfigList{})
}
