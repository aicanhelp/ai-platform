/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster,shortName=up
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Paths",type=integer,JSONPath=`.status.pathCount`

// UpgradePath defines allowed version upgrade routes loaded from an OCI artifact.
type UpgradePath struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   UpgradePathSpec   `json:"spec,omitempty"`
	Status UpgradePathStatus `json:"status,omitempty"`
}

// UpgradePathSpec defines the desired upgrade path catalog.
type UpgradePathSpec struct {
	Paths    []UpgradePathRule `json:"paths,omitempty"`
	Versions []VersionEntry    `json:"versions,omitempty"`
}

// VersionEntry defines version info.
type VersionEntry struct {
	Version     string `json:"version,omitempty"`
	Installable bool   `json:"installable,omitempty"`
	Deprecated  bool   `json:"deprecated,omitempty"`
	Notes       string `json:"notes,omitempty"`
}

// UpgradePathRule is one directed upgrade edge between versions.
type UpgradePathRule struct {
	From       string      `json:"from,omitempty"`
	To         string      `json:"to,omitempty"`
	Blocked    bool        `json:"blocked,omitempty"`
	Deprecated bool        `json:"deprecated,omitempty"`
	Notes      string      `json:"notes,omitempty"`
	PreCheck   []CheckStep `json:"preCheck,omitempty"`
	PostCheck  []CheckStep `json:"postCheck,omitempty"`
}

// CheckStep describes a upgrade validation step.
type CheckStep struct {
	Name     string `json:"name,omitempty"`
	Required bool   `json:"required,omitempty"`
}

// UpgradePathPhase is the validation/lifecycle phase of an UpgradePath.
// +kubebuilder:validation:Enum=Active;Blocked;Invalid
type UpgradePathPhase string

const (
	UpgradePathPhaseActive  UpgradePathPhase = "Active"
	UpgradePathPhaseBlocked UpgradePathPhase = "Blocked"
	UpgradePathPhaseInvalid UpgradePathPhase = "Invalid"
)

// UpgradePathStatus defines the observed upgrade path catalog state.
type UpgradePathStatus struct {
	Phase         UpgradePathPhase   `json:"phase,omitempty"`
	LastDigest    string             `json:"lastDigest,omitempty"`
	PathCount     int                `json:"pathCount"`
	LastCheckedAt *metav1.Time       `json:"lastCheckedAt,omitempty"`
	Conditions    []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true

// UpgradePathList contains a list of UpgradePath.
type UpgradePathList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []UpgradePath `json:"items"`
}

func init() {
	SchemeBuilder.Register(&UpgradePath{}, &UpgradePathList{})
}
