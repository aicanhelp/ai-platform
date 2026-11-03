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
// +kubebuilder:resource:shortName=cv
// +kubebuilder:printcolumn:name="Desired",type=string,JSONPath=`.spec.desiredVersion`
// +kubebuilder:printcolumn:name="Current",type=string,JSONPath=`.status.currentVersion`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`

// ClusterVersion tracks desired and current openFuyao cluster version.
// Association with BKECluster is via OwnerReference and matching metadata.name.
type ClusterVersion struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ClusterVersionSpec   `json:"spec,omitempty"`
	Status ClusterVersionStatus `json:"status,omitempty"`
}

// ClusterVersionSpec defines the desired cluster version state.
type ClusterVersionSpec struct {
	// DesiredVersion is the target openFuyao version for the cluster.
	DesiredVersion string `json:"desiredVersion,omitempty"`
}

// ClusterVersionPhase is the lifecycle phase of a ClusterVersion.
// +kubebuilder:validation:Enum=Pending;Installing;Installed;Ready;PreChecking;Upgrading;Upgraded;Blocked;PreCheckFailed;Failed
type ClusterVersionPhase string

const (
	ClusterVersionPhasePending        ClusterVersionPhase = "Pending"
	ClusterVersionPhaseInstalling     ClusterVersionPhase = "Installing"
	ClusterVersionPhaseInstalled      ClusterVersionPhase = "Installed"
	ClusterVersionPhaseReady          ClusterVersionPhase = "Ready"
	ClusterVersionPhasePreChecking    ClusterVersionPhase = "PreChecking"
	ClusterVersionPhaseUpgrading      ClusterVersionPhase = "Upgrading"
	ClusterVersionPhaseUpgraded       ClusterVersionPhase = "Upgraded"
	ClusterVersionPhaseBlocked        ClusterVersionPhase = "Blocked"
	ClusterVersionPhasePreCheckFailed ClusterVersionPhase = "PreCheckFailed"
	ClusterVersionPhaseFailed         ClusterVersionPhase = "Failed"
)

// ClusterVersionStatus defines the observed cluster version state.
// Written by BKECluster Reconciler per design; not by ClusterVersion Reconciler.
type ClusterVersionStatus struct {
	CurrentVersion string                    `json:"currentVersion,omitempty"`
	Phase          ClusterVersionPhase       `json:"phase,omitempty"`
	UpgradeHistory []ClusterUpgradeRecord    `json:"upgradeHistory,omitempty"`
	Conditions     []ClusterVersionCondition `json:"conditions,omitempty"`
}

// ClusterUpgradeRecord records one upgrade attempt.
type ClusterUpgradeRecord struct {
	From        string                     `json:"from,omitempty"`
	To          string                     `json:"to,omitempty"`
	StartedAt   *metav1.Time               `json:"startedAt,omitempty"`
	CompletedAt *metav1.Time               `json:"completedAt,omitempty"`
	Status      ClusterUpgradeRecordStatus `json:"status,omitempty"`
}

// ClusterUpgradeRecordStatus is the result of an upgrade record.
// +kubebuilder:validation:Enum=Succeeded;Failed;RolledBack
type ClusterUpgradeRecordStatus string

const (
	ClusterUpgradeRecordStatusSucceeded  ClusterUpgradeRecordStatus = "Succeeded"
	ClusterUpgradeRecordStatusFailed     ClusterUpgradeRecordStatus = "Failed"
	ClusterUpgradeRecordStatusRolledBack ClusterUpgradeRecordStatus = "RolledBack"
)

// ClusterVersionCondition reports fine-grained readiness.
type ClusterVersionCondition struct {
	Type               string                 `json:"type,omitempty"`
	Status             metav1.ConditionStatus `json:"status,omitempty"`
	Reason             string                 `json:"reason,omitempty"`
	Message            string                 `json:"message,omitempty"`
	LastTransitionTime metav1.Time            `json:"lastTransitionTime,omitempty"`
}

// +kubebuilder:object:root=true

// ClusterVersionList contains a list of ClusterVersion.
type ClusterVersionList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ClusterVersion `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ClusterVersion{}, &ClusterVersionList{})
}
