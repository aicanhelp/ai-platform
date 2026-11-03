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
// +kubebuilder:resource:shortName=ri
// +kubebuilder:printcolumn:name="Version",type=string,JSONPath=`.spec.version`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`

// ReleaseImage describes an OCI release image and its install/upgrade component manifests.
type ReleaseImage struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ReleaseImageSpec   `json:"spec,omitempty"`
	Status ReleaseImageStatus `json:"status,omitempty"`
}

// ReleaseImageSpec defines the desired release image state.
type ReleaseImageSpec struct {
	Version            string                   `json:"version,omitempty"`
	Digest             string                   `json:"digest,omitempty"`
	VerifySignature    bool                     `json:"verifySignature,omitempty"`
	SignatureKey       string                   `json:"signatureKey,omitempty"`
	AllowCacheFallback bool                     `json:"allowCacheFallback,omitempty"`
	Install            *ReleaseImageInstallSpec `json:"install,omitempty"`
	Upgrade            *ReleaseImageUpgradeSpec `json:"upgrade,omitempty"`
}

// ReleaseImageInstallSpec lists components to install from the release image.
type ReleaseImageInstallSpec struct {
	Components []ReleaseImageInstallComponent `json:"components,omitempty"`
}

// ReleaseImageInstallComponent is one installable component.
type ReleaseImageInstallComponent struct {
	Name    string `json:"name,omitempty"`
	Version string `json:"version,omitempty"`
}

// ReleaseImageUpgradeSpec lists components and upgrade handlers for a release.
type ReleaseImageUpgradeSpec struct {
	Components []ReleaseImageUpgradeComponent `json:"components,omitempty"`
}

// ReleaseImageUpgradeComponent is one upgradable component, optionally with an inline handler.
type ReleaseImageUpgradeComponent struct {
	Name    string                     `json:"name,omitempty"`
	Version string                     `json:"version,omitempty"`
	Inline  *ReleaseImageUpgradeInline `json:"inline,omitempty"`
}

// ReleaseImageUpgradeInline references an inline upgrade handler implementation.
type ReleaseImageUpgradeInline struct {
	Handler string `json:"handler,omitempty"`
	Version string `json:"version,omitempty"`
}

// ReleaseImagePhase is the validation/lifecycle phase of a ReleaseImage.
// +kubebuilder:validation:Enum=Valid;Invalid;ManifestMissing;CompatibilityFailed
type ReleaseImagePhase string

const (
	ReleaseImagePhaseValid               ReleaseImagePhase = "Valid"
	ReleaseImagePhaseInvalid             ReleaseImagePhase = "Invalid"
	ReleaseImagePhaseManifestMissing     ReleaseImagePhase = "ManifestMissing"
	ReleaseImagePhaseCompatibilityFailed ReleaseImagePhase = "CompatibilityFailed"
)

// ReleaseImageStatus defines the observed release image state.
type ReleaseImageStatus struct {
	Phase               ReleaseImagePhase `json:"phase,omitempty"`
	ComponentCount      int               `json:"componentCount,omitempty"`
	Components          []ComponentStatus `json:"components,omitempty"`
	ValidatedAt         *metav1.Time      `json:"validatedAt,omitempty"`
	Digest              string            `json:"digest,omitempty"`
	Source              string            `json:"source,omitempty"`
	CacheFallback       bool              `json:"cacheFallback,omitempty"`
	Message             string            `json:"message,omitempty"`
	CompatibilityReport string            `json:"compatibilityReport,omitempty"`
}

// ComponentStatus records component metadata parsed from the release image OCI manifest.
type ComponentStatus struct {
	Name    string        `json:"name,omitempty"`
	Version string        `json:"version,omitempty"`
	Type    ComponentType `json:"type,omitempty"`
}

// +kubebuilder:object:root=true

// ReleaseImageList contains a list of ReleaseImage.
type ReleaseImageList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ReleaseImage `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ReleaseImage{}, &ReleaseImageList{})
}
