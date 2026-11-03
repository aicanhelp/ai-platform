// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package v1beta1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// WebterminalTemplateSpec defines the desired state of WebterminalTemplate
type WebterminalTemplateSpec struct {
	DefaultImage   string      `json:"defaultImage,omitempty"`
	SessionTimeout int         `json:"sessionTimeout,omitempty"`
	ExistsTime     metav1.Time `json:"existstime,omitempty"`
	RenewTime      metav1.Time `json:"renewTime,omitempty"`
	PodTemplate    PodTemplate `json:"podTemplate"`
}

type PodTemplate struct {
	ObjectMeta PodTemplateObjectMeta `json:"objectmeta"`
	Spec       PodTemplateSpec       `json:"spec"`
}

type PodTemplateObjectMeta struct {
	Name      string            `json:"name"`
	Namespace string            `json:"namespace"`
	Labels    map[string]string `json:"labels,omitempty"`
}

type PodTemplateSpec struct {
	InitContainers []corev1.Container   `json:"initContainers,omitempty"`
	Containers     []corev1.Container   `json:"containers"`
	Volumes        []corev1.Volume      `json:"volumes"`
	RestartPolicy  corev1.RestartPolicy `json:"restartPolicy"`
}

// WebterminalTemplateStatus defines the observed state of WebterminalTemplate
type WebterminalTemplateStatus struct {
	Phase      WebTerminalTemplatePhase       `json:"phase,omitempty"`
	Conditions []WebTerminalTemplateCondition `json:"conditions,omitempty"`
}

type WebTerminalTemplatePhase string

// valid phase
const (
	WebTerminalTemplateStarting WebTerminalTemplatePhase = "Starting"
	WebTerminalTemplateRunning  WebTerminalTemplatePhase = "Runnings"
	WebTerminalTemplateStopped  WebTerminalTemplatePhase = "Stopped"
	WebTerminalTemplateStopping WebTerminalTemplatePhase = "Stopping"
	WebTerminalTemplateError    WebTerminalTemplatePhase = "Error"
)

type WebTerminalTemplateCondition struct {
	Status             corev1.ConditionStatus `json:"status"`
	LastTransitionTime metav1.Time            `json:"lastTransitionTime,omitempty"`
	Reason             string                 `json:"reason,omitempty"`
	Message            string                 `json:"message,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status

// WebterminalTemplate is the Schema for the webterminaltemplates API
type WebterminalTemplate struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   WebterminalTemplateSpec   `json:"spec,omitempty"`
	Status WebterminalTemplateStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// WebterminalTemplateList contains a list of WebterminalTemplate
type WebterminalTemplateList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []WebterminalTemplate `json:"items"`
}

func init() {
	SchemeBuilder.Register(&WebterminalTemplate{}, &WebterminalTemplateList{})
}
