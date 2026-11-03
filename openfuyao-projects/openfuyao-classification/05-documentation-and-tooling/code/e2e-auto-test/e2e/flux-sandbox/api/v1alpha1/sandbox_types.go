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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// Sandbox is the schema for a single sandbox instance created from a SandboxGroup.
type Sandbox struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              SandboxSpec   `json:"spec,omitempty"`   // Desired state of the sandbox.
	Status            SandboxStatus `json:"status,omitempty"` // Observed state of the sandbox.
}

// DeepCopyObject returns a deep copy of the Sandbox.
func (in *Sandbox) DeepCopyObject() runtime.Object {
	if in == nil {
		return nil
	}
	out := new(Sandbox)
	*out = *in
	out.ObjectMeta = *in.ObjectMeta.DeepCopy()
	out.Spec = *in.Spec.DeepCopy()
	out.Status = *in.Status.DeepCopy()
	return out
}

// SandboxList is a list of Sandbox instances.
type SandboxList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Sandbox `json:"items"` // List of Sandbox objects.
}

// DeepCopyObject returns a deep copy of the SandboxList.
func (in *SandboxList) DeepCopyObject() runtime.Object {
	if in == nil {
		return nil
	}
	out := new(SandboxList)
	*out = *in
	out.ListMeta = in.ListMeta
	if in.Items != nil {
		out.Items = make([]Sandbox, len(in.Items))
		for i := range in.Items {
			out.Items[i] = *in.Items[i].DeepCopyObject().(*Sandbox)
		}
	}
	return out
}

// SandboxSpec defines the desired state of a Sandbox.
type SandboxSpec struct {
	SandboxGroup    string                      `json:"sandboxGroup,omitempty"`    // Parent SandboxGroup name.
	SubSandboxGroup string                      `json:"subSandboxGroup,omitempty"` // Parent SubSandboxGroup name.
	Image           string                      `json:"image"`                     // Container image to run.
	Command         []string                    `json:"command,omitempty"`         // Entrypoint command overriding the image default.
	Args            []string                    `json:"args,omitempty"`            // Arguments appended to the entrypoint command.
	Resources       corev1.ResourceRequirements `json:"resources,omitempty"`       // Resource requirements for the sandbox container.
	ExpireTime      *metav1.Time                `json:"expireTime,omitempty"`      // Optional deadline after which the sandbox is reaped.
	FailurePolicy   string                      `json:"failurePolicy,omitempty"`   // Policy applied when the sandbox fails.
}

// DeepCopy returns a deep copy of the SandboxSpec.
func (in *SandboxSpec) DeepCopy() *SandboxSpec {
	if in == nil {
		return nil
	}
	out := new(SandboxSpec)
	*out = *in
	if in.Command != nil {
		out.Command = append([]string(nil), in.Command...)
	}
	if in.Args != nil {
		out.Args = append([]string(nil), in.Args...)
	}
	out.Resources = *in.Resources.DeepCopy()
	if in.ExpireTime != nil {
		out.ExpireTime = in.ExpireTime.DeepCopy()
	}
	return out
}

// SandboxStatus holds the observed state of a Sandbox.
type SandboxStatus struct {
	Phase                   string `json:"phase,omitempty"`                   // Current lifecycle phase.
	SandboxID               string `json:"sandboxID,omitempty"`               // Unique identifier assigned by the scheduler.
	Scheduler               string `json:"scheduler,omitempty"`               // Name of the scheduler handling this sandbox.
	AssignedSandboxAgentPod string `json:"assignedSandboxAgentPod,omitempty"` // Name of the agent pod running this sandbox.
	SchedulerEndpoint       string `json:"schedulerEndpoint,omitempty"`       // Address of the scheduler serving this sandbox.
	SandboxAgentPodEndpoint string `json:"sandboxAgentPodEndpoint,omitempty"` // Address of the agent pod serving this sandbox.
	// Endpoints is the comma-separated list of sandbox access endpoints
	// (address:port). In E2B mode these come from the runtime (CNI IP:49983,
	// the envd daemon inside the microVM); in containerd mode the agent pod
	// address with the execd port.
	Endpoints          string             `json:"endpoints,omitempty"`          // Sandbox access endpoints used by the SDK.
	Reason             string             `json:"reason,omitempty"`             // Machine-readable reason for the current phase.
	Message            string             `json:"message,omitempty"`            // Human-readable message about the current phase.
	LastTransitionTime *metav1.Time       `json:"lastTransitionTime,omitempty"` // Time of the last phase transition.
	Conditions         []metav1.Condition `json:"conditions,omitempty"`         // Conditions describing the sandbox state.
}

// DeepCopy returns a deep copy of the SandboxStatus.
func (in *SandboxStatus) DeepCopy() *SandboxStatus {
	if in == nil {
		return nil
	}
	out := new(SandboxStatus)
	*out = *in
	if in.LastTransitionTime != nil {
		out.LastTransitionTime = in.LastTransitionTime.DeepCopy()
	}
	if in.Conditions != nil {
		out.Conditions = make([]metav1.Condition, len(in.Conditions))
		copy(out.Conditions, in.Conditions)
	}
	return out
}
