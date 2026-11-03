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

// SandboxGroup is the schema for a group of sandboxes sharing capacity and
// resource configuration. It is the top-level control-plane resource.
type SandboxGroup struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              SandboxGroupSpec   `json:"spec,omitempty"`   // Desired state of the SandboxGroup.
	Status            SandboxGroupStatus `json:"status,omitempty"` // Observed state of the SandboxGroup.
}

// DeepCopyObject returns a deep copy of the SandboxGroup.
func (in *SandboxGroup) DeepCopyObject() runtime.Object {
	if in == nil {
		return nil
	}
	out := new(SandboxGroup)
	*out = *in
	out.ObjectMeta = *in.ObjectMeta.DeepCopy()
	out.Spec = *in.Spec.DeepCopy()
	out.Status = *in.Status.DeepCopy()
	return out
}

// SandboxGroupList is a list of SandboxGroup instances.
type SandboxGroupList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []SandboxGroup `json:"items"` // List of SandboxGroup objects.
}

// DeepCopyObject returns a deep copy of the SandboxGroupList.
func (in *SandboxGroupList) DeepCopyObject() runtime.Object {
	if in == nil {
		return nil
	}
	out := new(SandboxGroupList)
	*out = *in
	out.ListMeta = in.ListMeta
	if in.Items != nil {
		out.Items = make([]SandboxGroup, len(in.Items))
		for i := range in.Items {
			out.Items[i] = *in.Items[i].DeepCopyObject().(*SandboxGroup)
		}
	}
	return out
}

// SandboxGroupSpec defines the desired state of a SandboxGroup.
type SandboxGroupSpec struct {
	Capacity         CapacitySpec                `json:"capacity"`         // Capacity targets for agent pods and sandbox buffers.
	SandboxResources corev1.ResourceRequirements `json:"sandboxResources"` // Resource requirements for each sandbox container.
	// AgentEnv holds extra environment variables injected into the agent pod
	// containers. Users can control data-plane agent behavior through this
	// field, for example:
	//   agentEnv:
	//     EXECD_BIN_PATH: ""    # Disable execd injection (dev agent image lacks /opt/opensandbox).
	//     RUNTIME_TYPE: "container"  # Override the runtime type.
	AgentEnv map[string]string `json:"agentEnv,omitempty"`
	// ExecdImage is the image containing the execd binary.
	// When set, the allocator auto-generates an init container that copies
	// /opt/opensandbox onto a shared volume. The agent main container reads
	// execd from the shared volume and bind-mounts it into each sandbox
	// container at creation time.
	// When unset (development), the agent image is expected to provide
	// /opt/opensandbox itself, or execd can be disabled via agentEnv.
	ExecdImage string `json:"execdImage,omitempty"`
	// SchedulingPolicy controls how sandboxes are placed across agent pods.
	// When set, the controller propagates it to each SubSandboxGroup it creates.
	SchedulingPolicy *SchedulingPolicy `json:"schedulingPolicy,omitempty"`
	// AgentPodScheduling holds node-scheduling constraints (tolerations,
	// nodeSelector, affinity) for the agent pods created for this group.
	// It lets a group target a specific (possibly tainted) node pool, e.g.:
	//   agentPodScheduling:
	//     tolerations:
	//     - key: sandbox-agent
	//       operator: Equal
	//       value: "true"
	//       effect: NoSchedule
	//     nodeSelector:
	//       sandbox-agent: "true"
	// When set, the controller propagates it into each SubSandboxGroup's
	// sandboxAgentPodTemplate.spec and from there onto every agent pod.
	AgentPodScheduling *AgentPodScheduling `json:"agentPodScheduling,omitempty"`
}

// AgentPodScheduling describes node-scheduling constraints applied to the
// agent pods of a SandboxGroup: tolerate specific taints and/or select a
// target node pool. All fields are optional; unset keeps agent pods
// unconstrained (the previous behavior).
type AgentPodScheduling struct {
	Tolerations  []corev1.Toleration `json:"tolerations,omitempty"`  // Taint tolerations for agent pods.
	NodeSelector map[string]string   `json:"nodeSelector,omitempty"` // Node labels agent pods are pinned to.
	Affinity     *corev1.Affinity    `json:"affinity,omitempty"`     // Affinity rules for agent pods.
}

// DeepCopy returns a deep copy of the AgentPodScheduling.
func (in *AgentPodScheduling) DeepCopy() *AgentPodScheduling {
	if in == nil {
		return nil
	}
	out := new(AgentPodScheduling)
	*out = *in
	if in.Tolerations != nil {
		out.Tolerations = make([]corev1.Toleration, len(in.Tolerations))
		for i := range in.Tolerations {
			out.Tolerations[i] = *in.Tolerations[i].DeepCopy()
		}
	}
	if in.NodeSelector != nil {
		out.NodeSelector = make(map[string]string, len(in.NodeSelector))
		for k, v := range in.NodeSelector {
			out.NodeSelector[k] = v
		}
	}
	if in.Affinity != nil {
		out.Affinity = in.Affinity.DeepCopy()
	}
	return out
}

// DeepCopy returns a deep copy of the SandboxGroupSpec.
func (in *SandboxGroupSpec) DeepCopy() *SandboxGroupSpec {
	if in == nil {
		return nil
	}
	out := new(SandboxGroupSpec)
	*out = *in
	out.SandboxResources = *in.SandboxResources.DeepCopy()
	if in.AgentEnv != nil {
		out.AgentEnv = make(map[string]string, len(in.AgentEnv))
		for k, v := range in.AgentEnv {
			out.AgentEnv[k] = v
		}
	}
	if in.AgentPodScheduling != nil {
		out.AgentPodScheduling = in.AgentPodScheduling.DeepCopy()
	}
	if in.SchedulingPolicy != nil {
		out.SchedulingPolicy = in.SchedulingPolicy.DeepCopy()
	}
	return out
}

// SandboxGroupStatus holds the observed state of a SandboxGroup.
type SandboxGroupStatus struct {
	Phase                 string             `json:"phase,omitempty"`                 // Current lifecycle phase.
	ObservedGeneration    int64              `json:"observedGeneration,omitempty"`    // Generation observed by the controller.
	SubSandboxGroups      []string           `json:"subSandboxGroups,omitempty"`      // Names of owned SubSandboxGroups.
	SubSandboxGroupsCount int                `json:"subSandboxGroupsCount,omitempty"` // Count of owned SubSandboxGroups.
	TotalAgentPods        int                `json:"totalAgentPods,omitempty"`        // Total agent pods across all SubSandboxGroups.
	TotalCapacity         int                `json:"totalCapacity,omitempty"`         // Aggregate sandbox capacity across SubSandboxGroups.
	Conditions            []metav1.Condition `json:"conditions,omitempty"`            // Conditions describing the group state.
}

// DeepCopy returns a deep copy of the SandboxGroupStatus.
func (in *SandboxGroupStatus) DeepCopy() *SandboxGroupStatus {
	if in == nil {
		return nil
	}
	out := new(SandboxGroupStatus)
	*out = *in
	if in.SubSandboxGroups != nil {
		out.SubSandboxGroups = append([]string(nil), in.SubSandboxGroups...)
	}
	if in.Conditions != nil {
		out.Conditions = make([]metav1.Condition, len(in.Conditions))
		copy(out.Conditions, in.Conditions)
	}
	return out
}
