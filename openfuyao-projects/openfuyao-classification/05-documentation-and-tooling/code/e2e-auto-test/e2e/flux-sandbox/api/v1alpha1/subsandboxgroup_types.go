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

// SubSandboxGroup is the schema for a slice of a SandboxGroup. It owns a set of
// agent pods and the sandbox capacity allocated to them.
type SubSandboxGroup struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              SubSandboxGroupSpec   `json:"spec,omitempty"`   // Desired state of the SubSandboxGroup.
	Status            SubSandboxGroupStatus `json:"status,omitempty"` // Observed state of the SubSandboxGroup.
}

// DeepCopyObject returns a deep copy of the SubSandboxGroup.
func (in *SubSandboxGroup) DeepCopyObject() runtime.Object {
	if in == nil {
		return nil
	}
	out := new(SubSandboxGroup)
	*out = *in
	out.ObjectMeta = *in.ObjectMeta.DeepCopy()
	out.Spec = *in.Spec.DeepCopy()
	out.Status = *in.Status.DeepCopy()
	return out
}

// SubSandboxGroupList is a list of SubSandboxGroup instances.
type SubSandboxGroupList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []SubSandboxGroup `json:"items"` // List of SubSandboxGroup objects.
}

// DeepCopyObject returns a deep copy of the SubSandboxGroupList.
func (in *SubSandboxGroupList) DeepCopyObject() runtime.Object {
	if in == nil {
		return nil
	}
	out := new(SubSandboxGroupList)
	*out = *in
	out.ListMeta = in.ListMeta
	if in.Items != nil {
		out.Items = make([]SubSandboxGroup, len(in.Items))
		for i := range in.Items {
			out.Items[i] = *in.Items[i].DeepCopyObject().(*SubSandboxGroup)
		}
	}
	return out
}

// SubSandboxGroupSpec defines the desired state of a SubSandboxGroup.
type SubSandboxGroupSpec struct {
	OwnerSandboxGroup             string                      `json:"ownerSandboxGroup"`                       // Name of the owning SandboxGroup.
	Scheduler                     string                      `json:"scheduler"`                               // Name of the scheduler responsible for this group.
	Capacity                      SubSandboxGroupCapacitySpec `json:"capacity"`                                // Capacity targets for agent pods and sandbox buffers.
	PerSlotResources              corev1.ResourceRequirements `json:"perSlotResources"`                        // Resource requirements for each sandbox slot.
	SandboxAgentPodResources      corev1.ResourceRequirements `json:"sandboxAgentPodResources,omitempty"`      // Resource requirements for each agent pod.
	SandboxAgentPodTotalResources corev1.ResourceRequirements `json:"sandboxAgentPodTotalResources,omitempty"` // Aggregate resource requirements for all agent pods.
	SandboxAgentPodTemplate       *SandboxAgentPodTemplate    `json:"sandboxAgentPodTemplate,omitempty"`       // Template for agent pods created by this group.
	SchedulingPolicy              *SchedulingPolicy           `json:"schedulingPolicy,omitempty"`              // Policy controlling how sandboxes are placed across agent pods.
}

// SubSandboxGroupCapacitySpec is the capacity declaration for a SubSandboxGroup.
// JSON tags align with the CRD fields (sandboxBufferMin/sandboxBufferMax) to stay
// consistent with the SandboxGroup CapacitySpec. The previous bufferMin/bufferMax
// tags did not match the CRD required fields, which caused SubSandboxGroup CRs to
// be rejected by the schema.
type SubSandboxGroupCapacitySpec struct {
	AgentPodMin        int `json:"agentPodMin"`                // Minimum number of agent pods to maintain.
	AgentPodMax        int `json:"agentPodMax"`                // Maximum number of agent pods allowed.
	SandboxBufferMin   int `json:"sandboxBufferMin,omitempty"` // Minimum idle sandboxes kept as a warm buffer.
	SandboxBufferMax   int `json:"sandboxBufferMax,omitempty"` // Maximum idle sandboxes kept as a warm buffer.
	MaxSandboxesPerPod int `json:"maxSandboxesPerPod"`         // Maximum sandboxes co-located on a single agent pod.
}

// SandboxAgentPodTemplate describes the metadata and pod spec template used to
// create agent pods.
type SandboxAgentPodTemplate struct {
	Metadata *PodTemplateMetadata `json:"metadata,omitempty"` // Optional metadata applied to agent pods.
	Spec     *PodTemplateSpec     `json:"spec,omitempty"`     // Optional pod spec applied to agent pods.
}

// PodTemplateMetadata holds labels and annotations applied to agent pods.
type PodTemplateMetadata struct {
	Labels      map[string]string `json:"labels,omitempty"`      // Labels applied to agent pods.
	Annotations map[string]string `json:"annotations,omitempty"` // Annotations applied to agent pods.
}

// PodTemplateSpec is the pod spec template for agent pods.
type PodTemplateSpec struct {
	HostPID        bool                 `json:"hostPID,omitempty"`        // Whether to share the host PID namespace.
	HostNetwork    bool                 `json:"hostNetwork,omitempty"`    // Whether to use the host network namespace.
	HostIPC        bool                 `json:"hostIPC,omitempty"`        // Whether to share the host IPC namespace.
	RestartPolicy  corev1.RestartPolicy `json:"restartPolicy,omitempty"`  // Restart policy for agent pods.
	InitContainers []corev1.Container   `json:"initContainers,omitempty"` // Init containers run before the main agent container.
	Containers     []corev1.Container   `json:"containers,omitempty"`     // Main containers of the agent pod.
	Volumes        []corev1.Volume      `json:"volumes,omitempty"`        // Volumes available to the agent pod.
	// Scheduling constraints propagated to agent pods, so a SandboxGroup can
	// target tainted or labeled node pools (tolerate a specific taint plus a
	// nodeSelector pin). Empty values keep the previous unconstrained behavior.
	Tolerations  []corev1.Toleration `json:"tolerations,omitempty"`  // Tolerations applied to agent pods.
	NodeSelector map[string]string   `json:"nodeSelector,omitempty"` // Node label selector for agent pods.
	Affinity     *corev1.Affinity    `json:"affinity,omitempty"`     // Affinity rules for agent pods.
}

// SchedulingPolicy controls how sandboxes are placed across agent pods.
type SchedulingPolicy struct {
	Strategy         string             `json:"strategy,omitempty"`         // Placement strategy: LegacyAffinity or Balanced.
	Weights          *SchedulingWeights `json:"weights,omitempty"`          // Optional weights tuning the placement strategy.
	TopologyAffinity *TopologyAffinity  `json:"topologyAffinity,omitempty"` // Optional topology-aware spreading configuration.
}

// TopologyAffinity configures topology-aware sandbox scheduling. When enabled,
// the scheduler scores Agent Pods by how many sandboxes are already allocated
// in each topology domain (derived from a k8s node label), spreading sandboxes
// evenly across domains to improve fault isolation and resource utilization.
type TopologyAffinity struct {
	Enabled     bool   `json:"enabled"`               // Whether topology-aware scheduling is active.
	Strategy    string `json:"strategy,omitempty"`    // Topology scheduling strategy; currently only "topologySpread" (even distribution).
	TopologyKey string `json:"topologyKey,omitempty"` // k8s node label key whose value identifies the topology domain (e.g. "topology.kubernetes.io/zone").
}

// SchedulingWeights tunes the relative importance of placement factors.
type SchedulingWeights struct {
	Load          int `json:"load,omitempty"`          // Weight for agent pod load.
	Image         int `json:"image,omitempty"`         // Weight for image locality.
	Fragmentation int `json:"fragmentation,omitempty"` // Weight for fragmentation avoidance.
	Topology      int `json:"topology,omitempty"`      // Weight for topology affinity.
}

// DeepCopy returns a deep copy of the SubSandboxGroupSpec.
func (in *SubSandboxGroupSpec) DeepCopy() *SubSandboxGroupSpec {
	if in == nil {
		return nil
	}
	out := new(SubSandboxGroupSpec)
	out.OwnerSandboxGroup = in.OwnerSandboxGroup
	out.Scheduler = in.Scheduler
	out.Capacity = in.Capacity
	out.PerSlotResources = *in.PerSlotResources.DeepCopy()
	out.SandboxAgentPodResources = *in.SandboxAgentPodResources.DeepCopy()
	out.SandboxAgentPodTotalResources = *in.SandboxAgentPodTotalResources.DeepCopy()
	if in.SandboxAgentPodTemplate != nil {
		out.SandboxAgentPodTemplate = in.SandboxAgentPodTemplate.DeepCopy()
	}
	if in.SchedulingPolicy != nil {
		out.SchedulingPolicy = in.SchedulingPolicy.DeepCopy()
	}
	return out
}

// DeepCopy returns a deep copy of the SandboxAgentPodTemplate.
func (in *SandboxAgentPodTemplate) DeepCopy() *SandboxAgentPodTemplate {
	if in == nil {
		return nil
	}
	out := new(SandboxAgentPodTemplate)
	if in.Metadata != nil {
		out.Metadata = new(PodTemplateMetadata)
		if in.Metadata.Labels != nil {
			out.Metadata.Labels = make(map[string]string, len(in.Metadata.Labels))
			for k, v := range in.Metadata.Labels {
				out.Metadata.Labels[k] = v
			}
		}
		if in.Metadata.Annotations != nil {
			out.Metadata.Annotations = make(map[string]string, len(in.Metadata.Annotations))
			for k, v := range in.Metadata.Annotations {
				out.Metadata.Annotations[k] = v
			}
		}
	}
	if in.Spec != nil {
		out.Spec = new(PodTemplateSpec)
		if in.Spec.Containers != nil {
			out.Spec.Containers = make([]corev1.Container, len(in.Spec.Containers))
			for i := range in.Spec.Containers {
				out.Spec.Containers[i] = *in.Spec.Containers[i].DeepCopy()
			}
		}
		if in.Spec.Tolerations != nil {
			out.Spec.Tolerations = make([]corev1.Toleration, len(in.Spec.Tolerations))
			for i := range in.Spec.Tolerations {
				out.Spec.Tolerations[i] = *in.Spec.Tolerations[i].DeepCopy()
			}
		}
		if in.Spec.NodeSelector != nil {
			out.Spec.NodeSelector = make(map[string]string, len(in.Spec.NodeSelector))
			for k, v := range in.Spec.NodeSelector {
				out.Spec.NodeSelector[k] = v
			}
		}
		if in.Spec.Affinity != nil {
			out.Spec.Affinity = in.Spec.Affinity.DeepCopy()
		}
	}
	return out
}

// DeepCopy returns a deep copy of the SchedulingPolicy.
func (in *SchedulingPolicy) DeepCopy() *SchedulingPolicy {
	if in == nil {
		return nil
	}
	out := new(SchedulingPolicy)
	out.Strategy = in.Strategy
	if in.Weights != nil {
		out.Weights = new(SchedulingWeights)
		*out.Weights = *in.Weights
	}
	if in.TopologyAffinity != nil {
		out.TopologyAffinity = new(TopologyAffinity)
		*out.TopologyAffinity = *in.TopologyAffinity
	}
	return out
}

// SubSandboxGroupStatus holds the observed state of a SubSandboxGroup.
type SubSandboxGroupStatus struct {
	Phase              string             `json:"phase,omitempty"`              // Current lifecycle phase.
	ObservedGeneration int64              `json:"observedGeneration,omitempty"` // Generation observed by the controller.
	ReadyAgentPods     int                `json:"readyAgentPods,omitempty"`     // Number of agent pods that are ready.
	DesiredAgentPods   int                `json:"desiredAgentPods,omitempty"`   // Number of agent pods the controller aims to maintain.
	Capacity           int                `json:"capacity,omitempty"`           // Total sandbox capacity across ready agent pods.
	Allocated          int                `json:"allocated,omitempty"`          // Sandboxes currently allocated to users.
	Idle               int                `json:"idle,omitempty"`               // Sandboxes available for immediate allocation.
	Conditions         []metav1.Condition `json:"conditions,omitempty"`         // Conditions describing the group state.
}

// DeepCopy returns a deep copy of the SubSandboxGroupStatus.
func (in *SubSandboxGroupStatus) DeepCopy() *SubSandboxGroupStatus {
	if in == nil {
		return nil
	}
	out := new(SubSandboxGroupStatus)
	*out = *in
	if in.Conditions != nil {
		out.Conditions = make([]metav1.Condition, len(in.Conditions))
		copy(out.Conditions, in.Conditions)
	}
	return out
}
