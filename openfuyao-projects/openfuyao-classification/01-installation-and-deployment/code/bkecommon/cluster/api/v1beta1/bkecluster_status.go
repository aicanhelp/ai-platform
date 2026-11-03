/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// +k8s:deepcopy-gen=package

package v1beta1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

const (
	ConditionTrue    ConditionStatus = "True"
	ConditionFalse   ConditionStatus = "False"
	ConditionUnknown ConditionStatus = "Unknown"
)

type BKEClusterPhase string
type BKEClusterPhases []BKEClusterPhase

func (in BKEClusterPhase) String() string { return string(in) }

func (in BKEClusterPhase) In(phases BKEClusterPhases) bool {
	for _, phase := range phases {
		if phase == in {
			return true
		}
	}
	return false
}

func (in BKEClusterPhase) NotIn(phases BKEClusterPhases) bool {
	return !in.In(phases)
}

func (in *BKEClusterPhases) Add(phases ...BKEClusterPhase) {
	for _, phase := range phases {
		*in = append(*in, phase)
	}

}

func (in *BKEClusterPhases) Remove(phases ...BKEClusterPhase) {
	// 创建一个新的切片，用于存储结果
	result := make([]BKEClusterPhase, 0, len(*in))

	for _, p := range *in {
		found := false

		// 检查 p 是否在 phases 中
		for _, phase := range phases {
			if p == phase {
				found = true
				break
			}
		}

		// 如果 p 不在 phases 中，将其添加到结果切片中
		if !found {
			result = append(result, p)
		}
	}

	// 将结果切片赋值回原始切片
	*in = result
}

type ClusterStatus string

type ClusterHealthState string

type BKEClusterPhaseStatus string

type ClusterConditionType string

type ConditionStatus string

// BKEClusterStatus defines the observed state of BKECluster
type BKEClusterStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	// +optional
	Ready bool `json:"ready"`

	// +optional
	OpenFuyaoVersion string `json:"openFuyaoVersion,omitempty"`

	// +optional
	KubernetesVersion string `json:"kubernetesVersion,omitempty"`

	// +optional
	EtcdVersion string `json:"etcdVersion,omitempty"`

	// +optional
	ContainerdVersion string `json:"containerdVersion,omitempty"`

	// +optional
	AgentStatus BKEAgentStatus `json:"agentStatus"`

	// Phase is the current phase of the cluster.
	// +optional
	Phase BKEClusterPhase `json:"phase,omitempty"`

	// ClusterStatus is the current operate status of the cluster.
	// +optional
	ClusterStatus ClusterStatus `json:"clusterStatus,omitempty"`

	// ClusterHealthState
	// +optional
	ClusterHealthState ClusterHealthState `json:"clusterHealthState,omitempty"`

	// AddonStatus is the current status of the addons.
	AddonStatus []Product `json:"addonStatus,omitempty"`

	// +kubebuilder:object:generate:=true
	// +optional
	PhaseStatus PhaseStatus `json:"phaseStatus,omitempty"`

	// +kubebuilder:object:generate:=true
	// +optional
	Conditions ClusterConditions `json:"conditions,omitempty"`
}

// +kubebuilder:object:generate=true

type ClusterConditions []ClusterCondition

type ClusterCondition struct {
	Type ClusterConditionType `json:"type"`

	// AddonName is the name of the current reconcile addon
	// +optional
	AddonName string `json:"addonName,omitempty"`

	// Status of the condition, one of True, False, Unknown.
	Status ConditionStatus `json:"status"`

	// Last time the condition transitioned from one status to another.
	// This should be when the underlying condition changed. If that is not known, then using the time when
	// the API field changed is acceptable.
	LastTransitionTime *metav1.Time `json:"lastTransitionTime,omitempty"`

	// The reason for the condition's last transition in CamelCase.
	// The specific API may choose whether or not this field is considered a guaranteed API.
	// This field may not be empty.
	// +optional
	Reason string `json:"reason,omitempty"`

	// A human readable message indicating details about the transition.
	// This field may be empty.
	// +optional
	Message string `json:"message,omitempty"`
}

type BKEAgentStatus struct {
	// +optional
	// +kubebuilder:default:=0
	Replies int32 `json:"replies,omitempty"`
	// +optional
	// +kubebuilder:default:=0
	UnavailableReplies int32 `json:"unavailableReplies,omitempty"`
	// +optional
	// +kubebuilder:default:="0/0"
	Status string `json:"status,omitempty"`
}

func (agentStatus *BKEAgentStatus) Reset() {
	agentStatus.Replies = 0
	agentStatus.UnavailableReplies = 0
	agentStatus.Status = "0/0"
}

func (agentStatus *BKEAgentStatus) Ready() bool {
	return agentStatus.UnavailableReplies == 0
}

func (agentStatus *BKEAgentStatus) Equal(other *BKEAgentStatus) bool {
	return agentStatus.Replies == other.Replies &&
		agentStatus.UnavailableReplies == other.UnavailableReplies &&
		agentStatus.Status == other.Status
}

type PhaseStatus []PhaseState

type PhaseState struct {
	// Name is the name of the phase name
	// +required
	Name BKEClusterPhase `json:"name,omitempty"`
	// StartTime is the start time of the phase
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`
	// EndTime is the end time of the phase
	// +optional
	EndTime *metav1.Time `json:"endTime,omitempty"`
	// Status is the status of the phase
	// +required
	Status BKEClusterPhaseStatus `json:"status,omitempty"`
	// Message is the message of the phase
	// +optional
	Message string `json:"message,omitempty"`
}
