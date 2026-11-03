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

// +k8s:deepcopy-gen=package

package v1beta1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// BKENode GVK definition
var (
	BKENodeGVK = schema.GroupVersionKind{
		Group:   "bke.bocloud.com",
		Version: "v1beta1",
		Kind:    "BKENode",
	}
)

// NodeState represents the state of a BKENode
type NodeState string

// NodeState constants
const (
	NodeNotReady    NodeState = "NotReady"
	NodeReady       NodeState = "Ready"
	NodePending     NodeState = "Pending"
	NodeFailed      NodeState = "Failed"
	NodeDeleting    NodeState = "Deleting"
	NodeUpgrading   NodeState = "Upgrading"
	NodeProvisioned NodeState = "Provisioned"
)

// BKENodeSpec defines the desired state of BKENode
type BKENodeSpec struct {
	// Role defines the role of the node in target cluster
	// +optional
	Role []string `json:"role,omitempty"`

	// IP node IP
	// +required
	IP string `json:"ip"`

	// Port node Port used for SSH
	// +optional
	Port string `json:"port,omitempty"`

	// Username node Username used for SSH
	// +optional
	Username string `json:"username,omitempty"`

	// Password node Password used for SSH (encrypted)
	// +optional
	Password string `json:"password,omitempty"`

	// Hostname specifies the hostname of the node
	// +optional
	Hostname string `json:"hostname,omitempty"`

	// ControlPlane rewrite the cluster's ControlPlane configuration
	// +optional
	ControlPlane `json:",omitempty"`

	// Kubelet rewrite the cluster's Kubelet configuration
	// +optional
	Kubelet *Kubelet `json:"kubelet,omitempty"`

	// Labels defines the node labels
	// +optional
	Labels []Label `json:"labels,omitempty"`
}

// BKENodeStatus defines the observed state of BKENode
type BKENodeStatus struct {
	// State is the current state of the node
	// +optional
	State NodeState `json:"state,omitempty"`

	// StateCode is the bit flag representing the node state
	// +optional
	StateCode int `json:"stateCode,omitempty"`

	// Message is a human-readable message indicating details about the node state
	// +optional
	Message string `json:"message,omitempty"`

	// NeedSkip indicates whether this node should be skipped during operations
	// +optional
	NeedSkip bool `json:"needSkip,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="IP",type="string",JSONPath=".spec.ip"
// +kubebuilder:printcolumn:name="Hostname",type="string",JSONPath=".spec.hostname"
// +kubebuilder:printcolumn:name="Role",type="string",JSONPath=".spec.role"
// +kubebuilder:printcolumn:name="State",type="string",JSONPath=".status.state"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// BKENode is the Schema for the bkenodes API
type BKENode struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   BKENodeSpec   `json:"spec,omitempty"`
	Status BKENodeStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// BKENodeList contains a list of BKENode
type BKENodeList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []BKENode `json:"items"`
}

// SetState sets the node state
func (n *BKENode) SetState(state NodeState) {
	n.Status.State = state
}

// SetStateWithMessage sets the node state along with a message
func (n *BKENode) SetStateWithMessage(state NodeState, message string) {
	n.Status.State = state
	n.Status.Message = message
}

// ToNode converts BKENode to the legacy Node type
func (n *BKENode) ToNode() Node {
	return Node{
		Role:         n.Spec.Role,
		IP:           n.Spec.IP,
		Port:         n.Spec.Port,
		Username:     n.Spec.Username,
		Password:     n.Spec.Password,
		Hostname:     n.Spec.Hostname,
		ControlPlane: n.Spec.ControlPlane,
		Kubelet:      n.Spec.Kubelet,
		Labels:       n.Spec.Labels,
	}
}

// FromNode creates a BKENodeSpec from a legacy Node type
func FromNode(node Node) BKENodeSpec {
	return BKENodeSpec{
		Role:         node.Role,
		IP:           node.IP,
		Port:         node.Port,
		Username:     node.Username,
		Password:     node.Password,
		Hostname:     node.Hostname,
		ControlPlane: node.ControlPlane,
		Kubelet:      node.Kubelet,
		Labels:       node.Labels,
	}
}
