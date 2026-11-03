/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package v1beta1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

const (
	BKEMachineFinalizer = "bkemachine.infrastructure.cluster.x-k8s.io"
)

// MachineAddressType describes a valid MachineAddress type.
type MachineAddressType string

// Define the MachineAddressType constants.
const (
	MachineHostName    MachineAddressType = "Hostname"
	MachineExternalIP  MachineAddressType = "ExternalIP"
	MachineInternalIP  MachineAddressType = "InternalIP"
	MachineExternalDNS MachineAddressType = "ExternalDNS"
	MachineInternalDNS MachineAddressType = "InternalDNS"
)

// BKEMachineSpec defines the desired state of BKEMachine
type BKEMachineSpec struct {
	// +optional
	// 标识唯一的主机 cluster-api需要的参数，可以用hostname或者ip填充
	ProviderID *string `json:"providerID,omitempty"`

	// Pause is used to pause reconciliation of the BKEMachine.
	// +optional
	Pause bool `json:"pause,omitempty"`

	// DryRun is used to dry run the BKEMachine.
	// +optional
	DryRun bool `json:"dryRun,omitempty"`
}

// BKEMachineStatus defines the observed state of BKEMachine
type BKEMachineStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	// Ready denotes that the machine is ready
	// +optional
	Ready bool `json:"ready"`

	//Bootstrapped means that the machine already has bootstrapped
	// +optional
	Bootstrapped bool `json:"bootstrapped"`

	// +optional
	Addresses []MachineAddress `json:"addresses,omitempty"`

	// +optional
	Conditions clusterv1.Conditions `json:"conditions,omitempty"`

	// +optional
	Node *confv1beta1.Node `json:"node,omitempty"`
}

// MachineAddress contains information of the node's address.
type MachineAddress struct {
	// The machine address type, could be one of Hostname, ExternalIP or InternalIP.
	Type MachineAddressType `json:"type"`

	// The address of machine
	Address string `json:"address"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="HOSTNAME",type="string",JSONPath=".status.node.hostname",description="The hostname of the machine"
// +kubebuilder:printcolumn:name="IP",type="string",JSONPath=".status.node.ip",description="The ip of the machine"
// +kubebuilder:printcolumn:name="PROVIDER-ID",type="string",JSONPath=".spec.providerID"
// +kubebuilder:printcolumn:name="BOOTSTRAPPED",type="boolean",JSONPath=".status.bootstrapped"
// +kubebuilder:metadata:labels={"cluster.x-k8s.io/provider=infrastructure-bke", "cluster.x-k8s.io/v1beta1=v1beta1"}
// BKEMachine is the Schema for the bkemachines API
type BKEMachine struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   BKEMachineSpec   `json:"spec,omitempty"`
	Status BKEMachineStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
// BKEMachineList contains a list of BKEMachine
type BKEMachineList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []BKEMachine `json:"items"`
}

func init() {
	SchemeBuilder.Register(&BKEMachine{}, &BKEMachineList{})
}

// GetConditions returns the set of conditions for this object.
func (c *BKEMachine) GetConditions() clusterv1.Conditions {
	return c.Status.Conditions
}

// SetConditions sets the conditions on this object.
func (c *BKEMachine) SetConditions(conditions clusterv1.Conditions) {
	c.Status.Conditions = conditions
}
