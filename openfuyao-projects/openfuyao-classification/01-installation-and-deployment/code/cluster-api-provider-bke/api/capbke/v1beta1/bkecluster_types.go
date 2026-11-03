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

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=bc
// +kubebuilder:printcolumn:name="PHASE",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="STATE",type="string",JSONPath=".status.clusterHealthState"
// +kubebuilder:printcolumn:name="CLUSTER STATUS",type="string",JSONPath=".status.clusterStatus"
// +kubebuilder:printcolumn:name="ENDPOINT",type="string",JSONPath=".spec.controlPlaneEndpoint.host"
// +kubebuilder:printcolumn:name="ENDPOINT PORT",type="string",JSONPath=".spec.controlPlaneEndpoint.port"
// +kubebuilder:printcolumn:name="VERSION",type="string",JSONPath=".status.kubernetesVersion"
// +kubebuilder:printcolumn:name="AGENT STATUS",type="string",JSONPath=".status.agentStatus.status"
// +kubebuilder:printcolumn:name="CONTAINER RUNTIME",type="string",JSONPath=".spec.clusterConfig.cluster.containerRuntime.cri",priority=1
// +kubebuilder:printcolumn:name="AGE",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:metadata:labels={"cluster.x-k8s.io/provider=infrastructure-bke", "cluster.x-k8s.io/v1beta1=v1beta1"}
// BKECluster is the Schema for the bkeclusters API
type BKECluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   confv1beta1.BKEClusterSpec   `json:"spec,omitempty"`
	Status confv1beta1.BKEClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
// BKEClusterList contains a list of BKECluster
type BKEClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []BKECluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&BKECluster{}, &BKEClusterList{})
}
