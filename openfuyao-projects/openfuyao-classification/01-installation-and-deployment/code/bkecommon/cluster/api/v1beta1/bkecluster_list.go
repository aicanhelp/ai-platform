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

package v1beta1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	GVK = schema.GroupVersionKind{
		Group:   "bke.bocloud.com",
		Version: "v1beta1",
		Kind:    "BKECluster",
	}

	// BKENodeGVK is defined in bkenode_types.go
	// Re-exported here for convenience
	NodeGVK = schema.GroupVersionKind{
		Group:   "bke.bocloud.com",
		Version: "v1beta1",
		Kind:    "BKENode",
	}
)

//+kubebuilder:object:root=true

// BKECluster is the Schema for the bkeclusters API
type BKECluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   BKEClusterSpec   `json:"spec,omitempty"`
	Status BKEClusterStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// BKEClusterList contains a list of BKECluster
type BKEClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []BKECluster `json:"items"`
}
