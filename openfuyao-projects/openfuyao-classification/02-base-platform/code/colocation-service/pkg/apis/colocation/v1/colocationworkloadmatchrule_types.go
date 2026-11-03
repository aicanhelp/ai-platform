/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ColocationType 混部工作负载的类型
// +enum
type ColocationType string

// QosType 混部Qos类型
type QosType string

const (
	// QosHLS 在线时延敏感Qos标签
	QosHLS QosType = "HLS"
	// QosLS 普通在线业务Qos标签
	QosLS QosType = "LS"
	// QosBE 离线业务Qos标签
	QosBE QosType = "BE"
	// QosNONE 没有设置Qos的业务
	QosNONE QosType = "NONE"
	// QosWorkloadAnn 工作负载的Qos对应的注解
	QosWorkloadAnn = "openfuyao.com/qos-level"

	// ColocationIgnore 混部controller不处理
	ColocationIgnore ColocationType = "ignore"
	// NonColocation 非混部
	NonColocation ColocationType = "non"
	// ColocationOnline 混部在线工作负载
	ColocationOnline ColocationType = "online"
	// ColocationOffline 混部离线工作负载
	ColocationOffline ColocationType = "offline"
	// Oversubscription 混部超卖工作负载
	Oversubscription ColocationType = "oversold"
	// ColocationNodeLabel 混部节点标签
	ColocationNodeLabel string = "node.openfuyao.com/colocation"
	// OversubNodeLabel 超卖节点标签
	OversubNodeLabel string = "node.openfuyao.com/oversold"
	// ColocationTaintLabel 混部污点标签
	ColocationTaintLabel string = ColocationNodeLabel
	// ColocationWorloadLabel 混部工作负载标签
	ColocationWorloadLabel string = "volcano.sh/preemptable"
	// OversubscriptionWorkloadAnn 超卖工作负载注解
	OversubscriptionWorkloadAnn string = "openfuyao.com/oversold"
	// ColocationAnnotationLabel 混部工作负载注解标签
	ColocationAnnotationLabel string = "generated.openfuyao.com/colocation"
	// OversubscriptionCPUResourceAnn 超卖cpu资源扩展资源标签
	OversubscriptionCPUResourceAnn = "kubernetes.io/be-cpu"
	// OversubscriptionMemResourceAnn 超卖memory资源扩展资源标签
	OversubscriptionMemResourceAnn = "kubernetes.io/be-memory"
)

// +genclient
// +kubebuilder:object:root=true
// +kubebuilder:resource:categories=colocation,shortName=ctn
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:printcolumn:name="Colocation",type=string,JSONPath=`.metadata.colocation`

// ColocationWorkloadMatchRule is the Schema for the colocationworkloadmatchrules API
type ColocationWorkloadMatchRule struct {
	metav1.TypeMeta `json:",inline"`

	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec ColocationWorkloadMatchRule的详细规格
	Spec ColocationWorkloadMatchRuleSpec `json:"spec,omitempty"`

	// Status ColocationWorkloadMatchRule的当前状态.
	// +optional
	Status ColocationWorkloadMatchRuleStatus `json:"status,omitempty"`
}

type ColocationWorkloadMatchRuleSpec struct {
	// Selectors 工作负载选择器，用于选择需要混部的工作负载
	// +optional
	// +listType=atomic
	// +kubebuilder:validation:MinItems=0
	// +kubebuilder:validation:MaxItems=64
	Selectors []Selector `json:"selectors"`

	// Colocation 工作负载的混部类型：Online与Offline(默认为在线)
	// +optional
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Enum=Online;Offline
	Colocation ColocationType `json:"colocation,omitempty"`
}

type Selector struct {
	// Namespaces 命名空间，空为所有命名空间
	// +optional
	Namespace string `json:"namespace"`

	// WorkloadTypes 工作负载类型，空为所有工作负载类型
	// +optional
	// +listType=atomic
	WorkloadTypes []string `json:"workLoadTypes,omitempty"`

	// Name 工作负载名称
	// +optional
	Name string `json:"name"`

	// Selector 选择器，用于选择需要混部的工作负载
	// +optional
	Selector *metav1.LabelSelector `json:"selector"`
}

// ColocationWorkloadMatchRuleStatus 混部工作负载匹配规则状态
type ColocationWorkloadMatchRuleStatus struct {
}

//+kubebuilder:object:root=true

// ColocationWorkloadMatchRuleList contains a list of ColocationWorkloadMatchRule
type ColocationWorkloadMatchRuleList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty"`

	// Items ColocationWorkloadMatchRule集合
	Items []ColocationWorkloadMatchRule `json:"items"`
}
