/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

// Package controller
package controller

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// 注册 Oenuma 类型到 scheme
var (
	schemeGroupVersion = schema.GroupVersion{Group: "resource.sop.huawei.com", Version: "v1alpha1"}
	schemeBuilder      = runtime.NewSchemeBuilder(addOenumaKnownTypes)
	addOenumaToScheme  = schemeBuilder.AddToScheme
)

func addOenumaKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(schemeGroupVersion,
		&Oenuma{},
		&OenumaList{},
	)
	metav1.AddToGroupVersion(scheme, schemeGroupVersion)
	return nil
}

// Oenuma 是 oenumas API 的结构体定义
type Oenuma struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   OenumaSpec   `json:"spec,omitempty"`
	Status OenumaStatus `json:"status,omitempty"`
}

// OenumaSpec 定义了 Oenuma 的期望状态
type OenumaSpec struct {
	Name         string       `json:"name,omitempty"`
	Replicas     int32        `json:"replicas,omitempty"`
	Updateenable string       `json:"updateenable,omitempty"`
	Node         []OenumaNode `json:"node,omitempty"`
}

// OenumaNode 定义了节点配置
type OenumaNode struct {
	Name        string              `json:"name,omitempty"`
	Numa        []NumaConfig        `json:"numa,omitempty"`
	PodAffinity []PodAffinityConfig `json:"podAffinity,omitempty"`
}

// NumaConfig 定义了NUMA配置
type NumaConfig struct {
	NumaNum int32  `json:"numaNum"`
	CPUSet  string `json:"cpuset,omitempty"`
	MemSet  string `json:"memset,omitempty"`
}

// PodAffinityConfig 定义了Pod亲和性配置
type PodAffinityConfig struct {
	NumaNum    int32                     `json:"numaNum"`
	Namespace  string                    `json:"namespace,omitempty"`
	PodName    string                    `json:"podName,omitempty"`
	Containers []ContainerAffinityConfig `json:"containers,omitempty"`
}

// ContainerAffinityConfig 定义了容器亲和性配置
type ContainerAffinityConfig struct {
	Name        string `json:"name"`
	ContainerId string `json:"containerId"`
	CPUSet      string `json:"cpuset"`
	MemSet      string `json:"memset"`
}

// OenumaStatus 定义了 Oenuma 的观察状态
type OenumaStatus struct {
	// 可以根据需要添加状态字段
}

// OenumaList 包含 Oenuma 对象的列表
type OenumaList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Oenuma `json:"items"`
}

// DeepCopyInto 实现 runtime.Object 接口
func (in *Oenuma) DeepCopyInto(out *Oenuma) {
	*out = *in
	out.TypeMeta = in.TypeMeta
	in.ObjectMeta.DeepCopyInto(&out.ObjectMeta)
	in.Spec.DeepCopyInto(&out.Spec)
	out.Status = in.Status
}

// DeepCopy 实现 runtime.Object 接口
func (in *Oenuma) DeepCopy() *Oenuma {
	if in == nil {
		return nil
	}
	out := new(Oenuma)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyObject 实现 runtime.Object 接口
func (in *Oenuma) DeepCopyObject() runtime.Object {
	if c := in.DeepCopy(); c != nil {
		return c
	}
	return nil
}

// DeepCopyInto 实现 OenumaList 的深拷贝
func (in *OenumaList) DeepCopyInto(out *OenumaList) {
	*out = *in
	out.TypeMeta = in.TypeMeta
	in.ListMeta.DeepCopyInto(&out.ListMeta)
	if in.Items != nil {
		in, out := &in.Items, &out.Items
		*out = make([]Oenuma, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}
}

// DeepCopy 实现 OenumaList 的深拷贝
func (in *OenumaList) DeepCopy() *OenumaList {
	if in == nil {
		return nil
	}
	out := new(OenumaList)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyObject 实现 runtime.Object 接口
func (in *OenumaList) DeepCopyObject() runtime.Object {
	if c := in.DeepCopy(); c != nil {
		return c
	}
	return nil
}

// DeepCopyInto 实现 OenumaSpec 的深拷贝
func (in *OenumaSpec) DeepCopyInto(out *OenumaSpec) {
	*out = *in
	if in.Node != nil {
		in, out := &in.Node, &out.Node
		*out = make([]OenumaNode, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}
}

// DeepCopy 实现 OenumaSpec 的深拷贝
func (in *OenumaSpec) DeepCopy() *OenumaSpec {
	if in == nil {
		return nil
	}
	out := new(OenumaSpec)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto 实现 OenumaNode 的深拷贝
func (in *OenumaNode) DeepCopyInto(out *OenumaNode) {
	*out = *in
	if in.Numa != nil {
		in, out := &in.Numa, &out.Numa
		*out = make([]NumaConfig, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}
	if in.PodAffinity != nil {
		in, out := &in.PodAffinity, &out.PodAffinity
		*out = make([]PodAffinityConfig, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}
}

// DeepCopy 实现 OenumaNode 的深拷贝
func (in *OenumaNode) DeepCopy() *OenumaNode {
	if in == nil {
		return nil
	}
	out := new(OenumaNode)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto 实现 NumaConfig 的深拷贝
func (in *NumaConfig) DeepCopyInto(out *NumaConfig) {
	*out = *in
}

// DeepCopy 实现 NumaConfig 的深拷贝
func (in *NumaConfig) DeepCopy() *NumaConfig {
	if in == nil {
		return nil
	}
	out := new(NumaConfig)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto 实现 PodAffinityConfig 的深拷贝
func (in *PodAffinityConfig) DeepCopyInto(out *PodAffinityConfig) {
	*out = *in
	if in.Containers != nil {
		in, out := &in.Containers, &out.Containers
		*out = make([]ContainerAffinityConfig, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}
}

// DeepCopy 实现 PodAffinityConfig 的深拷贝
func (in *PodAffinityConfig) DeepCopy() *PodAffinityConfig {
	if in == nil {
		return nil
	}
	out := new(PodAffinityConfig)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto 实现 ContainerAffinityConfig 的深拷贝
func (in *ContainerAffinityConfig) DeepCopyInto(out *ContainerAffinityConfig) {
	*out = *in
}

// DeepCopy 实现 ContainerAffinityConfig 的深拷贝
func (in *ContainerAffinityConfig) DeepCopy() *ContainerAffinityConfig {
	if in == nil {
		return nil
	}
	out := new(ContainerAffinityConfig)
	in.DeepCopyInto(out)
	return out
}
