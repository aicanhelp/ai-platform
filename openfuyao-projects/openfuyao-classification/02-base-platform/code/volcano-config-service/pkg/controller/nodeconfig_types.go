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

var (
	groupVersion = schema.GroupVersion{Group: "volcano.sh", Version: "v1"}
	builder      = runtime.NewSchemeBuilder(addKnownTypes)
	addToScheme  = builder.AddToScheme
)

func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(groupVersion,
		&NodeConfig{},
		&NodeConfigList{},
	)
	metav1.AddToGroupVersion(scheme, groupVersion)
	return nil
}

// NodeConfig 相关类型定义
type NodeConfig struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   NodeConfigSpec   `json:"spec,omitempty"`
	Status NodeConfigStatus `json:"status,omitempty"`
}

// NodeConfiguration 代表单个节点的配置
type NodeConfiguration struct {
	NodeName               string `json:"nodeName"`
	EnableStaticCPUManager bool   `json:"enableStaticCPUManager"`
	EnableNUMADistance     bool   `json:"enableNUMADistance"`
	KubeletConfigPath      string `json:"kubeletConfigPath,omitempty"`
}

// NodeConfigSpec 定义了节点配置的期望状态
type NodeConfigSpec struct {
	Nodes []NodeConfiguration `json:"nodes"`
}

// NodeConfigStatus 定义了节点配置的当前状态
type NodeConfigStatus struct {
	NodeStatuses   []NodeStatus `json:"nodeStatuses"`
	LastUpdateTime metav1.Time  `json:"lastUpdateTime,omitempty"`
}

// NodeStatus 代表单个节点的状态
type NodeStatus struct {
	NodeName                string      `json:"nodeName"`
	StaticCPUManagerEnabled bool        `json:"staticCPUManagerEnabled"`
	NUMADistanceEnabled     bool        `json:"numaDistanceEnabled"`
	State                   string      `json:"state"`
	Error                   string      `json:"error,omitempty"`
	LastUpdateTime          metav1.Time `json:"lastUpdateTime"`
}

// NodeConfigList 包含了 NodeConfig 的列表
type NodeConfigList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []NodeConfig `json:"items"`
}

// NodeConfigState表示资源状态
const (
	NodeConfigStateConfiguring = "Configuring"
	NodeConfigStateConfigured  = "Configured"
	NodeConfigStateFailed      = "Failed"
)
