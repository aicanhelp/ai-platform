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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// KAEPolicySpec 定义 KAEPolicy 的期望状态
type KAEPolicySpec struct {
	// Operator KAE Operator 规格
	Operator OperatorSpec `json:"operator"`

	// DaemonSet 定义所有 DaemonSets 的通用配置
	DaemonSets DaemonSetsSpec `json:"daemonSets"`

	// Driver 驱动配置
	Driver DriverSpec `json:"driver"`

	// DevicePlugin 设备插件配置
	DevicePlugin DevicePluginSpec `json:"devicePlugin"`

	// PSA  PodSecurityAdmission 配置
	PSA PSASpec `json:"psa,omitempty"`
}

// OperatorSpec Operator 配置
type OperatorSpec struct {
	// EnabledHPRE是否使能 HPRE 硬件
	EnabledHPRE *bool `json:"enabledHPRE,omitempty"`

	// 当前设备插件仅实现了 HPRE 设备，此版本暂时只支持 HPRE 设备，此处为预留字段
	// EnabledSEC 是否使能 SEC 设备
	EnabledSEC *bool `json:"enabledSEC,omitempty"`

	// EnabledZIP 是否使能 ZIP 设备
	EnabledZIP *bool `json:"enabledZIP,omitempty"`
}

// DaemonSetsSpec Operator 管理的所有 DaemonSets 的通用配置
type DaemonSetsSpec struct {
	// Labels 可选：通用标签配置
	Labels map[string]string `json:"labels,omitempty"`

	// Annotations 可选：通用注解配置
	Annotations map[string]string `json:"annotations,omitempty"`

	// Tolerations 可选：通用容忍配置
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`

	// PriorityClassName 可选：通用优先级配置
	// +kubebuilder:validation:Optional
	PriorityClassName string `json:"priorityClassName,omitempty"`

	// UpdateStrategy 可选：更新策略
	// +kubebuilder:validation:Optional
	// +kubebuilder:default=RollingUpdate
	// +kubebuilder:validation:Enum=RollingUpdate;OnDelete
	UpdateStrategy string `json:"updateStrategy,omitempty"`

	// RollingUpdate 可选: DaemonSet 所有 Pod 的滚动更新配置
	RollingUpdate *RollingUpdateSpec `json:"rollingUpdate,omitempty"`
}

// DriverSpec KAE 驱动程序部署的属性
type DriverSpec struct {
	// Enabled 是否已启用通过 Operator 部署 KAE 驱动程序
	Enabled *bool `json:"enabled,omitempty"`

	// EnabledCleanup 表示是否启用了在服务结束时清理驱动的功能。
	// +kubebuilder:validation:Optional
	EnabledCleanup *bool `json:"enabledCleanup"`

	// Repository KAE 驱动程序镜像仓库
	// +kubebuilder:validation:Optional
	Repository string `json:"repository,omitempty"`

	// Image 镜像名称
	// +kubebuilder:validation:Pattern=[a-zA-Z0-9\-]+
	Image string `json:"image,omitempty"`

	// Version 镜像标签
	// +kubebuilder:validation:Optional
	Version string `json:"version,omitempty"`

	// ImagePullPolicy 镜像拉取策略
	// +kubebuilder:validation:Optional
	ImagePullPolicy string `json:"imagePullPolicy,omitempty"`

	// ImagePullSecrets 镜像拉取密钥
	// +kubebuilder:validation:Optional
	ImagePullSecrets []string `json:"imagePullSecrets,omitempty"`

	// Resources 可选: 定义每个 pod 的资源请求和限制
	Resources *ResourceRequirements `json:"resources,omitempty"`

	// Args 可选: 参数列表
	Args []string `json:"args,omitempty"`

	// Env 可选: 环境变量列表
	Env []EnvVar `json:"env,omitempty"`
}

// DevicePluginSpec KAE 设备插件程序部署的属性
type DevicePluginSpec struct {
	// EnabledHPRE 是否已启用通过 Operator 部署 KAE hpre 设备插件程序
	EnabledHPRE *bool `json:"enabledHPRE,omitempty"`

	// EnabledSEC 是否已启用通过 Operator 部署 KAE sec 设备插件程序，此版本暂时只支持 HPRE 设备，此处为预留字段
	EnabledSEC *bool `json:"enabledSEC,omitempty"`

	// EnabledZIP 是否已启用通过 Operator 部署 KAE zip 设备插件程序，此版本暂时只支持 HPRE 设备，此处为预留字段
	EnabledZIP *bool `json:"enabledZIP,omitempty"`

	// Repository KAE 设备插件程序镜像仓库
	// +kubebuilder:validation:Optional
	Repository string `json:"repository,omitempty"`

	// Image KAE Device Plugin 镜像名称
	// +kubebuilder:validation:Pattern=[a-zA-Z0-9\-]+
	Image string `json:"image,omitempty"`

	// Version KAE Device Plugin 镜像标签
	// +kubebuilder:validation:Optional
	Version string `json:"version,omitempty"`

	// ImagePullPolicy KAE Device Plugin 镜像拉取策略
	// +kubebuilder:validation:Optional
	ImagePullPolicy string `json:"imagePullPolicy,omitempty"`

	// ImagePullSecrets KAE Device Plugin 镜像拉取密钥
	ImagePullSecrets []string `json:"imagePullSecrets,omitempty"`

	// Resources 可选: 定义每个 pod 的资源请求和限制
	Resources *ResourceRequirements `json:"resources,omitempty"`

	// Args 可选: 参数列表
	Args []string `json:"args,omitempty"`

	// Env 可选: 环境变量列表
	Env []EnvVar `json:"env,omitempty"`
}

// PSASpec 适用于所有 Pod 的 PodSecurityAdmission 配置
type PSASpec struct {
	// Enabled 是否需要为所有 Pod 启用 PodSecurityAdmission 配置
	Enabled *bool `json:"enabled,omitempty"`
}

// RollingUpdateSpec 定义所有 DaemonSet pod 的滚动更新配置
type RollingUpdateSpec struct {
	// MaxUnavailable 最大不可用 Pod 数
	// +kubebuilder:validation:Optional
	MaxUnavailable string `json:"maxUnavailable,omitempty"`
}

// ResourceRequirements 计算资源需求
type ResourceRequirements struct {
	// Limits 可选，描述了允许使用的最大计算资源数量。
	Limits corev1.ResourceList `json:"limits,omitempty"`

	// Requests 可选，描述了所需计算资源的最低数量。
	Requests corev1.ResourceList `json:"requests,omitempty"`
}

// EnvVar 容器中的环境变量。
type EnvVar struct {
	// Name 环境变量名称
	Name string `json:"name"`

	// Value 环境变量值
	Value string `json:"value,omitempty"`
}

// State 组件状态
type State string

const (
	// Ignored 表示 CR 实例重复，被忽略。
	Ignored State = "Ignored"

	// Ready 表示 CR 的所有组件均已就绪
	Ready State = "Ready"

	// NotReady 标识 CR 的部分或全部组件没有就绪
	NotReady State = "NotReady"

	// Disabled 标识组件不启用
	Disabled State = "Disabled"
)

// KAEPolicyStatus 定义 KAEPolicy 的观测状态
type KAEPolicyStatus struct {
	// State KAEPolicy 的状态
	// +kubebuilder:validation:Enum=Ignored;Ready;NotReady;Disabled
	State State `json:"state"`

	// Namespace 命名空间
	Namespace string `json:"namespace,omitempty"`

	// Conditions 当前状态的条件列表
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// +kubebuilder:printcolumn:name="Status",type=string,JSONPath=`.status.state`,priority=0
// +kubebuilder:printcolumn:name="Age",type=string,JSONPath=`.metadata.creationTimestamp`,priority=0

// KAEPolicy kaepolicies 的 API 规范
type KAEPolicy struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   KAEPolicySpec   `json:"spec,omitempty"`
	Status KAEPolicyStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// KAEPolicyList KAEPolicy 列表
type KAEPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []KAEPolicy `json:"items"`
}

func init() {
	SchemeBuilder.Register(&KAEPolicy{}, &KAEPolicyList{})
}
