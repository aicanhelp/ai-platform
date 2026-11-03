/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// ComponentVersion is the Schema for the componentversions API
type ComponentVersion struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ComponentVersionSpec   `json:"spec,omitempty"`
	Status ComponentVersionStatus `json:"status,omitempty"`
}

// ComponentVersionSpec defines the desired state of ComponentVersion
type ComponentVersionSpec struct {
	Name    string        `json:"name"`
	Type    ComponentType `json:"type"`
	Version string        `json:"version"`
	Inline  *InlineSpec   `json:"inline,omitempty"`
	// YAML configures yaml/manifest component install behavior when Type=yaml.
	// Nil keeps legacy behavior for existing ComponentVersion objects.
	YAML *YAMLSpec `json:"yaml,omitempty"`
	// Helm configures Helm chart install/upgrade behavior when Type=helm.
	// Nil keeps legacy behavior for existing ComponentVersion objects.
	Helm            *HelmSpec           `json:"helm,omitempty"`
	SubComponents   []SubComponent      `json:"subComponents,omitempty"`
	Compatibility   CompatibilitySpec   `json:"compatibility,omitempty"`
	Dependencies    []Dependency        `json:"dependencies,omitempty"`
	UpgradeStrategy UpgradeStrategySpec `json:"upgradeStrategy,omitempty"`
	Resources       []ResourceSpec      `json:"resources,omitempty"`
}

// ComponentType defines the type of component installation
type ComponentType string

const (
	ComponentTypeYAML   ComponentType = "yaml"
	ComponentTypeHelm   ComponentType = "helm"
	ComponentTypeInline ComponentType = "inline"
	ComponentTypeBinary ComponentType = "binary"
)

// InlineSpec defines the inline handler configuration
type InlineSpec struct {
	Handler string `json:"handler"`
	Version string `json:"version"`
}

// YAMLSpec configures YAML/Manifest component Apply / Uninstall / Prune / HealthCheck.
type YAMLSpec struct {
	// Namespace is the optional target namespace for deployment and prune listing.
	Namespace string `json:"namespace,omitempty"`
	// ApplyStrategy selects ServerSideApply (default), Replace, or CreateOnly.
	// +kubebuilder:validation:Enum=ServerSideApply;Replace;CreateOnly
	ApplyStrategy string `json:"applyStrategy,omitempty"`
	// Prune enables pruning resources that match PruneLabelSelector but are absent from the current manifests.
	Prune bool `json:"prune,omitempty"`
	// PruneLabelSelector is required when Prune is true; used to list prune candidates in the cluster.
	PruneLabelSelector map[string]string `json:"pruneLabelSelector,omitempty"`
	// HealthCheck runs after successful Apply when Enabled is true.
	HealthCheck *HealthCheckSpec `json:"healthCheck,omitempty"`
}

// HelmSpec configures Helm chart install/upgrade behavior for Type=helm components.
type HelmSpec struct {
	// Namespace is the Helm release namespace. Empty defaults to the execution namespace.
	Namespace string `json:"namespace,omitempty"`
	// ReleaseName is the Helm release name. Empty defaults to the component name.
	ReleaseName string `json:"releaseName,omitempty"`
	// Chart identifies the chart in the configured chart repository.
	Chart ChartSpec `json:"chart"`
	// ValuesFiles are release-image files placed next to component.yaml, usually values.yaml.
	ValuesFiles []string `json:"valuesFiles,omitempty"`
	// +kubebuilder:pruning:PreserveUnknownFields
	// Values is inline Helm values. It has higher priority than ValuesFiles.
	Values map[string]interface{} `json:"values,omitempty"`
	// Strategy maps selected options to Helm SDK install/upgrade actions.
	Strategy HelmStrategySpec `json:"strategy,omitempty"`
	// HealthCheck runs after a successful Helm install/upgrade when Enabled is true.
	HealthCheck *HealthCheckSpec `json:"healthCheck,omitempty"`
}

// ChartSpec identifies a chart in BKECluster.spec.clusterConfig.cluster.chartRepo.
type ChartSpec struct {
	// Name is the chart name. Empty defaults to the component name.
	Name string `json:"name,omitempty"`
	// Version is the chart version. Empty defaults to the component version.
	Version string `json:"version,omitempty"`
}

// HelmStrategySpec maps ComponentVersion Helm behavior to Helm SDK action options.
type HelmStrategySpec struct {
	// Mode optionally forces Install or Upgrade. Empty means auto by release existence.
	// +kubebuilder:validation:Enum=Install;Upgrade
	Mode string `json:"mode,omitempty"`
	// Wait maps to Helm --wait.
	Wait bool `json:"wait,omitempty"`
	// WaitForJobs maps to Helm --wait-for-jobs.
	WaitForJobs bool `json:"waitForJobs,omitempty"`
	// Timeout is a Go duration string, e.g. "10m". Empty uses Helm defaults.
	Timeout string `json:"timeout,omitempty"`
	// Atomic maps to Helm --atomic. It only enables Helm's built-in atomic behavior.
	Atomic bool `json:"atomic,omitempty"`
	// CreateNamespace maps to Helm install --create-namespace.
	CreateNamespace bool `json:"createNamespace,omitempty"`
}

const (
	HelmModeInstall = "Install"
	HelmModeUpgrade = "Upgrade"
)

// HealthCheckSpec defines shared health-check configuration for YAML/Helm components.
// Runtime converts this CRD type to pkg/healthcheck.Spec.
type HealthCheckSpec struct {
	// Enabled controls whether health checks run after Apply.
	Enabled bool `json:"enabled"`
	// Timeout bounds the overall retry window (Go duration string, e.g. "5m").
	Timeout string `json:"timeout,omitempty"`
	// Interval is the poll period between attempts (Go duration string, e.g. "2s").
	Interval string `json:"interval,omitempty"`
	// Checks is the ordered list of probes; all must pass.
	Checks []HealthCheckItemSpec `json:"checks,omitempty"`
}

// HealthCheckItemSpec is one health probe. Type selects which nested config is used;
// the other nested fields should be nil.
type HealthCheckItemSpec struct {
	// Type is PodReady / EndpointReady / Custom.
	// +kubebuilder:validation:Enum=PodReady;EndpointReady;Custom
	Type string `json:"type"`
	// PodReady is used when Type=PodReady.
	PodReady *PodReadyCheckSpec `json:"podReady,omitempty"`
	// EndpointReady is used when Type=EndpointReady.
	EndpointReady *EndpointReadyCheckSpec `json:"endpointReady,omitempty"`
	// Custom is used when Type=Custom.
	Custom *CustomCheckSpec `json:"custom,omitempty"`
}

// PodReadyCheckSpec checks Pod readiness by label selector.
type PodReadyCheckSpec struct {
	// Namespace of pods to list.
	Namespace string `json:"namespace"`
	// LabelSelector is a Kubernetes label selector string (e.g. "k8s-app=kube-dns").
	LabelSelector string `json:"labelSelector"`
	// MinReady is the minimum Ready pod count; 0 means all listed pods must be Ready.
	MinReady int32 `json:"minReady,omitempty"`
}

// EndpointReadyCheckSpec checks that a Service has ready endpoints.
type EndpointReadyCheckSpec struct {
	// Namespace of the Service.
	Namespace string `json:"namespace"`
	// ServiceName is the Endpoints/Service name.
	ServiceName string `json:"serviceName"`
	// Port optionally restricts readiness to a specific port.
	Port int32 `json:"port,omitempty"`
}

// CustomCheckSpec runs a command in the controller Pod; exit code 0 means pass.
type CustomCheckSpec struct {
	// Command is executed via /bin/sh -c (e.g. "curl -s http://.../healthz").
	Command string `json:"command"`
}

// ApplyStrategy constants for YAMLSpec.ApplyStrategy.
const (
	ApplyStrategyServerSideApply = "ServerSideApply"
	ApplyStrategyReplace         = "Replace"
	ApplyStrategyCreateOnly      = "CreateOnly"
)

// SubComponent defines a sub-component reference
type SubComponent struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// CompatibilitySpec defines compatibility constraints
type CompatibilitySpec struct {
	Constraints []Constraint `json:"constraints,omitempty"`
}

// Constraint defines a single compatibility constraint
type Constraint struct {
	Component string `json:"component"`
	Rule      string `json:"rule"`
}

// Dependency defines a dependency on another component
type Dependency struct {
	Name  string `json:"name"`
	Phase string `json:"phase,omitempty"`
}

// UpgradeStrategySpec defines the upgrade strategy for the component
type UpgradeStrategySpec struct {
	Mode          string `json:"mode,omitempty"`
	BatchSize     int    `json:"batchSize,omitempty"`
	Timeout       string `json:"timeout,omitempty"`
	FailurePolicy string `json:"failurePolicy,omitempty"`
}

// ResourceSpec defines a Kubernetes resource to be applied
type ResourceSpec struct {
	Kind       string            `json:"kind"`
	APIVersion string            `json:"apiVersion"`
	Namespace  string            `json:"namespace,omitempty"`
	Name       string            `json:"name"`
	Labels     map[string]string `json:"labels,omitempty"`
	Data       map[string]string `json:"data,omitempty"`
	StringData map[string]string `json:"stringData,omitempty"`
	Manifest   string            `json:"manifest,omitempty"`
}

// ComponentVersionStatus defines the observed state of ComponentVersion
type ComponentVersionStatus struct {
	Phase      string             `json:"phase,omitempty"`
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true

// ComponentVersionList contains a list of ComponentVersion
type ComponentVersionList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ComponentVersion `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ComponentVersion{}, &ComponentVersionList{})
}
