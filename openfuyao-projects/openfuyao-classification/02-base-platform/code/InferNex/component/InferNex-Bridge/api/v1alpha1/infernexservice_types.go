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

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	igwapiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// SourceRef points to an optional LLMInferenceService.
type SourceRef struct {
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Name       string `json:"name"`
	Namespace  string `json:"namespace,omitempty"`
}

// LLMModelSpec defines model metadata for inference.
type LLMModelSpec struct {
	URI  string `json:"uri"`
	Name string `json:"name,omitempty"`
}

// InferenceEngineWorkloadSpec describes a single backend workload template.
type InferenceEngineWorkloadSpec struct {
	// Replicas sets how many independent workload instances to run horizontally:
	//   Deployment: spec.replicas = Pod count.
	//   LeaderWorkerSet: spec.replicas = group count (each group has groupSize Pods; see dataParallelSize/Local).
	// Omit to allow external autoscalers (ElasticScaler / RSG) to manage spec.replicas;
	// the controller defaults to 1 on first create only (1 Pod for Deployment, 1 group for LWS).
	// +optional
	Replicas *int32 `json:"replicas,omitempty"`
	// Template is the Pod template for aggregate or PD decode workloads.
	// Deployment pods and LWS leader pods use template.
	// +optional
	Template *corev1.PodTemplateSpec `json:"template,omitempty"`
	// Worker is the Pod template for LWS non-leader pods only (ignored when groupSize==1 / Deployment).
	// When omitted or empty (no containers) and groupSize>1, worker pods use template.
	// +optional
	Worker *corev1.PodTemplateSpec `json:"worker,omitempty"`
	// DataParallelSize is the global data-parallel size (vLLM --data-parallel-size). Defaults to 1.
	// groupSize = dataParallelSize / dataParallelSizeLocal selects Deployment (1) vs LeaderWorkerSet (>1).
	// +kubebuilder:validation:Minimum=1
	// +optional
	DataParallelSize *int32 `json:"dataParallelSize,omitempty"`
	// DataParallelSizeLocal is the number of DP ranks per Pod. Defaults to 1.
	// +kubebuilder:validation:Minimum=1
	// +optional
	DataParallelSizeLocal *int32 `json:"dataParallelSizeLocal,omitempty"`
}

// ComponentSpec describes a generic optional component.
type ComponentSpec struct {
	Enabled     *bool `json:"enabled,omitempty"`
	Replicas    int32 `json:"replicas,omitempty"`
	ServicePort int32 `json:"servicePort,omitempty"`
	// Template is the Pod template for the component Deployment (metadata + spec).
	// Container images and imagePullPolicy are set on spec.containers (same as a Deployment pod template).
	// +optional
	Template *corev1.PodTemplateSpec `json:"template,omitempty"`
}

// TemplateComponentSpec describes a component controlled by a parent enabled flag.
type TemplateComponentSpec struct {
	Replicas    int32                   `json:"replicas,omitempty"`
	ServicePort int32                   `json:"servicePort,omitempty"`
	Template    *corev1.PodTemplateSpec `json:"template,omitempty"`
}

// EnabledComponentSpec toggles an InferNex-managed optional component.
// The pod template is supplied by the controller's built-in platform assets.
type EnabledComponentSpec struct {
	Enabled *bool `json:"enabled,omitempty"`
}

// InferenceEngineSpec defines inference engine workload(s).
// Aggregate: set template (and optional worker/dp fields) only.
// PD: set template for decode at the root and prefill under prefill (KServe-style).
type InferenceEngineSpec struct {
	InferenceEngineWorkloadSpec `json:",inline"`
	// Prefill is the prefill role workload. When prefill has a non-empty template, mode is PD.
	// +optional
	Prefill *InferenceEngineWorkloadSpec `json:"prefill,omitempty"`
}

// NamedRef points to external route/gateway resources by name.
type NamedRef struct {
	Name string `json:"name"`
}

// HTTPRouteRefSpec defines BYO HTTPRoute refs and optional managed HTTPRoute spec.
type HTTPRouteRefSpec struct {
	Ref *NamedRef `json:"ref,omitempty"`
	// +optional
	Spec *gwapiv1.HTTPRouteSpec `json:"spec,omitempty"`
}

// GatewayRefSpec defines BYO Gateway refs and optional managed Gateway spec.
type GatewayRefSpec struct {
	Ref *NamedRef `json:"ref,omitempty"`
	// +optional
	Spec *gwapiv1.GatewaySpec `json:"spec,omitempty"`
}

// InferencePoolRefSpec defines BYO InferencePool refs and optional managed InferencePool spec.
type InferencePoolRefSpec struct {
	Ref *NamedRef `json:"ref,omitempty"`
	// +optional
	Spec *igwapiv1.InferencePoolSpec `json:"spec,omitempty"`
}

// IntelligentGatewayRoutingSpec defines router and gateway-api relation.
type IntelligentGatewayRoutingSpec struct {
	Router        *ComponentSpec        `json:"router,omitempty"`
	HTTPRoute     *HTTPRouteRefSpec     `json:"httpRoute,omitempty"`
	Gateway       *GatewayRefSpec       `json:"gateway,omitempty"`
	InferencePool *InferencePoolRefSpec `json:"inferencePool,omitempty"`
}

// CacheIndexerComponentSpec toggles cache-indexer; pod template comes from controller built-in assets.
// Engine discovery env is derived from the reconcile-time effective spec (including baseRefs merge).
type CacheIndexerComponentSpec struct {
	Enabled  *bool `json:"enabled,omitempty"`
	Replicas int32 `json:"replicas,omitempty"`
}

// MooncakeComponentSpec groups mooncake sub-components.
type MooncakeComponentSpec struct {
	Enabled *bool                  `json:"enabled,omitempty"`
	Master  *TemplateComponentSpec `json:"master,omitempty"`
}

// PDOrchestratorComponentSpec groups pd-orchestrator sub-components.
type PDOrchestratorComponentSpec struct {
	ElasticScaler        *EnabledComponentSpec `json:"elasticScaler,omitempty"`
	Tidal                *EnabledComponentSpec `json:"tidal,omitempty"`
	ResourceScalingGroup *EnabledComponentSpec `json:"resourceScalingGroup,omitempty"`
}

// EagleEyeComponentSpec groups eagle-eye sub-components.
type EagleEyeComponentSpec struct {
	HardwareMonitor            *EnabledComponentSpec `json:"hardwareMonitor,omitempty"`
	HardwareDiagnosis          *EnabledComponentSpec `json:"hardwareDiagnosis,omitempty"`
	NetworkPerformanceExporter *EnabledComponentSpec `json:"networkPerformanceExporter,omitempty"`
}

// InfernexComponentsSpec defines optional InferNex side components.
type InfernexComponentsSpec struct {
	CacheIndexer   *CacheIndexerComponentSpec   `json:"cacheIndexer,omitempty"`
	Mooncake       *MooncakeComponentSpec       `json:"mooncake,omitempty"`
	PDOrchestrator *PDOrchestratorComponentSpec `json:"pdOrchestrator,omitempty"`
	EagleEye       *EagleEyeComponentSpec       `json:"eagleEye,omitempty"`
}

// InferNexServiceSpec defines the desired state of InferNexService.
type InferNexServiceSpec struct {
	SourceRef *SourceRef `json:"sourceRef,omitempty"`
	// BaseRefs lists InferNexServiceConfig templates to merge in order.
	// Later refs override earlier refs when they set the same field.
	// If empty, controller falls back to mode-based defaults (aggregate/pd).
	BaseRefs                  []NamedRef                     `json:"baseRefs,omitempty"`
	Model                     *LLMModelSpec                  `json:"model,omitempty"`
	Engine                    *InferenceEngineSpec           `json:"engine,omitempty"`
	IntelligentGatewayRouting *IntelligentGatewayRoutingSpec `json:"intelligentGatewayRouting,omitempty"`
	Components                *InfernexComponentsSpec        `json:"components,omitempty"`
}

// ComponentStatus describes one component runtime status.
type ComponentStatus struct {
	Ready   bool   `json:"ready"`
	Message string `json:"message,omitempty"`
}

// InferNexComponentStatuses groups status of managed InferNex components.
type InferNexComponentStatuses struct {
	InferenceEngine *ComponentStatus `json:"inferenceEngine,omitempty"`
	HermesRouter    *ComponentStatus `json:"hermesRouter,omitempty"`
	ProxyServer     *ComponentStatus `json:"proxyServer,omitempty"`
	CacheIndexer    *ComponentStatus `json:"cacheIndexer,omitempty"`
	Mooncake        *ComponentStatus `json:"mooncake,omitempty"`
	PDOrchestrator  *ComponentStatus `json:"pdOrchestrator,omitempty"`
	EagleEye        *ComponentStatus `json:"eagleEye,omitempty"`
}

// InferNexServiceStatus defines the observed state of InferNexService.
type InferNexServiceStatus struct {
	// Mode is the effective serving mode: aggregate or pd.
	Mode string `json:"mode,omitempty"`

	// Ready indicates whether InferNex managed components are overall ready.
	Ready bool `json:"ready,omitempty"`

	// ObservedGeneration is the latest generation acted on by controller.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Components summarizes per-component readiness.
	// +optional
	Components *InferNexComponentStatuses `json:"components,omitempty"`

	// conditions represent the current state of the InferNexService resource.
	// Each condition has a unique type and reflects the status of a specific aspect of the resource.
	//
	// Standard condition types include:
	// - "Available": the resource is fully functional
	// - "Progressing": the resource is being created or updated
	// - "Degraded": the resource failed to reach or maintain its desired state
	//
	// The status of each condition is one of True, False, or Unknown.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:resource:shortName=insvc
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Ready",type=boolean,JSONPath=".status.ready"
// +kubebuilder:printcolumn:name="Reason",type=string,JSONPath=".status.conditions[?(@.type=='Ready')].reason"
// +kubebuilder:printcolumn:name="Mode",type=string,JSONPath=".status.mode"
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=".metadata.creationTimestamp"

// InferNexService is the Schema for the infernexservices API
type InferNexService struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is a standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec defines the desired state of InferNexService
	// +required
	Spec InferNexServiceSpec `json:"spec"`

	// status defines the observed state of InferNexService
	// +optional
	Status InferNexServiceStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// InferNexServiceList contains a list of InferNexService
type InferNexServiceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferNexService `json:"items"`
}

func init() {
	SchemeBuilder.Register(&InferNexService{}, &InferNexServiceList{})
}
