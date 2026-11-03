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

package controller

import (
	corev1 "k8s.io/api/core/v1"
	igwapiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

// mergeSpecFromTemplate overlays src (InferNexServiceConfig) onto dst (InferNexService spec copy).
// Pointers from the API server are not aliased into dst except via DeepCopy.
func mergeSpecFromTemplate(dst *infernexv1alpha1.InferNexServiceSpec, src infernexv1alpha1.InferNexServiceSpec) {
	if dst.Components == nil && src.Components != nil {
		dst.Components = src.Components.DeepCopy()
	}
	if dst.Components != nil && src.Components != nil {
		mergeComponents(dst.Components, src.Components)
	}

	if dst.Engine == nil && src.Engine != nil {
		dst.Engine = src.Engine.DeepCopy()
	}
	if dst.Engine != nil && src.Engine != nil {
		mergeEngine(dst.Engine, src.Engine)
	}

	if dst.Model == nil && src.Model != nil {
		dst.Model = src.Model.DeepCopy()
	}

	if dst.IntelligentGatewayRouting == nil && src.IntelligentGatewayRouting != nil {
		dst.IntelligentGatewayRouting = src.IntelligentGatewayRouting.DeepCopy()
	} else if dst.IntelligentGatewayRouting != nil && src.IntelligentGatewayRouting != nil {
		mergeIntelligentGatewayRouting(dst.IntelligentGatewayRouting, src.IntelligentGatewayRouting)
	}
}

func mergeIntelligentGatewayRouting(dst, src *infernexv1alpha1.IntelligentGatewayRoutingSpec) {
	if src.Router != nil {
		if dst.Router == nil {
			dst.Router = src.Router.DeepCopy()
		} else {
			mergeComponentSpecFields(dst.Router, src.Router)
		}
	}
	mergeGatewayRefSpec(&dst.Gateway, src.Gateway)
	mergeInferencePoolRefSpec(&dst.InferencePool, src.InferencePool)
	mergeHTTPRouteRefSpec(&dst.HTTPRoute, src.HTTPRoute)
}

func mergeGatewayRefSpec(dst **infernexv1alpha1.GatewayRefSpec, src *infernexv1alpha1.GatewayRefSpec) {
	mergeOptionalRefSpec(dst, src, (*infernexv1alpha1.GatewayRefSpec).DeepCopy, func(dst, src *infernexv1alpha1.GatewayRefSpec) {
		mergeIfMissing(&dst.Ref, src.Ref, (*infernexv1alpha1.NamedRef).DeepCopy)
		mergeIfMissing(&dst.Spec, src.Spec, (*gwapiv1.GatewaySpec).DeepCopy)
	})
}

func mergeInferencePoolRefSpec(dst **infernexv1alpha1.InferencePoolRefSpec, src *infernexv1alpha1.InferencePoolRefSpec) {
	mergeOptionalRefSpec(dst, src, (*infernexv1alpha1.InferencePoolRefSpec).DeepCopy, func(dst, src *infernexv1alpha1.InferencePoolRefSpec) {
		mergeIfMissing(&dst.Ref, src.Ref, (*infernexv1alpha1.NamedRef).DeepCopy)
		mergeIfMissing(&dst.Spec, src.Spec, (*igwapiv1.InferencePoolSpec).DeepCopy)
	})
}

func mergeHTTPRouteRefSpec(dst **infernexv1alpha1.HTTPRouteRefSpec, src *infernexv1alpha1.HTTPRouteRefSpec) {
	mergeOptionalRefSpec(dst, src, (*infernexv1alpha1.HTTPRouteRefSpec).DeepCopy, func(dst, src *infernexv1alpha1.HTTPRouteRefSpec) {
		mergeIfMissing(&dst.Ref, src.Ref, (*infernexv1alpha1.NamedRef).DeepCopy)
		mergeIfMissing(&dst.Spec, src.Spec, (*gwapiv1.HTTPRouteSpec).DeepCopy)
	})
}

func mergeOptionalRefSpec[T any](dst **T, src *T, copy func(*T) *T, merge func(dst, src *T)) {
	if src == nil {
		return
	}
	if *dst == nil {
		*dst = copy(src)
		return
	}
	merge(*dst, src)
}

func mergeIfMissing[T any](dst **T, src *T, copy func(*T) *T) {
	if *dst == nil && src != nil {
		*dst = copy(src)
	}
}

func mergeComponents(dst, src *infernexv1alpha1.InfernexComponentsSpec) {
	if dst.CacheIndexer == nil && src.CacheIndexer != nil {
		dst.CacheIndexer = src.CacheIndexer.DeepCopy()
	}
	if dst.CacheIndexer != nil && src.CacheIndexer != nil {
		mergeCacheIndexer(dst.CacheIndexer, src.CacheIndexer)
	}

	if dst.Mooncake == nil && src.Mooncake != nil {
		dst.Mooncake = src.Mooncake.DeepCopy()
	}
	if dst.Mooncake != nil && src.Mooncake != nil {
		mergeMooncake(dst.Mooncake, src.Mooncake)
	}

	if dst.PDOrchestrator == nil && src.PDOrchestrator != nil {
		dst.PDOrchestrator = src.PDOrchestrator.DeepCopy()
	}
	if dst.PDOrchestrator != nil && src.PDOrchestrator != nil {
		mergePDOrchestrator(dst.PDOrchestrator, src.PDOrchestrator)
	}

	if dst.EagleEye == nil && src.EagleEye != nil {
		dst.EagleEye = src.EagleEye.DeepCopy()
	}
	if dst.EagleEye != nil && src.EagleEye != nil {
		mergeEagleEye(dst.EagleEye, src.EagleEye)
	}
}

func mergeCacheIndexer(dst, src *infernexv1alpha1.CacheIndexerComponentSpec) {
	if dst.Enabled == nil {
		dst.Enabled = src.Enabled
	}
	if dst.Replicas == 0 && src.Replicas != 0 {
		dst.Replicas = src.Replicas
	}
}

func mergeMooncake(dst, src *infernexv1alpha1.MooncakeComponentSpec) {
	if dst.Enabled == nil {
		dst.Enabled = src.Enabled
	}
	mergePtrTemplateComponentSpec(&dst.Master, src.Master)
}

func mergePDOrchestrator(dst, src *infernexv1alpha1.PDOrchestratorComponentSpec) {
	mergePtrEnabledComponentSpec(&dst.ElasticScaler, src.ElasticScaler)
	mergePtrEnabledComponentSpec(&dst.Tidal, src.Tidal)
	mergePtrEnabledComponentSpec(&dst.ResourceScalingGroup, src.ResourceScalingGroup)
}

func mergeEagleEye(dst, src *infernexv1alpha1.EagleEyeComponentSpec) {
	mergePtrEnabledComponentSpec(&dst.HardwareMonitor, src.HardwareMonitor)
	mergePtrEnabledComponentSpec(&dst.HardwareDiagnosis, src.HardwareDiagnosis)
	mergePtrEnabledComponentSpec(&dst.NetworkPerformanceExporter, src.NetworkPerformanceExporter)
}

func mergePtrEnabledComponentSpec(dst **infernexv1alpha1.EnabledComponentSpec, src *infernexv1alpha1.EnabledComponentSpec) {
	if src == nil {
		return
	}
	if *dst == nil {
		*dst = src.DeepCopy()
		return
	}
	if (*dst).Enabled == nil {
		(*dst).Enabled = src.Enabled
	}
}

func mergePtrComponentSpec(dst **infernexv1alpha1.ComponentSpec, src *infernexv1alpha1.ComponentSpec) {
	if src == nil {
		return
	}
	if *dst == nil {
		*dst = src.DeepCopy()
		return
	}
	mergeComponentSpecFields(*dst, src)
}

func mergePtrTemplateComponentSpec(dst **infernexv1alpha1.TemplateComponentSpec, src *infernexv1alpha1.TemplateComponentSpec) {
	if src == nil {
		return
	}
	if *dst == nil {
		*dst = src.DeepCopy()
		return
	}
	mergeTemplateComponentSpecFields(*dst, src)
}

func mergeTemplateComponentSpecFields(dst, src *infernexv1alpha1.TemplateComponentSpec) {
	if dst.Replicas == 0 && src.Replicas != 0 {
		dst.Replicas = src.Replicas
	}
	if dst.ServicePort == 0 && src.ServicePort != 0 {
		dst.ServicePort = src.ServicePort
	}
	if dst.Template == nil && src.Template != nil {
		dst.Template = src.Template.DeepCopy()
	}
	if dst.Template != nil && src.Template != nil {
		mergePodTemplateSpecs(dst.Template, src.Template)
	}
}

func mergeEngine(dst, src *infernexv1alpha1.InferenceEngineSpec) {
	if dst == nil || src == nil {
		return
	}
	mergeInferenceEngineWorkloadSpec(&dst.InferenceEngineWorkloadSpec, &src.InferenceEngineWorkloadSpec)
	if dst.Prefill == nil && src.Prefill != nil {
		dst.Prefill = src.Prefill.DeepCopy()
	}
	if dst.Prefill != nil && src.Prefill != nil {
		mergeInferenceEngineWorkloadSpec(dst.Prefill, src.Prefill)
	}
}

func mergeInferenceEngineWorkloadSpec(dst, src *infernexv1alpha1.InferenceEngineWorkloadSpec) {
	if dst.Replicas == nil && src.Replicas != nil {
		v := *src.Replicas
		dst.Replicas = &v
	}
	if dst.DataParallelSize == nil && src.DataParallelSize != nil {
		v := *src.DataParallelSize
		dst.DataParallelSize = &v
	}
	if dst.DataParallelSizeLocal == nil && src.DataParallelSizeLocal != nil {
		v := *src.DataParallelSizeLocal
		dst.DataParallelSizeLocal = &v
	}
	if dst.Template == nil && src.Template != nil {
		dst.Template = src.Template.DeepCopy()
	}
	if dst.Template != nil && src.Template != nil {
		mergePodTemplateSpecs(dst.Template, src.Template)
	}
	if dst.Worker == nil && src.Worker != nil {
		dst.Worker = src.Worker.DeepCopy()
	}
	if dst.Worker != nil && src.Worker != nil {
		mergePodTemplateSpecs(dst.Worker, src.Worker)
	}
}

// WorkloadWorkerTemplateEffective returns the worker PodTemplateSpec when merge-complete
// and at least one container is defined; otherwise nil (worker not set).
func WorkloadWorkerTemplateEffective(w *infernexv1alpha1.InferenceEngineWorkloadSpec) *corev1.PodTemplateSpec {
	if w == nil || w.Worker == nil || len(w.Worker.Spec.Containers) == 0 {
		return nil
	}
	return w.Worker
}

func mergeComponentSpecFields(dst, src *infernexv1alpha1.ComponentSpec) {
	if dst.Replicas == 0 && src.Replicas != 0 {
		dst.Replicas = src.Replicas
	}
	if dst.ServicePort == 0 && src.ServicePort != 0 {
		dst.ServicePort = src.ServicePort
	}
	if dst.Enabled == nil {
		dst.Enabled = src.Enabled
	}
	if dst.Template == nil && src.Template != nil {
		dst.Template = src.Template.DeepCopy()
	}
	if dst.Template != nil && src.Template != nil {
		mergePodTemplateSpecs(dst.Template, src.Template)
	}
}

// mergePodTemplateSpecs fills gaps in dst using src (dst is user/service, src is config template).
func mergePodTemplateSpecs(dst, src *corev1.PodTemplateSpec) {
	if src == nil {
		return
	}
	if dst.Labels == nil {
		dst.Labels = map[string]string{}
	}
	for k, v := range src.Labels {
		if _, ok := dst.Labels[k]; !ok {
			dst.Labels[k] = v
		}
	}
	if dst.Annotations == nil {
		dst.Annotations = map[string]string{}
	}
	for k, v := range src.Annotations {
		if _, ok := dst.Annotations[k]; !ok {
			dst.Annotations[k] = v
		}
	}
	if len(dst.Spec.Containers) == 0 && len(src.Spec.Containers) > 0 {
		dst.Spec = *src.Spec.DeepCopy()
		return
	}
	for i := range dst.Spec.Containers {
		if i >= len(src.Spec.Containers) {
			break
		}
		if dst.Spec.Containers[i].Image == "" {
			dst.Spec.Containers[i].Image = src.Spec.Containers[i].Image
		}
		if dst.Spec.Containers[i].ImagePullPolicy == "" {
			dst.Spec.Containers[i].ImagePullPolicy = src.Spec.Containers[i].ImagePullPolicy
		}
	}
}
