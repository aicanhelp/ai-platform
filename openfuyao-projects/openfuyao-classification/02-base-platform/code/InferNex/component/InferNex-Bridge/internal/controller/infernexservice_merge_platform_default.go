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
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

// Merge order (effective spec for one InferNexService reconcile):
//  1. Start from InferNexService.spec (deep copy).
//  2. For each baseRefs entry in order, mergeSpecFromTemplate(effective, config):
//     config fills gaps in effective; where both sides set values, field-level rules apply
//     (e.g. mergePodTemplateSpecs keeps user labels/annotations and fills empty container images from config).
//  3. Direct mode: serving mode is re-derived from merged engine.prefill (aggregate vs pd).
//  4. mergePlatformDefaultTemplates: load infernex-default-aggregate-template or
//     infernex-default-pd-template (from mode).
//     - Linked LLMInferenceService (spec.sourceRef): merges platform spec.components into effective so sidecars
//       follow the same chart defaults as KServe-adjacent flows; then fills any still-missing pod templates.
//     - Direct InferNexService (no sourceRef): KServe-like well-known overlay — after user/baseRefs merge,
//       mergeSpecFromTemplate(effective, platformSpec) fills any remaining gaps from the chart-installed
//       InferNexServiceConfig (same objects as infernexserviceconfig-{aggregate,pd}.yaml templates), including
//       servicePort, replicas where zero, nil pod templates, and omitted subtrees. User/baseRefs values win
//       when already set (see mergeComponentSpecFields / mergePodTemplateSpecs). If the platform Config is
//       missing and something still needs a pod template, reconcile fails with the same error as before.

func (r *InferNexServiceReconciler) mergePlatformDefaultTemplates(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
	effective *infernexv1alpha1.InferNexServiceSpec,
	platformCfgName string,
) error {
	linked := infsvc.Spec.SourceRef != nil
	cfg, err := r.getInferNexServiceConfig(ctx, platformCfgName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if needsPlatformTemplateFill(effective) {
				return fmt.Errorf(
					"InferNexServiceConfig %q not found (required to fill missing fields/templates); install chart defaults (%s / %s) or set spec.template on enabled components",
					platformCfgName,
					defaultAggregateTemplateName,
					defaultPDTemplateName,
				)
			}
			return nil
		}
		return err
	}
	plat := &cfg.Spec.InferNexServiceSpec
	if linked {
		mergePlatformDefaultComponents(effective, plat)
	} else {
		mergeSpecFromTemplate(effective, *plat)
	}
	fillMissingTemplatesFromPlatformDefault(effective, plat)
	return nil
}

// mergePlatformDefaultComponents overlays platform InferNexServiceConfig.components onto effective.
// Used for InferNexService linked to LLMInferenceService (sourceRef) so platform chart defaults apply.
func mergePlatformDefaultComponents(dst *infernexv1alpha1.InferNexServiceSpec, src *infernexv1alpha1.InferNexServiceSpec) {
	if dst == nil || src == nil || src.Components == nil {
		return
	}
	if dst.Components == nil {
		dst.Components = src.Components.DeepCopy()
		return
	}
	mergeComponents(dst.Components, src.Components)
}

func needsPlatformTemplateFill(spec *infernexv1alpha1.InferNexServiceSpec) bool {
	return routerNeedsTemplate(spec) ||
		componentsNeedTemplate(spec) ||
		engineNeedsTemplate(spec)
}

func routerNeedsTemplate(spec *infernexv1alpha1.InferNexServiceSpec) bool {
	if spec == nil || spec.IntelligentGatewayRouting == nil || spec.IntelligentGatewayRouting.Router == nil {
		return false
	}
	router := spec.IntelligentGatewayRouting.Router
	return enabled(router.Enabled) && router.Template == nil
}

func componentsNeedTemplate(spec *infernexv1alpha1.InferNexServiceSpec) bool {
	if spec == nil || spec.Components == nil {
		return false
	}
	return mooncakeNeedsTemplate(spec.Components.Mooncake)
}

func componentTemplateMissing(spec *infernexv1alpha1.ComponentSpec) bool {
	return spec != nil && enabled(spec.Enabled) && spec.Template == nil
}

func mooncakeNeedsTemplate(spec *infernexv1alpha1.MooncakeComponentSpec) bool {
	return spec != nil && enabled(spec.Enabled) &&
		(spec.Master == nil || templateComponentTemplateMissing(spec.Master))
}

func templateComponentTemplateMissing(spec *infernexv1alpha1.TemplateComponentSpec) bool {
	return spec != nil && spec.Template == nil
}

func engineNeedsTemplate(spec *infernexv1alpha1.InferNexServiceSpec) bool {
	if spec == nil || spec.Engine == nil {
		return false
	}
	engine := spec.Engine
	if !EngineIsPDMode(engine) {
		return engine.Template == nil
	}
	return inferenceWorkloadTemplateMissing(EnginePrefillWorkload(engine)) ||
		inferenceWorkloadTemplateMissing(EngineDecodeWorkload(engine))
}

func inferenceWorkloadTemplateMissing(spec *infernexv1alpha1.InferenceEngineWorkloadSpec) bool {
	return spec != nil && spec.Template == nil
}

func fillMissingTemplatesFromPlatformDefault(
	dst *infernexv1alpha1.InferNexServiceSpec,
	src *infernexv1alpha1.InferNexServiceSpec,
) {
	if dst == nil || src == nil {
		return
	}
	if dst.IntelligentGatewayRouting != nil && dst.IntelligentGatewayRouting.Router != nil &&
		src.IntelligentGatewayRouting != nil && src.IntelligentGatewayRouting.Router != nil {
		fillComponentTemplateIfNeeded(dst.IntelligentGatewayRouting.Router, src.IntelligentGatewayRouting.Router)
	}
	if dst.Components != nil && src.Components != nil {
		fillMooncakeTemplates(dst.Components.Mooncake, src.Components.Mooncake)
	}
	fillEngineTemplates(dst.Engine, src.Engine)
}

func fillComponentTemplateIfNeeded(dst, src *infernexv1alpha1.ComponentSpec) {
	if dst == nil || src == nil {
		return
	}
	if !enabled(dst.Enabled) {
		return
	}
	if dst.Template != nil {
		return
	}
	if src.Template != nil {
		dst.Template = src.Template.DeepCopy()
	}
}

func fillMooncakeTemplates(dst, src *infernexv1alpha1.MooncakeComponentSpec) {
	if dst == nil || src == nil || !enabled(dst.Enabled) {
		return
	}
	if dst.Master == nil && src.Master != nil {
		dst.Master = src.Master.DeepCopy()
	}
	if dst.Master != nil && src.Master != nil {
		fillTemplateComponentIfNeeded(dst.Master, src.Master)
	}
}

func fillTemplateComponentIfNeeded(dst, src *infernexv1alpha1.TemplateComponentSpec) {
	fillPodTemplateIfNeeded(dst, src,
		func(spec *infernexv1alpha1.TemplateComponentSpec) *corev1.PodTemplateSpec { return spec.Template },
		func(spec *infernexv1alpha1.TemplateComponentSpec, tpl *corev1.PodTemplateSpec) { spec.Template = tpl },
	)
}

func fillEngineTemplates(dst, src *infernexv1alpha1.InferenceEngineSpec) {
	if dst == nil || src == nil {
		return
	}
	fillWorkloadTemplate(&dst.InferenceEngineWorkloadSpec, &src.InferenceEngineWorkloadSpec)
	if dst.Prefill != nil && src.Prefill != nil {
		fillWorkloadTemplate(dst.Prefill, src.Prefill)
	}
}

func fillWorkloadTemplate(dst, src *infernexv1alpha1.InferenceEngineWorkloadSpec) {
	fillPodTemplateIfNeeded(dst, src,
		func(spec *infernexv1alpha1.InferenceEngineWorkloadSpec) *corev1.PodTemplateSpec { return spec.Template },
		func(spec *infernexv1alpha1.InferenceEngineWorkloadSpec, tpl *corev1.PodTemplateSpec) { spec.Template = tpl },
	)
}

func fillPodTemplateIfNeeded[T any](
	dst, src *T,
	getTemplate func(*T) *corev1.PodTemplateSpec,
	setTemplate func(*T, *corev1.PodTemplateSpec),
) {
	if dst == nil || src == nil {
		return
	}
	if getTemplate(dst) != nil {
		return
	}
	if tpl := getTemplate(src); tpl != nil {
		setTemplate(dst, tpl.DeepCopy())
	}
}
