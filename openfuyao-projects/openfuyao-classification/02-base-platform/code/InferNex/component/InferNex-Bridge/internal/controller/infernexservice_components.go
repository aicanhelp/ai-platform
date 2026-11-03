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
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

type componentPlan struct {
	// Replicas is nil for inference engine workloads when spec omits replicas (external scaling).
	Replicas       *int32
	Template       *corev1.PodTemplateSpec
	WorkerTemplate *corev1.PodTemplateSpec
	ServicePort    int32
	WorkloadKind   string
	GroupSize      int32
	DisableService bool
	// Empty means default infernex-component; PD workloads set app.kubernetes.io/name to pd group identity (InferNexService name or sourceRef).
	AppKubernetesIOName string

	IsProxyServer       bool
	ProxyPDWorkloadName string
	ProxyPDWorkloadNS   string
}

const (
	defaultWorkloadServicePort = 8000
	workloadKindDeployment     = "Deployment"
	workloadKindDaemonSet      = "DaemonSet"

	// Marks pods created by InferNexService reconcile (distinct from KServe-managed workloads).
	labelInfernexManagedBy       = "infernex.io/managed-by"
	valueInfernexBridgeManagedBy = "infernex-bridge"
)

func (r *InferNexServiceReconciler) buildDesiredComponents(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
	spec infernexv1alpha1.InferNexServiceSpec,
) (map[string]componentPlan, error) {
	out := map[string]componentPlan{}
	manageInferenceRuntime := infsvc.Spec.SourceRef == nil
	if manageInferenceRuntime && spec.Engine != nil {
		if wl := EngineAggregateWorkload(spec.Engine); wl != nil {
			plan, err := buildEngineWorkloadPlan("engine-aggregate", wl)
			if err != nil {
				return nil, err
			}
			out["engine-aggregate"] = plan
		}
		if EngineIsPDMode(spec.Engine) {
			wlName, _ := pdWorkloadIdentity(infsvc)
			if prefill := EnginePrefillWorkload(spec.Engine); prefill != nil {
				plan, err := buildEngineWorkloadPlan("engine-pd-prefill", prefill)
				if err != nil {
					return nil, err
				}
				plan.AppKubernetesIOName = wlName
				out["engine-pd-prefill"] = plan
			}
			if decode := EngineDecodeWorkload(spec.Engine); decode != nil {
				plan, err := buildEngineWorkloadPlan("engine-pd-decode", decode)
				if err != nil {
					return nil, err
				}
				plan.AppKubernetesIOName = wlName
				out["engine-pd-decode"] = plan
			}
		}
	}
	if manageInferenceRuntime &&
		spec.IntelligentGatewayRouting != nil &&
		spec.IntelligentGatewayRouting.Router != nil &&
		enabled(spec.IntelligentGatewayRouting.Router.Enabled) {
		router := spec.IntelligentGatewayRouting.Router
		tpl, err := buildComponentPodTemplate(hermesRouterComponent, router)
		if err != nil {
			return nil, err
		}
		routerPort := router.ServicePort
		if routerPort <= 0 {
			routerPort = defaultEndpointPickerPort
		}
		out[hermesRouterComponent] = componentPlan{
			Replicas:    ptr.To(replicas(router.Replicas)),
			Template:    tpl,
			ServicePort: routerPort,
		}
	}
	if spec.Components != nil {
		if c := spec.Components.CacheIndexer; c != nil && enabled(c.Enabled) {
			comp, err := resolveCacheIndexerComponent(c)
			if err != nil {
				return nil, err
			}
			tpl, err := buildComponentPodTemplate(cacheIndexerComponent, comp)
			if err != nil {
				return nil, err
			}
			out[cacheIndexerComponent] = componentPlan{
				Replicas:    ptr.To(replicas(comp.Replicas)),
				Template:    tpl,
				ServicePort: comp.ServicePort,
			}
		}
		if c := spec.Components.Mooncake; c != nil && enabled(c.Enabled) {
			if c.Master == nil {
				return nil, fmt.Errorf("component %q: spec.components.mooncake.master is required when mooncake is enabled", "mooncake-master")
			}
			tpl, err := buildTemplateComponentPodTemplate("mooncake-master", c.Master)
			if err != nil {
				return nil, err
			}
			mergePodTemplateLabelIfAbsent(tpl, labelOpenFuyaoKVManager, openfuyaoKVManagerMooncake)
			out["mooncake-master"] = componentPlan{
				Replicas:    ptr.To(replicas(c.Master.Replicas)),
				Template:    tpl,
				ServicePort: c.Master.ServicePort,
			}

			comp, err := resolveManagedComponent("mooncake-metadata", &infernexv1alpha1.EnabledComponentSpec{Enabled: ptr.To(true)})
			if err != nil {
				return nil, err
			}
			tpl, err = buildComponentPodTemplate("mooncake-metadata", comp)
			if err != nil {
				return nil, err
			}
			out["mooncake-metadata"] = componentPlan{
				Replicas:    ptr.To(replicas(comp.Replicas)),
				Template:    tpl,
				ServicePort: comp.ServicePort,
			}
		}
		if c := spec.Components.PDOrchestrator; c != nil {
			if c.ElasticScaler != nil && enabled(c.ElasticScaler.Enabled) {
				comp, err := resolveManagedComponent("pd-orchestrator-elastic-scaler", c.ElasticScaler)
				if err != nil {
					return nil, err
				}
				tpl, err := buildComponentPodTemplate("pd-orchestrator-elastic-scaler", comp)
				if err != nil {
					return nil, err
				}
				out["pd-orchestrator-elastic-scaler"] = componentPlan{
					Replicas:    ptr.To(replicas(comp.Replicas)),
					Template:    tpl,
					ServicePort: comp.ServicePort,
				}
			}
			if c.Tidal != nil && enabled(c.Tidal.Enabled) {
				comp, err := resolveManagedComponent("pd-orchestrator-tidal", c.Tidal)
				if err != nil {
					return nil, err
				}
				tpl, err := buildComponentPodTemplate("pd-orchestrator-tidal", comp)
				if err != nil {
					return nil, err
				}
				out["pd-orchestrator-tidal"] = componentPlan{
					Replicas:    ptr.To(replicas(comp.Replicas)),
					Template:    tpl,
					ServicePort: comp.ServicePort,
				}
			}
			if c.ResourceScalingGroup != nil && enabled(c.ResourceScalingGroup.Enabled) {
				comp, err := resolveManagedComponent("pd-orchestrator-rsg", c.ResourceScalingGroup)
				if err != nil {
					return nil, err
				}
				tpl, err := buildComponentPodTemplate("pd-orchestrator-rsg", comp)
				if err != nil {
					return nil, err
				}
				out["pd-orchestrator-rsg"] = componentPlan{
					Replicas:    ptr.To(replicas(comp.Replicas)),
					Template:    tpl,
					ServicePort: comp.ServicePort,
				}
			}
		}
		if c := spec.Components.EagleEye; c != nil {
			if c.HardwareMonitor != nil && enabled(c.HardwareMonitor.Enabled) {
				comp, err := resolveManagedComponent("eagle-eye-hardware-monitor", c.HardwareMonitor)
				if err != nil {
					return nil, err
				}
				tpl, err := buildComponentPodTemplate("eagle-eye-hardware-monitor", comp)
				if err != nil {
					return nil, err
				}
				out["eagle-eye-hardware-monitor"] = componentPlan{
					Replicas:       ptr.To(replicas(comp.Replicas)),
					Template:       tpl,
					ServicePort:    comp.ServicePort,
					WorkloadKind:   workloadKindDaemonSet,
					DisableService: true,
				}
			}
			if c.HardwareDiagnosis != nil && enabled(c.HardwareDiagnosis.Enabled) {
				comp, err := resolveManagedComponent("eagle-eye-hardware-diagnosis", c.HardwareDiagnosis)
				if err != nil {
					return nil, err
				}
				tpl, err := buildComponentPodTemplate("eagle-eye-hardware-diagnosis", comp)
				if err != nil {
					return nil, err
				}
				out["eagle-eye-hardware-diagnosis"] = componentPlan{
					Replicas:       ptr.To(replicas(comp.Replicas)),
					Template:       tpl,
					ServicePort:    comp.ServicePort,
					DisableService: true,
				}
			}
			if c.NetworkPerformanceExporter != nil && enabled(c.NetworkPerformanceExporter.Enabled) {
				comp, err := resolveManagedComponent("eagle-eye-network-performance-exporter", c.NetworkPerformanceExporter)
				if err != nil {
					return nil, err
				}
				tpl, err := buildComponentPodTemplate("eagle-eye-network-performance-exporter", comp)
				if err != nil {
					return nil, err
				}
				out["eagle-eye-network-performance-exporter"] = componentPlan{
					Replicas:     ptr.To(replicas(comp.Replicas)),
					Template:     tpl,
					ServicePort:  comp.ServicePort,
					WorkloadKind: workloadKindDaemonSet,
				}
			}
		}
	}

	// Proxy-server is reconciled for PD workloads (direct or LLMInferenceService-linked); not used in aggregate mode.
	launchProxy, err := r.shouldLaunchProxyServer(ctx, infsvc, spec)
	if err != nil {
		return nil, err
	}
	if launchProxy {
		comp, err := resolveManagedComponent("proxy-server", &infernexv1alpha1.EnabledComponentSpec{Enabled: ptr.To(true)})
		if err != nil {
			return nil, err
		}
		tpl, err := buildComponentPodTemplate("proxy-server", comp)
		if err != nil {
			return nil, err
		}
		wlName, wlNS := pdWorkloadIdentity(infsvc)
		out["proxy-server"] = componentPlan{
			Replicas:            ptr.To(replicas(comp.Replicas)),
			Template:            tpl,
			ServicePort:         comp.ServicePort,
			AppKubernetesIOName: wlName,
			IsProxyServer:       true,
			ProxyPDWorkloadName: wlName,
			ProxyPDWorkloadNS:   wlNS,
		}
	}

	return out, nil
}

func buildEngineWorkloadPlan(component string, w *infernexv1alpha1.InferenceEngineWorkloadSpec) (componentPlan, error) {
	if w == nil {
		return componentPlan{}, fmt.Errorf("component %q: workload is nil", component)
	}
	if w.Template == nil {
		return componentPlan{}, fmt.Errorf("component %q: spec.template is required", component)
	}
	tpl, err := normalizeEnginePodTemplate(component, "template", w.Template)
	if err != nil {
		return componentPlan{}, err
	}
	var workerTpl *corev1.PodTemplateSpec
	if eff := WorkloadWorkerTemplateEffective(w); eff != nil {
		workerTpl, err = normalizeEnginePodTemplate(component, "worker", eff)
		if err != nil {
			return componentPlan{}, err
		}
	}
	groupSize, err := EngineWorkloadGroupSize(w)
	if err != nil {
		return componentPlan{}, fmt.Errorf("component %q: %w", component, err)
	}
	plan := componentPlan{
		Replicas:       w.Replicas,
		Template:       tpl,
		WorkerTemplate: workerTpl,
		ServicePort:    firstContainerServicePort(tpl, defaultWorkloadServicePort),
		WorkloadKind:   workloadKindDeployment,
		GroupSize:      groupSize,
	}
	if groupSize > 1 {
		plan.WorkloadKind = workloadKindLeaderWorkerSet
	}
	return plan, nil
}

func normalizeEnginePodTemplate(component, field string, src *corev1.PodTemplateSpec) (*corev1.PodTemplateSpec, error) {
	if src == nil {
		return nil, fmt.Errorf("component %q: spec.%s is required", component, field)
	}
	tpl := src.DeepCopy()
	if len(tpl.Spec.Containers) == 0 {
		return nil, fmt.Errorf("component %q: spec.%s.spec.containers must not be empty", component, field)
	}
	for i := range tpl.Spec.Containers {
		if strings.TrimSpace(tpl.Spec.Containers[i].Image) == "" {
			return nil, fmt.Errorf("component %q: spec.%s container %d must set image", component, field, i)
		}
		if strings.TrimSpace(tpl.Spec.Containers[i].Name) == "" {
			if len(tpl.Spec.Containers) == 1 {
				tpl.Spec.Containers[i].Name = component
			} else {
				tpl.Spec.Containers[i].Name = fmt.Sprintf("%s-%d", component, i)
			}
		}
	}
	return tpl, nil
}

func buildTemplateComponentPodTemplate(component string, c *infernexv1alpha1.TemplateComponentSpec) (*corev1.PodTemplateSpec, error) {
	if c == nil {
		return nil, fmt.Errorf("component %q: spec is nil", component)
	}
	if c.Template == nil {
		return nil, fmt.Errorf("component %q: spec.template is required when the component is enabled", component)
	}
	spec := &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(true), Template: c.Template}
	return buildComponentPodTemplate(component, spec)
}

func firstContainerServicePort(tpl *corev1.PodTemplateSpec, defaultPort int32) int32 {
	if tpl == nil || len(tpl.Spec.Containers) == 0 || len(tpl.Spec.Containers[0].Ports) == 0 {
		return defaultPort
	}
	if p := tpl.Spec.Containers[0].Ports[0].ContainerPort; p > 0 {
		return p
	}
	return defaultPort
}

// buildComponentPodTemplate validates and normalizes PodTemplateSpec from the component spec.
// Images must be set on template.spec.containers (same as Deployment).
func buildComponentPodTemplate(component string, c *infernexv1alpha1.ComponentSpec) (*corev1.PodTemplateSpec, error) {
	if c == nil {
		return nil, fmt.Errorf("component %q: spec is nil", component)
	}
	if c.Template == nil {
		return nil, fmt.Errorf("component %q: spec.template is required when the component is enabled", component)
	}
	t := c.Template.DeepCopy()
	if len(t.Spec.Containers) == 0 {
		return nil, fmt.Errorf("component %q: spec.template.spec.containers must not be empty", component)
	}
	for i := range t.Spec.Containers {
		if strings.TrimSpace(t.Spec.Containers[i].Image) == "" {
			return nil, fmt.Errorf("component %q: container %d must set image in spec.template.spec.containers", component, i)
		}
		if t.Spec.Containers[i].Name == "" {
			if len(t.Spec.Containers) == 1 {
				t.Spec.Containers[i].Name = component
			} else {
				t.Spec.Containers[i].Name = fmt.Sprintf("%s-%d", component, i)
			}
		}
	}
	return t, nil
}

func pdEngineWorkloadsPresent(engine *infernexv1alpha1.InferenceEngineSpec) bool {
	return EngineIsPDMode(engine)
}

func (r *InferNexServiceReconciler) shouldLaunchProxyServer(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
	spec infernexv1alpha1.InferNexServiceSpec,
) (bool, error) {
	if infsvc.Spec.SourceRef == nil {
		return pdEngineWorkloadsPresent(spec.Engine), nil
	}

	hasPD, err := r.linkedLLMHasPDMode(ctx, infsvc)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return hasPD, nil
}

func (r *InferNexServiceReconciler) linkedLLMHasPDMode(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
) (bool, error) {
	sourceNS := infsvc.Spec.SourceRef.Namespace
	if sourceNS == "" {
		sourceNS = infsvc.Namespace
	}
	sourceName := strings.TrimSpace(infsvc.Spec.SourceRef.Name)

	llm, err := r.getLinkedLLM(ctx, infsvc, sourceNS, sourceName)
	if err != nil {
		return false, err
	}
	hasPrefill, err := llmHasPrefillSpec(llm)
	if err != nil {
		return false, err
	}
	if hasPrefill {
		return true, nil
	}

	hasPrefill, err = r.llmBaseRefsResolvePrefill(ctx, sourceNS, llm)
	if err != nil {
		return false, err
	}
	if hasPrefill {
		return true, nil
	}

	return r.prefillDeploymentExists(ctx, sourceNS, sourceName)
}

// kserveStopAnnotationKey mirrors KServe constants.StopAnnotationKey
// (serving.kserve.io/stop). When set to "true" on an LLMInferenceService, KServe
// tears down its own serving workloads; we mirror that here so the InferNex-managed
// sidecar workloads are released too, instead of leaking until the LLMISVC is deleted.
const kserveStopAnnotationKey = "serving.kserve.io/stop"

// linkedLLMIsStopped reports whether this sourceRef-managed InferNexService should
// be treated as stopped by reading the KServe stop annotation off the source
// LLMInferenceService, mirroring utils.GetForceStopRuntime.
func (r *InferNexServiceReconciler) linkedLLMIsStopped(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
) (bool, error) {
	if infsvc.Spec.SourceRef == nil {
		return false, nil
	}
	sourceNS := infsvc.Spec.SourceRef.Namespace
	if sourceNS == "" {
		sourceNS = infsvc.Namespace
	}
	sourceName := strings.TrimSpace(infsvc.Spec.SourceRef.Name)
	llm, err := r.getLinkedLLM(ctx, infsvc, sourceNS, sourceName)
	if err != nil {
		// A missing source LLMISVC is handled by the delete path elsewhere; don't treat
		// it as stopped here.
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return annotationIsTrue(llm.GetAnnotations(), kserveStopAnnotationKey), nil
}

func annotationIsTrue(anns map[string]string, key string) bool {
	if anns == nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(anns[key]), "true")
}

// infernexDisabledComponentsAnnotationKey lets source LLMInferenceServices opt a
// managed component out per inference service without editing the namespace-wide
// default InferNexServiceConfig template. Value is a comma-separated list of
// component keys (e.g. "eagle-eye-hardware-monitor") or group aliases
// ("eagle-eye", "mooncake", "pd-orchestrator").
const infernexDisabledComponentsAnnotationKey = "infernex.io/disabled-components"

// componentGroupAliases expands a coarse group name to the concrete component keys.
var componentGroupAliases = map[string][]string{
	"eagle-eye": {
		"eagle-eye-hardware-monitor",
		"eagle-eye-hardware-diagnosis",
		"eagle-eye-network-performance-exporter",
	},
	"mooncake":        {"mooncake-master", "mooncake-metadata"},
	"pd-orchestrator": {"pd-orchestrator-elastic-scaler", "pd-orchestrator-tidal", "pd-orchestrator-rsg"},
}

// disabledComponents returns the set of component keys to suppress for this
// InferNexService. It reads infernex.io/disabled-components off the linked
// LLMInferenceService (sourceRef path) so the toggle propagates on annotation change.
func (r *InferNexServiceReconciler) disabledComponents(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
) (map[string]struct{}, error) {
	if infsvc.Spec.SourceRef == nil {
		return nil, nil
	}
	sourceNS := infsvc.Spec.SourceRef.Namespace
	if sourceNS == "" {
		sourceNS = infsvc.Namespace
	}
	llm, err := r.getLinkedLLM(ctx, infsvc, sourceNS, strings.TrimSpace(infsvc.Spec.SourceRef.Name))
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	raw := llm.GetAnnotations()[infernexDisabledComponentsAnnotationKey]
	return parseDisabledComponents(raw), nil
}

func parseDisabledComponents(raw string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, tok := range strings.Split(raw, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		if expanded, ok := componentGroupAliases[tok]; ok {
			for _, c := range expanded {
				out[c] = struct{}{}
			}
			continue
		}
		out[tok] = struct{}{}
	}
	return out
}

func (r *InferNexServiceReconciler) getLinkedLLM(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
	sourceNS, sourceName string,
) (*unstructured.Unstructured, error) {
	preferred := ""
	if infsvc.Spec.SourceRef != nil {
		preferred = infsvc.Spec.SourceRef.APIVersion
	}
	return r.getLLMInferenceService(ctx, types.NamespacedName{Namespace: sourceNS, Name: sourceName}, preferred)
}

func preferredLLMInferenceServiceAPIVersions(sourceRefAPIVersion string) []string {
	var versions []string
	appendIfMissing := func(version string) {
		version = strings.TrimSpace(version)
		if version == "" {
			return
		}
		for _, existing := range versions {
			if existing == version {
				return
			}
		}
		versions = append(versions, version)
	}
	appendIfMissing(sourceRefAPIVersion)
	appendIfMissing("serving.kserve.io/v1alpha2")
	appendIfMissing("serving.kserve.io/v1alpha1")
	return versions
}

func llmHasPrefillSpec(llm *unstructured.Unstructured) (bool, error) {
	_, hasPrefill, err := unstructured.NestedFieldNoCopy(llm.Object, "spec", "prefill")
	if err != nil {
		return false, err
	}
	return hasPrefill, nil
}

func (r *InferNexServiceReconciler) llmBaseRefsResolvePrefill(
	ctx context.Context,
	sourceNS string,
	llm *unstructured.Unstructured,
) (bool, error) {
	baseRefs, found, err := unstructured.NestedSlice(llm.Object, "spec", "baseRefs")
	if err != nil || !found {
		return false, err
	}
	for _, item := range baseRefs {
		ref, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := ref["name"].(string)
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		cfg, err := r.getLLMInferenceServiceConfig(ctx, sourceNS, name)
		if err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			return false, err
		}
		hasPrefill, err := llmHasPrefillSpec(cfg)
		if err != nil {
			return false, err
		}
		if hasPrefill {
			return true, nil
		}
	}
	return false, nil
}

func (r *InferNexServiceReconciler) getLLMInferenceServiceConfig(
	ctx context.Context,
	namespace, name string,
) (*unstructured.Unstructured, error) {
	for _, apiVersion := range llmInferenceServiceConfigAPIVersions() {
		cfg := &unstructured.Unstructured{}
		cfg.SetKind("LLMInferenceServiceConfig")
		cfg.SetAPIVersion(apiVersion)
		if err := r.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, cfg); err != nil {
			// Skip unserved API versions (RESTMapper miss) like NotFound so fallback can try the next version.
			if apierrors.IsNotFound(err) || meta.IsNoMatchError(err) {
				continue
			}
			return nil, err
		}
		return cfg, nil
	}
	return nil, apierrors.NewNotFound(infernexv1alpha1.GroupVersion.WithResource("llminferenceserviceconfigs").GroupResource(), name)
}

func (r *InferNexServiceReconciler) prefillDeploymentExists(
	ctx context.Context,
	namespace, workloadName string,
) (bool, error) {
	deployments := &appsv1.DeploymentList{}
	if err := r.List(ctx, deployments,
		client.InNamespace(namespace),
		client.MatchingLabels{
			labelAppKubernetesIOName:      workloadName,
			labelAppKubernetesIOPartOf:    valueKServeAppPartOf,
			labelAppKubernetesIOComponent: kserveWorkloadComponentPrefill,
		},
	); err != nil {
		return false, err
	}
	return len(deployments.Items) > 0, nil
}

// pdWorkloadIdentity returns the PD logical group id and namespace used for openfuyao.com/pdGroupID,
// infernex.io/pdEngineGroup (direct PD), and app.kubernetes.io/name on PD/proxy pods.
// Direct InferNexService: name = InferNexService.metadata.name (unique per namespaced CR).
// Linked mode (sourceRef): name = referenced LLMInferenceService name in its namespace.
func pdWorkloadIdentity(infsvc *infernexv1alpha1.InferNexService) (name, ns string) {
	if infsvc.Spec.SourceRef != nil {
		ns = strings.TrimSpace(infsvc.Spec.SourceRef.Namespace)
		if ns == "" {
			ns = infsvc.Namespace
		}
		name = strings.TrimSpace(infsvc.Spec.SourceRef.Name)
		if name == "" {
			name = infsvc.Name
		}
		return name, ns
	}
	return infsvc.Name, infsvc.Namespace
}

func enabled(v *bool) bool {
	return v == nil || *v
}

func componentWorkloadKind(plan componentPlan) string {
	if strings.TrimSpace(plan.WorkloadKind) == "" {
		return workloadKindDeployment
	}
	return plan.WorkloadKind
}

func replicas(v int32) int32 {
	if v > 0 {
		return v
	}
	return 1
}

func isInferenceEngineComponent(component string) bool {
	switch component {
	case "engine-aggregate", "engine-pd-prefill", "engine-pd-decode":
		return true
	default:
		return false
	}
}

// applyDeploymentReplicas sets dep.Spec.Replicas from plan. Inference engine workloads omit
// replicas in merged spec (nil) to allow external autoscalers; first create defaults to 1.
func applyDeploymentReplicas(dep *appsv1.Deployment, component string, plan componentPlan) {
	if plan.Replicas != nil {
		dep.Spec.Replicas = plan.Replicas
		return
	}
	if !isInferenceEngineComponent(component) {
		dep.Spec.Replicas = ptr.To(int32(1))
		return
	}
	if dep.UID == "" {
		dep.Spec.Replicas = ptr.To(int32(1))
	}
}

func (r *InferNexServiceReconciler) reconcileComponentWorkload(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
	plan componentPlan,
	effectiveEngine *infernexv1alpha1.InferenceEngineSpec,
) error {
	if plan.IsProxyServer && plan.ProxyPDWorkloadNS == owner.Namespace {
		if err := r.ensureProxyPodRBAC(ctx, owner); err != nil {
			return err
		}
	}
	if err := r.ensureComponentControllerRBAC(ctx, owner, component); err != nil {
		return err
	}
	switch componentWorkloadKind(plan) {
	case workloadKindDaemonSet:
		return r.reconcileComponentDaemonSet(ctx, owner, component, plan, effectiveEngine)
	case workloadKindLeaderWorkerSet:
		return r.reconcileComponentLeaderWorkerSet(ctx, owner, component, plan, effectiveEngine)
	default:
		return r.reconcileComponentDeployment(ctx, owner, component, plan, effectiveEngine)
	}
}

func (r *InferNexServiceReconciler) buildManagedComponentPodTemplate(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
	plan componentPlan,
	resourceLabels map[string]string,
	matchLabels map[string]string,
	effectiveEngine *infernexv1alpha1.InferenceEngineSpec,
) (*corev1.PodTemplateSpec, error) {
	tpl := plan.Template.DeepCopy()
	// Pod labels: union of resource labels (infernex.io/*, app.kubernetes.io/name) and selector match labels
	// (openfuyao.com/pdRole|pdGroupID for direct PD engines/proxy). PD merge fills remaining keys without
	// overwriting non-empty user labels (mergePodTemplateLabelIfAbsent).
	if tpl.Labels == nil {
		tpl.Labels = map[string]string{}
	}
	for k, v := range resourceLabels {
		tpl.Labels[k] = v
	}
	for k, v := range matchLabels {
		tpl.Labels[k] = v
	}
	// Managed-by is integration metadata; do not clobber a non-empty user value.
	mergePodTemplateLabelIfAbsent(tpl, labelInfernexManagedBy, valueInfernexBridgeManagedBy)
	if component == cacheIndexerComponent {
		cmName, cmErr := r.reconcileCacheIndexerConfigMap(ctx, owner, effectiveEngine)
		if cmErr != nil {
			return nil, cmErr
		}
		applyCacheIndexerPodTemplate(tpl, cmName)
	}
	if component == "engine-aggregate" {
		mergeAggregateInferenceWorkloadPodTemplateLabels(tpl, owner.Namespace, owner.Name)
	}
	if component == "engine-pd-prefill" || component == "engine-pd-decode" {
		wlGroup, _ := pdWorkloadIdentity(owner)
		prefill := component == "engine-pd-prefill"
		mergePDInferenceWorkloadPodTemplateLabels(tpl, owner.Namespace, wlGroup, prefill)
	}
	if plan.IsProxyServer {
		mergeProxyPDLabelsAndEnv(tpl, plan.ProxyPDWorkloadName, plan.ProxyPDWorkloadNS, 10, owner.Spec.SourceRef != nil)
		if strings.TrimSpace(tpl.Spec.ServiceAccountName) == "" && plan.ProxyPDWorkloadNS == owner.Namespace {
			tpl.Spec.ServiceAccountName = proxySANamespacedName(owner.Name)
		}
	}
	if !plan.IsProxyServer && strings.TrimSpace(tpl.Spec.ServiceAccountName) == "" {
		if component == hermesRouterComponent {
			tpl.Spec.ServiceAccountName = hermesRouterServiceAccountName(owner.Name)
		} else if saName := componentControllerSAName(component); saName != "" {
			tpl.Spec.ServiceAccountName = saName
		} else {
			// No dedicated service account for this component: keep the pod template default.
		}
	}
	if componentNeedsWebhookCert(component) {
		secretName, certErr := r.ensureComponentWebhookCertSecret(ctx, owner, component)
		if certErr != nil {
			return nil, certErr
		}
		ensureWebhookCertVolumeAndMount(tpl, secretName)
	}
	return tpl, nil
}

func (r *InferNexServiceReconciler) reconcileComponentDeployment(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
	plan componentPlan,
	effectiveEngine *infernexv1alpha1.InferenceEngineSpec,
) error {
	singleton := isSingletonComponentService(component)
	name := fmt.Sprintf("%s-%s", owner.Name, component)
	resourceLabels := deploymentResourceLabels(owner.Name, component, plan.AppKubernetesIOName)
	matchLabels := deploymentPodMatchLabels(owner, component, plan)
	if singleton {
		name = mooncakeSingletonWorkloadName(component)
		resourceLabels = mooncakeSingletonWorkloadLabels(component)
		matchLabels = mooncakeSingletonWorkloadSelector(component)
	}
	dep := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: owner.Namespace}}
	if singleton {
		if external, err := r.externalMooncakeServiceExists(ctx, owner, component); err != nil {
			return err
		} else if external {
			// An existing unowned Service is an external endpoint. Do not create
			// an internal Deployment behind it. If a Bridge-managed Deployment was
			// left over from before the external endpoint was installed, stop it so
			// its Pods cannot become accidental Service endpoints.
			if managed, exists, managedErr := r.isManagedMooncakeSingleton(ctx, dep); managedErr != nil {
				return managedErr
			} else if exists && managed {
				return r.scaleMooncakeDeployment(ctx, dep, 0)
			}
			return nil
		}
		managed, exists, err := r.isManagedMooncakeSingleton(ctx, dep)
		if err != nil {
			return err
		}
		if exists && !managed {
			// An externally managed canonical Deployment is a valid endpoint.
			return nil
		}
		if exists {
			// The singleton PodTemplate is namespace-scoped, not owned by this
			// InferNexService. Once it exists, preserve its image, command, args,
			// resources and volumes; a later service must not overwrite it or fail
			// solely because its per-service template differs.
			//
			// In particular, Deployment.spec.selector is immutable, so attempting
			// to "repair" it here would make upgrades fail. Canonical singleton
			// resources must be created with the fixed selector from the start.
			//
			// Replica count is not adjusted here. syncMooncakeSingletonReplicas
			// runs after all component workloads in reconcileWorkloadsAndRBAC
			// (and on deletion) and restores replicas from namespace activity.
			return nil
		}
	}
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, dep, func() error {
		// Mooncake master and metadata are namespace singletons. Their Deployment,
		// Pod labels, Service selector, and lifecycle must all use this same fixed
		// identity; they are intentionally not owned by an InferNexService.
		dep.Labels = resourceLabels
		if singleton {
			dep.Spec.Replicas = ptr.To(int32(1))
		} else {
			applyDeploymentReplicas(dep, component, plan)
		}
		dep.Spec.Selector = &metav1.LabelSelector{MatchLabels: matchLabels}
		tpl, tplErr := r.buildManagedComponentPodTemplate(ctx, owner, component, plan, resourceLabels, matchLabels, effectiveEngine)
		if tplErr != nil {
			return tplErr
		}
		dep.Spec.Template = *tpl
		if singleton {
			dep.OwnerReferences = nil
			return nil
		}
		return controllerutil.SetControllerReference(owner, dep, r.Scheme)
	})
	return err
}

func (r *InferNexServiceReconciler) externalMooncakeServiceExists(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
) (bool, error) {
	svc := &corev1.Service{}
	name := componentServiceNames(owner, component)[0]
	if err := r.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: name}, svc); err != nil {
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return !isBridgeManagedMooncakeService(svc), nil
}

// isManagedMooncakeSingleton reports whether the canonical Deployment belongs to
// Bridge. Existing chart/external Deployments are valid endpoints and must not be
// overwritten by the InferNexService controller.
func (r *InferNexServiceReconciler) isManagedMooncakeSingleton(ctx context.Context, dep *appsv1.Deployment) (managed, exists bool, err error) {
	if err := r.Get(ctx, client.ObjectKeyFromObject(dep), dep); err != nil {
		if apierrors.IsNotFound(err) {
			return true, false, nil
		}
		return false, false, err
	}
	managed = dep.Labels["infernex.io/singleton"] == "mooncake" ||
		dep.Labels["app.kubernetes.io/managed-by"] == "infernex-bridge"
	return managed, true, nil
}

func mooncakeSingletonWorkloadName(component string) string {
	switch component {
	case "mooncake-master":
		return "mooncake-master-deployment"
	case "mooncake-metadata":
		return "redis-server-deployment"
	default:
		return ""
	}
}

func mooncakeSingletonWorkloadLabels(component string) map[string]string {
	switch component {
	case "mooncake-master":
		return map[string]string{
			"app":                   "mooncake-master",
			"infernex.io/singleton": "mooncake",
		}
	case "mooncake-metadata":
		return map[string]string{
			"app":                   "redis",
			"infernex.io/singleton": "mooncake",
		}
	default:
		return nil
	}
}

func mooncakeSingletonWorkloadSelector(component string) map[string]string {
	switch component {
	case "mooncake-master":
		return map[string]string{"app": "mooncake-master"}
	case "mooncake-metadata":
		return map[string]string{"app": "redis"}
	default:
		return nil
	}
}

// syncMooncakeSingletonReplicas keeps the namespace singleton running while at
// least one InferNexService exists and scales it to zero after the last service
// is deleted. The Deployment and Service intentionally remain for stable DNS.
func (r *InferNexServiceReconciler) syncMooncakeSingletonReplicas(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	active := false
	var services infernexv1alpha1.InferNexServiceList
	if err := r.List(ctx, &services, client.InNamespace(owner.Namespace)); err != nil {
		return err
	}
	for i := range services.Items {
		if services.Items[i].DeletionTimestamp.IsZero() {
			active = true
			break
		}
	}
	// List then scale is the normal Kubernetes controller window, not a bug.
	// There is no cross-object transaction; a second List or generation check
	// cannot close it. InferNexService reconcile and the singleton watches
	// requeue the namespace and restore the correct replica count.
	replicas := int32(0)
	if active {
		replicas = 1
	}
	for _, component := range []string{"mooncake-master", "mooncake-metadata"} {
		external, err := r.externalMooncakeServiceExists(ctx, owner, component)
		if err != nil {
			return err
		}
		if external {
			// An external canonical Service is authoritative. Never scale a local
			// Deployment that happens to have the same canonical name.
			continue
		}
		dep := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{
			Namespace: owner.Namespace,
			Name:      mooncakeSingletonWorkloadName(component),
		}}
		managed, exists, err := r.isManagedMooncakeSingleton(ctx, dep)
		if err != nil {
			return err
		}
		if !exists || !managed {
			// No Bridge-managed singleton exists, or the canonical Deployment is
			// externally managed. In both cases scaling it would be a takeover.
			continue
		}
		if err := r.scaleMooncakeDeployment(ctx, dep, replicas); err != nil {
			return err
		}
	}
	return nil
}

func (r *InferNexServiceReconciler) scaleMooncakeDeployment(ctx context.Context, dep *appsv1.Deployment, replicas int32) error {
	if dep == nil || (dep.Spec.Replicas != nil && *dep.Spec.Replicas == replicas) {
		return nil
	}
	base := dep.DeepCopy()
	dep.Spec.Replicas = ptr.To(replicas)
	if err := r.Patch(ctx, dep, client.MergeFrom(base)); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

func (r *InferNexServiceReconciler) reconcileComponentDaemonSet(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
	plan componentPlan,
	effectiveEngine *infernexv1alpha1.InferenceEngineSpec,
) error {
	name := fmt.Sprintf("%s-%s", owner.Name, component)
	resourceLabels := deploymentResourceLabels(owner.Name, component, plan.AppKubernetesIOName)
	matchLabels := deploymentPodMatchLabels(owner, component, plan)
	ds := &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: owner.Namespace}}
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, ds, func() error {
		ds.Labels = resourceLabels
		ds.Spec.Selector = &metav1.LabelSelector{MatchLabels: matchLabels}
		tpl, tplErr := r.buildManagedComponentPodTemplate(ctx, owner, component, plan, resourceLabels, matchLabels, effectiveEngine)
		if tplErr != nil {
			return tplErr
		}
		ds.Spec.Template = *tpl
		return controllerutil.SetControllerReference(owner, ds, r.Scheme)
	})
	return err
}

// deploymentResourceLabels labels the Deployment (and pod template) with infernex.io/* for ownership/prune.
func deploymentResourceLabels(ownerName, component, appKubernetesIONameOverride string) map[string]string {
	appName := strings.TrimSpace(appKubernetesIONameOverride)
	if appName == "" {
		appName = "infernex-component"
	}
	return map[string]string{
		"app.kubernetes.io/name": appName,
		"infernex.io/owner":      ownerName,
		"infernex.io/component":  component,
	}
}

// deploymentPodMatchLabels is the Deployment/Service pod selector. For direct PD engines and proxy-server it
// matches Hermes filter-by-pd-label (openfuyao.com/pdRole + openfuyao.com/pdGroupID); other components use
// deploymentResourceLabels.
func deploymentPodMatchLabels(owner *infernexv1alpha1.InferNexService, component string, plan componentPlan) map[string]string {
	if owner == nil {
		return deploymentResourceLabels("", component, plan.AppKubernetesIOName)
	}
	wl, _ := pdWorkloadIdentity(owner)
	wl = strings.TrimSpace(wl)
	if wl == "" {
		wl = owner.Name
	}
	switch component {
	case "engine-pd-prefill":
		return map[string]string{
			labelOpenFuyaoPDRole:  openfuyaoPDRolePrefill,
			labelOpenFuyaoPDGroup: wl,
		}
	case "engine-pd-decode":
		return map[string]string{
			labelOpenFuyaoPDRole:  openfuyaoPDRoleDecode,
			labelOpenFuyaoPDGroup: wl,
		}
	case "proxy-server":
		if owner.Spec.SourceRef != nil {
			return deploymentResourceLabels(owner.Name, component, plan.AppKubernetesIOName)
		}
		g := strings.TrimSpace(plan.ProxyPDWorkloadName)
		if g == "" {
			g = wl
		}
		return map[string]string{
			labelOpenFuyaoPDRole:  openfuyaoPDRoleLeader,
			labelOpenFuyaoPDGroup: g,
		}
	default:
		return deploymentResourceLabels(owner.Name, component, plan.AppKubernetesIOName)
	}
}

func (r *InferNexServiceReconciler) pruneComponentDeployments(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	desired map[string]componentPlan,
) error {
	var deps appsv1.DeploymentList
	if err := r.List(
		ctx,
		&deps,
		client.InNamespace(owner.Namespace),
		client.MatchingLabels{"infernex.io/owner": owner.Name},
	); err != nil {
		return err
	}
	for i := range deps.Items {
		d := &deps.Items[i]
		component := d.Labels["infernex.io/component"]
		if isSingletonComponentService(component) {
			// Before singleton reconciliation, Mooncake workloads were named per
			// InferNexService. Remove those legacy owner-scoped Deployments; the
			// canonical singleton is not selected by this owner label.
			if err := r.deleteLegacyMooncakeDeployment(ctx, d, component); err != nil {
				return err
			}
			continue
		}
		if plan, ok := desired[component]; ok && componentWorkloadKind(plan) == workloadKindDeployment {
			continue
		}
		if err := r.Delete(ctx, d); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func (r *InferNexServiceReconciler) deleteLegacyMooncakeDeployment(ctx context.Context, d *appsv1.Deployment, component string) error {
	if d.Name == mooncakeSingletonWorkloadName(component) {
		return nil
	}
	err := r.Delete(ctx, d)
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

func (r *InferNexServiceReconciler) pruneComponentDaemonSets(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	desired map[string]componentPlan,
) error {
	var daemonSets appsv1.DaemonSetList
	if err := r.List(
		ctx,
		&daemonSets,
		client.InNamespace(owner.Namespace),
		client.MatchingLabels{"infernex.io/owner": owner.Name},
	); err != nil {
		return err
	}
	for i := range daemonSets.Items {
		ds := &daemonSets.Items[i]
		component := ds.Labels["infernex.io/component"]
		if plan, ok := desired[component]; ok && componentWorkloadKind(plan) == workloadKindDaemonSet {
			continue
		}
		if err := r.Delete(ctx, ds); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}
