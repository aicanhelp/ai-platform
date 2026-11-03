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
	"context"
	"fmt"
	"os"
	"strings"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
	"gitcode.com/openFuyao/InferNex/internal/controller"
)

const llmInferenceServiceKind = "LLMInferenceService"

// +kubebuilder:webhook:path=/validate-infernex-infernex-io-v1alpha1-infernexservice,mutating=false,failurePolicy=fail,sideEffects=None,groups=infernex.infernex.io,resources=infernexservices,verbs=create;update,versions=v1alpha1,name=vinfernexservice-v1alpha1.kb.io,admissionReviewVersions=v1

func SetupInferNexServiceValidationWebhookWithManager(mgr ctrl.Manager) {
	dec := admission.NewDecoder(mgr.GetScheme())
	server := mgr.GetWebhookServer()
	tmplNS := infernexTemplateNamespaceForWebhook()
	server.Register("/validate-infernex-infernex-io-v1alpha1-infernexservice", &admission.Webhook{
		Handler: &inferNexServiceValidatingHandler{
			decoder:           dec,
			apiClient:         mgr.GetClient(),
			templateNamespace: tmplNS,
		},
	})
}

func infernexTemplateNamespaceForWebhook() string {
	// Keep in sync with cmd/main.go controllerTemplateNamespace().
	if ns := strings.TrimSpace(os.Getenv("INFERNEX_TEMPLATE_NAMESPACE")); ns != "" {
		return ns
	}
	return strings.TrimSpace(os.Getenv("POD_NAMESPACE"))
}

type inferNexServiceValidatingHandler struct {
	decoder           admission.Decoder
	apiClient         client.Client
	templateNamespace string
}

func (h *inferNexServiceValidatingHandler) Handle(ctx context.Context, req admission.Request) admission.Response {
	insvc := &infernexv1alpha1.InferNexService{}
	if err := h.decoder.Decode(req, insvc); err != nil {
		return admission.Errored(400, err)
	}
	if isLLMISourceMode(insvc) {
		return admission.Allowed("sourceRef mode: skip infernexservice validation for LLMInferenceService-linked object")
	}
	if err := validateRawRouterExplicitEnabledIfPresent(insvc); err != nil {
		return admission.Denied(err.Error())
	}
	if err := validateDirectInferNexServiceExplicitComponents(insvc); err != nil {
		return admission.Denied(err.Error())
	}
	// Updates that only drop the controller finalizer (or other metadata) still run through
	// validation. Merged-spec checks load InferNexServiceConfig from baseRefs; if templates were
	// deleted first, those lookups fail and the InferNexService stays Terminating forever.
	if !insvc.DeletionTimestamp.IsZero() {
		return admission.Allowed("terminating InferNexService: skip effective-spec merge validation")
	}
	if req.Operation == admissionv1.Update && len(req.OldObject.Raw) > 0 {
		old := &infernexv1alpha1.InferNexService{}
		if err := h.decoder.DecodeRaw(req.OldObject, old); err == nil && !old.DeletionTimestamp.IsZero() {
			return admission.Allowed("terminating InferNexService update: skip effective-spec merge validation")
		}
	}
	effective, _, err := controller.BuildEffectiveSpecForWebhook(ctx, h.apiClient, h.templateNamespace, insvc)
	if err != nil {
		return admission.Denied(fmt.Sprintf("effective spec (merge baseRefs / defaults): %v", err))
	}
	if err := validateGatewayRefSpec(effective.IntelligentGatewayRouting); err != nil {
		return admission.Denied(err.Error())
	}
	if err := validateMergedDirectInferNexService(effective); err != nil {
		return admission.Denied(err.Error())
	}
	return admission.Allowed("validated direct infernexservice spec")
}

func validateRawRouterExplicitEnabledIfPresent(insvc *infernexv1alpha1.InferNexService) error {
	if igr := insvc.Spec.IntelligentGatewayRouting; igr != nil && igr.Router != nil {
		return requireExplicitEnabled("spec.intelligentGatewayRouting.router", igr.Router.Enabled)
	}
	return nil
}

func isLLMISourceMode(insvc *infernexv1alpha1.InferNexService) bool {
	return insvc != nil &&
		insvc.Spec.SourceRef != nil &&
		strings.EqualFold(strings.TrimSpace(insvc.Spec.SourceRef.Kind), llmInferenceServiceKind)
}

// validateMergedDirectInferNexService validates engine and router on the reconcile-time effective spec
// (after baseRefs and platform template merge), so engine may come only from InferNexServiceConfig.
func validateMergedDirectInferNexService(effective infernexv1alpha1.InferNexServiceSpec) error {
	if effective.Engine == nil {
		return fmt.Errorf("merged spec must define spec.engine (set on InferNexService or supply it via baseRefs / InferNexServiceConfig)")
	}
	if effective.Engine.Template == nil && !controller.EnginePrefillEffective(effective.Engine) {
		return fmt.Errorf("merged spec.engine must set template (aggregate/decode) or prefill.template (PD)")
	}
	if err := validateMergedEngineTemplates(effective.Engine); err != nil {
		return err
	}
	return validateMergedRouterTemplate(effective.IntelligentGatewayRouting)
}

func validateMergedEngineTemplates(engine *infernexv1alpha1.InferenceEngineSpec) error {
	if engine == nil {
		return nil
	}
	if !controller.EngineIsPDMode(engine) {
		if err := validateWorkloadTemplate("spec.engine", engine.Template); err != nil {
			return err
		}
		root := controller.EngineRootWorkload(engine)
		if err := controller.ValidateEngineWorkloadDataParallel("spec.engine", root); err != nil {
			return err
		}
		return validateEngineWorkloadWorker("spec.engine", root)
	}
	if err := validateWorkloadTemplate("spec.engine", engine.Template); err != nil {
		return err
	}
	decode := controller.EngineDecodeWorkload(engine)
	if err := controller.ValidateEngineWorkloadDataParallel("spec.engine", decode); err != nil {
		return err
	}
	if err := validateEngineWorkloadWorker("spec.engine", decode); err != nil {
		return err
	}
	prefill := controller.EnginePrefillWorkload(engine)
	if prefill == nil {
		return fmt.Errorf("merged spec.engine.prefill must be provided in PD mode")
	}
	if err := validateWorkloadTemplate("spec.engine.prefill", prefill.Template); err != nil {
		return err
	}
	if err := controller.ValidateEngineWorkloadDataParallel("spec.engine.prefill", prefill); err != nil {
		return err
	}
	return validateEngineWorkloadWorker("spec.engine.prefill", prefill)
}

func validateMergedRouterTemplate(igr *infernexv1alpha1.IntelligentGatewayRoutingSpec) error {
	if igr == nil || igr.Router == nil {
		return nil
	}
	if err := requireExplicitEnabled("spec.intelligentGatewayRouting.router", igr.Router.Enabled); err != nil {
		return err
	}
	if !isComponentEnabled(igr.Router.Enabled) {
		return nil
	}
	path := "spec.intelligentGatewayRouting.router.template"
	if err := validatePodTemplate(path, igr.Router.Template); err != nil {
		return err
	}
	return validateHermesRouterEPPContainer(path, igr.Router.Template)
}

const hermesRouterEPPContainerName = "main"

// validateHermesRouterEPPContainer requires the gateway-api-inference-extension (Hermes)
// process container to be named "main" with a named port "grpc", matching controller Service
// targetPort resolution. Sidecars (tokenizer, prediction, …) may appear in any order.
func validateHermesRouterEPPContainer(path string, tpl *corev1.PodTemplateSpec) error {
	if tpl == nil {
		return nil
	}
	var epp *corev1.Container
	for i := range tpl.Spec.Containers {
		if tpl.Spec.Containers[i].Name == hermesRouterEPPContainerName {
			epp = &tpl.Spec.Containers[i]
			break
		}
	}
	if epp == nil {
		return fmt.Errorf(
			"%s.spec.containers must include a container named %q (Hermes EPP); sidecars may use any other name and order",
			path, hermesRouterEPPContainerName,
		)
	}
	for _, p := range epp.Ports {
		if p.Name == "grpc" && p.ContainerPort > 0 {
			return nil
		}
	}
	return fmt.Errorf(
		"%s.spec.containers (name=%q).ports must declare named port %q with containerPort > 0",
		path, hermesRouterEPPContainerName, "grpc",
	)
}

// validateDirectInferNexServiceExplicitComponents requires every declared optional component block
// to set enabled explicitly (true/false), so admission matches controller semantics (nil enabled
// would otherwise mean "on") and avoids accidental sidecars.
func validateDirectInferNexServiceExplicitComponents(insvc *infernexv1alpha1.InferNexService) error {
	c := insvc.Spec.Components
	if c == nil {
		return nil
	}
	return validateDeclaredComponentsExplicitEnabled(c)
}

func validateDeclaredComponentsExplicitEnabled(c *infernexv1alpha1.InfernexComponentsSpec) error {
	if c == nil {
		return nil
	}
	checks := []struct {
		path string
		spec *infernexv1alpha1.ComponentSpec
	}{
		{path: "spec.components.cacheIndexer", spec: componentSpecFromCacheIndexer(c.CacheIndexer)},
		{path: "spec.components.pdOrchestrator.elasticScaler", spec: enabledComponentSpecOrNil(c.PDOrchestrator, func(p *infernexv1alpha1.PDOrchestratorComponentSpec) *infernexv1alpha1.EnabledComponentSpec {
			return p.ElasticScaler
		})},
		{path: "spec.components.pdOrchestrator.tidal", spec: enabledComponentSpecOrNil(c.PDOrchestrator, func(p *infernexv1alpha1.PDOrchestratorComponentSpec) *infernexv1alpha1.EnabledComponentSpec {
			return p.Tidal
		})},
		{path: "spec.components.pdOrchestrator.resourceScalingGroup", spec: enabledComponentSpecOrNil(c.PDOrchestrator, func(p *infernexv1alpha1.PDOrchestratorComponentSpec) *infernexv1alpha1.EnabledComponentSpec {
			return p.ResourceScalingGroup
		})},
		{path: "spec.components.eagleEye.hardwareMonitor", spec: enabledComponentSpecOrNil(c.EagleEye, func(e *infernexv1alpha1.EagleEyeComponentSpec) *infernexv1alpha1.EnabledComponentSpec {
			return e.HardwareMonitor
		})},
		{path: "spec.components.eagleEye.hardwareDiagnosis", spec: enabledComponentSpecOrNil(c.EagleEye, func(e *infernexv1alpha1.EagleEyeComponentSpec) *infernexv1alpha1.EnabledComponentSpec {
			return e.HardwareDiagnosis
		})},
		{path: "spec.components.eagleEye.networkPerformanceExporter", spec: enabledComponentSpecOrNil(c.EagleEye, func(e *infernexv1alpha1.EagleEyeComponentSpec) *infernexv1alpha1.EnabledComponentSpec {
			return e.NetworkPerformanceExporter
		})},
	}
	if err := requireEnabledFromComponentSpec("spec.components.mooncake", mooncakeRootComponent(c.Mooncake)); err != nil {
		return err
	}
	for _, check := range checks {
		if err := requireEnabledFromComponentSpec(check.path, check.spec); err != nil {
			return err
		}
	}
	return nil
}

func componentSpecFromCacheIndexer(spec *infernexv1alpha1.CacheIndexerComponentSpec) *infernexv1alpha1.ComponentSpec {
	if spec == nil {
		return nil
	}
	return &infernexv1alpha1.ComponentSpec{Enabled: spec.Enabled}
}

func mooncakeRootComponent(spec *infernexv1alpha1.MooncakeComponentSpec) *infernexv1alpha1.ComponentSpec {
	if spec == nil {
		return nil
	}
	return &infernexv1alpha1.ComponentSpec{Enabled: spec.Enabled}
}

func componentSpecOrNil[T any](spec *T, getter func(*T) *infernexv1alpha1.ComponentSpec) *infernexv1alpha1.ComponentSpec {
	if spec == nil {
		return nil
	}
	return getter(spec)
}

func enabledComponentSpecOrNil[T any](spec *T, getter func(*T) *infernexv1alpha1.EnabledComponentSpec) *infernexv1alpha1.ComponentSpec {
	if spec == nil {
		return nil
	}
	enabledSpec := getter(spec)
	if enabledSpec == nil {
		return nil
	}
	return &infernexv1alpha1.ComponentSpec{Enabled: enabledSpec.Enabled}
}

func requireEnabledFromComponentSpec(path string, spec *infernexv1alpha1.ComponentSpec) error {
	if spec == nil {
		return nil
	}
	return requireExplicitEnabled(path, spec.Enabled)
}

func requireExplicitEnabled(path string, enabled *bool) error {
	if enabled == nil {
		return fmt.Errorf("%s.enabled is required for direct InferNexService (set true or false explicitly)", path)
	}
	return nil
}

func validateGatewayRefSpec(igr *infernexv1alpha1.IntelligentGatewayRoutingSpec) error {
	if igr == nil {
		return nil
	}
	if igr.Router == nil || !isComponentEnabled(igr.Router.Enabled) {
		return nil
	}
	if err := validateHTTPRouteRefsAndSpecExclusivity("spec.intelligentGatewayRouting.httpRoute", igr.HTTPRoute); err != nil {
		return err
	}
	if err := validateGatewayRefsAndSpecExclusivity("spec.intelligentGatewayRouting.gateway", igr.Gateway); err != nil {
		return err
	}
	if err := validateInferencePoolRefsAndSpecExclusivity("spec.intelligentGatewayRouting.inferencePool", igr.InferencePool); err != nil {
		return err
	}
	return nil
}

func validateHTTPRouteRefsAndSpecExclusivity(path string, cfg *infernexv1alpha1.HTTPRouteRefSpec) error {
	if cfg == nil {
		return nil
	}
	if cfg.Ref != nil && cfg.Spec != nil {
		return fmt.Errorf("%s.ref and %s.spec are mutually exclusive", path, path)
	}
	return nil
}

func validateGatewayRefsAndSpecExclusivity(path string, cfg *infernexv1alpha1.GatewayRefSpec) error {
	if cfg == nil {
		return nil
	}
	if cfg.Ref != nil && cfg.Spec != nil {
		return fmt.Errorf("%s.ref and %s.spec are mutually exclusive", path, path)
	}
	return nil
}

func validateInferencePoolRefsAndSpecExclusivity(path string, cfg *infernexv1alpha1.InferencePoolRefSpec) error {
	if cfg == nil {
		return nil
	}
	if cfg.Ref != nil && cfg.Spec != nil {
		return fmt.Errorf("%s.ref and %s.spec are mutually exclusive", path, path)
	}
	return nil
}

func isComponentEnabled(enabled *bool) bool {
	return enabled == nil || *enabled
}

func validateEngineWorkloadWorker(path string, w *infernexv1alpha1.InferenceEngineWorkloadSpec) error {
	if w == nil {
		return nil
	}
	groupSize, err := controller.EngineWorkloadGroupSize(w)
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	worker := controller.WorkloadWorkerTemplateEffective(w)
	if groupSize == 1 && worker != nil {
		return fmt.Errorf("%s: worker must not be set when groupSize is 1 (Deployment path)", path)
	}
	if groupSize > 1 && worker != nil {
		return validatePodTemplate(path+".worker", worker)
	}
	return nil
}

func validateWorkloadTemplate(path string, tpl *corev1.PodTemplateSpec) error {
	return validatePodTemplate(path+".template", tpl)
}

func validatePodTemplate(path string, tpl *corev1.PodTemplateSpec) error {
	if tpl == nil {
		return fmt.Errorf("%s is required", path)
	}
	if len(tpl.Spec.Containers) == 0 {
		return fmt.Errorf("%s.spec.containers must not be empty", path)
	}
	for i := range tpl.Spec.Containers {
		if strings.TrimSpace(tpl.Spec.Containers[i].Image) == "" {
			return fmt.Errorf("%s.spec.containers[%d].image is required", path, i)
		}
	}
	return nil
}
