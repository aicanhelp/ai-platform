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

// Package v1alpha1 provides webhooks for KServe v1alpha1 resources.
package v1alpha1

import (
	"context"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	runtimeLabelKey   = "infernex.io/runtime"
	runtimeLabelValue = "true"
	llmKind           = "LLMInferenceService"
	llmGroup          = "serving.kserve.io"

	configMutatedAnnotationKey = "infernex.io/llm-inference-service-config-mutated"
	mutatedAnnotationValue     = "infernex"
)

var supportedLLMVersions = map[string]struct{}{
	"v1alpha1": {},
	"v1alpha2": {},
}

// +kubebuilder:webhook:path=/mutate-serving-kserve-io-v1alpha1-llminferenceservice,mutating=true,failurePolicy=fail,sideEffects=NoneOnDryRun,groups=serving.kserve.io,resources=llminferenceservices,verbs=create;update,versions=v1alpha1;v1alpha2,name=mllminferenceservice.kb.io,admissionReviewVersions=v1
// +kubebuilder:rbac:groups=serving.kserve.io,resources=llminferenceserviceconfigs,verbs=get;list;watch;update;patch

// SetupLLMInferenceServiceWebhooksWithManager registers the mutating webhook on LLMInferenceService.
// KServe / llmisvc field validation is left to upstream; this webhook only patches well-known
// LLMInferenceServiceConfig objects when infernex.io/runtime=true.
func SetupLLMInferenceServiceWebhooksWithManager(mgr ctrl.Manager) {
	dec := admission.NewDecoder(mgr.GetScheme())
	server := mgr.GetWebhookServer()
	server.Register("/mutate-serving-kserve-io-v1alpha1-llminferenceservice", &admission.Webhook{
		Handler: &llmInferenceServiceMutatingHandler{
			client:  mgr.GetClient(),
			decoder: dec,
		},
	})
}

type llmInferenceServiceMutatingHandler struct {
	client  client.Client
	decoder admission.Decoder
}

func (h *llmInferenceServiceMutatingHandler) Handle(ctx context.Context, req admission.Request) admission.Response {
	llm := &unstructured.Unstructured{}
	if err := h.decoder.Decode(req, llm); err != nil {
		return admission.Errored(400, err)
	}
	if !isSupportedLLMInferenceServiceGVK(llm) {
		return admission.Allowed("skip mutation for unexpected GVK")
	}
	if !infernexRuntimeLabels(readObjectLabels(llm.Object)) {
		return admission.Allowed("skip mutation for non-infernex runtime")
	}

	ns := strings.TrimSpace(llm.GetNamespace())
	if err := h.mutateInstalledConfigs(ctx, ns); err != nil {
		return admission.Errored(500, err)
	}

	return admission.Allowed("mutated installed llmisvc configs for infernex runtime")
}

func isSupportedLLMInferenceServiceGVK(obj *unstructured.Unstructured) bool {
	if obj == nil || obj.GetKind() != llmKind {
		return false
	}
	gv, err := schema.ParseGroupVersion(strings.TrimSpace(obj.GetAPIVersion()))
	if err != nil || gv.Group != llmGroup {
		return false
	}
	_, ok := supportedLLMVersions[gv.Version]
	return ok
}

func readObjectLabels(obj map[string]interface{}) map[string]string {
	labels := map[string]string{}
	raw, found, err := unstructured.NestedMap(obj, "metadata", "labels")
	if err != nil || !found {
		return labels
	}
	for k, v := range raw {
		switch val := v.(type) {
		case string:
			labels[k] = val
		default:
			labels[k] = strings.TrimSpace(fmt.Sprint(val))
		}
	}
	return labels
}

func (h *llmInferenceServiceMutatingHandler) mutateInstalledConfigs(ctx context.Context, namespace string) error {
	candidateNS := make([]string, 0, 2)
	if namespace != "" {
		candidateNS = append(candidateNS, namespace)
	}
	if namespace != "kserve" {
		candidateNS = append(candidateNS, "kserve")
	}
	for _, ns := range candidateNS {
		if err := h.mutateKnownConfigsInNamespace(ctx, ns); err != nil {
			return err
		}
	}
	return nil
}

func (h *llmInferenceServiceMutatingHandler) mutateKnownConfigsInNamespace(
	ctx context.Context,
	namespace string,
) error {
	for _, name := range infernexMutateTargetLLMInferenceServiceConfigNames {
		if strings.TrimSpace(name) == "" {
			continue
		}
		if _, err := h.tryMutateConfigInNamespace(ctx, namespace, name); err != nil {
			return err
		}
	}
	return nil
}

func (h *llmInferenceServiceMutatingHandler) tryMutateConfigInNamespace(
	ctx context.Context,
	namespace,
	configName string,
) (bool, error) {
	var cfg *unstructured.Unstructured
	key := types.NamespacedName{Namespace: namespace, Name: configName}
	for _, apiVersion := range []string{"serving.kserve.io/v1alpha2", "serving.kserve.io/v1alpha1"} {
		candidate := &unstructured.Unstructured{}
		candidate.SetAPIVersion(apiVersion)
		candidate.SetKind("LLMInferenceServiceConfig")
		if err := h.client.Get(ctx, key, candidate); err != nil {
			if client.IgnoreNotFound(err) == nil || meta.IsNoMatchError(err) {
				continue
			}
			return false, err
		}
		cfg = candidate
		break
	}
	if cfg == nil {
		return false, nil
	}

	obj := cfg.Object
	annotations := getStringMap(obj, "metadata", "annotations")
	if annotations == nil {
		annotations = map[string]string{}
	}
	if strings.EqualFold(strings.TrimSpace(annotations[configMutatedAnnotationKey]), mutatedAnnotationValue) {
		return true, nil
	}

	applyInferNexPatchesToLLMInferenceServiceConfig(obj)

	annotations[configMutatedAnnotationKey] = mutatedAnnotationValue
	setStringMap(obj, annotations, "metadata", "annotations")
	cfg.Object = obj
	if err := h.client.Update(ctx, cfg); err != nil {
		return false, err
	}
	return true, nil
}

func infernexRuntimeLabels(labels map[string]string) bool {
	if labels == nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(labels[runtimeLabelKey]), runtimeLabelValue)
}

func getStringMap(obj map[string]interface{}, path ...string) map[string]string {
	cur := interface{}(obj)
	for _, p := range path {
		m, ok := cur.(map[string]interface{})
		if !ok {
			return nil
		}
		cur, ok = m[p]
		if !ok {
			return nil
		}
	}
	rawMap, ok := cur.(map[string]interface{})
	if !ok {
		return nil
	}
	out := make(map[string]string, len(rawMap))
	for k, v := range rawMap {
		if s, ok := v.(string); ok {
			out[k] = s
		}
	}
	return out
}

func setStringMap(obj map[string]interface{}, val map[string]string, path ...string) {
	if len(path) == 0 {
		return
	}
	cur := obj
	for i := 0; i < len(path)-1; i++ {
		next, ok := cur[path[i]]
		if !ok {
			n := map[string]interface{}{}
			cur[path[i]] = n
			cur = n
			continue
		}
		nextMap, ok := next.(map[string]interface{})
		if !ok {
			n := map[string]interface{}{}
			cur[path[i]] = n
			cur = n
			continue
		}
		cur = nextMap
	}
	last := path[len(path)-1]
	m := make(map[string]interface{}, len(val))
	for k, v := range val {
		m[k] = v
	}
	cur[last] = m
}

// applyInferNexPatchesToLLMInferenceServiceConfig patches well-known KServe LLMInferenceServiceConfig
// objects for InferNex/Hermes. Workload presets get initContainers cleared; capabilities.drop ALL is
// preserved so workloads stay compatible with restricted Pod Security Admission. NPU device access is
// granted via pod-level supplementalGroups (injected separately), not by stripping drop ALL.
// kserve-config-llm-scheduler is intentionally not mutated: Hermes LLMISVC templates overlay the
// upstream scheduler preset (including tokenizer probes) without Bridge sanitization.
func applyInferNexPatchesToLLMInferenceServiceConfig(obj map[string]interface{}) {
	// Main workload templates (decode / aggregate): clear llm-d init sidecars. drop ALL is kept.
	clearInitContainers(obj, "spec", "template")
}

func clearInitContainers(obj map[string]interface{}, path ...string) {
	p := append(path, "initContainers")
	_ = unstructured.SetNestedSlice(obj, []interface{}{}, p...)
}
