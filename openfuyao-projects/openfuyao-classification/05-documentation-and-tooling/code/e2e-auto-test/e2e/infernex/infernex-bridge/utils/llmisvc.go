package utils

import (
	"context"
	"fmt"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

var (
	gvrLLMInferenceService = schema.GroupVersionResource{
		Group: "serving.kserve.io", Version: "v1alpha1", Resource: "llminferenceservices",
	}
	gvrLLMInferenceServiceConfig = schema.GroupVersionResource{
		Group: "serving.kserve.io", Version: "v1alpha1", Resource: "llminferenceserviceconfigs",
	}
)

func LLMInferenceServiceGVR() schema.GroupVersionResource { return gvrLLMInferenceService }

func KServeConfigNames() []string {
	return []string{
		"kserve-config-llm-template",
		"kserve-config-llm-worker-data-parallel",
		"kserve-config-llm-decode-template",
		"kserve-config-llm-decode-worker-data-parallel",
		"kserve-config-llm-prefill-template",
		"kserve-config-llm-prefill-worker-data-parallel",
	}
}

func GetLLMInferenceService(ctx context.Context, dyn dynamic.Interface, ns, name string) (*unstructured.Unstructured, error) {
	return dyn.Resource(gvrLLMInferenceService).Namespace(ns).Get(ctx, name, metav1.GetOptions{})
}

func DeleteLLMInferenceService(ctx context.Context, dyn dynamic.Interface, ns, name string) error {
	return dyn.Resource(gvrLLMInferenceService).Namespace(ns).Delete(ctx, name, metav1.DeleteOptions{})
}

func LLMInferenceServiceExists(ctx context.Context, dyn dynamic.Interface, ns, name string) (bool, error) {
	_, err := GetLLMInferenceService(ctx, dyn, ns, name)
	if err != nil {
		if isNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func DeleteLLMInferenceServiceConfig(ctx context.Context, dyn dynamic.Interface, ns, name string) error {
	return dyn.Resource(gvrLLMInferenceServiceConfig).Namespace(ns).Delete(ctx, name, metav1.DeleteOptions{})
}

func GetLLMInferenceServiceConfig(ctx context.Context, dyn dynamic.Interface, ns, name string) (*unstructured.Unstructured, error) {
	return dyn.Resource(gvrLLMInferenceServiceConfig).Namespace(ns).Get(ctx, name, metav1.GetOptions{})
}

func ConfigMutatedAnnotation(obj *unstructured.Unstructured) string {
	v, _, _ := unstructured.NestedString(obj.Object, "metadata", "annotations", LLMConfigMutatedAnnotation)
	return strings.TrimSpace(v)
}

func PatchLLMInferenceServiceLabels(ns, name, patchJSON string) (string, string, error) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl patch llminferenceservice %q -n %q --type=merge -p %q",
		kc, name, ns, patchJSON)
	return ExecLocal(cmd)
}

func CRDExists(fitID, group, kind string) bool {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl api-resources | grep -E '%s.*%s'", kc, kind, group)
	out, _, err := ExecLocal(cmd)
	if err != nil {
		LogStep(fitID, "crd check %s/%s failed: %v", group, kind, err)
		return false
	}
	return strings.Contains(out, kind) && strings.Contains(out, group)
}
