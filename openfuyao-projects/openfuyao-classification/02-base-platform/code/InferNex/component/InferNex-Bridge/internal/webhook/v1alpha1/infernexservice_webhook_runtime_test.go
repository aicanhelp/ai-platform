package v1alpha1

import (
	"context"
	"encoding/json"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func webhookRuntimeTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	return runtime.NewScheme()
}

func newLLMISVCConfig(ns, name string) *unstructured.Unstructured {
	cfg := &unstructured.Unstructured{}
	cfg.SetAPIVersion("serving.kserve.io/v1alpha1")
	cfg.SetKind("LLMInferenceServiceConfig")
	cfg.SetName(name)
	cfg.SetNamespace(ns)
	cfg.Object["spec"] = map[string]interface{}{
		"template": map[string]interface{}{
			"containers": []interface{}{
				map[string]interface{}{
					"name":  "main",
					"image": "router:v1",
					"securityContext": map[string]interface{}{
						"capabilities": map[string]interface{}{
							"drop": []interface{}{"ALL"},
						},
					},
				},
			},
			"initContainers": []interface{}{
				map[string]interface{}{"name": "sidecar", "image": "sidecar:v1"},
			},
		},
	}
	return cfg
}

func fetchConfig(t *testing.T, c client.Client, ns, name string) *unstructured.Unstructured {
	t.Helper()
	cfg := &unstructured.Unstructured{}
	cfg.SetAPIVersion("serving.kserve.io/v1alpha1")
	cfg.SetKind("LLMInferenceServiceConfig")
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: ns, Name: name}, cfg); err != nil {
		t.Fatalf("get %s/%s: %v", ns, name, err)
	}
	return cfg
}

func TestInfernexRuntimeLabels(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		labels map[string]string
		want   bool
	}{
		{name: "nil labels", labels: nil, want: false},
		{name: "missing key", labels: map[string]string{"foo": "bar"}, want: false},
		{name: "false", labels: map[string]string{runtimeLabelKey: "false"}, want: false},
		{name: "true", labels: map[string]string{runtimeLabelKey: "true"}, want: true},
		{name: "case insensitive", labels: map[string]string{runtimeLabelKey: "True"}, want: true},
		{name: "trim spaces", labels: map[string]string{runtimeLabelKey: " true "}, want: true},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := infernexRuntimeLabels(tt.labels)
			if got != tt.want {
				t.Fatalf("infernexRuntimeLabels()=%v want %v", got, tt.want)
			}
		})
	}
}

func TestTryMutateConfigInNamespace_IdempotentAnnotatedConfig(t *testing.T) {
	t.Parallel()
	s := webhookRuntimeTestScheme(t)
	cfg := newLLMISVCConfig("kserve", infernexMutateTargetLLMInferenceServiceConfigNames[0])
	unstructured.SetNestedField(cfg.Object, mutatedAnnotationValue, "metadata", "annotations", configMutatedAnnotationKey)

	cl := fake.NewClientBuilder().WithScheme(s).WithRuntimeObjects(cfg).Build()
	h := &llmInferenceServiceMutatingHandler{client: cl}

	mutated, err := h.tryMutateConfigInNamespace(context.Background(), "kserve", cfg.GetName())
	if err != nil {
		t.Fatalf("tryMutateConfigInNamespace error: %v", err)
	}
	if !mutated {
		t.Fatal("expected annotated config to be treated as mutated")
	}

	after := fetchConfig(t, cl, "kserve", cfg.GetName())
	initContainers, found, err := unstructured.NestedSlice(after.Object, "spec", "template", "initContainers")
	if err != nil {
		t.Fatalf("read initContainers: %v", err)
	}
	if !found || len(initContainers) != 1 {
		t.Fatalf("expected existing initContainers untouched, found=%v len=%d", found, len(initContainers))
	}
}

func TestMutateInstalledConfigs_MutatesNamespaceAndKServe(t *testing.T) {
	t.Parallel()
	s := webhookRuntimeTestScheme(t)
	targetName := infernexMutateTargetLLMInferenceServiceConfigNames[0]
	cfgUser := newLLMISVCConfig("user-ns", targetName)
	cfgKServe := newLLMISVCConfig("kserve", targetName)

	cl := fake.NewClientBuilder().WithScheme(s).WithRuntimeObjects(cfgUser, cfgKServe).Build()
	h := &llmInferenceServiceMutatingHandler{client: cl}
	if err := h.mutateInstalledConfigs(context.Background(), "user-ns"); err != nil {
		t.Fatalf("mutateInstalledConfigs error: %v", err)
	}

	for _, ns := range []string{"user-ns", "kserve"} {
		after := fetchConfig(t, cl, ns, targetName)
		ann, _, _ := unstructured.NestedString(after.Object, "metadata", "annotations", configMutatedAnnotationKey)
		if ann != mutatedAnnotationValue {
			t.Fatalf("%s/%s annotation=%q want %q", ns, targetName, ann, mutatedAnnotationValue)
		}
		inits, _, _ := unstructured.NestedSlice(after.Object, "spec", "template", "initContainers")
		if len(inits) != 0 {
			t.Fatalf("%s/%s expected initContainers cleared", ns, targetName)
		}
	}
}

func TestLLMInferenceServiceMutatingHandler_Handle_SkipWhenNoRuntimeLabel(t *testing.T) {
	t.Parallel()
	s := webhookRuntimeTestScheme(t)
	targetName := infernexMutateTargetLLMInferenceServiceConfigNames[0]
	cfg := newLLMISVCConfig("user-ns", targetName)

	cl := fake.NewClientBuilder().WithScheme(s).WithRuntimeObjects(cfg).Build()
	h := &llmInferenceServiceMutatingHandler{
		client:  cl,
		decoder: admission.NewDecoder(s),
	}

	llm := &unstructured.Unstructured{}
	llm.SetAPIVersion("serving.kserve.io/v1alpha1")
	llm.SetKind("LLMInferenceService")
	llm.SetName("demo")
	llm.SetNamespace("user-ns")
	llm.SetLabels(map[string]string{})
	raw, err := json.Marshal(llm)
	if err != nil {
		t.Fatal(err)
	}
	resp := h.Handle(context.Background(), admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
		Object: runtime.RawExtension{Raw: raw},
	}})
	if !resp.Allowed {
		t.Fatalf("expected Allowed, got %v", resp.Result)
	}

	after := fetchConfig(t, cl, "user-ns", targetName)
	ann, found, _ := unstructured.NestedString(after.Object, "metadata", "annotations", configMutatedAnnotationKey)
	if found || ann != "" {
		t.Fatalf("unexpected mutation annotation when runtime label missing: found=%v ann=%q", found, ann)
	}
}

func TestLLMInferenceServiceMutatingHandler_Handle_SkipWhenUnexpectedGVK(t *testing.T) {
	t.Parallel()
	s := webhookRuntimeTestScheme(t)
	targetName := infernexMutateTargetLLMInferenceServiceConfigNames[0]
	cfg := newLLMISVCConfig("user-ns", targetName)

	cl := fake.NewClientBuilder().WithScheme(s).WithRuntimeObjects(cfg).Build()
	h := &llmInferenceServiceMutatingHandler{
		client:  cl,
		decoder: admission.NewDecoder(s),
	}

	llm := &unstructured.Unstructured{}
	llm.SetAPIVersion("serving.kserve.io/v1beta1")
	llm.SetKind("LLMInferenceService")
	llm.SetName("demo")
	llm.SetNamespace("user-ns")
	llm.SetLabels(map[string]string{runtimeLabelKey: runtimeLabelValue})

	raw, err := json.Marshal(llm)
	if err != nil {
		t.Fatal(err)
	}
	resp := h.Handle(context.Background(), admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
		Object: runtime.RawExtension{Raw: raw},
	}})
	if !resp.Allowed {
		t.Fatalf("expected Allowed for unexpected GVK skip, got %v", resp.Result)
	}

	after := fetchConfig(t, cl, "user-ns", targetName)
	ann, found, _ := unstructured.NestedString(after.Object, "metadata", "annotations", configMutatedAnnotationKey)
	if found || ann != "" {
		t.Fatalf("unexpected mutation annotation when GVK is unsupported: found=%v ann=%q", found, ann)
	}
}

