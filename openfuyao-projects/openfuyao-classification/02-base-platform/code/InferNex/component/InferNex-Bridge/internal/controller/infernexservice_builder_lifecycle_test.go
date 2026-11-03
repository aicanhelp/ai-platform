package controller

import (
	"context"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

type llmAwareClient struct {
	client.Client
	llms map[types.NamespacedName]*unstructured.Unstructured
}

func (c *llmAwareClient) Get(ctx context.Context, key types.NamespacedName, obj client.Object, opts ...client.GetOption) error {
	if u, ok := obj.(*unstructured.Unstructured); ok && u.GetKind() == "LLMInferenceService" {
		if found, ok := c.llms[key]; ok {
			u.SetAPIVersion(found.GetAPIVersion())
			u.SetKind(found.GetKind())
			u.SetNamespace(found.GetNamespace())
			u.SetName(found.GetName())
			u.SetLabels(found.GetLabels())
			u.SetAnnotations(found.GetAnnotations())
			u.Object["spec"] = found.Object["spec"]
			return nil
		}
		return apierrors.NewNotFound(schema.GroupResource{Group: "serving.kserve.io", Resource: "llminferenceservices"}, key.Name)
	}
	return c.Client.Get(ctx, key, obj, opts...)
}

func controllerTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	if err := infernexv1alpha1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	gvk := schema.GroupVersionKind{Group: "serving.kserve.io", Version: "v1alpha1", Kind: "LLMInferenceService"}
	s.AddKnownTypeWithName(gvk, &unstructured.Unstructured{})
	s.AddKnownTypeWithName(gvk.GroupVersion().WithKind("LLMInferenceServiceList"), &unstructured.UnstructuredList{})
	return s
}

func newLinkedLLM(ns, name, runtimeVal string) *unstructured.Unstructured {
	u := &unstructured.Unstructured{}
	u.SetAPIVersion("serving.kserve.io/v1alpha1")
	u.SetKind("LLMInferenceService")
	u.SetNamespace(ns)
	u.SetName(name)
	if runtimeVal != "" {
		u.SetLabels(map[string]string{infernexRuntimeLabel: runtimeVal})
	}
	return u
}

func newLinkedInferNexService(ns, name string) *infernexv1alpha1.InferNexService {
	return &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{
				APIVersion: "serving.kserve.io/v1alpha1",
				Kind:       "LLMInferenceService",
				Name:       name,
				Namespace:  ns,
			},
		},
	}
}

func getInferNexServiceOrNil(t *testing.T, c client.Client, ns, name string) *infernexv1alpha1.InferNexService {
	t.Helper()
	obj := &infernexv1alpha1.InferNexService{}
	err := c.Get(context.Background(), types.NamespacedName{Namespace: ns, Name: name}, obj)
	if err == nil {
		return obj
	}
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		t.Fatalf("get infsvc %s/%s: %v", ns, name, err)
	}
	return nil
}

func TestBuildInferNexServiceFromLLMInferenceService_CreatesFromRuntime(t *testing.T) {
	t.Parallel()
	s := controllerTestScheme(t)
	llm := newLinkedLLM("ns-a", "demo", infernexRuntimeValue)
	base := fake.NewClientBuilder().WithScheme(s).Build()
	cl := &llmAwareClient{
		Client: base,
		llms: map[types.NamespacedName]*unstructured.Unstructured{
			{Namespace: "ns-a", Name: "demo"}: llm,
		},
	}
	r := &InferNexServiceReconciler{Client: cl}

	nn := types.NamespacedName{Namespace: "ns-a", Name: "demo"}
	if err := r.buildInferNexServiceFromLLMInferenceService(context.Background(), nn); err != nil {
		t.Fatalf("buildInferNexServiceFromLLMInferenceService error: %v", err)
	}
	created := getInferNexServiceOrNil(t, cl, nn.Namespace, nn.Name)
	if created == nil {
		t.Fatal("expected InferNexService to be created")
	}
	if created.Spec.SourceRef == nil {
		t.Fatal("expected sourceRef populated")
	}
	if created.Spec.SourceRef.Kind != "LLMInferenceService" || created.Spec.SourceRef.Name != "demo" {
		t.Fatalf("unexpected sourceRef: %#v", created.Spec.SourceRef)
	}
}

func TestBuildInferNexServiceFromLLMInferenceService_DeletesWhenRuntimeMissing(t *testing.T) {
	t.Parallel()
	s := controllerTestScheme(t)
	llm := newLinkedLLM("ns-a", "demo", "")
	existing := newLinkedInferNexService("ns-a", "demo")
	base := fake.NewClientBuilder().WithScheme(s).WithObjects(existing).Build()
	cl := &llmAwareClient{
		Client: base,
		llms: map[types.NamespacedName]*unstructured.Unstructured{
			{Namespace: "ns-a", Name: "demo"}: llm,
		},
	}
	r := &InferNexServiceReconciler{Client: cl}

	if err := r.buildInferNexServiceFromLLMInferenceService(context.Background(), types.NamespacedName{
		Namespace: "ns-a",
		Name:      "demo",
	}); err != nil {
		t.Fatalf("buildInferNexServiceFromLLMInferenceService error: %v", err)
	}
	if got := getInferNexServiceOrNil(t, cl, "ns-a", "demo"); got != nil {
		t.Fatal("expected InferNexService to be deleted when runtime label missing")
	}
}

func TestSyncSourceRefLifecycle_DeletesWhenLinkedSourceMissingOrDisabled(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		llm     client.Object
		deleted bool
	}{
		{
			name:    "llm missing",
			llm:     nil,
			deleted: true,
		},
		{
			name:    "llm runtime disabled",
			llm:     newLinkedLLM("ns-a", "demo", "false"),
			deleted: true,
		},
		{
			name:    "llm runtime enabled",
			llm:     newLinkedLLM("ns-a", "demo", "true"),
			deleted: false,
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			s := controllerTestScheme(t)
			objs := []client.Object{newLinkedInferNexService("ns-a", "demo")}
			base := fake.NewClientBuilder().WithScheme(s).WithObjects(objs...).Build()
			lc := &llmAwareClient{Client: base, llms: map[types.NamespacedName]*unstructured.Unstructured{}}
			if tt.llm != nil {
				lc.llms[types.NamespacedName{Namespace: "ns-a", Name: "demo"}] = tt.llm.(*unstructured.Unstructured)
			}
			cl := lc
			r := &InferNexServiceReconciler{Client: cl}

			deleted, err := r.syncSourceRefLifecycle(context.Background(), newLinkedInferNexService("ns-a", "demo"))
			if err != nil {
				t.Fatalf("syncSourceRefLifecycle error: %v", err)
			}
			if deleted != tt.deleted {
				t.Fatalf("deleted=%v want %v", deleted, tt.deleted)
			}

			got := getInferNexServiceOrNil(t, cl, "ns-a", "demo")
			if tt.deleted && got != nil {
				t.Fatal("expected InferNexService deleted")
			}
			if !tt.deleted && got == nil {
				t.Fatal("expected InferNexService kept")
			}
		})
	}
}
