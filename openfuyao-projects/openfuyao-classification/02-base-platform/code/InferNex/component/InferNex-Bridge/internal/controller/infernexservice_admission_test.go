package controller

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestBuildEffectiveSpecForWebhook_UsesTemplateNamespaceFallback(t *testing.T) {
	t.Parallel()
	s := gatewayTestScheme(t)
	cfg := &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: defaultAggregateTemplateName, Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceConfigSpec{
			InferNexServiceSpec: infernexv1alpha1.InferNexServiceSpec{
				Components: &infernexv1alpha1.InfernexComponentsSpec{
					CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{
						Enabled:  ptr.To(true),
						Replicas: 2,
					},
				},
				Engine: &infernexv1alpha1.InferenceEngineSpec{
					InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
						Template: &corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{{Name: "main", Image: "engine:v1"}},
							},
						},
					},
				},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(cfg).Build()
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	effective, mode, err := BuildEffectiveSpecForWebhook(context.Background(), cl, "", insvc)
	if err != nil {
		t.Fatalf("BuildEffectiveSpecForWebhook error: %v", err)
	}
	if mode != "aggregate" {
		t.Fatalf("expected aggregate mode, got %q", mode)
	}
	if effective.Components == nil || effective.Components.CacheIndexer == nil || effective.Components.CacheIndexer.Replicas != 2 {
		t.Fatalf("expected config merged from namespace fallback, got %#v", effective.Components)
	}
}

func TestGetInferNexServiceConfig(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	cfg := platformAggregateConfig()
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(cfg).Build()

	t.Run("empty template namespace errors", func(t *testing.T) {
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		if _, err := r.getInferNexServiceConfig(ctx, defaultAggregateTemplateName); err == nil {
			t.Fatal("expected error for empty template namespace")
		}
	})

	t.Run("loads config from template namespace", func(t *testing.T) {
		r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}
		got, err := r.getInferNexServiceConfig(ctx, defaultAggregateTemplateName)
		if err != nil {
			t.Fatalf("getInferNexServiceConfig error: %v", err)
		}
		if got == nil || got.Name != defaultAggregateTemplateName {
			t.Fatalf("unexpected config: %#v", got)
		}
	})
}

func TestBuildEffectiveSpec_CustomBaseRefAndPDMode(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	custom := &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "custom-pd", Namespace: "tpl-ns"},
		Spec: infernexv1alpha1.InferNexServiceConfigSpec{
			InferNexServiceSpec: infernexv1alpha1.InferNexServiceSpec{
				Engine: &infernexv1alpha1.InferenceEngineSpec{
					InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
						Template: testTemplate("decode:v1", 8001),
					},
					Prefill: &infernexv1alpha1.InferenceEngineWorkloadSpec{
						Template: testTemplate("prefill:v1", 8000),
					},
				},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(custom, platformPDConfig()).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			BaseRefs: []infernexv1alpha1.NamedRef{{Name: "custom-pd"}},
		},
	}
	effective, mode, err := r.buildEffectiveSpec(ctx, insvc)
	if err != nil {
		t.Fatalf("buildEffectiveSpec error: %v", err)
	}
	if mode != "pd" {
		t.Fatalf("expected pd mode, got %q", mode)
	}
	if effective.Engine == nil || effective.Engine.Prefill == nil {
		t.Fatalf("expected pd engine merged from baseRef, got %#v", effective.Engine)
	}
}

func TestComputeServingMode(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := controllerTestScheme(t)
	llm := newLinkedLLM("ns-a", "demo", infernexRuntimeValue)
	if err := unstructured.SetNestedField(llm.Object, map[string]interface{}{"replicas": int64(1)}, "spec", "prefill"); err != nil {
		t.Fatal(err)
	}
	cl := &llmAwareClient{
		Client: fake.NewClientBuilder().WithScheme(s).Build(),
		llms: map[types.NamespacedName]*unstructured.Unstructured{
			{Namespace: "ns-a", Name: "demo"}: llm,
		},
	}
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	t.Run("linked llm pd mode", func(t *testing.T) {
		mode, err := r.computeServingMode(ctx, newLinkedInferNexService("ns-a", "demo"))
		if err != nil || mode != "pd" {
			t.Fatalf("expected linked pd mode, got mode=%q err=%v", mode, err)
		}
	})

	t.Run("direct aggregate mode", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			Spec: infernexv1alpha1.InferNexServiceSpec{
				Engine: &infernexv1alpha1.InferenceEngineSpec{},
			},
		}
		mode, err := r.computeServingMode(ctx, insvc)
		if err != nil || mode != "aggregate" {
			t.Fatalf("expected direct aggregate mode, got mode=%q err=%v", mode, err)
		}
	})

	t.Run("direct pd mode", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			Spec: infernexv1alpha1.InferNexServiceSpec{
				Engine: testPDEnginePrefillOnly(),
			},
		}
		mode, err := r.computeServingMode(ctx, insvc)
		if err != nil || mode != "pd" {
			t.Fatalf("expected direct pd mode, got mode=%q err=%v", mode, err)
		}
	})
}

