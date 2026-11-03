package controller

import (
	"context"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func llmTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := controllerTestScheme(t)
	for _, gvk := range []schema.GroupVersionKind{
		{Group: "serving.kserve.io", Version: "v1alpha2", Kind: "LLMInferenceService"},
		{Group: "serving.kserve.io", Version: "v1alpha2", Kind: "LLMInferenceServiceConfig"},
	} {
		s.AddKnownTypeWithName(gvk, &unstructured.Unstructured{})
	}
	return s
}

func TestPreferredLLMInferenceServiceAPIVersions(t *testing.T) {
	t.Parallel()
	got := preferredLLMInferenceServiceAPIVersions("serving.kserve.io/v1alpha2")
	if len(got) != 2 || got[0] != "serving.kserve.io/v1alpha2" || got[1] != "serving.kserve.io/v1alpha1" {
		t.Fatalf("unexpected api version preference order: %v", got)
	}
	if versions := preferredLLMInferenceServiceAPIVersions(""); len(versions) != 2 {
		t.Fatalf("expected default versions when source ref empty, got %v", versions)
	}
}

func TestGetLinkedLLM_FallsBackAcrossAPIVersions(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := llmTestScheme(t)
	llm := &unstructured.Unstructured{}
	llm.SetAPIVersion("serving.kserve.io/v1alpha2")
	llm.SetKind("LLMInferenceService")
	llm.SetNamespace("ns-a")
	llm.SetName("demo")
	base := fake.NewClientBuilder().WithScheme(s).WithObjects(llm).Build()
	cl := &llmAwareClient{
		Client: base,
		llms: map[types.NamespacedName]*unstructured.Unstructured{
			{Namespace: "ns-a", Name: "demo"}: llm,
		},
	}
	r := &InferNexServiceReconciler{Client: cl}
	insvc := &infernexv1alpha1.InferNexService{
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{
				APIVersion: "serving.kserve.io/v1alpha1",
				Kind:       "LLMInferenceService",
				Name:       "demo",
			},
		},
	}
	got, err := r.getLinkedLLM(ctx, insvc, "ns-a", "demo")
	if err != nil || got == nil || got.GetAPIVersion() != "serving.kserve.io/v1alpha2" {
		t.Fatalf("expected v1alpha2 llm via fallback, got %#v err=%v", got, err)
	}
}

func TestLinkedLLMHasPDMode_PrefillDeploymentExists(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := llmTestScheme(t)
	if err := appsv1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	llm := newLinkedLLM("ns-a", "demo", infernexRuntimeValue)
	base := fake.NewClientBuilder().WithScheme(s).Build()
	cl := &llmAwareClient{
		Client: base,
		llms: map[types.NamespacedName]*unstructured.Unstructured{
			{Namespace: "ns-a", Name: "demo"}: llm,
		},
	}
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "demo-prefill",
			Namespace: "ns-a",
			Labels: map[string]string{
				labelAppKubernetesIOName:      "demo",
				labelAppKubernetesIOPartOf:    valueKServeAppPartOf,
				labelAppKubernetesIOComponent: kserveWorkloadComponentPrefill,
			},
		},
	}
	if err := cl.Create(ctx, dep); err != nil {
		t.Fatalf("create prefill deployment: %v", err)
	}
	r := &InferNexServiceReconciler{Client: cl}
	insvc := newLinkedInferNexService("ns-a", "demo")
	hasPD, err := r.linkedLLMHasPDMode(ctx, insvc)
	if err != nil || !hasPD {
		t.Fatalf("expected pd mode from prefill deployment, got hasPD=%v err=%v", hasPD, err)
	}
}

func TestGetLLMInferenceServiceConfig_LoadsFromBaseRef(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := llmTestScheme(t)
	cfg := &unstructured.Unstructured{}
	cfg.SetAPIVersion("serving.kserve.io/v1alpha2")
	cfg.SetKind("LLMInferenceServiceConfig")
	cfg.SetNamespace("ns-a")
	cfg.SetName("pd-template")
	if err := unstructured.SetNestedField(cfg.Object, map[string]interface{}{"replicas": int64(1)}, "spec", "prefill"); err != nil {
		t.Fatal(err)
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(cfg).Build()
	r := &InferNexServiceReconciler{Client: cl}
	got, err := r.getLLMInferenceServiceConfig(ctx, "ns-a", "pd-template")
	if err != nil || got == nil || got.GetName() != "pd-template" {
		t.Fatalf("expected llm config loaded, got %#v err=%v", got, err)
	}
	hasPrefill, err := llmHasPrefillSpec(got)
	if err != nil || !hasPrefill {
		t.Fatalf("expected prefill in config, got hasPrefill=%v err=%v", hasPrefill, err)
	}
}

func TestShouldLaunchProxyServer_DirectAndLinked(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := llmTestScheme(t)
	cl := &llmAwareClient{Client: fake.NewClientBuilder().WithScheme(s).Build(), llms: map[types.NamespacedName]*unstructured.Unstructured{}}
	r := &InferNexServiceReconciler{Client: cl}

	t.Run("direct aggregate does not launch proxy", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			Spec: infernexv1alpha1.InferNexServiceSpec{
				Engine: &infernexv1alpha1.InferenceEngineSpec{},
			},
		}
		launch, err := r.shouldLaunchProxyServer(ctx, insvc, insvc.Spec)
		if err != nil || launch {
			t.Fatalf("expected no proxy for aggregate, got launch=%v err=%v", launch, err)
		}
	})

	t.Run("direct pd launches proxy", func(t *testing.T) {
		spec := infernexv1alpha1.InferNexServiceSpec{
			Engine: testPDEnginePrefillOnly(),
		}
		launch, err := r.shouldLaunchProxyServer(ctx, &infernexv1alpha1.InferNexService{}, spec)
		if err != nil || !launch {
			t.Fatalf("expected proxy for direct pd, got launch=%v err=%v", launch, err)
		}
	})

	t.Run("linked missing llm returns false", func(t *testing.T) {
		launch, err := r.shouldLaunchProxyServer(ctx, newLinkedInferNexService("ns-a", "missing"), infernexv1alpha1.InferNexServiceSpec{})
		if err != nil || launch {
			t.Fatalf("expected false when linked llm missing, got launch=%v err=%v", launch, err)
		}
	})
}

func TestSourceRefAnnotationHelpersOnlyReadLinkedLLM(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := llmTestScheme(t)
	llm := newLinkedLLM("ns-a", "demo", infernexRuntimeValue)
	llm.SetAnnotations(map[string]string{
		kserveStopAnnotationKey:                 " true ",
		infernexDisabledComponentsAnnotationKey: "eagle-eye,mooncake",
	})
	base := fake.NewClientBuilder().WithScheme(s).WithRuntimeObjects(llm).Build()
	r := &InferNexServiceReconciler{
		Client: &llmAwareClient{
			Client: base,
			llms: map[types.NamespacedName]*unstructured.Unstructured{
				{Namespace: "ns-a", Name: "demo"}: llm,
			},
		},
	}

	linked := newLinkedInferNexService("ns-a", "demo")
	stopped, err := r.linkedLLMIsStopped(ctx, linked)
	if err != nil {
		t.Fatalf("linkedLLMIsStopped linked path error: %v", err)
	}
	if !stopped {
		t.Fatal("expected linked LLM stop annotation to stop InferNexService sidecars")
	}
	disabled, err := r.disabledComponents(ctx, linked)
	if err != nil {
		t.Fatalf("disabledComponents linked path error: %v", err)
	}
	for _, component := range []string{"eagle-eye-hardware-monitor", "eagle-eye-hardware-diagnosis", "eagle-eye-network-performance-exporter", "mooncake-master", "mooncake-metadata"} {
		if _, ok := disabled[component]; !ok {
			t.Fatalf("expected linked LLM annotation to disable %q, got %#v", component, disabled)
		}
	}

	direct := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{
		Name:      "direct",
		Namespace: "ns-a",
		Annotations: map[string]string{
			kserveStopAnnotationKey:                 "true",
			infernexDisabledComponentsAnnotationKey: cacheIndexerComponent,
		},
	}}
	stopped, err = r.linkedLLMIsStopped(ctx, direct)
	if err != nil {
		t.Fatalf("linkedLLMIsStopped direct path error: %v", err)
	}
	if stopped {
		t.Fatal("direct InferNexService must not honor serving.kserve.io/stop annotation")
	}
	disabled, err = r.disabledComponents(ctx, direct)
	if err != nil {
		t.Fatalf("disabledComponents direct path error: %v", err)
	}
	if len(disabled) != 0 {
		t.Fatalf("direct InferNexService must not honor disabled-components annotation, got %#v", disabled)
	}
}

func TestBuildInferNexServiceFromLLM_AdditionalPaths(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := controllerTestScheme(t)
	nn := types.NamespacedName{Namespace: "ns-a", Name: "demo"}

	t.Run("llm missing deletes orphan infernexservice", func(t *testing.T) {
		existing := newLinkedInferNexService("ns-a", "demo")
		base := fake.NewClientBuilder().WithScheme(s).WithObjects(existing).Build()
		cl := &llmAwareClient{Client: base, llms: map[types.NamespacedName]*unstructured.Unstructured{}}
		r := &InferNexServiceReconciler{Client: cl}
		if err := r.buildInferNexServiceFromLLMInferenceService(ctx, nn); err != nil {
			t.Fatalf("build from missing llm error: %v", err)
		}
		if got := getInferNexServiceOrNil(t, cl, "ns-a", "demo"); got != nil {
			t.Fatal("expected orphan infernexservice deleted when llm missing")
		}
	})

	t.Run("existing infernexservice is left unchanged", func(t *testing.T) {
		llm := newLinkedLLM("ns-a", "demo", infernexRuntimeValue)
		existing := newLinkedInferNexService("ns-a", "demo")
		existing.Spec.Components = &infernexv1alpha1.InfernexComponentsSpec{
			CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{Enabled: ptr.To(true), Replicas: 2},
		}
		base := fake.NewClientBuilder().WithScheme(s).WithObjects(existing).Build()
		cl := &llmAwareClient{
			Client: base,
			llms: map[types.NamespacedName]*unstructured.Unstructured{
				{Namespace: "ns-a", Name: "demo"}: llm,
			},
		}
		r := &InferNexServiceReconciler{Client: cl}
		if err := r.buildInferNexServiceFromLLMInferenceService(ctx, nn); err != nil {
			t.Fatalf("build when infsvc exists error: %v", err)
		}
		got := getInferNexServiceOrNil(t, cl, "ns-a", "demo")
		if got == nil || got.Spec.Components == nil || got.Spec.Components.CacheIndexer.Replicas != 2 {
			t.Fatalf("expected existing infernexservice preserved, got %#v", got)
		}
	})
}
