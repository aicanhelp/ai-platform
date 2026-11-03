package controller

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestGatewayDefaultHelpers(t *testing.T) {
	t.Parallel()
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}

	t.Run("inferBackendComponent aggregate and pd", func(t *testing.T) {
		agg := inferBackendComponent(infernexv1alpha1.InferNexServiceSpec{})
		if agg != "engine-aggregate" {
			t.Fatalf("expected aggregate backend, got %q", agg)
		}
		pdDecode := inferBackendComponent(infernexv1alpha1.InferNexServiceSpec{
			Engine: testPDEngineSpec(),
		})
		if pdDecode != "engine-pd-decode" {
			t.Fatalf("decode should win when both pd workloads present, got %q", pdDecode)
		}
	})

	t.Run("inferBackendServiceName pd uses shared workload svc", func(t *testing.T) {
		name := inferBackendServiceName(owner, infernexv1alpha1.InferNexServiceSpec{
			Engine: testPDEnginePrefillOnly(),
		})
		if name != "demo-workload-svc" {
			t.Fatalf("expected pd shared service name, got %q", name)
		}
		aggName := inferBackendServiceName(owner, infernexv1alpha1.InferNexServiceSpec{
			Engine: &infernexv1alpha1.InferenceEngineSpec{},
		})
		if aggName != "demo-engine-aggregate" {
			t.Fatalf("expected aggregate deployment service name, got %q", aggName)
		}
		aggLWS := inferBackendServiceName(owner, infernexv1alpha1.InferNexServiceSpec{
			Engine: &infernexv1alpha1.InferenceEngineSpec{
				InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
					DataParallelSize:      ptr.To(int32(2)),
					DataParallelSizeLocal: ptr.To(int32(1)),
					Template:              testTemplate("engine:v1", 8000),
				},
			},
		})
		if aggLWS != "demo-workload-svc" {
			t.Fatalf("expected aggregate LWS shared service name, got %q", aggLWS)
		}
	})

	t.Run("inference pool match labels direct pd vs linked", func(t *testing.T) {
		direct := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
			Spec: infernexv1alpha1.InferNexServiceSpec{
				Engine: testPDEnginePrefillOnly(),
			},
		}
		labels := infernexManagedInferencePoolMatchLabels(direct, direct.Spec)
		if labels == nil || string(labels["infernex.io/owner"]) != "demo" {
			t.Fatalf("expected direct pd owner label, got %#v", labels)
		}
		linked := direct.DeepCopy()
		linked.Spec.SourceRef = &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "llm"}
		linkedLabels := infernexManagedInferencePoolMatchLabels(linked, linked.Spec)
		if linkedLabels == nil || string(linkedLabels["app.kubernetes.io/part-of"]) != valueKServeAppPartOf {
			t.Fatalf("expected linked pd kserve part-of label, got %#v", linkedLabels)
		}
	})

	t.Run("managed httproute spec has single pool rule plus catch-all", func(t *testing.T) {
		parentRefs := managedInfernexGatewayParentRefs(owner)
		spec := infernexManagedHTTPRouteSpec(owner, infernexv1alpha1.InferNexServiceSpec{
			Engine: &infernexv1alpha1.InferenceEngineSpec{},
		}, parentRefs, "demo-inference-pool")
		if len(spec.Rules) != 2 {
			t.Fatalf("expected 2 httproute rules, got %d", len(spec.Rules))
		}
		poolPath := spec.Rules[0].Matches[0].Path.Value
		if poolPath == nil || *poolPath != "/ns-a/demo/v1" {
			t.Fatalf("expected single /v1 pool prefix, got %v", poolPath)
		}
		rewriteTo := spec.Rules[0].Filters[0].URLRewrite.Path.ReplacePrefixMatch
		if rewriteTo == nil || *rewriteTo != "/v1" {
			t.Fatalf("expected pool rule rewrite /v1, got %v", rewriteTo)
		}
	})

	t.Run("resolvedGatewayNamespacedName gateway spec branch", func(t *testing.T) {
		r := &InferNexServiceReconciler{}
		igr := &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Gateway: &infernexv1alpha1.GatewayRefSpec{
				Spec: &gwapiv1.GatewaySpec{},
			},
		}
		key := r.resolvedGatewayNamespacedName(owner, igr)
		if key.Name != "demo-"+managedGatewaySuffix {
			t.Fatalf("expected managed gateway name when spec provided, got %v", key)
		}
		custom := r.resolvedGatewayNamespacedName(owner, &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Gateway: &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "custom-gw"}},
		})
		if custom.Name != "custom-gw" {
			t.Fatalf("expected custom gateway ref name, got %v", custom)
		}
	})
}

func TestRouterNeedsTemplate(t *testing.T) {
	t.Parallel()
	enabledTrue := true
	enabledFalse := false
	if !routerNeedsTemplate(&infernexv1alpha1.InferNexServiceSpec{
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
		},
	}) {
		t.Fatal("router enabled without template should need fill")
	}
	if routerNeedsTemplate(&infernexv1alpha1.InferNexServiceSpec{
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{Enabled: &enabledFalse, Template: testTemplate("r:v1", 9000)},
		},
	}) {
		t.Fatal("disabled router should not need template")
	}
	if got := routerNeedsTemplate(&infernexv1alpha1.InferNexServiceSpec{
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{
				Enabled:  &enabledTrue,
				Template: testTemplate("r:v1", 9000),
			},
		},
	}); got {
		t.Fatalf("enabled router with template should not need fill, got %v", got)
	}
}
