package controller

import (
	"context"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	igwapiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"
	lwsv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func gatewayTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	if err := infernexv1alpha1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	if err := gwapiv1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	if err := igwapiv1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	if err := corev1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	if err := appsv1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	if err := lwsv1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	return s
}

func newOwnerInfsvc(name, ns string) *infernexv1alpha1.InferNexService {
	return &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
	}
}

func mustGet[T client.Object](t *testing.T, c client.Client, nn types.NamespacedName, out T) {
	t.Helper()
	if err := c.Get(context.Background(), nn, out); err != nil {
		t.Fatalf("get %s/%s: %v", nn.Namespace, nn.Name, err)
	}
}

func mustNotFound[T client.Object](t *testing.T, c client.Client, nn types.NamespacedName, out T) {
	t.Helper()
	err := c.Get(context.Background(), nn, out)
	if client.IgnoreNotFound(err) != nil {
		t.Fatalf("get %s/%s: %v", nn.Namespace, nn.Name, err)
	}
	if err == nil {
		t.Fatalf("expected %s/%s to be deleted", nn.Namespace, nn.Name)
	}
}

func TestReconcileGatewayRouting_SourceRefSkipsCreation(t *testing.T) {
	t.Parallel()
	s := gatewayTestScheme(t)
	owner := newOwnerInfsvc("demo", "ns-a")
	owner.Spec.SourceRef = &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "demo"}
	cl := fake.NewClientBuilder().WithScheme(s).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	spec := infernexv1alpha1.InferNexServiceSpec{
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(true)},
		},
	}
	if err := r.reconcileGatewayRouting(context.Background(), owner, spec); err != nil {
		t.Fatalf("reconcileGatewayRouting error: %v", err)
	}
	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-gateway"}, &gwapiv1.Gateway{})
}

func TestReconcileGatewayRouting_RouterEnabledCreatesManagedObjects(t *testing.T) {
	t.Parallel()
	s := gatewayTestScheme(t)
	owner := newOwnerInfsvc("demo", "ns-a")
	cl := fake.NewClientBuilder().WithScheme(s).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	spec := infernexv1alpha1.InferNexServiceSpec{
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(true)},
		},
	}
	if err := r.reconcileGatewayRouting(context.Background(), owner, spec); err != nil {
		t.Fatalf("reconcileGatewayRouting error: %v", err)
	}

	gw := &gwapiv1.Gateway{}
	mustGet(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-gateway"}, gw)
	if gw.Labels["infernex.io/managed-gw-routing"] != "true" {
		t.Fatalf("gateway should be marked managed, labels=%v", gw.Labels)
	}
	route := &gwapiv1.HTTPRoute{}
	mustGet(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-route"}, route)
	pool := &igwapiv1.InferencePool{}
	mustGet(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-inference-pool"}, pool)

	proxy := &unstructured.Unstructured{}
	proxy.SetAPIVersion("gateway.envoyproxy.io/v1alpha1")
	proxy.SetKind("EnvoyProxy")
	mustGet(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-nodeport"}, proxy)
}

func TestReconcileGatewayRouting_RouterDisabledDeletesManagedObjects(t *testing.T) {
	t.Parallel()
	s := gatewayTestScheme(t)
	owner := newOwnerInfsvc("demo", "ns-a")

	managedLabels := managedGatewayLabels(owner.Name)
	gw := &gwapiv1.Gateway{ObjectMeta: metav1.ObjectMeta{
		Name: "demo-infernex-gateway", Namespace: "ns-a", Labels: managedLabels,
	}}
	route := &gwapiv1.HTTPRoute{ObjectMeta: metav1.ObjectMeta{
		Name: "demo-infernex-route", Namespace: "ns-a", Labels: managedLabels,
	}}
	pool := &igwapiv1.InferencePool{ObjectMeta: metav1.ObjectMeta{
		Name: "demo-inference-pool", Namespace: "ns-a", Labels: managedLabels,
	}}
	proxy := &unstructured.Unstructured{}
	proxy.SetAPIVersion("gateway.envoyproxy.io/v1alpha1")
	proxy.SetKind("EnvoyProxy")
	proxy.SetNamespace("ns-a")
	proxy.SetName("demo-infernex-nodeport")
	proxy.SetLabels(managedLabels)

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(gw, route, pool, proxy).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	spec := infernexv1alpha1.InferNexServiceSpec{
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(false)},
		},
	}
	if err := r.reconcileGatewayRouting(context.Background(), owner, spec); err != nil {
		t.Fatalf("reconcileGatewayRouting error: %v", err)
	}

	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-gateway"}, &gwapiv1.Gateway{})
	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-route"}, &gwapiv1.HTTPRoute{})
	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-inference-pool"}, &igwapiv1.InferencePool{})
	u := &unstructured.Unstructured{}
	u.SetAPIVersion("gateway.envoyproxy.io/v1alpha1")
	u.SetKind("EnvoyProxy")
	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-nodeport"}, u)
}

func TestDeleteManagedGatewayRouting_DoesNotDeleteUnownedObjects(t *testing.T) {
	t.Parallel()
	s := gatewayTestScheme(t)
	owner := newOwnerInfsvc("demo", "ns-a")
	route := &gwapiv1.HTTPRoute{ObjectMeta: metav1.ObjectMeta{
		Name:      "demo-infernex-route",
		Namespace: "ns-a",
		Labels:    map[string]string{"app": "external"},
	}}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(route).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	if err := r.deleteManagedGatewayRouting(context.Background(), owner); err != nil {
		t.Fatalf("deleteManagedGatewayRouting error: %v", err)
	}

	mustGet(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-route"}, &gwapiv1.HTTPRoute{})
}

func TestBuildEffectiveSpec_BaseRefsMergeOrder_FirstWinsForExistingFields(t *testing.T) {
	t.Parallel()
	s := gatewayTestScheme(t)
	cfgA := &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "tmpl"},
		Spec: infernexv1alpha1.InferNexServiceConfigSpec{
			InferNexServiceSpec: infernexv1alpha1.InferNexServiceSpec{
				Components: &infernexv1alpha1.InfernexComponentsSpec{
					CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{
						Enabled:  ptr.To(true),
						Replicas: 1,
					},
				},
			},
		},
	}
	cfgB := &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "b", Namespace: "tmpl"},
		Spec: infernexv1alpha1.InferNexServiceConfigSpec{
			InferNexServiceSpec: infernexv1alpha1.InferNexServiceSpec{
				Components: &infernexv1alpha1.InfernexComponentsSpec{
					CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{
						Enabled:  ptr.To(false),
						Replicas: 9,
					},
				},
			},
		},
	}
	platformAgg := &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: defaultAggregateTemplateName, Namespace: "tmpl"},
		Spec:       infernexv1alpha1.InferNexServiceConfigSpec{},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(cfgA, cfgB, platformAgg).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tmpl"}

	infsvc := newOwnerInfsvc("demo", "ns-a")
	infsvc.Spec.BaseRefs = []infernexv1alpha1.NamedRef{{Name: "a"}, {Name: "b"}}
	effective, _, err := r.buildEffectiveSpec(context.Background(), infsvc)
	if err != nil {
		t.Fatalf("buildEffectiveSpec error: %v", err)
	}
	if effective.Components == nil || effective.Components.CacheIndexer == nil {
		t.Fatal("expected cacheIndexer merged from baseRefs")
	}
	if effective.Components.CacheIndexer.Replicas != 1 {
		t.Fatalf("expected first baseRef replicas to stay (merge-if-missing), got %d", effective.Components.CacheIndexer.Replicas)
	}
	if effective.Components.CacheIndexer.Enabled == nil || !*effective.Components.CacheIndexer.Enabled {
		t.Fatalf("expected enabled from first baseRef, got %#v", effective.Components.CacheIndexer.Enabled)
	}
}

func TestGatewayRoutingSmallHelpers(t *testing.T) {
	t.Parallel()
	owner := newOwnerInfsvc("demo", "ns-a")
	parentRefs := toHTTPRouteParentRef("ns-a", infernexv1alpha1.NamedRef{Name: "gw-a"})
	if len(parentRefs) != 1 || string(parentRefs[0].Name) != "gw-a" {
		t.Fatalf("unexpected parent refs: %#v", parentRefs)
	}
	defRef := defaultNodePortEnvoyProxyRef(owner)
	if !isDefaultNodePortEnvoyProxyRef(defRef, owner) {
		t.Fatalf("expected default envoy proxy ref recognized: %#v", defRef)
	}
	if isDefaultNodePortEnvoyProxyRef(&gwapiv1.LocalParametersReference{Name: "other"}, owner) {
		t.Fatal("expected non-default envoy proxy ref not recognized")
	}
}

func TestReconcileGatewayRouting_ExternalRefs(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	owner := newOwnerInfsvc("demo", "ns-a")

	managedLabels := managedGatewayLabels(owner.Name)
	gw := &gwapiv1.Gateway{ObjectMeta: metav1.ObjectMeta{
		Name: "demo-infernex-gateway", Namespace: "ns-a", Labels: managedLabels,
	}}
	pool := &igwapiv1.InferencePool{ObjectMeta: metav1.ObjectMeta{
		Name: "demo-inference-pool", Namespace: "ns-a", Labels: managedLabels,
	}}
	route := &gwapiv1.HTTPRoute{ObjectMeta: metav1.ObjectMeta{
		Name: "demo-infernex-route", Namespace: "ns-a", Labels: managedLabels,
	}}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(gw, pool, route).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, Recorder: record.NewFakeRecorder(8)}

	spec := infernexv1alpha1.InferNexServiceSpec{
		Engine: &infernexv1alpha1.InferenceEngineSpec{},
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(true)},
			Gateway: &infernexv1alpha1.GatewayRefSpec{
				Ref: &infernexv1alpha1.NamedRef{Name: "external-gw"},
			},
			InferencePool: &infernexv1alpha1.InferencePoolRefSpec{
				Ref: &infernexv1alpha1.NamedRef{Name: "external-pool"},
			},
			HTTPRoute: &infernexv1alpha1.HTTPRouteRefSpec{
				Ref: &infernexv1alpha1.NamedRef{Name: "external-route"},
			},
		},
	}
	if err := r.reconcileGatewayRouting(ctx, owner, spec); err != nil {
		t.Fatalf("reconcileGatewayRouting external refs error: %v", err)
	}
	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-gateway"}, &gwapiv1.Gateway{})
	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-inference-pool"}, &igwapiv1.InferencePool{})
	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-route"}, &gwapiv1.HTTPRoute{})
}

func TestReconcileGateway_CustomGatewaySpec(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	owner := newOwnerInfsvc("demo", "ns-a")
	r := &InferNexServiceReconciler{Client: fake.NewClientBuilder().WithScheme(s).Build(), Scheme: s, Recorder: record.NewFakeRecorder(4)}

	customSpec := defaultGatewaySpec(owner)
	customSpec.GatewayClassName = "custom-class"
	parentRefs, err := r.reconcileGateway(ctx, owner, &infernexv1alpha1.GatewayRefSpec{Spec: customSpec})
	if err != nil {
		t.Fatalf("reconcileGateway custom spec error: %v", err)
	}
	if len(parentRefs) == 0 {
		t.Fatal("expected parent refs from managed gateway")
	}
	gw := &gwapiv1.Gateway{}
	mustGet(t, r.Client, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-gateway"}, gw)
	if string(gw.Spec.GatewayClassName) != "custom-class" {
		t.Fatalf("expected custom gateway class, got %q", gw.Spec.GatewayClassName)
	}
	proxy := &unstructured.Unstructured{}
	proxy.SetAPIVersion("gateway.envoyproxy.io/v1alpha1")
	proxy.SetKind("EnvoyProxy")
	mustGet(t, r.Client, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-nodeport"}, proxy)
}

func TestEnsureDefaultNodePortEnvoyProxyRef(t *testing.T) {
	t.Parallel()
	owner := newOwnerInfsvc("demo", "ns-a")
	spec := &gwapiv1.GatewaySpec{}
	if !ensureDefaultNodePortEnvoyProxyRef(spec, owner) {
		t.Fatal("expected default envoy proxy ref injection")
	}
	if spec.Infrastructure == nil || spec.Infrastructure.ParametersRef == nil {
		t.Fatalf("expected infrastructure parameters ref, got %#v", spec.Infrastructure)
	}
	again := &gwapiv1.GatewaySpec{Infrastructure: &gwapiv1.GatewayInfrastructure{
		ParametersRef: defaultNodePortEnvoyProxyRef(owner),
	}}
	if !ensureDefaultNodePortEnvoyProxyRef(again, owner) {
		t.Fatal("expected existing default ref recognized")
	}
}

