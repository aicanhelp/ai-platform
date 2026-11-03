package controller

import (
	"context"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	igwapiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestServiceHelpers(t *testing.T) {
	t.Parallel()
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	linkedOwner := owner.DeepCopy()
	linkedOwner.Spec.SourceRef = &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "demo"}

	t.Run("serviceManagerLabels", func(t *testing.T) {
		labels := serviceManagerLabels("demo", "cache-indexer")
		if labels["infernex.io/managed-service"] != "true" {
			t.Fatalf("expected managed service label, got %v", labels)
		}
	})

	t.Run("componentServiceNames mooncake and pd", func(t *testing.T) {
		if got := componentServiceNames(owner, "mooncake-master"); len(got) != 1 || got[0] != "mooncake-master-service" {
			t.Fatalf("unexpected mooncake-master service names: %v", got)
		}
		if got := componentServiceNames(owner, "engine-pd-prefill"); len(got) != 1 || !strings.Contains(got[0], "-workload-svc") {
			t.Fatalf("unexpected pd service names: %v", got)
		}
		if got := componentServiceNames(linkedOwner, "engine-pd-prefill"); len(got) != 1 || strings.Contains(got[0], "-workload-svc") {
			t.Fatalf("linked owner should not use shared pd workload service: %v", got)
		}
	})

	t.Run("managedServiceComponentLabel", func(t *testing.T) {
		if got := managedServiceComponentLabel(owner, "engine-pd-decode", componentPlan{}); got != pdWorkloadServiceComponent {
			t.Fatalf("expected pd-workload label, got %q", got)
		}
		if got := managedServiceComponentLabel(linkedOwner, "engine-pd-decode", componentPlan{}); got == pdWorkloadServiceComponent {
			t.Fatalf("linked source should not map to shared pd-workload label")
		}
		lwsPlan := componentPlan{WorkloadKind: workloadKindLeaderWorkerSet, GroupSize: 2}
		if got := managedServiceComponentLabel(owner, "engine-aggregate", lwsPlan); got != pdWorkloadServiceComponent {
			t.Fatalf("expected aggregate LWS to use shared workload label, got %q", got)
		}
	})

	t.Run("shouldKeepManagedService", func(t *testing.T) {
		desired := map[string]componentPlan{
			"engine-pd-prefill": {ServicePort: 8000},
			"cache-indexer":     {ServicePort: 8080},
		}
		if !shouldKeepManagedService(owner, desired, pdWorkloadServiceComponent, pdWorkloadServiceName(owner.Name)) {
			t.Fatal("expected shared pd workload service kept")
		}
		if !shouldKeepManagedService(owner, map[string]componentPlan{}, "mooncake-metadata", "redis-service") {
			t.Fatal("singleton mooncake service must be preserved by per-owner pruning")
		}
		desiredAggLWS := map[string]componentPlan{
			"engine-aggregate": {ServicePort: 8000, WorkloadKind: workloadKindLeaderWorkerSet, GroupSize: 2},
		}
		if !shouldKeepManagedService(owner, desiredAggLWS, pdWorkloadServiceComponent, pdWorkloadServiceName(owner.Name)) {
			t.Fatal("expected shared aggregate LWS workload service kept")
		}
		if shouldKeepManagedService(owner, desired, "cache-indexer", "unknown-service") {
			t.Fatal("unexpected keep for unmatched service name")
		}
	})

	t.Run("target port and service ports", func(t *testing.T) {
		tpl := &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{
					Name: "main",
					Ports: []corev1.ContainerPort{
						{Name: "grpc", ContainerPort: 9002},
						{Name: "grpc-health", ContainerPort: 9003},
					},
				}},
			},
		}
		if p := firstContainerTargetPort("cache-indexer", nil); p.IntVal != cacheIndexerServicePort {
			t.Fatalf("cache indexer default target port mismatch: %v", p)
		}
		if p := hermesNamedContainerTargetPort(tpl, "grpc", intstr.FromInt(80)); p.Type != intstr.String || p.StrVal != "grpc" {
			t.Fatalf("expected named target port grpc, got %v", p)
		}
		if hp, ok := hermesNamedContainerPort(tpl, "grpc-health"); !ok || hp != 9003 {
			t.Fatalf("expected grpc-health=9003, got %d %v", hp, ok)
		}
		ports := componentServicePorts(hermesRouterComponent, componentPlan{ServicePort: 9002, Template: tpl}, intstr.FromInt(9002))
		if len(ports) != 2 {
			t.Fatalf("expected grpc + grpc-health ports, got %v", ports)
		}
	})

	t.Run("hermes service targets main container when tokenizer is first", func(t *testing.T) {
		tpl := &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "tokenizer"},
					{
						Name: "main",
						Ports: []corev1.ContainerPort{
							{Name: "grpc", ContainerPort: 9002},
							{Name: "grpc-health", ContainerPort: 9003},
						},
					},
				},
			},
		}
		ports := componentServicePorts(hermesRouterComponent, componentPlan{ServicePort: 9002, Template: tpl}, intstr.FromInt(8000))
		if len(ports) < 1 || ports[0].TargetPort.StrVal != "grpc" {
			t.Fatalf("expected grpc targetPort on main, got %#v", ports)
		}
	})

	t.Run("hermes service ignores sidecars before and after main", func(t *testing.T) {
		tpl := &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "tokenizer", Ports: []corev1.ContainerPort{{Name: "grpc", ContainerPort: 1}}},
					{Name: "prediction"},
					{
						Name: "main",
						Ports: []corev1.ContainerPort{
							{Name: "grpc", ContainerPort: 9002},
							{Name: "grpc-health", ContainerPort: 9003},
						},
					},
				},
			},
		}
		ports := componentServicePorts(hermesRouterComponent, componentPlan{ServicePort: 9002, Template: tpl}, intstr.FromInt(8000))
		if len(ports) < 1 || ports[0].TargetPort.StrVal != "grpc" {
			t.Fatalf("expected grpc on main only, got %#v", ports)
		}
	})

	t.Run("hermes service falls back when main missing", func(t *testing.T) {
		tpl := &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "epp", Ports: []corev1.ContainerPort{{Name: "grpc", ContainerPort: 9002}}},
				},
			},
		}
		ports := componentServicePorts(hermesRouterComponent, componentPlan{ServicePort: 9002, Template: tpl}, intstr.FromInt(8000))
		if len(ports) < 1 || ports[0].TargetPort.IntVal != 8000 {
			t.Fatalf("expected numeric fallback without main, got %#v", ports)
		}
	})
}

func TestRolloutAndStatusHelpers(t *testing.T) {
	t.Parallel()
	t.Run("deploymentRolloutReady", func(t *testing.T) {
		d := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Generation: 2},
			Spec:       appsv1.DeploymentSpec{Replicas: ptr.To(int32(2))},
			Status: appsv1.DeploymentStatus{
				ObservedGeneration: 2,
				ReadyReplicas:      2,
			},
		}
		ok, msg := deploymentRolloutReady(d)
		if !ok || msg != "" {
			t.Fatalf("expected ready deployment, got ok=%v msg=%q", ok, msg)
		}
		d.Status.ReadyReplicas = 1
		ok, msg = deploymentRolloutReady(d)
		if ok || !strings.Contains(msg, "ready pods") {
			t.Fatalf("expected not ready pods, got ok=%v msg=%q", ok, msg)
		}
	})

	t.Run("daemonSetRolloutReady", func(t *testing.T) {
		ds := &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Generation: 1},
			Status: appsv1.DaemonSetStatus{
				ObservedGeneration:     1,
				DesiredNumberScheduled: 3,
				NumberReady:            3,
				UpdatedNumberScheduled: 3,
			},
		}
		ok, msg := daemonSetRolloutReady(ds)
		if !ok || msg != "" {
			t.Fatalf("expected ready daemonset, got ok=%v msg=%q", ok, msg)
		}
		ds.Status.NumberReady = 2
		ok, msg = daemonSetRolloutReady(ds)
		if ok || !strings.Contains(msg, "ready pods") {
			t.Fatalf("expected not ready daemonset, got ok=%v msg=%q", ok, msg)
		}
		ds = &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Generation: 2},
			Status: appsv1.DaemonSetStatus{
				ObservedGeneration:     1,
				DesiredNumberScheduled: 2,
				NumberReady:            2,
				UpdatedNumberScheduled: 2,
			},
		}
		ok, msg = daemonSetRolloutReady(ds)
		if ok || !strings.Contains(msg, "observe latest spec") {
			t.Fatalf("expected not ready on daemonset generation mismatch, got ok=%v msg=%q", ok, msg)
		}
		ds = &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Generation: 1},
			Status: appsv1.DaemonSetStatus{
				ObservedGeneration:     1,
				DesiredNumberScheduled: 3,
				NumberReady:            3,
				UpdatedNumberScheduled: 1,
			},
		}
		ok, msg = daemonSetRolloutReady(ds)
		if ok || !strings.Contains(msg, "updated pods") {
			t.Fatalf("expected not ready on daemonset rollout lag, got ok=%v msg=%q", ok, msg)
		}
		ds = &appsv1.DaemonSet{
			Status: appsv1.DaemonSetStatus{DesiredNumberScheduled: 0},
		}
		ok, msg = daemonSetRolloutReady(ds)
		if !ok || msg != "" {
			t.Fatalf("expected ready when no daemonset pods scheduled, got ok=%v msg=%q", ok, msg)
		}
	})

	t.Run("summary helpers", func(t *testing.T) {
		desired := map[string]componentPlan{"cache-indexer": {}, "proxy-server": {}}
		if !containsAnyKey(desired, "cache-indexer") {
			t.Fatal("containsAnyKey should match existing key")
		}
		comp := &infernexv1alpha1.InferNexComponentStatuses{
			CacheIndexer: &infernexv1alpha1.ComponentStatus{Ready: false, Message: "cache down"},
		}
		if !hasAnyComponentStatus(comp) {
			t.Fatal("hasAnyComponentStatus should be true")
		}
		msg := statusSummaryFromComponents(comp, false)
		if !strings.Contains(msg, "cacheIndexer: cache down") {
			t.Fatalf("unexpected summary: %q", msg)
		}
	})

	t.Run("gateway and route accepted helpers", func(t *testing.T) {
		gw := &gwapiv1.Gateway{
			ObjectMeta: metav1.ObjectMeta{Name: "gw", Namespace: "ns-a"},
			Status: gwapiv1.GatewayStatus{
				Conditions: []metav1.Condition{
					{Type: "Accepted", Status: metav1.ConditionTrue},
					{Type: "Programmed", Status: metav1.ConditionTrue},
				},
			},
		}
		ok, msg := gatewayProgrammed(gw)
		if !ok || msg != "" {
			t.Fatalf("expected gateway programmed, got ok=%v msg=%q", ok, msg)
		}
		route := &gwapiv1.HTTPRoute{
			ObjectMeta: metav1.ObjectMeta{Name: "route", Namespace: "ns-a"},
			Status: gwapiv1.HTTPRouteStatus{
				RouteStatus: gwapiv1.RouteStatus{
					Parents: []gwapiv1.RouteParentStatus{{
						ParentRef: gwapiv1.ParentReference{Name: "gw"},
						Conditions: []metav1.Condition{
							{Type: "Accepted", Status: metav1.ConditionTrue},
							{Type: "ResolvedRefs", Status: metav1.ConditionTrue},
						},
					}},
				},
			},
		}
		ok, msg = httpRouteAcceptedByGateway(route, "gw")
		if !ok || msg != "" {
			t.Fatalf("expected httproute accepted, got ok=%v msg=%q", ok, msg)
		}
		route.Status.Parents[0].Conditions[0].Status = metav1.ConditionFalse
		route.Status.Parents[0].Conditions[0].Message = "backend missing"
		ok, msg = httpRouteAcceptedByGateway(route, "gw")
		if ok || !strings.Contains(msg, "not accepted by gateway") {
			t.Fatalf("expected httproute not accepted, got ok=%v msg=%q", ok, msg)
		}
		route.Status.Parents[0].Conditions = []metav1.Condition{
			{Type: "ResolvedRefs", Status: metav1.ConditionTrue},
		}
		ok, msg = httpRouteAcceptedByGateway(route, "gw")
		if ok || !strings.Contains(msg, "missing Accepted condition") {
			t.Fatalf("expected missing Accepted condition, got ok=%v msg=%q", ok, msg)
		}
		route.Status.Parents[0].Conditions = []metav1.Condition{
			{Type: "Accepted", Status: metav1.ConditionTrue},
			{Type: "ResolvedRefs", Status: metav1.ConditionFalse, Message: "service not found"},
		}
		ok, msg = httpRouteAcceptedByGateway(route, "gw")
		if ok || !strings.Contains(msg, "unresolved refs") {
			t.Fatalf("expected unresolved refs failure, got ok=%v msg=%q", ok, msg)
		}
		route.Status.Parents[0].ParentRef.Name = "other-gw"
		ok, msg = httpRouteAcceptedByGateway(route, "gw")
		if ok || !strings.Contains(msg, "no parent status for gateway") {
			t.Fatalf("expected no matching parent, got ok=%v msg=%q", ok, msg)
		}
	})
}

func TestInferencePoolReady_FallbackByMatchLabels(t *testing.T) {
	t.Parallel()
	s := gatewayTestScheme(t)
	pool := &igwapiv1.InferencePool{
		ObjectMeta: metav1.ObjectMeta{Name: "pool", Namespace: "ns-a"},
		Spec: igwapiv1.InferencePoolSpec{
			Selector: igwapiv1.LabelSelector{
				MatchLabels: map[igwapiv1.LabelKey]igwapiv1.LabelValue{
					"app": "demo",
				},
			},
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "ns-a", Labels: map[string]string{"app": "demo"}},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(pool, pod).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	ok, msg, reason := r.inferencePoolReady(context.Background(), pool, "gw")
	if !ok || msg != "" || reason != "" {
		t.Fatalf("expected pool ready via match labels fallback, got ok=%v msg=%q reason=%q", ok, msg, reason)
	}
}

func TestInferencePoolReady_GatewayParentStatus(t *testing.T) {
	t.Parallel()
	s := gatewayTestScheme(t)
	ctx := context.Background()

	t.Run("accepted by gateway parent", func(t *testing.T) {
		pool := acceptedInferencePool("pool", "ns-a", "gw")
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(pool).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		ok, msg, reason := r.inferencePoolReady(ctx, pool, "gw")
		if !ok || msg != "" || reason != "" {
			t.Fatalf("expected pool ready via gateway parent, got ok=%v msg=%q reason=%q", ok, msg, reason)
		}
	})

	t.Run("not accepted by gateway parent", func(t *testing.T) {
		pool := acceptedInferencePool("pool", "ns-a", "gw")
		pool.Status.Parents[0].Conditions[0].Status = metav1.ConditionFalse
		pool.Status.Parents[0].Conditions[0].Message = "selector invalid"
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(pool).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		ok, msg, reason := r.inferencePoolReady(ctx, pool, "gw")
		if ok || reason != "InferencePoolNotAccepted" || !strings.Contains(msg, "not accepted by gateway") {
			t.Fatalf("expected InferencePoolNotAccepted, got ok=%v msg=%q reason=%q", ok, msg, reason)
		}
	})

	t.Run("missing accepted condition on parent", func(t *testing.T) {
		pool := acceptedInferencePool("pool", "ns-a", "gw")
		pool.Status.Parents[0].Conditions = []metav1.Condition{
			{Type: "ResolvedRefs", Status: metav1.ConditionTrue},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(pool).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		ok, msg, reason := r.inferencePoolReady(ctx, pool, "gw")
		if ok || reason != "InferencePoolNotAccepted" || !strings.Contains(msg, "missing Accepted condition") {
			t.Fatalf("expected missing Accepted on pool parent, got ok=%v msg=%q reason=%q", ok, msg, reason)
		}
	})

	t.Run("unresolved refs on parent", func(t *testing.T) {
		pool := acceptedInferencePool("pool", "ns-a", "gw")
		pool.Status.Parents[0].Conditions = []metav1.Condition{
			{Type: "Accepted", Status: metav1.ConditionTrue},
			{Type: "ResolvedRefs", Status: metav1.ConditionFalse, Message: "backend ref missing"},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(pool).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		ok, msg, reason := r.inferencePoolReady(ctx, pool, "gw")
		if ok || reason != "InferencePoolRefsUnresolved" || !strings.Contains(msg, "unresolved refs") {
			t.Fatalf("expected InferencePoolRefsUnresolved, got ok=%v msg=%q reason=%q", ok, msg, reason)
		}
	})
}

func TestStatusForWorkloadKeys_DeploymentAndDaemonSet(t *testing.T) {
	t.Parallel()
	s := gatewayTestScheme(t)
	replicas := int32(1)
	owner := "demo"
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: owner + "-proxy-server", Namespace: "ns-a", Generation: 1},
		Spec:       appsv1.DeploymentSpec{Replicas: &replicas},
		Status: appsv1.DeploymentStatus{
			ObservedGeneration: 1,
			ReadyReplicas:      1,
			Replicas:           1,
		},
	}
	ds := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: owner + "-cache-indexer", Namespace: "ns-a"},
		Spec:       appsv1.DaemonSetSpec{},
		Status: appsv1.DaemonSetStatus{
			ObservedGeneration:     1,
			DesiredNumberScheduled: 1,
			NumberReady:            1,
			UpdatedNumberScheduled: 1,
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(dep, ds).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	desired := map[string]componentPlan{
		"proxy-server":  {ServicePort: 8000},
		"cache-indexer": {ServicePort: 8080, WorkloadKind: workloadKindDaemonSet},
	}
	st, err := r.statusForWorkloadKeys(context.Background(), "ns-a", owner, []string{"proxy-server", "cache-indexer"}, desired)
	if err != nil {
		t.Fatalf("statusForWorkloadKeys error: %v", err)
	}
	if st == nil || !st.Ready {
		t.Fatalf("expected ready status, got %#v", st)
	}
}

func TestBuildBackendAndComponentConditions(t *testing.T) {
	t.Parallel()
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", Generation: 2},
	}
	cond := buildBackendReadyCondition(insvc, nil)
	if cond.Status != metav1.ConditionFalse || cond.Reason != "BackendNotConfigured" {
		t.Fatalf("expected backend not configured, got %#v", cond)
	}
	insvc.Spec.SourceRef = &infernexv1alpha1.SourceRef{Name: "llm", Kind: "LLMInferenceService"}
	cond = buildBackendReadyCondition(insvc, nil)
	if cond.Status != metav1.ConditionTrue || cond.Reason != "SourceRefManagedByLLMInferenceService" {
		t.Fatalf("expected sourceRef backend ready, got %#v", cond)
	}

	comp := &infernexv1alpha1.InferNexComponentStatuses{
		ProxyServer: &infernexv1alpha1.ComponentStatus{Ready: false, Message: "starting"},
	}
	readyCond := buildComponentsReadyCondition(insvc, comp, false, readyReasonProxyServerNotReady, "starting")
	if readyCond.Status != metav1.ConditionFalse || readyCond.Reason != readyReasonProxyServerNotReady {
		t.Fatalf("expected components not ready condition, got %#v", readyCond)
	}

	keys := keysPresentInDesired(map[string]componentPlan{"a": {}, "b": {}}, []string{"a", "x"})
	if len(keys) != 1 || keys[0] != "a" {
		t.Fatalf("unexpected keysPresentInDesired result: %v", keys)
	}

	specCond := buildSpecValidatedCondition(insvc)
	if specCond.Type != "SpecValidated" || specCond.Status != metav1.ConditionTrue {
		t.Fatalf("unexpected spec validated condition: %#v", specCond)
	}
}

func TestReconcileAndPruneComponentServices(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	tpl := &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Ports: []corev1.ContainerPort{{Name: "grpc", ContainerPort: 8000}}}},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	err := r.reconcileComponentService(ctx, owner, hermesRouterComponent, componentPlan{
		ServicePort: 9002,
		Template:    tpl,
	})
	if err != nil {
		t.Fatalf("reconcileComponentService error: %v", err)
	}
	svc := &corev1.Service{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: owner.Name + "-" + hermesRouterComponent}, svc); err != nil {
		t.Fatalf("expected service created: %v", err)
	}
	if len(svc.Spec.Ports) == 0 || svc.Spec.Ports[0].Port != 9002 {
		t.Fatalf("unexpected service ports: %#v", svc.Spec.Ports)
	}

	// keep desired service
	if err := r.pruneComponentServices(ctx, owner, map[string]componentPlan{
		hermesRouterComponent: {ServicePort: 9002},
	}); err != nil {
		t.Fatalf("pruneComponentServices keep error: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: owner.Name + "-" + hermesRouterComponent}, &corev1.Service{}); err != nil {
		t.Fatalf("service should still exist: %v", err)
	}

	// prune when no longer desired
	if err := r.pruneComponentServices(ctx, owner, map[string]componentPlan{}); err != nil {
		t.Fatalf("pruneComponentServices delete error: %v", err)
	}
	err = cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: owner.Name + "-" + hermesRouterComponent}, &corev1.Service{})
	if !apierrors.IsNotFound(err) {
		t.Fatalf("expected service deleted, got err=%v", err)
	}
}

func TestReconcileComponentService_MooncakeSingleton(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "isvc-a", Namespace: "ns-a"}}
	tpl := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Ports: []corev1.ContainerPort{{ContainerPort: 6379}}}}}}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	if err := r.reconcileComponentService(ctx, owner, "mooncake-metadata", componentPlan{ServicePort: 6379, Template: tpl}); err != nil {
		t.Fatalf("reconcile singleton service: %v", err)
	}
	svc := &corev1.Service{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "redis-service"}, svc); err != nil {
		t.Fatalf("expected canonical Redis Service: %v", err)
	}
	if svc.Spec.Selector["app"] != "redis" || len(svc.OwnerReferences) != 0 {
		t.Fatalf("expected fixed selector and no owner reference, got selector=%#v refs=%#v", svc.Spec.Selector, svc.OwnerReferences)
	}
}

func TestBuildGatewayRoutingReadyCondition(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", Generation: 3},
	}

	t.Run("sourceRef short-circuit", func(t *testing.T) {
		cl := fake.NewClientBuilder().WithScheme(s).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		linked := insvc.DeepCopy()
		linked.Spec.SourceRef = &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "llm"}
		cond, err := r.buildGatewayRoutingReadyCondition(ctx, linked, infernexv1alpha1.InferNexServiceSpec{})
		if err != nil {
			t.Fatalf("buildGatewayRoutingReadyCondition error: %v", err)
		}
		if cond.Reason != "SourceRefManagedByLLMInferenceService" || cond.Status != metav1.ConditionTrue {
			t.Fatalf("unexpected condition for sourceRef mode: %#v", cond)
		}
	})

	t.Run("router disabled", func(t *testing.T) {
		cl := fake.NewClientBuilder().WithScheme(s).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		enabledFalse := false
		spec := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router: &infernexv1alpha1.ComponentSpec{Enabled: &enabledFalse},
			},
		}
		cond, err := r.buildGatewayRoutingReadyCondition(ctx, insvc, spec)
		if err != nil {
			t.Fatalf("buildGatewayRoutingReadyCondition error: %v", err)
		}
		if cond.Reason != "RouterDisabled" || cond.Status != metav1.ConditionTrue {
			t.Fatalf("unexpected router disabled condition: %#v", cond)
		}
	})

	t.Run("gateway not found", func(t *testing.T) {
		cl := fake.NewClientBuilder().WithScheme(s).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		enabledTrue := true
		spec := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:  &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
				Gateway: &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
			},
		}
		cond, err := r.buildGatewayRoutingReadyCondition(ctx, insvc, spec)
		if err != nil {
			t.Fatalf("buildGatewayRoutingReadyCondition error: %v", err)
		}
		if cond.Reason != "GatewayNotFound" || cond.Status != metav1.ConditionFalse {
			t.Fatalf("unexpected gateway not found condition: %#v", cond)
		}
	})

	t.Run("gateway route and pool ready", func(t *testing.T) {
		enabledTrue := true
		spec := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:        &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
				Gateway:       &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
				HTTPRoute:     &infernexv1alpha1.HTTPRouteRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "route"}},
				InferencePool: &infernexv1alpha1.InferencePoolRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "pool"}},
			},
		}
		gw := &gwapiv1.Gateway{
			ObjectMeta: metav1.ObjectMeta{Name: "gw", Namespace: "ns-a"},
			Status: gwapiv1.GatewayStatus{Conditions: []metav1.Condition{
				{Type: "Accepted", Status: metav1.ConditionTrue},
				{Type: "Programmed", Status: metav1.ConditionTrue},
			}},
		}
		route := &gwapiv1.HTTPRoute{
			ObjectMeta: metav1.ObjectMeta{Name: "route", Namespace: "ns-a"},
			Status: gwapiv1.HTTPRouteStatus{
				RouteStatus: gwapiv1.RouteStatus{
					Parents: []gwapiv1.RouteParentStatus{{
						ParentRef: gwapiv1.ParentReference{Name: "gw"},
						Conditions: []metav1.Condition{
							{Type: "Accepted", Status: metav1.ConditionTrue},
							{Type: "ResolvedRefs", Status: metav1.ConditionTrue},
						},
					}},
				},
			},
		}
		pool := &igwapiv1.InferencePool{
			ObjectMeta: metav1.ObjectMeta{Name: "pool", Namespace: "ns-a"},
			Status: igwapiv1.InferencePoolStatus{
				Parents: []igwapiv1.ParentStatus{{
					ParentRef: igwapiv1.ParentReference{Name: "gw"},
					Conditions: []metav1.Condition{
						{Type: "Accepted", Status: metav1.ConditionTrue},
						{Type: "ResolvedRefs", Status: metav1.ConditionTrue},
					},
				}},
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(gw, route, pool).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		cond, err := r.buildGatewayRoutingReadyCondition(ctx, insvc, spec)
		if err != nil {
			t.Fatalf("buildGatewayRoutingReadyCondition error: %v", err)
		}
		if cond.Status != metav1.ConditionTrue || cond.Reason != "GatewayRoutingReady" || !strings.Contains(cond.Message, "gateway routing linked") {
			t.Fatalf("unexpected ready condition: %#v", cond)
		}
	})

	t.Run("gateway not programmed", func(t *testing.T) {
		enabledTrue := true
		spec := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:  &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
				Gateway: &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
			},
		}
		gw := &gwapiv1.Gateway{
			ObjectMeta: metav1.ObjectMeta{Name: "gw", Namespace: "ns-a"},
			Status: gwapiv1.GatewayStatus{Conditions: []metav1.Condition{
				{Type: "Accepted", Status: metav1.ConditionTrue},
				{Type: "Programmed", Status: metav1.ConditionFalse, Message: "waiting for controller"},
			}},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(gw).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		cond, err := r.buildGatewayRoutingReadyCondition(ctx, insvc, spec)
		if err != nil {
			t.Fatalf("buildGatewayRoutingReadyCondition error: %v", err)
		}
		if cond.Reason != "GatewayNotProgrammed" || cond.Status != metav1.ConditionFalse {
			t.Fatalf("unexpected gateway not programmed condition: %#v", cond)
		}
	})

	t.Run("httproute not found", func(t *testing.T) {
		enabledTrue := true
		spec := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:    &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
				Gateway:   &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
				HTTPRoute: &infernexv1alpha1.HTTPRouteRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "missing-route"}},
			},
		}
		gw := &gwapiv1.Gateway{
			ObjectMeta: metav1.ObjectMeta{Name: "gw", Namespace: "ns-a"},
			Status: gwapiv1.GatewayStatus{Conditions: []metav1.Condition{
				{Type: "Accepted", Status: metav1.ConditionTrue},
				{Type: "Programmed", Status: metav1.ConditionTrue},
			}},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(gw).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		cond, err := r.buildGatewayRoutingReadyCondition(ctx, insvc, spec)
		if err != nil {
			t.Fatalf("buildGatewayRoutingReadyCondition error: %v", err)
		}
		if cond.Reason != "HTTPRouteNotFound" || cond.Status != metav1.ConditionFalse {
			t.Fatalf("unexpected httproute not found condition: %#v", cond)
		}
	})

	t.Run("httproute not accepted", func(t *testing.T) {
		enabledTrue := true
		spec := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:    &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
				Gateway:   &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
				HTTPRoute: &infernexv1alpha1.HTTPRouteRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "route"}},
			},
		}
		gw := &gwapiv1.Gateway{
			ObjectMeta: metav1.ObjectMeta{Name: "gw", Namespace: "ns-a"},
			Status: gwapiv1.GatewayStatus{Conditions: []metav1.Condition{
				{Type: "Accepted", Status: metav1.ConditionTrue},
				{Type: "Programmed", Status: metav1.ConditionTrue},
			}},
		}
		route := &gwapiv1.HTTPRoute{
			ObjectMeta: metav1.ObjectMeta{Name: "route", Namespace: "ns-a"},
			Status: gwapiv1.HTTPRouteStatus{
				RouteStatus: gwapiv1.RouteStatus{
					Parents: []gwapiv1.RouteParentStatus{{
						ParentRef: gwapiv1.ParentReference{Name: "gw"},
						Conditions: []metav1.Condition{
							{Type: "Accepted", Status: metav1.ConditionFalse, Message: "backend missing"},
						},
					}},
				},
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(gw, route).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		cond, err := r.buildGatewayRoutingReadyCondition(ctx, insvc, spec)
		if err != nil {
			t.Fatalf("buildGatewayRoutingReadyCondition error: %v", err)
		}
		if cond.Reason != "HTTPRouteNotAccepted" || cond.Status != metav1.ConditionFalse {
			t.Fatalf("unexpected httproute not accepted condition: %#v", cond)
		}
	})

	t.Run("inference pool no endpoints", func(t *testing.T) {
		enabledTrue := true
		spec := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:        &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
				Gateway:       &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
				HTTPRoute:     &infernexv1alpha1.HTTPRouteRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "route"}},
				InferencePool: &infernexv1alpha1.InferencePoolRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "pool"}},
			},
		}
		gw := &gwapiv1.Gateway{
			ObjectMeta: metav1.ObjectMeta{Name: "gw", Namespace: "ns-a"},
			Status: gwapiv1.GatewayStatus{Conditions: []metav1.Condition{
				{Type: "Accepted", Status: metav1.ConditionTrue},
				{Type: "Programmed", Status: metav1.ConditionTrue},
			}},
		}
		route := &gwapiv1.HTTPRoute{
			ObjectMeta: metav1.ObjectMeta{Name: "route", Namespace: "ns-a"},
			Status: gwapiv1.HTTPRouteStatus{
				RouteStatus: gwapiv1.RouteStatus{
					Parents: []gwapiv1.RouteParentStatus{{
						ParentRef: gwapiv1.ParentReference{Name: "gw"},
						Conditions: []metav1.Condition{
							{Type: "Accepted", Status: metav1.ConditionTrue},
							{Type: "ResolvedRefs", Status: metav1.ConditionTrue},
						},
					}},
				},
			},
		}
		pool := &igwapiv1.InferencePool{
			ObjectMeta: metav1.ObjectMeta{Name: "pool", Namespace: "ns-a"},
			Spec: igwapiv1.InferencePoolSpec{
				Selector: igwapiv1.LabelSelector{
					MatchLabels: map[igwapiv1.LabelKey]igwapiv1.LabelValue{
						"app": "engine",
					},
				},
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(gw, route, pool).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		cond, err := r.buildGatewayRoutingReadyCondition(ctx, insvc, spec)
		if err != nil {
			t.Fatalf("buildGatewayRoutingReadyCondition error: %v", err)
		}
		if cond.Reason != "InferencePoolNoEndpoints" || cond.Status != metav1.ConditionFalse {
			t.Fatalf("unexpected inference pool no endpoints condition: %#v", cond)
		}
	})

	t.Run("inference pool not accepted by gateway parent", func(t *testing.T) {
		enabledTrue := true
		spec := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:        &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
				Gateway:       &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
				HTTPRoute:     &infernexv1alpha1.HTTPRouteRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "route"}},
				InferencePool: &infernexv1alpha1.InferencePoolRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "pool"}},
			},
		}
		pool := acceptedInferencePool("pool", "ns-a", "gw")
		pool.Status.Parents[0].Conditions[0].Status = metav1.ConditionFalse
		pool.Status.Parents[0].Conditions[0].Message = "selector invalid"
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			programmedGateway("gw", "ns-a"),
			acceptedHTTPRoute("route", "ns-a", "gw"),
			pool,
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		cond, err := r.buildGatewayRoutingReadyCondition(ctx, insvc, spec)
		if err != nil {
			t.Fatalf("buildGatewayRoutingReadyCondition error: %v", err)
		}
		if cond.Reason != "InferencePoolNotAccepted" || cond.Status != metav1.ConditionFalse {
			t.Fatalf("unexpected inference pool not accepted condition: %#v", cond)
		}
	})

	t.Run("inference pool unresolved refs on gateway parent", func(t *testing.T) {
		enabledTrue := true
		spec := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:        &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
				Gateway:       &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
				HTTPRoute:     &infernexv1alpha1.HTTPRouteRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "route"}},
				InferencePool: &infernexv1alpha1.InferencePoolRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "pool"}},
			},
		}
		pool := acceptedInferencePool("pool", "ns-a", "gw")
		pool.Status.Parents[0].Conditions = []metav1.Condition{
			{Type: "Accepted", Status: metav1.ConditionTrue},
			{Type: "ResolvedRefs", Status: metav1.ConditionFalse, Message: "backend ref missing"},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			programmedGateway("gw", "ns-a"),
			acceptedHTTPRoute("route", "ns-a", "gw"),
			pool,
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		cond, err := r.buildGatewayRoutingReadyCondition(ctx, insvc, spec)
		if err != nil {
			t.Fatalf("buildGatewayRoutingReadyCondition error: %v", err)
		}
		if cond.Reason != "InferencePoolRefsUnresolved" || cond.Status != metav1.ConditionFalse {
			t.Fatalf("unexpected inference pool refs unresolved condition: %#v", cond)
		}
	})

	t.Run("httproute unresolved refs surfaces HTTPRouteNotAccepted", func(t *testing.T) {
		enabledTrue := true
		spec := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:    &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
				Gateway:   &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
				HTTPRoute: &infernexv1alpha1.HTTPRouteRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "route"}},
			},
		}
		route := acceptedHTTPRoute("route", "ns-a", "gw")
		route.Status.Parents[0].Conditions = []metav1.Condition{
			{Type: "Accepted", Status: metav1.ConditionTrue},
			{Type: "ResolvedRefs", Status: metav1.ConditionFalse, Message: "service not found"},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			programmedGateway("gw", "ns-a"),
			route,
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		cond, err := r.buildGatewayRoutingReadyCondition(ctx, insvc, spec)
		if err != nil {
			t.Fatalf("buildGatewayRoutingReadyCondition error: %v", err)
		}
		if cond.Reason != "HTTPRouteNotAccepted" || cond.Status != metav1.ConditionFalse {
			t.Fatalf("unexpected httproute unresolved refs condition: %#v", cond)
		}
	})
}

func TestComputeInferNexServiceStatus_UsesGatewayConditionFlow(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	cl := fake.NewClientBuilder().WithScheme(s).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", Generation: 9},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "llm"},
		},
	}
	st, err := r.computeInferNexServiceStatus(ctx, insvc, map[string]componentPlan{}, infernexv1alpha1.InferNexServiceSpec{}, "pd")
	if err != nil {
		t.Fatalf("computeInferNexServiceStatus error: %v", err)
	}
	if !st.Ready || st.Mode != "pd" {
		t.Fatalf("unexpected status readiness/mode: %#v", st)
	}
	if len(st.Conditions) == 0 {
		t.Fatal("expected conditions set")
	}
	var hasSpecValidated, hasGateway bool
	for _, c := range st.Conditions {
		if c.Type == "SpecValidated" {
			hasSpecValidated = true
		}
		if c.Type == "GatewayRoutingReady" {
			hasGateway = true
		}
	}
	if !hasSpecValidated || !hasGateway {
		t.Fatalf("expected spec and gateway conditions in %#v", st.Conditions)
	}
	var readyCond metav1.Condition
	for _, c := range st.Conditions {
		if c.Type == conditionTypeReady {
			readyCond = c
		}
	}
	if readyCond.Status != metav1.ConditionTrue || readyCond.Reason != readyReasonAllManagedResourcesReady {
		t.Fatalf("expected Ready=true AllManagedResourcesReady, got %#v", readyCond)
	}
}

func TestReconcileInferNexServiceDeletion_CleansOwnedGatewayRouting(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	owner := newOwnerInfsvc("demo", "ns-a")
	owner.Finalizers = []string{infernexServiceFinalizer}
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
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner, gw, route, pool).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	res, err := r.reconcileInferNexServiceDeletion(ctx, owner)
	if err != nil {
		t.Fatalf("reconcileInferNexServiceDeletion error: %v", err)
	}
	if res.Requeue {
		t.Fatalf("unexpected requeue: %#v", res)
	}
	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-gateway"}, &gwapiv1.Gateway{})
	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-route"}, &gwapiv1.HTTPRoute{})
	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: "demo-inference-pool"}, &igwapiv1.InferencePool{})
	fresh := &infernexv1alpha1.InferNexService{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo"}, fresh); err != nil {
		t.Fatalf("get infernexservice after deletion reconcile: %v", err)
	}
	if len(fresh.Finalizers) != 0 {
		t.Fatalf("expected finalizer removed, got %v", fresh.Finalizers)
	}
}

func TestComputeInferNexServiceStatus_NotReadyWhenDeploymentPending(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	owner := "demo"
	replicas := int32(1)
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: owner + "-cache-indexer", Namespace: "ns-a", Generation: 1},
		Spec:       appsv1.DeploymentSpec{Replicas: &replicas},
		Status: appsv1.DeploymentStatus{
			ObservedGeneration: 1,
			Replicas:           1,
			ReadyReplicas:      0,
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(dep).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: owner, Namespace: "ns-a", Generation: 4},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "llm"},
		},
	}
	desired := map[string]componentPlan{
		cacheIndexerComponent: {ServicePort: 8080, WorkloadKind: workloadKindDeployment},
	}
	st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
	if err != nil {
		t.Fatalf("computeInferNexServiceStatus error: %v", err)
	}
	if st.Ready {
		t.Fatalf("expected overall not ready when deployment has 0 ready replicas, got %#v", st)
	}
	if st.Components == nil || st.Components.CacheIndexer == nil || st.Components.CacheIndexer.Ready {
		t.Fatalf("expected cache-indexer component not ready, got %#v", st.Components)
	}
	var avail, readyCond metav1.Condition
	for _, c := range st.Conditions {
		switch c.Type {
		case "Available":
			avail = c
		case conditionTypeReady:
			readyCond = c
		}
	}
	if avail.Status != metav1.ConditionFalse || avail.Reason != readyReasonCacheIndexerNotReady {
		t.Fatalf("expected Available=false CacheIndexerNotReady, got %#v", avail)
	}
	if readyCond.Type != conditionTypeReady || readyCond.Status != metav1.ConditionFalse ||
		readyCond.Reason != readyReasonCacheIndexerNotReady {
		t.Fatalf("expected Ready=false CacheIndexerNotReady, got %#v", readyCond)
	}
}

func TestComputeInferNexServiceStatus_NotReadyWhenOnlyEnginePending(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	owner := "demo"
	replicas := int32(1)
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: owner + "-engine-aggregate", Namespace: "ns-a", Generation: 1},
		Spec:       appsv1.DeploymentSpec{Replicas: &replicas},
		Status: appsv1.DeploymentStatus{
			ObservedGeneration: 1,
			Replicas:           1,
			ReadyReplicas:      0,
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(dep).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: owner, Namespace: "ns-a", Generation: 3},
	}
	desired := map[string]componentPlan{
		"engine-aggregate": {WorkloadKind: workloadKindDeployment, Replicas: &replicas},
	}
	st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
	if err != nil {
		t.Fatalf("computeInferNexServiceStatus error: %v", err)
	}
	if st.Ready {
		t.Fatalf("expected not ready when only engine deployment has 0 ready pods, got %#v", st)
	}
	if st.Components == nil || st.Components.InferenceEngine == nil || st.Components.InferenceEngine.Ready {
		t.Fatalf("expected inferenceEngine not ready, got %#v", st.Components)
	}
	var readyCond metav1.Condition
	for _, c := range st.Conditions {
		if c.Type == conditionTypeReady {
			readyCond = c
		}
	}
	if readyCond.Reason != readyReasonInferenceBackendNotReady {
		t.Fatalf("expected Ready reason InferenceBackendNotReady, got %#v", readyCond)
	}
}

func findStatusCondition(conds []metav1.Condition, typ string) (metav1.Condition, bool) {
	for i := range conds {
		if conds[i].Type == typ {
			return conds[i], true
		}
	}
	return metav1.Condition{}, false
}

func readyDeployment(name, ns string, replicas int32) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns, Generation: 1},
		Spec:       appsv1.DeploymentSpec{Replicas: &replicas},
		Status: appsv1.DeploymentStatus{
			ObservedGeneration: 1,
			Replicas:           replicas,
			ReadyReplicas:      replicas,
		},
	}
}

func igrEnabledEffectiveSpec(gwName, routeName, poolName string) infernexv1alpha1.InferNexServiceSpec {
	enabledTrue := true
	return infernexv1alpha1.InferNexServiceSpec{
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router:        &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
			Gateway:       &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: gwName}},
			HTTPRoute:     &infernexv1alpha1.HTTPRouteRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: routeName}},
			InferencePool: &infernexv1alpha1.InferencePoolRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: poolName}},
		},
	}
}

func programmedGateway(name, ns string) *gwapiv1.Gateway {
	return &gwapiv1.Gateway{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Status: gwapiv1.GatewayStatus{Conditions: []metav1.Condition{
			{Type: "Accepted", Status: metav1.ConditionTrue},
			{Type: "Programmed", Status: metav1.ConditionTrue},
		}},
	}
}

func acceptedHTTPRoute(name, ns, gwName string) *gwapiv1.HTTPRoute {
	return &gwapiv1.HTTPRoute{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Status: gwapiv1.HTTPRouteStatus{
			RouteStatus: gwapiv1.RouteStatus{
				Parents: []gwapiv1.RouteParentStatus{{
					ParentRef: gwapiv1.ParentReference{Name: gwapiv1.ObjectName(gwName)},
					Conditions: []metav1.Condition{
						{Type: "Accepted", Status: metav1.ConditionTrue},
						{Type: "ResolvedRefs", Status: metav1.ConditionTrue},
					},
				}},
			},
		},
	}
}

func acceptedInferencePool(name, ns, gwName string) *igwapiv1.InferencePool {
	return &igwapiv1.InferencePool{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Status: igwapiv1.InferencePoolStatus{
			Parents: []igwapiv1.ParentStatus{{
				ParentRef: igwapiv1.ParentReference{Name: igwapiv1.ObjectName(gwName)},
				Conditions: []metav1.Condition{
					{Type: "Accepted", Status: metav1.ConditionTrue},
					{Type: "ResolvedRefs", Status: metav1.ConditionTrue},
				},
			}},
		},
	}
}

func TestGatewayRoutingRequired(t *testing.T) {
	t.Parallel()
	insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
	enabledTrue := true
	enabledFalse := false
	spec := infernexv1alpha1.InferNexServiceSpec{
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
		},
	}
	if !gatewayRoutingRequired(insvc, spec) {
		t.Fatal("expected gateway required for direct insvc with router enabled")
	}
	spec.IntelligentGatewayRouting.Router.Enabled = &enabledFalse
	if gatewayRoutingRequired(insvc, spec) {
		t.Fatal("expected gateway not required when router disabled")
	}
	linked := insvc.DeepCopy()
	linked.Spec.SourceRef = &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "llm"}
	if gatewayRoutingRequired(linked, spec) {
		t.Fatal("expected gateway not required for llmisvc path")
	}
}

func TestStatusForWorkloadFamily_SkipsZeroReplicas(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	zero := int32(0)
	desired := map[string]componentPlan{
		"cache-indexer": {Replicas: &zero, WorkloadKind: workloadKindDeployment},
	}
	cl := fake.NewClientBuilder().WithScheme(s).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	st, err := r.statusForWorkloadFamily(ctx, "ns-a", "demo", infernexComponentGroups.CacheIndexer, desired)
	if err != nil {
		t.Fatalf("statusForWorkloadFamily error: %v", err)
	}
	if st != nil {
		t.Fatalf("expected nil status when replicas=0, got %#v", st)
	}
}

func TestEvaluateInferNexServiceReady_WorkloadFailureBeforeGateway(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	enabledTrue := true
	gw := &gwapiv1.Gateway{
		ObjectMeta: metav1.ObjectMeta{Name: "gw", Namespace: "ns-a"},
		Status: gwapiv1.GatewayStatus{Conditions: []metav1.Condition{
			{Type: "Accepted", Status: metav1.ConditionTrue},
			{Type: "Programmed", Status: metav1.ConditionFalse},
		}},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(gw).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
	spec := infernexv1alpha1.InferNexServiceSpec{
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router:  &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
			Gateway: &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
		},
	}
	comp := &infernexv1alpha1.InferNexComponentStatuses{
		InferenceEngine: &infernexv1alpha1.ComponentStatus{Ready: false, Message: "engine-aggregate: ready pods 0/1"},
	}
	ready, reason, _, err := r.evaluateInferNexServiceReady(ctx, insvc, spec, comp)
	if err != nil {
		t.Fatalf("evaluateInferNexServiceReady error: %v", err)
	}
	if ready || reason != readyReasonInferenceBackendNotReady {
		t.Fatalf("expected workload failure first, got ready=%v reason=%q", ready, reason)
	}
}

func TestEvaluateInferNexServiceReady_GatewayFailureWhenWorkloadsReady(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	gw := &gwapiv1.Gateway{
		ObjectMeta: metav1.ObjectMeta{Name: "gw", Namespace: "ns-a"},
		Status: gwapiv1.GatewayStatus{Conditions: []metav1.Condition{
			{Type: "Accepted", Status: metav1.ConditionTrue},
			{Type: "Programmed", Status: metav1.ConditionFalse, Message: "waiting"},
		}},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(gw).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
	spec := igrEnabledEffectiveSpec("gw", "route", "pool")
	comp := &infernexv1alpha1.InferNexComponentStatuses{
		InferenceEngine: &infernexv1alpha1.ComponentStatus{Ready: true},
	}
	ready, reason, _, err := r.evaluateInferNexServiceReady(ctx, insvc, spec, comp)
	if err != nil {
		t.Fatalf("evaluateInferNexServiceReady error: %v", err)
	}
	if ready || reason != "GatewayNotProgrammed" {
		t.Fatalf("expected gateway failure, got ready=%v reason=%q", ready, reason)
	}
}

func TestComputeInferNexServiceStatus_ReadySemantics(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	replicas := int32(1)

	t.Run("direct all workloads and gateway ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", Generation: 5},
		}
		effective := igrEnabledEffectiveSpec("gw", "route", "pool")
		objects := []client.Object{
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
			programmedGateway("gw", "ns-a"),
			acceptedHTTPRoute("route", "ns-a", "gw"),
			acceptedInferencePool("pool", "ns-a", "gw"),
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(objects...).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, effective, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		if !st.Ready {
			t.Fatalf("expected ready=true, got %#v", st)
		}
		readyCond, ok := findStatusCondition(st.Conditions, conditionTypeReady)
		if !ok || readyCond.Reason != readyReasonAllManagedResourcesReady {
			t.Fatalf("expected Ready AllManagedResourcesReady, got %#v", readyCond)
		}
		avail, _ := findStatusCondition(st.Conditions, "Available")
		if avail.Reason != readyReasonAllManagedResourcesReady {
			t.Fatalf("expected Available reason aligned, got %#v", avail)
		}
		backend, _ := findStatusCondition(st.Conditions, "BackendReady")
		if backend.Status != metav1.ConditionTrue || backend.Reason != "BackendConfigured" {
			t.Fatalf("expected BackendReady true, got %#v", backend)
		}
	})

	t.Run("direct gateway not programmed blocks ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", Generation: 5},
		}
		effective := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:  &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(true)},
				Gateway: &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
			},
		}
		gw := programmedGateway("gw", "ns-a")
		gw.Status.Conditions[1].Status = metav1.ConditionFalse
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
			gw,
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, effective, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		if st.Ready {
			t.Fatalf("expected ready=false when gateway not programmed, got %#v", st)
		}
		readyCond, ok := findStatusCondition(st.Conditions, conditionTypeReady)
		if !ok || readyCond.Reason != "GatewayNotProgrammed" {
			t.Fatalf("expected Ready GatewayNotProgrammed, got %#v", readyCond)
		}
		components, _ := findStatusCondition(st.Conditions, "ComponentsReady")
		if components.Reason != "GatewayNotProgrammed" {
			t.Fatalf("expected ComponentsReady reason aligned with Ready, got %#v", components)
		}
	})

	t.Run("direct router disabled skips gateway check", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		}
		effective := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router: &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(false)},
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, effective, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		if !st.Ready {
			t.Fatalf("expected ready=true without gateway requirement, got %#v", st)
		}
		gwCond, _ := findStatusCondition(st.Conditions, "GatewayRoutingReady")
		if gwCond.Reason != "RouterDisabled" {
			t.Fatalf("expected GatewayRoutingReady RouterDisabled, got %#v", gwCond)
		}
	})

	t.Run("direct hermes router not ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		}
		dep := readyDeployment("demo-"+hermesRouterComponent, "ns-a", replicas)
		dep.Status.ReadyReplicas = 0
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(dep).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			hermesRouterComponent: {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		if st.Ready {
			t.Fatal("expected not ready when hermes-router pending")
		}
		readyCond, _ := findStatusCondition(st.Conditions, conditionTypeReady)
		if readyCond.Reason != readyReasonHermesRouterNotReady {
			t.Fatalf("expected HermesRouterNotReady, got %#v", readyCond)
		}
		if st.Components == nil || st.Components.HermesRouter == nil || st.Components.HermesRouter.Ready {
			t.Fatalf("expected hermesRouter component status not ready, got %#v", st.Components)
		}
	})

	t.Run("direct zero replica workload excluded from ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		}
		zero := int32(0)
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
			"cache-indexer":    {Replicas: &zero, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		if !st.Ready {
			t.Fatalf("expected ready=true when only zero-replica optional workload pending, got %#v", st)
		}
		if st.Components != nil && st.Components.CacheIndexer != nil {
			t.Fatalf("expected no cache-indexer status when replicas=0, got %#v", st.Components.CacheIndexer)
		}
	})

	t.Run("llmisvc proxy server not ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
			Spec: infernexv1alpha1.InferNexServiceSpec{
				SourceRef: &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "llm"},
			},
		}
		dep := readyDeployment("demo-proxy-server", "ns-a", replicas)
		dep.Status.ReadyReplicas = 0
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(dep).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"proxy-server": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "pd")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		if st.Ready {
			t.Fatal("expected not ready for llmisvc proxy pending")
		}
		readyCond, _ := findStatusCondition(st.Conditions, conditionTypeReady)
		if readyCond.Reason != readyReasonProxyServerNotReady {
			t.Fatalf("expected ProxyServerNotReady, got %#v", readyCond)
		}
		backend, _ := findStatusCondition(st.Conditions, "BackendReady")
		if backend.Reason != "SourceRefManagedByLLMInferenceService" {
			t.Fatalf("expected backend delegated on llmisvc, got %#v", backend)
		}
	})

	t.Run("llmisvc bridge ready ignores missing engine", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
			Spec: infernexv1alpha1.InferNexServiceSpec{
				SourceRef: &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "llm"},
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-proxy-server", "ns-a", replicas),
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"proxy-server": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "pd")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		if !st.Ready {
			t.Fatalf("expected ready when bridge proxy is 1/1 on llmisvc path, got %#v", st)
		}
		if st.Components != nil && st.Components.InferenceEngine != nil {
			t.Fatal("llmisvc path should not report inferenceEngine status")
		}
	})

	t.Run("direct engine ready sets backend ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		backend, _ := findStatusCondition(st.Conditions, "BackendReady")
		if backend.Status != metav1.ConditionTrue || backend.Reason != "BackendConfigured" {
			t.Fatalf("expected BackendReady configured, got %#v", backend)
		}
		if st.Components == nil || st.Components.InferenceEngine == nil || !st.Components.InferenceEngine.Ready {
			t.Fatalf("expected inferenceEngine ready in components, got %#v", st.Components)
		}
	})

	t.Run("deployment generation not observed", func(t *testing.T) {
		d := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: "demo-engine-aggregate", Namespace: "ns-a", Generation: 2},
			Spec:       appsv1.DeploymentSpec{Replicas: &replicas},
			Status: appsv1.DeploymentStatus{
				ObservedGeneration: 1,
				ReadyReplicas:      1,
				Replicas:           1,
			},
		}
		ok, msg := deploymentRolloutReady(d)
		if ok || !strings.Contains(msg, "observe latest spec") {
			t.Fatalf("expected not ready on generation mismatch, got ok=%v msg=%q", ok, msg)
		}
	})

	t.Run("deployment progressing false", func(t *testing.T) {
		d := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: "demo-engine-aggregate", Namespace: "ns-a", Generation: 1},
			Spec:       appsv1.DeploymentSpec{Replicas: &replicas},
			Status: appsv1.DeploymentStatus{
				ObservedGeneration: 1,
				ReadyReplicas:      1,
				Conditions: []appsv1.DeploymentCondition{
					{Type: appsv1.DeploymentProgressing, Status: corev1.ConditionFalse, Message: "crash loop"},
				},
			},
		}
		ok, msg := deploymentRolloutReady(d)
		if ok || msg != "crash loop" {
			t.Fatalf("expected progressing failure, got ok=%v msg=%q", ok, msg)
		}
	})

	// P0: direct engine ready cache-indexer not ready
	t.Run("direct cache indexer not ready aligned conditions", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		cacheDep := readyDeployment("demo-cache-indexer", "ns-a", replicas)
		cacheDep.Status.ReadyReplicas = 0
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
			cacheDep,
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
			"cache-indexer":    {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonCacheIndexerNotReady)
	})

	// P0: direct proxy-server not ready
	t.Run("direct proxy server not ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		dep := readyDeployment("demo-proxy-server", "ns-a", replicas)
		dep.Status.ReadyReplicas = 0
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(dep).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"proxy-server": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonProxyServerNotReady)
	})

	// P0: PD prefill ready decode not ready
	t.Run("direct pd decode not ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		prefill := readyDeployment("demo-engine-pd-prefill", "ns-a", replicas)
		decode := readyDeployment("demo-engine-pd-decode", "ns-a", replicas)
		decode.Status.ReadyReplicas = 0
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(prefill, decode).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-pd-prefill": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
			"engine-pd-decode":  {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "pd")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonInferenceBackendNotReady)
		if st.Components == nil || st.Components.InferenceEngine == nil ||
			!strings.Contains(st.Components.InferenceEngine.Message, "engine-pd-decode") {
			t.Fatalf("expected decode key in inferenceEngine message, got %#v", st.Components)
		}
	})

	// P1: gateway HTTPRouteNotFound on total Ready
	t.Run("direct ready reason HTTPRouteNotFound", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		effective := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:    &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(true)},
				Gateway:   &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
				HTTPRoute: &infernexv1alpha1.HTTPRouteRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "missing-route"}},
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
			programmedGateway("gw", "ns-a"),
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, effective, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, "HTTPRouteNotFound")
	})

	// P1: gateway InferencePoolNoEndpoints on total Ready
	t.Run("direct ready reason InferencePoolNoEndpoints", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		effective := igrEnabledEffectiveSpec("gw", "route", "pool")
		pool := &igwapiv1.InferencePool{
			ObjectMeta: metav1.ObjectMeta{Name: "pool", Namespace: "ns-a"},
			Spec: igwapiv1.InferencePoolSpec{
				Selector: igwapiv1.LabelSelector{MatchLabels: map[igwapiv1.LabelKey]igwapiv1.LabelValue{"app": "engine"}},
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
			programmedGateway("gw", "ns-a"),
			acceptedHTTPRoute("route", "ns-a", "gw"),
			pool,
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, effective, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, "InferencePoolNoEndpoints")
	})

	// P1: deployment not found
	t.Run("direct cache deployment not found", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		cl := fake.NewClientBuilder().WithScheme(s).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"cache-indexer": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonCacheIndexerNotReady)
		if st.Components == nil || st.Components.CacheIndexer == nil ||
			!strings.Contains(st.Components.CacheIndexer.Message, "not found") {
			t.Fatalf("expected not found in message, got %#v", st.Components)
		}
	})

	t.Run("direct daemonset generation not observed", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		ds := &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Name: "demo-cache-indexer", Namespace: "ns-a", Generation: 2},
			Status: appsv1.DaemonSetStatus{
				ObservedGeneration:     1,
				DesiredNumberScheduled: 2,
				NumberReady:            2,
				UpdatedNumberScheduled: 2,
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(ds).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"cache-indexer": {Replicas: &replicas, WorkloadKind: workloadKindDaemonSet},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonCacheIndexerNotReady)
		if st.Components == nil || st.Components.CacheIndexer == nil ||
			!strings.Contains(st.Components.CacheIndexer.Message, "observe latest spec") {
			t.Fatalf("expected generation mismatch in cache-indexer message, got %#v", st.Components)
		}
	})

	t.Run("direct daemonset rollout not updated", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		ds := &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Name: "demo-cache-indexer", Namespace: "ns-a", Generation: 1},
			Status: appsv1.DaemonSetStatus{
				ObservedGeneration:     1,
				DesiredNumberScheduled: 3,
				NumberReady:            3,
				UpdatedNumberScheduled: 1,
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(ds).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"cache-indexer": {Replicas: &replicas, WorkloadKind: workloadKindDaemonSet},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonCacheIndexerNotReady)
		if st.Components == nil || st.Components.CacheIndexer == nil ||
			!strings.Contains(st.Components.CacheIndexer.Message, "updated pods") {
			t.Fatalf("expected updated pods message, got %#v", st.Components)
		}
	})

	// P2: daemonset cache-indexer not ready
	t.Run("direct daemonset cache indexer not ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		ds := &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Name: "demo-cache-indexer", Namespace: "ns-a", Generation: 1},
			Status: appsv1.DaemonSetStatus{
				ObservedGeneration:     1,
				DesiredNumberScheduled: 2,
				NumberReady:            1,
				UpdatedNumberScheduled: 2,
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(ds).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"cache-indexer": {Replicas: &replicas, WorkloadKind: workloadKindDaemonSet},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonCacheIndexerNotReady)
		if st.Components == nil || st.Components.CacheIndexer == nil ||
			!strings.Contains(st.Components.CacheIndexer.Message, "ready pods") {
			t.Fatalf("expected daemonset ready pods message, got %#v", st.Components)
		}
	})

	// P2: direct empty desired vacuous ready
	t.Run("direct no managed workloads vacuous ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		cl := fake.NewClientBuilder().WithScheme(s).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, map[string]componentPlan{}, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, true, readyReasonAllManagedResourcesReady)
		backend, _ := findStatusCondition(st.Conditions, "BackendReady")
		if backend.Status != metav1.ConditionFalse || backend.Reason != "BackendNotConfigured" {
			t.Fatalf("expected BackendNotConfigured without engine, got %#v", backend)
		}
	})

	// P2: engine and proxy both not ready engine reason wins
	t.Run("readiness check order engine before proxy", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		engine := readyDeployment("demo-engine-aggregate", "ns-a", replicas)
		engine.Status.ReadyReplicas = 0
		proxy := readyDeployment("demo-proxy-server", "ns-a", replicas)
		proxy.Status.ReadyReplicas = 0
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(engine, proxy).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
			"proxy-server":     {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonInferenceBackendNotReady)
	})

	t.Run("readiness check order proxy before cache", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		proxy := readyDeployment("demo-proxy-server", "ns-a", replicas)
		proxy.Status.ReadyReplicas = 0
		cache := readyDeployment("demo-cache-indexer", "ns-a", replicas)
		cache.Status.ReadyReplicas = 0
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(proxy, cache).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"proxy-server":  {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
			"cache-indexer": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonProxyServerNotReady)
	})

	t.Run("mooncake not ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		master := readyDeployment("mooncake-master-deployment", "ns-a", replicas)
		master.Status.ReadyReplicas = 0
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(master).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"mooncake-master": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonMooncakeNotReady)
	})

	t.Run("pd orchestrator not ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		dep := readyDeployment("demo-pd-orchestrator-elastic-scaler", "ns-a", replicas)
		dep.Status.ReadyReplicas = 0
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(dep).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"pd-orchestrator-elastic-scaler": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "pd")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonPDOrchestratorNotReady)
	})

	t.Run("eagle eye not ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		dep := readyDeployment("demo-eagle-eye-hardware-monitor", "ns-a", replicas)
		dep.Status.ReadyReplicas = 0
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(dep).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"eagle-eye-hardware-monitor": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonEagleEyeNotReady)
	})

	t.Run("engine nil replicas still checked", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		dep := readyDeployment("demo-engine-aggregate", "ns-a", replicas)
		dep.Status.ReadyReplicas = 0
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(dep).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, readyReasonInferenceBackendNotReady)
	})

	t.Run("direct ready reason InferencePoolNotAccepted", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		effective := igrEnabledEffectiveSpec("gw", "route", "pool")
		pool := acceptedInferencePool("pool", "ns-a", "gw")
		pool.Status.Parents[0].Conditions[0].Status = metav1.ConditionFalse
		pool.Status.Parents[0].Conditions[0].Message = "selector invalid"
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
			programmedGateway("gw", "ns-a"),
			acceptedHTTPRoute("route", "ns-a", "gw"),
			pool,
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, effective, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, "InferencePoolNotAccepted")
	})

	t.Run("direct ready reason InferencePoolRefsUnresolved", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		effective := igrEnabledEffectiveSpec("gw", "route", "pool")
		pool := acceptedInferencePool("pool", "ns-a", "gw")
		pool.Status.Parents[0].Conditions = []metav1.Condition{
			{Type: "Accepted", Status: metav1.ConditionTrue},
			{Type: "ResolvedRefs", Status: metav1.ConditionFalse, Message: "backend ref missing"},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
			programmedGateway("gw", "ns-a"),
			acceptedHTTPRoute("route", "ns-a", "gw"),
			pool,
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, effective, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, "InferencePoolRefsUnresolved")
	})

	t.Run("direct ready reason HTTPRouteNotAccepted", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		effective := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:    &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(true)},
				Gateway:   &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "gw"}},
				HTTPRoute: &infernexv1alpha1.HTTPRouteRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "route"}},
			},
		}
		route := acceptedHTTPRoute("route", "ns-a", "gw")
		route.Status.Parents[0].Conditions[0].Status = metav1.ConditionFalse
		route.Status.Parents[0].Conditions[0].Message = "backend missing"
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
			programmedGateway("gw", "ns-a"),
			route,
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, effective, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, "HTTPRouteNotAccepted")
	})

	t.Run("gateway GatewayNotFound on total Ready", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		effective := infernexv1alpha1.InferNexServiceSpec{
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router:  &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(true)},
				Gateway: &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "missing-gw"}},
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
			readyDeployment("demo-engine-aggregate", "ns-a", replicas),
		).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		desired := map[string]componentPlan{
			"engine-aggregate": {Replicas: &replicas, WorkloadKind: workloadKindDeployment},
		}
		st, err := r.computeInferNexServiceStatus(ctx, insvc, desired, effective, "aggregate")
		if err != nil {
			t.Fatalf("computeInferNexServiceStatus error: %v", err)
		}
		assertReadySemantics(t, st, false, "GatewayNotFound")
	})
}

func assertReadySemantics(t *testing.T, st infernexv1alpha1.InferNexServiceStatus, wantReady bool, wantReason string) {
	t.Helper()
	if st.Ready != wantReady {
		t.Fatalf("status.ready=%v want %v", st.Ready, wantReady)
	}
	readyCond, ok := findStatusCondition(st.Conditions, conditionTypeReady)
	if !ok {
		t.Fatal("missing Ready condition")
	}
	if wantReady {
		if readyCond.Status != metav1.ConditionTrue || readyCond.Reason != wantReason {
			t.Fatalf("Ready condition %#v want true reason %q", readyCond, wantReason)
		}
		return
	}
	if readyCond.Status != metav1.ConditionFalse || readyCond.Reason != wantReason {
		t.Fatalf("Ready condition %#v want false reason %q", readyCond, wantReason)
	}
	avail, ok := findStatusCondition(st.Conditions, "Available")
	if !ok || avail.Reason != wantReason {
		t.Fatalf("Available %#v want reason %q", avail, wantReason)
	}
	compReady, ok := findStatusCondition(st.Conditions, "ComponentsReady")
	if !ok || compReady.Reason != wantReason {
		t.Fatalf("ComponentsReady %#v want reason %q", compReady, wantReason)
	}
}
