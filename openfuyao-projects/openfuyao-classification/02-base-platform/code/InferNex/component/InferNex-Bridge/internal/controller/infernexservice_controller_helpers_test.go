package controller

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestMigrateLegacySingletonServiceOwnerRef_PreservesService(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", UID: types.UID("uid-demo")},
	}
	svc := &corev1.Service{ObjectMeta: metav1.ObjectMeta{
		Name:      "redis-service",
		Namespace: "ns-a",
		Labels: map[string]string{
			"app.kubernetes.io/name":      "infernex-component",
			"infernex.io/component":       "mooncake-metadata",
			"infernex.io/managed-service": "true",
		},
		OwnerReferences: []metav1.OwnerReference{{
			APIVersion: "infernex.io/v1alpha1", Kind: "InferNexService", Name: owner.Name, UID: owner.UID,
		}},
	}}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner, svc).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	if err := r.migrateLegacySingletonServices(ctx, owner.Namespace); err != nil {
		t.Fatalf("migrateLegacySingletonServices error: %v", err)
	}
	got := &corev1.Service{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: "redis-service"}, got); err != nil {
		t.Fatalf("legacy singleton Service should remain: %v", err)
	}
	if len(got.OwnerReferences) != 0 {
		t.Fatalf("expected legacy owner reference removed, got %#v", got.OwnerReferences)
	}
	if got.Labels["infernex.io/singleton"] != "mooncake" ||
		got.Labels["infernex.io/component"] != "mooncake-metadata" ||
		got.Labels["infernex.io/managed-service"] != "true" {
		t.Fatalf("expected singleton labels migrated, got %#v", got.Labels)
	}
	if got.Spec.Selector["app"] != "redis" {
		t.Fatalf("expected singleton selector migrated to redis, got %#v", got.Spec.Selector)
	}
}

func TestMooncakeSingletonWatchHelpers(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	ownerA := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "isvc-a", Namespace: "ns-a"}}
	ownerB := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "isvc-b", Namespace: "ns-a"}}
	otherNamespace := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "isvc-c", Namespace: "ns-b"}}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(ownerA, ownerB, otherNamespace).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	cases := []struct {
		name string
		obj  client.Object
		want bool
	}{
		{name: "canonical service", obj: &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: "redis-service", Namespace: "ns-a"}}, want: true},
		{name: "canonical deployment", obj: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "redis-server-deployment", Namespace: "ns-a"}}, want: true},
		{name: "singleton label", obj: &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: "custom-name", Namespace: "ns-a", Labels: map[string]string{"infernex.io/singleton": "mooncake"}}}, want: true},
		{name: "unrelated service", obj: &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: "other-service", Namespace: "ns-a"}}, want: false},
	}
	for _, tc := range cases {
		if got := isMooncakeSingletonObject(tc.obj); got != tc.want {
			t.Errorf("isMooncakeSingletonObject(%s) = %v, want %v", tc.name, got, tc.want)
		}
	}

	requests := r.enqueueFromMooncakeSingleton(ctx, &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: "redis-service", Namespace: "ns-a"}})
	if len(requests) != 2 {
		t.Fatalf("expected two requests for namespace ns-a, got %#v", requests)
	}
	for _, req := range requests {
		if req.Namespace != "ns-a" || (req.Name != ownerA.Name && req.Name != ownerB.Name) {
			t.Fatalf("unexpected singleton enqueue request: %#v", req)
		}
	}
}

func TestReconcileInferNexServiceDeletion_RemovesFinalizer(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "demo",
			Namespace:  "ns-a",
			Finalizers: []string{infernexServiceFinalizer},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	res, err := r.reconcileInferNexServiceDeletion(ctx, insvc)
	if err != nil {
		t.Fatalf("reconcileInferNexServiceDeletion error: %v", err)
	}
	if res.Requeue {
		t.Fatalf("unexpected requeue: %#v", res)
	}
	fresh := &infernexv1alpha1.InferNexService{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo"}, fresh); err != nil {
		t.Fatalf("get updated infernexservice failed: %v", err)
	}
	if len(fresh.Finalizers) != 0 {
		t.Fatalf("expected finalizers removed, got %v", fresh.Finalizers)
	}
}

func TestReconcileInferNexServiceDeletion_ReleasesSingletonOwnerRefs(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	ownerA := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "isvc-a",
			Namespace:  "ns-a",
			UID:        types.UID("uid-a"),
			Finalizers: []string{infernexServiceFinalizer},
		},
	}
	ownerB := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "isvc-b",
			Namespace:  "ns-a",
			UID:        types.UID("uid-b"),
			Finalizers: []string{infernexServiceFinalizer},
		},
	}
	refs := []metav1.OwnerReference{
		{APIVersion: "infernex.io/v1alpha1", Kind: "InferNexService", Name: ownerA.Name, UID: ownerA.UID},
		{APIVersion: "infernex.io/v1alpha1", Kind: "InferNexService", Name: ownerB.Name, UID: ownerB.UID},
	}
	// Mooncake Services are namespace singletons without InferNexService owner
	// references; only the shared component ServiceAccount is released here.
	svc := &corev1.Service{ObjectMeta: metav1.ObjectMeta{
		Name:      "redis-service",
		Namespace: "ns-a",
	}}
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{
		Name:            componentControllerSAName(cacheIndexerComponent),
		Namespace:       "ns-a",
		OwnerReferences: refs,
	}}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(ownerA, ownerB, svc, sa).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	if _, err := r.reconcileInferNexServiceDeletion(ctx, ownerA); err != nil {
		t.Fatalf("reconcileInferNexServiceDeletion(%s) error: %v", ownerA.Name, err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "redis-service"}, svc); err != nil {
		t.Fatalf("expected singleton service to remain: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: componentControllerSAName(cacheIndexerComponent)}, sa); err != nil {
		t.Fatalf("expected singleton serviceaccount after first owner deletion: %v", err)
	}
	assertMissingOwnerRef(t, sa.OwnerReferences, ownerA.Name, ownerA.UID)
	assertNonControllerOwnerRef(t, sa.OwnerReferences, ownerB.Name, ownerB.UID)

	if _, err := r.reconcileInferNexServiceDeletion(ctx, ownerB); err != nil {
		t.Fatalf("reconcileInferNexServiceDeletion(%s) error: %v", ownerB.Name, err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "redis-service"}, svc); err != nil {
		t.Fatalf("singleton service should remain after all InferNexServices are deleted: %v", err)
	}
	mustNotFound(t, cl, types.NamespacedName{Namespace: "ns-a", Name: componentControllerSAName(cacheIndexerComponent)}, &corev1.ServiceAccount{})
}

func TestReconcileFetchInferNexService(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := controllerTestScheme(t)
	req := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}}

	t.Run("returns existing object", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		got, err := r.reconcileFetchInferNexService(ctx, req)
		if err != nil {
			t.Fatalf("reconcileFetchInferNexService error: %v", err)
		}
		if got == nil || got.Name != "demo" {
			t.Fatalf("expected fetched object, got %#v", got)
		}
	})

	t.Run("notfound triggers llm build path", func(t *testing.T) {
		llm := newLinkedLLM("ns-a", "demo", infernexRuntimeValue)
		base := fake.NewClientBuilder().WithScheme(s).WithObjects(llm).Build()
		cl := &llmAwareClient{
			Client: base,
			llms:   map[types.NamespacedName]*unstructured.Unstructured{{Namespace: "ns-a", Name: "demo"}: llm},
		}
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		got, err := r.reconcileFetchInferNexService(ctx, req)
		if err != nil {
			t.Fatalf("reconcileFetchInferNexService build-from-llm error: %v", err)
		}
		if got != nil {
			t.Fatalf("expected nil result on create-on-miss path, got %#v", got)
		}
		created := &infernexv1alpha1.InferNexService{}
		if err := base.Get(ctx, req.NamespacedName, created); err != nil {
			t.Fatalf("expected infernexservice created from llm: %v", err)
		}
	})
}

func TestEmitGatewayRoutingConditionEvent(t *testing.T) {
	t.Parallel()
	rec := record.NewFakeRecorder(10)
	r := &InferNexServiceReconciler{Recorder: rec}
	obj := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}

	r.emitGatewayRoutingConditionEvent(obj, nil, []metav1.Condition{{
		Type:    "GatewayRoutingReady",
		Status:  metav1.ConditionFalse,
		Reason:  "GatewayNotFound",
		Message: "gateway ns-a/gw not found",
	}})
	select {
	case e := <-rec.Events:
		if e == "" {
			t.Fatal("expected warning event content")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for warning event")
	}

	r.emitGatewayRoutingConditionEvent(obj, nil, []metav1.Condition{{
		Type:    "GatewayRoutingReady",
		Status:  metav1.ConditionTrue,
		Reason:  "GatewayRoutingReady",
		Message: "gateway routing linked",
	}})
	select {
	case e := <-rec.Events:
		if e == "" {
			t.Fatal("expected normal event content")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for normal event")
	}

	// unchanged condition should not emit a new event.
	old := []metav1.Condition{{Type: "GatewayRoutingReady", Status: metav1.ConditionTrue, Reason: "GatewayRoutingReady", Message: "same"}}
	r.emitGatewayRoutingConditionEvent(obj, old, old)
	select {
	case <-rec.Events:
		t.Fatal("unexpected event for unchanged condition")
	case <-time.After(200 * time.Millisecond):
	}
}

func TestRecordWarn_NoPanicPaths(t *testing.T) {
	t.Parallel()
	r := &InferNexServiceReconciler{}
	r.recordWarn(nil, "X", "msg")
	r.recordNormal(nil, "X", "msg")
	r.Recorder = record.NewFakeRecorder(4)
	obj := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
	r.recordWarn(obj, "Warn", "hello %s", "x")
	r.recordNormal(obj, "Normal", "hello")
}

func TestReconcile_AddsFinalizerAndRequeues(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", Generation: 1},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "llm"},
		},
	}
	llm := newLinkedLLM("ns-a", "llm", infernexRuntimeValue)
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc, llm).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}

	res, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}})
	if err != nil {
		t.Fatalf("Reconcile add-finalizer error: %v", err)
	}
	if !res.Requeue {
		t.Fatalf("expected requeue after adding finalizer, got %#v", res)
	}
	fresh := &infernexv1alpha1.InferNexService{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo"}, fresh); err != nil {
		t.Fatalf("get infernexservice: %v", err)
	}
	if !containsString(fresh.Finalizers, infernexServiceFinalizer) {
		t.Fatalf("expected finalizer added, got %v", fresh.Finalizers)
	}
}

func TestReconcilePersistStatus_UpdatesStatus(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", Generation: 3},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "llm"},
		},
	}
	cl := fake.NewClientBuilder().
		WithScheme(s).
		WithStatusSubresource(&infernexv1alpha1.InferNexService{}).
		WithObjects(insvc).
		Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, Recorder: record.NewFakeRecorder(4)}

	res, err := r.reconcilePersistStatus(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}}, map[string]componentPlan{}, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
	if err != nil {
		t.Fatalf("reconcilePersistStatus error: %v", err)
	}
	if res.Requeue {
		t.Fatalf("unexpected requeue: %#v", res)
	}
	fresh := &infernexv1alpha1.InferNexService{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo"}, fresh); err != nil {
		t.Fatalf("get infernexservice: %v", err)
	}
	if fresh.Status.Mode != "aggregate" || fresh.Status.ObservedGeneration != 3 {
		t.Fatalf("unexpected status after persist: %#v", fresh.Status)
	}
	if len(fresh.Status.Conditions) == 0 {
		t.Fatal("expected conditions written to status")
	}
}

func containsString(list []string, target string) bool {
	for _, s := range list {
		if s == target {
			return true
		}
	}
	return false
}

// --- Tests for ErrTemplateNotFound sentinel & Reconcile error classification ---

// stubReconcilerForReconcile builds an InferNexServiceReconciler with a fake client
// and scheme, suitable for calling Reconcile directly in unit tests.
func stubReconcilerForReconcile(t *testing.T, objs ...client.Object) *InferNexServiceReconciler {
	t.Helper()
	s := gatewayTestScheme(t)
	return &InferNexServiceReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(s).
			WithStatusSubresource(&infernexv1alpha1.InferNexService{}).
			WithObjects(objs...).
			Build(),
		Scheme:            s,
		TemplateNamespace: "template-ns",
		Recorder:          record.NewFakeRecorder(8),
	}
}

func TestReconcile_TemplateNotFound_StopsRetryAndWritesStatus(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "demo",
			Namespace:  "ns-a",
			Finalizers: []string{infernexServiceFinalizer},
		},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			Engine: &infernexv1alpha1.InferenceEngineSpec{
				InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
					Template: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{Name: "main", Image: "busybox"}},
						},
					},
				},
			},
		},
	}
	r := stubReconcilerForReconcile(t, insvc)

	res, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}})
	if err != nil {
		t.Fatalf("expected no error (should stop retry), got: %v", err)
	}
	if res.Requeue {
		t.Fatalf("expected no requeue, got: %#v", res)
	}

	fresh := &infernexv1alpha1.InferNexService{}
	if getErr := r.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo"}, fresh); getErr != nil {
		t.Fatalf("get infernexservice: %v", getErr)
	}
	if fresh.Status.Ready {
		t.Fatal("expected Ready=false")
	}
	readyCond := apimeta.FindStatusCondition(fresh.Status.Conditions, "Ready")
	if readyCond == nil {
		t.Fatal("expected Ready condition in status")
	}
	if readyCond.Status != metav1.ConditionFalse {
		t.Fatalf("expected Ready=False, got %s", readyCond.Status)
	}
	if readyCond.Reason != "TemplateNotFound" {
		t.Fatalf("expected Reason=TemplateNotFound, got %s", readyCond.Reason)
	}
	if !strings.Contains(readyCond.Message, ErrTemplateNotFound.Error()) {
		t.Fatalf("expected Message to contain %q, got: %s", ErrTemplateNotFound.Error(), readyCond.Message)
	}
}

func TestReconcile_NonTemplateNotFound_StillReturnsError(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "demo",
			Namespace:  "ns-a",
			Finalizers: []string{infernexServiceFinalizer},
		},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			Engine: &infernexv1alpha1.InferenceEngineSpec{
				InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
					Template: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{Name: "main", Image: "busybox"}},
						},
					},
				},
			},
		},
	}
	s := gatewayTestScheme(t)
	r := &InferNexServiceReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(s).
			WithStatusSubresource(&infernexv1alpha1.InferNexService{}).
			WithObjects(insvc).
			Build(),
		Scheme:            s,
		TemplateNamespace: "", // empty → non-ErrTemplateNotFound error
		Recorder:          record.NewFakeRecorder(8),
	}

	res, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}})
	if err == nil {
		t.Fatal("expected error for non-template NotFound, got nil")
	}
	if errors.Is(err, ErrTemplateNotFound) {
		t.Fatal("error must not be ErrTemplateNotFound for non-template path")
	}
	_ = res
}

func TestReconcile_EmptyTemplateNamespace_ReturnsError(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "demo",
			Namespace:  "ns-a",
			Finalizers: []string{infernexServiceFinalizer},
		},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			Engine: &infernexv1alpha1.InferenceEngineSpec{
				InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
					Template: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{Name: "main", Image: "busybox"}},
						},
					},
				},
			},
		},
	}
	s := gatewayTestScheme(t)
	r := &InferNexServiceReconciler{
		Client:            fake.NewClientBuilder().WithScheme(s).WithObjects(insvc).Build(),
		Scheme:            s,
		TemplateNamespace: "", // empty
		Recorder:          record.NewFakeRecorder(8),
	}

	res, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}})
	if err == nil {
		t.Fatal("expected error when template namespace is empty")
	}
	if errors.Is(err, ErrTemplateNotFound) {
		t.Fatal("empty namespace error must not be ErrTemplateNotFound")
	}
	if res.Requeue {
		t.Fatalf("expected no requeue on error return, got: %#v", res)
	}
}

// --- Tests for svcRefersToConfig ---

func TestSvcRefersToConfig_BaseRefsMatch(t *testing.T) {
	t.Parallel()
	svc := &infernexv1alpha1.InferNexService{
		Spec: infernexv1alpha1.InferNexServiceSpec{
			BaseRefs: []infernexv1alpha1.NamedRef{
				{Name: " my-template "},
				{Name: "other"},
			},
		},
	}
	if !svcRefersToConfig(svc, "my-template") {
		t.Fatal("expected svc to refer to 'my-template' via baseRefs")
	}
	if svcRefersToConfig(svc, "unknown") {
		t.Fatal("expected svc NOT to refer to 'unknown'")
	}
}

func TestSvcRefersToConfig_DefaultTemplates(t *testing.T) {
	t.Parallel()
	svc := &infernexv1alpha1.InferNexService{
		Spec: infernexv1alpha1.InferNexServiceSpec{},
	}
	if !svcRefersToConfig(svc, defaultAggregateTemplateName) {
		t.Fatalf("expected svc to refer to default aggregate template %q", defaultAggregateTemplateName)
	}
	if !svcRefersToConfig(svc, defaultPDTemplateName) {
		t.Fatalf("expected svc to refer to default PD template %q", defaultPDTemplateName)
	}
	if svcRefersToConfig(svc, "some-other-config") {
		t.Fatal("expected svc NOT to refer to arbitrary config when baseRefs is empty")
	}
}

func TestSvcRefersToConfig_NilAndEmpty(t *testing.T) {
	t.Parallel()
	if svcRefersToConfig(nil, "x") {
		t.Fatal("nil svc should not refer to anything")
	}
	if svcRefersToConfig(&infernexv1alpha1.InferNexService{}, "") {
		t.Fatal("empty cfgName should not match")
	}
}

// --- Tests for enqueueFromConfig ---

func TestEnqueueFromConfig_WrongNamespace(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	cl := fake.NewClientBuilder().WithScheme(s).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "template-ns"}

	cfg := &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "my-config", Namespace: "other-ns"},
	}
	reqs := r.enqueueFromConfig(ctx, cfg)
	if len(reqs) != 0 {
		t.Fatalf("expected no requests for config in wrong namespace, got %d", len(reqs))
	}
}

func TestEnqueueFromConfig_MatchesBaseRefs(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	svcA := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "svc-a", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			BaseRefs: []infernexv1alpha1.NamedRef{{Name: "my-config"}},
		},
	}
	svcB := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "svc-b", Namespace: "ns-b"},
		Spec:       infernexv1alpha1.InferNexServiceSpec{},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(svcA, svcB).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "template-ns"}

	cfg := &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "my-config", Namespace: "template-ns"},
	}
	reqs := r.enqueueFromConfig(ctx, cfg)
	if len(reqs) != 1 {
		t.Fatalf("expected 1 request, got %d", len(reqs))
	}
	if reqs[0].Name != "svc-a" || reqs[0].Namespace != "ns-a" {
		t.Fatalf("expected svc-a request, got %s/%s", reqs[0].Namespace, reqs[0].Name)
	}
}

func TestEnqueueFromConfig_MatchesDefaultTemplate(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	svc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "svc-d", Namespace: "ns-d"},
		Spec:       infernexv1alpha1.InferNexServiceSpec{},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(svc).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "template-ns"}

	cfg := &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: defaultAggregateTemplateName, Namespace: "template-ns"},
	}
	reqs := r.enqueueFromConfig(ctx, cfg)
	if len(reqs) != 1 {
		t.Fatalf("expected 1 request for default template, got %d", len(reqs))
	}
	if reqs[0].Name != "svc-d" {
		t.Fatalf("expected svc-d, got %s", reqs[0].Name)
	}
}
