package controller

import (
	"context"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	ctrl "sigs.k8s.io/controller-runtime"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"
	lwsv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func reconcileTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := rbacTestScheme(t)
	gvk := schema.GroupVersionKind{Group: "serving.kserve.io", Version: "v1alpha1", Kind: "LLMInferenceService"}
	s.AddKnownTypeWithName(gvk, &unstructured.Unstructured{})
	s.AddKnownTypeWithName(gvk.GroupVersion().WithKind("LLMInferenceServiceList"), &unstructured.UnstructuredList{})
	return s
}

func platformAggregateConfig() *infernexv1alpha1.InferNexServiceConfig {
	return &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: defaultAggregateTemplateName, Namespace: "tpl-ns"},
		Spec:       infernexv1alpha1.InferNexServiceConfigSpec{},
	}
}

func platformPDConfig() *infernexv1alpha1.InferNexServiceConfig {
	return &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: defaultPDTemplateName, Namespace: "tpl-ns"},
		Spec:       infernexv1alpha1.InferNexServiceConfigSpec{},
	}
}

func TestReconcileWorkloadsAndRBAC_DirectCacheIndexer(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			Components: &infernexv1alpha1.InfernexComponentsSpec{
				CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{Enabled: ptr.To(true), Replicas: 1},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc, platformAggregateConfig()).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}

	desired, _, mode, err := r.reconcileWorkloadsAndRBAC(ctx, insvc)
	if err != nil {
		t.Fatalf("reconcileWorkloadsAndRBAC error: %v", err)
	}
	if mode != "aggregate" {
		t.Fatalf("expected aggregate mode, got %q", mode)
	}
	if _, ok := desired[cacheIndexerComponent]; !ok {
		t.Fatalf("expected cache-indexer in desired components, got %#v", desired)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-cache-indexer"}, &appsv1.Deployment{}); err != nil {
		t.Fatalf("expected cache-indexer deployment: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-cache-indexer"}, &corev1.Service{}); err != nil {
		t.Fatalf("expected cache-indexer service: %v", err)
	}
}

func TestReconcileWorkloadsAndRBAC_DirectPDLaunchesProxy(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			Engine: testPDEngineSpec(),
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc, platformPDConfig()).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}

	desired, _, mode, err := r.reconcileWorkloadsAndRBAC(ctx, insvc)
	if err != nil {
		t.Fatalf("reconcileWorkloadsAndRBAC pd error: %v", err)
	}
	if mode != "pd" {
		t.Fatalf("expected pd mode, got %q", mode)
	}
	for _, comp := range []string{"engine-pd-prefill", "engine-pd-decode", "proxy-server"} {
		if _, ok := desired[comp]; !ok {
			t.Fatalf("expected %s in desired, got %#v", comp, desired)
		}
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-workload-svc"}, &corev1.Service{}); err != nil {
		t.Fatalf("expected shared pd workload service: %v", err)
	}
}

func TestReconcileWorkloadsAndRBAC_LinkedStopReleasesSidecars(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := reconcileTestScheme(t)
	llm := newLinkedLLM("ns-a", "demo", infernexRuntimeValue)
	insvc := newLinkedInferNexService("ns-a", "demo")
	insvc.Spec.Components = &infernexv1alpha1.InfernexComponentsSpec{
		CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{Enabled: ptr.To(true), Replicas: 1},
	}
	base := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc, platformAggregateConfig()).Build()
	cl := &llmAwareClient{
		Client: base,
		llms: map[types.NamespacedName]*unstructured.Unstructured{
			{Namespace: "ns-a", Name: "demo"}: llm,
		},
	}
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}

	if _, _, _, err := r.reconcileWorkloadsAndRBAC(ctx, insvc); err != nil {
		t.Fatalf("initial reconcile: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-cache-indexer"}, &appsv1.Deployment{}); err != nil {
		t.Fatalf("expected cache-indexer deployment: %v", err)
	}

	llm.SetAnnotations(map[string]string{kserveStopAnnotationKey: "true"})
	if _, _, _, err := r.reconcileWorkloadsAndRBAC(ctx, insvc); err != nil {
		t.Fatalf("stop reconcile: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-cache-indexer"}, &appsv1.Deployment{}); !apierrors.IsNotFound(err) {
		t.Fatalf("expected cache-indexer deployment deleted on stop, got err=%v", err)
	}
}

func TestReconcileWorkloadsAndRBAC_LinkedDisabledComponentsReleasesSidecar(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := reconcileTestScheme(t)
	llm := newLinkedLLM("ns-a", "demo", infernexRuntimeValue)
	insvc := newLinkedInferNexService("ns-a", "demo")
	insvc.Spec.Components = &infernexv1alpha1.InfernexComponentsSpec{
		CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{Enabled: ptr.To(true), Replicas: 1},
	}
	base := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc, platformAggregateConfig()).Build()
	cl := &llmAwareClient{
		Client: base,
		llms: map[types.NamespacedName]*unstructured.Unstructured{
			{Namespace: "ns-a", Name: "demo"}: llm,
		},
	}
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}

	if _, _, _, err := r.reconcileWorkloadsAndRBAC(ctx, insvc); err != nil {
		t.Fatalf("initial reconcile: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-cache-indexer"}, &appsv1.Deployment{}); err != nil {
		t.Fatalf("expected cache-indexer deployment: %v", err)
	}

	llm.SetAnnotations(map[string]string{infernexDisabledComponentsAnnotationKey: cacheIndexerComponent})
	if _, _, _, err := r.reconcileWorkloadsAndRBAC(ctx, insvc); err != nil {
		t.Fatalf("disabled reconcile: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-cache-indexer"}, &appsv1.Deployment{}); !apierrors.IsNotFound(err) {
		t.Fatalf("expected cache-indexer deployment deleted when disabled via annotation, got err=%v", err)
	}
	// Singleton SA is shared: disable should prune this owner's ref, not break reconcile.
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: componentControllerSAName(cacheIndexerComponent)}, &corev1.ServiceAccount{}); err != nil {
		t.Fatalf("expected singleton cache-indexer SA removed after last owner disabled: %v", err)
	}
}

func TestReconcileWorkloadsAndRBAC_DirectLWSStopNotViaAnnotation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			Engine: &infernexv1alpha1.InferenceEngineSpec{
				InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
					Template:              testTemplate("engine:v1", 8000),
					DataParallelSize:      ptr.To(int32(2)),
					DataParallelSizeLocal: ptr.To(int32(1)),
				},
			},
		},
	}
	insvc.Annotations = map[string]string{kserveStopAnnotationKey: "true"}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc, platformAggregateConfig()).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}

	if _, _, _, err := r.reconcileWorkloadsAndRBAC(ctx, insvc); err != nil {
		t.Fatalf("reconcileWorkloadsAndRBAC: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-aggregate"}, &lwsv1.LeaderWorkerSet{}); err != nil {
		t.Fatalf("direct insvc must not honor stop annotation; expected LWS created, got err=%v", err)
	}
}

func TestBuildDesiredComponents_PDComponentsAndEagleEye(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	enabledTrue := true
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			Components: &infernexv1alpha1.InfernexComponentsSpec{
				PDOrchestrator: &infernexv1alpha1.PDOrchestratorComponentSpec{
					ElasticScaler:        &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledTrue},
					Tidal:                &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledTrue},
					ResourceScalingGroup: &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledTrue},
				},
				EagleEye: &infernexv1alpha1.EagleEyeComponentSpec{
					HardwareMonitor:            &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledTrue},
					HardwareDiagnosis:          &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledTrue},
					NetworkPerformanceExporter: &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledTrue},
				},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	desired, err := r.buildDesiredComponents(ctx, insvc, insvc.Spec)
	if err != nil {
		t.Fatalf("buildDesiredComponents error: %v", err)
	}
	for _, comp := range []string{
		"pd-orchestrator-elastic-scaler",
		"pd-orchestrator-tidal",
		"pd-orchestrator-rsg",
		"eagle-eye-hardware-monitor",
		"eagle-eye-hardware-diagnosis",
		"eagle-eye-network-performance-exporter",
	} {
		if _, ok := desired[comp]; !ok {
			t.Fatalf("expected %s in desired map", comp)
		}
	}
	if desired["eagle-eye-hardware-monitor"].WorkloadKind != workloadKindDaemonSet {
		t.Fatalf("hardware monitor should be daemonset, got %q", desired["eagle-eye-hardware-monitor"].WorkloadKind)
	}
	if !desired["eagle-eye-hardware-monitor"].DisableService {
		t.Fatal("hardware monitor should disable service")
	}
	if desired["eagle-eye-network-performance-exporter"].WorkloadKind != workloadKindDaemonSet {
		t.Fatalf("network performance exporter should be daemonset, got %q", desired["eagle-eye-network-performance-exporter"].WorkloadKind)
	}
	if desired["eagle-eye-network-performance-exporter"].DisableService {
		t.Fatal("network performance exporter should expose metrics service")
	}
	if desired["eagle-eye-network-performance-exporter"].ServicePort != 8222 {
		t.Fatalf("network performance exporter service port = %d, want 8222", desired["eagle-eye-network-performance-exporter"].ServicePort)
	}
}

func TestBuildDesiredComponents_MooncakeEnabled(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	enabledTrue := true
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			Components: &infernexv1alpha1.InfernexComponentsSpec{
				Mooncake: &infernexv1alpha1.MooncakeComponentSpec{
					Enabled: &enabledTrue,
					Master: &infernexv1alpha1.TemplateComponentSpec{
						Replicas: 1,
						Template: testTemplate("mooncake:v1", 8080),
					},
				},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	desired, err := r.buildDesiredComponents(ctx, insvc, insvc.Spec)
	if err != nil {
		t.Fatalf("buildDesiredComponents mooncake error: %v", err)
	}
	if _, ok := desired["mooncake-master"]; !ok {
		t.Fatalf("expected mooncake-master in desired map, got %#v", desired)
	}
	if _, ok := desired["mooncake-metadata"]; !ok {
		t.Fatalf("expected mooncake-metadata in desired map, got %#v", desired)
	}
}

func TestReconcile_SourceRefLinkedFullPass(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := reconcileTestScheme(t)
	insvc := newLinkedInferNexService("ns-a", "demo")
	insvc.Finalizers = []string{infernexServiceFinalizer}
	insvc.Spec.Components = &infernexv1alpha1.InfernexComponentsSpec{
		CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{Enabled: ptr.To(true), Replicas: 1},
	}
	llm := newLinkedLLM("ns-a", "demo", infernexRuntimeValue)
	base := fake.NewClientBuilder().
		WithScheme(s).
		WithStatusSubresource(&infernexv1alpha1.InferNexService{}).
		WithObjects(insvc, platformAggregateConfig()).
		Build()
	cl := &llmAwareClient{
		Client: base,
		llms: map[types.NamespacedName]*unstructured.Unstructured{
			{Namespace: "ns-a", Name: "demo"}: llm,
		},
	}
	r := &InferNexServiceReconciler{
		Client:            cl,
		Scheme:            s,
		TemplateNamespace: "tpl-ns",
		Recorder:          record.NewFakeRecorder(8),
	}

	res, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}})
	if err != nil {
		t.Fatalf("Reconcile linked full pass error: %v", err)
	}
	if res.Requeue {
		t.Fatalf("unexpected requeue: %#v", res)
	}
	fresh := &infernexv1alpha1.InferNexService{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo"}, fresh); err != nil {
		t.Fatalf("get infernexservice: %v", err)
	}
	if fresh.Status.Mode != "aggregate" {
		t.Fatalf("expected aggregate mode in status, got %q", fresh.Status.Mode)
	}
	if len(fresh.Status.Conditions) == 0 {
		t.Fatal("expected status conditions after reconcile")
	}
}

func TestReconcile_DirectAggregateFullPass(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "demo",
			Namespace:  "ns-a",
			Finalizers: []string{infernexServiceFinalizer},
		},
		Spec: infernexv1alpha1.InferNexServiceSpec{
				Engine: &infernexv1alpha1.InferenceEngineSpec{
					InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
						Replicas: ptr.To(int32(1)),
						Template: testTemplate("engine:v1", 8000),
					},
				},
			Components: &infernexv1alpha1.InfernexComponentsSpec{
				CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{Enabled: ptr.To(true), Replicas: 1},
			},
		},
	}
	cl := fake.NewClientBuilder().
		WithScheme(s).
		WithStatusSubresource(&infernexv1alpha1.InferNexService{}).
		WithObjects(insvc, platformAggregateConfig()).
		Build()
	r := &InferNexServiceReconciler{
		Client:            cl,
		Scheme:            s,
		TemplateNamespace: "tpl-ns",
		Recorder:          record.NewFakeRecorder(8),
	}

	res, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}})
	if err != nil {
		t.Fatalf("Reconcile direct aggregate error: %v", err)
	}
	if res.Requeue {
		t.Fatalf("unexpected requeue: %#v", res)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-aggregate"}, &appsv1.Deployment{}); err != nil {
		t.Fatalf("expected engine deployment: %v", err)
	}
	fresh := &infernexv1alpha1.InferNexService{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo"}, fresh); err != nil {
		t.Fatalf("get infernexservice: %v", err)
	}
	if fresh.Status.Mode != "aggregate" {
		t.Fatalf("expected aggregate mode, got %q", fresh.Status.Mode)
	}
}

func TestReconcile_DirectAggregateWithGatewayRouting(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "demo",
			Namespace:  "ns-a",
			Finalizers: []string{infernexServiceFinalizer},
		},
		Spec: infernexv1alpha1.InferNexServiceSpec{
				Engine: &infernexv1alpha1.InferenceEngineSpec{
					InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
						Replicas: ptr.To(int32(1)),
						Template: testTemplate("engine:v1", 8000),
					},
				},
			IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
				Router: &infernexv1alpha1.ComponentSpec{
					Enabled:  ptr.To(true),
					Replicas: 1,
					Template: testTemplate("router:v1", 9000),
				},
			},
		},
	}
	cl := fake.NewClientBuilder().
		WithScheme(s).
		WithStatusSubresource(&infernexv1alpha1.InferNexService{}).
		WithObjects(insvc, platformAggregateConfig()).
		Build()
	r := &InferNexServiceReconciler{
		Client:            cl,
		Scheme:            s,
		TemplateNamespace: "tpl-ns",
		Recorder:          record.NewFakeRecorder(8),
	}

	res, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}})
	if err != nil {
		t.Fatalf("Reconcile with gateway routing error: %v", err)
	}
	if res.Requeue {
		t.Fatalf("unexpected requeue: %#v", res)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-infernex-gateway"}, &gwapiv1.Gateway{}); err != nil {
		t.Fatalf("expected managed gateway created: %v", err)
	}
}

func TestReconcile_TerminatingAndErrorPaths(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	now := metav1.Now()

	t.Run("terminating infernexservice triggers deletion reconcile", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{
				Name:              "demo",
				Namespace:         "ns-a",
				Finalizers:        []string{infernexServiceFinalizer},
				DeletionTimestamp: &now,
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s}
		res, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}})
		if err != nil {
			t.Fatalf("Reconcile terminating error: %v", err)
		}
		if res.Requeue {
			t.Fatalf("unexpected requeue: %#v", res)
		}
	})

	t.Run("workloads reconcile failure surfaces error", func(t *testing.T) {
		insvc := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "demo",
				Namespace:  "ns-a",
				Finalizers: []string{infernexServiceFinalizer},
			},
			Spec: infernexv1alpha1.InferNexServiceSpec{
				BaseRefs: []infernexv1alpha1.NamedRef{{Name: "missing-template"}},
			},
		}
		cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns", Recorder: record.NewFakeRecorder(4)}
		_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}})
		if err == nil {
			t.Fatal("expected reconcile error when baseRef template missing")
		}
	})
}

func TestReconcileInferNexServiceDeletion_NoFinalizerNoop(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	res, err := r.reconcileInferNexServiceDeletion(ctx, insvc)
	if err != nil {
		t.Fatalf("reconcileInferNexServiceDeletion noop error: %v", err)
	}
	if res.Requeue {
		t.Fatalf("unexpected requeue: %#v", res)
	}
}

func TestReconcilePersistStatus_RecordsSuccessEvent(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", Generation: 5},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{Kind: "LLMInferenceService", Name: "llm"},
		},
	}
	cl := fake.NewClientBuilder().
		WithScheme(s).
		WithStatusSubresource(&infernexv1alpha1.InferNexService{}).
		WithObjects(insvc).
		Build()
	rec := record.NewFakeRecorder(4)
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, Recorder: rec}

	res, err := r.reconcilePersistStatus(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "ns-a", Name: "demo"}}, map[string]componentPlan{}, infernexv1alpha1.InferNexServiceSpec{}, "aggregate")
	if err != nil {
		t.Fatalf("reconcilePersistStatus error: %v", err)
	}
	if res.Requeue {
		t.Fatalf("unexpected requeue: %#v", res)
	}
	select {
	case e := <-rec.Events:
		if e == "" {
			t.Fatal("expected reconcile success event")
		}
	default:
		t.Fatal("expected event recorded on successful status persist")
	}
}
