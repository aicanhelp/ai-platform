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
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func testTemplate(image string, ports ...int32) *corev1.PodTemplateSpec {
	cports := make([]corev1.ContainerPort, 0, len(ports))
	for _, p := range ports {
		cports = append(cports, corev1.ContainerPort{ContainerPort: p})
	}
	return &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Image: image,
				Ports: cports,
			}},
		},
	}
}

func testPDEngineSpec() *infernexv1alpha1.InferenceEngineSpec {
	return &infernexv1alpha1.InferenceEngineSpec{
		InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
			Template: testTemplate("decode:v1", 8000),
		},
		Prefill: &infernexv1alpha1.InferenceEngineWorkloadSpec{
			Template: testTemplate("prefill:v1", 8000),
		},
	}
}

func testPDEnginePrefillOnly() *infernexv1alpha1.InferenceEngineSpec {
	return &infernexv1alpha1.InferenceEngineSpec{
		Prefill: &infernexv1alpha1.InferenceEngineWorkloadSpec{
			Template: testTemplate("prefill:v1", 8000),
		},
	}
}

func TestBuildEngineAndComponentTemplates(t *testing.T) {
	t.Parallel()

	t.Run("buildEngineWorkloadPlan success and defaults", func(t *testing.T) {
		p, err := buildEngineWorkloadPlan("engine-aggregate", &infernexv1alpha1.InferenceEngineWorkloadSpec{
			Replicas: ptr.To(int32(2)),
			Template: testTemplate("engine:v1", 8080),
		})
		if err != nil {
			t.Fatalf("buildEngineWorkloadPlan error: %v", err)
		}
		if p.ServicePort != 8080 || p.Replicas == nil || *p.Replicas != 2 {
			t.Fatalf("unexpected plan from buildEngineWorkloadPlan: %#v", p)
		}
		if p.Template.Spec.Containers[0].Name == "" {
			t.Fatal("expected unnamed container auto-filled")
		}
	})

	t.Run("buildEngineWorkloadPlan validation errors", func(t *testing.T) {
		if _, err := buildEngineWorkloadPlan("x", nil); err == nil {
			t.Fatal("expected error for nil workload")
		}
		if _, err := buildEngineWorkloadPlan("x", &infernexv1alpha1.InferenceEngineWorkloadSpec{}); err == nil {
			t.Fatal("expected error for nil template")
		}
		if _, err := buildEngineWorkloadPlan("x", &infernexv1alpha1.InferenceEngineWorkloadSpec{
			Template: &corev1.PodTemplateSpec{},
		}); err == nil {
			t.Fatal("expected error for empty containers")
		}
	})

	t.Run("buildTemplateComponentPodTemplate and firstContainerServicePort", func(t *testing.T) {
		tpl, err := buildTemplateComponentPodTemplate("mooncake-master", &infernexv1alpha1.TemplateComponentSpec{
			Template: testTemplate("mooncake:v1", 6379),
		})
		if err != nil {
			t.Fatalf("buildTemplateComponentPodTemplate error: %v", err)
		}
		if got := firstContainerServicePort(tpl, 8000); got != 6379 {
			t.Fatalf("unexpected first container service port: %d", got)
		}
		if got := firstContainerServicePort(nil, 8000); got != 8000 {
			t.Fatalf("expected default service port when template nil, got %d", got)
		}
	})
}

func TestManagedAssetsAndBuildManagedComponentPodTemplate(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	ci, err := resolveCacheIndexerComponent(&infernexv1alpha1.CacheIndexerComponentSpec{Enabled: ptr.To(true), Replicas: 0})
	if err != nil {
		t.Fatalf("resolveCacheIndexerComponent error: %v", err)
	}
	if ci == nil || ci.Template == nil || ci.Replicas != 1 {
		t.Fatalf("unexpected cache-indexer resolve result: %#v", ci)
	}
	disabled, err := resolveManagedComponent("proxy-server", &infernexv1alpha1.EnabledComponentSpec{Enabled: ptr.To(false)})
	if err != nil {
		t.Fatalf("resolveManagedComponent disabled error: %v", err)
	}
	if disabled != nil {
		t.Fatalf("expected nil for disabled managed component, got %#v", disabled)
	}

	plan := componentPlan{
		Replicas:       ptr.To(int32(1)),
		Template:       testTemplate("cache:v1", 28080),
		ServicePort:    28080,
		WorkloadKind:   workloadKindDeployment,
		DisableService: false,
	}
	tpl, err := r.buildManagedComponentPodTemplate(
		ctx,
		owner,
		cacheIndexerComponent,
		plan,
		map[string]string{"infernex.io/owner": owner.Name},
		map[string]string{"infernex.io/component": cacheIndexerComponent},
		nil,
	)
	if err != nil {
		t.Fatalf("buildManagedComponentPodTemplate error: %v", err)
	}
	if tpl.Labels[labelInfernexManagedBy] != valueInfernexBridgeManagedBy {
		t.Fatalf("expected managed-by label filled, got %v", tpl.Labels)
	}
	if tpl.Spec.ServiceAccountName != componentControllerSAName(cacheIndexerComponent) {
		t.Fatalf("expected cache-indexer service account assigned, got %q", tpl.Spec.ServiceAccountName)
	}
	c := preferredContainer(tpl, "cache-indexer", "main")
	if c == nil {
		t.Fatal("expected cache-indexer container")
	}
	foundNS := false
	for _, e := range c.Env {
		if e.Name == envPodNamespace && e.ValueFrom != nil {
			foundNS = true
		}
	}
	if !foundNS {
		t.Fatalf("expected POD_NAMESPACE env, got %#v", c.Env)
	}
	cm := &corev1.ConfigMap{}
	if err := cl.Get(ctx, client.ObjectKey{Namespace: "ns-a", Name: cacheIndexerConfigMapName("demo")}, cm); err != nil {
		t.Fatalf("expected cache-indexer configmap reconciled: %v", err)
	}
}

func TestMooncakeSingletonDeploymentAndServiceUseFixedIdentity(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	ownerA := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "isvc-a", Namespace: "ns-a"}}
	ownerB := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "isvc-b", Namespace: "ns-a"}}
	plan := componentPlan{Replicas: ptr.To(int32(1)), ServicePort: 30089, Template: testTemplate("mooncake:v1", 30089)}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(ownerA, ownerB).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	if err := r.reconcileComponentDeployment(ctx, ownerA, "mooncake-master", plan, nil); err != nil {
		t.Fatalf("create singleton Deployment: %v", err)
	}
	if err := r.reconcileComponentService(ctx, ownerA, "mooncake-master", plan); err != nil {
		t.Fatalf("create singleton Service: %v", err)
	}
	// A second InferNexService must not replace the namespace singleton template.
	// Its different per-service template is ignored because the singleton already
	// has an established configuration.
	if err := r.reconcileComponentDeployment(ctx, ownerB, "mooncake-master", componentPlan{ServicePort: 30089, Template: testTemplate("different:v2", 30089)}, nil); err != nil {
		t.Fatalf("reconcile second singleton Deployment: %v", err)
	}

	dep := &appsv1.Deployment{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "mooncake-master-deployment"}, dep); err != nil {
		t.Fatalf("get singleton Deployment: %v", err)
	}
	if got := dep.Spec.Template.Spec.Containers[0].Image; got != "mooncake:v1" {
		t.Fatalf("second owner overwrote singleton image: got %q", got)
	}
	if dep.Spec.Selector.MatchLabels["app"] != "mooncake-master" || dep.Spec.Template.Labels["app"] != "mooncake-master" {
		t.Fatalf("singleton Deployment identity is inconsistent: selector=%#v labels=%#v", dep.Spec.Selector.MatchLabels, dep.Spec.Template.Labels)
	}
	if len(dep.OwnerReferences) != 0 {
		t.Fatalf("singleton Deployment must not have an InferNexService owner: %#v", dep.OwnerReferences)
	}

	svc := &corev1.Service{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "mooncake-master-service"}, svc); err != nil {
		t.Fatalf("get singleton Service: %v", err)
	}
	if svc.Spec.Selector["app"] != "mooncake-master" || len(svc.OwnerReferences) != 0 {
		t.Fatalf("singleton Service identity is inconsistent: selector=%#v refs=%#v", svc.Spec.Selector, svc.OwnerReferences)
	}

	if err := r.syncMooncakeSingletonReplicas(ctx, ownerA); err != nil {
		t.Fatalf("scale singleton up: %v", err)
	}
	if err := cl.Get(ctx, client.ObjectKeyFromObject(dep), dep); err != nil {
		t.Fatalf("get scaled singleton Deployment: %v", err)
	}
	if dep.Spec.Replicas == nil || *dep.Spec.Replicas != 1 {
		t.Fatalf("expected singleton replicas=1 with active owners, got %v", dep.Spec.Replicas)
	}
	if err := cl.Delete(ctx, ownerA); err != nil {
		t.Fatalf("delete owner A: %v", err)
	}
	if err := cl.Delete(ctx, ownerB); err != nil {
		t.Fatalf("delete owner B: %v", err)
	}
	if err := r.syncMooncakeSingletonReplicas(ctx, ownerA); err != nil {
		t.Fatalf("scale singleton down: %v", err)
	}
	if err := cl.Get(ctx, client.ObjectKeyFromObject(dep), dep); err != nil {
		t.Fatalf("get scaled-down singleton Deployment: %v", err)
	}
	if dep.Spec.Replicas == nil || *dep.Spec.Replicas != 0 {
		t.Fatalf("expected singleton replicas=0 with no active owners, got %v", dep.Spec.Replicas)
	}
}

func TestReconcileComponentWorkloadAndPruneWorkloads(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	depPlan := componentPlan{
		Replicas:     ptr.To(int32(1)),
		Template:     testTemplate("cache:v1", 8080),
		ServicePort:  8080,
		WorkloadKind: workloadKindDeployment,
	}
	if err := r.reconcileComponentWorkload(ctx, owner, cacheIndexerComponent, depPlan, nil); err != nil {
		t.Fatalf("reconcileComponentWorkload deployment error: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: owner.Name + "-" + cacheIndexerComponent}, &appsv1.Deployment{}); err != nil {
		t.Fatalf("expected deployment created: %v", err)
	}

	dsPlan := componentPlan{
		Replicas:     ptr.To(int32(1)),
		Template:     testTemplate("daemon:v1", 8081),
		ServicePort:  8081,
		WorkloadKind: workloadKindDaemonSet,
	}
	if err := r.reconcileComponentWorkload(ctx, owner, "daemon-comp", dsPlan, nil); err != nil {
		t.Fatalf("reconcileComponentWorkload daemonset error: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: owner.Name + "-daemon-comp"}, &appsv1.DaemonSet{}); err != nil {
		t.Fatalf("expected daemonset created: %v", err)
	}

	if err := r.pruneComponentDeployments(ctx, owner, map[string]componentPlan{
		cacheIndexerComponent: depPlan,
	}); err != nil {
		t.Fatalf("pruneComponentDeployments keep error: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: owner.Name + "-" + cacheIndexerComponent}, &appsv1.Deployment{}); err != nil {
		t.Fatalf("expected deployment still present: %v", err)
	}
	if err := r.pruneComponentDeployments(ctx, owner, map[string]componentPlan{}); err != nil {
		t.Fatalf("pruneComponentDeployments delete error: %v", err)
	}
	err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: owner.Name + "-" + cacheIndexerComponent}, &appsv1.Deployment{})
	if !apierrors.IsNotFound(err) {
		t.Fatalf("expected deployment deleted, got %v", err)
	}

	if err := r.pruneComponentDaemonSets(ctx, owner, map[string]componentPlan{
		"daemon-comp": dsPlan,
	}); err != nil {
		t.Fatalf("pruneComponentDaemonSets keep error: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: owner.Name + "-daemon-comp"}, &appsv1.DaemonSet{}); err != nil {
		t.Fatalf("expected daemonset still present: %v", err)
	}
	if err := r.pruneComponentDaemonSets(ctx, owner, map[string]componentPlan{}); err != nil {
		t.Fatalf("pruneComponentDaemonSets delete error: %v", err)
	}
	err = cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: owner.Name + "-daemon-comp"}, &appsv1.DaemonSet{})
	if !apierrors.IsNotFound(err) {
		t.Fatalf("expected daemonset deleted, got %v", err)
	}
}

func TestPlatformDefaultTemplateHelpers(t *testing.T) {
	t.Parallel()
	if !componentTemplateMissing(&infernexv1alpha1.ComponentSpec{Enabled: ptr.To(true)}) {
		t.Fatal("expected component template missing when enabled template is nil")
	}
	if componentTemplateMissing(&infernexv1alpha1.ComponentSpec{Enabled: ptr.To(false)}) {
		t.Fatal("disabled component should not require template")
	}
	if !inferenceWorkloadTemplateMissing(&infernexv1alpha1.InferenceEngineWorkloadSpec{}) {
		t.Fatal("workload with nil template should be missing")
	}

	dst := &infernexv1alpha1.InferNexServiceSpec{
		Components: &infernexv1alpha1.InfernexComponentsSpec{
			Mooncake: &infernexv1alpha1.MooncakeComponentSpec{
				Enabled: ptr.To(true),
				Master:  &infernexv1alpha1.TemplateComponentSpec{Replicas: 1},
			},
		},
	}
	if !needsPlatformTemplateFill(dst) {
		t.Fatal("expected mooncake needs platform template fill")
	}
	src := &infernexv1alpha1.InferNexServiceSpec{
		Components: &infernexv1alpha1.InfernexComponentsSpec{
			Mooncake: &infernexv1alpha1.MooncakeComponentSpec{
				Enabled: ptr.To(true),
				Master: &infernexv1alpha1.TemplateComponentSpec{
					Template: testTemplate("moon:v1", 8080),
				},
			},
		},
	}
	fillMissingTemplatesFromPlatformDefault(dst, src)
	if dst.Components.Mooncake.Master.Template == nil {
		t.Fatal("expected mooncake template filled")
	}
	mergePlatformDefaultComponents(dst, src)
	if dst.Components.Mooncake == nil {
		t.Fatal("expected mooncake merged")
	}
}

func TestMergePlatformDefaultTemplates_BasicPaths(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	infsvc := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}

	t.Run("missing config and no template need", func(t *testing.T) {
		cl := fake.NewClientBuilder().WithScheme(s).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}
		if err := r.mergePlatformDefaultTemplates(ctx, infsvc, &infernexv1alpha1.InferNexServiceSpec{}, "missing"); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("missing config but template required", func(t *testing.T) {
		cl := fake.NewClientBuilder().WithScheme(s).Build()
		r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}
		effective := &infernexv1alpha1.InferNexServiceSpec{
			Engine: &infernexv1alpha1.InferenceEngineSpec{},
		}
		err := r.mergePlatformDefaultTemplates(ctx, infsvc, effective, "missing")
		if err == nil || !strings.Contains(err.Error(), "not found") {
			t.Fatalf("expected not found error, got %v", err)
		}
	})
}
