package controller

import (
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestComponentUnitHelpers(t *testing.T) {
	t.Parallel()

	t.Run("replicas and workload kind defaults", func(t *testing.T) {
		if got := replicas(0); got != 1 {
			t.Fatalf("expected default replica 1, got %d", got)
		}
		if got := componentWorkloadKind(componentPlan{}); got != workloadKindDeployment {
			t.Fatalf("expected deployment default, got %q", got)
		}
		if !isInferenceEngineComponent("engine-pd-prefill") || isInferenceEngineComponent("cache-indexer") {
			t.Fatal("unexpected isInferenceEngineComponent results")
		}
	})

	t.Run("applyDeploymentReplicas engine nil vs create", func(t *testing.T) {
		existing := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{UID: "existing-uid"}}
		applyDeploymentReplicas(existing, "engine-aggregate", componentPlan{})
		if existing.Spec.Replicas != nil {
			t.Fatalf("existing engine deployment should keep nil replicas for autoscaler, got %#v", existing.Spec.Replicas)
		}
		applyDeploymentReplicas(existing, "cache-indexer", componentPlan{})
		if existing.Spec.Replicas == nil || *existing.Spec.Replicas != 1 {
			t.Fatalf("non-engine component should default replicas=1, got %#v", existing.Spec.Replicas)
		}
		newDep := &appsv1.Deployment{}
		applyDeploymentReplicas(newDep, "engine-aggregate", componentPlan{})
		if newDep.Spec.Replicas == nil || *newDep.Spec.Replicas != 1 {
			t.Fatalf("new engine deployment should default replicas=1, got %#v", newDep.Spec.Replicas)
		}
		planReplicas := int32(3)
		withReplicas := &appsv1.Deployment{}
		applyDeploymentReplicas(withReplicas, "engine-aggregate", componentPlan{Replicas: &planReplicas})
		if withReplicas.Spec.Replicas == nil || *withReplicas.Spec.Replicas != 3 {
			t.Fatalf("plan replicas should win, got %#v", withReplicas.Spec.Replicas)
		}
	})

	t.Run("buildComponentPodTemplate normalizes container names", func(t *testing.T) {
		tpl, err := buildComponentPodTemplate("demo-comp", &infernexv1alpha1.ComponentSpec{
			Template: &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Image: "img:v1"}},
				},
			},
		})
		if err != nil || tpl.Spec.Containers[0].Name != "demo-comp" {
			t.Fatalf("expected container name normalized, got tpl=%#v err=%v", tpl, err)
		}
	})

	t.Run("buildTemplateComponentPodTemplate requires template", func(t *testing.T) {
		if _, err := buildTemplateComponentPodTemplate("mooncake-master", &infernexv1alpha1.TemplateComponentSpec{}); err == nil {
			t.Fatal("expected error when template missing")
		}
		masterTpl, err := buildTemplateComponentPodTemplate("mooncake-master", &infernexv1alpha1.TemplateComponentSpec{
			Template: testTemplate("mooncake:v1", 8080),
		})
		if err != nil || masterTpl == nil {
			t.Fatalf("expected mooncake master template built, got %#v err=%v", masterTpl, err)
		}
	})

	t.Run("pdWorkloadIdentity direct and linked", func(t *testing.T) {
		direct := &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		}
		name, ns := pdWorkloadIdentity(direct)
		if name != "demo" || ns != "ns-a" {
			t.Fatalf("unexpected direct identity: %s/%s", name, ns)
		}
		linked := direct.DeepCopy()
		linked.Spec.SourceRef = &infernexv1alpha1.SourceRef{Name: "llm", Namespace: "other-ns"}
		name, ns = pdWorkloadIdentity(linked)
		if name != "llm" || ns != "other-ns" {
			t.Fatalf("unexpected linked identity: %s/%s", name, ns)
		}
	})

	t.Run("enabled helper treats nil as true", func(t *testing.T) {
		if !enabled(nil) || !enabled(ptr.To(true)) || enabled(ptr.To(false)) {
			t.Fatal("unexpected enabled() semantics")
		}
	})
}
