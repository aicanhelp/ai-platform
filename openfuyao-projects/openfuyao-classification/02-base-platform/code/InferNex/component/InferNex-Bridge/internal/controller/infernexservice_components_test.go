/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */


package controller

import (
	"context"
	"sort"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

// Linked InferNexService (spec.sourceRef): KServe owns engine pods; InferNex must still
// plan proxy-server when merged effective spec enables it and the linked LLM is PD (prefill).
func TestBuildDesiredComponents_LinkedPD_IncludesProxyServer(t *testing.T) {
	t.Parallel()
	scheme := runtime.NewScheme()
	if err := appsv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := infernexv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	llm := &unstructured.Unstructured{}
	llm.SetAPIVersion("serving.kserve.io/v1alpha1")
	llm.SetKind("LLMInferenceService")
	llm.SetName("linked-llm")
	llm.SetNamespace("ns-a")
	if err := unstructured.SetNestedField(llm.Object, map[string]interface{}{"replicas": int64(1)}, "spec", "prefill"); err != nil {
		t.Fatal(err)
	}

	infsvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "linked-llm", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{
				APIVersion: "serving.kserve.io/v1alpha1",
				Kind:       "LLMInferenceService",
				Name:       "linked-llm",
				Namespace:  "ns-a",
			},
		},
	}

	effective := infernexv1alpha1.InferNexServiceSpec{}

	cl := fake.NewClientBuilder().WithScheme(scheme).WithRuntimeObjects(llm).Build()
	r := &InferNexServiceReconciler{Client: cl}

	desired, err := r.buildDesiredComponents(context.Background(), infsvc, effective)
	if err != nil {
		t.Fatalf("buildDesiredComponents: %v", err)
	}
	if _, ok := desired["proxy-server"]; !ok {
		t.Fatalf("expected proxy-server in desired map, got keys %v", sortedKeys(desired))
	}
	plan := desired["proxy-server"]
	if !plan.IsProxyServer {
		t.Fatal("expected IsProxyServer")
	}
	if plan.ProxyPDWorkloadName != "linked-llm" || plan.ProxyPDWorkloadNS != "ns-a" {
		t.Fatalf("proxy workload identity: got name=%q ns=%q", plan.ProxyPDWorkloadName, plan.ProxyPDWorkloadNS)
	}
}

func TestBuildDesiredComponents_LinkedPD_FromBaseRefTemplate_IncludesProxyServer(t *testing.T) {
	t.Parallel()
	scheme := runtime.NewScheme()
	if err := appsv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := infernexv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	llm := &unstructured.Unstructured{}
	llm.SetAPIVersion("serving.kserve.io/v1alpha2")
	llm.SetKind("LLMInferenceService")
	llm.SetName("linked-llm")
	llm.SetNamespace("ns-a")
	if err := unstructured.SetNestedSlice(llm.Object, []interface{}{
		map[string]interface{}{"name": "pd-template"},
	}, "spec", "baseRefs"); err != nil {
		t.Fatal(err)
	}

	cfg := &unstructured.Unstructured{}
	cfg.SetAPIVersion("serving.kserve.io/v1alpha2")
	cfg.SetKind("LLMInferenceServiceConfig")
	cfg.SetName("pd-template")
	cfg.SetNamespace("ns-a")
	if err := unstructured.SetNestedField(cfg.Object, map[string]interface{}{"replicas": int64(1)}, "spec", "prefill"); err != nil {
		t.Fatal(err)
	}

	infsvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "linked-llm", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{
				APIVersion: "serving.kserve.io/v1alpha2",
				Kind:       "LLMInferenceService",
				Name:       "linked-llm",
				Namespace:  "ns-a",
			},
		},
	}

	cl := fake.NewClientBuilder().WithScheme(scheme).WithRuntimeObjects(llm, cfg).Build()
	r := &InferNexServiceReconciler{Client: cl}

	desired, err := r.buildDesiredComponents(context.Background(), infsvc, infernexv1alpha1.InferNexServiceSpec{})
	if err != nil {
		t.Fatalf("buildDesiredComponents: %v", err)
	}
	if _, ok := desired["proxy-server"]; !ok {
		t.Fatalf("expected proxy-server for linked PD baseRef, got keys %v", sortedKeys(desired))
	}
	mode, err := r.computeServingMode(context.Background(), infsvc)
	if err != nil {
		t.Fatalf("computeServingMode: %v", err)
	}
	if mode != "pd" {
		t.Fatalf("expected linked llmisvc mode pd from baseRef template, got %q", mode)
	}
}

func TestBuildDesiredComponents_DirectPD_IncludesProxyServer(t *testing.T) {
	t.Parallel()
	infsvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "pd-svc", Namespace: "ns-a"},
	}
	effective := infernexv1alpha1.InferNexServiceSpec{
		Engine: testPDEnginePrefillOnly(),
	}
	r := &InferNexServiceReconciler{}
	desired, err := r.buildDesiredComponents(context.Background(), infsvc, effective)
	if err != nil {
		t.Fatalf("buildDesiredComponents: %v", err)
	}
	if _, ok := desired["proxy-server"]; !ok {
		t.Fatalf("expected proxy-server for direct PD, got keys %v", sortedKeys(desired))
	}
}

func TestBuildDesiredComponents_LinkedAggregate_NoProxyWhenLLMHasNoPrefill(t *testing.T) {
	t.Parallel()
	scheme := runtime.NewScheme()
	if err := appsv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := infernexv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	llm := &unstructured.Unstructured{}
	llm.SetAPIVersion("serving.kserve.io/v1alpha1")
	llm.SetKind("LLMInferenceService")
	llm.SetName("agg-llm")
	llm.SetNamespace("ns-b")
	// no spec.prefill => aggregate mode; shouldLaunchProxyServer returns false

	infsvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "agg-llm", Namespace: "ns-b"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{
				APIVersion: "serving.kserve.io/v1alpha1",
				Kind:       "LLMInferenceService",
				Name:       "agg-llm",
				Namespace:  "ns-b",
			},
		},
	}

	effective := infernexv1alpha1.InferNexServiceSpec{}

	cl := fake.NewClientBuilder().WithScheme(scheme).WithRuntimeObjects(llm).Build()
	r := &InferNexServiceReconciler{Client: cl}

	desired, err := r.buildDesiredComponents(context.Background(), infsvc, effective)
	if err != nil {
		t.Fatalf("buildDesiredComponents: %v", err)
	}
	if _, ok := desired["proxy-server"]; ok {
		t.Fatalf("did not expect proxy-server for aggregate linked LLM, got keys %v", sortedKeys(desired))
	}
}

func sortedKeys(m map[string]componentPlan) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func TestDeploymentPodMatchLabels_DirectPDEnginesUseOpenFuyao(t *testing.T) {
	t.Parallel()
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "qwen3-8b-pd"},
		Spec:       infernexv1alpha1.InferNexServiceSpec{},
	}
	plan := componentPlan{AppKubernetesIOName: "qwen3-8b-pd"}
	pref := deploymentPodMatchLabels(owner, "engine-pd-prefill", plan)
	if got := pref[labelOpenFuyaoPDRole]; got != openfuyaoPDRolePrefill {
		t.Fatalf("prefill role: got %q", got)
	}
	if got := pref[labelOpenFuyaoPDGroup]; got != "qwen3-8b-pd" {
		t.Fatalf("prefill group: got %q", got)
	}
	dec := deploymentPodMatchLabels(owner, "engine-pd-decode", plan)
	if got := dec[labelOpenFuyaoPDRole]; got != openfuyaoPDRoleDecode {
		t.Fatalf("decode role: got %q", got)
	}
	if got := dec[labelOpenFuyaoPDGroup]; got != "qwen3-8b-pd" {
		t.Fatalf("decode group: got %q", got)
	}
}

func TestDeploymentPodMatchLabels_DirectPDProxyUsesLeaderOpenFuyao(t *testing.T) {
	t.Parallel()
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "qwen3-8b-pd"},
		Spec:       infernexv1alpha1.InferNexServiceSpec{},
	}
	plan := componentPlan{
		AppKubernetesIOName: "qwen3-8b-pd",
		ProxyPDWorkloadName: "qwen3-8b-pd",
	}
	got := deploymentPodMatchLabels(owner, "proxy-server", plan)
	if got[labelOpenFuyaoPDRole] != openfuyaoPDRoleLeader || got[labelOpenFuyaoPDGroup] != "qwen3-8b-pd" {
		t.Fatalf("proxy selector: got %#v", got)
	}
}

func TestDeploymentPodMatchLabels_LinkedProxyUsesInfernexSelector(t *testing.T) {
	t.Parallel()
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "linked", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{
				Kind: "LLMInferenceService",
				Name: "llm",
			},
		},
	}
	plan := componentPlan{AppKubernetesIOName: "llm"}
	got := deploymentPodMatchLabels(owner, "proxy-server", plan)
	if got["infernex.io/owner"] != "linked" || got["infernex.io/component"] != "proxy-server" {
		t.Fatalf("linked proxy should keep infernex selector, got %#v", got)
	}
}

func TestDeploymentPodMatchLabels_HermesRouterUsesInfernex(t *testing.T) {
	t.Parallel()
	owner := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "x"}}
	plan := componentPlan{}
	got := deploymentPodMatchLabels(owner, hermesRouterComponent, plan)
	if got["infernex.io/component"] != hermesRouterComponent {
		t.Fatalf("got %#v", got)
	}
}

func TestMergeInferenceEngineWorkloadSpec_DataParallelFields(t *testing.T) {
	t.Parallel()
	dst := &infernexv1alpha1.InferenceEngineWorkloadSpec{}
	src := &infernexv1alpha1.InferenceEngineWorkloadSpec{
		DataParallelSize:      ptr.To(int32(2)),
		DataParallelSizeLocal: ptr.To(int32(1)),
	}
	mergeInferenceEngineWorkloadSpec(dst, src)
	if dst.DataParallelSize == nil || *dst.DataParallelSize != 2 {
		t.Fatalf("expected dataParallelSize 2, got %v", dst.DataParallelSize)
	}
	if dst.DataParallelSizeLocal == nil || *dst.DataParallelSizeLocal != 1 {
		t.Fatalf("expected dataParallelSizeLocal 1, got %v", dst.DataParallelSizeLocal)
	}
	dst2 := &infernexv1alpha1.InferenceEngineWorkloadSpec{DataParallelSize: ptr.To(int32(4))}
	src2 := &infernexv1alpha1.InferenceEngineWorkloadSpec{DataParallelSize: ptr.To(int32(2))}
	mergeInferenceEngineWorkloadSpec(dst2, src2)
	if dst2.DataParallelSize == nil || *dst2.DataParallelSize != 4 {
		t.Fatalf("user dataParallelSize must win over template, got %v", dst2.DataParallelSize)
	}
}

func TestMergeInferenceEngineWorkloadSpec_ReplicasNilNotOverriddenByZero(t *testing.T) {
	t.Parallel()
	dst := &infernexv1alpha1.InferenceEngineWorkloadSpec{}
	src := &infernexv1alpha1.InferenceEngineWorkloadSpec{Replicas: ptr.To(int32(2))}
	mergeInferenceEngineWorkloadSpec(dst, src)
	if dst.Replicas == nil || *dst.Replicas != 2 {
		t.Fatalf("expected replicas 2 from template, got %v", dst.Replicas)
	}
	dst2 := &infernexv1alpha1.InferenceEngineWorkloadSpec{Replicas: ptr.To(int32(3))}
	src2 := &infernexv1alpha1.InferenceEngineWorkloadSpec{}
	mergeInferenceEngineWorkloadSpec(dst2, src2)
	if dst2.Replicas == nil || *dst2.Replicas != 3 {
		t.Fatalf("user replicas must win over empty template, got %v", dst2.Replicas)
	}
}

func TestApplyDeploymentReplicas_EngineExternalScaling(t *testing.T) {
	t.Parallel()
	plan := componentPlan{Replicas: nil}
	created := &appsv1.Deployment{}
	applyDeploymentReplicas(created, "engine-aggregate", plan)
	if created.Spec.Replicas == nil || *created.Spec.Replicas != 1 {
		t.Fatalf("new engine deploy want default 1, got %v", created.Spec.Replicas)
	}
	existing := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{UID: "abc"}}
	existing.Spec.Replicas = ptr.To(int32(5))
	applyDeploymentReplicas(existing, "engine-aggregate", plan)
	if existing.Spec.Replicas == nil || *existing.Spec.Replicas != 5 {
		t.Fatalf("existing engine deploy should preserve 5, got %v", existing.Spec.Replicas)
	}
	planFixed := componentPlan{Replicas: ptr.To(int32(2))}
	existing2 := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{UID: "abc"}}
	existing2.Spec.Replicas = ptr.To(int32(5))
	applyDeploymentReplicas(existing2, "engine-aggregate", planFixed)
	if existing2.Spec.Replicas == nil || *existing2.Spec.Replicas != 2 {
		t.Fatalf("explicit replicas should apply, got %v", existing2.Spec.Replicas)
	}
}
