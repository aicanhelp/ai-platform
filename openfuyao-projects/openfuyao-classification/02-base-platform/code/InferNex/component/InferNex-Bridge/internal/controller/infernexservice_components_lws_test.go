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
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	lwsv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestEngineWorkloadGroupSize(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		w       *infernexv1alpha1.InferenceEngineWorkloadSpec
		want    int32
		wantErr bool
	}{
		{name: "defaults", w: &infernexv1alpha1.InferenceEngineWorkloadSpec{}, want: 1},
		{name: "pd04 deployment", w: &infernexv1alpha1.InferenceEngineWorkloadSpec{
			DataParallelSize: ptr.To(int32(2)), DataParallelSizeLocal: ptr.To(int32(2)),
		}, want: 1},
		{name: "lws", w: &infernexv1alpha1.InferenceEngineWorkloadSpec{
			DataParallelSize: ptr.To(int32(2)), DataParallelSizeLocal: ptr.To(int32(1)),
		}, want: 2},
		{name: "invalid", w: &infernexv1alpha1.InferenceEngineWorkloadSpec{
			DataParallelSize: ptr.To(int32(3)), DataParallelSizeLocal: ptr.To(int32(2)),
		}, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := EngineWorkloadGroupSize(tc.w)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("groupSize = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestBuildEngineWorkloadPlanGroupSizeRouting(t *testing.T) {
	t.Parallel()

	depPlan, err := buildEngineWorkloadPlan("engine-aggregate", &infernexv1alpha1.InferenceEngineWorkloadSpec{
		Template:              testTemplate("engine:v1"),
		DataParallelSize:      ptr.To(int32(2)),
		DataParallelSizeLocal: ptr.To(int32(2)),
	})
	if err != nil {
		t.Fatalf("buildEngineWorkloadPlan deployment: %v", err)
	}
	if depPlan.WorkloadKind != workloadKindDeployment || depPlan.GroupSize != 1 {
		t.Fatalf("expected Deployment groupSize=1, got kind=%q groupSize=%d", depPlan.WorkloadKind, depPlan.GroupSize)
	}

	lwsPlan, err := buildEngineWorkloadPlan("engine-aggregate", &infernexv1alpha1.InferenceEngineWorkloadSpec{
		Template:              testTemplate("engine:v1"),
		DataParallelSize:      ptr.To(int32(2)),
		DataParallelSizeLocal: ptr.To(int32(1)),
	})
	if err != nil {
		t.Fatalf("buildEngineWorkloadPlan lws: %v", err)
	}
	if lwsPlan.WorkloadKind != workloadKindLeaderWorkerSet || lwsPlan.GroupSize != 2 {
		t.Fatalf("expected LWS groupSize=2, got kind=%q groupSize=%d", lwsPlan.WorkloadKind, lwsPlan.GroupSize)
	}
}

func TestReconcileComponentLeaderWorkerSet(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	plan := componentPlan{
		Replicas:     ptr.To(int32(1)),
		Template:     testTemplate("engine:v1", 8000),
		ServicePort:  8000,
		WorkloadKind: workloadKindLeaderWorkerSet,
		GroupSize:    2,
	}
	if err := r.reconcileComponentLeaderWorkerSet(ctx, owner, "engine-aggregate", plan, nil); err != nil {
		t.Fatalf("reconcileComponentLeaderWorkerSet error: %v", err)
	}
	lws := &lwsv1.LeaderWorkerSet{}
	key := types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-aggregate"}
	if err := cl.Get(ctx, key, lws); err != nil {
		t.Fatalf("get LWS: %v", err)
	}
	if lws.Spec.Replicas == nil || *lws.Spec.Replicas != 1 {
		t.Fatalf("unexpected replicas: %#v", lws.Spec.Replicas)
	}
	if lws.Spec.LeaderWorkerTemplate.Size == nil || *lws.Spec.LeaderWorkerTemplate.Size != 2 {
		t.Fatalf("unexpected group size: %#v", lws.Spec.LeaderWorkerTemplate.Size)
	}
	if lws.Spec.LeaderWorkerTemplate.LeaderTemplate == nil {
		t.Fatal("expected LeaderTemplate set")
	}
	if lws.Spec.StartupPolicy != lwsv1.LeaderCreatedStartupPolicy {
		t.Fatalf("unexpected startup policy: %q", lws.Spec.StartupPolicy)
	}
}

func TestApplyLWSReplicasExternalScaling(t *testing.T) {
	t.Parallel()
	existing := &lwsv1.LeaderWorkerSet{
		ObjectMeta: metav1.ObjectMeta{Name: "demo-engine-aggregate", Namespace: "ns-a", UID: "uid-1"},
		Spec: lwsv1.LeaderWorkerSetSpec{
			Replicas: ptr.To(int32(5)),
		},
	}
	applyLWSReplicas(existing, "engine-aggregate", componentPlan{Replicas: nil})
	if existing.Spec.Replicas == nil || *existing.Spec.Replicas != 5 {
		t.Fatalf("expected autoscaler replicas preserved, got %#v", existing.Spec.Replicas)
	}

	fresh := &lwsv1.LeaderWorkerSet{}
	applyLWSReplicas(fresh, "engine-aggregate", componentPlan{Replicas: nil})
	if fresh.Spec.Replicas == nil || *fresh.Spec.Replicas != 1 {
		t.Fatalf("expected default replicas=1 on create, got %#v", fresh.Spec.Replicas)
	}
}

func TestPruneComponentLeaderWorkerSets(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	stale := &lwsv1.LeaderWorkerSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: "demo-engine-aggregate", Namespace: "ns-a",
			Labels: map[string]string{"infernex.io/owner": "demo", "infernex.io/component": "engine-aggregate"},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner, stale).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	desired := map[string]componentPlan{
		"engine-aggregate": {WorkloadKind: workloadKindDeployment},
	}
	if err := r.pruneComponentLeaderWorkerSets(ctx, owner, desired); err != nil {
		t.Fatalf("pruneComponentLeaderWorkerSets error: %v", err)
	}
	lws := &lwsv1.LeaderWorkerSet{}
	key := types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-aggregate"}
	if err := cl.Get(ctx, key, lws); !apierrors.IsNotFound(err) {
		t.Fatalf("expected stale LWS deleted, get err=%v", err)
	}
}

func TestLWSRolloutReady(t *testing.T) {
	t.Parallel()
	ready := &lwsv1.LeaderWorkerSet{
		Spec:   lwsv1.LeaderWorkerSetSpec{Replicas: ptr.To(int32(2))},
		Status: lwsv1.LeaderWorkerSetStatus{ReadyReplicas: 2},
	}
	ok, msg := lwsRolloutReady(ready)
	if !ok || msg != "" {
		t.Fatalf("expected ready LWS, ok=%v msg=%q", ok, msg)
	}
	pending := &lwsv1.LeaderWorkerSet{
		Spec:   lwsv1.LeaderWorkerSetSpec{Replicas: ptr.To(int32(2))},
		Status: lwsv1.LeaderWorkerSetStatus{ReadyReplicas: 1},
	}
	ok, msg = lwsRolloutReady(pending)
	if ok || msg == "" {
		t.Fatalf("expected not ready, ok=%v msg=%q", ok, msg)
	}
}

func TestReconcileLWS_DataParallelTwoReplicasNilMeansOneGroup(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	plan, err := buildEngineWorkloadPlan("engine-aggregate", &infernexv1alpha1.InferenceEngineWorkloadSpec{
		Template:              testTemplate("engine:v1"),
		DataParallelSize:      ptr.To(int32(2)),
		DataParallelSizeLocal: ptr.To(int32(1)),
	})
	if err != nil {
		t.Fatalf("buildEngineWorkloadPlan: %v", err)
	}
	if err := r.reconcileComponentLeaderWorkerSet(ctx, owner, "engine-aggregate", plan, nil); err != nil {
		t.Fatalf("reconcileComponentLeaderWorkerSet: %v", err)
	}
	lws := &lwsv1.LeaderWorkerSet{}
	key := types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-aggregate"}
	if err := cl.Get(ctx, key, lws); err != nil {
		t.Fatalf("get LWS: %v", err)
	}
	// replicas nil → 1 LWS group (horizontal); groupSize 2 → 2 Pods per group.
	if lws.Spec.Replicas == nil || *lws.Spec.Replicas != 1 {
		t.Fatalf("expected LWS.spec.replicas=1 (one group), got %#v", lws.Spec.Replicas)
	}
	if lws.Spec.LeaderWorkerTemplate.Size == nil || *lws.Spec.LeaderWorkerTemplate.Size != 2 {
		t.Fatalf("expected LWS group size=2 (two Pods per group), got %#v", lws.Spec.LeaderWorkerTemplate.Size)
	}
}

func TestReconcileMixedPD_PrefillDeploymentDecodeLWS(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	insvc := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			Engine: &infernexv1alpha1.InferenceEngineSpec{
				InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
					Template:              testTemplate("decode:v1", 8000),
					DataParallelSize:      ptr.To(int32(2)),
					DataParallelSizeLocal: ptr.To(int32(1)),
				},
				Prefill: &infernexv1alpha1.InferenceEngineWorkloadSpec{
					Template:              testTemplate("prefill:v1", 8000),
					DataParallelSize:      ptr.To(int32(1)),
					DataParallelSizeLocal: ptr.To(int32(1)),
				},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(insvc, platformPDConfig()).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s, TemplateNamespace: "tpl-ns"}

	if _, _, _, err := r.reconcileWorkloadsAndRBAC(ctx, insvc); err != nil {
		t.Fatalf("reconcileWorkloadsAndRBAC: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-pd-prefill"}, &appsv1.Deployment{}); err != nil {
		t.Fatalf("expected prefill Deployment: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-pd-decode"}, &lwsv1.LeaderWorkerSet{}); err != nil {
		t.Fatalf("expected decode LWS: %v", err)
	}
	depList := &appsv1.DeploymentList{}
	if err := cl.List(ctx, depList, client.InNamespace("ns-a"), client.MatchingLabels{"infernex.io/owner": "demo"}); err != nil {
		t.Fatalf("list deployments: %v", err)
	}
	for i := range depList.Items {
		if depList.Items[i].Labels["infernex.io/component"] == "engine-pd-decode" {
			t.Fatal("decode must not remain Deployment when groupSize>1")
		}
	}
}

func TestPruneIsolation_DoesNotDeleteOtherInferNexService(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	ownerA := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "insvc-a", Namespace: "ns-a"}}
	ownerB := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "insvc-b", Namespace: "ns-a"}}
	otherLWS := &lwsv1.LeaderWorkerSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: "insvc-b-engine-aggregate", Namespace: "ns-a",
			Labels: map[string]string{"infernex.io/owner": "insvc-b", "infernex.io/component": "engine-aggregate"},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(ownerA, ownerB, otherLWS).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	desired := map[string]componentPlan{
		"engine-aggregate": {WorkloadKind: workloadKindDeployment},
	}
	if err := r.pruneComponentLeaderWorkerSets(ctx, ownerA, desired); err != nil {
		t.Fatalf("pruneComponentLeaderWorkerSets: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "insvc-b-engine-aggregate"}, &lwsv1.LeaderWorkerSet{}); err != nil {
		t.Fatalf("must not delete other insvc LWS: %v", err)
	}
}

func TestLWSPodTemplateGetsPDLabelsForProxyDiscovery(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	plan := componentPlan{
		Replicas:     ptr.To(int32(1)),
		Template:     testTemplate("decode:v1"),
		WorkloadKind: workloadKindLeaderWorkerSet,
		GroupSize:    2,
	}
	if err := r.reconcileComponentLeaderWorkerSet(ctx, owner, "engine-pd-decode", plan, nil); err != nil {
		t.Fatalf("reconcileComponentLeaderWorkerSet: %v", err)
	}
	lws := &lwsv1.LeaderWorkerSet{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-pd-decode"}, lws); err != nil {
		t.Fatalf("get LWS: %v", err)
	}
	labels := lws.Spec.LeaderWorkerTemplate.WorkerTemplate.Labels
	if labels[labelOpenFuyaoPDRole] != openfuyaoPDRoleDecode {
		t.Fatalf("LWS pod missing pdRole for proxy discovery, got %#v", labels)
	}
	if labels[labelOpenFuyaoPDGroup] != "demo" {
		t.Fatalf("LWS pod missing pdGroupID, got %#v", labels)
	}
}

func TestPruneTransition_RemovesStaleDeploymentWhenSwitchingToLWS(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
	staleDep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "demo-engine-aggregate", Namespace: "ns-a",
			Labels: map[string]string{"infernex.io/owner": "demo", "infernex.io/component": "engine-aggregate"},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner, staleDep).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	desired := map[string]componentPlan{
		"engine-aggregate": {WorkloadKind: workloadKindLeaderWorkerSet, GroupSize: 2},
	}
	if err := r.pruneOrphanedComponents(ctx, owner, desired); err != nil {
		t.Fatalf("pruneOrphanedComponents: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-aggregate"}, &appsv1.Deployment{}); !apierrors.IsNotFound(err) {
		t.Fatalf("expected stale Deployment deleted, get err=%v", err)
	}
}

func TestPruneTransition_RemovesStaleLWSWhenSwitchingToDeployment(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
	staleLWS := &lwsv1.LeaderWorkerSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: "demo-engine-aggregate", Namespace: "ns-a",
			Labels: map[string]string{"infernex.io/owner": "demo", "infernex.io/component": "engine-aggregate"},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner, staleLWS).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	desired := map[string]componentPlan{
		"engine-aggregate": {WorkloadKind: workloadKindDeployment, GroupSize: 1},
	}
	if err := r.pruneOrphanedComponents(ctx, owner, desired); err != nil {
		t.Fatalf("pruneOrphanedComponents: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-aggregate"}, &lwsv1.LeaderWorkerSet{}); !apierrors.IsNotFound(err) {
		t.Fatalf("expected stale LWS deleted, get err=%v", err)
	}
}

func TestReconcileComponentService_LWSEngineUsesPodMatchLabels(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	plan := componentPlan{
		Template:    testTemplate("engine:v1", 8000),
		ServicePort: 8000,
		WorkloadKind: workloadKindLeaderWorkerSet,
		GroupSize:   2,
	}
	if err := r.reconcileComponentService(ctx, owner, "engine-aggregate", plan); err != nil {
		t.Fatalf("reconcileComponentService: %v", err)
	}
	svc := &corev1.Service{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-workload-svc"}, svc); err != nil {
		t.Fatalf("get shared workload service: %v", err)
	}
	wantSel := deploymentPodMatchLabels(owner, "engine-aggregate", plan)
	for k, v := range wantSel {
		if svc.Spec.Selector[k] != v {
			t.Fatalf("service selector[%q]=%q, want %q", k, svc.Spec.Selector[k], v)
		}
	}
}

func TestReconcileComponentService_SharedPDServiceSelectsBothRoles(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"}}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	plan := componentPlan{
		Template:    testTemplate("decode:v1", 8000),
		ServicePort: 8000,
		WorkloadKind: workloadKindLeaderWorkerSet,
		GroupSize:   2,
	}
	if err := r.reconcileComponentService(ctx, owner, "engine-pd-decode", plan); err != nil {
		t.Fatalf("reconcileComponentService: %v", err)
	}
	svc := &corev1.Service{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-workload-svc"}, svc); err != nil {
		t.Fatalf("get shared pd service: %v", err)
	}
	if svc.Spec.Selector[labelAppKubernetesIOName] != "demo" {
		t.Fatalf("shared PD service selector: %#v", svc.Spec.Selector)
	}
}

func TestStatusForWorkloadKeys_LWS(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	owner := "demo"
	ns := "ns-a"
	desired := map[string]componentPlan{
		"engine-aggregate": {WorkloadKind: workloadKindLeaderWorkerSet, Replicas: ptr.To(int32(1))},
	}
	readyLWS := &lwsv1.LeaderWorkerSet{
		ObjectMeta: metav1.ObjectMeta{Name: "demo-engine-aggregate", Namespace: ns},
		Spec:       lwsv1.LeaderWorkerSetSpec{Replicas: ptr.To(int32(1))},
		Status: lwsv1.LeaderWorkerSetStatus{
			ReadyReplicas: 1,
			Conditions: []metav1.Condition{{
				Type:   string(lwsv1.LeaderWorkerSetAvailable),
				Status: metav1.ConditionTrue,
			}},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(readyLWS).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}
	st, err := r.statusForWorkloadKeys(ctx, ns, owner, []string{"engine-aggregate"}, desired)
	if err != nil {
		t.Fatalf("statusForWorkloadKeys: %v", err)
	}
	if st == nil || !st.Ready {
		t.Fatalf("expected ready LWS status, got %#v", st)
	}
}

func TestBuildEngineWorkloadPlanWorkerTemplate(t *testing.T) {
	t.Parallel()

	base := &infernexv1alpha1.InferenceEngineWorkloadSpec{
		Template:              testTemplate("engine:v1"),
		DataParallelSize:      ptr.To(int32(2)),
		DataParallelSizeLocal: ptr.To(int32(1)),
	}

	t.Run("empty worker treated as unset", func(t *testing.T) {
		w := base.DeepCopy()
		w.Worker = &corev1.PodTemplateSpec{}
		plan, err := buildEngineWorkloadPlan("engine-aggregate", w)
		if err != nil {
			t.Fatalf("buildEngineWorkloadPlan: %v", err)
		}
		if plan.WorkerTemplate != nil {
			t.Fatalf("expected nil WorkerTemplate for empty worker, got %#v", plan.WorkerTemplate)
		}
	})

	t.Run("effective worker copied to plan", func(t *testing.T) {
		w := base.DeepCopy()
		w.Worker = &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				NodeName:   "node-b",
				Containers: []corev1.Container{{Name: "w", Image: "worker:v1"}},
			},
		}
		plan, err := buildEngineWorkloadPlan("engine-aggregate", w)
		if err != nil {
			t.Fatalf("buildEngineWorkloadPlan: %v", err)
		}
		if plan.WorkerTemplate == nil || plan.WorkerTemplate.Spec.NodeName != "node-b" {
			t.Fatalf("expected worker template with node-b, got %#v", plan.WorkerTemplate)
		}
	})
}

func TestReconcileLWS_LeaderWorkerSplitByNodeName(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	leaderTpl := testTemplate("engine:v1", 8000)
	leaderTpl.Spec.NodeName = "node-a"
	workerTpl := testTemplate("engine:v1", 8000)
	workerTpl.Spec.NodeName = "node-b"
	plan := componentPlan{
		Replicas:       ptr.To(int32(1)),
		Template:       leaderTpl,
		WorkerTemplate: workerTpl,
		ServicePort:    8000,
		WorkloadKind:   workloadKindLeaderWorkerSet,
		GroupSize:      2,
	}
	if err := r.reconcileComponentLeaderWorkerSet(ctx, owner, "engine-aggregate", plan, nil); err != nil {
		t.Fatalf("reconcileComponentLeaderWorkerSet: %v", err)
	}
	lws := &lwsv1.LeaderWorkerSet{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-aggregate"}, lws); err != nil {
		t.Fatalf("get LWS: %v", err)
	}
	if lws.Spec.LeaderWorkerTemplate.LeaderTemplate == nil {
		t.Fatal("expected LeaderTemplate")
	}
	if lws.Spec.LeaderWorkerTemplate.LeaderTemplate.Spec.NodeName != "node-a" {
		t.Fatalf("leader nodeName = %q, want node-a", lws.Spec.LeaderWorkerTemplate.LeaderTemplate.Spec.NodeName)
	}
	if lws.Spec.LeaderWorkerTemplate.WorkerTemplate.Spec.NodeName != "node-b" {
		t.Fatalf("worker nodeName = %q, want node-b", lws.Spec.LeaderWorkerTemplate.WorkerTemplate.Spec.NodeName)
	}
}

func TestReconcileLWS_FallbackWorkerUsesTemplateWhenUnset(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := lwsTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	tpl := testTemplate("engine:v1", 8000)
	tpl.Spec.NodeName = "node-a"
	plan := componentPlan{
		Replicas:       ptr.To(int32(1)),
		Template:       tpl,
		WorkerTemplate: nil,
		ServicePort:    8000,
		WorkloadKind:   workloadKindLeaderWorkerSet,
		GroupSize:      2,
	}
	if err := r.reconcileComponentLeaderWorkerSet(ctx, owner, "engine-aggregate", plan, nil); err != nil {
		t.Fatalf("reconcileComponentLeaderWorkerSet: %v", err)
	}
	lws := &lwsv1.LeaderWorkerSet{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "demo-engine-aggregate"}, lws); err != nil {
		t.Fatalf("get LWS: %v", err)
	}
	if lws.Spec.LeaderWorkerTemplate.LeaderTemplate.Spec.NodeName != "node-a" {
		t.Fatalf("leader nodeName = %q, want node-a", lws.Spec.LeaderWorkerTemplate.LeaderTemplate.Spec.NodeName)
	}
	if lws.Spec.LeaderWorkerTemplate.WorkerTemplate.Spec.NodeName != "node-a" {
		t.Fatalf("worker fallback nodeName = %q, want node-a", lws.Spec.LeaderWorkerTemplate.WorkerTemplate.Spec.NodeName)
	}
}

func TestWorkloadWorkerTemplateEffective(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		w    *infernexv1alpha1.InferenceEngineWorkloadSpec
		want bool
	}{
		{name: "nil workload", w: nil, want: false},
		{name: "no worker field", w: &infernexv1alpha1.InferenceEngineWorkloadSpec{}, want: false},
		{name: "empty worker", w: &infernexv1alpha1.InferenceEngineWorkloadSpec{Worker: &corev1.PodTemplateSpec{}}, want: false},
		{name: "worker with container", w: &infernexv1alpha1.InferenceEngineWorkloadSpec{
			Worker: &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "w", Image: "i:v1"}}},
			},
		}, want: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := WorkloadWorkerTemplateEffective(tc.w) != nil
			if got != tc.want {
				t.Fatalf("effective = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestMergeInferenceEngineWorkloadSpecWorker(t *testing.T) {
	t.Parallel()
	dst := &infernexv1alpha1.InferenceEngineWorkloadSpec{
		Template: testTemplate("dst:v1"),
	}
	src := &infernexv1alpha1.InferenceEngineWorkloadSpec{
		Worker: &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "w", Image: "src:v1"}}},
		},
	}
	mergeInferenceEngineWorkloadSpec(dst, src)
	if dst.Worker == nil || len(dst.Worker.Spec.Containers) != 1 || dst.Worker.Spec.Containers[0].Image != "src:v1" {
		t.Fatalf("expected worker merged from src, got %#v", dst.Worker)
	}
}

func lwsTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	return rbacTestScheme(t)
}
