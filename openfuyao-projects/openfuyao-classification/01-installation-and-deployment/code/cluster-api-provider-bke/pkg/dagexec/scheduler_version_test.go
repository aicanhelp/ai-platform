/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package dagexec

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func TestManifestNeedsUpgrade(t *testing.T) {
	t.Parallel()

	vc := upgrade.NewVersionContext()
	vc.SetTarget(upgrade.ComponentCoreDNS, "v1.0.0")
	vc.SetCurrent(upgrade.ComponentCoreDNS, "v1.0.0")

	if manifestNeedsUpgrade(vc, upgrade.ComponentCoreDNS) {
		t.Fatal("expected false when current equals target")
	}
	if manifestNeedsUpgrade(vc, upgrade.ComponentKubeProxy) {
		t.Fatal("expected false when component has no target in VersionContext")
	}
	if !manifestNeedsUpgrade(nil, upgrade.ComponentCoreDNS) {
		t.Fatal("expected true when version context is nil")
	}

	vc.SetCurrent(upgrade.ComponentCoreDNS, "v0.9.0")
	if !manifestNeedsUpgrade(vc, upgrade.ComponentCoreDNS) {
		t.Fatal("expected true when current differs from target")
	}
}

type countingManifestApplier struct {
	calls atomic.Int32
}

func (a *countingManifestApplier) ApplyComponent(context.Context, *manifest.ComponentPackage) error {
	a.calls.Add(1)
	return nil
}

func (a *countingManifestApplier) DeleteComponent(context.Context, *manifest.ComponentPackage) error {
	return nil
}

func (a *countingManifestApplier) PruneResources(context.Context, map[string]string, string, [][]byte) error {
	return nil
}

func TestExecuteManifest_SkipsApplyWhenVersionMatched(t *testing.T) {
	vc := upgrade.NewVersionContext()
	vc.SetTarget(upgrade.ComponentCoreDNS, "v1.0.0")
	vc.SetCurrent(upgrade.ComponentCoreDNS, "v1.0.0")

	applier := &countingManifestApplier{}
	sched := NewScheduler(Config{
		ManifestStore:   skipManifestStore{},
		ManifestApplier: applier,
	})
	execCtx := NewExecutionContext(NewExecutionContextOptions{VersionContext: vc})

	node := &topology.ComponentNode{Name: upgrade.ComponentCoreDNS, Version: "v1.0.0"}
	if err := sched.executeManifest(context.Background(), execCtx, node, manifest.TemplateContext{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if applier.calls.Load() != 0 {
		t.Fatalf("expected manifest applier not called, got %d calls", applier.calls.Load())
	}
}

func TestExecuteManifest_AppliesWhenVersionDiffers(t *testing.T) {
	vc := upgrade.NewVersionContext()
	vc.SetTarget(upgrade.ComponentCoreDNS, "v1.0.0")
	vc.SetCurrent(upgrade.ComponentCoreDNS, "v0.9.0")

	applier := &countingManifestApplier{}
	sched := NewScheduler(Config{
		ManifestStore:   skipManifestStore{},
		ManifestApplier: applier,
	})
	execCtx := NewExecutionContext(NewExecutionContextOptions{VersionContext: vc})

	node := &topology.ComponentNode{Name: upgrade.ComponentCoreDNS, Version: "v1.0.0"}
	if err := sched.executeManifest(context.Background(), execCtx, node, manifest.TemplateContext{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if applier.calls.Load() != 1 {
		t.Fatalf("expected manifest applier called once, got %d", applier.calls.Load())
	}
}

type failManifestApplier struct{}

func (failManifestApplier) ApplyComponent(context.Context, *manifest.ComponentPackage) error {
	return fmt.Errorf("apply should not run")
}

func (failManifestApplier) DeleteComponent(context.Context, *manifest.ComponentPackage) error {
	return nil
}

func (failManifestApplier) PruneResources(context.Context, map[string]string, string, [][]byte) error {
	return nil
}

func TestExecuteManifest_VersionMatched_DoesNotReachApplier(t *testing.T) {
	vc := upgrade.NewVersionContext()
	vc.SetTarget(upgrade.ComponentProvider, "v1.0.0")
	vc.SetCurrent(upgrade.ComponentProvider, "v1.0.0")

	sched := NewScheduler(Config{
		ManifestStore:   skipManifestStore{},
		ManifestApplier: failManifestApplier{},
	})
	execCtx := NewExecutionContext(NewExecutionContextOptions{VersionContext: vc})

	node := &topology.ComponentNode{Name: upgrade.ComponentProvider, Version: "v1.0.0"}
	if err := sched.executeManifest(context.Background(), execCtx, node, manifest.TemplateContext{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestExecuteDAG_SkipsWhenVersionMatched(t *testing.T) {
	runner := &recordingInlineRunner{}
	sched := NewScheduler(Config{
		InlineRunner: runner,
		CVStore: mapCVStore{
			"etcd@v1": {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "etcd", Version: "v1", Type: cvv1alpha1.ComponentTypeInline,
				},
			},
		},
	})
	dag := topology.NewUpgradeDAG()
	_ = dag.AddNode(&topology.ComponentNode{
		Name:    "etcd",
		Version: "v1",
		Inline:  &topology.InlineRef{Handler: "H", Version: "v1"},
	})
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			DeclarativeUpgrade: &confv1beta1.DeclarativeUpgradeStatus{
				TargetVersion: "v2",
			},
		},
	}
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	cli := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc).WithStatusSubresource(bc).Build()

	vc := upgrade.NewVersionContext()
	vc.SetTarget("etcd", "v1")
	vc.SetCurrent("etcd", "v1")
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:        bc,
		Client:         cli,
		VersionContext: vc,
	})
	if err := sched.ExecuteDAG(context.Background(), execCtx, dag); err != nil {
		t.Fatal(err)
	}
	if runner.calls.Load() != 0 {
		t.Fatalf("version-matched component must be skipped, got %d calls", runner.calls.Load())
	}
	if len(bc.Status.DeclarativeUpgrade.Completed) != 0 {
		t.Fatalf("expected no completed records, got %v", bc.Status.DeclarativeUpgrade.Completed)
	}
}
