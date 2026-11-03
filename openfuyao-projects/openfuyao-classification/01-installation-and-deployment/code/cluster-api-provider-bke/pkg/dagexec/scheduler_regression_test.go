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
	"errors"
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func TestExecuteDAG_ValidationErrors(t *testing.T) {
	var s *Scheduler
	if err := s.ExecuteDAG(context.Background(), nil, nil); err == nil {
		t.Fatal("expected nil scheduler error")
	}
	sched := NewScheduler(Config{})
	if err := sched.ExecuteDAG(context.Background(), nil, nil); err == nil {
		t.Fatal("expected nil dag error")
	}
	dag := topology.NewUpgradeDAG()
	if err := sched.ExecuteDAG(context.Background(), nil, dag); err == nil {
		t.Fatal("expected nil execCtx error")
	}
	execCtx := NewExecutionContext(NewExecutionContextOptions{})
	if err := sched.ExecuteDAG(context.Background(), execCtx, dag); err == nil {
		t.Fatal("expected nil cluster error")
	}
}

func TestExecuteDAG_RegistryInlineSuccessWritesLifecycle(t *testing.T) {
	runner := &recordingInlineRunner{}
	updater := &recordingStatusUpdater{}
	sched := NewScheduler(Config{
		InlineRunner: runner,
		CVStore: mapCVStore{
			"etcd@v3.5.12": {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "etcd", Version: "v3.5.12", Type: cvv1alpha1.ComponentTypeInline,
				},
			},
		},
	})

	dag := topology.NewUpgradeDAG()
	_ = dag.AddNode(&topology.ComponentNode{
		Name:    "etcd",
		Version: "v3.5.12",
		Inline:  &topology.InlineRef{Handler: "EnsureEtcdUpgrade", Version: "v3.5.12"},
	})
	vc := upgrade.NewVersionContext()
	vc.SetCurrent("etcd", "v3.5.10")
	vc.SetTarget("etcd", "v3.5.12")

	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:                &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"}},
		VersionContext:         vc,
		ComponentStatusUpdater: updater,
	})
	if err := sched.ExecuteDAG(context.Background(), execCtx, dag); err != nil {
		t.Fatalf("ExecuteDAG: %v", err)
	}
	if runner.calls.Load() != 1 {
		t.Fatalf("expected inline runner once, got %d", runner.calls.Load())
	}
	if updater.pending.Load() != 1 || updater.installed.Load() != 1 {
		t.Fatalf("expected pending+installed, got pending=%d installed=%d",
			updater.pending.Load(), updater.installed.Load())
	}
}

func TestExecuteDAG_FailFastStopsBatch(t *testing.T) {
	sched := NewScheduler(Config{
		YamlExecutor: &failingExecutor{typ: ComponentTypeYAML},
		CVStore: mapCVStore{
			"a@v1": {Spec: cvv1alpha1.ComponentVersionSpec{Name: "a", Version: "v1", Type: cvv1alpha1.ComponentTypeYAML}},
			"b@v1": {Spec: cvv1alpha1.ComponentVersionSpec{Name: "b", Version: "v1", Type: cvv1alpha1.ComponentTypeYAML}},
		},
	})
	dag := topology.NewUpgradeDAG()
	_ = dag.AddNode(&topology.ComponentNode{
		Name:          "a",
		Version:       "v1",
		FailurePolicy: topology.FailurePolicyFailFast,
	})
	_ = dag.AddNode(&topology.ComponentNode{
		Name:    "b",
		Version: "v1",
	})
	_ = dag.AddDependency("a", "b")

	vc := upgrade.NewVersionContext()
	vc.SetTarget("a", "v1")
	vc.SetTarget("b", "v1")
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:        &bkev1beta1.BKECluster{},
		VersionContext: vc,
	})
	err := sched.ExecuteDAG(context.Background(), execCtx, dag)
	if err == nil {
		t.Fatal("expected fail-fast aggregate error")
	}
}

func TestExecuteDAG_YamlExecutionFailureWritesLifecycleFailed(t *testing.T) {
	bc := newUpdaterTestCluster("yaml-fail")
	cli := newUpdaterFakeClient(t, bc)
	updater := NewBKEComponentStatusUpdater(cli)

	sched := NewScheduler(Config{
		YamlExecutor: &errorExecutor{
			typ: ComponentTypeYAML,
			err: fmt.Errorf("apply failed: invalid manifest"),
		},
		CVStore: mapCVStore{
			"coredns@v1": {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "coredns", Version: "v1", Type: cvv1alpha1.ComponentTypeYAML,
				},
			},
		},
	})
	dag := topology.NewUpgradeDAG()
	_ = dag.AddNode(&topology.ComponentNode{
		Name:          "coredns",
		Version:       "v1",
		FailurePolicy: topology.FailurePolicyFailFast,
	})
	vc := upgrade.NewVersionContext()
	vc.SetTarget("coredns", "v1")
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:                bc,
		VersionContext:         vc,
		Client:                 cli,
		ComponentStatusUpdater: updater,
	})

	err := sched.ExecuteDAG(context.Background(), execCtx, dag)
	if err == nil {
		t.Fatal("expected aggregate error")
	}
	st := bc.Status.ClusterComponentStatuses["coredns"]
	if st.Phase != confv1beta1.LifecyclePhaseFailed {
		t.Fatalf("expected phase Failed, got %q", st.Phase)
	}
	if st.Message != "apply failed: invalid manifest" {
		t.Fatalf("unexpected failure message: %q", st.Message)
	}
	if st.LastTransitionTime == nil {
		t.Fatal("expected LastTransitionTime to be set")
	}
}

func TestExecuteDAG_YamlHealthCheckFailureWritesLifecycleFailed(t *testing.T) {
	bc := newUpdaterTestCluster("yaml-health-fail")
	cli := newUpdaterFakeClient(t, bc)
	updater := NewBKEComponentStatusUpdater(cli)

	sched := NewScheduler(Config{
		YamlExecutor: &errorExecutor{
			typ: ComponentTypeYAML,
			err: fmt.Errorf("healthcheck failed: PodReady timeout"),
		},
		CVStore: mapCVStore{
			"calico@v1": {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "calico", Version: "v1", Type: cvv1alpha1.ComponentTypeYAML,
				},
			},
		},
	})
	dag := topology.NewUpgradeDAG()
	_ = dag.AddNode(&topology.ComponentNode{
		Name:          "calico",
		Version:       "v1",
		FailurePolicy: topology.FailurePolicyFailFast,
	})
	vc := upgrade.NewVersionContext()
	vc.SetTarget("calico", "v1")
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:                bc,
		VersionContext:         vc,
		Client:                 cli,
		ComponentStatusUpdater: updater,
	})

	err := sched.ExecuteDAG(context.Background(), execCtx, dag)
	if err == nil {
		t.Fatal("expected aggregate error")
	}
	st := bc.Status.ClusterComponentStatuses["calico"]
	if st.Phase != confv1beta1.LifecyclePhaseFailed {
		t.Fatalf("expected phase Failed, got %q", st.Phase)
	}
	if st.Message != "healthcheck failed: PodReady timeout" {
		t.Fatalf("unexpected failure message: %q", st.Message)
	}
	if st.LastTransitionTime == nil {
		t.Fatal("expected LastTransitionTime to be set")
	}
}

func TestExecuteDAG_YamlHealthCheckSuccessWritesLifecycleInstalled(t *testing.T) {
	bc := newUpdaterTestCluster("yaml-health-ok")
	cli := newUpdaterFakeClient(t, bc)
	updater := NewBKEComponentStatusUpdater(cli)

	sched := NewScheduler(Config{
		YamlExecutor: &countingExecutor{typ: ComponentTypeYAML},
		CVStore: mapCVStore{
			"cni@v2": {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "cni", Version: "v2", Type: cvv1alpha1.ComponentTypeYAML,
				},
			},
		},
	})
	dag := topology.NewUpgradeDAG()
	_ = dag.AddNode(&topology.ComponentNode{
		Name:          "cni",
		Version:       "v2",
		FailurePolicy: topology.FailurePolicyFailFast,
	})
	vc := upgrade.NewVersionContext()
	vc.SetCurrent("cni", "v1")
	vc.SetTarget("cni", "v2")
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:                bc,
		VersionContext:         vc,
		Client:                 cli,
		ComponentStatusUpdater: updater,
	})

	if err := sched.ExecuteDAG(context.Background(), execCtx, dag); err != nil {
		t.Fatalf("expected success, got error: %v", err)
	}
	st := bc.Status.ClusterComponentStatuses["cni"]
	if st.Phase != confv1beta1.LifecyclePhaseInstalled {
		t.Fatalf("expected phase Installed, got %q", st.Phase)
	}
	if st.CurrentVersion != "v2" {
		t.Fatalf("expected current version v2, got %q", st.CurrentVersion)
	}
	if st.Message != "" {
		t.Fatalf("expected empty message on success, got %q", st.Message)
	}
	if st.LastTransitionTime == nil {
		t.Fatal("expected LastTransitionTime to be set")
	}
}

type failingExecutor struct {
	typ ComponentType
}

func (e *failingExecutor) GetComponentType() ComponentType { return e.typ }

func (e *failingExecutor) ExecuteComponent(context.Context, *topology.ComponentNode, *ExecutionContext) error {
	return fmt.Errorf("exec boom")
}

type errorExecutor struct {
	typ ComponentType
	err error
}

func (e *errorExecutor) GetComponentType() ComponentType { return e.typ }

func (e *errorExecutor) ExecuteComponent(context.Context, *topology.ComponentNode, *ExecutionContext) error {
	return e.err
}

func TestExecuteDAG_ContinueRunsSibling(t *testing.T) {
	okExec := &countingExecutor{typ: ComponentTypeYAML}
	sched := NewScheduler(Config{
		YamlExecutor: &selectingExecutor{
			failName: "bad",
			ok:       okExec,
		},
		CVStore: mapCVStore{
			"bad@v1":  {Spec: cvv1alpha1.ComponentVersionSpec{Name: "bad", Version: "v1", Type: cvv1alpha1.ComponentTypeYAML}},
			"good@v1": {Spec: cvv1alpha1.ComponentVersionSpec{Name: "good", Version: "v1", Type: cvv1alpha1.ComponentTypeYAML}},
		},
	})
	dag := topology.NewUpgradeDAG()
	_ = dag.AddNode(&topology.ComponentNode{
		Name:          "bad",
		Version:       "v1",
		FailurePolicy: topology.FailurePolicyContinue,
	})
	_ = dag.AddNode(&topology.ComponentNode{
		Name:          "good",
		Version:       "v1",
		FailurePolicy: topology.FailurePolicyContinue,
	})

	vc := upgrade.NewVersionContext()
	vc.SetTarget("bad", "v1")
	vc.SetTarget("good", "v1")
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:        &bkev1beta1.BKECluster{},
		VersionContext: vc,
	})
	err := sched.ExecuteDAG(context.Background(), execCtx, dag)
	if err == nil {
		t.Fatal("expected aggregate error from failed component")
	}
	if okExec.calls.Load() != 1 {
		t.Fatalf("expected sibling good to run once, got %d", okExec.calls.Load())
	}
}

type selectingExecutor struct {
	failName string
	ok       *countingExecutor
}

func (e *selectingExecutor) GetComponentType() ComponentType { return ComponentTypeYAML }

func (e *selectingExecutor) ExecuteComponent(
	_ context.Context,
	node *topology.ComponentNode,
	_ *ExecutionContext,
) error {
	if node != nil && node.Name == e.failName {
		return fmt.Errorf("boom")
	}
	return e.ok.ExecuteComponent(context.Background(), node, nil)
}

func TestExecuteComponent_LegacyInlinePath(t *testing.T) {
	runner := &recordingInlineRunner{}
	sched := NewScheduler(Config{InlineRunner: runner})
	node := &topology.ComponentNode{
		Name: "etcd",
		Inline: &topology.InlineRef{
			Handler: "EnsureEtcdUpgrade",
			Version: "",
		},
	}
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster: &bkev1beta1.BKECluster{},
	})
	if err := sched.executeComponent(componentExecution{
		ctx: context.Background(), execCtx: execCtx, newCluster: execCtx.Cluster,
		node: node, tmpl: manifest.TemplateContext{},
	}); err != nil {
		t.Fatalf("legacy inline: %v", err)
	}
	if runner.calls.Load() != 1 {
		t.Fatalf("expected legacy inline runner, got %d", runner.calls.Load())
	}
	if runner.version != defaultComponentVersion {
		t.Fatalf("expected default version, got %q", runner.version)
	}
}

func TestExecuteInline_Errors(t *testing.T) {
	sched := NewScheduler(Config{})
	err := sched.executeInline(context.Background(), nil, nil, &topology.ComponentNode{
		Name:   "x",
		Inline: &topology.InlineRef{Handler: ""},
	})
	if err == nil {
		t.Fatal("expected missing handler error")
	}
	err = sched.executeInline(context.Background(), nil, nil, &topology.ComponentNode{
		Name:   "x",
		Inline: &topology.InlineRef{Handler: "H"},
	})
	if err == nil {
		t.Fatal("expected nil runner error")
	}
}

func TestExecuteManifest_ErrorPaths(t *testing.T) {
	sched := NewScheduler(Config{})
	if err := sched.executeManifest(context.Background(), nil, nil, manifest.TemplateContext{}); err == nil {
		t.Fatal("expected nil node error")
	}
	node := &topology.ComponentNode{Name: "coredns", Version: "v1"}
	vc := upgrade.NewVersionContext()
	vc.SetTarget("coredns", "v1")
	execCtx := NewExecutionContext(NewExecutionContextOptions{VersionContext: vc})
	if err := sched.executeManifest(context.Background(), execCtx, node, manifest.TemplateContext{}); err == nil {
		t.Fatal("expected missing store error")
	}

	sched = NewScheduler(Config{ManifestStore: emptyManifestStore{}})
	if err := sched.executeManifest(context.Background(), execCtx, node, manifest.TemplateContext{}); err == nil {
		t.Fatal("expected empty manifests error")
	}

	sched = NewScheduler(Config{ManifestStore: skipManifestStore{}})
	if err := sched.executeManifest(context.Background(), execCtx, node, manifest.TemplateContext{}); err == nil {
		t.Fatal("expected missing applier error")
	}
}

type emptyManifestStore struct{}

func (emptyManifestStore) GetComponentManifests(context.Context, string, string, manifest.TemplateContext) (*manifest.ComponentPackage, error) {
	return &manifest.ComponentPackage{Name: "coredns", Version: "v1"}, nil
}

func TestMaxParallelAndNodeVersionKey(t *testing.T) {
	sched := NewScheduler(Config{MaxParallelPerBatch: 2})
	if got := sched.maxParallel(0); got != 1 {
		t.Fatalf("maxParallel(0)=%d", got)
	}
	if got := sched.maxParallel(10); got != 2 {
		t.Fatalf("maxParallel capped=%d", got)
	}
	sched.MaxParallelPerBatch = 0
	if got := sched.maxParallel(100); got != defaultMaxParallelPerBatch {
		t.Fatalf("default parallel=%d", got)
	}
	if got := sched.nodeVersionKey(nil); got != defaultComponentVersion {
		t.Fatalf("nil node version=%q", got)
	}
	if got := sched.nodeVersionKey(&topology.ComponentNode{Inline: &topology.InlineRef{}}); got != defaultComponentVersion {
		t.Fatalf("empty inline version=%q", got)
	}
	if got := sched.nodeVersionKey(&topology.ComponentNode{Inline: &topology.InlineRef{Version: "v9"}}); got != "v9" {
		t.Fatalf("inline version=%q", got)
	}
	if got := sched.nodeVersionKey(&topology.ComponentNode{Version: "v8"}); got != "v8" {
		t.Fatalf("node version=%q", got)
	}
}

func TestComponentStatusVersion(t *testing.T) {
	if got := componentStatusVersion(nil); got != defaultComponentVersion {
		t.Fatalf("nil=%q", got)
	}
	if got := componentStatusVersion(&topology.ComponentNode{Version: "v1"}); got != "v1" {
		t.Fatalf("version=%q", got)
	}
	if got := componentStatusVersion(&topology.ComponentNode{Inline: &topology.InlineRef{Version: "v2"}}); got != "v2" {
		t.Fatalf("inline=%q", got)
	}
}

func TestPersistBatchResults_SkipNotInstalled(t *testing.T) {
	updater := &recordingStatusUpdater{}
	sched := NewScheduler(Config{})
	node := &topology.ComponentNode{Name: "coredns", Version: "v1"}
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:                &bkev1beta1.BKECluster{},
		ComponentStatusUpdater: updater,
	})
	errs, failFast := sched.persistBatchResults(context.Background(), execCtx, []componentResult{
		{name: node.Name, node: node, err: manifest.NewSkipNotInstalledError("coredns"), viaRegistry: true},
	}, nil)
	if len(errs) != 0 || failFast {
		t.Fatalf("skip-not-installed should not persist failure: errs=%v failFast=%v", errs, failFast)
	}
	if updater.failed.Load() != 0 || updater.installed.Load() != 0 {
		t.Fatal("updater must not be called for skip-not-installed")
	}
	if updater.cleared.Load() != 1 {
		t.Fatalf("expected lifecycle clear after skip-not-installed, got cleared=%d", updater.cleared.Load())
	}
}

func TestNewScheduler_RegisterFailureFallsBack(t *testing.T) {
	sched := NewScheduler(Config{YamlExecutor: &emptyTypeExecutor{}})
	if sched.Registry.Has(ComponentTypeYAML) {
		t.Fatal("empty-type executor must not register")
	}
}

type emptyTypeExecutor struct{}

func (emptyTypeExecutor) GetComponentType() ComponentType { return "" }
func (emptyTypeExecutor) ExecuteComponent(context.Context, *topology.ComponentNode, *ExecutionContext) error {
	return nil
}

func TestLoggerFacade(t *testing.T) {
	l := NewLogger(nil)
	l.Info("info %s", "x")
	l.Warn("warn %s", "x")
	l.Error("error %s", "x")
	if loggerFrom(nil) == nil {
		t.Fatal("loggerFrom nil")
	}
	_ = loggerFrom(&ExecutionContext{})
}

func TestMarkLifecycleHelpersNilSafe(t *testing.T) {
	sched := NewScheduler(Config{})
	if err := sched.markLifecyclePending(context.Background(), nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := sched.markLifecycleInstalled(context.Background(), nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := sched.markLifecycleFailed(context.Background(), nil, nil, errors.New("x")); err != nil {
		t.Fatal(err)
	}
	if err := sched.markLifecycleClear(context.Background(), nil, nil); err != nil {
		t.Fatal(err)
	}
}

func TestResolveComponentType_NilNodeAndEmptyCV(t *testing.T) {
	sched := NewScheduler(Config{CVStore: mapCVStore{}})
	if _, ok := sched.resolveComponentType(context.Background(), nil); ok {
		t.Fatal("nil node")
	}
	node := &topology.ComponentNode{Name: "x", Version: "v1"}
	if _, ok := sched.resolveComponentType(context.Background(), node); ok {
		t.Fatal("missing cv should be unresolved")
	}
}

func TestExecuteDAG_SkipsWhenDeclarativeCompleted(t *testing.T) {
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
				Completed: []confv1beta1.DeclarativeUpgradeComponentRecord{{
					Name:        "etcd",
					Version:     "v1",
					CompletedAt: metav1.Now(),
				}},
			},
		},
	}
	vc := upgrade.NewVersionContext()
	vc.SetTarget("etcd", "v1")
	vc.SetCurrent("etcd", "v0")
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:        bc,
		VersionContext: vc,
	})
	if err := sched.ExecuteDAG(context.Background(), execCtx, dag); err != nil {
		t.Fatal(err)
	}
	if runner.calls.Load() != 0 {
		t.Fatalf("completed component must be skipped, got %d calls", runner.calls.Load())
	}
}
