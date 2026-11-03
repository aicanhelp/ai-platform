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

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
)

type countingExecutor struct {
	typ   ComponentType
	calls atomic.Int32
}

func (e *countingExecutor) GetComponentType() ComponentType { return e.typ }

func (e *countingExecutor) ExecuteComponent(
	context.Context,
	*topology.ComponentNode,
	*ExecutionContext,
) error {
	e.calls.Add(1)
	return nil
}

type mapCVStore map[string]*cvv1alpha1.ComponentVersion

func (m mapCVStore) GetComponentVersion(_ context.Context, name, version string) (*cvv1alpha1.ComponentVersion, error) {
	key := name + "@" + version
	cv, ok := m[key]
	if !ok {
		return nil, fmt.Errorf("cv %s not found", key)
	}
	return cv, nil
}

func TestNewScheduler_RegistersInlineAlwaysAndOptionalYamlHelm(t *testing.T) {
	yamlExec := &countingExecutor{typ: ComponentTypeYAML}
	helmExec := &countingExecutor{typ: ComponentTypeHelm}

	sched := NewScheduler(Config{
		YamlExecutor: yamlExec,
		HelmExecutor: helmExec,
	})
	if !sched.Registry.Has(ComponentTypeInline) {
		t.Fatal("inline must always be registered")
	}
	if !sched.Registry.Has(ComponentTypeYAML) || !sched.Registry.Has(ComponentTypeHelm) {
		t.Fatal("yaml/helm should be registered when provided")
	}

	schedOnlyInline := NewScheduler(Config{})
	if !schedOnlyInline.Registry.Has(ComponentTypeInline) {
		t.Fatal("inline must always be registered")
	}
	if schedOnlyInline.Registry.Has(ComponentTypeYAML) || schedOnlyInline.Registry.Has(ComponentTypeHelm) {
		t.Fatal("yaml/helm must not register when nil")
	}
}

func TestExecuteComponent_RegistryHitUsesExecutor(t *testing.T) {
	yamlExec := &countingExecutor{typ: ComponentTypeYAML}
	applier := &countingManifestApplier{}
	sched := NewScheduler(Config{
		YamlExecutor:    yamlExec,
		ManifestStore:   skipManifestStore{},
		ManifestApplier: applier,
		CVStore: mapCVStore{
			"coredns@v1.0.0": {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "coredns", Version: "v1.0.0", Type: cvv1alpha1.ComponentTypeYAML,
				},
			},
		},
	})
	node := &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"}
	execCtx := NewExecutionContext(NewExecutionContextOptions{})
	if err := sched.executeComponent(componentExecution{
		ctx: context.Background(), execCtx: execCtx, node: node, tmpl: manifest.TemplateContext{},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if yamlExec.calls.Load() != 1 {
		t.Fatalf("expected yaml executor called once, got %d", yamlExec.calls.Load())
	}
	if applier.calls.Load() != 0 {
		t.Fatalf("legacy manifest must not run on registry hit, got %d", applier.calls.Load())
	}
}

func TestExecuteComponent_UnregisteredTypeFallsBackToLegacy(t *testing.T) {
	applier := &countingManifestApplier{}
	sched := NewScheduler(Config{
		ManifestStore:   skipManifestStore{},
		ManifestApplier: applier,
		CVStore: mapCVStore{
			"coredns@v1.0.0": {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "coredns", Version: "v1.0.0", Type: cvv1alpha1.ComponentTypeYAML,
				},
			},
		},
	})
	node := &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"}
	execCtx := NewExecutionContext(NewExecutionContextOptions{})
	if err := sched.executeComponent(componentExecution{
		ctx: context.Background(), execCtx: execCtx, node: node, tmpl: manifest.TemplateContext{},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if applier.calls.Load() != 1 {
		t.Fatalf("expected legacy manifest apply once, got %d", applier.calls.Load())
	}
}

func TestExecuteComponent_EmptyTypeFallsBackToLegacy(t *testing.T) {
	applier := &countingManifestApplier{}
	sched := NewScheduler(Config{
		ManifestStore:   skipManifestStore{},
		ManifestApplier: applier,
	})
	node := &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"}
	execCtx := NewExecutionContext(NewExecutionContextOptions{})
	if err := sched.executeComponent(componentExecution{
		ctx: context.Background(), execCtx: execCtx, node: node, tmpl: manifest.TemplateContext{},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if applier.calls.Load() != 1 {
		t.Fatalf("expected legacy manifest apply once, got %d", applier.calls.Load())
	}
}

func TestResolveComponentType_FromCVStore(t *testing.T) {
	store := mapCVStore{
		"coredns@v1.0.0": {
			Spec: cvv1alpha1.ComponentVersionSpec{
				Name:    "coredns",
				Type:    cvv1alpha1.ComponentTypeYAML,
				Version: "v1.0.0",
			},
		},
	}
	sched := NewScheduler(Config{CVStore: store})
	node := &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"}
	typ, ok := sched.resolveComponentType(context.Background(), node)
	if !ok || typ != ComponentTypeYAML {
		t.Fatalf("expected yaml from CVStore, got ok=%v typ=%q", ok, typ)
	}
}

type recordingStatusUpdater struct {
	pending   atomic.Int32
	installed atomic.Int32
	failed    atomic.Int32
	cleared   atomic.Int32
}

func (r *recordingStatusUpdater) MarkPending(context.Context, ComponentMarkRef) error {
	r.pending.Add(1)
	return nil
}
func (r *recordingStatusUpdater) MarkInstalled(context.Context, ComponentMarkRef, string) error {
	r.installed.Add(1)
	return nil
}
func (r *recordingStatusUpdater) MarkFailed(context.Context, ComponentMarkRef, error) error {
	r.failed.Add(1)
	return nil
}
func (r *recordingStatusUpdater) MarkRollingBack(context.Context, ComponentMarkRef) error {
	return nil
}
func (r *recordingStatusUpdater) MarkUninstalling(context.Context, ComponentMarkRef) error {
	return nil
}
func (r *recordingStatusUpdater) MarkRemoved(context.Context, ComponentMarkRef) error {
	return nil
}
func (r *recordingStatusUpdater) ClearComponentStatus(context.Context, ComponentMarkRef) error {
	r.cleared.Add(1)
	return nil
}

func TestPersistBatchResults_RegistryWritesLifecycleStatus(t *testing.T) {
	updater := &recordingStatusUpdater{}
	sched := NewScheduler(Config{})
	node := &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"}
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:                &bkev1beta1.BKECluster{},
		ComponentStatusUpdater: updater,
	})
	_, failFast := sched.persistBatchResults(context.Background(), execCtx, []componentResult{
		{name: node.Name, node: node, viaRegistry: true},
	}, nil)
	if failFast {
		t.Fatal("unexpected fail-fast")
	}
	if updater.installed.Load() != 1 || updater.failed.Load() != 0 {
		t.Fatalf("expected lifecycle installed on registry persist, got installed=%d failed=%d",
			updater.installed.Load(), updater.failed.Load())
	}
}

func TestPersistBatchResults_LegacySkipsLifecycleStatus(t *testing.T) {
	updater := &recordingStatusUpdater{}
	sched := NewScheduler(Config{})
	node := &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"}
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:                &bkev1beta1.BKECluster{},
		ComponentStatusUpdater: updater,
	})
	_, _ = sched.persistBatchResults(context.Background(), execCtx, []componentResult{
		{name: node.Name, node: node, viaRegistry: false},
	}, nil)
	if updater.pending.Load() != 0 || updater.installed.Load() != 0 || updater.failed.Load() != 0 {
		t.Fatalf("legacy must not call ComponentStatusUpdater on persist, got pending=%d installed=%d failed=%d",
			updater.pending.Load(), updater.installed.Load(), updater.failed.Load())
	}
}

func TestPersistBatchResults_RegistryMarksFailed(t *testing.T) {
	updater := &recordingStatusUpdater{}
	sched := NewScheduler(Config{})
	node := &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"}
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:                &bkev1beta1.BKECluster{},
		ComponentStatusUpdater: updater,
	})
	errs, _ := sched.persistBatchResults(context.Background(), execCtx, []componentResult{
		{name: node.Name, node: node, err: fmt.Errorf("exec boom"), viaRegistry: true},
	}, nil)
	if len(errs) == 0 {
		t.Fatal("expected batch error")
	}
	if updater.failed.Load() != 1 || updater.installed.Load() != 0 {
		t.Fatalf("expected lifecycle failed, got installed=%d failed=%d",
			updater.installed.Load(), updater.failed.Load())
	}
}

func TestExecuteComponent_LegacyPathSkipsComponentStatus(t *testing.T) {
	applier := &countingManifestApplier{}
	updater := &recordingStatusUpdater{}
	sched := NewScheduler(Config{
		ManifestStore:   skipManifestStore{},
		ManifestApplier: applier,
	})
	node := &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"}
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster:                &bkev1beta1.BKECluster{},
		ComponentStatusUpdater: updater,
	})
	if err := sched.executeComponent(componentExecution{
		ctx: context.Background(), execCtx: execCtx, node: node, tmpl: manifest.TemplateContext{},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if applier.calls.Load() != 1 {
		t.Fatalf("expected legacy apply once")
	}
	if updater.pending.Load() != 0 || updater.installed.Load() != 0 || updater.failed.Load() != 0 {
		t.Fatalf("executeComponent itself must not call ComponentStatusUpdater, got pending=%d installed=%d failed=%d",
			updater.pending.Load(), updater.installed.Load(), updater.failed.Load())
	}
}

func TestUsesRegistryExecutor(t *testing.T) {
	sched := NewScheduler(Config{
		YamlExecutor: &countingExecutor{typ: ComponentTypeYAML},
		CVStore: mapCVStore{
			"coredns@v1.0.0": {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "coredns", Version: "v1.0.0", Type: cvv1alpha1.ComponentTypeYAML,
				},
			},
		},
	})
	node := &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"}
	if !sched.usesRegistryExecutor(context.Background(), node) {
		t.Fatal("expected registry hit via CVStore")
	}
	legacy := &topology.ComponentNode{Name: "coredns", Version: "v1.0.0"}
	noStore := NewScheduler(Config{YamlExecutor: &countingExecutor{typ: ComponentTypeYAML}})
	if noStore.usesRegistryExecutor(context.Background(), legacy) {
		t.Fatal("without CVStore should not use registry")
	}
}
