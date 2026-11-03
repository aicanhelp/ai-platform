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

package yamlinstaller_test

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/dagexec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/yamlinstaller"
)

type capturingApplier struct {
	applyCalls  atomic.Int32
	applyErr    error
	deleteCalls atomic.Int32
}

func (a *capturingApplier) ApplyComponent(context.Context, *manifest.ComponentPackage) error {
	a.applyCalls.Add(1)
	return a.applyErr
}

func (a *capturingApplier) DeleteComponent(context.Context, *manifest.ComponentPackage) error {
	a.deleteCalls.Add(1)
	return nil
}

func (a *capturingApplier) PruneResources(context.Context, map[string]string, string, [][]byte) error {
	return nil
}

type staticStore struct {
	pkg *manifest.ComponentPackage
}

func (s staticStore) GetComponentManifests(
	context.Context, string, string, manifest.TemplateContext,
) (*manifest.ComponentPackage, error) {
	return s.pkg, nil
}

type mapCVStore map[string]*cvv1alpha1.ComponentVersion

func (m mapCVStore) GetComponentVersion(_ context.Context, name, version string) (*cvv1alpha1.ComponentVersion, error) {
	cv, ok := m[name+"@"+version]
	if !ok {
		return nil, errors.New("not found")
	}
	return cv, nil
}

type recordingStatusUpdater struct {
	pending   atomic.Int32
	installed atomic.Int32
	failed    atomic.Int32
	cleared   atomic.Int32
}

func (r *recordingStatusUpdater) MarkPending(context.Context, dagexec.ComponentMarkRef) error {
	r.pending.Add(1)
	return nil
}
func (r *recordingStatusUpdater) MarkInstalled(context.Context, dagexec.ComponentMarkRef, string) error {
	r.installed.Add(1)
	return nil
}
func (r *recordingStatusUpdater) MarkFailed(context.Context, dagexec.ComponentMarkRef, error) error {
	r.failed.Add(1)
	return nil
}
func (r *recordingStatusUpdater) MarkRollingBack(context.Context, dagexec.ComponentMarkRef) error {
	return nil
}
func (r *recordingStatusUpdater) MarkUninstalling(context.Context, dagexec.ComponentMarkRef) error {
	return nil
}
func (r *recordingStatusUpdater) MarkRemoved(context.Context, dagexec.ComponentMarkRef) error {
	return nil
}
func (r *recordingStatusUpdater) ClearComponentStatus(context.Context, dagexec.ComponentMarkRef) error {
	r.cleared.Add(1)
	return nil
}

func yamlExecutorForSchedulerTest(applier manifest.Applier, version string) *yamlinstaller.YamlComponentExecutor {
	cv := &cvv1alpha1.ComponentVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "coredns"},
		Spec: cvv1alpha1.ComponentVersionSpec{
			Name:    "coredns",
			Version: version,
			Type:    cvv1alpha1.ComponentTypeYAML,
		},
	}
	return &yamlinstaller.YamlComponentExecutor{
		Installer: yamlinstaller.NewYamlInstaller(yamlinstaller.YamlInstallerConfig{
			Store: staticStore{pkg: &manifest.ComponentPackage{
				Name:      "coredns",
				Version:   version,
				Manifests: [][]byte{[]byte("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: coredns\n")},
			}},
			Applier: applier,
		}),
		CVStore: mapCVStore{"coredns@" + version: cv},
	}
}

func TestExecuteDAG_YamlComponentExecutor_Installed(t *testing.T) {
	applier := &capturingApplier{}
	exec := yamlExecutorForSchedulerTest(applier, "v1.1.0")
	updater := &recordingStatusUpdater{}

	sched := dagexec.NewScheduler(dagexec.Config{
		YamlExecutor: exec,
		CVStore: mapCVStore{
			"coredns@v1.1.0": {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "coredns", Version: "v1.1.0", Type: cvv1alpha1.ComponentTypeYAML,
				},
			},
		},
	})
	dag := topology.NewUpgradeDAG()
	_ = dag.AddNode(&topology.ComponentNode{Name: "coredns", Version: "v1.1.0"})

	vc := upgrade.NewVersionContext()
	vc.SetCurrent("coredns", "v1.0.0")
	vc.SetTarget("coredns", "v1.1.0")
	execCtx := dagexec.NewExecutionContext(dagexec.NewExecutionContextOptions{
		Cluster:                &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"}},
		VersionContext:         vc,
		ComponentStatusUpdater: updater,
	})
	if err := sched.ExecuteDAG(context.Background(), execCtx, dag); err != nil {
		t.Fatalf("ExecuteDAG: %v", err)
	}
	if applier.applyCalls.Load() != 1 {
		t.Fatalf("expected apply once, got %d", applier.applyCalls.Load())
	}
	if updater.pending.Load() != 1 || updater.installed.Load() != 1 || updater.failed.Load() != 0 {
		t.Fatalf("lifecycle pending=%d installed=%d failed=%d",
			updater.pending.Load(), updater.installed.Load(), updater.failed.Load())
	}
}

func TestExecuteDAG_YamlComponentExecutor_Failed(t *testing.T) {
	applier := &capturingApplier{applyErr: errors.New("ssa conflict")}
	exec := yamlExecutorForSchedulerTest(applier, "v1.1.0")
	updater := &recordingStatusUpdater{}

	sched := dagexec.NewScheduler(dagexec.Config{
		YamlExecutor: exec,
		CVStore: mapCVStore{
			"coredns@v1.1.0": {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "coredns", Version: "v1.1.0", Type: cvv1alpha1.ComponentTypeYAML,
				},
			},
		},
	})
	dag := topology.NewUpgradeDAG()
	_ = dag.AddNode(&topology.ComponentNode{
		Name: "coredns", Version: "v1.1.0", FailurePolicy: topology.FailurePolicyFailFast,
	})
	vc := upgrade.NewVersionContext()
	vc.SetCurrent("coredns", "v1.0.0")
	vc.SetTarget("coredns", "v1.1.0")
	execCtx := dagexec.NewExecutionContext(dagexec.NewExecutionContextOptions{
		Cluster:                &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"}},
		VersionContext:         vc,
		ComponentStatusUpdater: updater,
	})
	err := sched.ExecuteDAG(context.Background(), execCtx, dag)
	if err == nil {
		t.Fatal("expected aggregate error")
	}
	if updater.failed.Load() != 1 {
		t.Fatalf("expected MarkFailed once, got %d", updater.failed.Load())
	}
	if !strings.Contains(err.Error(), "ssa conflict") {
		t.Fatalf("expected ssa conflict in aggregate error, got %v", err)
	}
}

func TestExecuteDAG_YamlComponentExecutor_SkipVersionsMatch(t *testing.T) {
	applier := &capturingApplier{}
	exec := yamlExecutorForSchedulerTest(applier, "v1.0.0")
	updater := &recordingStatusUpdater{}

	sched := dagexec.NewScheduler(dagexec.Config{
		YamlExecutor: exec,
		CVStore: mapCVStore{
			"coredns@v1.0.0": {
				Spec: cvv1alpha1.ComponentVersionSpec{
					Name: "coredns", Version: "v1.0.0", Type: cvv1alpha1.ComponentTypeYAML,
				},
			},
		},
	})
	dag := topology.NewUpgradeDAG()
	_ = dag.AddNode(&topology.ComponentNode{Name: "coredns", Version: "v1.0.0"})

	vc := upgrade.NewVersionContext()
	vc.SetCurrent("coredns", "v1.0.0")
	vc.SetTarget("coredns", "v1.0.0")
	execCtx := dagexec.NewExecutionContext(dagexec.NewExecutionContextOptions{
		Cluster:                &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"}},
		VersionContext:         vc,
		ComponentStatusUpdater: updater,
	})
	if err := sched.ExecuteDAG(context.Background(), execCtx, dag); err != nil {
		t.Fatalf("ExecuteDAG: %v", err)
	}
	if applier.applyCalls.Load() != 0 {
		t.Fatalf("skip must not apply, calls=%d", applier.applyCalls.Load())
	}
	if updater.pending.Load() != 0 || updater.installed.Load() != 0 {
		t.Fatalf("version-matched component is filtered before executor, pending=%d installed=%d",
			updater.pending.Load(), updater.installed.Load())
	}
}
