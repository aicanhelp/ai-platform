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
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func TestExecuteDAG_ProviderSkipStillEnsuresWebhook(t *testing.T) {
	var calls atomic.Int32
	prev := ensureProviderWebhook
	ensureProviderWebhook = func(context.Context, client.Client) error {
		calls.Add(1)
		return nil
	}
	t.Cleanup(func() { ensureProviderWebhook = prev })

	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			DeclarativeUpgrade: &confv1beta1.DeclarativeUpgradeStatus{
				TargetVersion: "v26.09",
				Completed: []confv1beta1.DeclarativeUpgradeComponentRecord{
					{Name: upgrade.ComponentProvider, Version: "26.9.0", CompletedAt: metav1.Now()},
				},
			},
		},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	cli := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc).WithStatusSubresource(bc).Build()

	dag, err := topology.BuildUpgradeDAG(
		[]cvv1alpha1.ReleaseImageUpgradeComponent{{Name: upgrade.ComponentProvider, Version: "26.9.0"}},
		topology.DefaultDependencyResolver(),
	)
	require.NoError(t, err)

	sched := NewScheduler(Config{})
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		OldCluster: bc,
		Cluster:    bc,
		Client:     cli,
	})
	require.NoError(t, sched.ExecuteDAG(context.Background(), execCtx, dag))
	assert.Equal(t, int32(1), calls.Load())
}

func TestExecuteDAG_ProviderVersionMatchStillEnsuresWebhook(t *testing.T) {
	var calls atomic.Int32
	prev := ensureProviderWebhook
	ensureProviderWebhook = func(context.Context, client.Client) error {
		calls.Add(1)
		return nil
	}
	t.Cleanup(func() { ensureProviderWebhook = prev })

	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			DeclarativeUpgrade: &confv1beta1.DeclarativeUpgradeStatus{TargetVersion: "v26.09"},
		},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	cli := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc).WithStatusSubresource(bc).Build()

	dag, err := topology.BuildUpgradeDAG(
		[]cvv1alpha1.ReleaseImageUpgradeComponent{{Name: upgrade.ComponentProvider, Version: "26.9.0"}},
		topology.DefaultDependencyResolver(),
	)
	require.NoError(t, err)

	vc := upgrade.NewVersionContext()
	vc.SetTarget(upgrade.ComponentProvider, "26.9.0")
	vc.SetCurrent(upgrade.ComponentProvider, "26.9.0")

	sched := NewScheduler(Config{})
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		OldCluster:     bc,
		Cluster:        bc,
		Client:         cli,
		VersionContext: vc,
	})
	require.NoError(t, sched.ExecuteDAG(context.Background(), execCtx, dag))
	assert.Equal(t, int32(1), calls.Load())
}

func TestExecuteDAG_ProviderApplySuccessEnsuresWebhook(t *testing.T) {
	var calls atomic.Int32
	prev := ensureProviderWebhook
	ensureProviderWebhook = func(context.Context, client.Client) error {
		calls.Add(1)
		return nil
	}
	t.Cleanup(func() { ensureProviderWebhook = prev })

	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			DeclarativeUpgrade: &confv1beta1.DeclarativeUpgradeStatus{TargetVersion: "v26.09"},
		},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	cli := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc).WithStatusSubresource(bc).Build()

	dag, err := topology.BuildUpgradeDAG(
		[]cvv1alpha1.ReleaseImageUpgradeComponent{{Name: upgrade.ComponentProvider, Version: "26.9.0"}},
		topology.DefaultDependencyResolver(),
	)
	require.NoError(t, err)

	vc := upgrade.NewVersionContext()
	vc.SetTarget(upgrade.ComponentProvider, "26.9.0")

	cvStore := mapCVStore{
		"provider@26.9.0": {
			Spec: cvv1alpha1.ComponentVersionSpec{
				Name:    upgrade.ComponentProvider,
				Version: "26.9.0",
				Type:    cvv1alpha1.ComponentTypeYAML,
			},
		},
	}
	yamlExec := &recordingYAMLExecutor{}
	sched := NewScheduler(Config{
		YamlExecutor: yamlExec,
		CVStore:      cvStore,
	})
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		OldCluster:     bc,
		Cluster:        bc,
		Client:         cli,
		VersionContext: vc,
	})
	require.NoError(t, sched.ExecuteDAG(context.Background(), execCtx, dag))
	assert.Equal(t, 1, yamlExec.calls)
	assert.Equal(t, int32(1), calls.Load())
}

func TestExecuteDAG_ProviderWebhookEnsureErrorFails(t *testing.T) {
	prev := ensureProviderWebhook
	ensureProviderWebhook = func(context.Context, client.Client) error {
		return errors.New("ensure failed")
	}
	t.Cleanup(func() { ensureProviderWebhook = prev })

	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			DeclarativeUpgrade: &confv1beta1.DeclarativeUpgradeStatus{
				TargetVersion: "v26.09",
				Completed: []confv1beta1.DeclarativeUpgradeComponentRecord{
					{Name: upgrade.ComponentProvider, Version: "26.9.0", CompletedAt: metav1.Now()},
				},
			},
		},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	cli := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc).WithStatusSubresource(bc).Build()

	dag, err := topology.BuildUpgradeDAG(
		[]cvv1alpha1.ReleaseImageUpgradeComponent{{
			Name:    upgrade.ComponentProvider,
			Version: "26.9.0",
		}},
		topology.DefaultDependencyResolver(),
	)
	require.NoError(t, err)
	// Default FailurePolicy is FailFast for nodes from BuildUpgradeDAG.
	sched := NewScheduler(Config{})
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		OldCluster: bc,
		Cluster:    bc,
		Client:     cli,
	})
	err = sched.ExecuteDAG(context.Background(), execCtx, dag)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ensure bkenode/vbkenode webhooks")
}

type recordingYAMLExecutor struct {
	calls int
}

func (e *recordingYAMLExecutor) GetComponentType() ComponentType { return ComponentTypeYAML }

func (e *recordingYAMLExecutor) ExecuteComponent(
	context.Context, *topology.ComponentNode, *ExecutionContext,
) error {
	e.calls++
	return nil
}

func (e *recordingYAMLExecutor) UninstallComponent(
	context.Context, *topology.ComponentNode, *ExecutionContext,
) error {
	return nil
}
