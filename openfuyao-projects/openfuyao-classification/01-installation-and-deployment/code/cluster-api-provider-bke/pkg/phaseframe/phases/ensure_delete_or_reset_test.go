/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phases

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestEnsureDeleteOrResetConstants(t *testing.T) {
	assert.Equal(t, "EnsureDeleteOrReset", string(EnsureDeleteOrResetName))
}

func TestNewEnsureDeleteOrReset(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	phase := NewEnsureDeleteOrReset(ctx)
	assert.NotNil(t, phase)

	e, ok := phase.(*EnsureDeleteOrReset)
	assert.True(t, ok)
	assert.Equal(t, EnsureDeleteOrResetName, e.PhaseName)
}

func TestEnsureDeleteOrReset_NeedExecute_DeletionTimestamp(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test",
			Namespace:         "default",
			DeletionTimestamp: &metav1.Time{Time: time.Now()},
		},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureDeleteOrReset{BasePhase: phaseframe.BasePhase{Ctx: ctx, PhaseName: EnsureDeleteOrResetName}}

	result := e.NeedExecute(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.True(t, result)
}

func TestEnsureDeleteOrReset_NeedExecute_Reset(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec:       confv1beta1.BKEClusterSpec{Reset: true},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureDeleteOrReset{BasePhase: phaseframe.BasePhase{Ctx: ctx, PhaseName: EnsureDeleteOrResetName}}

	result := e.NeedExecute(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.True(t, result)
}

func TestEnsureDeleteOrReset_NeedExecute_False(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureDeleteOrReset{BasePhase: phaseframe.BasePhase{Ctx: ctx, PhaseName: EnsureDeleteOrResetName}}

	result := e.NeedExecute(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureDeleteOrResetPostHook_Success(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	phase := &EnsureDeleteOrReset{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	err := ensureDeleteOrResetPostHook(phase, nil)
	assert.NoError(t, err)
}

func TestEnsureDeleteOrResetPostHook_WithError(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	phase := &EnsureDeleteOrReset{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	err := ensureDeleteOrResetPostHook(phase, assert.AnError)
	assert.NoError(t, err)
}

func TestEnsureDeleteOrReset_Execute_ReconcileSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureDeleteOrReset{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "reconcileDelete", func(_ *EnsureDeleteOrReset, _ context.Context) error {
		return nil
	})

	result, err := e.Execute()
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureDeleteOrReset_EnsureClusterStatusDeleting_AlreadyDeleting(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Status:     confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterDeleting},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureDeleteOrReset{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	err := e.ensureClusterStatusDeleting(c, bkeCluster, ctx.Log)
	assert.NoError(t, err)
}

func TestEnsureDeleteOrReset_HandleClusterDeletion_NoCluster(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Cluster:    nil,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureDeleteOrReset{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	err := e.handleClusterDeletion(context.Background(), nil, ctx.Log)
	assert.NoError(t, err)
}

func deleteScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, agentv1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	return scheme
}

func newDeletePhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster, objs ...client.Object) *EnsureDeleteOrReset {
	t.Helper()
	scheme := deleteScheme(t)
	builder := ctrlfake.NewClientBuilder().WithScheme(scheme)
	if bkeCluster != nil {
		builder = builder.WithObjects(bkeCluster)
	}
	for _, o := range objs {
		builder = builder.WithObjects(o)
	}
	c := builder.Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureDeleteOrReset{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

func TestEnsureDeleteOrResetHandleBKEMachineDeletion(t *testing.T) {
	t.Run("no bkeMachines returns nil", func(t *testing.T) {
		e := newDeletePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		err := e.handleBKEMachineDeletion(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.NoError(t, err)
	})

	t.Run("unbootstrapped machine with owner waits for delete", func(t *testing.T) {
		machine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: "m1", Namespace: "ns", OwnerReferences: []metav1.OwnerReference{{Kind: "KubeadmControlPlane"}}},
			Status:     bkev1beta1.BKEMachineStatus{Bootstrapped: false},
		}
		e := newDeletePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}, machine)
		err := e.handleBKEMachineDeletion(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait for bkeMachine delete")
	})

	t.Run("bootstrapped machine with owner waits for delete", func(t *testing.T) {
		machine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: "m1", Namespace: "ns", OwnerReferences: []metav1.OwnerReference{{Kind: "KubeadmControlPlane"}}},
			Status:     bkev1beta1.BKEMachineStatus{Bootstrapped: true},
		}
		e := newDeletePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}, machine)
		err := e.handleBKEMachineDeletion(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait for bkeMachine delete")
	})
}

// ---- deleteRelatedResources (real body) ----

func TestEnsureDeleteOrResetDeleteRelatedResources(t *testing.T) {
	t.Run("empty resources returns nil", func(t *testing.T) {
		e := newDeletePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		require.NoError(t, e.deleteRelatedResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log))
	})

	t.Run("with commands deletes them", func(t *testing.T) {
		cmd := &agentv1beta1.Command{
			ObjectMeta: metav1.ObjectMeta{Name: "cmd1", Namespace: "ns"},
		}
		e := newDeletePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}, cmd)
		require.NoError(t, e.deleteRelatedResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log))
	})
}

// ---- cleanupClusterResources (real body) ----

func TestEnsureDeleteOrResetCleanupClusterResources(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })

	t.Run("success removes finalizer", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name: "c1", Namespace: "ns",
				Finalizers: []string{bkev1beta1.ClusterFinalizer},
			},
		}
		e := newDeletePhaseCov(t, cluster)
		require.NoError(t, e.cleanupClusterResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log))
		assert.Empty(t, e.Ctx.BKECluster.Finalizers)
	})
}

// ---- handleNamespaceDeletion (real body) ----

func TestEnsureDeleteOrResetHandleNamespaceDeletion(t *testing.T) {
	t.Run("annotation false with no other cluster deletes namespace", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name: "c1", Namespace: "ns",
				Annotations: map[string]string{annotation.DeleteIgnoreNamespaceAnnotationKey: "false"},
			},
		}
		e := newDeletePhaseCov(t, cluster)
		require.NoError(t, e.handleNamespaceDeletion(context.Background(), e.Ctx.Client, cluster, e.Ctx.Log))
	})

	t.Run("annotation true ignores", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name: "c1", Namespace: "ns",
				Annotations: map[string]string{annotation.DeleteIgnoreNamespaceAnnotationKey: "true"},
			},
		}
		e := newDeletePhaseCov(t, cluster)
		require.NoError(t, e.handleNamespaceDeletion(context.Background(), e.Ctx.Client, cluster, e.Ctx.Log))
	})

	t.Run("no annotation ignores", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		e := newDeletePhaseCov(t, cluster)
		require.NoError(t, e.handleNamespaceDeletion(context.Background(), e.Ctx.Client, cluster, e.Ctx.Log))
	})
}

func TestEnsureDeleteOrResetReconcileDelete(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newDeletePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		patches.ApplyPrivateMethod(e, "ensureClusterStatusDeleting", func(_ *EnsureDeleteOrReset, _ client.Client, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) error {
			return nil
		})
		patches.ApplyPrivateMethod(e, "handleClusterDeletion", func(_ *EnsureDeleteOrReset, _ context.Context, _ client.Client, _ *bkev1beta1.BKELogger) error {
			return nil
		})
		patches.ApplyPrivateMethod(e, "handleBKEMachineDeletion", func(_ *EnsureDeleteOrReset, _ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) error {
			return nil
		})
		patches.ApplyPrivateMethod(e, "deleteRelatedResources", func(_ *EnsureDeleteOrReset, _ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) error {
			return nil
		})
		patches.ApplyPrivateMethod(e, "cleanupClusterResources", func(_ *EnsureDeleteOrReset, _ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) error {
			return nil
		})
		patches.ApplyPrivateMethod(e, "handleNamespaceDeletion", func(_ *EnsureDeleteOrReset, _ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) error {
			return nil
		})
		require.NoError(t, e.reconcileDelete(context.Background()))
	})

	t.Run("bke machine deletion error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newDeletePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		patches.ApplyPrivateMethod(e, "ensureClusterStatusDeleting", func(_ *EnsureDeleteOrReset, _ client.Client, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) error {
			return nil
		})
		patches.ApplyPrivateMethod(e, "handleClusterDeletion", func(_ *EnsureDeleteOrReset, _ context.Context, _ client.Client, _ *bkev1beta1.BKELogger) error {
			return nil
		})
		patches.ApplyPrivateMethod(e, "handleBKEMachineDeletion", func(_ *EnsureDeleteOrReset, _ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKELogger) error {
			return assertErr("machine delete failed")
		})
		err := e.reconcileDelete(context.Background())
		require.Error(t, err)
		assert.Contains(t, err.Error(), "machine delete failed")
	})
}

// ---- helpers ----

func deleteOrResetGapsScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, agentv1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, clusterv1.AddToScheme(scheme))
	return scheme
}

func deleteOrResetGapsPhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster, objs ...client.Object) *EnsureDeleteOrReset {
	t.Helper()
	scheme := deleteOrResetGapsScheme(t)
	builder := ctrlfake.NewClientBuilder().WithScheme(scheme)
	if bkeCluster != nil {
		builder = builder.WithObjects(bkeCluster)
	}
	for _, o := range objs {
		builder = builder.WithObjects(o)
	}
	c := builder.Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureDeleteOrReset{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

// deleteOrResetGapsPhaseWithClient builds a phase whose client is wrapped.
func deleteOrResetGapsPhaseWithClient(t *testing.T, c client.Client, bkeCluster *bkev1beta1.BKECluster) *EnsureDeleteOrReset {
	t.Helper()
	scheme := deleteOrResetGapsScheme(t)
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureDeleteOrReset{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

// deleteOrResetGapsForbiddenOnBKENodeClient returns Forbidden on DeleteAllOf for BKENode.
type deleteOrResetGapsForbiddenOnBKENodeClient struct {
	client.Client
}

func (f *deleteOrResetGapsForbiddenOnBKENodeClient) DeleteAllOf(ctx context.Context, obj client.Object, opts ...client.DeleteAllOfOption) error {
	if _, ok := obj.(*confv1beta1.BKENode); ok {
		return apierrors.NewForbidden(schema.GroupResource{Group: "bke.bocloud.com", Resource: "bkenodes"}, "", errors.New("forbidden"))
	}
	return f.Client.DeleteAllOf(ctx, obj, opts...)
}

// deleteOrResetGapsListErrClient returns a configurable error on all List calls when listErr is set.
type deleteOrResetGapsListErrClient struct {
	client.Client
	listErr error
}

func (f *deleteOrResetGapsListErrClient) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	if f.listErr != nil {
		return f.listErr
	}
	return f.Client.List(ctx, list, opts...)
}

// deleteOrResetGapsDeleteErrClient returns an error on Delete for clusterv1.Cluster.
type deleteOrResetGapsDeleteErrClient struct {
	client.Client
}

func (f *deleteOrResetGapsDeleteErrClient) Delete(ctx context.Context, obj client.Object, opts ...client.DeleteOption) error {
	if _, ok := obj.(*clusterv1.Cluster); ok {
		return apierrors.NewInternalError(errors.New("delete error"))
	}
	return f.Client.Delete(ctx, obj, opts...)
}

// ---- Execute ----

func TestEnsureDeleteOrResetExecuteGaps(t *testing.T) {
	t.Run("paused resumes cluster and removes commands", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		cmd := &agentv1beta1.Command{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "cmd1",
				Namespace:  "ns",
				Labels:     map[string]string{clusterv1.ClusterNameLabel: "c1"},
				Finalizers: []string{agentv1beta1.CommandFinalizer},
			},
		}
		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec:       confv1beta1.BKEClusterSpec{Pause: true},
		}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster, cmd)

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })

		result, err := e.Execute()
		require.NoError(t, err)
		assert.True(t, result.Requeue)
	})

	t.Run("paused list error still returns requeue", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec:       confv1beta1.BKEClusterSpec{Pause: true},
		}
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
		wrappedC := &deleteOrResetGapsListErrClient{Client: fakeC, listErr: errors.New("list boom")}
		e := deleteOrResetGapsPhaseWithClient(t, wrappedC, bkeCluster)

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })

		result, err := e.Execute()
		require.NoError(t, err)
		assert.True(t, result.Requeue)
	})

	t.Run("paused SyncStatusUntilComplete error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec:       confv1beta1.BKEClusterSpec{Pause: true},
		}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
			return errors.New("sync error")
		})

		result, err := e.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sync error")
		assert.False(t, result.Requeue)
	})

	t.Run("paused RefreshCtxBKECluster error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec:       confv1beta1.BKEClusterSpec{Pause: true},
		}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return errors.New("refresh error") })

		result, err := e.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "refresh error")
		assert.False(t, result.Requeue)
	})

	t.Run("not paused reconcileDelete success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)

		patches.ApplyPrivateMethod(e, "reconcileDelete", func(_ *EnsureDeleteOrReset, _ context.Context) error { return nil })

		result, err := e.Execute()
		require.NoError(t, err)
		assert.False(t, result.Requeue)
	})

	t.Run("not paused reconcileDelete timeout", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)

		patches.ApplyFunc(wait.PollImmediateUntil, func(_ time.Duration, _ wait.ConditionFunc, _ <-chan struct{}) error {
			return wait.ErrWaitTimeout
		})

		result, err := e.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Wait delete timeout")
		assert.False(t, result.Requeue)
	})
}

// ---- handleClusterDeletion ----

func TestEnsureDeleteOrResetHandleClusterDeletionGaps(t *testing.T) {
	t.Run("cluster not deleting delete succeeds", func(t *testing.T) {
		cluster := &clusterv1.Cluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Status:     clusterv1.ClusterStatus{Phase: string(clusterv1.ClusterPhaseProvisioned)},
		}
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster, cluster)
		e.Ctx.Cluster = cluster

		err := e.handleClusterDeletion(context.Background(), e.Ctx.Client, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait for the deletion of cluster api obj")
	})

	t.Run("cluster not deleting delete fails", func(t *testing.T) {
		cluster := &clusterv1.Cluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Status:     clusterv1.ClusterStatus{Phase: string(clusterv1.ClusterPhaseProvisioned)},
		}
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster, cluster).Build()
		wrappedC := &deleteOrResetGapsDeleteErrClient{Client: fakeC}
		e := deleteOrResetGapsPhaseWithClient(t, wrappedC, bkeCluster)
		e.Ctx.Cluster = cluster

		err := e.handleClusterDeletion(context.Background(), e.Ctx.Client, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to delete cluster")
	})

	t.Run("cluster already deleting returns nil", func(t *testing.T) {
		cluster := &clusterv1.Cluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Status:     clusterv1.ClusterStatus{Phase: string(clusterv1.ClusterPhaseDeleting)},
		}
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)
		e.Ctx.Cluster = cluster

		err := e.handleClusterDeletion(context.Background(), e.Ctx.Client, e.Ctx.Log)
		require.NoError(t, err)
	})

	t.Run("cluster nil returns nil", func(t *testing.T) {
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)

		err := e.handleClusterDeletion(context.Background(), e.Ctx.Client, e.Ctx.Log)
		require.NoError(t, err)
	})
}

// ---- handleBKEMachineDeletion ----

func TestEnsureDeleteOrResetHandleBKEMachineDeletionGaps(t *testing.T) {
	t.Run("bootstrapped machine no owner removes finalizer", func(t *testing.T) {
		machine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "m1",
				Namespace:  "ns",
				Finalizers: []string{bkev1beta1.BKEMachineFinalizer},
			},
			Status: bkev1beta1.BKEMachineStatus{Bootstrapped: true},
		}
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster, machine)

		err := e.handleBKEMachineDeletion(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait for bkeMachine delete")
	})

	t.Run("list error returns error", func(t *testing.T) {
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
		wrappedC := &deleteOrResetGapsListErrClient{Client: fakeC, listErr: errors.New("list boom")}
		e := deleteOrResetGapsPhaseWithClient(t, wrappedC, bkeCluster)

		err := e.handleBKEMachineDeletion(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to list bkeMachine")
	})

	t.Run("list not found returns error", func(t *testing.T) {
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
		notFoundErr := apierrors.NewNotFound(schema.GroupResource{Group: "bke.bocloud.com", Resource: "bkemachines"}, "")
		wrappedC := &deleteOrResetGapsListErrClient{Client: fakeC, listErr: notFoundErr}
		e := deleteOrResetGapsPhaseWithClient(t, wrappedC, bkeCluster)

		err := e.handleBKEMachineDeletion(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to list bkeMachine")
	})

	t.Run("unbootstrapped machine no owner delete then remove finalizer fails", func(t *testing.T) {
		machine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "m1",
				Namespace:  "ns",
				Finalizers: []string{bkev1beta1.BKEMachineFinalizer},
			},
			Status: bkev1beta1.BKEMachineStatus{Bootstrapped: false},
		}
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster, machine)

		err := e.handleBKEMachineDeletion(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
	})
}

// ---- cleanupClusterResources ----

func TestEnsureDeleteOrResetCleanupClusterResourcesGaps(t *testing.T) {
	t.Run("forbidden fallback deletes individually", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "c1",
				Namespace:  "ns",
				Finalizers: []string{bkev1beta1.ClusterFinalizer},
			},
		}
		bkeNode := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "node1",
				Namespace: "ns",
				Labels:    map[string]string{clusterv1.ClusterNameLabel: "c1"},
			},
		}
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster, bkeNode).Build()
		wrappedC := &deleteOrResetGapsForbiddenOnBKENodeClient{Client: fakeC}
		e := deleteOrResetGapsPhaseWithClient(t, wrappedC, bkeCluster)

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })

		err := e.cleanupClusterResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.NoError(t, err)
		assert.Empty(t, e.Ctx.BKECluster.Finalizers)
	})

	t.Run("forbidden fallback list error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "c1",
				Namespace:  "ns",
				Finalizers: []string{bkev1beta1.ClusterFinalizer},
			},
		}
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
		// Wrap to make DeleteAllOf forbidden AND List fail for BKENodeList
		wrappedC := &deleteOrResetGapsForbiddenAndListErrClient{
			Client:         fakeC,
			bkeNodeListErr: errors.New("list boom"),
		}
		e := deleteOrResetGapsPhaseWithClient(t, wrappedC, bkeCluster)

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })

		err := e.cleanupClusterResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "list boom")
	})

	t.Run("SyncStatusUntilComplete error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "c1",
				Namespace:  "ns",
				Finalizers: []string{bkev1beta1.ClusterFinalizer},
			},
		}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
			return errors.New("sync status error")
		})

		err := e.cleanupClusterResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to update bkeCluster Status")
	})
}

// ---- cleanupClusterResources: BKENode 先置 Deleting 再删除（issue 312） ----

func TestEnsureDeleteOrResetCleanupClusterResourcesBKENodeState(t *testing.T) {
	newBKENode := func(name string, state confv1beta1.NodeState) *confv1beta1.BKENode {
		return &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: "ns",
				Labels:    map[string]string{clusterv1.ClusterNameLabel: "c1"},
			},
			Status: confv1beta1.BKENodeStatus{State: state},
		}
	}

	newDeletingCluster := func() *bkev1beta1.BKECluster {
		return &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "c1",
				Namespace:  "ns",
				Finalizers: []string{bkev1beta1.ClusterFinalizer},
			},
		}
	}

	t.Run("initializing node set to Deleting before deletecollection", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })

		bkeCluster := newDeletingCluster()
		bkeNode := newBKENode("node1", bkev1beta1.NodeInitializing)

		// 捕获 DeleteAllOf 执行时服务端对象的实际状态
		var stateAtDelete confv1beta1.NodeState
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).
			WithObjects(bkeCluster, bkeNode).
			WithStatusSubresource(&confv1beta1.BKENode{}).
			WithInterceptorFuncs(interceptor.Funcs{
				DeleteAllOf: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteAllOfOption) error {
					if _, ok := obj.(*confv1beta1.BKENode); ok {
						latest := &confv1beta1.BKENode{}
						if err := c.Get(ctx, client.ObjectKey{Namespace: "ns", Name: "node1"}, latest); err == nil {
							stateAtDelete = latest.Status.State
						}
					}
					return c.DeleteAllOf(ctx, obj, opts...)
				},
			}).Build()
		e := deleteOrResetGapsPhaseWithClient(t, fakeC, bkeCluster)

		require.NoError(t, e.cleanupClusterResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log))
		assert.Equal(t, confv1beta1.NodeDeleting, stateAtDelete)
		assert.Empty(t, e.Ctx.BKECluster.Finalizers)

		err := fakeC.Get(context.Background(), client.ObjectKey{Namespace: "ns", Name: "node1"}, &confv1beta1.BKENode{})
		assert.True(t, apierrors.IsNotFound(err))
	})

	t.Run("forbidden fallback deletes node in Deleting state", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })

		bkeCluster := newDeletingCluster()
		bkeNode := newBKENode("node1", bkev1beta1.NodeInitializing)

		// DeleteAllOf 模拟被拦截（RBAC/webhook），回退到逐个删除时校验状态
		var deletedStates []confv1beta1.NodeState
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).
			WithObjects(bkeCluster, bkeNode).
			WithStatusSubresource(&confv1beta1.BKENode{}).
			WithInterceptorFuncs(interceptor.Funcs{
				DeleteAllOf: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteAllOfOption) error {
					if _, ok := obj.(*confv1beta1.BKENode); ok {
						return apierrors.NewForbidden(schema.GroupResource{Group: "bke.bocloud.com", Resource: "bkenodes"}, "", errors.New("forbidden"))
					}
					return c.DeleteAllOf(ctx, obj, opts...)
				},
				Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
					if _, ok := obj.(*confv1beta1.BKENode); ok {
						latest := &confv1beta1.BKENode{}
						if err := c.Get(ctx, client.ObjectKey{Namespace: obj.GetNamespace(), Name: obj.GetName()}, latest); err == nil {
							deletedStates = append(deletedStates, latest.Status.State)
						}
					}
					return c.Delete(ctx, obj, opts...)
				},
			}).Build()
		e := deleteOrResetGapsPhaseWithClient(t, fakeC, bkeCluster)

		require.NoError(t, e.cleanupClusterResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log))
		assert.Equal(t, []confv1beta1.NodeState{confv1beta1.NodeDeleting}, deletedStates)
		assert.Empty(t, e.Ctx.BKECluster.Finalizers)

		err := fakeC.Get(context.Background(), client.ObjectKey{Namespace: "ns", Name: "node1"}, &confv1beta1.BKENode{})
		assert.True(t, apierrors.IsNotFound(err))
	})

	t.Run("already deleting node skipped patch and deleted", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })

		bkeCluster := newDeletingCluster()
		bkeNode := newBKENode("node1", bkev1beta1.NodeDeleting)

		patchCalls := 0
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).
			WithObjects(bkeCluster, bkeNode).
			WithStatusSubresource(&confv1beta1.BKENode{}).
			WithInterceptorFuncs(interceptor.Funcs{
				SubResourcePatch: func(ctx context.Context, c client.Client, subResourceName string, obj client.Object, patch client.Patch, opts ...client.SubResourcePatchOption) error {
					patchCalls++
					return c.SubResource(subResourceName).Patch(ctx, obj, patch, opts...)
				},
			}).Build()
		e := deleteOrResetGapsPhaseWithClient(t, fakeC, bkeCluster)

		require.NoError(t, e.cleanupClusterResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log))
		assert.Equal(t, 0, patchCalls)
		assert.Empty(t, e.Ctx.BKECluster.Finalizers)

		err := fakeC.Get(context.Background(), client.ObjectKey{Namespace: "ns", Name: "node1"}, &confv1beta1.BKENode{})
		assert.True(t, apierrors.IsNotFound(err))
	})

	t.Run("status patch failure does not block deletion", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })

		bkeCluster := newDeletingCluster()
		bkeNode := newBKENode("node1", bkev1beta1.NodeInitializing)

		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).
			WithObjects(bkeCluster, bkeNode).
			WithStatusSubresource(&confv1beta1.BKENode{}).
			WithInterceptorFuncs(interceptor.Funcs{
				SubResourcePatch: func(ctx context.Context, c client.Client, subResourceName string, obj client.Object, patch client.Patch, opts ...client.SubResourcePatchOption) error {
					return apierrors.NewForbidden(schema.GroupResource{Group: "bke.bocloud.com", Resource: "bkenodes"}, obj.GetName(), errors.New("patch forbidden"))
				},
			}).Build()
		e := deleteOrResetGapsPhaseWithClient(t, fakeC, bkeCluster)

		require.NoError(t, e.cleanupClusterResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log))
		assert.Empty(t, e.Ctx.BKECluster.Finalizers)

		err := fakeC.Get(context.Background(), client.ObjectKey{Namespace: "ns", Name: "node1"}, &confv1beta1.BKENode{})
		assert.True(t, apierrors.IsNotFound(err))
	})

	t.Run("node without cluster label left untouched", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })

		bkeCluster := newDeletingCluster()
		// 无 cluster label 的节点不属于该集群，不应被逐个清理逻辑处理
		orphanNode := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "orphan", Namespace: "ns"},
			Status:     confv1beta1.BKENodeStatus{State: bkev1beta1.NodeInitializing},
		}

		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).
			WithObjects(bkeCluster, orphanNode).
			WithStatusSubresource(&confv1beta1.BKENode{}).Build()
		e := deleteOrResetGapsPhaseWithClient(t, fakeC, bkeCluster)

		require.NoError(t, e.cleanupClusterResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log))
		assert.Empty(t, e.Ctx.BKECluster.Finalizers)

		// 带 label 过滤的 DeleteAllOf 清扫不会删除无 label 节点
		err := fakeC.Get(context.Background(), client.ObjectKey{Namespace: "ns", Name: "orphan"}, &confv1beta1.BKENode{})
		assert.NoError(t, err)
	})
}

// deleteOrResetGapsForbiddenAndListErrClient makes DeleteAllOf forbidden for BKENode
// and optionally returns an error on List for BKENodeList.
type deleteOrResetGapsForbiddenAndListErrClient struct {
	client.Client
	bkeNodeListErr error
}

func (f *deleteOrResetGapsForbiddenAndListErrClient) DeleteAllOf(ctx context.Context, obj client.Object, opts ...client.DeleteAllOfOption) error {
	if _, ok := obj.(*confv1beta1.BKENode); ok {
		return apierrors.NewForbidden(schema.GroupResource{Group: "bke.bocloud.com", Resource: "bkenodes"}, "", errors.New("forbidden"))
	}
	return f.Client.DeleteAllOf(ctx, obj, opts...)
}

func (f *deleteOrResetGapsForbiddenAndListErrClient) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	if _, ok := list.(*confv1beta1.BKENodeList); ok && f.bkeNodeListErr != nil {
		return f.bkeNodeListErr
	}
	return f.Client.List(ctx, list, opts...)
}

// ---- ensureClusterStatusDeleting ----

func TestEnsureDeleteOrResetEnsureClusterStatusDeletingGaps(t *testing.T) {
	t.Run("not deleting marks deleting success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })

		err := e.ensureClusterStatusDeleting(e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.NoError(t, err)
		assert.Equal(t, bkev1beta1.ClusterDeleting, e.Ctx.BKECluster.Status.ClusterStatus)
	})

	t.Run("not deleting SyncStatus error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
			return errors.New("sync error")
		})

		err := e.ensureClusterStatusDeleting(e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to update bkeCluster Status")
	})

	t.Run("already deleting returns nil", func(t *testing.T) {
		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Status:     confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterDeleting},
		}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)

		err := e.ensureClusterStatusDeleting(e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.NoError(t, err)
	})
}

func TestEnsureDeleteOrResetNeedExecuteGaps(t *testing.T) {
	t.Run("deletion timestamp set returns true", func(t *testing.T) {
		e := deleteOrResetGapsPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		now := metav1.Now()
		newCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns", DeletionTimestamp: &now},
		}
		assert.True(t, e.NeedExecute(nil, newCluster))
	})

	t.Run("spec reset returns true", func(t *testing.T) {
		e := deleteOrResetGapsPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		newCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec:       confv1beta1.BKEClusterSpec{Reset: true},
		}
		assert.True(t, e.NeedExecute(nil, newCluster))
	})

	t.Run("neither returns false", func(t *testing.T) {
		e := deleteOrResetGapsPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		newCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		}
		assert.False(t, e.NeedExecute(nil, newCluster))
	})
}

// ---- NewEnsureDeleteOrReset ----

func TestNewEnsureDeleteOrResetGaps(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	e := deleteOrResetGapsPhaseCov(t, bkeCluster)
	p := NewEnsureDeleteOrReset(e.Ctx)
	assert.NotNil(t, p)
}

// ---- ensureDeleteOrResetPostHook ----

func TestEnsureDeleteOrResetPostHookGaps(t *testing.T) {
	t.Run("nil err unregisters metrics", func(t *testing.T) {
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)
		err := ensureDeleteOrResetPostHook(e, nil)
		require.NoError(t, err)
	})

	t.Run("non nil err skips unregister", func(t *testing.T) {
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		e := deleteOrResetGapsPhaseCov(t, bkeCluster)
		err := ensureDeleteOrResetPostHook(e, errors.New("some error"))
		require.NoError(t, err)
	})
}

func TestEnsureDeleteOrResetHandleNamespaceDeletionGaps(t *testing.T) {
	t.Run("annotation false empty cluster list deletes namespace", func(t *testing.T) {
		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "c1",
				Namespace:   "ns",
				Annotations: map[string]string{annotation.DeleteIgnoreNamespaceAnnotationKey: "false"},
			},
		}
		// BKECluster not in fake client so List returns empty
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).Build()
		e := deleteOrResetGapsPhaseWithClient(t, fakeC, bkeCluster)

		err := e.handleNamespaceDeletion(context.Background(), e.Ctx.Client, bkeCluster, e.Ctx.Log)
		require.NoError(t, err)
	})

	t.Run("annotation false list error", func(t *testing.T) {
		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "c1",
				Namespace:   "ns",
				Annotations: map[string]string{annotation.DeleteIgnoreNamespaceAnnotationKey: "false"},
			},
		}
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
		wrappedC := &deleteOrResetGapsListErrClient{Client: fakeC, listErr: errors.New("list boom")}
		e := deleteOrResetGapsPhaseWithClient(t, wrappedC, bkeCluster)

		err := e.handleNamespaceDeletion(context.Background(), e.Ctx.Client, bkeCluster, e.Ctx.Log)
		require.NoError(t, err)
	})
}

// ---- deleteRelatedResources (list error branch) ----

func TestEnsureDeleteOrResetDeleteRelatedResourcesGaps(t *testing.T) {
	t.Run("list command error logs and returns nil", func(t *testing.T) {
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		scheme := deleteOrResetGapsScheme(t)
		fakeC := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
		wrappedC := &deleteOrResetGapsListErrClient{Client: fakeC, listErr: errors.New("list boom")}
		e := deleteOrResetGapsPhaseWithClient(t, wrappedC, bkeCluster)

		err := e.deleteRelatedResources(context.Background(), e.Ctx.Client, e.Ctx.BKECluster, e.Ctx.Log)
		require.NoError(t, err)
	})
}
