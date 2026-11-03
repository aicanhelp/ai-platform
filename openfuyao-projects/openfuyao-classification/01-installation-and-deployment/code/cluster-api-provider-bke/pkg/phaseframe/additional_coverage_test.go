/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 ******************************************************************/

package phaseframe

import (
	"context"
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	apiv1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func TestBasePhaseDefaultHooks(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	refreshedBKECluster := false
	refreshedCluster := false
	reported := false
	patches.ApplyMethodFunc(
		&PhaseContext{},
		"RefreshCtxBKECluster",
		func(_ ...context.Context) error {
			refreshedBKECluster = true
			return nil
		},
	)
	patches.ApplyMethodFunc(
		&PhaseContext{},
		"RefreshCtxCluster",
		func(_ ...context.Context) error {
			refreshedCluster = true
			return nil
		},
	)
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
		reported = true
		return nil
	})

	pc := NewReconcilePhaseCtx(context.Background()).
		SetBKECluster(&bkev1beta1.BKECluster{}).
		SetLogger(&bkev1beta1.BKELogger{})
	phase := NewBasePhase(pc, "EnsureDeleteOrReset")
	preHookCalled := false
	phase.RegisterPreHooks(func(p Phase) error {
		preHookCalled = true
		assert.Equal(t, confv1beta1.BKEClusterPhase("EnsureDeleteOrReset"), p.Name())
		return nil
	})

	require.NoError(t, phase.ExecutePreHook())
	assert.True(t, refreshedBKECluster)
	assert.True(t, refreshedCluster)
	assert.True(t, preHookCalled)
	assert.True(t, reported)
	assert.Equal(t, bkev1beta1.PhaseRunning, phase.GetStatus())

	postHookCalled := false
	phase.RegisterPostHooks(func(p Phase, err error) error {
		postHookCalled = true
		assert.NoError(t, err)
		return nil
	})
	require.NoError(t, phase.ExecutePostHook(nil))
	assert.True(t, postHookCalled)
	assert.Equal(t, bkev1beta1.PhaseSucceeded, phase.GetStatus())
}

func TestBasePhaseDefaultHookErrors(t *testing.T) {
	t.Run("pre hook refresh failed", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyMethodFunc(&PhaseContext{}, "RefreshCtxBKECluster", func(_ ...context.Context) error {
			return errors.New("refresh failed")
		})

		phase := NewBasePhase(NewReconcilePhaseCtx(context.Background()).SetBKECluster(&bkev1beta1.BKECluster{}), "EnsureDeleteOrReset")
		require.Error(t, phase.ExecutePreHook())
	})

	t.Run("pre custom hook failed", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyMethodFunc(&PhaseContext{}, "RefreshCtxBKECluster", func(_ ...context.Context) error { return nil })
		patches.ApplyMethodFunc(&PhaseContext{}, "RefreshCtxCluster", func(_ ...context.Context) error { return nil })

		phase := NewBasePhase(NewReconcilePhaseCtx(context.Background()).SetBKECluster(&bkev1beta1.BKECluster{}), "EnsureDeleteOrReset")
		phase.RegisterPreHooks(func(p Phase) error { return errors.New("pre failed") })
		require.Error(t, phase.ExecutePreHook())
	})

	t.Run("post custom hook failed", func(t *testing.T) {
		phase := NewBasePhase(NewReconcilePhaseCtx(context.Background()).SetBKECluster(&bkev1beta1.BKECluster{}), "EnsureDeleteOrReset")
		phase.RegisterPostHooks(func(p Phase, err error) error { return errors.New("post failed") })
		require.Error(t, phase.ExecutePostHook(nil))
		assert.Equal(t, bkev1beta1.PhaseSucceeded, phase.GetStatus())
	})
}

func TestBasePhaseDefaultPostHookStatuses(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
		return nil
	})

	pc := NewReconcilePhaseCtx(context.Background()).
		SetBKECluster(&bkev1beta1.BKECluster{}).
		SetLogger(&bkev1beta1.BKELogger{})

	skipped := NewBasePhase(pc, "EnsureDeleteOrReset")
	skipped.SetStatus(bkev1beta1.PhaseSkipped)
	require.NoError(t, skipped.ExecutePostHook(errors.New("skip reason")))
	assert.Equal(t, bkev1beta1.PhaseSkipped, skipped.GetStatus())

	failed := NewBasePhase(pc, "EnsureDeleteOrReset")
	require.NoError(t, failed.ExecutePostHook(errors.New("boom")))
	assert.Equal(t, bkev1beta1.PhaseFailed, failed.GetStatus())
}

func TestBuildAndSetVersionContextFromBundle(t *testing.T) {
	targetBundle := &releasemanifest.Bundle{
		Release: apiv1.ReleaseImage{
			Spec: apiv1.ReleaseImageSpec{
				Install: &apiv1.ReleaseImageInstallSpec{Components: []apiv1.ReleaseImageInstallComponent{
					{Name: upgrade.ComponentEtcd, Version: "v3.6.0"},
				}},
				Upgrade: &apiv1.ReleaseImageUpgradeSpec{Components: []apiv1.ReleaseImageUpgradeComponent{
					{Name: upgrade.ComponentKubernetesMaster, Version: "v1.30.0"},
				}},
			},
		},
	}
	currentBundle := &releasemanifest.Bundle{
		Release: apiv1.ReleaseImage{
			Spec: apiv1.ReleaseImageSpec{
				Install: &apiv1.ReleaseImageInstallSpec{Components: []apiv1.ReleaseImageInstallComponent{
					{Name: upgrade.ComponentEtcd, Version: "v3.5.0"},
				}},
			},
		},
	}
	pc := NewReconcilePhaseCtx(context.Background()).
		SetBKECluster(&bkev1beta1.BKECluster{})

	got := pc.BuildAndSetVersionContextFromBundle(targetBundle, currentBundle)
	require.Same(t, pc, got)
	require.NotNil(t, pc.VersionContext)
	assert.Equal(t, "v3.6.0", pc.VersionContext.GetTarget(upgrade.ComponentEtcd))
	assert.Equal(t, "v3.5.0", pc.VersionContext.GetCurrent(upgrade.ComponentEtcd))
	assert.True(t, pc.VersionContext.NeedsUpgrade(upgrade.ComponentEtcd))
	assert.Equal(t, "v1.30.0", pc.VersionContext.GetTarget(upgrade.ComponentKubernetesMaster))
}

func TestSyncUpgradeTargetsToClusterSpec(t *testing.T) {
	t.Run("nil inputs are no-op", func(t *testing.T) {
		var pc *PhaseContext
		require.NoError(t, pc.SyncUpgradeTargetsToClusterSpec())
		require.NoError(t, (&PhaseContext{}).SyncUpgradeTargetsToClusterSpec())
	})

	t.Run("applies targets without client", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Spec: confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{}},
		}
		vc := upgrade.NewVersionContext()
		vc.SetTarget(upgrade.ComponentEtcd, "v3.6.0")
		vc.SetTarget(upgrade.ComponentContainerd, "v1.7.0")
		vc.SetTarget(upgrade.ComponentKubernetesWorker, "v1.30.0")
		pc := NewReconcilePhaseCtx(context.Background()).SetBKECluster(cluster).SetVersionContext(vc)

		require.NoError(t, pc.SyncUpgradeTargetsToClusterSpec())
		assert.Equal(t, "v3.6.0", cluster.Spec.ClusterConfig.Cluster.EtcdVersion)
		assert.Equal(t, "v1.7.0", cluster.Spec.ClusterConfig.Cluster.ContainerdVersion)
		assert.Equal(t, "v1.30.0", cluster.Spec.ClusterConfig.Cluster.KubernetesVersion)
	})

	t.Run("uses sync when spec already has targets and client exists", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		synced := false
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, bc *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
			synced = true
			for _, patch := range patchs {
				patch(bc)
			}
			return nil
		})

		cluster := &bkev1beta1.BKECluster{
			Spec: confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{EtcdVersion: "old"},
			}},
		}
		vc := upgrade.NewVersionContext()
		vc.SetTarget(upgrade.ComponentEtcd, "v3.6.0")
		pc := NewReconcilePhaseCtx(context.Background()).SetClient(&fakeClient{}).SetBKECluster(cluster).SetVersionContext(vc)

		require.NoError(t, pc.SyncUpgradeTargetsToClusterSpec())
		assert.True(t, synced)
		assert.Equal(t, "v3.6.0", cluster.Spec.ClusterConfig.Cluster.EtcdVersion)
	})
}

func TestWatchBKEClusterStatusEarlyReturns(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyMethodFunc(&bkev1beta1.BKELogger{}, "Error", func(reason, msg string, args ...interface{}) {})
	patches.ApplyMethodFunc(&bkev1beta1.BKELogger{}, "Warn", func(reason, msg string, args ...interface{}) {})

	pc := NewReconcilePhaseCtx(context.Background()).SetLogger(&bkev1beta1.BKELogger{})
	pc.WatchBKEClusterStatus()

	patches.ApplyMethodFunc(&PhaseContext{}, "GetNewestBKECluster", func(_ ...context.Context) (*bkev1beta1.BKECluster, error) {
		return nil, errors.New("get failed")
	})
	pc.SetBKECluster(&bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"}})
	pc.WatchBKEClusterStatus()
}
