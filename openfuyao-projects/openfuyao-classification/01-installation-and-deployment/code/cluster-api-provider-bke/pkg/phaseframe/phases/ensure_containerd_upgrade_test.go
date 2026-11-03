package phases

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureContainerdUpgradeConstants(t *testing.T) {
	assert.Equal(t, "EnsureContainerdUpgrade", string(EnsureContainerdUpgradeName))
}

func TestNewEnsureContainerdUpgrade(t *testing.T) {
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

	phase := NewEnsureContainerdUpgrade(ctx)
	assert.NotNil(t, phase)

	e, ok := phase.(*EnsureContainerdUpgrade)
	assert.True(t, ok)
	assert.Equal(t, EnsureContainerdUpgradeName, e.PhaseName)
}

func TestEnsureContainerdUpgrade_IsContainerdNeedUpgrade_NodeFetcherError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "1.7.0"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "1.6.0"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyMethod(ctx, "NodeFetcher", func(_ *phaseframe.PhaseContext) interface{} {
		return nil
	})

	result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureContainerdUpgrade_Execute_Success(t *testing.T) {
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
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "rolloutContainerd", func(_ *EnsureContainerdUpgrade) (ctrl.Result, error) {
		return ctrl.Result{}, nil
	})

	result, err := e.Execute()
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureContainerdUpgrade_NeedExecute_DefaultFalse(t *testing.T) {
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
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx, PhaseName: EnsureContainerdUpgradeName}}

	patches.ApplyMethod(&e.BasePhase, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return false
	})

	result := e.NeedExecute(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureContainerdUpgrade_IsContainerdNeedUpgrade_EmptyStatus(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "1.6.0"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{ContainerdVersion: ""},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureContainerdUpgrade_IsContainerdNeedUpgrade_InvalidOldVersion(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "1.6.0"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "invalid"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureContainerdUpgrade_IsContainerdNeedUpgrade_InvalidNewVersion(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "invalid"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "1.5.0"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureContainerdUpgrade_IsContainerdNeedUpgrade_SameVersion(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "1.6.0"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "1.6.0"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureContainerdUpgrade_IsContainerdNeedUpgrade_Downgrade(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "1.5.0"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "1.6.0"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureContainerdUpgrade_NeedExecute_ContainerdNotNeedUpgrade(t *testing.T) {
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
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx, PhaseName: EnsureContainerdUpgradeName}}

	patches.ApplyMethod(&e.BasePhase, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return true
	})
	patches.ApplyPrivateMethod(e, "isContainerdNeedUpgrade", func(_ *EnsureContainerdUpgrade, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return false
	})

	result := e.NeedExecute(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureContainerdUpgrade_NeedExecute_BothTrue(t *testing.T) {
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
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx, PhaseName: EnsureContainerdUpgradeName}}

	patches.ApplyMethod(&e.BasePhase, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return true
	})
	patches.ApplyPrivateMethod(e, "isContainerdNeedUpgrade", func(_ *EnsureContainerdUpgrade, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return true
	})

	result := e.NeedExecute(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.True(t, result)
}

func TestEnsureContainerdUpgrade_Execute_RolloutError(t *testing.T) {
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
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "rolloutContainerd", func(_ *EnsureContainerdUpgrade) (ctrl.Result, error) {
		return ctrl.Result{}, assert.AnError
	})

	result, err := e.Execute()
	assert.Error(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureContainerdUpgrade_RolloutContainerd_ResetError(t *testing.T) {
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
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "resetContainerd", func(_ *EnsureContainerdUpgrade) error {
		return assert.AnError
	})

	result, err := e.rolloutContainerd()
	assert.Error(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureContainerdUpgrade_RolloutContainerd_RedeployError(t *testing.T) {
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
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "resetContainerd", func(_ *EnsureContainerdUpgrade) error {
		return nil
	})
	patches.ApplyPrivateMethod(e, "redeployContainerd", func(_ *EnsureContainerdUpgrade) error {
		return assert.AnError
	})

	result, err := e.rolloutContainerd()
	assert.Error(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureContainerdUpgrade_RolloutContainerd_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "1.6.0"},
			},
		},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "resetContainerd", func(_ *EnsureContainerdUpgrade) error {
		return nil
	})
	patches.ApplyPrivateMethod(e, "redeployContainerd", func(_ *EnsureContainerdUpgrade) error {
		return nil
	})

	result, err := e.rolloutContainerd()
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
	assert.Equal(t, "1.6.0", bkeCluster.Status.ContainerdVersion)
}

func TestEnsureContainerdUpgrade_RolloutContainerd_UsesVersionContextTarget(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "1.6.0"},
			},
		},
	}
	ctx := &phaseframe.PhaseContext{
		Context:        context.Background(),
		BKECluster:     bkeCluster,
		VersionContext: upgrade.NewVersionContext(),
		Log:            bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	ctx.VersionContext.SetTarget(upgrade.ComponentContainerd, "1.7.0")

	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "resetContainerd", func(_ *EnsureContainerdUpgrade) error {
		assert.Equal(t, "1.7.0", bkeCluster.Spec.ClusterConfig.Cluster.ContainerdVersion)
		return nil
	})
	patches.ApplyPrivateMethod(e, "redeployContainerd", func(_ *EnsureContainerdUpgrade) error {
		assert.Equal(t, "1.7.0", bkeCluster.Spec.ClusterConfig.Cluster.ContainerdVersion)
		return nil
	})

	result, err := e.rolloutContainerd()
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
	assert.Equal(t, "1.7.0", bkeCluster.Status.ContainerdVersion)
	assert.Equal(t, "1.7.0", bkeCluster.Spec.ClusterConfig.Cluster.ContainerdVersion)
}

func TestEnsureContainerdUpgrade_IsContainerdNeedUpgrade_UpgradeNeeded(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "1.7.0"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "1.6.0"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "isContainerdNeedUpgrade", func(_ *EnsureContainerdUpgrade, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return true
	})

	result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.True(t, result)
}

func TestEnsureContainerdUpgrade_GetCommand_Success(t *testing.T) {
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
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "getCommand", func(_ *EnsureContainerdUpgrade) interface{} {
		return nil
	})

	result := e.getCommand()
	assert.Nil(t, result)
}

func TestEnsureContainerdUpgrade_ResetContainerd_GetCommandFail(t *testing.T) {
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
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "getCommand", func(_ *EnsureContainerdUpgrade) interface{} {
		return nil
	})

	err := e.resetContainerd()
	assert.Error(t, err)
}

func TestEnsureContainerdUpgrade_RedeployContainerd_GetCommandFail(t *testing.T) {
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
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "getCommand", func(_ *EnsureContainerdUpgrade) interface{} {
		return nil
	})

	err := e.redeployContainerd()
	assert.Error(t, err)
}

func TestEnsureContainerdUpgrade_IsContainerdNeedUpgrade_AllNodesFailed(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "1.7.0"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "1.6.0"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureContainerdUpgrade_IsContainerdNeedUpgrade_AllNodesSkipped(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "1.7.0"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "1.6.0"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.False(t, result)
}

func TestEnsureContainerdUpgrade_NeedExecute_SetStatusWaiting(t *testing.T) {
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
	e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx, PhaseName: EnsureContainerdUpgradeName}}

	patches.ApplyMethod(&e.BasePhase, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return true
	})
	patches.ApplyPrivateMethod(e, "isContainerdNeedUpgrade", func(_ *EnsureContainerdUpgrade, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return true
	})

	result := e.NeedExecute(&bkev1beta1.BKECluster{}, bkeCluster)
	assert.True(t, result)
}

func newContainerdUpgradePhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureContainerdUpgrade {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	c := ctrlfake.NewClientBuilder().WithScheme(scheme).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

func containerdCluster(ver string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{Cluster: confv1beta1.Cluster{ContainerdVersion: ver}}},
	}
}

// ---- Version ----

func TestEnsureContainerdUpgradeVersionCov(t *testing.T) {
	t.Run("returns containerd version", func(t *testing.T) {
		cluster := containerdCluster("1.7.0")
		cluster.Status.ContainerdVersion = "1.6.0"
		e := newContainerdUpgradePhaseCov(t, cluster)
		assert.Equal(t, "1.6.0", e.Version())
	})
	t.Run("nil ctx", func(t *testing.T) {
		assert.Equal(t, "", (&EnsureContainerdUpgrade{}).Version())
	})
	t.Run("nil bkecluster", func(t *testing.T) {
		e := &EnsureContainerdUpgrade{BasePhase: phaseframe.BasePhase{Ctx: &phaseframe.PhaseContext{}}}
		assert.Equal(t, "", e.Version())
	})
}

// ---- resetContainerd (real body via getCommand seam + ENV methods) ----

func TestEnsureContainerdUpgradeResetContainerd(t *testing.T) {
	t.Run("get command nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newContainerdUpgradePhaseCov(t, containerdCluster("1.7.0"))
		patches.ApplyPrivateMethod(e, "getCommand", func(_ *EnsureContainerdUpgrade) *command.ENV { return nil })
		err := e.resetContainerd()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "new containerd command fail")
	})

	t.Run("reset success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newContainerdUpgradePhaseCov(t, containerdCluster("1.7.0"))
		patches.ApplyPrivateMethod(e, "getCommand", func(_ *EnsureContainerdUpgrade) *command.ENV { return &command.ENV{} })
		patches.ApplyMethod(&command.ENV{}, "NewConatinerdReset", func(_ *command.ENV) error { return nil })
		patches.ApplyMethod(&command.ENV{}, "Wait", func(_ *command.ENV) (error, []string, []string) {
			return nil, []string{"node1"}, nil
		})
		require.NoError(t, e.resetContainerd())
	})

	t.Run("new containerd reset error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newContainerdUpgradePhaseCov(t, containerdCluster("1.7.0"))
		patches.ApplyPrivateMethod(e, "getCommand", func(_ *EnsureContainerdUpgrade) *command.ENV { return &command.ENV{} })
		patches.ApplyMethod(&command.ENV{}, "NewConatinerdReset", func(_ *command.ENV) error { return assertErr("reset create failed") })
		err := e.resetContainerd()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "reset create failed")
	})

	t.Run("wait returns failed nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newContainerdUpgradePhaseCov(t, containerdCluster("1.7.0"))
		patches.ApplyPrivateMethod(e, "getCommand", func(_ *EnsureContainerdUpgrade) *command.ENV { return &command.ENV{} })
		patches.ApplyMethod(&command.ENV{}, "NewConatinerdReset", func(_ *command.ENV) error { return nil })
		patches.ApplyMethod(&command.ENV{}, "Wait", func(_ *command.ENV) (error, []string, []string) {
			return assertErr("wait failed"), []string{"node1"}, []string{"node2"}
		})
		err := e.resetContainerd()
		require.Error(t, err)
	})
}

// ---- redeployContainerd (real body) ----

func TestEnsureContainerdUpgradeRedeployContainerd(t *testing.T) {
	t.Run("get command nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newContainerdUpgradePhaseCov(t, containerdCluster("1.7.0"))
		patches.ApplyPrivateMethod(e, "getCommand", func(_ *EnsureContainerdUpgrade) *command.ENV { return nil })
		err := e.redeployContainerd()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "new containerd command fail")
	})

	t.Run("redeploy success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newContainerdUpgradePhaseCov(t, containerdCluster("1.7.0"))
		patches.ApplyPrivateMethod(e, "getCommand", func(_ *EnsureContainerdUpgrade) *command.ENV { return &command.ENV{} })
		patches.ApplyMethod(&command.ENV{}, "NewConatinerdRedeploy", func(_ *command.ENV) error { return nil })
		patches.ApplyMethod(&command.ENV{}, "Wait", func(_ *command.ENV) (error, []string, []string) {
			return nil, []string{"node1"}, nil
		})
		require.NoError(t, e.redeployContainerd())
	})

	t.Run("new containerd redeploy error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newContainerdUpgradePhaseCov(t, containerdCluster("1.7.0"))
		patches.ApplyPrivateMethod(e, "getCommand", func(_ *EnsureContainerdUpgrade) *command.ENV { return &command.ENV{} })
		patches.ApplyMethod(&command.ENV{}, "NewConatinerdRedeploy", func(_ *command.ENV) error { return assertErr("redeploy create failed") })
		err := e.redeployContainerd()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "redeploy create failed")
	})

	t.Run("wait returns failed nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newContainerdUpgradePhaseCov(t, containerdCluster("1.7.0"))
		patches.ApplyPrivateMethod(e, "getCommand", func(_ *EnsureContainerdUpgrade) *command.ENV { return &command.ENV{} })
		patches.ApplyMethod(&command.ENV{}, "NewConatinerdRedeploy", func(_ *command.ENV) error { return nil })
		patches.ApplyMethod(&command.ENV{}, "Wait", func(_ *command.ENV) (error, []string, []string) {
			return assertErr("wait failed"), []string{}, []string{"node2"}
		})
		err := e.redeployContainerd()
		require.Error(t, err)
	})
}

// containerdGapsCluster builds a BKECluster with a ClusterConfig (required so that
// getCommand's access to bkeCluster.Spec.ClusterConfig.Addons does not panic) and
// optional control-plane endpoint / addons configuration.
func containerdGapsCluster() *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{ContainerdVersion: "1.7.0"},
			},
		},
	}
}

// withLoadBalancerEndpoint sets a valid control-plane endpoint on the cluster.
func withLoadBalancerEndpoint(c *bkev1beta1.BKECluster, host string, port int32) *bkev1beta1.BKECluster {
	c.Spec.ControlPlaneEndpoint = confv1beta1.APIEndpoint{Host: host, Port: port}
	return c
}

// withIngressAddon attaches a beyondELB addon with the given lbVIP.
func withIngressAddon(c *bkev1beta1.BKECluster, vip string) *bkev1beta1.BKECluster {
	c.Spec.ClusterConfig.Addons = append(c.Spec.ClusterConfig.Addons, confv1beta1.Product{
		Name:  "beyondELB",
		Param: map[string]string{"lbVIP": vip},
	})
	return c
}

// ---- getCommand (real body, was 0%) ----

func TestEnsureContainerdUpgradeGetCommand_Gaps(t *testing.T) {
	t.Run("success_with_loadbalancer_and_ingress_extras", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		cluster := withIngressAddon(withLoadBalancerEndpoint(containerdGapsCluster(), "10.0.0.1", 6443), "10.0.0.2")
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return bkev1beta1.BKENodes{}, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				// node IP does NOT match LB host -> AvailableLoadBalancerEndPoint returns true
				return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "192.168.1.1"}}}, nil
			})

		cmd := e.getCommand()
		require.NotNil(t, cmd)
		assert.IsType(t, &command.ENV{}, cmd)
		// LB host and ingress vip are both appended to Extra
		assert.Contains(t, cmd.Extra, "10.0.0.1")
		assert.Contains(t, cmd.Extra, "10.0.0.2")
		// ExtraHosts holds the master HA domain mapping
		assert.NotEmpty(t, cmd.ExtraHosts)
	})

	t.Run("success_lb_host_matches_node_no_extra", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		cluster := withLoadBalancerEndpoint(containerdGapsCluster(), "192.168.1.1", 6443)
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return bkev1beta1.BKENodes{}, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				// node IP matches LB host -> AvailableLoadBalancerEndPoint returns false
				return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "192.168.1.1"}}}, nil
			})

		cmd := e.getCommand()
		require.NotNil(t, cmd)
		assert.Empty(t, cmd.Extra)
		assert.Empty(t, cmd.ExtraHosts)
	})

	t.Run("success_invalid_endpoint_no_extras", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		// No control-plane endpoint set -> IsValid false -> AvailableLoadBalancerEndPoint false
		cluster := containerdGapsCluster()
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return bkev1beta1.BKENodes{}, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "192.168.1.1"}}}, nil
			})

		cmd := e.getCommand()
		require.NotNil(t, cmd)
		assert.Empty(t, cmd.Extra)
		assert.Empty(t, cmd.ExtraHosts)
	})

	t.Run("success_ingress_vip_equals_host", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		// LB host and ingress vip are the same -> second extra append is skipped
		cluster := withIngressAddon(withLoadBalancerEndpoint(containerdGapsCluster(), "10.0.0.1", 6443), "10.0.0.1")
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return bkev1beta1.BKENodes{}, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "192.168.1.1"}}}, nil
			})

		cmd := e.getCommand()
		require.NotNil(t, cmd)
		// Only the LB host is appended; ingress vip equals host so it is not added again
		assert.Equal(t, []string{"10.0.0.1"}, cmd.Extra)
	})

	t.Run("get_bke_nodes_wrapper_error_returns_nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		cluster := containerdGapsCluster()
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return nil, assertErr("fetch bke nodes failed")
			})

		cmd := e.getCommand()
		assert.Nil(t, cmd)
	})

	t.Run("get_nodes_error_returns_nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		cluster := containerdGapsCluster()
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return bkev1beta1.BKENodes{}, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return nil, assertErr("fetch nodes failed")
			})

		cmd := e.getCommand()
		assert.Nil(t, cmd)
	})

	t.Run("build_common_env_command_error_returns_nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		cluster := containerdGapsCluster()
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return bkev1beta1.BKENodes{}, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "192.168.1.1"}}}, nil
			})
		patches.ApplyFunc(BuildCommonEnvCommand,
			func(_ BuildCommonEnvCommandParams) (*command.ENV, error) {
				return nil, assertErr("build env command failed")
			})

		cmd := e.getCommand()
		assert.Nil(t, cmd)
	})

	t.Run("containerd_version_diff_selects_nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		cluster := containerdGapsCluster()
		// Keep openFuyaoVersion unchanged so the assertion proves getCommand now
		// relies on containerdVersion difference for node selection.
		cluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion = "vfit-cur"
		cluster.Spec.ClusterConfig.Cluster.ContainerdVersion = "v2.1.2"
		cluster.Status.OpenFuyaoVersion = "vfit-cur"
		cluster.Status.ContainerdVersion = "v2.1.1"
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return bkev1beta1.BKENodes{{
					Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1"},
					Status: confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentReadyFlag},
				}}, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "192.168.1.1"}}}, nil
			})

		cmd := e.getCommand()
		require.NotNil(t, cmd)
		require.Len(t, cmd.Nodes, 1)
		assert.Equal(t, "192.168.1.1", cmd.Nodes[0].IP)
		assert.Equal(t, "v2.1.2", cmd.ContainerdVersion)
	})
}

// ---- isContainerdNeedUpgrade (upgrade / case 1 branch, was 27.3%) ----

func TestEnsureContainerdUpgradeIsContainerdNeedUpgrade_Gaps(t *testing.T) {
	t.Run("upgrade_with_healthy_node_returns_true", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{ContainerdVersion: "v1.7.0"},
				},
			},
			Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "v1.6.0"},
		}
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				// single node, not skipped, no failed flag -> should return true
				return &nodeutil.FetchResult{BKENodes: []confv1beta1.BKENode{{
					Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1"},
					Status: confv1beta1.BKENodeStatus{NeedSkip: false, StateCode: 0},
				}}}, nil
			})

		result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, cluster)
		assert.True(t, result)
	})

	t.Run("upgrade_all_nodes_skipped_returns_false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{ContainerdVersion: "v1.7.0"},
				},
			},
			Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "v1.6.0"},
		}
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return &nodeutil.FetchResult{BKENodes: []confv1beta1.BKENode{{
					Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1"},
					Status: confv1beta1.BKENodeStatus{NeedSkip: true, StateCode: 0},
				}}}, nil
			})

		result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, cluster)
		assert.False(t, result)
	})

	t.Run("upgrade_all_nodes_failed_returns_false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{ContainerdVersion: "v1.7.0"},
				},
			},
			Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "v1.6.0"},
		}
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return &nodeutil.FetchResult{BKENodes: []confv1beta1.BKENode{{
					Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"},
					// NodeFailedFlag set -> GetNodeStateFlag returns true -> skip
					Status: confv1beta1.BKENodeStatus{NeedSkip: false, StateCode: bkev1beta1.NodeFailedFlag},
				}}}, nil
			})

		result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, cluster)
		assert.False(t, result)
	})

	t.Run("upgrade_get_bke_nodes_error_returns_false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{ContainerdVersion: "v1.7.0"},
				},
			},
			Status: confv1beta1.BKEClusterStatus{ContainerdVersion: "v1.6.0"},
		}
		e := newContainerdUpgradePhaseCov(t, cluster)

		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return nil, assertErr("fetch bke nodes failed")
			})

		result := e.isContainerdNeedUpgrade(&bkev1beta1.BKECluster{}, cluster)
		assert.False(t, result)
	})
}
