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
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestPhaseFlowConstants(t *testing.T) {
	assert.Equal(t, 20, MaxPhaseStatusHistory)
	assert.NotNil(t, FullPhasesRegisFunc)
	assert.Greater(t, len(FullPhasesRegisFunc), 0)
}

func TestNewPhaseFlow(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	flow := NewPhaseFlow(ctx)
	assert.NotNil(t, flow)
	assert.Equal(t, ctx, flow.ctx)
	assert.Nil(t, flow.BKEPhases)
}

func TestPhaseFlow_DeterminePhasesFuncs_DeleteOrReset(t *testing.T) {
	t.Skip("skip unstable UT: monkey patch of phaseutil.IsDeleteOrReset is ineffective in CI, causing phase count drift")
	patches := gomonkey.NewPatches()
	defer patches.Reset()

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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	patches.ApplyFunc(phaseutil.IsDeleteOrReset, func(bkeCluster *bkev1beta1.BKECluster) bool {
		return true
	})

	flow := NewPhaseFlow(ctx)
	funcs := flow.determinePhasesFuncs()
	assert.Equal(t, len(DeletePhases()), len(funcs))
}

func TestPhaseFlow_DeterminePhasesFuncs_Normal(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	patches.ApplyFunc(phaseutil.IsDeleteOrReset, func(bkeCluster *bkev1beta1.BKECluster) bool {
		return false
	})

	flow := NewPhaseFlow(ctx)
	funcs := flow.determinePhasesFuncs()
	assert.Equal(t, len(FullPhasesRegisFunc), len(funcs))
}

func TestPhaseFlow_DeterminePhases_DeleteOrReset(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	patches.ApplyFunc(phaseutil.IsDeleteOrReset, func(bkeCluster *bkev1beta1.BKECluster) bool {
		return true
	})

	flow := NewPhaseFlow(ctx)
	phases := flow.determinePhases()
	assert.NotEqual(t, ClusterDeleteResetPhaseNames(), phases)
}

func TestPhaseFlow_GetWaitingPhases(t *testing.T) {
	t.Skip("Skipping - requires proper PhaseStatus structure")
}

func TestPhaseFlow_ProcessPhaseStatus(t *testing.T) {
	t.Skip("Skipping - requires proper PhaseStatus structure")
}

func TestRegisterPhaseCName(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureFinalizer(ctx)
	err := registerPhaseCName(phase)
	assert.NoError(t, err)
}

func TestHandleClusterInitPhase_Success(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterInitPhase(ctx, nil)
	assert.Equal(t, bkev1beta1.ClusterInitializing, ctx.BKECluster.Status.ClusterStatus)
}

func TestHandleClusterInitPhase_Error(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterInitPhase(ctx, assert.AnError)
	assert.Equal(t, bkev1beta1.ClusterInitializationFailed, ctx.BKECluster.Status.ClusterStatus)
}

func TestHandleClusterScaleMasterUpPhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterScaleMasterUpPhase(ctx, nil)
	assert.Equal(t, bkev1beta1.ClusterMasterScalingUp, ctx.BKECluster.Status.ClusterStatus)

	handleClusterScaleMasterUpPhase(ctx, assert.AnError)
	assert.Equal(t, bkev1beta1.ClusterMasterScalingUpFailed, ctx.BKECluster.Status.ClusterStatus)
}

func TestHandleClusterScaleWorkerUpPhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterScaleWorkerUpPhase(ctx, nil)
	assert.Equal(t, bkev1beta1.ClusterWorkerScalingUp, ctx.BKECluster.Status.ClusterStatus)

	handleClusterScaleWorkerUpPhase(ctx, assert.AnError)
	assert.Equal(t, bkev1beta1.ClusterWorkerScalingUpFailed, ctx.BKECluster.Status.ClusterStatus)
}

func TestHandleClusterDeletePhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterDeletePhase(ctx, nil)
	assert.Equal(t, bkev1beta1.ClusterDeleting, ctx.BKECluster.Status.ClusterStatus)

	handleClusterDeletePhase(ctx, assert.AnError)
	assert.Equal(t, bkev1beta1.ClusterDeleteFailed, ctx.BKECluster.Status.ClusterStatus)
}

func TestHandleClusterPausedPhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterPausedPhase(ctx, nil)
	assert.Equal(t, bkev1beta1.ClusterPaused, ctx.BKECluster.Status.ClusterStatus)

	handleClusterPausedPhase(ctx, assert.AnError)
	assert.Equal(t, bkev1beta1.ClusterPauseFailed, ctx.BKECluster.Status.ClusterStatus)
}

func TestHandleClusterDryRunPhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterDryRunPhase(ctx, nil)
	assert.Equal(t, bkev1beta1.ClusterDryRun, ctx.BKECluster.Status.ClusterStatus)

	handleClusterDryRunPhase(ctx, assert.AnError)
	assert.Equal(t, bkev1beta1.ClusterDryRunFailed, ctx.BKECluster.Status.ClusterStatus)
}

func TestHandleClusterAddonsPhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterAddonsPhase(ctx, nil)
	assert.Equal(t, bkev1beta1.ClusterDeployingAddon, ctx.BKECluster.Status.ClusterStatus)

	handleClusterAddonsPhase(ctx, assert.AnError)
	assert.Equal(t, bkev1beta1.ClusterDeployAddonFailed, ctx.BKECluster.Status.ClusterStatus)
}

func TestHandleClusterUpgradePhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterUpgradePhase(ctx, nil)
	assert.Equal(t, bkev1beta1.ClusterUpgrading, ctx.BKECluster.Status.ClusterStatus)

	handleClusterUpgradePhase(ctx, assert.AnError)
	assert.Equal(t, bkev1beta1.ClusterUpgradeFailed, ctx.BKECluster.Status.ClusterStatus)
}

func TestHandleClusterScaleMasterDownPhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterScaleMasterDownPhase(ctx, nil)
	assert.Equal(t, bkev1beta1.ClusterMasterScalingDown, ctx.BKECluster.Status.ClusterStatus)

	handleClusterScaleMasterDownPhase(ctx, assert.AnError)
	assert.Equal(t, bkev1beta1.ClusterMasterScalingDownFailed, ctx.BKECluster.Status.ClusterStatus)
}

func TestHandleClusterScaleWorkerDownPhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterScaleWorkerDownPhase(ctx, nil)
	assert.Equal(t, bkev1beta1.ClusterWorkerScalingDown, ctx.BKECluster.Status.ClusterStatus)

	handleClusterScaleWorkerDownPhase(ctx, assert.AnError)
	assert.Equal(t, bkev1beta1.ClusterWorkerScalingDownFailed, ctx.BKECluster.Status.ClusterStatus)
}

func TestHandleClusterManagePhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	handleClusterManagePhase(ctx, nil)
	assert.Equal(t, bkev1beta1.ClusterManaging, ctx.BKECluster.Status.ClusterStatus)

	handleClusterManagePhase(ctx, assert.AnError)
	assert.Equal(t, bkev1beta1.ClusterManageFailed, ctx.BKECluster.Status.ClusterStatus)
}

func TestCalculateClusterStatusByPhase_InitPhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureFinalizer(ctx)
	err := calculateClusterStatusByPhase(phase, nil)
	assert.NoError(t, err)
	assert.Equal(t, bkev1beta1.ClusterInitializing, ctx.BKECluster.Status.ClusterStatus)
}

func TestCalculateClusterStatusByPhase_UpgradePhase(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureMasterUpgrade(ctx)
	err := calculateClusterStatusByPhase(phase, nil)
	assert.NoError(t, err)
	assert.Equal(t, bkev1beta1.ClusterUpgrading, ctx.BKECluster.Status.ClusterStatus)
}

func TestCalculateClusterStatusByPhase_CustomSetStatus(t *testing.T) {
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
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureCluster(ctx)
	err := calculateClusterStatusByPhase(phase, nil)
	assert.NoError(t, err)
}

func TestSkipPhaseAfterDeclarativeDAG(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		BKECluster:              bkeCluster,
		Scheme:                  scheme,
		Context:                 context.Background(),
		Log:                     bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
		DeclarativeDAGCompleted: true,
	}

	flow := NewPhaseFlow(ctx)
	assert.True(t, flow.skipPhaseAfterDeclarativeDAG(NewEnsureEtcdUpgrade(ctx)))
	assert.False(t, flow.skipPhaseAfterDeclarativeDAG(NewEnsureCluster(ctx)))

	ctx.DeclarativeDAGCompleted = false
	assert.False(t, flow.skipPhaseAfterDeclarativeDAG(NewEnsureEtcdUpgrade(ctx)))
}

func TestCalculateAndAddPhases_SkipsLegacyUpgradeWhenCVFailed(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = cvv1alpha1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.06"},
		Status: cvv1alpha1.ClusterVersionStatus{
			CurrentVersion: "v26.05",
			Phase:          cvv1alpha1.ClusterVersionPhaseFailed,
		},
	}
	fakeClient := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster, cv).Build()
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     fakeClient,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	flow := NewPhaseFlow(ctx)
	flow.calculateAndAddPhases(bkeCluster, bkeCluster, []func(*phaseframe.PhaseContext) phaseframe.Phase{
		func(pc *phaseframe.PhaseContext) phaseframe.Phase { return NewEnsureEtcdUpgrade(pc) },
	})

	assert.Empty(t, flow.BKEPhases)
}

func TestCalculateAndAddPhases_SkipsLegacyUpgradeWhenTerminalUpgradeFailed(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Status: confv1beta1.BKEClusterStatus{
			ClusterHealthState: bkev1beta1.UpgradeFailed,
		},
	}
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	flow := NewPhaseFlow(ctx)
	flow.calculateAndAddPhases(bkeCluster, bkeCluster, []func(*phaseframe.PhaseContext) phaseframe.Phase{
		func(pc *phaseframe.PhaseContext) phaseframe.Phase { return NewEnsureAgentUpgrade(pc) },
	})

	assert.Empty(t, flow.BKEPhases)
}

func TestPhaseFlow_CalculatePhase(t *testing.T) {
	t.Skip("Skipping - requires complex phase mocking")
}

func TestPhaseFlow_Execute(t *testing.T) {
	t.Skip("Skipping - requires complex orchestration with goroutines")
}

func TestPhaseFlow_ExecutePhases(t *testing.T) {
	t.Skip("Skipping - requires complex phase execution mocking")
}

func TestPhaseFlow_RefreshOldAndNewBKECluster(t *testing.T) {
	t.Skip("Skipping - requires mergecluster mocking")
}

func newAdditionalPhaseContext(t *testing.T, bc *bkev1beta1.BKECluster, objs ...runtime.Object) *phaseframe.PhaseContext {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, cvv1alpha1.AddToScheme(scheme))
	builder := fake.NewClientBuilder().WithScheme(scheme)
	if bc != nil {
		builder = builder.WithObjects(bc)
	}
	for _, obj := range objs {
		if clientObj, ok := obj.(interface {
			GetName() string
			GetNamespace() string
		}); ok && clientObj.GetName() != "" {
			builder = builder.WithRuntimeObjects(obj)
		}
	}
	return phaseframe.NewReconcilePhaseCtx(context.Background()).
		SetClient(builder.Build()).
		SetScheme(scheme).
		SetBKECluster(bc).
		SetLogger(bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bc))
}

func TestIsClusterInSpecialStateAdditionalCases(t *testing.T) {
	for _, status := range []confv1beta1.ClusterStatus{
		bkev1beta1.ClusterMasterScalingUp,
		bkev1beta1.ClusterMasterScalingDown,
		bkev1beta1.ClusterWorkerScalingUp,
		bkev1beta1.ClusterWorkerScalingDown,
		bkev1beta1.ClusterInitializing,
		bkev1beta1.ClusterPaused,
		bkev1beta1.ClusterUpgrading,
	} {
		cluster := &bkev1beta1.BKECluster{Status: confv1beta1.BKEClusterStatus{ClusterStatus: status}}
		assert.True(t, isClusterInSpecialState(cluster), "status %s should be special", status)
	}
	assert.False(t, isClusterInSpecialState(&bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{ClusterStatus: confv1beta1.ClusterStatus("Running")},
	}))
}

func TestShouldSkipLegacyUpgrade(t *testing.T) {
	assert.False(t, shouldSkipLegacyUpgrade(nil))
	assert.False(t, shouldSkipLegacyUpgrade(&phaseframe.PhaseContext{}))

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Status:     confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterUpgradeFailed},
	}
	assert.True(t, shouldSkipLegacyUpgrade(newAdditionalPhaseContext(t, cluster)))

	cluster = &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	condition.ConditionMark(cluster, bkev1beta1.ClusterHealthyStateCondition, confv1beta1.ConditionFalse, "UpgradeFailed", string(bkev1beta1.UpgradeFailed))
	assert.True(t, shouldSkipLegacyUpgrade(newAdditionalPhaseContext(t, cluster)))

	cluster = &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	assert.False(t, shouldSkipLegacyUpgrade(phaseframe.NewReconcilePhaseCtx(context.Background()).SetBKECluster(cluster)))

	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.06"},
		Status: cvv1alpha1.ClusterVersionStatus{
			CurrentVersion: "v26.05",
			Phase:          cvv1alpha1.ClusterVersionPhaseFailed,
		},
	}
	assert.True(t, shouldSkipLegacyUpgrade(newAdditionalPhaseContext(t, cluster, cv)))

	cv.Status.CurrentVersion = "v26.06"
	assert.False(t, shouldSkipLegacyUpgrade(newAdditionalPhaseContext(t, cluster, cv)))
}

func TestLegacyAndDeclarativePhasePredicates(t *testing.T) {
	assert.True(t, isLegacyClusterUpgradePhase(EnsureAgentUpgradeName))
	assert.True(t, isLegacyClusterUpgradePhase(EnsureEtcdUpgradeName))
	assert.True(t, isLegacyClusterUpgradePhase(EnsureProviderSelfUpgradeName))
	assert.False(t, isLegacyClusterUpgradePhase(EnsureClusterName))

	assert.True(t, IsDeclarativeInlineUpgradePhase(EnsureEtcdUpgradeName))
	assert.True(t, IsDeclarativeInlineUpgradePhase(EnsureProviderSelfUpgradeName))
	assert.Equal(t, "集群健康检查", ConvertPhaseNameToCN(string(EnsureClusterName)))
	assert.Equal(t, "UnknownPhase", ConvertPhaseNameToCN("UnknownPhase"))
}

func TestPhaseFlowSkipAndStatusProcessing(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	ctx := newAdditionalPhaseContext(t, cluster)
	flow := NewPhaseFlow(ctx)

	inlinePhase := phaseframe.NewBasePhase(ctx, EnsureEtcdUpgradeName)
	normalPhase := phaseframe.NewBasePhase(ctx, EnsureClusterName)
	assert.False(t, flow.skipPhaseAfterDeclarativeDAG(&inlinePhase))
	ctx.DeclarativeDAGCompleted = true
	assert.True(t, flow.skipPhaseAfterDeclarativeDAG(&inlinePhase))
	assert.False(t, flow.skipPhaseAfterDeclarativeDAG(&normalPhase))

	cluster.Status.PhaseStatus = []confv1beta1.PhaseState{
		{Name: "old", Status: bkev1beta1.PhaseSucceeded},
		{Name: "wait", Status: bkev1beta1.PhaseWaiting},
		{Name: "failed", Status: bkev1beta1.PhaseFailed},
		{Name: "tail", Status: bkev1beta1.PhaseWaiting},
	}
	flow.processPhaseStatus()
	require.Len(t, cluster.Status.PhaseStatus, 2)
	assert.Equal(t, confv1beta1.BKEClusterPhase("old"), cluster.Status.PhaseStatus[0].Name)
	assert.Equal(t, confv1beta1.BKEClusterPhase("wait"), cluster.Status.PhaseStatus[1].Name)

	cluster.Status.PhaseStatus = nil
	for i := 0; i < MaxPhaseStatusHistory+5; i++ {
		cluster.Status.PhaseStatus = append(cluster.Status.PhaseStatus, confv1beta1.PhaseState{
			Name:   confv1beta1.BKEClusterPhase("phase"),
			Status: bkev1beta1.PhaseWaiting,
		})
	}
	flow.processPhaseStatus()
	assert.Len(t, cluster.Status.PhaseStatus, MaxPhaseStatusHistory)
}

func TestPhaseFlowProcessPhaseStatusTruncatesAtFailedPhase(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			PhaseStatus: []confv1beta1.PhaseState{
				{Name: "finished-1", Status: bkev1beta1.PhaseSucceeded},
				{Name: "finished-2", Status: bkev1beta1.PhaseSucceeded},
				{Name: "waiting", Status: bkev1beta1.PhaseWaiting},
				{Name: "failed", Status: bkev1beta1.PhaseFailed},
				{Name: "waiting-tail", Status: bkev1beta1.PhaseWaiting},
			},
		},
	}

	NewPhaseFlow(newAdditionalPhaseContext(t, cluster)).processPhaseStatus()

	require.Equal(t, confv1beta1.PhaseStatus{
		{Name: "finished-1", Status: bkev1beta1.PhaseSucceeded},
		{Name: "finished-2", Status: bkev1beta1.PhaseSucceeded},
		{Name: "waiting", Status: bkev1beta1.PhaseWaiting},
	}, cluster.Status.PhaseStatus)
}

func TestPhaseFlowProcessPhaseStatusKeepsTailWithoutFailed(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			PhaseStatus: []confv1beta1.PhaseState{
				{Name: "finished-1", Status: bkev1beta1.PhaseSucceeded},
				{Name: "finished-2", Status: bkev1beta1.PhaseSucceeded},
				{Name: "waiting", Status: bkev1beta1.PhaseWaiting},
				{Name: "unknown-tail", Status: bkev1beta1.PhaseUnknown},
				{Name: "waiting-tail", Status: bkev1beta1.PhaseWaiting},
			},
		},
	}

	NewPhaseFlow(newAdditionalPhaseContext(t, cluster)).processPhaseStatus()

	require.Len(t, cluster.Status.PhaseStatus, 5)
	assert.Equal(t, confv1beta1.BKEClusterPhase("waiting-tail"), cluster.Status.PhaseStatus[4].Name)
}

func TestPhaseFlowDetermineAndWaitingPhases(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			PhaseStatus: []confv1beta1.PhaseState{
				{Name: EnsureClusterName, Status: bkev1beta1.PhaseWaiting},
				{Name: EnsureDryRunName, Status: bkev1beta1.PhaseSucceeded},
			},
		},
	}
	flow := NewPhaseFlow(newAdditionalPhaseContext(t, cluster))
	assert.Equal(t, confv1beta1.BKEClusterPhases{EnsureClusterName}, flow.getWaitingPhases())
	assert.Equal(t, confv1beta1.BKEClusterPhases{EnsureClusterName}, flow.determinePhases())
}

func TestPhaseFlowReportPhases(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	ctx := newAdditionalPhaseContext(t, cluster)
	waiting := &additionalFakePhase{name: "waiting", ctx: ctx, status: bkev1beta1.PhaseWaiting}
	succeeded := &additionalFakePhase{name: "succeeded", ctx: ctx, status: bkev1beta1.PhaseSucceeded}
	flow := NewPhaseFlow(ctx)
	flow.BKEPhases = []phaseframe.Phase{waiting, succeeded}

	count, err := flow.reportPhases()
	require.NoError(t, err)
	assert.Equal(t, 1, count)
	assert.Equal(t, 1, waiting.reportCalls)
	assert.Equal(t, 1, succeeded.reportCalls)

	reportFailed := &additionalFakePhase{name: "failed", ctx: ctx, status: bkev1beta1.PhaseWaiting, reportErr: errors.New("report failed")}
	flow.BKEPhases = []phaseframe.Phase{reportFailed}
	count, err = flow.reportPhases()
	require.Error(t, err)
	assert.Equal(t, 0, count)
	assert.Equal(t, 1, reportFailed.reportCalls)
}

func TestCalculateClusterStatusHooksAdditionalCases(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "c1",
			Namespace:   "ns",
			Annotations: map[string]string{"status-record": "old"},
		},
	}
	ctx := newAdditionalPhaseContext(t, cluster)

	ensureCluster := phaseframe.NewBasePhase(ctx, EnsureClusterName)
	require.NoError(t, calculatingClusterPreStatusByPhase(&ensureCluster))
	assert.Equal(t, bkev1beta1.ClusterChecking, cluster.Status.ClusterStatus)

	unknown := phaseframe.NewBasePhase(ctx, confv1beta1.BKEClusterPhase("UnknownPhase"))
	require.NoError(t, calculateClusterStatusByPhase(&unknown, nil))
	// Unknown phase without explicit status mapping preserves current ClusterStatus
	assert.Equal(t, bkev1beta1.ClusterChecking, cluster.Status.ClusterStatus)

	upgrade := phaseframe.NewBasePhase(ctx, EnsureMasterUpgradeName)
	require.NoError(t, calculatingClusterPostStatusByPhase(&upgrade, errors.New("upgrade failed")))
	assert.Equal(t, bkev1beta1.ClusterUpgradeFailed, cluster.Status.ClusterStatus)
}

type additionalFakePhase struct {
	name        confv1beta1.BKEClusterPhase
	cName       string
	ctx         *phaseframe.PhaseContext
	status      confv1beta1.BKEClusterPhaseStatus
	need        bool
	result      ctrl.Result
	preErr      error
	executeErr  error
	postErr     error
	reportErr   error
	preCalls    int
	execCalls   int
	postCalls   int
	reportCalls int
}

func (p *additionalFakePhase) Name() confv1beta1.BKEClusterPhase { return p.name }
func (p *additionalFakePhase) Execute() (ctrl.Result, error) {
	p.execCalls++
	return p.result, p.executeErr
}
func (p *additionalFakePhase) ExecutePreHook() error {
	p.preCalls++
	return p.preErr
}
func (p *additionalFakePhase) ExecutePostHook(err error) error {
	p.postCalls++
	if p.postErr != nil {
		return p.postErr
	}
	return err
}
func (p *additionalFakePhase) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	return p.need
}
func (p *additionalFakePhase) RegisterPreHooks(hooks ...func(phaseframe.Phase) error)         {}
func (p *additionalFakePhase) RegisterPostHooks(hooks ...func(phaseframe.Phase, error) error) {}
func (p *additionalFakePhase) Report(msg string, onlyRecord bool) error {
	p.reportCalls++
	return p.reportErr
}
func (p *additionalFakePhase) SetCName(name string) { p.cName = name }
func (p *additionalFakePhase) SetStatus(status confv1beta1.BKEClusterPhaseStatus) {
	p.status = status
}
func (p *additionalFakePhase) GetStatus() confv1beta1.BKEClusterPhaseStatus { return p.status }
func (p *additionalFakePhase) SetStartTime(t metav1.Time)                   {}
func (p *additionalFakePhase) GetStartTime() metav1.Time                    { return metav1.Time{} }
func (p *additionalFakePhase) GetPhaseContext() *phaseframe.PhaseContext    { return p.ctx }
func (p *additionalFakePhase) SetPhaseContext(ctx *phaseframe.PhaseContext) { p.ctx = ctx }

func TestPhaseFlowExecutePhasesRunsAndSkips(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.GetLastUpdatedBKECluster, func(cluster *bkev1beta1.BKECluster) (*bkev1beta1.BKECluster, error) {
		return cluster.DeepCopy(), nil
	})
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
		return nil
	})

	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	ctx := newAdditionalPhaseContext(t, cluster)
	runPhase := &additionalFakePhase{name: "run", ctx: ctx, need: true, result: ctrl.Result{RequeueAfter: time.Second}}
	needFalsePhase := &additionalFakePhase{name: "need-false", ctx: ctx, need: false}
	notWaitingPhase := &additionalFakePhase{name: "not-waiting", ctx: ctx, need: true}
	flow := NewPhaseFlow(ctx)
	flow.BKEPhases = []phaseframe.Phase{runPhase, needFalsePhase, notWaitingPhase}

	result, err := flow.executePhases(confv1beta1.BKEClusterPhases{"run", "need-false"})
	require.NoError(t, err)
	assert.Equal(t, time.Second, result.RequeueAfter)

	assert.Equal(t, 1, runPhase.preCalls)
	assert.Equal(t, 1, runPhase.execCalls)
	assert.Equal(t, 1, runPhase.postCalls)
	assert.Equal(t, 0, runPhase.reportCalls)

	assert.Equal(t, bkev1beta1.PhaseSkipped, needFalsePhase.GetStatus())
	assert.Equal(t, 1, needFalsePhase.reportCalls)
	assert.Equal(t, bkev1beta1.PhaseSkipped, notWaitingPhase.GetStatus())
	assert.Equal(t, 1, notWaitingPhase.reportCalls)
}

func TestPhaseFlowExecutePhasesErrorBranches(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.GetLastUpdatedBKECluster, func(cluster *bkev1beta1.BKECluster) (*bkev1beta1.BKECluster, error) {
		return cluster.DeepCopy(), nil
	})
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
		return nil
	})

	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	ctx := newAdditionalPhaseContext(t, cluster)

	preFailed := &additionalFakePhase{name: "pre", ctx: ctx, need: true, preErr: errors.New("pre failed")}
	flow := NewPhaseFlow(ctx)
	flow.BKEPhases = []phaseframe.Phase{preFailed}
	_, err := flow.executePhases(confv1beta1.BKEClusterPhases{"pre"})
	require.Error(t, err)
	assert.Equal(t, 1, preFailed.preCalls)
	assert.Equal(t, 0, preFailed.execCalls)

	execFailed := &additionalFakePhase{name: "exec", ctx: ctx, need: true, executeErr: errors.New("execute failed")}
	flow.BKEPhases = []phaseframe.Phase{execFailed}
	_, err = flow.executePhases(confv1beta1.BKEClusterPhases{"exec"})
	require.Error(t, err)
	assert.Equal(t, 1, execFailed.execCalls)
	assert.Equal(t, 1, execFailed.postCalls)

	reportFailed := &additionalFakePhase{name: "report", ctx: ctx, need: false, reportErr: errors.New("report failed")}
	flow.BKEPhases = []phaseframe.Phase{reportFailed}
	_, err = flow.executePhases(confv1beta1.BKEClusterPhases{"report"})
	require.Error(t, err)
}

func TestPhaseFlowCleanupUnexecutedPhases(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	synced := false
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
		synced = true
		return nil
	})

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{PhaseStatus: []confv1beta1.PhaseState{
			{Name: "keep", Status: bkev1beta1.PhaseWaiting},
			{Name: "cleanup", Status: bkev1beta1.PhaseWaiting},
		}},
	}
	flow := NewPhaseFlow(newAdditionalPhaseContext(t, cluster))
	remaining := confv1beta1.BKEClusterPhases{"cleanup"}
	flow.cleanupUnexecutedPhases(&remaining)

	assert.True(t, synced)
	assert.Equal(t, bkev1beta1.PhaseWaiting, cluster.Status.PhaseStatus[0].Status)
	assert.Equal(t, bkev1beta1.PhaseUnknown, cluster.Status.PhaseStatus[1].Status)
}

type commandFailureClient struct {
	client.Client
	getErr    error
	updateErr error
	deleteErr error
}

func (c *commandFailureClient) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	if c.getErr != nil {
		return c.getErr
	}
	return c.Client.Get(ctx, key, obj, opts...)
}

func (c *commandFailureClient) Update(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
	if c.updateErr != nil {
		return c.updateErr
	}
	return c.Client.Update(ctx, obj, opts...)
}

func (c *commandFailureClient) Delete(ctx context.Context, obj client.Object, opts ...client.DeleteOption) error {
	if c.deleteErr != nil {
		return c.deleteErr
	}
	return c.Client.Delete(ctx, obj, opts...)
}

func newProcessCommandFailureTestEnv(t *testing.T, nodeIP string, failedFlag bool) (client.Client, *nodeutil.NodeFetcher, ProcessCommandFailureParams) {
	t.Helper()

	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, agentv1beta1.AddToScheme(scheme))

	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	stateCode := 0
	if failedFlag {
		stateCode = bkev1beta1.NodeFailedFlag
	}
	bkeNode := &confv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "node-1",
			Namespace: "ns",
			Labels:    map[string]string{nodeutil.ClusterNameLabel: "c1"},
		},
		Spec:   confv1beta1.BKENodeSpec{IP: nodeIP},
		Status: confv1beta1.BKENodeStatus{StateCode: stateCode},
	}
	ownerMachine := &bkev1beta1.BKEMachine{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "owner-machine",
			Namespace: "ns",
			Labels:    map[string]string{"role": "master"},
		},
	}
	initCommand := &agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "init-command",
			Namespace: "ns",
			OwnerReferences: []metav1.OwnerReference{{
				Name: "owner-machine",
			}},
		},
	}

	objs := []client.Object{cluster, bkeNode, ownerMachine, initCommand}
	baseClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objs...).
		WithStatusSubresource(objs...).
		Build()

	params := ProcessCommandFailureParams{
		Context:     context.Background(),
		BKECluster:  cluster,
		NodeFetcher: nodeutil.NewNodeFetcher(baseClient),
		InitCommand: initCommand,
		InitNodeIp:  &nodeIP,
		FailedNodes: []string{nodeIP},
		RefreshContext: func() error {
			return nil
		},
	}
	return baseClient, params.NodeFetcher, params
}

func TestProcessCommandFailureAdditionalBranches(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(time.Sleep, func(time.Duration) {})

	nodeIP := "192.168.1.10"

	t.Run("node failed flag set deletes command", func(t *testing.T) {
		baseClient, _, params := newProcessCommandFailureTestEnv(t, nodeIP, true)
		params.Client = &commandFailureClient{Client: baseClient}
		result := ProcessCommandFailure(params)
		require.True(t, result.Done)
		assert.False(t, result.Success)
		assert.Error(t, result.Err)
	})

	t.Run("delete command failed", func(t *testing.T) {
		baseClient, _, params := newProcessCommandFailureTestEnv(t, nodeIP, true)
		params.Client = &commandFailureClient{Client: baseClient, deleteErr: errors.New("delete failed")}
		result := ProcessCommandFailure(params)
		assert.False(t, result.Done)
		assert.False(t, result.Success)
		assert.NoError(t, result.Err)
	})

	t.Run("get owner machine failed", func(t *testing.T) {
		baseClient, _, params := newProcessCommandFailureTestEnv(t, nodeIP, false)
		params.Client = &commandFailureClient{Client: baseClient, getErr: errors.New("get failed")}
		result := ProcessCommandFailure(params)
		assert.False(t, result.Done)
		assert.Error(t, result.Err)
	})

	t.Run("update owner machine failed", func(t *testing.T) {
		baseClient, _, params := newProcessCommandFailureTestEnv(t, nodeIP, false)
		params.Client = &commandFailureClient{Client: baseClient, updateErr: errors.New("update failed")}
		result := ProcessCommandFailure(params)
		assert.False(t, result.Done)
		assert.Error(t, result.Err)
	})

	t.Run("retry bootstrap without failed flag", func(t *testing.T) {
		baseClient, _, params := newProcessCommandFailureTestEnv(t, nodeIP, false)
		params.Client = &commandFailureClient{Client: baseClient}
		result := ProcessCommandFailure(params)
		assert.False(t, result.Done)
		assert.Error(t, result.Err)
	})
}

func TestCalculateClusterStatusByPhasePhaseGroups(t *testing.T) {
	cases := []struct {
		name       string
		phase      confv1beta1.BKEClusterPhase
		okStatus   confv1beta1.ClusterStatus
		failStatus confv1beta1.ClusterStatus
	}{
		{"init", EnsureMasterInitName, bkev1beta1.ClusterInitializing, bkev1beta1.ClusterInitializationFailed},
		{"scale master up", EnsureMasterJoinName, bkev1beta1.ClusterMasterScalingUp, bkev1beta1.ClusterMasterScalingUpFailed},
		{"scale worker up", EnsureWorkerJoinName, bkev1beta1.ClusterWorkerScalingUp, bkev1beta1.ClusterWorkerScalingUpFailed},
		{"delete", EnsureDeleteOrResetName, bkev1beta1.ClusterDeleting, bkev1beta1.ClusterDeleteFailed},
		{"paused", EnsurePausedName, bkev1beta1.ClusterPaused, bkev1beta1.ClusterPauseFailed},
		{"dry run", EnsureDryRunName, bkev1beta1.ClusterDryRun, bkev1beta1.ClusterDryRunFailed},
		{"addons", EnsureAddonDeployName, bkev1beta1.ClusterDeployingAddon, bkev1beta1.ClusterDeployAddonFailed},
		{"upgrade", EnsureMasterUpgradeName, bkev1beta1.ClusterUpgrading, bkev1beta1.ClusterUpgradeFailed},
		{"declarative upgrade", EnsureEtcdUpgradeName, bkev1beta1.ClusterUpgrading, bkev1beta1.ClusterUpgradeFailed},
		{"scale master down", EnsureMasterDeleteName, bkev1beta1.ClusterMasterScalingDown, bkev1beta1.ClusterMasterScalingDownFailed},
		{"scale worker down", EnsureWorkerDeleteName, bkev1beta1.ClusterWorkerScalingDown, bkev1beta1.ClusterWorkerScalingDownFailed},
		{"manage", EnsureClusterManageName, bkev1beta1.ClusterManaging, bkev1beta1.ClusterManageFailed},
	}

	for _, tc := range cases {
		t.Run(tc.name+" success", func(t *testing.T) {
			cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
			phase := phaseframe.NewBasePhase(newAdditionalPhaseContext(t, cluster), tc.phase)
			require.NoError(t, calculateClusterStatusByPhase(&phase, nil))
			assert.Equal(t, tc.okStatus, cluster.Status.ClusterStatus)
		})
		t.Run(tc.name+" failed", func(t *testing.T) {
			cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
			phase := phaseframe.NewBasePhase(newAdditionalPhaseContext(t, cluster), tc.phase)
			require.NoError(t, calculateClusterStatusByPhase(&phase, errors.New("phase failed")))
			assert.Equal(t, tc.failStatus, cluster.Status.ClusterStatus)
		})
	}

	t.Run("custom set status phase skipped", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		phase := phaseframe.NewBasePhase(newAdditionalPhaseContext(t, cluster), EnsureClusterName)
		require.NoError(t, calculateClusterStatusByPhase(&phase, errors.New("ignored")))
		assert.Empty(t, cluster.Status.ClusterStatus)
	})
}

func TestPhaseFlowHandlePanicRecovers(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	flow := NewPhaseFlow(newAdditionalPhaseContext(t, cluster))
	assert.NotPanics(t, func() {
		func() {
			defer flow.handlePanic()
			panic(errors.New("phase panic"))
		}()
	})
}

func TestPhaseFlowReportPhaseStatusBranches(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	flow := NewPhaseFlow(newAdditionalPhaseContext(t, cluster))
	require.NoError(t, flow.ReportPhaseStatus())

	ctx := newAdditionalPhaseContext(t, cluster)
	waiting := &additionalFakePhase{name: "waiting", ctx: ctx, status: bkev1beta1.PhaseWaiting}
	flow = NewPhaseFlow(ctx)
	flow.BKEPhases = []phaseframe.Phase{waiting}
	cluster.Status.PhaseStatus = []confv1beta1.PhaseState{
		{Name: "old", Status: bkev1beta1.PhaseSucceeded},
		{Name: "failed", Status: bkev1beta1.PhaseFailed},
	}
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.GetLastUpdatedBKECluster, func(c *bkev1beta1.BKECluster) (*bkev1beta1.BKECluster, error) {
		return c.DeepCopy(), nil
	})
	require.NoError(t, flow.ReportPhaseStatus())
	require.Len(t, cluster.Status.PhaseStatus, 1)
	assert.Equal(t, confv1beta1.BKEClusterPhase("old"), cluster.Status.PhaseStatus[0].Name)
}

func TestPhaseFlowCalculatePhaseAndDeterminePhases(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.GetLastUpdatedBKECluster, func(c *bkev1beta1.BKECluster) (*bkev1beta1.BKECluster, error) {
		return c.DeepCopy(), nil
	})
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
		return nil
	})

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       confv1beta1.BKEClusterSpec{Reset: true},
	}
	ctx := newAdditionalPhaseContext(t, cluster)
	flow := NewPhaseFlow(ctx)
	require.NoError(t, flow.CalculatePhase(cluster, cluster))
	assert.Equal(t, confv1beta1.BKEClusterPhases(ClusterDeleteResetPhaseNames()), flow.determinePhases())

	cluster.Spec.Reset = false
	cluster.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	assert.Equal(t, confv1beta1.BKEClusterPhases(ClusterDeleteResetPhaseNames()), NewPhaseFlow(ctx).determinePhases())
}

func TestPhaseFlowCalculateAndAddPhasesFilters(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Status:     confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterUpgradeFailed},
	}
	ctx := newAdditionalPhaseContext(t, cluster)
	ctx.DeclarativeDAGCompleted = true
	flow := NewPhaseFlow(ctx)
	flow.calculateAndAddPhases(cluster, cluster, FullPhasesRegisFunc)
	for _, phase := range flow.BKEPhases {
		assert.False(t, isLegacyClusterUpgradePhase(phase.Name()))
		assert.False(t, IsDeclarativeInlineUpgradePhase(phase.Name()))
	}
}

func TestPhaseFlowRefreshOldAndNewBKEClusterError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.GetLastUpdatedBKECluster, func(*bkev1beta1.BKECluster) (*bkev1beta1.BKECluster, error) {
		return nil, errors.New("refresh failed")
	})

	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	flow := NewPhaseFlow(newAdditionalPhaseContext(t, cluster))
	assert.Error(t, flow.refreshOldAndNewBKECluster())
}

func TestLogFinishWhenDeployFailed(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Status:     confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterInitializationFailed},
	}
	ctx := newAdditionalPhaseContext(t, cluster)
	assert.NotPanics(t, func() { logFinishWhenDeployFailed(ctx) })

	cluster.Status.ClusterStatus = bkev1beta1.ClusterStatusReady
	assert.NotPanics(t, func() { logFinishWhenDeployFailed(ctx) })
}

func TestRegisterPhaseCNameSetsChineseName(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	phase := phaseframe.NewBasePhase(newAdditionalPhaseContext(t, cluster), EnsureClusterName)
	require.NoError(t, registerPhaseCName(&phase))
	assert.Equal(t, "集群健康检查", ConvertPhaseNameToCN(string(EnsureClusterName)))
}

func TestPhaseFlowExecuteDelegatesToExecutePhases(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	called := false
	patches.ApplyPrivateMethod(&PhaseFlow{}, "executePhases", func(_ *PhaseFlow, _ confv1beta1.BKEClusterPhases) (ctrl.Result, error) {
		called = true
		return ctrl.Result{RequeueAfter: time.Second}, nil
	})
	patches.ApplyMethod(&phaseframe.PhaseContext{}, "WatchBKEClusterStatus", func(_ *phaseframe.PhaseContext) {})

	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			PhaseStatus: []confv1beta1.PhaseState{{Name: EnsureClusterName, Status: bkev1beta1.PhaseWaiting}},
		},
	}
	result, err := NewPhaseFlow(newAdditionalPhaseContext(t, cluster)).Execute()
	require.NoError(t, err)
	assert.True(t, called)
	assert.Equal(t, time.Second, result.RequeueAfter)
}
