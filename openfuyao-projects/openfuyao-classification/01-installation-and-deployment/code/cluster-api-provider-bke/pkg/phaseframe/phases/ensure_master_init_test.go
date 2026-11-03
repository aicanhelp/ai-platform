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
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util/conditions"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func createTestEnsureMasterInit() *EnsureMasterInit {
	logger := createTestLogger()
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{},
	}

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: bkeCluster,
		Scheme:     scheme,
		Log:        logger,
	}

	return &EnsureMasterInit{
		BasePhase: phaseframe.NewBasePhase(ctx, EnsureMasterInitName),
	}
}

func TestEnsureMasterInitConstants(t *testing.T) {
	assert.Equal(t, "EnsureMasterInit", string(EnsureMasterInitName))
	assert.Equal(t, 10, MasterInitLogIntervalCount)
	assert.Equal(t, 2, MasterInitSleepSeconds)
	assert.Equal(t, 1, MasterInitPollIntervalSeconds)
}

func TestNewEnsureMasterInit(t *testing.T) {
	logger := createTestLogger()
	ctx := &phaseframe.PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
		Scheme: runtime.NewScheme(),
		Log:    logger,
	}

	phase := NewEnsureMasterInit(ctx)
	assert.NotNil(t, phase)
	assert.IsType(t, &EnsureMasterInit{}, phase)
}

func TestEnsureMasterInit_ValidateMasterNodes_NoMasterNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureMasterInit()

	patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
		return &nodeutil.FetchResult{Nodes: bkenode.Nodes{}}, nil
	})

	params := ValidateMasterNodesParams{
		Ctx: e.Ctx,
	}

	nodes, count, err := e.validateMasterNodes(params)
	assert.Error(t, err)
	assert.Nil(t, nodes)
	assert.Equal(t, 0, count)
}

func TestEnsureMasterInit_NeedExecute_NotReady(t *testing.T) {
	e := createTestEnsureMasterInit()
	now := metav1.Now()
	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			DeletionTimestamp: &now,
		},
	}

	result := e.NeedExecute(old, new)
	assert.False(t, result)
}

func TestValidateMasterNodesParams_Structure(t *testing.T) {
	e := createTestEnsureMasterInit()
	params := ValidateMasterNodesParams{
		Ctx: e.Ctx,
	}
	assert.NotNil(t, params.Ctx)
}

func TestSetupConditionAndRefreshParams_Structure(t *testing.T) {
	e := createTestEnsureMasterInit()
	params := SetupConditionAndRefreshParams{
		Ctx: e.Ctx,
	}
	assert.NotNil(t, params.Ctx)
}

func newMasterInitPhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureMasterInit {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	builder := ctrlfake.NewClientBuilder().WithScheme(scheme)
	if bkeCluster != nil {
		builder = builder.WithObjects(bkeCluster)
	}
	c := builder.Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureMasterInit{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

func clusterWithInitCond(t *testing.T, initialized bool) *clusterv1.Cluster {
	t.Helper()
	cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	if initialized {
		conditions.Set(cluster, &clusterv1.Condition{Type: clusterv1.ControlPlaneInitializedCondition, Status: corev1.ConditionTrue})
	}
	return cluster
}

func masterInitPollParams(e *EnsureMasterInit) MasterInitPollParams {
	cf, mf := false, false
	ip := ""
	return MasterInitPollParams{Ctx: e.Ctx, Timeout: context.Background(), CommandCompleteFlag: &cf, MachineBootFlag: &mf, InitNodeIp: &ip}
}

// ---- ExecutePreHook ----

func TestEnsureMasterInitExecutePreHook(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	patches.ApplyMethod(&e.BasePhase, "DefaultPreHook", func(_ *phaseframe.BasePhase) error { return nil })
	assert.NoError(t, e.ExecutePreHook())
}

// ---- setupConditionAndRefresh ----

func TestEnsureMasterInitSetupConditionAndRefresh(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		require.NoError(t, e.setupConditionAndRefresh(SetupConditionAndRefreshParams{Ctx: e.Ctx}))
	})

	t.Run("refresh error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error {
			return assertErr("refresh failed")
		})
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		err := e.setupConditionAndRefresh(SetupConditionAndRefreshParams{Ctx: e.Ctx})
		require.Error(t, err)
	})
}

// ---- validateMasterNodes ----

func TestEnsureMasterInitValidateMasterNodes(t *testing.T) {
	t.Run("no master nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: bkenode.Nodes{}}, nil
		})
		_, _, err := e.validateMasterNodes(ValidateMasterNodesParams{Ctx: e.Ctx})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no master node")
	})

	t.Run("some ready nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"master"}}, {IP: "10.0.0.2", Role: []string{"master"}}}}, nil
		})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlag, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ int) (bool, error) {
			return true, nil // env ready
		})
		nodes, count, err := e.validateMasterNodes(ValidateMasterNodesParams{Ctx: e.Ctx})
		require.NoError(t, err)
		assert.Len(t, nodes, 2)
		assert.Equal(t, 0, count)
	})

	t.Run("all not ready", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"master"}}}}, nil
		})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlag, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ int) (bool, error) {
			return false, nil // env not ready
		})
		_, _, err := e.validateMasterNodes(ValidateMasterNodesParams{Ctx: e.Ctx})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not ready")
	})
}

// ---- checkClusterInitialized ----

func TestEnsureMasterInitCheckClusterInitialized(t *testing.T) {
	pollCount := 0
	t.Run("refresh cluster error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return assertErr("refresh failed") })
		_, err := e.checkClusterInitialized(CheckClusterInitializedParams{Ctx: e.Ctx, PollCount: &pollCount})
		require.Error(t, err)
	})

	t.Run("already initialized", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.Ctx.Cluster = clusterWithInitCond(t, true)
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		done, err := e.checkClusterInitialized(CheckClusterInitializedParams{Ctx: e.Ctx, PollCount: &pollCount})
		require.NoError(t, err)
		assert.True(t, done)
	})

	t.Run("not initialized", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.Ctx.Cluster = clusterWithInitCond(t, false)
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		done, err := e.checkClusterInitialized(CheckClusterInitializedParams{Ctx: e.Ctx, PollCount: &pollCount})
		require.NoError(t, err)
		assert.False(t, done)
	})
}

// ---- checkClusterInitializedStep + checkClusterFinalStep ----

func TestEnsureMasterInitCheckSteps(t *testing.T) {
	t.Run("initialized step done", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.Ctx.Cluster = clusterWithInitCond(t, true)
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		params := masterInitPollParams(e)
		done, success, err := e.checkClusterInitializedStep(params, 1)
		require.NoError(t, err)
		assert.True(t, done)
		assert.True(t, success)
	})

	t.Run("final step not initialized", func(t *testing.T) {
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.Ctx.Cluster = clusterWithInitCond(t, false)
		params := masterInitPollParams(e)
		done, _, err := e.checkClusterFinalStep(params, 1)
		require.NoError(t, err)
		assert.False(t, done)
	})
}

// ---- getInitCommandStep ----

func TestEnsureMasterInitGetInitCommandStep(t *testing.T) {
	t.Run("command complete flag set", func(t *testing.T) {
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		params := masterInitPollParams(e)
		*params.CommandCompleteFlag = true
		cmd, shouldContinue, err := e.getInitCommandStep(params, 1)
		require.NoError(t, err)
		assert.Nil(t, cmd)
		assert.False(t, shouldContinue)
	})

	t.Run("command not found", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetMasterInitCommand, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
			return nil, assertErr("command not found")
		})
		params := masterInitPollParams(e)
		cmd, shouldContinue, err := e.getInitCommandStep(params, 1)
		require.NoError(t, err)
		assert.Nil(t, cmd)
		assert.False(t, shouldContinue)
	})

	t.Run("other error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetMasterInitCommand, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
			return nil, assertErr("boom")
		})
		params := masterInitPollParams(e)
		_, _, err := e.getInitCommandStep(params, 1)
		require.Error(t, err)
	})

	t.Run("success sets init node ip", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetMasterInitCommand, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
			return &agentv1beta1.Command{Spec: agentv1beta1.CommandSpec{NodeName: "10.0.0.1"}}, nil
		})
		params := masterInitPollParams(e)
		cmd, shouldContinue, err := e.getInitCommandStep(params, 1)
		require.NoError(t, err)
		assert.NotNil(t, cmd)
		assert.True(t, shouldContinue)
		assert.Equal(t, "10.0.0.1", *params.InitNodeIp)
	})
}

// ---- processCommandComplete ----

func TestEnsureMasterInitProcessCommandComplete(t *testing.T) {
	t.Run("complete with failed nodes -> processCommandFailure", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "processCommandFailure", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ client.Client, _ *agentv1beta1.Command, _ []string, _ int) (bool, bool, error) {
			return true, false, assertErr("failure handled")
		})
		params := masterInitPollParams(e)
		done, success, err := e.processCommandComplete(ProcessCommandCompleteParams{
			MasterInitPollParams: params, Complete: true, FailedNodes: []string{"10.0.0.1"}, PollCount: 1,
		})
		require.Error(t, err)
		assert.True(t, done)
		assert.False(t, success)
	})

	t.Run("complete with success nodes sets flag", func(t *testing.T) {
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		params := masterInitPollParams(e)
		_, _, err := e.processCommandComplete(ProcessCommandCompleteParams{
			MasterInitPollParams: params, Complete: true, SuccessNodes: []string{"10.0.0.1"}, PollCount: 1,
		})
		require.NoError(t, err)
		assert.True(t, *params.CommandCompleteFlag)
	})

	t.Run("not complete", func(t *testing.T) {
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		params := masterInitPollParams(e)
		done, _, err := e.processCommandComplete(ProcessCommandCompleteParams{MasterInitPollParams: params, Complete: false, PollCount: 1})
		require.NoError(t, err)
		assert.False(t, done)
	})
}

// ---- waitForMachineBootstrapStep + waitForMachineBootstrap ----

func TestEnsureMasterInitWaitForMachineBootstrap(t *testing.T) {
	t.Run("machine boot flag set", func(t *testing.T) {
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		params := masterInitPollParams(e)
		*params.MachineBootFlag = true
		done, _, err := e.waitForMachineBootstrapStep(params, 1)
		require.NoError(t, err)
		assert.False(t, done)
	})

	t.Run("get machine error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetControlPlaneInitBKEMachine, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*bkev1beta1.BKEMachine, error) {
			return nil, assertErr("machine not found")
		})
		params := masterInitPollParams(e)
		done, _, err := e.waitForMachineBootstrapStep(params, 1)
		require.NoError(t, err)
		assert.False(t, done)
	})

	t.Run("machine not bootstrapped", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetControlPlaneInitBKEMachine, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*bkev1beta1.BKEMachine, error) {
			return &bkev1beta1.BKEMachine{Status: bkev1beta1.BKEMachineStatus{Bootstrapped: false}}, nil
		})
		params := masterInitPollParams(e)
		done, _, err := e.waitForMachineBootstrapStep(params, 1)
		require.NoError(t, err)
		assert.False(t, done)
	})

	t.Run("machine bootstrapped sets flag", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetControlPlaneInitBKEMachine, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*bkev1beta1.BKEMachine, error) {
			return &bkev1beta1.BKEMachine{Status: bkev1beta1.BKEMachineStatus{Bootstrapped: true}}, nil
		})
		params := masterInitPollParams(e)
		_, _, err := e.waitForMachineBootstrapStep(params, 1)
		require.NoError(t, err)
		assert.True(t, *params.MachineBootFlag)
	})
}

// ---- Execute ----

func TestEnsureMasterInitExecute(t *testing.T) {
	t.Run("setup error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "setupConditionAndRefresh", func(_ *EnsureMasterInit, _ SetupConditionAndRefreshParams) error {
			return assertErr("setup failed")
		})
		_, err := e.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "setup failed")
	})

	t.Run("poll success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.Ctx.Cluster = clusterWithInitCond(t, false)
		patches.ApplyPrivateMethod(e, "setupConditionAndRefresh", func(_ *EnsureMasterInit, _ SetupConditionAndRefreshParams) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		patches.ApplyFunc(phaseutil.GetBootTimeOut, func(_ *bkev1beta1.BKECluster) (time.Duration, error) { return time.Second, nil })
		// make masterInitPollFunc return immediately with success
		patches.ApplyPrivateMethod(e, "checkClusterInitializedStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return true, true, nil
		})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		_, err := e.Execute()
		require.NoError(t, err)
	})
}

// masterInitGapsPhase builds an EnsureMasterInit phase backed by a fake client
// for testing the legacy 0%-covered methods on ensure_master_init.go.
func masterInitGapsPhase(t *testing.T) *EnsureMasterInit {
	t.Helper()
	return newMasterInitPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
}

// masterInitGapsWaitParams builds WaitForInitCommandCompleteParams with fresh
// out-parameters for a single subtest invocation.
func masterInitGapsWaitParams(e *EnsureMasterInit, pollCount int) (WaitForInitCommandCompleteParams, *string, *bool, *int) {
	ip := ""
	flag := false
	pc := pollCount
	return WaitForInitCommandCompleteParams{
		Ctx:                 e.Ctx,
		InitNodeIp:          &ip,
		CommandCompleteFlag: &flag,
		PollCount:           &pc,
	}, &ip, &flag, &pc
}

// ---- waitForInitCommandComplete ----

func TestEnsureMasterInitGapsWaitForInitCommandComplete(t *testing.T) {
	t.Run("command not found returns false nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(phaseutil.GetMasterInitCommand, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
			return nil, assertErr("command not found")
		})
		params, _, _, _ := masterInitGapsWaitParams(e, 0)
		done, err := e.waitForInitCommandComplete(params)
		require.NoError(t, err)
		assert.False(t, done)
	})

	t.Run("other error returns err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(phaseutil.GetMasterInitCommand, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
			return nil, assertErr("boom")
		})
		params, _, _, _ := masterInitGapsWaitParams(e, 0)
		done, err := e.waitForInitCommandComplete(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "boom")
		assert.False(t, done)
	})

	t.Run("complete with failed nodes done returns success and err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(phaseutil.GetMasterInitCommand, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
			return &agentv1beta1.Command{Spec: agentv1beta1.CommandSpec{NodeName: "10.0.0.1"}}, nil
		})
		patches.ApplyFunc(command.CheckCommandStatus, func(_ *agentv1beta1.Command) (bool, []string, []string) {
			return true, nil, []string{"10.0.0.1"}
		})
		patches.ApplyFunc(ProcessCommandFailure, func(_ ProcessCommandFailureParams) ProcessCommandFailureResult {
			return ProcessCommandFailureResult{Done: true, Success: false, Err: assertErr("node failed")}
		})
		params, ip, _, _ := masterInitGapsWaitParams(e, 0)
		done, err := e.waitForInitCommandComplete(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "node failed")
		assert.False(t, done) // success=false
		assert.Equal(t, "10.0.0.1", *ip)
	})

	t.Run("complete with failed nodes not done returns false and err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(phaseutil.GetMasterInitCommand, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
			return &agentv1beta1.Command{Spec: agentv1beta1.CommandSpec{NodeName: "10.0.0.1"}}, nil
		})
		patches.ApplyFunc(command.CheckCommandStatus, func(_ *agentv1beta1.Command) (bool, []string, []string) {
			return true, nil, []string{"10.0.0.1"}
		})
		patches.ApplyFunc(ProcessCommandFailure, func(_ ProcessCommandFailureParams) ProcessCommandFailureResult {
			return ProcessCommandFailureResult{Done: false, Success: false, Err: assertErr("retrying")}
		})
		params, _, _, _ := masterInitGapsWaitParams(e, 0)
		done, err := e.waitForInitCommandComplete(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "retrying")
		assert.False(t, done)
	})

	t.Run("complete with success nodes sets flag and returns true", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(phaseutil.GetMasterInitCommand, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
			return &agentv1beta1.Command{Spec: agentv1beta1.CommandSpec{NodeName: "10.0.0.1"}}, nil
		})
		patches.ApplyFunc(command.CheckCommandStatus, func(_ *agentv1beta1.Command) (bool, []string, []string) {
			return true, []string{"10.0.0.1"}, nil
		})
		params, _, flag, _ := masterInitGapsWaitParams(e, 0)
		done, err := e.waitForInitCommandComplete(params)
		require.NoError(t, err)
		assert.True(t, done)
		assert.True(t, *flag)
	})

	t.Run("not complete returns false nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(phaseutil.GetMasterInitCommand, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
			return &agentv1beta1.Command{Spec: agentv1beta1.CommandSpec{NodeName: "10.0.0.1"}}, nil
		})
		patches.ApplyFunc(command.CheckCommandStatus, func(_ *agentv1beta1.Command) (bool, []string, []string) {
			return false, nil, nil
		})
		params, _, _, _ := masterInitGapsWaitParams(e, 0)
		done, err := e.waitForInitCommandComplete(params)
		require.NoError(t, err)
		assert.False(t, done)
	})

	t.Run("init node ip from match labels", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(phaseutil.GetMasterInitCommand, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
			return &agentv1beta1.Command{Spec: agentv1beta1.CommandSpec{
				NodeSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"10.0.0.2": ""}},
			}}, nil
		})
		patches.ApplyFunc(command.CheckCommandStatus, func(_ *agentv1beta1.Command) (bool, []string, []string) {
			return false, nil, nil
		})
		params, ip, _, _ := masterInitGapsWaitParams(e, 1)
		done, err := e.waitForInitCommandComplete(params)
		require.NoError(t, err)
		assert.False(t, done)
		assert.Equal(t, "10.0.0.2", *ip)
	})
}

// ---- waitForMachineBootstrap ----

func TestEnsureMasterInitGapsWaitForMachineBootstrap(t *testing.T) {
	t.Run("error returns false nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(phaseutil.GetControlPlaneInitBKEMachine, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*bkev1beta1.BKEMachine, error) {
			return nil, assertErr("machine not found")
		})
		pollCount := 0
		done, err := e.waitForMachineBootstrap(WaitForMachineBootstrapParams{Ctx: e.Ctx, PollCount: &pollCount})
		require.NoError(t, err)
		assert.False(t, done)
	})

	t.Run("error pollCount nonzero skips log", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(phaseutil.GetControlPlaneInitBKEMachine, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*bkev1beta1.BKEMachine, error) {
			return nil, assertErr("machine not found")
		})
		pollCount := 1
		done, err := e.waitForMachineBootstrap(WaitForMachineBootstrapParams{Ctx: e.Ctx, PollCount: &pollCount})
		require.NoError(t, err)
		assert.False(t, done)
	})

	t.Run("not bootstrapped returns false nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(phaseutil.GetControlPlaneInitBKEMachine, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*bkev1beta1.BKEMachine, error) {
			return &bkev1beta1.BKEMachine{Status: bkev1beta1.BKEMachineStatus{Bootstrapped: false}}, nil
		})
		pollCount := 0
		done, err := e.waitForMachineBootstrap(WaitForMachineBootstrapParams{Ctx: e.Ctx, PollCount: &pollCount})
		require.NoError(t, err)
		assert.False(t, done)
	})

	t.Run("bootstrapped returns true", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(phaseutil.GetControlPlaneInitBKEMachine, func(context.Context, client.Client, *bkev1beta1.BKECluster) (*bkev1beta1.BKEMachine, error) {
			return &bkev1beta1.BKEMachine{Status: bkev1beta1.BKEMachineStatus{Bootstrapped: true}}, nil
		})
		pollCount := 0
		done, err := e.waitForMachineBootstrap(WaitForMachineBootstrapParams{Ctx: e.Ctx, PollCount: &pollCount})
		require.NoError(t, err)
		assert.True(t, done)
	})
}

// ---- processCommandFailure (method) ----

func TestEnsureMasterInitGapsProcessCommandFailure(t *testing.T) {
	t.Run("done true returns success and err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(ProcessCommandFailure, func(_ ProcessCommandFailureParams) ProcessCommandFailureResult {
			return ProcessCommandFailureResult{Done: true, Success: false, Err: assertErr("init failed")}
		})
		params := masterInitPollParams(e)
		done, success, err := e.processCommandFailure(params, e.Ctx.Client, &agentv1beta1.Command{}, []string{"10.0.0.1"}, 1)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "init failed")
		assert.True(t, done)
		assert.False(t, success)
	})

	t.Run("done false returns false nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(ProcessCommandFailure, func(_ ProcessCommandFailureParams) ProcessCommandFailureResult {
			return ProcessCommandFailureResult{Done: false, Success: false, Err: nil}
		})
		params := masterInitPollParams(e)
		done, success, err := e.processCommandFailure(params, e.Ctx.Client, &agentv1beta1.Command{}, []string{"10.0.0.1"}, 1)
		require.NoError(t, err)
		assert.False(t, done)
		assert.False(t, success)
	})

	t.Run("success true returns true true", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyFunc(ProcessCommandFailure, func(_ ProcessCommandFailureParams) ProcessCommandFailureResult {
			return ProcessCommandFailureResult{Done: true, Success: true, Err: nil}
		})
		params := masterInitPollParams(e)
		done, success, err := e.processCommandFailure(params, e.Ctx.Client, &agentv1beta1.Command{}, []string{"10.0.0.1"}, 1)
		require.NoError(t, err)
		assert.True(t, done)
		assert.True(t, success)
	})
}

// ---- waitForCommandCompleteStep ----

func TestEnsureMasterInitGapsWaitForCommandCompleteStep(t *testing.T) {
	t.Run("getInitCommandStep shouldContinue false returns continue", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyPrivateMethod(e, "getInitCommandStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (*agentv1beta1.Command, bool, error) {
			return nil, false, nil
		})
		params := masterInitPollParams(e)
		done, success, err := e.waitForCommandCompleteStep(params, 1)
		require.NoError(t, err)
		assert.False(t, done)
		assert.False(t, success)
	})

	t.Run("getInitCommandStep error returns err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyPrivateMethod(e, "getInitCommandStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (*agentv1beta1.Command, bool, error) {
			return nil, false, assertErr("boom")
		})
		params := masterInitPollParams(e)
		done, success, err := e.waitForCommandCompleteStep(params, 1)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "boom")
		assert.False(t, done)
		assert.False(t, success)
	})

	t.Run("nil command shouldContinue true returns continue", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyPrivateMethod(e, "getInitCommandStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (*agentv1beta1.Command, bool, error) {
			return nil, true, nil
		})
		params := masterInitPollParams(e)
		done, success, err := e.waitForCommandCompleteStep(params, 1)
		require.NoError(t, err)
		assert.False(t, done)
		assert.False(t, success)
	})

	t.Run("command not complete delegates to processCommandComplete", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyPrivateMethod(e, "getInitCommandStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (*agentv1beta1.Command, bool, error) {
			return &agentv1beta1.Command{Spec: agentv1beta1.CommandSpec{NodeName: "10.0.0.1"}}, true, nil
		})
		patches.ApplyFunc(command.CheckCommandStatus, func(_ *agentv1beta1.Command) (bool, []string, []string) {
			return false, nil, nil
		})
		called := false
		patches.ApplyPrivateMethod(e, "processCommandComplete", func(_ *EnsureMasterInit, _ ProcessCommandCompleteParams) (bool, bool, error) {
			called = true
			return false, false, nil
		})
		params := masterInitPollParams(e)
		done, success, err := e.waitForCommandCompleteStep(params, 1)
		require.NoError(t, err)
		assert.False(t, done)
		assert.False(t, success)
		assert.True(t, called)
	})

	t.Run("command complete propagates processCommandComplete result", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyPrivateMethod(e, "getInitCommandStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (*agentv1beta1.Command, bool, error) {
			return &agentv1beta1.Command{Spec: agentv1beta1.CommandSpec{NodeName: "10.0.0.1"}}, true, nil
		})
		patches.ApplyFunc(command.CheckCommandStatus, func(_ *agentv1beta1.Command) (bool, []string, []string) {
			return true, []string{"10.0.0.1"}, nil
		})
		called := false
		patches.ApplyPrivateMethod(e, "processCommandComplete", func(_ *EnsureMasterInit, _ ProcessCommandCompleteParams) (bool, bool, error) {
			called = true
			return true, true, nil
		})
		params := masterInitPollParams(e)
		done, success, err := e.waitForCommandCompleteStep(params, 5)
		require.NoError(t, err)
		assert.True(t, done)
		assert.True(t, success)
		assert.True(t, called)
	})

	t.Run("step1 err returns false err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyPrivateMethod(e, "checkClusterInitializedStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, assertErr("step1 err")
		})
		params := masterInitPollParams(e)
		pollFn := e.masterInitPollFunc(params)
		done, err := pollFn()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "step1 err")
		assert.False(t, done)
	})

	t.Run("step1 continue step2 done returns success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyPrivateMethod(e, "checkClusterInitializedStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, nil
		})
		patches.ApplyPrivateMethod(e, "waitForCommandCompleteStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return true, true, nil
		})
		params := masterInitPollParams(e)
		pollFn := e.masterInitPollFunc(params)
		done, err := pollFn()
		require.NoError(t, err)
		assert.True(t, done)
	})

	t.Run("step1 continue step2 err returns false err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyPrivateMethod(e, "checkClusterInitializedStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, nil
		})
		patches.ApplyPrivateMethod(e, "waitForCommandCompleteStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, assertErr("step2 err")
		})
		params := masterInitPollParams(e)
		pollFn := e.masterInitPollFunc(params)
		done, err := pollFn()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "step2 err")
		assert.False(t, done)
	})

	t.Run("step1 continue step2 continue step3 err returns false err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyPrivateMethod(e, "checkClusterInitializedStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, nil
		})
		patches.ApplyPrivateMethod(e, "waitForCommandCompleteStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, nil
		})
		patches.ApplyPrivateMethod(e, "waitForMachineBootstrapStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, assertErr("step3 err")
		})
		params := masterInitPollParams(e)
		pollFn := e.masterInitPollFunc(params)
		done, err := pollFn()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "step3 err")
		assert.False(t, done)
	})

	t.Run("step1-3 continue step4 done returns success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyPrivateMethod(e, "checkClusterInitializedStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, nil
		})
		patches.ApplyPrivateMethod(e, "waitForCommandCompleteStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, nil
		})
		patches.ApplyPrivateMethod(e, "waitForMachineBootstrapStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, nil
		})
		patches.ApplyPrivateMethod(e, "checkClusterFinalStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return true, true, nil
		})
		params := masterInitPollParams(e)
		pollFn := e.masterInitPollFunc(params)
		done, err := pollFn()
		require.NoError(t, err)
		assert.True(t, done)
	})

	t.Run("all steps continue returns false nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := masterInitGapsPhase(t)
		patches.ApplyPrivateMethod(e, "checkClusterInitializedStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, nil
		})
		patches.ApplyPrivateMethod(e, "waitForCommandCompleteStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, nil
		})
		patches.ApplyPrivateMethod(e, "waitForMachineBootstrapStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, nil
		})
		patches.ApplyPrivateMethod(e, "checkClusterFinalStep", func(_ *EnsureMasterInit, _ MasterInitPollParams, _ int) (bool, bool, error) {
			return false, false, nil
		})
		params := masterInitPollParams(e)
		pollFn := e.masterInitPollFunc(params)
		done, err := pollFn()
		require.NoError(t, err)
		assert.False(t, done)
	})
}

// ---- NewEnsureMasterInit ----

func TestEnsureMasterInitGapsNewEnsureMasterInit(t *testing.T) {
	e := masterInitGapsPhase(t)
	phase := NewEnsureMasterInit(e.Ctx)
	require.NotNil(t, phase)
	_, ok := phase.(*EnsureMasterInit)
	assert.True(t, ok)
}
