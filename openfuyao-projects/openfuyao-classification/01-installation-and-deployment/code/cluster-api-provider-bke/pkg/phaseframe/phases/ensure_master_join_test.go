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
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	controlv1beta1 "sigs.k8s.io/cluster-api/controlplane/kubeadm/api/v1beta1"
	"sigs.k8s.io/cluster-api/util/conditions"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func createTestEnsureMasterJoin() *EnsureMasterJoin {
	logger := createTestLogger()
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{},
		},
	}

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: bkeCluster,
		Scheme:     scheme,
		Log:        logger,
	}

	return &EnsureMasterJoin{
		BasePhase:   phaseframe.NewBasePhase(ctx, EnsureMasterJoinName),
		nodesToJoin: bkenode.Nodes{},
	}
}

func TestEnsureMasterJoinConstants(t *testing.T) {
	assert.Equal(t, "EnsureMasterJoin", string(EnsureMasterJoinName))
	assert.Equal(t, 10, LogOutputInterval)
}

func TestNewEnsureMasterJoin(t *testing.T) {
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

	phase := NewEnsureMasterJoin(ctx)
	assert.NotNil(t, phase)
	assert.IsType(t, &EnsureMasterJoin{}, phase)
}

func TestEnsureMasterJoin_NeedExecute_DefaultNeedExecuteFalse(t *testing.T) {
	e := createTestEnsureMasterJoin()
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

func TestMasterJoinParams_Structure(t *testing.T) {
	params := MasterJoinParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Log:        createTestLogger(),
	}
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.Log)
}

func newMasterJoinPhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureMasterJoin {
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
	return &EnsureMasterJoin{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

func clusterWithControlPlaneInit(t *testing.T, initialized bool) *clusterv1.Cluster {
	t.Helper()
	cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	if initialized {
		conditions.Set(cluster, &clusterv1.Condition{Type: clusterv1.ControlPlaneInitializedCondition, Status: corev1.ConditionTrue})
	} else {
		conditions.Set(cluster, &clusterv1.Condition{Type: clusterv1.ControlPlaneInitializedCondition, Status: corev1.ConditionFalse})
	}
	return cluster
}

// ---- Execute ----

func TestEnsureMasterJoinExecute(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "reconcileMasterJoin", func(_ *EnsureMasterJoin) error { return nil })
		_, err := e.Execute()
		require.NoError(t, err)
	})
	t.Run("reconcile error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "reconcileMasterJoin", func(_ *EnsureMasterJoin) error { return assertErr("reconcile failed") })
		_, err := e.Execute()
		require.Error(t, err)
	})
}

// ---- reconcileMasterJoin ----

func TestEnsureMasterJoinReconcileMasterJoin(t *testing.T) {
	t.Run("precondition error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "checkPreconditions", func(_ *EnsureMasterJoin, _ MasterJoinParams) error { return assertErr("precondition failed") })
		err := e.reconcileMasterJoin()
		require.Error(t, err)
	})

	t.Run("get joinable nodes error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "checkPreconditions", func(_ *EnsureMasterJoin, _ MasterJoinParams) error { return nil })
		patches.ApplyPrivateMethod(e, "getJoinableNodes", func(_ *EnsureMasterJoin, _ MasterJoinParams) (int, []string, error) {
			return 0, nil, assertErr("no nodes")
		})
		err := e.reconcileMasterJoin()
		require.Error(t, err)
	})

	t.Run("zero nodes sync and return", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "checkPreconditions", func(_ *EnsureMasterJoin, _ MasterJoinParams) error { return nil })
		patches.ApplyPrivateMethod(e, "getJoinableNodes", func(_ *EnsureMasterJoin, _ MasterJoinParams) (int, []string, error) {
			return 0, nil, nil
		})
		require.NoError(t, e.reconcileMasterJoin())
	})

	t.Run("bocloud cluster distributes kubeconfig then joins", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name: "c1", Namespace: "ns",
				Annotations: map[string]string{"bke.bocloud.com/cluster-from": "bocloud"},
			},
		}
		e := newMasterJoinPhaseCov(t, cluster)
		patches.ApplyPrivateMethod(e, "checkPreconditions", func(_ *EnsureMasterJoin, _ MasterJoinParams) error { return nil })
		patches.ApplyPrivateMethod(e, "getJoinableNodes", func(_ *EnsureMasterJoin, _ MasterJoinParams) (int, []string, error) {
			return 1, []string{"node1"}, nil
		})
		patches.ApplyFunc(phaseutil.DistributeKubeProxyKubeConfig, func(context.Context, client.Client, *bkev1beta1.BKECluster, bkenode.Nodes, *bkev1beta1.BKELogger) error {
			return nil
		})
		patches.ApplyPrivateMethod(e, "scaleAndJoinMasterNodes", func(_ *EnsureMasterJoin, _ MasterJoinScaleParams) error { return nil })
		require.NoError(t, e.reconcileMasterJoin())
	})

	t.Run("scale and join error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "checkPreconditions", func(_ *EnsureMasterJoin, _ MasterJoinParams) error { return nil })
		patches.ApplyPrivateMethod(e, "getJoinableNodes", func(_ *EnsureMasterJoin, _ MasterJoinParams) (int, []string, error) {
			return 1, []string{"node1"}, nil
		})
		patches.ApplyPrivateMethod(e, "scaleAndJoinMasterNodes", func(_ *EnsureMasterJoin, _ MasterJoinScaleParams) error { return assertErr("scale failed") })
		err := e.reconcileMasterJoin()
		require.Error(t, err)
	})
}

// ---- checkPreconditions ----

func TestEnsureMasterJoinCheckPreconditions(t *testing.T) {
	params := func(e *EnsureMasterJoin) MasterJoinParams {
		return MasterJoinParams{Ctx: context.Background(), Client: e.Ctx.Client, BKECluster: e.Ctx.BKECluster, Log: e.Ctx.Log}
	}

	readyMasters := bkev1beta1.BKENodes{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "master-1", Namespace: "ns"},
			Spec:       confv1beta1.BKENodeSpec{IP: "10.0.0.1", Role: []string{bkenode.MasterNodeRole}},
			Status:     confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentReadyFlag},
		},
	}

	t.Run("get bke nodes error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return nil, assertErr("fetch failed")
		})
		err := e.checkPreconditions(params(e))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get BKENodes")
	})

	t.Run("no master node", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "worker-1", Namespace: "ns"},
					Spec:       confv1beta1.BKENodeSpec{IP: "10.0.0.2", Role: []string{bkenode.WorkerNodeRole}},
					Status:     confv1beta1.BKENodeStatus{StateCode: 0},
				},
			}, nil
		})
		err := e.checkPreconditions(params(e))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no master node")
	})

	t.Run("master agent not ready", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "master-1", Namespace: "ns"},
					Spec:       confv1beta1.BKENodeSpec{IP: "10.0.0.1", Role: []string{bkenode.MasterNodeRole}},
					Status:     confv1beta1.BKENodeStatus{StateCode: 0},
				},
			}, nil
		})
		err := e.checkPreconditions(params(e))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "master agent is not ready")
		assert.Contains(t, err.Error(), "10.0.0.1")
	})

	t.Run("worker agent not ready does not block", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			// Cluster-level AgentStatus still reflects a bad worker, but must not block master join.
			Status: confv1beta1.BKEClusterStatus{AgentStatus: confv1beta1.BKEAgentStatus{UnavailableReplies: 1}},
		})
		e.Ctx.Cluster = clusterWithControlPlaneInit(t, true)
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "master-1", Namespace: "ns"},
					Spec:       confv1beta1.BKENodeSpec{IP: "10.0.0.1", Role: []string{bkenode.MasterNodeRole}},
					Status:     confv1beta1.BKENodeStatus{StateCode: bkev1beta1.NodeAgentReadyFlag},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "worker-1", Namespace: "ns"},
					Spec:       confv1beta1.BKENodeSpec{IP: "10.0.0.2", Role: []string{bkenode.WorkerNodeRole}},
					Status:     confv1beta1.BKENodeStatus{StateCode: 0, NeedSkip: true},
				},
			}, nil
		})
		require.NoError(t, e.checkPreconditions(params(e)))
	})

	t.Run("control plane not initialized returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.Ctx.Cluster = clusterWithControlPlaneInit(t, false)
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return readyMasters, nil
		})
		require.NoError(t, e.checkPreconditions(params(e)))
	})

	t.Run("ok", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.Ctx.Cluster = clusterWithControlPlaneInit(t, true)
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return readyMasters, nil
		})
		require.NoError(t, e.checkPreconditions(params(e)))
	})
}

// ---- getJoinableNodes ----

func TestEnsureMasterJoinGetJoinableNodes(t *testing.T) {
	params := func(e *EnsureMasterJoin) MasterJoinParams {
		return MasterJoinParams{Ctx: context.Background(), Client: e.Ctx.Client, BKECluster: e.Ctx.BKECluster, Log: e.Ctx.Log}
	}

	t.Run("get bke nodes error returns zero", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return nil, assertErr("fetch failed")
		})
		count, _, err := e.getJoinableNodes(params(e))
		require.NoError(t, err)
		assert.Equal(t, 0, count)
	})

	t.Run("filters already-joined nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{}, nil
		})
		patches.ApplyFunc(phaseutil.GetNeedJoinMasterNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1", Hostname: "node1"}, {IP: "10.0.0.2", Hostname: "node2"}}
		})
		patches.ApplyFunc(phaseutil.NodeToMachine, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, node confv1beta1.Node) (*clusterv1.Machine, error) {
			if node.IP == "10.0.0.1" {
				return &clusterv1.Machine{}, nil // already joined
			}
			return nil, assertErr("not found")
		})
		patches.ApplyFunc((*nodeutil.NodeFetcher).MarkNodeStateFlagForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ int) error {
			return nil
		})
		count, infos, err := e.getJoinableNodes(params(e))
		require.NoError(t, err)
		assert.Equal(t, 1, count)
		require.Len(t, infos, 1)
	})
}

// ---- waitMasterJoin ----

func TestEnsureMasterJoinWaitMasterJoin(t *testing.T) {
	t.Run("timeout", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetBootTimeOut, func(_ *bkev1beta1.BKECluster) (time.Duration, error) { return time.Minute, nil })
		patches.ApplyFunc(waitForNodesJoin, func(_ WaitForNodesJoinParams) error { return wait.ErrWaitTimeout })
		err := e.waitMasterJoin(1)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Wait master join failed")
	})

	t.Run("other error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetBootTimeOut, func(_ *bkev1beta1.BKECluster) (time.Duration, error) { return time.Minute, nil })
		patches.ApplyFunc(waitForNodesJoin, func(_ WaitForNodesJoinParams) error { return assertErr("join boom") })
		err := e.waitMasterJoin(1)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "join boom")
	})

	t.Run("success refreshes cluster", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetBootTimeOut, func(_ *bkev1beta1.BKECluster) (time.Duration, error) { return time.Minute, nil })
		patches.ApplyFunc(waitForNodesJoin, func(_ WaitForNodesJoinParams) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		require.NoError(t, e.waitMasterJoin(1))
	})
}

// ---- waitForNodesJoin (real body) ----

func TestWaitForNodesJoin(t *testing.T) {
	t.Run("all nodes joined", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.NodeToMachine, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ confv1beta1.Node) (*clusterv1.Machine, error) {
			return &clusterv1.Machine{Status: clusterv1.MachineStatus{NodeRef: &corev1.ObjectReference{Name: "node1"}}}, nil
		})
		params := WaitForNodesJoinParams{
			Ctx: context.Background(), Client: e.Ctx.Client, BKECluster: e.Ctx.BKECluster, Log: e.Ctx.Log,
			NodesToJoin:     bkenode.Nodes{{IP: "10.0.0.1", Hostname: "node1"}},
			Timeout:         context.Background(),
			SuccessJoinNode: map[int]confv1beta1.Node{},
		}
		require.NoError(t, waitForNodesJoin(params))
	})

	t.Run("node never joins -> timeout", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.NodeToMachine, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ confv1beta1.Node) (*clusterv1.Machine, error) {
			return nil, assertErr("not found")
		})
		// immediately-canceled context -> poll exits with ErrWaitTimeout
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		params := WaitForNodesJoinParams{
			Ctx: context.Background(), Client: e.Ctx.Client, BKECluster: e.Ctx.BKECluster, Log: e.Ctx.Log,
			NodesToJoin:     bkenode.Nodes{{IP: "10.0.0.1", Hostname: "node1"}},
			Timeout:         ctx,
			SuccessJoinNode: map[int]confv1beta1.Node{},
		}
		err := waitForNodesJoin(params)
		require.Error(t, err)
		assert.True(t, err == wait.ErrWaitTimeout)
	})
}

// ---- NeedExecute (covers all branches of NeedExecute) ----

// masterJoinGapsPatchRefreshCtxCluster patches PhaseContext.RefreshCtxCluster
// so NeedExecute does not hit the API server. Returns the configured error.
func masterJoinGapsPatchRefreshCtxCluster(patches *gomonkey.Patches, err error) {
	patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster",
		func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return err })
}

// masterJoinGapsPatchGetBKENodesWrapper patches the deep
// (*nodeutil.NodeFetcher).GetBKENodesWrapper so the short wrapper
// GetBKENodesWrapperForCluster returns the configured values.
func masterJoinGapsPatchGetBKENodesWrapper(patches *gomonkey.Patches, nodes bkev1beta1.BKENodes, err error) {
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return nodes, err
		})
}

func TestEnsureMasterJoinNeedExecute(t *testing.T) {
	baseCluster := func() *bkev1beta1.BKECluster {
		return &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	}

	t.Run("default need execute false returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		// DeletionTimestamp set -> checkCommonNeedExecute returns false -> DefaultNeedExecute false
		now := metav1.Now()
		bc := baseCluster()
		bc.DeletionTimestamp = &now
		e := newMasterJoinPhaseCov(t, bc)
		assert.False(t, e.NeedExecute(bc, bc))
	})

	t.Run("get bke nodes error returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		bc := baseCluster()
		e := newMasterJoinPhaseCov(t, bc)
		e.Ctx.Cluster = clusterWithControlPlaneInit(t, true)
		masterJoinGapsPatchRefreshCtxCluster(patches, nil)
		masterJoinGapsPatchGetBKENodesWrapper(patches, nil, assertErr("fetch failed"))
		assert.False(t, e.NeedExecute(bc, bc))
	})

	t.Run("first create one node not inited returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		bc := baseCluster()
		e := newMasterJoinPhaseCov(t, bc)
		e.Ctx.Cluster = clusterWithControlPlaneInit(t, false)
		masterJoinGapsPatchRefreshCtxCluster(patches, nil)
		masterJoinGapsPatchGetBKENodesWrapper(patches, bkev1beta1.BKENodes{}, nil)
		patches.ApplyFunc(phaseutil.GetNeedJoinMasterNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}}
			})
		assert.False(t, e.NeedExecute(bc, bc))
	})

	t.Run("master inited no nodes returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		bc := baseCluster()
		e := newMasterJoinPhaseCov(t, bc)
		e.Ctx.Cluster = clusterWithControlPlaneInit(t, true)
		masterJoinGapsPatchRefreshCtxCluster(patches, nil)
		masterJoinGapsPatchGetBKENodesWrapper(patches, bkev1beta1.BKENodes{}, nil)
		patches.ApplyFunc(phaseutil.GetNeedJoinMasterNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return nil
			})
		assert.False(t, e.NeedExecute(bc, bc))
	})

	t.Run("not inited no nodes returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		bc := baseCluster()
		e := newMasterJoinPhaseCov(t, bc)
		e.Ctx.Cluster = clusterWithControlPlaneInit(t, false)
		masterJoinGapsPatchRefreshCtxCluster(patches, nil)
		masterJoinGapsPatchGetBKENodesWrapper(patches, bkev1beta1.BKENodes{}, nil)
		patches.ApplyFunc(phaseutil.GetNeedJoinMasterNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return nil
			})
		assert.False(t, e.NeedExecute(bc, bc))
	})

	t.Run("refresh cluster error leaves master not inited returns false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		bc := baseCluster()
		e := newMasterJoinPhaseCov(t, bc)
		// Cluster stays nil-ish; RefreshCtxCluster errors so the conditions branch is skipped
		e.Ctx.Cluster = clusterWithControlPlaneInit(t, true)
		masterJoinGapsPatchRefreshCtxCluster(patches, assertErr("refresh boom"))
		masterJoinGapsPatchGetBKENodesWrapper(patches, bkev1beta1.BKENodes{}, nil)
		patches.ApplyFunc(phaseutil.GetNeedJoinMasterNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}, {IP: "10.0.0.2"}}
			})
		// masterInited=false, nodes=2 -> falls through all three early returns -> true
		assert.True(t, e.NeedExecute(bc, bc))
		assert.Equal(t, bkev1beta1.PhaseWaiting, e.GetStatus())
	})

	t.Run("inited with nodes returns true and sets waiting", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		bc := baseCluster()
		e := newMasterJoinPhaseCov(t, bc)
		e.Ctx.Cluster = clusterWithControlPlaneInit(t, true)
		masterJoinGapsPatchRefreshCtxCluster(patches, nil)
		masterJoinGapsPatchGetBKENodesWrapper(patches, bkev1beta1.BKENodes{}, nil)
		patches.ApplyFunc(phaseutil.GetNeedJoinMasterNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}, {IP: "10.0.0.2"}}
			})
		assert.True(t, e.NeedExecute(bc, bc))
		assert.Equal(t, bkev1beta1.PhaseWaiting, e.GetStatus())
	})
}

// ---- scaleAndJoinMasterNodes (covers the 0% function) ----

// masterJoinGapsPatchGetNodes patches the deep (*nodeutil.NodeFetcher).GetNodes
// so the short wrapper GetNodesForBKECluster returns the configured nodes.
func masterJoinGapsPatchGetNodes(patches *gomonkey.Patches, nodes bkenode.Nodes) {
	patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: nodes}, nil
		})
}

func masterJoinGapsScaleParams(e *EnsureMasterJoin, nodesCount int) MasterJoinScaleParams {
	return MasterJoinScaleParams{
		Ctx:        context.Background(),
		Client:     e.Ctx.Client,
		BKECluster: e.Ctx.BKECluster,
		Log:        e.Ctx.Log,
		NodesCount: nodesCount,
	}
}

func TestEnsureMasterJoinScaleAndJoinMasterNodes(t *testing.T) {
	t.Run("get associate objs error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return nil, assertErr("capi failed")
			})
		err := e.scaleAndJoinMasterNodes(masterJoinGapsScaleParams(e, 1))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "capi failed")
	})

	t.Run("kubeadm control plane nil returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{}, nil
			})
		// upstream change: KCP nil with err nil now returns an error.
		err := e.scaleAndJoinMasterNodes(masterJoinGapsScaleParams(e, 1))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "associate objs failed")
	})

	t.Run("scale up resume error returns error and rollback", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		kcp := &controlv1beta1.KubeadmControlPlane{Spec: controlv1beta1.KubeadmControlPlaneSpec{Replicas: int32p(1)}}
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{KubeadmControlPlane: kcp}, nil
			})
		// Provide master nodes so GetNodes path works
		masterJoinGapsPatchGetNodes(patches, bkenode.Nodes{
			{IP: "10.0.0.1", Role: []string{"master"}},
			{IP: "10.0.0.2", Role: []string{"master"}},
			{IP: "10.0.0.3", Role: []string{"master"}},
		})
		patches.ApplyFunc(phaseutil.UpdateKubeadmControlPlaneReplicas,
			func(_ context.Context, _ client.Client, kcp *controlv1beta1.KubeadmControlPlane, replicas int32) error {
				kcp.Spec.Replicas = &replicas
				return nil
			})
		resumeCalls := 0
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error {
				resumeCalls++
				return assertErr("resume failed")
			})
		err := e.scaleAndJoinMasterNodes(masterJoinGapsScaleParams(e, 1))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "resume failed")
		// Resume called once for scale up (defer runs and also fails -> defer error path covered)
		assert.Equal(t, 2, resumeCalls)
		// Replicas should have been rolled back to original (1) by the defer
		assert.Equal(t, int32(1), *kcp.Spec.Replicas)
	})

	t.Run("wait master join error returns error and rollback ok", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		kcp := &controlv1beta1.KubeadmControlPlane{Spec: controlv1beta1.KubeadmControlPlaneSpec{Replicas: int32p(1)}}
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{KubeadmControlPlane: kcp}, nil
			})
		masterJoinGapsPatchGetNodes(patches, bkenode.Nodes{
			{IP: "10.0.0.1", Role: []string{"master"}},
			{IP: "10.0.0.2", Role: []string{"master"}},
			{IP: "10.0.0.3", Role: []string{"master"}},
		})
		patches.ApplyFunc(phaseutil.UpdateKubeadmControlPlaneReplicas,
			func(_ context.Context, _ client.Client, kcp *controlv1beta1.KubeadmControlPlane, replicas int32) error {
				kcp.Spec.Replicas = &replicas
				return nil
			})
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		patches.ApplyPrivateMethod(e, "waitMasterJoin", func(_ *EnsureMasterJoin, _ int) error {
			return assertErr("wait join failed")
		})
		err := e.scaleAndJoinMasterNodes(masterJoinGapsScaleParams(e, 1))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait join failed")
		// Replicas rolled back to original
		assert.Equal(t, int32(1), *kcp.Spec.Replicas)
	})

	t.Run("success scales up and waits", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		kcp := &controlv1beta1.KubeadmControlPlane{Spec: controlv1beta1.KubeadmControlPlaneSpec{Replicas: int32p(1)}}
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{KubeadmControlPlane: kcp}, nil
			})
		masterJoinGapsPatchGetNodes(patches, bkenode.Nodes{
			{IP: "10.0.0.1", Role: []string{"master"}},
			{IP: "10.0.0.2", Role: []string{"master"}},
			{IP: "10.0.0.3", Role: []string{"master"}},
		})
		patches.ApplyFunc(phaseutil.UpdateKubeadmControlPlaneReplicas,
			func(_ context.Context, _ client.Client, kcp *controlv1beta1.KubeadmControlPlane, replicas int32) error {
				kcp.Spec.Replicas = &replicas
				return nil
			})
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		patches.ApplyPrivateMethod(e, "waitMasterJoin", func(_ *EnsureMasterJoin, _ int) error { return nil })
		require.NoError(t, e.scaleAndJoinMasterNodes(masterJoinGapsScaleParams(e, 1)))
		// Replicas scaled up: 1 + 1 = 2
		assert.Equal(t, int32(2), *kcp.Spec.Replicas)
	})

	t.Run("except replicas floored to master node count", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		// currentReplicas=3, nodesCount=2 -> exceptReplicas=5, but masterNodes.Length()=3 -> floored to 3
		kcp := &controlv1beta1.KubeadmControlPlane{Spec: controlv1beta1.KubeadmControlPlaneSpec{Replicas: int32p(3)}}
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{KubeadmControlPlane: kcp}, nil
			})
		masterJoinGapsPatchGetNodes(patches, bkenode.Nodes{
			{IP: "10.0.0.1", Role: []string{"master"}},
			{IP: "10.0.0.2", Role: []string{"master"}},
			{IP: "10.0.0.3", Role: []string{"master"}},
		})
		patches.ApplyFunc(phaseutil.UpdateKubeadmControlPlaneReplicas,
			func(_ context.Context, _ client.Client, kcp *controlv1beta1.KubeadmControlPlane, replicas int32) error {
				kcp.Spec.Replicas = &replicas
				return nil
			})
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		patches.ApplyPrivateMethod(e, "waitMasterJoin", func(_ *EnsureMasterJoin, _ int) error { return nil })
		require.NoError(t, e.scaleAndJoinMasterNodes(masterJoinGapsScaleParams(e, 2)))
		// Floored to 3 (master node count)
		assert.Equal(t, int32(3), *kcp.Spec.Replicas)
	})
}

// ---- reconcileMasterJoin: SyncStatusUntilComplete error branch ----

func TestEnsureMasterJoinReconcileMasterJoinSyncError(t *testing.T) {
	t.Run("zero nodes sync status error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return assertErr("sync failed")
			})
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "checkPreconditions", func(_ *EnsureMasterJoin, _ MasterJoinParams) error { return nil })
		patches.ApplyPrivateMethod(e, "getJoinableNodes", func(_ *EnsureMasterJoin, _ MasterJoinParams) (int, []string, error) {
			return 0, nil, nil
		})
		err := e.reconcileMasterJoin()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sync failed")
	})
}

// ---- waitMasterJoin: GetBootTimeOut error branch ----

func TestEnsureMasterJoinWaitMasterJoinBootTimeoutError(t *testing.T) {
	t.Run("get boot timeout error still waits", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetBootTimeOut, func(_ *bkev1beta1.BKECluster) (time.Duration, error) {
			return 0, assertErr("boot timeout error")
		})
		patches.ApplyFunc(waitForNodesJoin, func(_ WaitForNodesJoinParams) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster",
			func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		require.NoError(t, e.waitMasterJoin(1))
	})

	t.Run("get boot timeout error then wait error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetBootTimeOut, func(_ *bkev1beta1.BKECluster) (time.Duration, error) {
			return 0, assertErr("boot timeout error")
		})
		patches.ApplyFunc(waitForNodesJoin, func(_ WaitForNodesJoinParams) error {
			return wait.ErrWaitTimeout
		})
		err := e.waitMasterJoin(1)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Wait master join failed")
	})
}

// ---- waitForNodesJoin: LogOutputInterval branch (pollCount%LogOutputInterval==0) ----

func TestWaitForNodesJoinLogOutputInterval(t *testing.T) {
	t.Run("logs at interval when waiting for partial join", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		// NodeToMachine always returns error -> node never joins -> poll loops until LogOutputInterval hits
		patches.ApplyFunc(phaseutil.NodeToMachine, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ confv1beta1.Node) (*clusterv1.Machine, error) {
			return nil, assertErr("not found")
		})
		// Use a context that cancels after LogOutputInterval+2 seconds to let pollCount reach 10
		ctx, cancel := context.WithTimeout(context.Background(), (LogOutputInterval+2)*time.Second)
		defer cancel()
		params := WaitForNodesJoinParams{
			Ctx:         context.Background(),
			Client:      e.Ctx.Client,
			BKECluster:  e.Ctx.BKECluster,
			NodesToJoin: bkenode.Nodes{{IP: "10.0.0.1", Hostname: "node1"}},
			Log:         e.Ctx.Log,
			Timeout:     ctx,
			// Pre-populate one success but not all -> len(SuccessJoinNode) != len(NodesToJoin) is true
			// Actually we have 1 node and it's not pre-populated, so 0 != 1 is true -> enters the log branch
			SuccessJoinNode: map[int]confv1beta1.Node{},
		}
		err := waitForNodesJoin(params)
		require.Error(t, err)
		// Should be wait.ErrWaitTimeout since context timed out
		assert.True(t, err == wait.ErrWaitTimeout)
	})
}
