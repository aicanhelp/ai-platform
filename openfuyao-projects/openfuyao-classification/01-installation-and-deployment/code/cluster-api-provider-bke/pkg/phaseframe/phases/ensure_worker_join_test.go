/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package phases

import (
	"context"
	"sync"
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
	"sigs.k8s.io/cluster-api/util/conditions"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureWorkerJoinConstants(t *testing.T) {
	assert.Equal(t, "EnsureWorkerJoin", string(EnsureWorkerJoinName))
}

func TestNewEnsureWorkerJoin(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

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

	phase := NewEnsureWorkerJoin(ctx)
	assert.NotNil(t, phase)

	workerJoin, ok := phase.(*EnsureWorkerJoin)
	assert.True(t, ok)
	assert.NotNil(t, workerJoin)
}

func TestEnsureWorkerJoin_CategorizeJoinedNodes(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)
	phase.nodesToJoin = bkenode.Nodes{
		{IP: "192.168.1.1", Hostname: "node1"},
		{IP: "192.168.1.2", Hostname: "node2"},
		{IP: "192.168.1.3", Hostname: "node3"},
	}

	successMap := &sync.Map{}
	successMap.Store(0, phase.nodesToJoin[0])
	successMap.Store(2, phase.nodesToJoin[2])

	successNodes, failedNodes := phase.categorizeJoinedNodes(successMap)
	assert.Equal(t, 2, len(successNodes))
	assert.Equal(t, 1, len(failedNodes))
	assert.Equal(t, "192.168.1.2", failedNodes[0].IP)
}

func TestEnsureWorkerJoin_IsAllNodesProcessed(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)
	phase.nodesToJoin = bkenode.Nodes{
		{IP: "192.168.1.1", Hostname: "node1"},
		{IP: "192.168.1.2", Hostname: "node2"},
	}

	successMap := &sync.Map{}
	failedMap := &sync.Map{}

	// Not all processed
	done, success, failed := phase.isAllNodesProcessed(successMap, failedMap)
	assert.False(t, done)
	assert.Equal(t, 0, success)
	assert.Equal(t, 0, failed)

	// All processed
	successMap.Store(0, phase.nodesToJoin[0])
	failedMap.Store(1, phase.nodesToJoin[1])

	done, success, failed = phase.isAllNodesProcessed(successMap, failedMap)
	assert.True(t, done)
	assert.Equal(t, 1, success)
	assert.Equal(t, 1, failed)
}

func TestEnsureWorkerJoin_LogProgressIfNeeded(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)
	phase.nodesToJoin = bkenode.Nodes{
		{IP: "192.168.1.1", Hostname: "node1"},
	}

	successMap := &sync.Map{}
	failedMap := &sync.Map{}

	// Should not log (pollCount not multiple of 10)
	phase.logProgressIfNeeded(5, successMap, failedMap, ctx.Log)

	// Should log (pollCount is multiple of 10)
	phase.logProgressIfNeeded(10, successMap, failedMap, ctx.Log)
}

func TestEnsureWorkerJoin_LogFailedNodesSummary(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)
	phase.nodesToJoin = bkenode.Nodes{
		{IP: "192.168.1.1", Hostname: "node1"},
		{IP: "192.168.1.2", Hostname: "node2"},
	}

	successNodes := bkenode.Nodes{{IP: "192.168.1.1", Hostname: "node1"}}
	failedNodes := bkenode.Nodes{{IP: "192.168.1.2", Hostname: "node2"}}

	phase.logFailedNodesSummary(ctx.Log, successNodes, failedNodes)
}

func TestEnsureWorkerJoin_LogFailedNodesGuidance(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)
	phase.logFailedNodesGuidance(ctx.Log)
}

func TestEnsureWorkerJoin_LogSuccessResult(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)
	phase.nodesToJoin = bkenode.Nodes{
		{IP: "192.168.1.1", Hostname: "node1"},
		{IP: "192.168.1.2", Hostname: "node2"},
	}

	successNodes := bkenode.Nodes{{IP: "192.168.1.1", Hostname: "node1"}}
	failedNodes := bkenode.Nodes{{IP: "192.168.1.2", Hostname: "node2"}}

	phase.logSuccessResult(ctx.Log, successNodes, failedNodes)
}

func TestEnsureWorkerJoin_LogTimeoutResult(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)
	failedNodes := bkenode.Nodes{{IP: "192.168.1.1", Hostname: "node1"}}

	phase.logTimeoutResult(ctx.Log, failedNodes)
}

func TestEnsureWorkerJoin_HandleBocloudClusterConfig_NotBocloudCluster(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)

	params := HandleBocloudClusterConfigParams{
		Ctx:        context.Background(),
		Client:     c,
		BKECluster: bkeCluster,
		Log:        ctx.Log,
	}

	err := phase.handleBocloudClusterConfig(params)
	assert.NoError(t, err)
}

func TestEnsureWorkerJoin_DetermineDeploymentResult_AllSuccess(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)
	phase.nodesToJoin = bkenode.Nodes{
		{IP: "192.168.1.1", Hostname: "node1"},
	}

	successNodes := bkenode.Nodes{{IP: "192.168.1.1", Hostname: "node1"}}
	failedNodes := bkenode.Nodes{}

	err := phase.determineDeploymentResult(successNodes, failedNodes, nil)
	assert.NoError(t, err)
}

func TestEnsureWorkerJoin_DetermineDeploymentResult_SomeSuccess(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)
	phase.nodesToJoin = bkenode.Nodes{
		{IP: "192.168.1.1", Hostname: "node1"},
		{IP: "192.168.1.2", Hostname: "node2"},
	}

	successNodes := bkenode.Nodes{{IP: "192.168.1.1", Hostname: "node1"}}
	failedNodes := bkenode.Nodes{{IP: "192.168.1.2", Hostname: "node2"}}

	err := phase.determineDeploymentResult(successNodes, failedNodes, nil)
	assert.NoError(t, err)
}

func TestEnsureWorkerJoin_DetermineDeploymentResult_AllFailed(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)
	phase.nodesToJoin = bkenode.Nodes{
		{IP: "192.168.1.1", Hostname: "node1"},
	}

	successNodes := bkenode.Nodes{}
	failedNodes := bkenode.Nodes{{IP: "192.168.1.1", Hostname: "node1"}}

	err := phase.determineDeploymentResult(successNodes, failedNodes, assert.AnError)
	assert.Error(t, err)
}

func TestEnsureWorkerJoin_WaitWorkerJoin_EmptyNodes(t *testing.T) {
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

	phase := NewEnsureWorkerJoin(ctx).(*EnsureWorkerJoin)
	phase.nodesToJoin = nil

	err := phase.waitWorkerJoin()
	assert.NoError(t, err)
}

func newWorkerJoinPhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureWorkerJoin {
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
	return &EnsureWorkerJoin{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

func int32p(i int32) *int32 { return &i }

// ---- Execute ----

func TestEnsureWorkerJoinExecute(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	patches.ApplyPrivateMethod(e, "reconcileWorkerJoin", func(_ *EnsureWorkerJoin) error { return assertErr("reconcile failed") })
	_, err := e.Execute()
	require.Error(t, err)
}

// ---- reconcileWorkerJoin ----

func TestEnsureWorkerJoinReconcileWorkerJoin(t *testing.T) {
	t.Run("control plane not initialized returns nil", func(t *testing.T) {
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.Ctx.Cluster = &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		conditions.Set(e.Ctx.Cluster, &clusterv1.Condition{Type: clusterv1.ControlPlaneInitializedCondition, Status: corev1.ConditionFalse})
		require.NoError(t, e.reconcileWorkerJoin())
	})

	t.Run("no except join nodes returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.Ctx.Cluster = &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		conditions.Set(e.Ctx.Cluster, &clusterv1.Condition{Type: clusterv1.ControlPlaneInitializedCondition, Status: corev1.ConditionTrue})
		patches.ApplyPrivateMethod(e, "getExceptJoinNodes", func(_ *EnsureWorkerJoin) bkenode.Nodes { return nil })
		require.NoError(t, e.reconcileWorkerJoin())
	})

	t.Run("get joinable nodes error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.Ctx.Cluster = &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		conditions.Set(e.Ctx.Cluster, &clusterv1.Condition{Type: clusterv1.ControlPlaneInitializedCondition, Status: corev1.ConditionTrue})
		patches.ApplyPrivateMethod(e, "getExceptJoinNodes", func(_ *EnsureWorkerJoin) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1", Hostname: "n1"}}
		})
		patches.ApplyPrivateMethod(e, "getJoinableNodesInfo", func(_ *EnsureWorkerJoin, _ bkenode.Nodes) ([]string, int, error) {
			return nil, 0, assertErr("no joinable")
		})
		err := e.reconcileWorkerJoin()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no joinable")
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.Ctx.Cluster = &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		conditions.Set(e.Ctx.Cluster, &clusterv1.Condition{Type: clusterv1.ControlPlaneInitializedCondition, Status: corev1.ConditionTrue})
		patches.ApplyPrivateMethod(e, "getExceptJoinNodes", func(_ *EnsureWorkerJoin) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1", Hostname: "n1"}}
		})
		patches.ApplyPrivateMethod(e, "getJoinableNodesInfo", func(_ *EnsureWorkerJoin, _ bkenode.Nodes) ([]string, int, error) {
			return []string{"n1"}, 1, nil
		})
		patches.ApplyPrivateMethod(e, "handleBocloudClusterConfig", func(_ *EnsureWorkerJoin, _ HandleBocloudClusterConfigParams) error { return nil })
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs, func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
			return &phaseutil.ClusterAPIObjs{MachineDeployment: &clusterv1.MachineDeployment{Spec: clusterv1.MachineDeploymentSpec{Replicas: int32p(1)}}}, nil
		})
		patches.ApplyPrivateMethod(e, "scaleMachineDeployment", func(_ *EnsureWorkerJoin, _ ScaleMachineDeploymentParams) error { return nil })
		require.NoError(t, e.reconcileWorkerJoin())
	})
}

// ---- getExceptJoinNodes (real body via seams) ----

func TestEnsureWorkerJoinGetExceptJoinNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
		return bkev1beta1.BKENodes{}, nil
	})
	patches.ApplyFunc(phaseutil.GetNeedJoinWorkerNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
		return bkenode.Nodes{{IP: "10.0.0.1"}, {IP: "10.0.0.2"}, {IP: "10.0.0.3"}}
	})
	// node1: skip; node2: env/ready ok; node3: not ready
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateNeedSkip, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, ip string) (bool, error) {
		return ip == "10.0.0.1", nil
	})
	// GetNodeStateFlagForCluster is an inlined wrapper -> patch the deepest GetNodeStateFlag.
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlag, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, ip string, flag int) (bool, error) {
		if ip == "10.0.0.3" && flag == bkev1beta1.NodeAgentReadyFlag { // not ready
			return false, nil
		}
		return true, nil
	})
	nodes := e.getExceptJoinNodes()
	require.Len(t, nodes, 1)
	assert.Equal(t, "10.0.0.2", nodes[0].IP)
}

// ---- getJoinableNodesInfo (real body) ----

func TestEnsureWorkerJoinGetJoinableNodesInfo(t *testing.T) {
	t.Run("filters already-joined and syncs when zero", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.NodeToMachine, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ confv1beta1.Node) (*clusterv1.Machine, error) {
			return &clusterv1.Machine{}, nil // already joined
		})
		patches.ApplyFunc((*nodeutil.NodeFetcher).MarkNodeStateFlagForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ int) error {
			return nil
		})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		infos, count, err := e.getJoinableNodesInfo(bkenode.Nodes{{IP: "10.0.0.1", Hostname: "n1"}})
		require.NoError(t, err)
		assert.Equal(t, 0, count)
		assert.Nil(t, infos)
	})

	t.Run("returns joinable nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.NodeToMachine, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ confv1beta1.Node) (*clusterv1.Machine, error) {
			return nil, assertErr("not found")
		})
		infos, count, err := e.getJoinableNodesInfo(bkenode.Nodes{{IP: "10.0.0.1", Hostname: "n1"}})
		require.NoError(t, err)
		assert.Equal(t, 1, count)
		require.Len(t, infos, 1)
	})
}

// ---- scaleMachineDeployment (real body) ----

func TestEnsureWorkerJoinScaleMachineDeployment(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"worker"}}, {IP: "10.0.0.2", Role: []string{"worker"}}}}, nil
		})
		patches.ApplyFunc(phaseutil.UpdateMachineDeploymentReplicas,
			func(_ context.Context, _ client.Client, md *clusterv1.MachineDeployment, replicas int32) error {
				md.Spec.Replicas = &replicas
				return nil
			})
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj, func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		patches.ApplyPrivateMethod(e, "waitWorkerJoin", func(_ *EnsureWorkerJoin) error { return nil })
		params := ScaleMachineDeploymentParams{
			Ctx: context.Background(), Client: e.Ctx.Client, BKECluster: e.Ctx.BKECluster,
			Scope:      &phaseutil.ClusterAPIObjs{MachineDeployment: &clusterv1.MachineDeployment{Spec: clusterv1.MachineDeploymentSpec{Replicas: int32p(1)}}},
			NodesCount: 1,
		}
		require.NoError(t, e.scaleMachineDeployment(params))
	})

	t.Run("wait error rolls back", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"worker"}}}}, nil
		})
		patches.ApplyFunc(phaseutil.UpdateMachineDeploymentReplicas,
			func(_ context.Context, _ client.Client, md *clusterv1.MachineDeployment, replicas int32) error {
				md.Spec.Replicas = &replicas
				return nil
			})
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj, func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		patches.ApplyPrivateMethod(e, "waitWorkerJoin", func(_ *EnsureWorkerJoin) error { return assertErr("wait failed") })
		params := ScaleMachineDeploymentParams{
			Ctx: context.Background(), Client: e.Ctx.Client, BKECluster: e.Ctx.BKECluster,
			Scope:      &phaseutil.ClusterAPIObjs{MachineDeployment: &clusterv1.MachineDeployment{Spec: clusterv1.MachineDeploymentSpec{Replicas: int32p(1)}}},
			NodesCount: 1,
		}
		err := e.scaleMachineDeployment(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait failed")
	})
}

// ---- waitWorkerJoin ----

func TestEnsureWorkerJoinWaitWorkerJoin(t *testing.T) {
	t.Run("empty nodes returns nil", func(t *testing.T) {
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		require.NoError(t, e.waitWorkerJoin())
	})

	t.Run("poll error then success nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.nodesToJoin = bkenode.Nodes{{IP: "10.0.0.1", Hostname: "n1"}}
		patches.ApplyFunc(phaseutil.GetBootTimeOut, func(_ *bkev1beta1.BKECluster) (time.Duration, error) { return time.Minute, nil })
		patches.ApplyPrivateMethod(e, "pollWorkerJoinStatus", func(_ *EnsureWorkerJoin, _ context.Context) (*sync.Map, error) {
			return &sync.Map{}, assertErr("poll failed")
		})
		patches.ApplyPrivateMethod(e, "updateSuccessNodesStatus", func(_ *EnsureWorkerJoin, _ client.Client, _ bkenode.Nodes) error { return nil })
		patches.ApplyPrivateMethod(e, "handleFailedNodes", func(_ *EnsureWorkerJoin, _ client.Client, _, _ bkenode.Nodes) {})
		// all nodes failed -> determineDeploymentResult returns pollErr (not timeout)
		err := e.waitWorkerJoin()
		require.Error(t, err)
	})
}

// ---- updateSuccessNodesStatus ----

func TestEnsureWorkerJoinUpdateSuccessNodesStatus(t *testing.T) {
	t.Run("empty returns nil", func(t *testing.T) {
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		require.NoError(t, e.updateSuccessNodesStatus(e.Ctx.Client, bkenode.Nodes{}))
	})

	t.Run("with success nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		patches.ApplyFunc((*nodeutil.NodeFetcher).SetNodeStateWithMessageForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ confv1beta1.NodeState, _ string) error {
			return nil
		})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		require.NoError(t, e.updateSuccessNodesStatus(e.Ctx.Client, bkenode.Nodes{{IP: "10.0.0.1"}}))
	})
}

// ---- handleFailedNodes (real body) ----

func TestEnsureWorkerJoinHandleFailedNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	e.nodesToJoin = bkenode.Nodes{{IP: "10.0.0.1"}}
	patches.ApplyFunc((*nodeutil.NodeFetcher).SetNodeNeedSkip, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ bool) error { return nil })
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeByIP, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string) (*confv1beta1.BKENode, error) {
		return &confv1beta1.BKENode{Status: confv1beta1.BKENodeStatus{State: "Failed"}}, nil
	})
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
	assert.NotPanics(t, func() {
		e.handleFailedNodes(e.Ctx.Client, bkenode.Nodes{}, bkenode.Nodes{{IP: "10.0.0.1"}})
	})
}

// ---- pollWorkerJoinStatus (timeout via canceled ctx) ----

func TestEnsureWorkerJoinPollWorkerJoinStatus(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	e.nodesToJoin = bkenode.Nodes{{IP: "10.0.0.1", Hostname: "n1"}}
	patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
	patches.ApplyPrivateMethod(e, "checkAllNodesStatus", func(_ *EnsureWorkerJoin, _ *sync.Map, _ *sync.Map) {})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := e.pollWorkerJoinStatus(ctx)
	require.Error(t, err)
	assert.True(t, err == wait.ErrWaitTimeout)
}

// ---- checkSingleNodeStatus (real body via seams) ----

func TestEnsureWorkerJoinCheckSingleNodeStatus(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newWorkerJoinPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlag, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, ip string, flag int) (bool, error) {
		return ip == "10.0.0.1" && flag == bkev1beta1.NodeFailedFlag, nil
	})
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeByIP, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, ip string) (*confv1beta1.BKENode, error) {
		state := confv1beta1.NodeState("")
		if ip == "10.0.0.2" {
			state = bkev1beta1.NodeBootStrapFailed
		}
		return &confv1beta1.BKENode{Status: confv1beta1.BKENodeStatus{State: state}}, nil
	})
	patches.ApplyFunc((*nodeutil.NodeFetcher).SetNodeNeedSkip, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ bool) error { return nil })
	patches.ApplyFunc(phaseutil.NodeToMachine, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ confv1beta1.Node) (*clusterv1.Machine, error) {
		return &clusterv1.Machine{Status: clusterv1.MachineStatus{NodeRef: &corev1.ObjectReference{Name: "n3"}}}, nil
	})

	t.Run("failed flag marks node failed", func(t *testing.T) {
		success, failed := &sync.Map{}, &sync.Map{}
		e.checkSingleNodeStatus(0, confv1beta1.Node{IP: "10.0.0.1"}, success, failed)
		_, ok := failed.Load(0)
		assert.True(t, ok)
	})

	t.Run("bootstrap failed state marks node failed", func(t *testing.T) {
		success, failed := &sync.Map{}, &sync.Map{}
		e.checkSingleNodeStatus(1, confv1beta1.Node{IP: "10.0.0.2"}, success, failed)
		_, ok := failed.Load(1)
		assert.True(t, ok)
	})

	t.Run("machine with noderef marks success", func(t *testing.T) {
		success, failed := &sync.Map{}, &sync.Map{}
		e.checkSingleNodeStatus(2, confv1beta1.Node{IP: "10.0.0.3"}, success, failed)
		_, ok := success.Load(2)
		assert.True(t, ok)
	})

	t.Run("already processed skips", func(t *testing.T) {
		success, failed := &sync.Map{}, &sync.Map{}
		success.Store(0, confv1beta1.Node{IP: "10.0.0.1"})
		e.checkSingleNodeStatus(0, confv1beta1.Node{IP: "10.0.0.1"}, success, failed)
		// no panic, remains processed
		_, ok := success.Load(0)
		assert.True(t, ok)
	})
}
