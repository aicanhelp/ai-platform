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
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	controlv1beta1 "sigs.k8s.io/cluster-api/controlplane/kubeadm/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func createTestEnsureMasterDelete() *EnsureMasterDelete {
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
	_ = clusterv1.AddToScheme(scheme)

	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: bkeCluster,
		Scheme:     scheme,
		Log:        logger,
	}

	return &EnsureMasterDelete{
		BasePhase:                    phaseframe.NewBasePhase(ctx, EnsureMasterDeleteName),
		machinesAndNodesToDelete:     make(map[string]phaseutil.MachineAndNode),
		machinesAndNodesToWaitDelete: make(map[string]phaseutil.MachineAndNode),
	}
}

func TestEnsureMasterDeleteConstants(t *testing.T) {
	assert.Equal(t, "EnsureMasterDelete", string(EnsureMasterDeleteName))
	assert.Equal(t, 4, WaitMasterDeleteTimeoutMinutes)
	assert.Equal(t, 2, WaitMasterDeletePollIntervalSeconds)
}

func TestNewEnsureMasterDelete(t *testing.T) {
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

	phase := NewEnsureMasterDelete(ctx)
	assert.NotNil(t, phase)
	assert.IsType(t, &EnsureMasterDelete{}, phase)

	emd := phase.(*EnsureMasterDelete)
	assert.NotNil(t, emd.machinesAndNodesToDelete)
	assert.NotNil(t, emd.machinesAndNodesToWaitDelete)
}

func TestEnsureMasterDelete_NeedExecute_DefaultNeedExecuteFalse(t *testing.T) {
	e := createTestEnsureMasterDelete()
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

func TestEnsureMasterDelete_NeedExecute_WithDeleteNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureMasterDelete()

	patches.ApplyFunc(phaseutil.GetNeedDeleteMasterNodes, func(ctx context.Context, c client.Client, cluster *bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{{IP: "192.168.1.1"}}
	})

	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{}

	result := e.NeedExecute(old, new)
	assert.True(t, result)
}

func TestEnsureMasterDelete_NeedExecute_NoDeleteNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureMasterDelete()

	patches.ApplyFunc(phaseutil.GetNeedDeleteMasterNodes, func(ctx context.Context, c client.Client, cluster *bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{}
	})

	patches.ApplyFunc(getDeleteTargetNodesIfDeployed, func(ctx *phaseframe.PhaseContext, cluster *bkev1beta1.BKECluster) (bkenode.Nodes, bool) {
		return nil, false
	})

	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{}

	result := e.NeedExecute(old, new)
	assert.False(t, result)
}

func TestEnsureMasterDelete_GetTargetClusterNodes_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureMasterDelete()

	patches.ApplyFunc(GetTargetClusterNodes, func(ctx context.Context, c client.Client, cluster *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
		return bkenode.Nodes{{IP: "192.168.1.1"}}, nil
	})

	nodes, err := e.getTargetClusterNodes(e.Ctx.BKECluster)
	assert.NoError(t, err)
	assert.Len(t, nodes, 1)
}

func TestEnsureMasterDelete_PrepareMachinesAndNodesToWaitDelete_Empty(t *testing.T) {
	e := createTestEnsureMasterDelete()
	result := e.prepareMachinesAndNodesToWaitDelete()
	assert.NotNil(t, result)
	assert.Empty(t, result)
}

func TestEnsureMasterDelete_PrepareMachinesAndNodesToWaitDelete_WithData(t *testing.T) {
	e := createTestEnsureMasterDelete()
	machine := &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}}
	node := confv1beta1.Node{IP: "192.168.1.1"}

	e.machinesAndNodesToDelete = map[string]phaseutil.MachineAndNode{
		"machine1": {Machine: machine, Node: node},
	}
	e.machinesAndNodesToWaitDelete = map[string]phaseutil.MachineAndNode{
		"machine2": {Machine: &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "machine2"}}, Node: confv1beta1.Node{IP: "192.168.1.2"}},
	}

	result := e.prepareMachinesAndNodesToWaitDelete()
	assert.Len(t, result, 2)
}

func TestEnsureMasterDelete_WaitForMachinesDelete_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureMasterDelete()
	machine := &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Namespace: "default"}}

	patches.ApplyMethod(&fakeClient{}, "Get", func(_ *fakeClient, ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
		return apierrors.NewNotFound(schema.GroupResource{}, "machine1")
	})

	params := WaitForMachinesDeleteParams{
		Ctx:    context.Background(),
		Client: &fakeClient{},
		MachinesAndNodesToWaitDelete: map[string]phaseutil.MachineAndNode{
			"machine1": {Machine: machine, Node: confv1beta1.Node{IP: "192.168.1.1"}},
		},
		Log: createTestLogger(),
	}

	result, err := e.waitForMachinesDelete(params)
	assert.NoError(t, err)
	assert.Len(t, result, 1)
}

func TestEnsureMasterDelete_WaitMasterDelete_NoMachines(t *testing.T) {
	e := createTestEnsureMasterDelete()
	err := e.waitMasterDelete()
	assert.NoError(t, err)
}

func TestEnsureMasterDelete_NeedExecute_WithTargetNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureMasterDelete()

	patches.ApplyFunc(phaseutil.GetNeedDeleteMasterNodes, func(ctx context.Context, c client.Client, cluster *bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{}
	})

	patches.ApplyFunc(getDeleteTargetNodesIfDeployed, func(ctx *phaseframe.PhaseContext, cluster *bkev1beta1.BKECluster) (bkenode.Nodes, bool) {
		return bkenode.Nodes{{IP: "192.168.1.1"}}, true
	})

	patches.ApplyFunc(phaseutil.GetNeedDeleteMasterNodesWithTargetNodes, func(ctx context.Context, c client.Client, cluster *bkev1beta1.BKECluster, targetNodes bkenode.Nodes) bkenode.Nodes {
		return bkenode.Nodes{{IP: "192.168.1.1"}}
	})

	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{}

	result := e.NeedExecute(old, new)
	assert.True(t, result)
}

func TestPauseAndScaleDownControlPlaneParams_Structure(t *testing.T) {
	params := PauseAndScaleDownControlPlaneParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		DeleteMap:  make(map[string]phaseutil.MachineAndNode),
		Log:        createTestLogger(),
	}
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.DeleteMap)
	assert.NotNil(t, params.Log)
}

func TestWaitForMachinesDeleteParams_Structure(t *testing.T) {
	params := WaitForMachinesDeleteParams{
		Ctx:                          context.Background(),
		Client:                       &fakeClient{},
		MachinesAndNodesToWaitDelete: make(map[string]phaseutil.MachineAndNode),
		Log:                          createTestLogger(),
	}
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.MachinesAndNodesToWaitDelete)
	assert.NotNil(t, params.Log)
}

func TestCleanupDeletedNodePodsParams_Structure(t *testing.T) {
	params := CleanupDeletedNodePodsParams{
		Ctx:                context.Background(),
		Client:             &fakeClient{},
		BKECluster:         &bkev1beta1.BKECluster{},
		SuccessDeletedNode: make(map[string]confv1beta1.Node),
	}
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.SuccessDeletedNode)
}

func newMasterDeleteCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster, objs ...client.Object) *EnsureMasterDelete {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, clusterv1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
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
		Cluster:    &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}},
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return NewEnsureMasterDelete(ctx).(*EnsureMasterDelete)
}

func mdPatchCommonSeams(patches *gomonkey.Patches) {
	patches.ApplyFunc(time.Sleep, func(_ time.Duration) {})
	patches.ApplyFunc((*nodeutil.NodeFetcher).DeleteBKENodeForCluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string) error { return nil })
	patches.ApplyMethod(&statusmanage.StatusManager{}, "RemoveSingleNodeStatusCache",
		func(_ *statusmanage.StatusManager, _ *bkev1beta1.BKECluster, _ string) {})
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
}

// ---- Execute ----

func TestMasterDeleteExecute(t *testing.T) {
	t.Run("reconcile error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "reconcileMasterDelete",
			func(_ *EnsureMasterDelete) error { return assertErr("reconcile failed") })
		_, err := e.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "reconcile failed")
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "reconcileMasterDelete",
			func(_ *EnsureMasterDelete) error { return nil })
		patches.ApplyPrivateMethod(e, "waitMasterDelete",
			func(_ *EnsureMasterDelete) error { return nil })
		_, err := e.Execute()
		require.NoError(t, err)
	})
}

// ---- reconcileMasterDelete ----

func TestMasterDeleteReconcileMasterDelete(t *testing.T) {
	t.Run("no nodes to delete returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(GetTargetClusterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return nil, nil
			})
		patches.ApplyFunc(phaseutil.GetNeedDeleteMasterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) bkenode.Nodes {
				return bkenode.Nodes{}
			})
		patches.ApplyFunc(ProcessNodeMachineMapping,
			func(_ ProcessNodeMachineMappingParams) (ProcessNodeMachineMappingResult, error) {
				return ProcessNodeMachineMappingResult{NodesCount: 0}, nil
			})
		require.NoError(t, e.reconcileMasterDelete())
	})

	t.Run("getTargetClusterNodes error continues with nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(GetTargetClusterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return nil, assertErr("target nodes error")
			})
		patches.ApplyFunc(phaseutil.GetNeedDeleteMasterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) bkenode.Nodes {
				return bkenode.Nodes{}
			})
		patches.ApplyFunc(ProcessNodeMachineMapping,
			func(_ ProcessNodeMachineMappingParams) (ProcessNodeMachineMappingResult, error) {
				return ProcessNodeMachineMappingResult{NodesCount: 0}, nil
			})
		require.NoError(t, e.reconcileMasterDelete())
	})

	t.Run("ProcessNodeMachineMapping error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(GetTargetClusterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return nil, nil
			})
		patches.ApplyFunc(phaseutil.GetNeedDeleteMasterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}}
			})
		patches.ApplyFunc(ProcessNodeMachineMapping,
			func(_ ProcessNodeMachineMappingParams) (ProcessNodeMachineMappingResult, error) {
				return ProcessNodeMachineMappingResult{}, assertErr("mapping failed")
			})
		err := e.reconcileMasterDelete()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "mapping failed")
	})

	t.Run("success with target nodes fallback", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(GetTargetClusterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return bkenode.Nodes{{IP: "10.0.0.1"}}, nil
			})
		// Legacy returns empty -> fallback to target nodes mode
		patches.ApplyFunc(phaseutil.GetNeedDeleteMasterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) bkenode.Nodes {
				return bkenode.Nodes{}
			})
		patches.ApplyFunc(phaseutil.GetNeedDeleteMasterNodesWithTargetNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ bkenode.Nodes) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}}
			})
		patches.ApplyFunc(ProcessNodeMachineMapping,
			func(_ ProcessNodeMachineMappingParams) (ProcessNodeMachineMappingResult, error) {
				return ProcessNodeMachineMappingResult{
					NodesCount:    1,
					DeleteMap:     map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
					WaitDeleteMap: map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
				}, nil
			})
		patches.ApplyPrivateMethod(e, "pauseAndScaleDownControlPlane",
			func(_ *EnsureMasterDelete, _ PauseAndScaleDownControlPlaneParams) error { return nil })
		require.NoError(t, e.reconcileMasterDelete())
	})

	t.Run("pauseAndScaleDownControlPlane error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(GetTargetClusterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return nil, nil
			})
		patches.ApplyFunc(phaseutil.GetNeedDeleteMasterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}}
			})
		patches.ApplyFunc(ProcessNodeMachineMapping,
			func(_ ProcessNodeMachineMappingParams) (ProcessNodeMachineMappingResult, error) {
				return ProcessNodeMachineMappingResult{
					NodesCount: 1,
					DeleteMap:  map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
				}, nil
			})
		patches.ApplyPrivateMethod(e, "pauseAndScaleDownControlPlane",
			func(_ *EnsureMasterDelete, _ PauseAndScaleDownControlPlaneParams) error {
				return assertErr("pause failed")
			})
		err := e.reconcileMasterDelete()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "pause failed")
	})
}

// ---- pauseAndScaleDownControlPlane ----

func TestMasterDeletePauseAndScaleDownControlPlane(t *testing.T) {
	t.Run("KubeadmControlPlane nil returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{}, nil // KCP is nil
			})
		params := PauseAndScaleDownControlPlaneParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			DeleteMap:  map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:        e.Ctx.Log,
		}
		// upstream change: KCP nil with err nil now returns an error.
		err := e.pauseAndScaleDownControlPlane(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "associate objs failed")
	})

	t.Run("GetClusterAPIAssociateObjs error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return nil, assertErr("capi failed")
			})
		params := PauseAndScaleDownControlPlaneParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			DeleteMap:  map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:        e.Ctx.Log,
		}
		err := e.pauseAndScaleDownControlPlane(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "capi failed")
	})

	t.Run("PauseClusterAPIObj error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{KubeadmControlPlane: &controlv1beta1.KubeadmControlPlane{Spec: controlv1beta1.KubeadmControlPlaneSpec{Replicas: int32p(3)}}}, nil
			})
		patches.ApplyFunc(phaseutil.PauseClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error {
				return assertErr("pause failed")
			})
		params := PauseAndScaleDownControlPlaneParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			DeleteMap:  map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:        e.Ctx.Log,
		}
		err := e.pauseAndScaleDownControlPlane(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "pause failed")
	})

	t.Run("all MarkMachineForDeletion fail returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{KubeadmControlPlane: &controlv1beta1.KubeadmControlPlane{Spec: controlv1beta1.KubeadmControlPlaneSpec{Replicas: int32p(3)}}}, nil
			})
		patches.ApplyFunc(phaseutil.PauseClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		patches.ApplyFunc(phaseutil.UpdateKubeadmControlPlaneReplicas,
			func(_ context.Context, _ client.Client, kcp *controlv1beta1.KubeadmControlPlane, replicas int32) error {
				kcp.Spec.Replicas = &replicas
				return nil
			})
		patches.ApplyFunc(phaseutil.MarkMachineForDeletion,
			func(_ context.Context, _ client.Client, _ *clusterv1.Machine) error {
				return assertErr("mark failed")
			})
		// Need to patch ResumeClusterAPIObj for the defer rollback
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		params := PauseAndScaleDownControlPlaneParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			DeleteMap:  map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "m1"}}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:        e.Ctx.Log,
		}
		// When all machines fail to mark, deleteMap is empty -> returns nil
		err := e.pauseAndScaleDownControlPlane(params)
		require.NoError(t, err)
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		kcp := &controlv1beta1.KubeadmControlPlane{Spec: controlv1beta1.KubeadmControlPlaneSpec{Replicas: int32p(3)}}
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{KubeadmControlPlane: kcp}, nil
			})
		patches.ApplyFunc(phaseutil.PauseClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		patches.ApplyFunc(phaseutil.UpdateKubeadmControlPlaneReplicas,
			func(_ context.Context, _ client.Client, kcp *controlv1beta1.KubeadmControlPlane, replicas int32) error {
				kcp.Spec.Replicas = &replicas
				return nil
			})
		patches.ApplyFunc(phaseutil.MarkMachineForDeletion,
			func(_ context.Context, _ client.Client, _ *clusterv1.Machine) error { return nil })
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		params := PauseAndScaleDownControlPlaneParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			DeleteMap:  map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "m1"}}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:        e.Ctx.Log,
		}
		err := e.pauseAndScaleDownControlPlane(params)
		require.NoError(t, err)
		// Replicas should be scaled down: 3 - 1 = 2
		assert.Equal(t, int32(2), *kcp.Spec.Replicas)
	})

	t.Run("ResumeClusterAPIObj error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		kcp := &controlv1beta1.KubeadmControlPlane{Spec: controlv1beta1.KubeadmControlPlaneSpec{Replicas: int32p(3)}}
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{KubeadmControlPlane: kcp}, nil
			})
		patches.ApplyFunc(phaseutil.PauseClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		patches.ApplyFunc(phaseutil.UpdateKubeadmControlPlaneReplicas,
			func(_ context.Context, _ client.Client, kcp *controlv1beta1.KubeadmControlPlane, replicas int32) error {
				kcp.Spec.Replicas = &replicas
				return nil
			})
		patches.ApplyFunc(phaseutil.MarkMachineForDeletion,
			func(_ context.Context, _ client.Client, _ *clusterv1.Machine) error { return nil })
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error {
				return assertErr("resume failed")
			})
		params := PauseAndScaleDownControlPlaneParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			DeleteMap:  map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "m1"}}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:        e.Ctx.Log,
		}
		err := e.pauseAndScaleDownControlPlane(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "resume failed")
	})

	t.Run("replicas floored to 1", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		kcp := &controlv1beta1.KubeadmControlPlane{Spec: controlv1beta1.KubeadmControlPlaneSpec{Replicas: int32p(1)}}
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{KubeadmControlPlane: kcp}, nil
			})
		patches.ApplyFunc(phaseutil.PauseClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		patches.ApplyFunc(phaseutil.UpdateKubeadmControlPlaneReplicas,
			func(_ context.Context, _ client.Client, kcp *controlv1beta1.KubeadmControlPlane, replicas int32) error {
				kcp.Spec.Replicas = &replicas
				return nil
			})
		patches.ApplyFunc(phaseutil.MarkMachineForDeletion,
			func(_ context.Context, _ client.Client, _ *clusterv1.Machine) error { return nil })
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		params := PauseAndScaleDownControlPlaneParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			DeleteMap: map[string]phaseutil.MachineAndNode{
				"m1": {Machine: &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "m1"}}, Node: confv1beta1.Node{IP: "10.0.0.1"}},
				"m2": {Machine: &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "m2"}}, Node: confv1beta1.Node{IP: "10.0.0.2"}},
			},
			Log: e.Ctx.Log,
		}
		// 1 - 2 = -1, floored to 1
		err := e.pauseAndScaleDownControlPlane(params)
		require.NoError(t, err)
		assert.Equal(t, int32(1), *kcp.Spec.Replicas)
	})
}

// ---- cleanupDeletedNodePods ----

func TestMasterDeleteCleanupDeletedNodePods(t *testing.T) {
	t.Run("success with mockClient and pods", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		mdPatchCommonSeams(patches)
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "ns"}, Spec: corev1.PodSpec{NodeName: "node1"}}
		e.mockClient = k8sfake.NewSimpleClientset(pod)
		params := CleanupDeletedNodePodsParams{
			Ctx:                context.Background(),
			Client:             e.Ctx.Client,
			BKECluster:         e.Ctx.BKECluster,
			SuccessDeletedNode: map[string]confv1beta1.Node{"m1": {IP: "10.0.0.1", Hostname: "node1"}},
		}
		require.NoError(t, e.cleanupDeletedNodePods(params))
	})

	t.Run("NewRemoteClientByBKECluster error returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		mdPatchCommonSeams(patches)
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		// mockClient is nil -> uses NewRemoteClientByBKECluster
		patches.ApplyFunc(kube.NewRemoteClientByBKECluster,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (kube.RemoteKubeClient, error) {
				return nil, assertErr("remote client failed")
			})
		params := CleanupDeletedNodePodsParams{
			Ctx:                context.Background(),
			Client:             e.Ctx.Client,
			BKECluster:         e.Ctx.BKECluster,
			SuccessDeletedNode: map[string]confv1beta1.Node{"m1": {IP: "10.0.0.1", Hostname: "node1"}},
		}
		require.NoError(t, e.cleanupDeletedNodePods(params))
	})

	t.Run("list error continues", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		mdPatchCommonSeams(patches)
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		fakeCs := k8sfake.NewSimpleClientset()
		fakeCs.PrependReactor("list", "pods", func(_ k8stesting.Action) (bool, runtime.Object, error) {
			return true, nil, assertErr("list error")
		})
		e.mockClient = fakeCs
		params := CleanupDeletedNodePodsParams{
			Ctx:                context.Background(),
			Client:             e.Ctx.Client,
			BKECluster:         e.Ctx.BKECluster,
			SuccessDeletedNode: map[string]confv1beta1.Node{"m1": {IP: "10.0.0.1", Hostname: "node1"}},
		}
		// List error -> continues to next node -> SyncStatusUntilComplete (patched to nil)
		require.NoError(t, e.cleanupDeletedNodePods(params))
	})

	t.Run("delete error continues", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		mdPatchCommonSeams(patches)
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "ns"}, Spec: corev1.PodSpec{NodeName: "node1"}}
		fakeCs := k8sfake.NewSimpleClientset(pod)
		fakeCs.PrependReactor("delete", "pods", func(_ k8stesting.Action) (bool, runtime.Object, error) {
			return true, nil, assertErr("delete error")
		})
		e.mockClient = fakeCs
		params := CleanupDeletedNodePodsParams{
			Ctx:                context.Background(),
			Client:             e.Ctx.Client,
			BKECluster:         e.Ctx.BKECluster,
			SuccessDeletedNode: map[string]confv1beta1.Node{"m1": {IP: "10.0.0.1", Hostname: "node1"}},
		}
		// Delete error -> continues -> SyncStatusUntilComplete (patched to nil)
		require.NoError(t, e.cleanupDeletedNodePods(params))
	})
}

// ---- waitMasterDelete ----

func TestMasterDeleteWaitMasterDelete(t *testing.T) {
	t.Run("empty map returns nil", func(t *testing.T) {
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		require.NoError(t, e.waitMasterDelete())
	})

	t.Run("timeout error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.machinesAndNodesToWaitDelete = map[string]phaseutil.MachineAndNode{
			"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}},
		}
		patches.ApplyPrivateMethod(e, "waitForMachinesDelete",
			func(_ *EnsureMasterDelete, _ WaitForMachinesDeleteParams) (map[string]confv1beta1.Node, error) {
				return nil, wait.ErrWaitTimeout
			})
		err := e.waitMasterDelete()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Wait master node delete failed")
	})

	t.Run("other error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.machinesAndNodesToWaitDelete = map[string]phaseutil.MachineAndNode{
			"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}},
		}
		patches.ApplyPrivateMethod(e, "waitForMachinesDelete",
			func(_ *EnsureMasterDelete, _ WaitForMachinesDeleteParams) (map[string]confv1beta1.Node, error) {
				return nil, assertErr("get error")
			})
		err := e.waitMasterDelete()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "get error")
	})

	t.Run("success with cleanup", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.machinesAndNodesToWaitDelete = map[string]phaseutil.MachineAndNode{
			"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}},
		}
		patches.ApplyPrivateMethod(e, "waitForMachinesDelete",
			func(_ *EnsureMasterDelete, _ WaitForMachinesDeleteParams) (map[string]confv1beta1.Node, error) {
				return map[string]confv1beta1.Node{"m1": {IP: "10.0.0.1", Hostname: "node1"}}, nil
			})
		patches.ApplyPrivateMethod(e, "cleanupDeletedNodePods",
			func(_ *EnsureMasterDelete, _ CleanupDeletedNodePodsParams) error { return nil })
		require.NoError(t, e.waitMasterDelete())
	})

	t.Run("success without deleted nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.machinesAndNodesToWaitDelete = map[string]phaseutil.MachineAndNode{
			"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}},
		}
		patches.ApplyPrivateMethod(e, "waitForMachinesDelete",
			func(_ *EnsureMasterDelete, _ WaitForMachinesDeleteParams) (map[string]confv1beta1.Node, error) {
				return map[string]confv1beta1.Node{}, nil
			})
		// No success deleted nodes -> returns nil without cleanup
		require.NoError(t, e.waitMasterDelete())
	})
}

// ---- kubeClient ----

func TestMasterDeleteKubeClient(t *testing.T) {
	e := newMasterDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	assert.Nil(t, e.kubeClient())
	e.mockClient = k8sfake.NewSimpleClientset()
	assert.NotNil(t, e.kubeClient())
}
