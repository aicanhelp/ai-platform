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
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	kubedrain "k8s.io/kubectl/pkg/drain"
	"k8s.io/utils/pointer"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureWorkerDeleteConstants(t *testing.T) {
	assert.Equal(t, "EnsureWorkerDelete", string(EnsureWorkerDeleteName))
	assert.Equal(t, 10, WorkerDeleteRequeueAfterSeconds)
	assert.Equal(t, 4, WorkerDeleteWaitTimeoutMinutes)
	assert.Equal(t, 2, WorkerDeletePollIntervalSeconds)
}

func TestNewEnsureWorkerDelete(t *testing.T) {
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

	phase := NewEnsureWorkerDelete(ctx)
	assert.NotNil(t, phase)

	workerDelete, ok := phase.(*EnsureWorkerDelete)
	assert.True(t, ok)
	assert.NotNil(t, workerDelete)
	assert.NotNil(t, workerDelete.machinesAndNodesToWaitDelete)
	assert.NotNil(t, workerDelete.machinesAndNodesToDelete)
}

func TestPrepareMachinesAndNodesToWaitDelete(t *testing.T) {
	machine1 := &clusterv1.Machine{
		ObjectMeta: metav1.ObjectMeta{Name: "machine1", Namespace: "default"},
	}
	machine2 := &clusterv1.Machine{
		ObjectMeta: metav1.ObjectMeta{Name: "machine2", Namespace: "default"},
	}

	node1 := confv1beta1.Node{IP: "192.168.1.1", Hostname: "node1"}
	node2 := confv1beta1.Node{IP: "192.168.1.2", Hostname: "node2"}

	tests := []struct {
		name                         string
		machinesAndNodesToWaitDelete map[string]phaseutil.MachineAndNode
		machinesAndNodesToDelete     map[string]phaseutil.MachineAndNode
		expectedCount                int
	}{
		{
			name: "Both maps have items",
			machinesAndNodesToWaitDelete: map[string]phaseutil.MachineAndNode{
				"machine1": {Machine: machine1, Node: node1},
			},
			machinesAndNodesToDelete: map[string]phaseutil.MachineAndNode{
				"machine2": {Machine: machine2, Node: node2},
			},
			expectedCount: 2,
		},
		{
			name: "Only wait delete map has items",
			machinesAndNodesToWaitDelete: map[string]phaseutil.MachineAndNode{
				"machine1": {Machine: machine1, Node: node1},
			},
			machinesAndNodesToDelete: nil,
			expectedCount:            1,
		},
		{
			name:                         "Only delete map has items",
			machinesAndNodesToWaitDelete: map[string]phaseutil.MachineAndNode{},
			machinesAndNodesToDelete: map[string]phaseutil.MachineAndNode{
				"machine2": {Machine: machine2, Node: node2},
			},
			expectedCount: 1,
		},
		{
			name:                         "Both maps empty",
			machinesAndNodesToWaitDelete: map[string]phaseutil.MachineAndNode{},
			machinesAndNodesToDelete:     map[string]phaseutil.MachineAndNode{},
			expectedCount:                0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := PrepareMachinesAndNodesToWaitDeleteParams{
				MachinesAndNodesToWaitDelete: tt.machinesAndNodesToWaitDelete,
				MachinesAndNodesToDelete:     tt.machinesAndNodesToDelete,
			}
			result := prepareMachinesAndNodesToWaitDelete(params)
			assert.Equal(t, tt.expectedCount, len(result))
		})
	}
}

func TestEnsureWorkerDelete_PrepareMachinesAndNodesToWaitDelete(t *testing.T) {
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

	phase := NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)

	machine := &clusterv1.Machine{
		ObjectMeta: metav1.ObjectMeta{Name: "machine1", Namespace: "default"},
	}
	node := confv1beta1.Node{IP: "192.168.1.1", Hostname: "node1"}

	phase.machinesAndNodesToWaitDelete = map[string]phaseutil.MachineAndNode{
		"machine1": {Machine: machine, Node: node},
	}
	phase.machinesAndNodesToDelete = map[string]phaseutil.MachineAndNode{
		"machine2": {Machine: machine, Node: node},
	}

	result := phase.prepareMachinesAndNodesToWaitDelete()
	assert.Equal(t, 2, len(result))
}

func TestEnsureWorkerDelete_NeedExecute(t *testing.T) {
	t.Skip("Skipping - requires complex setup with GetNeedDeleteWorkerNodes")
}

func TestEnsureWorkerDelete_Execute(t *testing.T) {
	t.Skip("Skipping - requires complex mocking of reconcileWorkerDelete")
}

func TestEnsureWorkerDelete_GetTargetClusterNodes(t *testing.T) {
	t.Skip("Skipping - requires complex setup with GetTargetClusterNodes")
}

func TestEnsureWorkerDelete_DrainNodes(t *testing.T) {
	t.Skip("Skipping - requires complex mocking of kubernetes client")
}

func TestEnsureWorkerDelete_MarkMachinesForDeletion(t *testing.T) {
	t.Skip("Skipping - requires complex mocking of MarkMachineForDeletion")
}

func TestEnsureWorkerDelete_InitialSetup(t *testing.T) {
	t.Skip("Skipping - requires complex setup with ProcessNodeMachineMapping")
}

func TestEnsureWorkerDelete_ProcessDrainAndMark(t *testing.T) {
	t.Skip("Skipping - requires complex mocking")
}

func TestEnsureWorkerDelete_FinalizeDeletion(t *testing.T) {
	t.Skip("Skipping - requires complex mocking")
}

func TestEnsureWorkerDelete_ReconcileWorkerDelete(t *testing.T) {
	t.Skip("Skipping - requires complex mocking")
}

func TestEnsureWorkerDelete_WaitWorkerDelete(t *testing.T) {
	t.Skip("Skipping - requires complex mocking")
}

func TestEnsureWorkerDelete_WaitForMachinesDelete(t *testing.T) {
	t.Skip("Skipping - requires complex mocking")
}

func TestEnsureWorkerDelete_ProcessSuccessfulDeletions(t *testing.T) {
	t.Skip("Skipping - requires complex mocking")
}

func TestEnsureWorkerDelete_CleanupNodePods(t *testing.T) {
	t.Skip("Skipping - requires complex mocking of kubernetes client")
}

func TestEnsureWorkerDelete_FinalizeDeletion_NoNodesToDelete(t *testing.T) {
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

	phase := NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)

	md := &clusterv1.MachineDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-md", Namespace: "default"},
		Spec: clusterv1.MachineDeploymentSpec{
			Replicas: pointer.Int32(3),
		},
	}

	scope := &phaseutil.ClusterAPIObjs{
		MachineDeployment: md,
	}

	params := FinalizeDeletionParams{
		Ctx:        context.Background(),
		Client:     c,
		BKECluster: bkeCluster,
		MarkResult: MarkMachinesForDeletionResult{
			FinalMachineToNodeDeleteMap:       map[string]phaseutil.MachineAndNode{},
			FinalCanNotDeleteMachinesAndNodes: map[string]phaseutil.MachineAndNode{},
		},
		Scope:           scope,
		CurrentReplicas: pointer.Int32(3),
		Log:             ctx.Log,
	}

	result := phase.finalizeDeletion(params)
	assert.Error(t, result.Error)
	assert.Contains(t, result.Error.Error(), "cannot be completely deleted")
}

func TestEnsureWorkerDelete_FinalizeDeletion_WithCanNotDeleteNodes(t *testing.T) {
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

	phase := NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)

	machine := &clusterv1.Machine{
		ObjectMeta: metav1.ObjectMeta{Name: "machine1", Namespace: "default"},
	}
	node := confv1beta1.Node{IP: "192.168.1.1", Hostname: "node1"}

	md := &clusterv1.MachineDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-md", Namespace: "default"},
		Spec: clusterv1.MachineDeploymentSpec{
			Replicas: pointer.Int32(3),
		},
	}

	scope := &phaseutil.ClusterAPIObjs{
		MachineDeployment: md,
	}

	params := FinalizeDeletionParams{
		Ctx:        context.Background(),
		Client:     c,
		BKECluster: bkeCluster,
		MarkResult: MarkMachinesForDeletionResult{
			FinalMachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{},
			FinalCanNotDeleteMachinesAndNodes: map[string]phaseutil.MachineAndNode{
				"machine1": {Machine: machine, Node: node},
			},
		},
		Scope:           scope,
		CurrentReplicas: pointer.Int32(3),
		Log:             ctx.Log,
	}

	result := phase.finalizeDeletion(params)
	assert.Error(t, result.Error)
	assert.Equal(t, time.Duration(WorkerDeleteRequeueAfterSeconds)*time.Second, result.Result.RequeueAfter)
}

func TestEnsureWorkerDelete_FinalizeDeletion_Success(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}

	machine := &clusterv1.Machine{
		ObjectMeta: metav1.ObjectMeta{Name: "machine1", Namespace: "default"},
	}
	node := confv1beta1.Node{IP: "192.168.1.1", Hostname: "node1"}

	md := &clusterv1.MachineDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-md", Namespace: "default"},
		Spec: clusterv1.MachineDeploymentSpec{
			Replicas: pointer.Int32(3),
		},
	}

	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster, md).Build()

	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)

	scope := &phaseutil.ClusterAPIObjs{
		MachineDeployment: md,
	}

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(phaseutil.ResumeClusterAPIObj, func(ctx context.Context, c any, obj any) error {
		return nil
	})

	params := FinalizeDeletionParams{
		Ctx:        context.Background(),
		Client:     c,
		BKECluster: bkeCluster,
		MarkResult: MarkMachinesForDeletionResult{
			FinalMachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{
				"machine1": {Machine: machine, Node: node},
			},
			FinalCanNotDeleteMachinesAndNodes: map[string]phaseutil.MachineAndNode{},
		},
		Scope:           scope,
		CurrentReplicas: pointer.Int32(3),
		Log:             ctx.Log,
	}

	result := phase.finalizeDeletion(params)
	assert.NoError(t, result.Error)
	assert.Equal(t, int32(2), *scope.MachineDeployment.Spec.Replicas)
}

func TestEnsureWorkerDelete_FinalizeDeletion_NegativeReplicas(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}

	machine1 := &clusterv1.Machine{
		ObjectMeta: metav1.ObjectMeta{Name: "machine1", Namespace: "default"},
	}
	machine2 := &clusterv1.Machine{
		ObjectMeta: metav1.ObjectMeta{Name: "machine2", Namespace: "default"},
	}
	node1 := confv1beta1.Node{IP: "192.168.1.1", Hostname: "node1"}
	node2 := confv1beta1.Node{IP: "192.168.1.2", Hostname: "node2"}

	md := &clusterv1.MachineDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-md", Namespace: "default"},
		Spec: clusterv1.MachineDeploymentSpec{
			Replicas: pointer.Int32(1),
		},
	}

	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster, md).Build()

	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)

	scope := &phaseutil.ClusterAPIObjs{
		MachineDeployment: md,
	}

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(phaseutil.ResumeClusterAPIObj, func(ctx context.Context, c any, obj any) error {
		return nil
	})

	params := FinalizeDeletionParams{
		Ctx:        context.Background(),
		Client:     c,
		BKECluster: bkeCluster,
		MarkResult: MarkMachinesForDeletionResult{
			FinalMachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{
				"machine1": {Machine: machine1, Node: node1},
				"machine2": {Machine: machine2, Node: node2},
			},
			FinalCanNotDeleteMachinesAndNodes: map[string]phaseutil.MachineAndNode{},
		},
		Scope:           scope,
		CurrentReplicas: pointer.Int32(1),
		Log:             ctx.Log,
	}

	result := phase.finalizeDeletion(params)
	assert.NoError(t, result.Error)
	assert.Equal(t, int32(0), *scope.MachineDeployment.Spec.Replicas)
}

func TestEnsureWorkerDelete_GetTargetClusterNodes_Success(t *testing.T) {
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

	phase := NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	expectedNodes := bkenode.Nodes{
		{IP: "192.168.1.1", Hostname: "node1"},
		{IP: "192.168.1.2", Hostname: "node2"},
	}

	patches.ApplyFunc(GetTargetClusterNodes, func(ctx context.Context, c any, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
		return expectedNodes, nil
	})

	nodes, err := phase.getTargetClusterNodes(bkeCluster)
	assert.NoError(t, err)
	assert.Equal(t, 2, len(nodes))
}

func TestEnsureWorkerDelete_WaitWorkerDelete_EmptyMap(t *testing.T) {
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

	phase := NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)
	phase.machinesAndNodesToWaitDelete = map[string]phaseutil.MachineAndNode{}
	phase.machinesAndNodesToDelete = map[string]phaseutil.MachineAndNode{}

	err := phase.waitWorkerDelete()
	assert.NoError(t, err)
}

func TestEnsureWorkerDelete_NeedExecute_WithLegacyNodes(t *testing.T) {
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

	phase := NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodes, func(ctx context.Context, c any, bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{
			{IP: "192.168.1.1", Hostname: "node1"},
		}
	})

	result := phase.NeedExecute(nil, bkeCluster)
	assert.True(t, result)
}

func TestEnsureWorkerDelete_NeedExecute_NoNodes(t *testing.T) {
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

	phase := NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodes, func(ctx context.Context, c any, bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{}
	})

	patches.ApplyFunc(getDeleteTargetNodesIfDeployed, func(ctx *phaseframe.PhaseContext, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, bool) {
		return nil, false
	})

	result := phase.NeedExecute(nil, bkeCluster)
	assert.False(t, result)
}

func TestEnsureWorkerDelete_Execute_Success(t *testing.T) {
	t.Skip("Skipping - cannot mock private methods with gomonkey")
}

func TestEnsureWorkerDelete_NeedExecute_WithTargetNodes(t *testing.T) {
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

	phase := NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodes, func(ctx context.Context, c any, bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{}
	})

	patches.ApplyFunc(getDeleteTargetNodesIfDeployed, func(ctx *phaseframe.PhaseContext, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, bool) {
		return bkenode.Nodes{
			{IP: "192.168.1.1", Hostname: "node1"},
		}, true
	})

	patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodesWithTargetNodes, func(ctx context.Context, c any, bkeCluster *bkev1beta1.BKECluster, targetNodes bkenode.Nodes) bkenode.Nodes {
		return bkenode.Nodes{
			{IP: "192.168.1.1", Hostname: "node1"},
		}
	})

	result := phase.NeedExecute(nil, bkeCluster)
	assert.True(t, result)
}

func TestEnsureWorkerDelete_NeedExecute_TargetNodesButNoDelete(t *testing.T) {
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

	phase := NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodes, func(ctx context.Context, c any, bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
		return bkenode.Nodes{}
	})

	patches.ApplyFunc(getDeleteTargetNodesIfDeployed, func(ctx *phaseframe.PhaseContext, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, bool) {
		return bkenode.Nodes{
			{IP: "192.168.1.1", Hostname: "node1"},
		}, true
	})

	patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodesWithTargetNodes, func(ctx context.Context, c any, bkeCluster *bkev1beta1.BKECluster, targetNodes bkenode.Nodes) bkenode.Nodes {
		return bkenode.Nodes{}
	})

	result := phase.NeedExecute(nil, bkeCluster)
	assert.False(t, result)
}

func newWorkerDeleteCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster, objs ...client.Object) *EnsureWorkerDelete {
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
	return NewEnsureWorkerDelete(ctx).(*EnsureWorkerDelete)
}

// Common seam patches for worker delete helpers
func wdPatchCommonSeams(patches *gomonkey.Patches) {
	patches.ApplyFunc(time.Sleep, func(_ time.Duration) {})
	patches.ApplyFunc((*nodeutil.NodeFetcher).DeleteBKENodeForCluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string) error { return nil })
	patches.ApplyFunc((*nodeutil.NodeFetcher).SetNodeStateWithMessageForCluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ confv1beta1.NodeState, _ string) error {
			return nil
		})
	patches.ApplyMethod(&statusmanage.StatusManager{}, "RemoveSingleNodeStatusCache",
		func(_ *statusmanage.StatusManager, _ *bkev1beta1.BKECluster, _ string) {})
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
}

// ---- Execute ----

func TestWorkerDeleteExecute(t *testing.T) {
	t.Run("reconcile error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "reconcileWorkerDelete",
			func(_ *EnsureWorkerDelete) (ctrl.Result, error) { return ctrl.Result{}, assertErr("reconcile failed") })
		_, err := e.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "reconcile failed")
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "reconcileWorkerDelete",
			func(_ *EnsureWorkerDelete) (ctrl.Result, error) { return ctrl.Result{}, nil })
		_, err := e.Execute()
		require.NoError(t, err)
	})
}

// ---- reconcileWorkerDelete ----

func TestWorkerDeleteReconcileWorkerDelete(t *testing.T) {
	t.Run("initialSetup error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(GetTargetClusterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return nil, nil
			})
		patches.ApplyPrivateMethod(e, "initialSetup",
			func(_ *EnsureWorkerDelete, _ InitialSetupParams) InitialSetupResult {
				return InitialSetupResult{Error: assertErr("setup failed")}
			})
		_, err := e.reconcileWorkerDelete()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "setup failed")
	})

	t.Run("no nodes to delete returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(GetTargetClusterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return nil, nil
			})
		patches.ApplyPrivateMethod(e, "initialSetup",
			func(_ *EnsureWorkerDelete, _ InitialSetupParams) InitialSetupResult {
				return InitialSetupResult{Error: nil}
			})
		_, err := e.reconcileWorkerDelete()
		require.NoError(t, err)
	})

	t.Run("getTargetClusterNodes error continues with nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(GetTargetClusterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return nil, assertErr("target nodes error")
			})
		patches.ApplyPrivateMethod(e, "initialSetup",
			func(_ *EnsureWorkerDelete, _ InitialSetupParams) InitialSetupResult {
				return InitialSetupResult{Error: nil}
			})
		_, err := e.reconcileWorkerDelete()
		require.NoError(t, err)
	})

	t.Run("success path with private method patches", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(GetTargetClusterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return nil, nil
			})
		md := &clusterv1.MachineDeployment{Spec: clusterv1.MachineDeploymentSpec{Replicas: int32p(3)}}
		patches.ApplyPrivateMethod(e, "initialSetup",
			func(_ *EnsureWorkerDelete, _ InitialSetupParams) InitialSetupResult {
				return InitialSetupResult{
					NodeMappingResult: ProcessNodeMachineMappingResult{NodesCount: 1},
					Scope:             &phaseutil.ClusterAPIObjs{MachineDeployment: md},
					CurrentReplicas:   int32p(3),
				}
			})
		patches.ApplyPrivateMethod(e, "processDrainAndMark",
			func(_ *EnsureWorkerDelete, _ ProcessDrainAndMarkParams) ProcessDrainAndMarkResult {
				return ProcessDrainAndMarkResult{
					MarkResult: MarkMachinesForDeletionResult{
						FinalMachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{
							"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}},
						},
					},
				}
			})
		patches.ApplyFunc(phaseutil.UpdateMachineDeploymentReplicas,
			func(_ context.Context, _ client.Client, md *clusterv1.MachineDeployment, replicas int32) error {
				md.Spec.Replicas = &replicas
				return nil
			})
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		patches.ApplyPrivateMethod(e, "waitWorkerDelete",
			func(_ *EnsureWorkerDelete) error { return nil })
		_, err := e.reconcileWorkerDelete()
		require.NoError(t, err)
	})

	t.Run("finalizeDeletion error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(GetTargetClusterNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return nil, nil
			})
		md := &clusterv1.MachineDeployment{Spec: clusterv1.MachineDeploymentSpec{Replicas: int32p(3)}}
		patches.ApplyPrivateMethod(e, "initialSetup",
			func(_ *EnsureWorkerDelete, _ InitialSetupParams) InitialSetupResult {
				return InitialSetupResult{
					NodeMappingResult: ProcessNodeMachineMappingResult{NodesCount: 1},
					Scope:             &phaseutil.ClusterAPIObjs{MachineDeployment: md},
					CurrentReplicas:   int32p(3),
				}
			})
		patches.ApplyPrivateMethod(e, "processDrainAndMark",
			func(_ *EnsureWorkerDelete, _ ProcessDrainAndMarkParams) ProcessDrainAndMarkResult {
				return ProcessDrainAndMarkResult{}
			})
		patches.ApplyFunc(phaseutil.UpdateMachineDeploymentReplicas,
			func(_ context.Context, _ client.Client, md *clusterv1.MachineDeployment, replicas int32) error {
				md.Spec.Replicas = &replicas
				return nil
			})
		patches.ApplyFunc(phaseutil.ResumeClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		_, err := e.reconcileWorkerDelete()
		require.Error(t, err)
	})
}

// ---- initialSetup ----

func TestWorkerDeleteInitialSetup(t *testing.T) {
	t.Run("ProcessNodeMachineMapping error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}}
			})
		patches.ApplyFunc(ProcessNodeMachineMapping,
			func(_ ProcessNodeMachineMappingParams) (ProcessNodeMachineMappingResult, error) {
				return ProcessNodeMachineMappingResult{}, assertErr("mapping failed")
			})
		result := e.initialSetup(InitialSetupParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Cluster:    e.Ctx.Cluster,
			Log:        e.Ctx.Log,
		})
		require.Error(t, result.Error)
		assert.Contains(t, result.Error.Error(), "mapping failed")
	})

	t.Run("no nodes to delete returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) bkenode.Nodes {
				return bkenode.Nodes{}
			})
		patches.ApplyFunc(ProcessNodeMachineMapping,
			func(_ ProcessNodeMachineMappingParams) (ProcessNodeMachineMappingResult, error) {
				return ProcessNodeMachineMappingResult{NodesCount: 0}, nil
			})
		result := e.initialSetup(InitialSetupParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Cluster:    e.Ctx.Cluster,
			Log:        e.Ctx.Log,
		})
		require.NoError(t, result.Error)
	})

	t.Run("GetClusterAPIAssociateObjs error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodes,
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
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return nil, assertErr("capi failed")
			})
		result := e.initialSetup(InitialSetupParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Cluster:    e.Ctx.Cluster,
			Log:        e.Ctx.Log,
		})
		require.Error(t, result.Error)
		assert.Contains(t, result.Error.Error(), "capi failed")
	})

	t.Run("MachineDeployment nil returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}}
			})
		patches.ApplyFunc(ProcessNodeMachineMapping,
			func(_ ProcessNodeMachineMappingParams) (ProcessNodeMachineMappingResult, error) {
				return ProcessNodeMachineMappingResult{NodesCount: 1}, nil
			})
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{}, nil
			})
		result := e.initialSetup(InitialSetupParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Cluster:    e.Ctx.Cluster,
			Log:        e.Ctx.Log,
		})
		// upstream change: MachineDeployment nil with err nil now returns an error.
		require.Error(t, result.Error)
		assert.Contains(t, result.Error.Error(), "associate objs failed")
	})

	t.Run("PauseClusterAPIObj error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}}
			})
		patches.ApplyFunc(ProcessNodeMachineMapping,
			func(_ ProcessNodeMachineMappingParams) (ProcessNodeMachineMappingResult, error) {
				return ProcessNodeMachineMappingResult{NodesCount: 1}, nil
			})
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{MachineDeployment: &clusterv1.MachineDeployment{}}, nil
			})
		patches.ApplyFunc(phaseutil.PauseClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error {
				return assertErr("pause failed")
			})
		result := e.initialSetup(InitialSetupParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Cluster:    e.Ctx.Cluster,
			Log:        e.Ctx.Log,
		})
		require.Error(t, result.Error)
		assert.Contains(t, result.Error.Error(), "pause failed")
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodes,
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
		patches.ApplyFunc(phaseutil.GetClusterAPIAssociateObjs,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*phaseutil.ClusterAPIObjs, error) {
				return &phaseutil.ClusterAPIObjs{MachineDeployment: &clusterv1.MachineDeployment{Spec: clusterv1.MachineDeploymentSpec{Replicas: int32p(3)}}}, nil
			})
		patches.ApplyFunc(phaseutil.PauseClusterAPIObj,
			func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error { return nil })
		result := e.initialSetup(InitialSetupParams{
			Ctx:        context.Background(),
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Cluster:    e.Ctx.Cluster,
			Log:        e.Ctx.Log,
		})
		require.NoError(t, result.Error)
		assert.NotNil(t, result.Scope)
		assert.NotNil(t, result.CurrentReplicas)
	})

	t.Run("target nodes fallback mode", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		// Legacy mode returns empty -> fallback to target nodes mode
		patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) bkenode.Nodes {
				return bkenode.Nodes{}
			})
		patches.ApplyFunc(phaseutil.GetNeedDeleteWorkerNodesWithTargetNodes,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ bkenode.Nodes) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}}
			})
		patches.ApplyFunc(ProcessNodeMachineMapping,
			func(_ ProcessNodeMachineMappingParams) (ProcessNodeMachineMappingResult, error) {
				return ProcessNodeMachineMappingResult{NodesCount: 0}, nil
			})
		result := e.initialSetup(InitialSetupParams{
			Ctx:         context.Background(),
			Client:      e.Ctx.Client,
			BKECluster:  e.Ctx.BKECluster,
			Cluster:     e.Ctx.Cluster,
			Log:         e.Ctx.Log,
			TargetNodes: bkenode.Nodes{{IP: "10.0.0.1"}},
		})
		require.NoError(t, result.Error)
	})
}

// ---- drainNodes ----

func TestWorkerDeleteDrainNodes(t *testing.T) {
	wdCommonPatches := func(patches *gomonkey.Patches) {
		patches.ApplyFunc(phaseutil.NewDrainer,
			func(_ context.Context, _ kubernetes.Interface, _ dynamic.Interface, _ bool, _ *bkev1beta1.BKELogger) *kubedrain.Helper {
				return &kubedrain.Helper{}
			})
		patches.ApplyFunc(kube.GetTargetClusterClient,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
				return nil, nil, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).SetNodeStateWithMessageForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ confv1beta1.NodeState, _ string) error {
				return nil
			})
	}

	t.Run("node not found deletes directly", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdCommonPatches(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetRemoteNodeByBKENode,
			func(_ context.Context, _ *kubernetes.Clientset, _ confv1beta1.Node) (*corev1.Node, error) {
				return nil, apierrors.NewNotFound(schema.GroupResource{Resource: "nodes"}, "node1")
			})
		params := DrainNodesParams{
			Ctx:                    context.Background(),
			Client:                 e.Ctx.Client,
			BKECluster:             e.Ctx.BKECluster,
			MachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:                    e.Ctx.Log,
		}
		result := e.drainNodes(params)
		// Node not found -> continue, remains in map
		assert.Len(t, result.UpdatedMachineToNodeDeleteMap, 1)
		assert.Empty(t, result.CanNotDeleteMachinesAndNodes)
	})

	t.Run("drain success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdCommonPatches(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetRemoteNodeByBKENode,
			func(_ context.Context, _ *kubernetes.Clientset, _ confv1beta1.Node) (*corev1.Node, error) {
				return &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1"}}, nil
			})
		patches.ApplyFunc((*kubedrain.Helper).GetPodsForDeletion,
			func(_ *kubedrain.Helper, _ string) (*kubedrain.PodDeleteList, []error) {
				return &kubedrain.PodDeleteList{}, nil
			})
		patches.ApplyFunc(kubedrain.RunNodeDrain, func(_ *kubedrain.Helper, _ string) error { return nil })
		params := DrainNodesParams{
			Ctx:                    context.Background(),
			Client:                 e.Ctx.Client,
			BKECluster:             e.Ctx.BKECluster,
			MachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:                    e.Ctx.Log,
		}
		result := e.drainNodes(params)
		assert.Len(t, result.UpdatedMachineToNodeDeleteMap, 1)
		assert.Empty(t, result.CanNotDeleteMachinesAndNodes)
	})

	t.Run("RunNodeDrain error moves to cannot delete", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdCommonPatches(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetRemoteNodeByBKENode,
			func(_ context.Context, _ *kubernetes.Clientset, _ confv1beta1.Node) (*corev1.Node, error) {
				return &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1"}}, nil
			})
		patches.ApplyFunc((*kubedrain.Helper).GetPodsForDeletion,
			func(_ *kubedrain.Helper, _ string) (*kubedrain.PodDeleteList, []error) {
				return &kubedrain.PodDeleteList{}, nil
			})
		patches.ApplyFunc(kubedrain.RunNodeDrain, func(_ *kubedrain.Helper, _ string) error {
			return assertErr("drain failed")
		})
		params := DrainNodesParams{
			Ctx:                    context.Background(),
			Client:                 e.Ctx.Client,
			BKECluster:             e.Ctx.BKECluster,
			MachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:                    e.Ctx.Log,
		}
		result := e.drainNodes(params)
		assert.Empty(t, result.UpdatedMachineToNodeDeleteMap)
		assert.Len(t, result.CanNotDeleteMachinesAndNodes, 1)
	})

	t.Run("GetPodsForDeletion error moves to cannot delete", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdCommonPatches(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.GetRemoteNodeByBKENode,
			func(_ context.Context, _ *kubernetes.Clientset, _ confv1beta1.Node) (*corev1.Node, error) {
				return &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1"}}, nil
			})
		patches.ApplyFunc((*kubedrain.Helper).GetPodsForDeletion,
			func(_ *kubedrain.Helper, _ string) (*kubedrain.PodDeleteList, []error) {
				return &kubedrain.PodDeleteList{}, []error{assertErr("get pods failed")}
			})
		patches.ApplyFunc(kubedrain.RunNodeDrain, func(_ *kubedrain.Helper, _ string) error { return nil })
		params := DrainNodesParams{
			Ctx:                    context.Background(),
			Client:                 e.Ctx.Client,
			BKECluster:             e.Ctx.BKECluster,
			MachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:                    e.Ctx.Log,
		}
		result := e.drainNodes(params)
		assert.Empty(t, result.UpdatedMachineToNodeDeleteMap)
		assert.Len(t, result.CanNotDeleteMachinesAndNodes, 1)
	})
}

// ---- markMachinesForDeletion ----

func TestWorkerDeleteMarkMachinesForDeletion(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdPatchCommonSeams(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.MarkMachineForDeletion,
			func(_ context.Context, _ client.Client, _ *clusterv1.Machine) error { return nil })
		params := MarkMachinesForDeletionParams{
			Ctx:                    context.Background(),
			Client:                 e.Ctx.Client,
			BKECluster:             e.Ctx.BKECluster,
			MachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:                    e.Ctx.Log,
		}
		result := e.markMachinesForDeletion(params)
		assert.Len(t, result.FinalMachineToNodeDeleteMap, 1)
	})

	t.Run("mark error moves to cannot delete", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdPatchCommonSeams(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(phaseutil.MarkMachineForDeletion,
			func(_ context.Context, _ client.Client, _ *clusterv1.Machine) error {
				return assertErr("mark failed")
			})
		params := MarkMachinesForDeletionParams{
			Ctx:                          context.Background(),
			Client:                       e.Ctx.Client,
			BKECluster:                   e.Ctx.BKECluster,
			MachineToNodeDeleteMap:       map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			CanNotDeleteMachinesAndNodes: map[string]phaseutil.MachineAndNode{},
			Log:                          e.Ctx.Log,
		}
		result := e.markMachinesForDeletion(params)
		assert.Empty(t, result.FinalMachineToNodeDeleteMap)
		assert.Len(t, result.FinalCanNotDeleteMachinesAndNodes, 1)
	})
}

// ---- processDrainAndMark ----

func TestWorkerDeleteProcessDrainAndMark(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(kube.GetTargetClusterClient,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
				return nil, nil, nil
			})
		patches.ApplyPrivateMethod(e, "drainNodes",
			func(_ *EnsureWorkerDelete, _ DrainNodesParams) DrainNodesResult {
				return DrainNodesResult{
					UpdatedMachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{
						"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}},
					},
				}
			})
		patches.ApplyPrivateMethod(e, "markMachinesForDeletion",
			func(_ *EnsureWorkerDelete, _ MarkMachinesForDeletionParams) MarkMachinesForDeletionResult {
				return MarkMachinesForDeletionResult{
					FinalMachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{
						"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}},
					},
				}
			})
		params := ProcessDrainAndMarkParams{
			Ctx:               context.Background(),
			Client:            e.Ctx.Client,
			BKECluster:        e.Ctx.BKECluster,
			NodeMappingResult: ProcessNodeMachineMappingResult{NodesCount: 1},
			Scope:             &phaseutil.ClusterAPIObjs{MachineDeployment: &clusterv1.MachineDeployment{}},
			Log:               e.Ctx.Log,
		}
		result := e.processDrainAndMark(params)
		assert.Len(t, result.MarkResult.FinalMachineToNodeDeleteMap, 1)
	})
}

// ---- waitForMachinesDelete ----

func TestWorkerDeleteWaitForMachinesDelete(t *testing.T) {
	t.Run("success machine deleted", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		// Empty fake client -> Get returns NotFound
		params := WaitMachinesDeleteParams{
			Ctx:                          context.Background(),
			Client:                       e.Ctx.Client,
			BKECluster:                   e.Ctx.BKECluster,
			MachinesAndNodesToWaitDelete: map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "m1", Namespace: "ns"}}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:                          e.Ctx.Log,
		}
		result := e.waitForMachinesDelete(params)
		require.NoError(t, result.Error)
		assert.Len(t, result.SuccessDeletedNode, 1)
	})

	t.Run("timeout via canceled context", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(time.Sleep, func(_ time.Duration) {})
		// Machine exists in client -> Get returns nil, not deleted
		machine := &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "m1", Namespace: "ns"}}
		e2 := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}, machine)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		params := WaitMachinesDeleteParams{
			Ctx:                          ctx,
			Client:                       e2.Ctx.Client,
			BKECluster:                   e2.Ctx.BKECluster,
			MachinesAndNodesToWaitDelete: map[string]phaseutil.MachineAndNode{"m1": {Machine: machine, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:                          e2.Ctx.Log,
		}
		result := e2.waitForMachinesDelete(params)
		require.Error(t, result.Error)
		assert.Contains(t, result.Error.Error(), "Wait worker node delete failed")
	})

	t.Run("get error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(time.Sleep, func(_ time.Duration) {})
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		// Use fakeClient whose Get returns nil but we patch to return error
		fc := &fakeClient{}
		patches.ApplyMethod(&fakeClient{}, "Get",
			func(_ *fakeClient, _ context.Context, _ client.ObjectKey, _ client.Object, _ ...client.GetOption) error {
				return assertErr("server error")
			})
		params := WaitMachinesDeleteParams{
			Ctx:                          context.Background(),
			Client:                       fc,
			BKECluster:                   e.Ctx.BKECluster,
			MachinesAndNodesToWaitDelete: map[string]phaseutil.MachineAndNode{"m1": {Machine: &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: "m1", Namespace: "ns"}}, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:                          e.Ctx.Log,
		}
		result := e.waitForMachinesDelete(params)
		require.Error(t, result.Error)
	})

	t.Run("machine with draining condition", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(time.Sleep, func(_ time.Duration) {})
		// Machine exists with draining condition false
		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{Name: "m1", Namespace: "ns"},
			Status: clusterv1.MachineStatus{
				Conditions: clusterv1.Conditions{
					{
						Type:   clusterv1.DrainingSucceededCondition,
						Status: corev1.ConditionFalse,
						Reason: clusterv1.DrainingFailedReason,
					},
					{
						Type:   clusterv1.VolumeDetachSucceededCondition,
						Status: corev1.ConditionFalse,
					},
					{
						Type:   clusterv1.MachineNodeHealthyCondition,
						Status: corev1.ConditionFalse,
						Reason: clusterv1.DeletionFailedReason,
					},
				},
			},
		}
		e2 := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}, machine)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		params := WaitMachinesDeleteParams{
			Ctx:                          ctx,
			Client:                       e2.Ctx.Client,
			BKECluster:                   e2.Ctx.BKECluster,
			MachinesAndNodesToWaitDelete: map[string]phaseutil.MachineAndNode{"m1": {Machine: machine, Node: confv1beta1.Node{IP: "10.0.0.1"}}},
			Log:                          e2.Ctx.Log,
		}
		result := e2.waitForMachinesDelete(params)
		require.Error(t, result.Error)
		assert.Contains(t, result.Error.Error(), "Wait worker node delete failed")
	})
}

// ---- waitWorkerDelete ----

func TestWorkerDeleteWaitWorkerDelete(t *testing.T) {
	t.Run("empty map returns nil", func(t *testing.T) {
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		require.NoError(t, e.waitWorkerDelete())
	})

	t.Run("waitForMachinesDelete error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.machinesAndNodesToWaitDelete = map[string]phaseutil.MachineAndNode{
			"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}},
		}
		patches.ApplyPrivateMethod(e, "waitForMachinesDelete",
			func(_ *EnsureWorkerDelete, _ WaitMachinesDeleteParams) WaitMachinesDeleteResult {
				return WaitMachinesDeleteResult{Error: assertErr("wait failed")}
			})
		err := e.waitWorkerDelete()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait failed")
	})

	t.Run("success with processSuccessfulDeletions", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		e.machinesAndNodesToWaitDelete = map[string]phaseutil.MachineAndNode{
			"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}},
		}
		patches.ApplyPrivateMethod(e, "waitForMachinesDelete",
			func(_ *EnsureWorkerDelete, _ WaitMachinesDeleteParams) WaitMachinesDeleteResult {
				return WaitMachinesDeleteResult{
					SuccessDeletedNode: map[string]confv1beta1.Node{"m1": {IP: "10.0.0.1", Hostname: "node1"}},
				}
			})
		patches.ApplyPrivateMethod(e, "processSuccessfulDeletions",
			func(_ *EnsureWorkerDelete, _ ProcessSuccessfulDeletionsParams) error { return nil })
		require.NoError(t, e.waitWorkerDelete())
	})
}

// ---- processSuccessfulDeletions ----

func TestWorkerDeleteProcessSuccessfulDeletions(t *testing.T) {
	t.Run("empty success deleted node returns nil", func(t *testing.T) {
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		err := e.processSuccessfulDeletions(ProcessSuccessfulDeletionsParams{
			Ctx:                context.Background(),
			Client:             e.Ctx.Client,
			BKECluster:         e.Ctx.BKECluster,
			SuccessDeletedNode: map[string]confv1beta1.Node{},
			Log:                e.Ctx.Log,
		})
		require.NoError(t, err)
	})

	t.Run("NewRemoteClientByBKECluster error returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdPatchCommonSeams(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(kube.NewRemoteClientByBKECluster,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (kube.RemoteKubeClient, error) {
				return nil, assertErr("remote client failed")
			})
		err := e.processSuccessfulDeletions(ProcessSuccessfulDeletionsParams{
			Ctx:                context.Background(),
			Client:             e.Ctx.Client,
			BKECluster:         e.Ctx.BKECluster,
			SuccessDeletedNode: map[string]confv1beta1.Node{"m1": {IP: "10.0.0.1", Hostname: "node1"}},
			Log:                e.Ctx.Log,
		})
		require.NoError(t, err) // returns nil on remote client error
	})

	t.Run("success with mockClient", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdPatchCommonSeams(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		// Inject fake clientset via mockClient
		e.mockClient = k8sfake.NewSimpleClientset()
		err := e.processSuccessfulDeletions(ProcessSuccessfulDeletionsParams{
			Ctx:                context.Background(),
			Client:             e.Ctx.Client,
			BKECluster:         e.Ctx.BKECluster,
			SuccessDeletedNode: map[string]confv1beta1.Node{"m1": {IP: "10.0.0.1", Hostname: "node1"}},
			Log:                e.Ctx.Log,
		})
		require.NoError(t, err)
	})
}

// ---- cleanupNodePods ----

func TestWorkerDeleteCleanupNodePods(t *testing.T) {
	t.Run("success with pods", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdPatchCommonSeams(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "ns"}, Spec: corev1.PodSpec{NodeName: "node1"}}
		fakeCs := k8sfake.NewSimpleClientset(pod)
		err := e.cleanupNodePods(context.Background(), fakeCs, e.Ctx.BKECluster, confv1beta1.Node{IP: "10.0.0.1", Hostname: "node1"}, e.Ctx.Log)
		require.NoError(t, err)
	})

	t.Run("list error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdPatchCommonSeams(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		fakeCs := k8sfake.NewSimpleClientset()
		fakeCs.PrependReactor("list", "pods", func(_ k8stesting.Action) (bool, runtime.Object, error) {
			return true, nil, assertErr("list error")
		})
		err := e.cleanupNodePods(context.Background(), fakeCs, e.Ctx.BKECluster, confv1beta1.Node{IP: "10.0.0.1", Hostname: "node1"}, e.Ctx.Log)
		require.Error(t, err)
	})

	t.Run("delete error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdPatchCommonSeams(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "ns"}, Spec: corev1.PodSpec{NodeName: "node1"}}
		fakeCs := k8sfake.NewSimpleClientset(pod)
		fakeCs.PrependReactor("delete", "pods", func(_ k8stesting.Action) (bool, runtime.Object, error) {
			return true, nil, assertErr("delete error")
		})
		err := e.cleanupNodePods(context.Background(), fakeCs, e.Ctx.BKECluster, confv1beta1.Node{IP: "10.0.0.1", Hostname: "node1"}, e.Ctx.Log)
		require.Error(t, err)
	})

	t.Run("no pods returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		wdPatchCommonSeams(patches)
		e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		fakeCs := k8sfake.NewSimpleClientset()
		err := e.cleanupNodePods(context.Background(), fakeCs, e.Ctx.BKECluster, confv1beta1.Node{IP: "10.0.0.1", Hostname: "node1"}, e.Ctx.Log)
		require.NoError(t, err)
	})
}

// ---- kubeClient ----

func TestWorkerDeleteKubeClient(t *testing.T) {
	e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	assert.Nil(t, e.kubeClient())
	e.mockClient = k8sfake.NewSimpleClientset()
	assert.NotNil(t, e.kubeClient())
}

// ---- finalizeDeletion ResumeClusterAPIObj error ----

func TestWorkerDeleteFinalizeDeletionResumeError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newWorkerDeleteCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	md := &clusterv1.MachineDeployment{Spec: clusterv1.MachineDeploymentSpec{Replicas: int32p(3)}}
	scope := &phaseutil.ClusterAPIObjs{MachineDeployment: md}
	patches.ApplyFunc(phaseutil.UpdateMachineDeploymentReplicas,
		func(_ context.Context, _ client.Client, md *clusterv1.MachineDeployment, replicas int32) error {
			md.Spec.Replicas = &replicas
			return nil
		})
	patches.ApplyFunc(phaseutil.ResumeClusterAPIObj,
		func(_ context.Context, _ client.Client, _ client.Object, _ ...string) error {
			return assertErr("resume failed")
		})
	params := FinalizeDeletionParams{
		Ctx:        context.Background(),
		Client:     e.Ctx.Client,
		BKECluster: e.Ctx.BKECluster,
		MarkResult: MarkMachinesForDeletionResult{
			FinalMachineToNodeDeleteMap: map[string]phaseutil.MachineAndNode{
				"m1": {Machine: &clusterv1.Machine{}, Node: confv1beta1.Node{IP: "10.0.0.1"}},
			},
		},
		Scope:           scope,
		CurrentReplicas: int32p(3),
		Log:             e.Ctx.Log,
	}
	result := e.finalizeDeletion(params)
	require.Error(t, result.Error)
	assert.Contains(t, result.Error.Error(), "resume failed")
}
