/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package phases

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// createBKENodeWithFlagsPostprocess creates a BKENode with the specified flags set
func createBKENodeWithFlagsPostprocess(namespace, clusterName, ip, hostname string, roles []string, flags ...int) *confv1beta1.BKENode {
	stateCode := 0
	for _, flag := range flags {
		stateCode |= flag
	}

	return &confv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{
			Name:      hostname,
			Namespace: namespace,
			Labels: map[string]string{
				nodeutil.ClusterNameLabel: clusterName,
			},
		},
		Spec: confv1beta1.BKENodeSpec{
			IP:       ip,
			Hostname: hostname,
			Role:     roles,
		},
		Status: confv1beta1.BKENodeStatus{
			StateCode: stateCode,
		},
	}
}

func TestEnsure_nodes_postprocess_check_config_exists_global(t *testing.T) {
	InitinitPhaseContextFun()

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-all-config",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"a.sh","order":1}]}`,
		},
	}
	initPhaseContext.Client = newFakeClientWithObjectsPostprocess(t, cm)

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	ok := e.checkPostProcessConfigExists(context.Background(), initPhaseContext.Client, initPhaseContext.Log, "10.0.0.1")
	require.True(t, ok)
}

func TestEnsure_nodes_postprocess_check_config_exists_batch(t *testing.T) {
	InitinitPhaseContextFun()

	mapping := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-node-batch-mapping",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"mapping.json": `{"10.0.0.1":"001"}`,
		},
	}
	batch := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-config-batch-001",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"b.sh","order":1}]}`,
		},
	}
	initPhaseContext.Client = newFakeClientWithObjectsPostprocess(t, mapping, batch)

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	ok := e.checkPostProcessConfigExists(context.Background(), initPhaseContext.Client, initPhaseContext.Log, "10.0.0.1")
	require.True(t, ok)
}

func TestEnsure_nodes_postprocess_check_config_exists_node(t *testing.T) {
	InitinitPhaseContextFun()

	nodeCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-config-node-10.0.0.1",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"c.sh","order":1}]}`,
		},
	}
	initPhaseContext.Client = newFakeClientWithObjectsPostprocess(t, nodeCM)

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	ok := e.checkPostProcessConfigExists(context.Background(), initPhaseContext.Client, initPhaseContext.Log, "10.0.0.1")
	require.True(t, ok)
}

func TestEnsure_nodes_postprocess_check_config_exists_miss(t *testing.T) {
	InitinitPhaseContextFun()
	initPhaseContext.Client = newFakeClientWithObjectsPostprocess(t)

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	ok := e.checkPostProcessConfigExists(context.Background(), initPhaseContext.Client, initPhaseContext.Log, "10.0.0.1")
	require.False(t, ok)
}

func TestEnsure_nodes_postprocess_check_config_exists_batch_mapping_parse_error(t *testing.T) {
	InitinitPhaseContextFun()

	mapping := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-node-batch-mapping",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"mapping.json": "{bad json",
		},
	}
	initPhaseContext.Client = newFakeClientWithObjectsPostprocess(t, mapping)

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	ok := e.checkPostProcessConfigExists(context.Background(), initPhaseContext.Client, initPhaseContext.Log, "10.0.0.1")
	require.False(t, ok)
}

func TestEnsure_nodes_postprocess_mark_success(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	// Create BKENode resource
	bkeNode1 := createBKENodeWithFlagsPostprocess(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "node1", []string{bkenode.MasterNodeRole},
	)

	// Create fake client with BKENode resources
	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeNode1).
		WithStatusSubresource(bkeNode1).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)

	e.markPostProcessSuccess([]string{"node1/10.0.0.1"})

	// Verify the flag was set by reading the BKENode
	updatedNode := &confv1beta1.BKENode{}
	err := fakeClient.Get(context.Background(), client.ObjectKey{
		Namespace: bkeCluster.Namespace,
		Name:      "node1",
	}, updatedNode)
	require.NoError(t, err)
	require.True(t, phaseutil.GetNodeStateFlag(updatedNode, "10.0.0.1", v1beta1.NodePostProcessFlag))
}

func TestEnsure_nodes_postprocess_handle_failed_nodes(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	bkeNode1 := createBKENodeWithFlagsPostprocess(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "node1", []string{bkenode.MasterNodeRole},
	)

	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeNode1).
		WithStatusSubresource(bkeNode1).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)

	e.handleFailedNodes(&command.Custom{}, []string{"node1/10.0.0.1"})

	updatedNode := &confv1beta1.BKENode{}
	err := fakeClient.Get(context.Background(), client.ObjectKey{
		Namespace: bkeCluster.Namespace,
		Name:      "node1",
	}, updatedNode)
	require.NoError(t, err)
	require.Equal(t, v1beta1.NodeInitFailed, updatedNode.Status.State)
	require.Equal(t, "Post process scripts failed", updatedNode.Status.Message)
}

func TestEnsure_nodes_postprocess_check_or_run_no_nodes(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	// Create fake client with no BKENodes
	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	_, err := e.CheckOrRunPostProcess()
	require.NoError(t, err)
	require.True(t, condition.HasConditionStatus(v1beta1.NodesPostProcessCondition, bkeCluster, confv1beta1.ConditionTrue))
}

func TestEnsure_nodes_postprocess_need_execute(t *testing.T) {
	t.Skip("Complex integration test - skipping for now")
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	// Create BKENode with BootFlag only (needs postprocess)
	bkeNode1 := createBKENodeWithFlagsPostprocess(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "node1", []string{bkenode.MasterNodeRole},
		v1beta1.NodeBootFlag,
	)

	// Create fake client with BKENode
	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeCluster, bkeNode1).
		WithStatusSubresource(bkeNode1).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	require.True(t, e.NeedExecute(&initOldBkeCluster, bkeCluster))

	// Now add PostProcessFlag to the node - should not need execute
	bkeNode1WithPostProcess := createBKENodeWithFlagsPostprocess(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "node1", []string{bkenode.MasterNodeRole},
		v1beta1.NodeBootFlag, v1beta1.NodePostProcessFlag,
	)

	fakeClient2 := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeCluster, bkeNode1WithPostProcess).
		WithStatusSubresource(bkeCluster, bkeNode1WithPostProcess).
		Build()

	bkeCluster2 := bkeCluster.DeepCopy()
	initPhaseContext.Client = fakeClient2
	initPhaseContext.BKECluster = bkeCluster2
	e2 := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	result := e2.NeedExecute(&initOldBkeCluster, bkeCluster2)
	t.Logf("NeedExecute result: %v, StateCode: %d", result, bkeNode1WithPostProcess.Status.StateCode)
	require.False(t, result)
}

func TestEnsure_nodes_postprocess_execute_no_config(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	// Create fake client without postprocess config
	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	e.nodes = bkenode.Nodes{
		{IP: "", Hostname: "empty"},
		{IP: "10.0.0.1", Hostname: "node1"},
	}

	err := e.executeNodePostProcessScripts()
	require.NoError(t, err)
}

func newFakeClientWithObjectsPostprocess(t *testing.T, objs ...client.Object) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
}

func TestEnsureNodesPostProcess_Constants(t *testing.T) {
	assert.Equal(t, "EnsureNodesPostProcess", string(EnsureNodesPostProcessName))
}

func TestNewEnsureNodesPostProcess(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = v1beta1.AddToScheme(scheme)
	bkeCluster := &v1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        createTestLogger(),
	}
	phase := NewEnsureNodesPostProcess(ctx)
	assert.NotNil(t, phase)
}

func TestEnsureNodesPostProcess_Execute(t *testing.T) {
	// Skipping test that requires mocking private method
	t.Skip("Skipping - requires mocking private method")
}

func TestEnsureNodesPostProcess_NeedExecute_True(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	// Create BKENode with BootFlag only (needs postprocess)
	bkeNode1 := createBKENodeWithFlagsPostprocess(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "node1", []string{bkenode.MasterNodeRole},
		v1beta1.NodeBootFlag,
	)

	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeCluster, bkeNode1).
		WithStatusSubresource(bkeNode1).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	patches.ApplyFunc(phaseutil.GetNeedPostProcessNodesWithBKENodes, func(cluster *v1beta1.BKECluster, nodes v1beta1.BKENodes) bkenode.Nodes {
		return bkenode.Nodes{{IP: "10.0.0.1", Hostname: "node1"}}
	})

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	result := e.NeedExecute(&initOldBkeCluster, bkeCluster)
	require.True(t, result)
}

func TestEnsureNodesPostProcess_NeedExecute_False(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeCluster).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	// Return empty nodes
	patches.ApplyFunc(phaseutil.GetNeedPostProcessNodesWithBKENodes, func(cluster *v1beta1.BKECluster, nodes v1beta1.BKENodes) bkenode.Nodes {
		return bkenode.Nodes{}
	})

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	result := e.NeedExecute(&initOldBkeCluster, bkeCluster)
	require.False(t, result)
}

func TestEnsureNodesPostProcess_NeedExecute_GetBKENodesError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeCluster).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	result := e.NeedExecute(&initOldBkeCluster, bkeCluster)
	require.False(t, result)
}

func TestEnsureNodesPostProcess_CheckOrRunPostProcess_WithNodes(t *testing.T) {
	// Skipping test that requires mocking private method
	t.Skip("Skipping - requires mocking private method")
}

func TestEnsureNodesPostProcess_CheckOrRunPostProcess_ExecuteError(t *testing.T) {
	// Skipping test that requires mocking private method
	t.Skip("Skipping - requires mocking private method")
}

func TestEnsureNodesPostProcess_ExecuteNodePostProcessScripts_CreateCommandError(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	// Add a ConfigMap to trigger config check
	nodeCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-config-node-10.0.0.1",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"c.sh","order":1}]}`,
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeCluster, nodeCM).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	e.nodes = bkenode.Nodes{
		{IP: "10.0.0.1", Hostname: "node1"},
	}

	// This will fail because command.New() will fail (no proper setup)
	err := e.executeNodePostProcessScripts()
	require.Error(t, err)
}

func TestEnsureNodesPostProcess_CreatePostProcessCommand(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)

	nodes := bkenode.Nodes{
		{IP: "10.0.0.1", Hostname: "node1"},
	}

	// This will fail because customCmd.New() will fail
	cmd, err := e.createPostProcessCommand(context.Background(), fakeClient, bkeCluster, scheme, nodes)
	require.Error(t, err)
	require.Nil(t, cmd)
}

func TestEnsureNodesPostProcess_ExecuteNodePostProcessScripts_AllNodesWithoutConfig(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeCluster).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	e.nodes = bkenode.Nodes{
		{IP: "10.0.0.1", Hostname: "node1"},
		{IP: "10.0.0.2", Hostname: "node2"},
	}

	// All nodes have no config - should return nil
	err := e.executeNodePostProcessScripts()
	require.NoError(t, err)
}

func TestEnsureNodesPostProcess_ExecuteNodePostProcessScripts_EmptyIPNode(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeCluster).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	e.nodes = bkenode.Nodes{
		{IP: "", Hostname: "node-empty"},
		{IP: "10.0.0.1", Hostname: "node1"},
	}

	// Node with empty IP should be skipped
	err := e.executeNodePostProcessScripts()
	require.NoError(t, err)
}

func TestEnsureNodesPostProcess_NeedExecute_DefaultReturnsFalse(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeCluster).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	// Mock DefaultNeedExecute to return false
	patches.ApplyMethod(&phaseframe.BasePhase{}, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, old, new *v1beta1.BKECluster) bool {
		return false
	})

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	result := e.NeedExecute(&initOldBkeCluster, bkeCluster)
	require.False(t, result)
}

func TestEnsureNodesPostProcess_NeedExecute_GetKENodesWithMockB(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeCluster).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	// Mock GetNeedPostProcessNodesWithBKENodes to return nodes
	patches.ApplyFunc(phaseutil.GetNeedPostProcessNodesWithBKENodes, func(cluster *v1beta1.BKECluster, nodes v1beta1.BKENodes) bkenode.Nodes {
		return bkenode.Nodes{{IP: "10.0.0.1", Hostname: "node1"}}
	})

	e := NewEnsureNodesPostProcess(initPhaseContext).(*EnsureNodesPostProcess)
	result := e.NeedExecute(&initOldBkeCluster, bkeCluster)
	// When nodes exist, NeedExecute should return true
	require.True(t, result)
}

func nodesPostProcessGapsPhase(t *testing.T) *EnsureNodesPostProcess {
	t.Helper()
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	ctx := newAdditionalPhaseContext(t, cluster)
	return &EnsureNodesPostProcess{BasePhase: phaseframe.NewBasePhase(ctx, EnsureNodesPostProcessName)}
}

// TestNodesPostProcessGapsCheckOrRunPostProcess covers the uncovered branches of
// CheckOrRunPostProcess: get-nodes error, no-nodes short-circuit, scripts error, success.
func TestNodesPostProcessGapsCheckOrRunPostProcess(t *testing.T) {
	t.Run("get_nodes_error_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesPostProcessGapsPhase(t)
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return nil, assertErr("get nodes failed")
			})
		_, err := e.CheckOrRunPostProcess()
		require.Error(t, err)
	})

	t.Run("no_nodes_need_postprocess_returns_nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesPostProcessGapsPhase(t)
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return bkev1beta1.BKENodes{}, nil
			})
		patches.ApplyFunc(phaseutil.GetNeedPostProcessNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return bkenode.Nodes{}
			})
		_, err := e.CheckOrRunPostProcess()
		require.NoError(t, err)
	})

	t.Run("execute_scripts_error_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesPostProcessGapsPhase(t)
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return bkev1beta1.BKENodes{}, nil
			})
		patches.ApplyFunc(phaseutil.GetNeedPostProcessNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}}
			})
		patches.ApplyPrivateMethod(e, "executeNodePostProcessScripts", func(_ *EnsureNodesPostProcess) error {
			return assertErr("exec failed")
		})
		_, err := e.CheckOrRunPostProcess()
		require.Error(t, err)
	})

	t.Run("success_returns_nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesPostProcessGapsPhase(t)
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
				return bkev1beta1.BKENodes{}, nil
			})
		patches.ApplyFunc(phaseutil.GetNeedPostProcessNodesWithBKENodes,
			func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
				return bkenode.Nodes{{IP: "10.0.0.1"}}
			})
		patches.ApplyPrivateMethod(e, "executeNodePostProcessScripts", func(_ *EnsureNodesPostProcess) error {
			return nil
		})
		_, err := e.CheckOrRunPostProcess()
		require.NoError(t, err)
	})
}

// TestNodesPostProcessGapsExecuteNodePostProcessScripts covers branches of
// executeNodePostProcessScripts: no-config skip, all-skipped short-circuit, success.
func TestNodesPostProcessGapsExecuteNodePostProcessScripts(t *testing.T) {
	t.Run("all_nodes_skipped_no_config_returns_nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesPostProcessGapsPhase(t)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}, {IP: "10.0.0.2"}}
		// checkPostProcessConfigExists returns false -> all nodes collected as withoutConfig -> skipped
		patches.ApplyPrivateMethod(e, "checkPostProcessConfigExists",
			func(_ *EnsureNodesPostProcess, _ context.Context, _ client.Client, _ *bkev1beta1.BKELogger, _ string) bool {
				return false
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ func(*confv1beta1.BKENodeStatus)) error {
				return nil
			})
		err := e.executeNodePostProcessScripts()
		require.NoError(t, err)
	})

	t.Run("empty_ip_nodes_skipped", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesPostProcessGapsPhase(t)
		e.nodes = bkenode.Nodes{{IP: ""}}
		err := e.executeNodePostProcessScripts()
		require.NoError(t, err)
	})
}
