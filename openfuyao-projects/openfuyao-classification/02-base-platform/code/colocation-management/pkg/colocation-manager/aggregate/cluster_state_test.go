/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-09-08
 */

package aggregate

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	colocationv1 "openfuyao.com/colocation-management/pkg/apis/v1"
)

func TestNewClusterState(t *testing.T) {
	ctx := context.Background()
	config := &apps.Configuration{}

	cluster := NewClusterState(ctx, config)

	assert.NotNil(t, cluster)
	assert.Equal(t, ctx, cluster.ctx)
	assert.Equal(t, config, cluster.config)
	assert.NotNil(t, cluster.nodes)
	assert.NotNil(t, cluster.pods)
	assert.NotNil(t, cluster.initialAggregateStateMap)
	assert.NotNil(t, cluster.namespacedNameToCheckpoints)
	assert.NotNil(t, cluster.aggregateKeyToCheckpoints)
}

func TestRegisterNodeCallback(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})

	addCalled := false
	deleteCalled := false

	addFunc := func(node string) { addCalled = true }
	deleteFunc := func(node string) { deleteCalled = true }

	cluster.RegisterNodeCallback(addFunc, deleteFunc)

	cluster.addNodeCallback("test-node")
	cluster.deleteNodeCallback("test-node")

	assert.True(t, addCalled)
	assert.True(t, deleteCalled)
}

func TestAddOrUpdateNode(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
	}

	addCallbackCalled := false
	cluster.addNodeCallback = func(node string) {
		addCallbackCalled = true
		assert.Equal(t, "node-1", node)
	}

	nodeState := cluster.AddOrUpdateNode(node)

	assert.NotNil(t, nodeState)
	assert.True(t, addCallbackCalled)
	assert.True(t, cluster.IsNodeExisted("node-1"))

	node.Labels = map[string]string{"test": "label"}
	nodeState = cluster.AddOrUpdateNode(node)

	assert.NotNil(t, nodeState)
	assert.True(t, cluster.IsNodeExisted("node-1"))
}

func TestDeleteNode(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}
	cluster.AddOrUpdateNode(node)

	deleteCallbackCalled := false
	cluster.deleteNodeCallback = func(node string) {
		deleteCallbackCalled = true
		assert.Equal(t, "node-1", node)
	}

	cluster.DeleteNode("node-1")

	assert.True(t, deleteCallbackCalled)
	assert.False(t, cluster.IsNodeExisted("node-1"))
}

func TestFindNodeState(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}
	cluster.AddOrUpdateNode(node)

	nodeState := cluster.FindNodeState("node-1")
	assert.NotNil(t, nodeState)

	nodeState = cluster.FindNodeState("nonexistent")
	assert.Nil(t, nodeState)
}

func TestIsNodeExisted(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}
	cluster.AddOrUpdateNode(node)

	assert.True(t, cluster.IsNodeExisted("node-1"))
	assert.False(t, cluster.IsNodeExisted("nonexistent"))
}

func TestGetAllNodeStates(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})

	nodes := []*corev1.Node{
		{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "node-2"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "node-3"}},
	}

	for _, node := range nodes {
		cluster.AddOrUpdateNode(node)
	}

	allNodes := cluster.GetAllNodeStates()

	assert.Equal(t, 3, len(allNodes))
	for _, nodeName := range []string{"node-1", "node-2", "node-3"} {
		assert.Contains(t, allNodes, nodeName)
		assert.NotNil(t, allNodes[nodeName])
	}
}

func TestAddPodAndContainer(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}
	cluster.AddOrUpdateNode(node)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
		},
		Spec: corev1.PodSpec{
			NodeName: "node-1",
			Containers: []corev1.Container{
				{
					Name: "test-container",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("128Mi"),
						},
					},
				},
			},
		},
	}

	err := cluster.AddPodAndContainer(pod)
	assert.NoError(t, err)

	emptyNodePod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Spec:       corev1.PodSpec{NodeName: ""},
	}
	err = cluster.AddPodAndContainer(emptyNodePod)
	assert.Error(t, err)

	nonexistentNodePod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Spec:       corev1.PodSpec{NodeName: "nonexistent"},
	}
	err = cluster.AddPodAndContainer(nonexistentNodePod)
	assert.Error(t, err)
}

func TestCalculateRequestedResources(t *testing.T) {
	container := corev1.Container{
		Name: "test-container",
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("250m"),
				corev1.ResourceMemory: resource.MustParse("512Mi"),
			},
		},
	}

	resources := calculateRequestedResources(container)

	assert.Equal(t, ResourceAmount(250), resources[ResourceCPU])
	assert.Equal(t, ResourceAmount(512*1024*1024), resources[ResourceMemory])
}

func TestDeleteNotExistedNodes(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})

	nodes := []*corev1.Node{
		{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "node-2"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "node-3"}},
	}

	for _, node := range nodes {
		cluster.AddOrUpdateNode(node)
	}

	curNodes := map[string]*corev1.Node{
		"node-1": nodes[0],
		"node-2": nodes[1],
	}

	deleteCallbackCount := 0
	cluster.deleteNodeCallback = func(node string) {
		deleteCallbackCount++
		assert.Equal(t, "node-3", node)
	}

	cluster.DeleteNotExistedNodes(curNodes)

	assert.Equal(t, 1, deleteCallbackCount)
	assert.True(t, cluster.IsNodeExisted("node-1"))
	assert.True(t, cluster.IsNodeExisted("node-2"))
	assert.False(t, cluster.IsNodeExisted("node-3"))
}

func TestDeleteNotExistedPods(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}
	cluster.AddOrUpdateNode(node)

	pods := []*corev1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "default"},
			Spec:       corev1.PodSpec{NodeName: "node-1"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-2", Namespace: "default"},
			Spec:       corev1.PodSpec{NodeName: "node-1"},
		},
	}

	for _, pod := range pods {
		err := cluster.AddPodAndContainer(pod)
		assert.NoError(t, err)
	}

	curPods := map[PodID]*corev1.Pod{
		{Namespace: "default", PodName: "pod-1"}: pods[0],
	}

	cluster.DeleteNotExistedPods(curPods)

	nodeState := cluster.FindNodeState("node-1")
	assert.NotNil(t, nodeState)
}

func TestDeletePod(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}
	cluster.AddOrUpdateNode(node)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Spec:       corev1.PodSpec{NodeName: "node-1"},
	}

	err := cluster.AddPodAndContainer(pod)
	assert.NoError(t, err)

	podID := PodID{Namespace: "default", PodName: "test-pod"}
	cluster.DeletePod(podID)

}

func TestSaveCheckpoints(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})

	nodes := []*corev1.Node{
		{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "node-2"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "node-3"}},
	}

	for i, node := range nodes {
		nodeState := cluster.AddOrUpdateNode(node)
		nodeState.checkpointWritten = time.Now().Add(time.Duration(i) * time.Minute)
	}

	keys, checkpoints := cluster.SaveCheckpoints(2)
	assert.NotNil(t, keys)
	assert.NotNil(t, checkpoints)

	keys, checkpoints = cluster.SaveCheckpoints(5)
	assert.NotNil(t, keys)
	assert.NotNil(t, checkpoints)
}

func TestAggregateStateOperations(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})

	key := aggregateStateKey{
		namespace:     "default",
		podName:       "test-pod",
		containerName: "test-container",
	}
	state := &AggregateContainerState{}

	cluster.AddOrUpdateInitialAggregateState(key, state)
	assert.Equal(t, 1, cluster.InitialAggregateStateSize())

	count := cluster.CleanupInitialAggregateState()
	assert.Equal(t, 1, count)
	assert.Equal(t, 0, cluster.InitialAggregateStateSize())
}

func TestCheckpointOperations(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})

	checkpoint := &colocationv1.ContainerCheckpoint{
		Spec: colocationv1.ContainerCheckpointSpec{
			Namespace:     "default",
			PodName:       "test-pod",
			ContainerName: "test-container",
		},
	}

	namespacedName := types.NamespacedName{Namespace: "default", Name: "test-pod-test-container"}

	cluster.AddOrUpdateCheckpoint(namespacedName, checkpoint)

	aggregateKey := MakeAggregateStateKey(PodID{Namespace: "default", PodName: "test-pod"}, "test-container")
	retrieved := cluster.GetCheckpoint(aggregateKey)
	assert.NotNil(t, retrieved)

	allCheckpoints := cluster.GetAllCheckpoint()
	assert.Equal(t, 1, len(allCheckpoints))

	cluster.DeleteCheckpoint(namespacedName)
	retrieved = cluster.GetCheckpoint(aggregateKey)
	assert.Nil(t, retrieved)
}

func BenchmarkAddOrUpdateNode(b *testing.B) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cluster.AddOrUpdateNode(node)
	}
}

func BenchmarkAddPodAndContainer(b *testing.B) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}
	cluster.AddOrUpdateNode(node)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Spec:       corev1.PodSpec{NodeName: "node-1"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cluster.AddPodAndContainer(pod)
	}
}

func TestMergeInitialAggregateContainerState(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})
	key := aggregateStateKey{
		namespace:     "default",
		podName:       "test-pod",
		containerName: "test-container",
	}
	initialState := NewAggregateContainerState()
	targetState := NewAggregateContainerState()

	cluster.AddOrUpdateInitialAggregateState(key, initialState)
	cluster.MergeInitialAggregateContainerState(key, targetState)

	assert.Equal(t, 0, cluster.InitialAggregateStateSize())
}

// TestDeletePodNotFound covers the early-return branch of ClusterState.DeletePod
// when the pod is not tracked in the cluster.
func TestDeletePodNotFound(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})

	// Deleting a pod that was never added should be a no-op (no panic, no error).
	missingPodID := PodID{Namespace: "default", PodName: "missing-pod"}
	assert.NotPanics(t, func() {
		cluster.DeletePod(missingPodID)
	})

	// When the pod is tracked but the underlying node has been removed, the
	// nodeState-ok branch's nil guard is exercised (still no panic).
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}
	cluster.AddOrUpdateNode(node)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "p", Namespace: "default"},
		Spec:       corev1.PodSpec{NodeName: "node-1"},
	}
	assert.NoError(t, cluster.AddPodAndContainer(pod))
	cluster.DeleteNode("node-1") // node removed but pod still in cluster.pods
	assert.NotPanics(t, func() {
		cluster.DeletePod(PodID{Namespace: "default", PodName: "p"})
	})
}

// TestDeleteCheckpointNotFound covers the early-return branch of
// ClusterState.DeleteCheckpoint when the checkpoint is not present.
func TestDeleteCheckpointNotFound(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})

	missingName := types.NamespacedName{Namespace: "default", Name: "missing"}
	assert.NotPanics(t, func() {
		cluster.DeleteCheckpoint(missingName)
	})

	// Sanity: nothing was added.
	assert.Equal(t, 0, len(cluster.GetAllCheckpoint()))
}

// TestGetCheckpointNotFound covers the not-found branch of GetCheckpoint.
func TestGetCheckpointNotFound(t *testing.T) {
	cluster := NewClusterState(context.Background(), &apps.Configuration{})
	key := MakeAggregateStateKey(PodID{Namespace: "default", PodName: "missing"}, "missing-container")
	assert.Nil(t, cluster.GetCheckpoint(key))
}
