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

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	"openfuyao.com/colocation-management/pkg/utils"
)

func getTestConfig() *apps.Configuration {
	return &apps.Configuration{
		AggregateStateGCInterval: time.Hour,
	}
}

func cpuToResourceAmount(quantity resource.Quantity) ResourceAmount {
	return CPUAmountFromCores(quantity.AsApproximateFloat64())
}

func memoryToResourceAmount(quantity resource.Quantity) ResourceAmount {
	return MemoryAmountFromBytes(quantity.AsApproximateFloat64())
}

func resourceAmountToCPU(amount ResourceAmount) resource.Quantity {
	return QuantityFromCPUAmount(amount)
}

func resourceAmountToMemory(amount ResourceAmount) resource.Quantity {
	return QuantityFromMemoryAmount(amount)
}

func createTestNode(name string) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}
}

func createTestPod(namespace, name string, phase corev1.PodPhase) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name: "test-container",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("100Mi"),
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{
			Phase: phase,
		},
	}
}

// TestNewNodeState tests the NewNodeState function.
func TestNewNodeState(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	if nodeState.Node.Name != "test-node" {
		t.Errorf("Expected node name 'test-node', got '%s'", nodeState.Node.Name)
	}

	if len(nodeState.pods) != 0 {
		t.Errorf("Expected empty pods map, got %d pods", len(nodeState.pods))
	}

	if len(nodeState.aggregateStateMap) != 0 {
		t.Errorf("Expected empty aggregateStateMap, got %d entries", len(nodeState.aggregateStateMap))
	}
}

// TestNodeStateUpdateNodeAndGetNode tests the UpdateNode and GetNode functions of the NodeState struct.
func TestNodeStateUpdateNodeAndGetNode(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("old-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	newNode := createTestNode("new-node")
	nodeState.UpdateNode(newNode)

	retrievedNode := nodeState.GetNode()
	if retrievedNode.Name != "new-node" {
		t.Errorf("Expected node name 'new-node', got '%s'", retrievedNode.Name)
	}
}

// TestNodeStateStateMapSize tests the StateMapSize function of the NodeState struct.
func TestNodeStateStateMapSize(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	if size := nodeState.StateMapSize(); size != 0 {
		t.Errorf("Expected initial size 0, got %d", size)
	}

	testPod := createTestPod("default", "test-pod", corev1.PodRunning)
	nodeState.AddOrUpdatePodAndContainer(testPod)

	if size := nodeState.StateMapSize(); size != 1 {
		t.Errorf("Expected size 1 after adding pod, got %d", size)
	}
}

// TestNodeStateAddOrUpdatePodAndContainer tests the AddOrUpdatePodAndContainer function of the NodeState struct.
func TestNodeStateAddOrUpdatePodAndContainer(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	testPod := createTestPod("default", "test-pod", corev1.PodRunning)
	nodeState.AddOrUpdatePodAndContainer(testPod)

	nodeState.podLock.RLock()
	if len(nodeState.pods) != 1 {
		t.Errorf("Expected 1 pod, got %d", len(nodeState.pods))
	}
	nodeState.podLock.RUnlock()

	containerID := ContainerID{
		PodID: PodID{
			Namespace: "default",
			PodName:   "test-pod",
		},
		ContainerName: "test-container",
	}
	container := nodeState.GetContainer(containerID)
	if container == nil {
		t.Error("Expected container to be found, got nil")
	}
}

// TestNodeStateAddOrUpdatePod tests the AddOrUpdatePod function of the NodeState struct.
func TestNodeStateAddOrUpdatePod(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	podID := PodID{Namespace: "default", PodName: "test-pod"}
	nodeState.AddOrUpdatePod(podID, corev1.PodRunning)

	nodeState.podLock.RLock()
	if len(nodeState.pods) != 1 {
		t.Errorf("Expected 1 pod, got %d", len(nodeState.pods))
	}

	pod, exists := nodeState.pods[podID]
	if !exists {
		t.Error("Pod should exist in map")
	}
	if pod.Phase != corev1.PodRunning {
		t.Errorf("Expected phase Running, got %s", pod.Phase)
	}
	nodeState.podLock.RUnlock()

	nodeState.AddOrUpdatePod(podID, corev1.PodFailed)

	nodeState.podLock.RLock()
	updatedPod := nodeState.pods[podID]
	if updatedPod.Phase != corev1.PodFailed {
		t.Errorf("Expected phase Failed, got %s", updatedPod.Phase)
	}
	nodeState.podLock.RUnlock()
}

// TestNodeStateDeletePod tests the DeletePod function of the NodeState struct.
func TestNodeStateDeletePod(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	podID := PodID{Namespace: "default", PodName: "test-pod"}
	nodeState.AddOrUpdatePod(podID, corev1.PodRunning)

	nodeState.podLock.RLock()
	if len(nodeState.pods) != 1 {
		t.Errorf("Expected 1 pod before deletion, got %d", len(nodeState.pods))
	}
	nodeState.podLock.RUnlock()

	nodeState.DeletePod(podID)

	nodeState.podLock.RLock()
	if len(nodeState.pods) != 0 {
		t.Errorf("Expected 0 pods after deletion, got %d", len(nodeState.pods))
	}
	nodeState.podLock.RUnlock()
}

// TestNodeStateAddOrUpdateContainer tests the AddOrUpdateContainer function of the NodeState struct.
func TestNodeStateAddOrUpdateContainer(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	podID := PodID{Namespace: "default", PodName: "test-pod"}
	nodeState.AddOrUpdatePod(podID, corev1.PodRunning)

	containerID := ContainerID{
		PodID:         podID,
		ContainerName: "test-container",
	}
	request := Resources{
		ResourceCPU:    cpuToResourceAmount(resource.MustParse("100m")),
		ResourceMemory: memoryToResourceAmount(resource.MustParse("100Mi")),
	}

	err := nodeState.AddOrUpdateContainer(containerID, request)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	container := nodeState.GetContainer(containerID)
	if container == nil {
		t.Error("Container should exist")
	}
}

// TestNodeStateAddSample tests the AddSample function of the NodeState struct.
func TestNodeStateAddSample(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	testPod := createTestPod("default", "test-pod", corev1.PodRunning)
	nodeState.AddOrUpdatePodAndContainer(testPod)

	containerID := ContainerID{
		PodID: PodID{
			Namespace: "default",
			PodName:   "test-pod",
		},
		ContainerName: "test-container",
	}

	cpuUsage := cpuToResourceAmount(resource.MustParse("50m"))

	sample := &ContainerUsageSampleWithKey{
		Container: containerID,
		ContainerUsageSample: ContainerUsageSample{
			MeasureStart: time.Now(),
			Usage:        cpuUsage,
			Resource:     ResourceCPU,
		},
	}

	err := nodeState.AddSample(sample)
	if err != nil {
		t.Errorf("Unexpected error adding sample: %v", err)
	}
}

// TestNodeStateGetTotalResourceEstimation tests the GetTotalResourceEstimation function of the NodeState struct.
func TestNodeStateGetTotalResourceEstimation(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	testPod := createTestPod("default", "test-pod", corev1.PodRunning)
	nodeState.AddOrUpdatePodAndContainer(testPod)

	estimateFunc := func(s *AggregateContainerState) Resources {
		return Resources{
			ResourceCPU:    cpuToResourceAmount(resource.MustParse("200m")),
			ResourceMemory: memoryToResourceAmount(resource.MustParse("200Mi")),
		}
	}

	result := nodeState.GetTotalResourceEstimation(estimateFunc)

	cpuQuantity := resourceAmountToCPU(result[ResourceCPU])
	memoryQuantity := resourceAmountToMemory(result[ResourceMemory])

	if cpuQuantity.MilliValue() != 200 {
		t.Errorf("Expected 200m CPU, got %dm", cpuQuantity.MilliValue())
	}

	expectedMemory := int64(200 * 1024 * 1024)
	if memoryQuantity.Value() != expectedMemory {
		t.Errorf("Expected %d memory bytes, got %d", expectedMemory, memoryQuantity.Value())
	}
}

// TestNodeStateSaveCheckpoints tests the SaveCheckpoints function of the NodeState struct.
func TestNodeStateSaveCheckpoints(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	testPod := createTestPod("default", "test-pod", corev1.PodRunning)
	nodeState.AddOrUpdatePodAndContainer(testPod)

	keys, checkpoints := nodeState.SaveCheckpoints()

	if len(keys) != 1 {
		t.Errorf("Expected 1 checkpoint key, got %d", len(keys))
	}
	if len(checkpoints) != 1 {
		t.Errorf("Expected 1 checkpoint, got %d", len(checkpoints))
	}
	if checkpoints[0] == nil {
		t.Error("Checkpoint should not be nil")
	}
}

// TestNodeStateErrorCases tests error cases of the NodeState struct.
func TestNodeStateErrorCases(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	containerID := ContainerID{
		PodID: PodID{
			Namespace: "default",
			PodName:   "non-existent-pod",
		},
		ContainerName: "test-container",
	}
	request := Resources{
		ResourceCPU: cpuToResourceAmount(resource.MustParse("100m")),
	}

	err := nodeState.AddOrUpdateContainer(containerID, request)
	if err == nil {
		t.Error("Expected error when adding container to non-existent pod")
	}

	cpuUsage := cpuToResourceAmount(resource.MustParse("50m"))

	sample := &ContainerUsageSampleWithKey{
		Container: containerID,
		ContainerUsageSample: ContainerUsageSample{
			MeasureStart: time.Now(),
			Usage:        cpuUsage,
			Resource:     ResourceCPU,
		},
	}

	err = nodeState.AddSample(sample)
	if err == nil {
		t.Error("Expected error when adding sample to non-existent container")
	}
}

// TestNodeStateGarbageCollection tests the garbage collection functionality of the NodeState struct.
func TestNodeStateGarbageCollection(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	runningPod := createTestPod("default", "running-pod", corev1.PodRunning)
	nodeState.AddOrUpdatePodAndContainer(runningPod)

	failedPod := createTestPod("default", "failed-pod", corev1.PodFailed)
	nodeState.AddOrUpdatePodAndContainer(failedPod)

	if nodeState.StateMapSize() != 2 {
		t.Errorf("Expected 2 aggregate states initially, got %d", nodeState.StateMapSize())
	}

	now := time.Now()
	nodeState.garbageCollectAggregateCollectionStates(now)
	nodeState.lastAggregateContainerStateGC = now

	if nodeState.StateMapSize() != 1 {
		t.Errorf("Expected 1 aggregate state after GC, got %d", nodeState.StateMapSize())
	}

	afterManualGC := nodeState.lastAggregateContainerStateGC

	shortTimeLater := now.Add(time.Second)
	nodeState.RateLimitedGarbageCollectAggregateCollectionStates(shortTimeLater)

	if !nodeState.lastAggregateContainerStateGC.Equal(afterManualGC) {
		t.Errorf("Rate-limited GC should not have run before interval elapsed")
	}

	timeAfterInterval := now.Add(nodeState.aggregateStateGCInterval + time.Second)
	nodeState.RateLimitedGarbageCollectAggregateCollectionStates(timeAfterInterval)

	if nodeState.lastAggregateContainerStateGC.Equal(afterManualGC) {
		t.Errorf("Rate-limited GC should have run after interval elapsed")
	}
}

// TestNodeStateGarbageCollectionEmptyStates tests the garbage collection functionality of the NodeState struct
// when there are empty aggregate states.
func TestNodeStateGarbageCollectionEmptyStates(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	runningPod := createTestPod("default", "running-pod", corev1.PodRunning)
	nodeState.AddOrUpdatePodAndContainer(runningPod)

	nodeState.aggregateStateMapLock.Lock()
	emptyKey := MakeAggregateStateKey(PodID{Namespace: "default", PodName: "non-existent-pod"}, "empty-container")
	nodeState.aggregateStateMap[emptyKey] = NewAggregateContainerState()
	nodeState.aggregateStateMapLock.Unlock()

	if nodeState.StateMapSize() != 2 {
		t.Errorf("Expected 2 aggregate states initially, got %d", nodeState.StateMapSize())
	}

	now := time.Now()
	nodeState.garbageCollectAggregateCollectionStates(now)

	if nodeState.StateMapSize() != 1 {
		t.Errorf("Expected 1 aggregate state after GC, got %d", nodeState.StateMapSize())
	}
}

// TestNodeStateGarbageCollectionExpiredStates tests the garbage collection functionality of the NodeState struct
func TestNodeStateGarbageCollectionExpiredStates(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	config := getTestConfig()

	mergeFunc := func(key AggregateStateKey, state *AggregateContainerState) {}
	nodeState := NewNodeState(ctx, node, config, mergeFunc)

	runningPod := createTestPod("default", "running-pod", corev1.PodRunning)
	nodeState.AddOrUpdatePodAndContainer(runningPod)

	nodeState.aggregateStateMapLock.Lock()
	expiredKey := MakeAggregateStateKey(PodID{Namespace: "default", PodName: "expired-pod"}, "expired-container")
	expiredState := NewAggregateContainerState()
	expiredState.CreationTime = time.Now().Add(-9 * 24 * time.Hour)
	nodeState.aggregateStateMap[expiredKey] = expiredState
	nodeState.aggregateStateMapLock.Unlock()

	if nodeState.StateMapSize() != 2 {
		t.Errorf("Expected 2 aggregate states initially, got %d", nodeState.StateMapSize())
	}

	now := time.Now()
	nodeState.garbageCollectAggregateCollectionStates(now)

	if nodeState.StateMapSize() != 1 {
		t.Errorf("Expected 1 aggregate state after GC, got %d", nodeState.StateMapSize())
	}
}

// TestUtilsContributivePod tests the ContributivePod function in the utils package.
func TestUtilsContributivePod(t *testing.T) {
	testCases := []struct {
		phase     corev1.PodPhase
		expected  bool
		phaseName string
	}{
		{corev1.PodRunning, true, "Running"},
		{corev1.PodPending, true, "Pending"},
		{corev1.PodUnknown, true, "Unknown"},
		{corev1.PodFailed, false, "Failed"},
		{corev1.PodSucceeded, false, "Succeeded"},
	}

	for _, tc := range testCases {
		t.Run(tc.phaseName, func(t *testing.T) {
			result := utils.ContributivePod(tc.phase)
			if result != tc.expected {
				t.Errorf("Expected ContributivePod(%s) = %v, got %v", tc.phaseName, tc.expected, result)
			}
		})
	}
}

// TestAddOrUpdatePodAndContainerWithInitContainers exercises the init-container
// branch of AddOrUpdatePodAndContainer that was previously uncovered.
func TestAddOrUpdatePodAndContainerWithInitContainers(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	nodeState := NewNodeState(ctx, node, getTestConfig(),
		func(key AggregateStateKey, state *AggregateContainerState) {})

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-init", Namespace: "default"},
		Spec: corev1.PodSpec{
			InitContainers: []corev1.Container{
				{
					Name: "init-container",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("50m"),
							corev1.ResourceMemory: resource.MustParse("64Mi"),
						},
					},
				},
			},
			Containers: []corev1.Container{
				{
					Name: "main-container",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("128Mi"),
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}

	nodeState.AddOrUpdatePodAndContainer(pod)

	// both init and main containers should be tracked
	if nodeState.StateMapSize() != 2 {
		t.Errorf("Expected 2 aggregate states (init + main), got %d", nodeState.StateMapSize())
	}
	initContainer := nodeState.GetContainer(ContainerID{
		PodID:         PodID{Namespace: "default", PodName: "pod-init"},
		ContainerName: "init-container",
	})
	if initContainer == nil {
		t.Error("Expected init container to be tracked")
	}
	mainContainer := nodeState.GetContainer(ContainerID{
		PodID:         PodID{Namespace: "default", PodName: "pod-init"},
		ContainerName: "main-container",
	})
	if mainContainer == nil {
		t.Error("Expected main container to be tracked")
	}
}

// TestAddOrUpdateContainerExistingContainer covers the branch of
// AddOrUpdateContainer where the container already exists and only its request
// is updated.
func TestAddOrUpdateContainerExistingContainer(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	nodeState := NewNodeState(ctx, node, getTestConfig(),
		func(key AggregateStateKey, state *AggregateContainerState) {})

	podID := PodID{Namespace: "default", PodName: "test-pod"}
	nodeState.AddOrUpdatePod(podID, corev1.PodRunning)
	containerID := ContainerID{PodID: podID, ContainerName: "test-container"}
	originalReq := Resources{
		ResourceCPU:    cpuToResourceAmount(resource.MustParse("100m")),
		ResourceMemory: memoryToResourceAmount(resource.MustParse("100Mi")),
	}
	if err := nodeState.AddOrUpdateContainer(containerID, originalReq); err != nil {
		t.Fatalf("first add failed: %v", err)
	}

	// Re-add: container already exists -> request is updated in place.
	updatedReq := Resources{
		ResourceCPU:    cpuToResourceAmount(resource.MustParse("500m")),
		ResourceMemory: memoryToResourceAmount(resource.MustParse("500Mi")),
	}
	if err := nodeState.AddOrUpdateContainer(containerID, updatedReq); err != nil {
		t.Fatalf("update add failed: %v", err)
	}

	container := nodeState.GetContainer(containerID)
	if container == nil {
		t.Fatal("container should exist after update")
	}
	if container.Request[ResourceCPU] != updatedReq[ResourceCPU] {
		t.Errorf("Expected updated CPU request %d, got %d", updatedReq[ResourceCPU], container.Request[ResourceCPU])
	}

	// Only one aggregate state should exist for the container.
	if nodeState.StateMapSize() != 1 {
		t.Errorf("Expected 1 aggregate state after re-add, got %d", nodeState.StateMapSize())
	}
}

// TestAddSampleContainerNotExistsAndOutOfOrder covers the two error branches
// of NodeState.AddSample that were previously uncovered.
func TestAddSampleContainerNotExistsAndOutOfOrder(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	nodeState := NewNodeState(ctx, node, getTestConfig(),
		func(key AggregateStateKey, state *AggregateContainerState) {})

	podID := PodID{Namespace: "default", PodName: "test-pod"}
	nodeState.AddOrUpdatePod(podID, corev1.PodRunning)

	// Sample for a container that does not exist -> NewKeyError(container).
	missingContainerSample := &ContainerUsageSampleWithKey{
		Container: ContainerID{PodID: podID, ContainerName: "missing-container"},
		ContainerUsageSample: ContainerUsageSample{
			MeasureStart: time.Now(),
			Usage:        cpuToResourceAmount(resource.MustParse("50m")),
			Resource:     ResourceCPU,
		},
	}
	if err := nodeState.AddSample(missingContainerSample); err == nil {
		t.Error("Expected error when adding sample to non-existent container")
	}

	// Create the container, add a CPU sample, then try an out-of-order sample.
	containerID := ContainerID{PodID: podID, ContainerName: "real-container"}
	if err := nodeState.AddOrUpdateContainer(containerID, Resources{
		ResourceCPU: cpuToResourceAmount(resource.MustParse("100m")),
	}); err != nil {
		t.Fatalf("add container failed: %v", err)
	}

	baseTime := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)
	first := &ContainerUsageSampleWithKey{
		Container: containerID,
		ContainerUsageSample: ContainerUsageSample{
			MeasureStart: baseTime,
			Usage:        cpuToResourceAmount(resource.MustParse("50m")),
			Resource:     ResourceCPU,
		},
	}
	if err := nodeState.AddSample(first); err != nil {
		t.Fatalf("first sample failed: %v", err)
	}

	// Out-of-order (earlier timestamp) -> discarded error.
	outOfOrder := &ContainerUsageSampleWithKey{
		Container: containerID,
		ContainerUsageSample: ContainerUsageSample{
			MeasureStart: baseTime.Add(-time.Minute), // earlier
			Usage:        cpuToResourceAmount(resource.MustParse("50m")),
			Resource:     ResourceCPU,
		},
	}
	if err := nodeState.AddSample(outOfOrder); err == nil {
		t.Error("Expected error for out-of-order / discarded sample")
	}
}

// TestGarbageCollectionNonContributiveExpired covers the remaining GC branches:
// an empty & non-contributive aggregate state, and an expired-but-present state.
func TestGarbageCollectionNonContributiveExpired(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	nodeState := NewNodeState(ctx, node, getTestConfig(),
		func(key AggregateStateKey, state *AggregateContainerState) {})

	// A failed pod is non-contributive. Its aggregate state is empty until a
	// sample is added, so it should be removed as "empty and not contributive".
	failedPod := createTestPod("default", "failed-pod", corev1.PodFailed)
	nodeState.AddOrUpdatePodAndContainer(failedPod)

	// An aggregate state tied to a contributive (running) pod but with an
	// expired last sample is removed via the isExpired branch.
	runningPod := createTestPod("default", "running-pod", corev1.PodRunning)
	nodeState.AddOrUpdatePodAndContainer(runningPod)
	runningKey := MakeAggregateStateKey(PodID{Namespace: "default", PodName: "running-pod"}, "test-container")
	nodeState.aggregateStateMapLock.Lock()
	if state, ok := nodeState.aggregateStateMap[runningKey]; ok {
		// Make it look expired: last sample > memory aggregation window ago.
		state.LastSampleStart = time.Now().Add(-9 * 24 * time.Hour)
		state.TotalSamplesCount = 1
	}
	nodeState.aggregateStateMapLock.Unlock()

	if nodeState.StateMapSize() != 2 {
		t.Fatalf("Expected 2 aggregate states before GC, got %d", nodeState.StateMapSize())
	}

	nodeState.garbageCollectAggregateCollectionStates(time.Now())

	// Both should have been collected (failed+empty, and running+expired).
	if nodeState.StateMapSize() != 0 {
		t.Errorf("Expected 0 aggregate states after GC, got %d", nodeState.StateMapSize())
	}
}

// TestGetContainerMissing covers the branch of GetContainer where the pod
// exists but the requested container does not (returns nil), plus the pod
// not found path.
func TestGetContainerMissing(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	nodeState := NewNodeState(ctx, node, getTestConfig(),
		func(key AggregateStateKey, state *AggregateContainerState) {})

	podID := PodID{Namespace: "default", PodName: "test-pod"}
	nodeState.AddOrUpdatePod(podID, corev1.PodRunning)
	nodeState.AddOrUpdateContainer(ContainerID{PodID: podID, ContainerName: "real"}, Resources{})

	// Pod exists, container does not -> nil.
	missing := nodeState.GetContainer(ContainerID{PodID: podID, ContainerName: "missing-container"})
	assert.Nil(t, missing)

	// Pod does not exist -> nil.
	missingPod := nodeState.GetContainer(ContainerID{
		PodID:         PodID{Namespace: "default", PodName: "no-such-pod"},
		ContainerName: "any",
	})
	assert.Nil(t, missingPod)
}

// TestAggregateStateKeyForContainerIDPanic covers the panic path of
// aggregateStateKeyForContainerID when the pod is not present.
func TestAggregateStateKeyForContainerIDPanic(t *testing.T) {
	ctx := context.Background()
	node := createTestNode("test-node")
	nodeState := NewNodeState(ctx, node, getTestConfig(),
		func(key AggregateStateKey, state *AggregateContainerState) {})

	containerID := ContainerID{
		PodID:         PodID{Namespace: "missing", PodName: "missing-pod"},
		ContainerName: "missing-container",
	}
	assert.Panics(t, func() {
		nodeState.aggregateStateKeyForContainerID(containerID)
	})
}
