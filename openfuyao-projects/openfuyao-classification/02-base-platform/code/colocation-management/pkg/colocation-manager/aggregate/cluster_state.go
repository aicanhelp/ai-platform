/*
 *
 *  * Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

// Package aggregate provides the cluster state management.
package aggregate

import (
	"context"
	"fmt"
	"sort"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	colocationv1 "openfuyao.com/colocation-management/pkg/apis/v1"
	"openfuyao.com/colocation-management/pkg/common"
)

// ClusterState is the state of the cluster.
type ClusterState struct {
	ctx    context.Context
	config *apps.Configuration

	// nodes in the cluster.
	nodes              map[string]*NodeState
	pods               map[PodID]string // key is podID, value is node name
	nodePodLock        sync.RWMutex
	addNodeCallback    func(node string)
	deleteNodeCallback func(node string)

	initialAggregateStateMap     AggregateContainerStatesMap
	initialAggregateStateMapLock sync.RWMutex

	namespacedNameToCheckpoints map[types.NamespacedName]*colocationv1.ContainerCheckpoint
	aggregateKeyToCheckpoints   map[AggregateStateKey]*colocationv1.ContainerCheckpoint
	checkpointLock              sync.RWMutex
}

// NewClusterState creates a new ClusterState.
func NewClusterState(ctx context.Context, config *apps.Configuration) *ClusterState {
	return &ClusterState{
		ctx:                         ctx,
		config:                      config,
		nodes:                       make(map[string]*NodeState),
		pods:                        make(map[PodID]string),
		nodePodLock:                 sync.RWMutex{},
		initialAggregateStateMap:    make(AggregateContainerStatesMap),
		namespacedNameToCheckpoints: make(map[types.NamespacedName]*colocationv1.ContainerCheckpoint),
		aggregateKeyToCheckpoints:   make(map[AggregateStateKey]*colocationv1.ContainerCheckpoint),
	}
}

// RegisterNodeCallback registers the callback function for node addition and deletion.
func (cluster *ClusterState) RegisterNodeCallback(add func(node string), delete func(node string)) {
	cluster.addNodeCallback = add
	cluster.deleteNodeCallback = delete
}

// AddOrUpdateNode adds or updates a node in the cluster state.
func (cluster *ClusterState) AddOrUpdateNode(node *corev1.Node) *NodeState {
	cluster.nodePodLock.Lock()
	defer cluster.nodePodLock.Unlock()

	state, ok := cluster.nodes[node.Name]
	if !ok {
		state = NewNodeState(cluster.ctx, node, cluster.config, cluster.MergeInitialAggregateContainerState)
		cluster.nodes[node.Name] = state
		if cluster.addNodeCallback != nil {
			cluster.addNodeCallback(node.Name)
		}
	}
	state.UpdateNode(node)

	return state
}

// DeleteNode deletes a node from the cluster state.
func (cluster *ClusterState) DeleteNode(name string) {
	cluster.nodePodLock.Lock()
	defer cluster.nodePodLock.Unlock()

	delete(cluster.nodes, name)
	if cluster.deleteNodeCallback != nil {
		cluster.deleteNodeCallback(name)
	}
}

// FindNodeState finds the node state by name.
func (cluster *ClusterState) FindNodeState(name string) *NodeState {
	cluster.nodePodLock.RLock()
	defer cluster.nodePodLock.RUnlock()

	nodeState, ok := cluster.nodes[name]
	if !ok {
		return nil
	}
	return nodeState
}

// IsNodeExisted checks if a node exists in the cluster state.
func (cluster *ClusterState) IsNodeExisted(name string) bool {
	cluster.nodePodLock.RLock()
	defer cluster.nodePodLock.RUnlock()

	_, ok := cluster.nodes[name]
	return ok
}

// GetAllNodeStates returns all node states in the cluster state.
func (cluster *ClusterState) GetAllNodeStates() map[string]*NodeState {
	cluster.nodePodLock.RLock()
	defer cluster.nodePodLock.RUnlock()

	result := make(map[string]*NodeState)
	for node, state := range cluster.nodes {
		result[node] = state
	}
	return result
}

// AddPodAndContainer adds or updates a pod and its containers in the cluster state.
func (cluster *ClusterState) AddPodAndContainer(pod *corev1.Pod) error {
	cluster.nodePodLock.Lock()
	defer cluster.nodePodLock.Unlock()

	node := pod.Spec.NodeName
	if node == "" {
		return fmt.Errorf("node name of pod is empty")
	}

	nodeState, ok := cluster.nodes[node]
	if !ok || nodeState == nil {
		return fmt.Errorf("node:%v is not existd in cluster", node)
	}

	cluster.pods[PodID{
		Namespace: pod.Namespace,
		PodName:   pod.Name,
	}] = node
	nodeState.AddOrUpdatePodAndContainer(pod)
	return nil
}

func calculateRequestedResources(container corev1.Container) Resources {
	cpuQuantity := container.Resources.Requests[corev1.ResourceCPU]
	cpuMillicores := cpuQuantity.MilliValue()

	memoryQuantity := container.Resources.Requests[corev1.ResourceMemory]
	memoryBytes := memoryQuantity.Value()

	return Resources{
		ResourceCPU:    ResourceAmount(cpuMillicores),
		ResourceMemory: ResourceAmount(memoryBytes),
	}
}

// DeleteNotExistedNodes deletes nodes that are not in the current node list from the cluster state.
func (cluster *ClusterState) DeleteNotExistedNodes(curNodes map[string]*corev1.Node) {
	cluster.nodePodLock.Lock()
	defer cluster.nodePodLock.Unlock()
	for node := range cluster.nodes {
		if _, ok := curNodes[node]; !ok {
			delete(cluster.nodes, node)
			if cluster.deleteNodeCallback != nil {
				cluster.deleteNodeCallback(node)
			}
			klog.V(common.AdvanceDebugLog).Infof("delete a node:%v from clusterState done", node)
		}
	}
}

// DeleteNotExistedPods deletes pods that are not in the current pod list from the cluster state.
func (cluster *ClusterState) DeleteNotExistedPods(curPods map[PodID]*corev1.Pod) {
	cluster.nodePodLock.Lock()
	defer cluster.nodePodLock.Unlock()

	for podID, node := range cluster.pods {
		if _, ok := curPods[podID]; !ok {
			delete(cluster.pods, podID)

			if nodeState, ok2 := cluster.nodes[node]; ok2 {
				nodeState.DeletePod(podID)
			}
		}
	}
}

// DeletePod deletes a pod from the cluster state.
func (cluster *ClusterState) DeletePod(podID PodID) {
	cluster.nodePodLock.Lock()
	defer cluster.nodePodLock.Unlock()

	node, ok := cluster.pods[podID]
	if !ok {
		return
	}
	delete(cluster.pods, podID)

	nodeState, ok := cluster.nodes[node]
	if ok && nodeState != nil {
		nodeState.DeletePod(podID)
	}
}

// SaveCheckpoints saves the checkpoints of all nodes in the cluster state.
func (cluster *ClusterState) SaveCheckpoints(parallelSaveNodeCount int) ([]AggregateStateKey,
	[]*colocationv1.ContainerCheckpointStatus) {
	cluster.nodePodLock.RLock()
	defer cluster.nodePodLock.RUnlock()

	nodeStates := make([]*NodeState, 0, len(cluster.nodes))
	for _, nodeState := range cluster.nodes {
		if nodeState != nil {
			nodeStates = append(nodeStates, nodeState)
		}
	}
	sort.Slice(nodeStates, func(i, j int) bool {
		return nodeStates[i].checkpointWritten.Before(nodeStates[j].checkpointWritten)
	})

	clusterKeys := make([]AggregateStateKey, 0)
	clusterCkpts := make([]*colocationv1.ContainerCheckpointStatus, 0)

	count := len(nodeStates)
	if parallelSaveNodeCount < count {
		count = parallelSaveNodeCount
	}
	for i := 0; i < count; i++ {
		nodeKeys, nodeCkpts := nodeStates[i].SaveCheckpoints()
		clusterKeys = append(clusterKeys, nodeKeys...)
		clusterCkpts = append(clusterCkpts, nodeCkpts...)
	}
	return clusterKeys, clusterCkpts
}

// AddOrUpdateInitialAggregateState adds or updates an initial aggregate state in the cluster state.
func (cluster *ClusterState) AddOrUpdateInitialAggregateState(aggregateKey AggregateStateKey,
	state *AggregateContainerState) {
	cluster.initialAggregateStateMapLock.Lock()
	defer cluster.initialAggregateStateMapLock.Unlock()

	cluster.initialAggregateStateMap[aggregateKey] = state
}

// InitialAggregateStateSize returns the size of the initial aggregate state map.
func (cluster *ClusterState) InitialAggregateStateSize() int {
	cluster.initialAggregateStateMapLock.RLock()
	defer cluster.initialAggregateStateMapLock.RUnlock()

	return len(cluster.initialAggregateStateMap)
}

// CleanupInitialAggregateState cleans up the initial aggregate state map and returns the size of the cleaned up map.
func (cluster *ClusterState) CleanupInitialAggregateState() int {
	cluster.initialAggregateStateMapLock.RLock()
	defer cluster.initialAggregateStateMapLock.RUnlock()

	count := len(cluster.initialAggregateStateMap)
	cluster.initialAggregateStateMap = nil
	return count
}

// AddOrUpdateCheckpoint adds or updates a checkpoint in the cluster state.
func (cluster *ClusterState) AddOrUpdateCheckpoint(
	namespaceName types.NamespacedName,
	checkpoint *colocationv1.ContainerCheckpoint,
) {
	cluster.checkpointLock.Lock()
	defer cluster.checkpointLock.Unlock()

	cluster.namespacedNameToCheckpoints[namespaceName] = checkpoint
	cluster.aggregateKeyToCheckpoints[MakeAggregateStateKey(PodID{
		Namespace: checkpoint.Spec.Namespace,
		PodName:   checkpoint.Spec.PodName,
	}, checkpoint.Spec.ContainerName)] = checkpoint
}

// DeleteCheckpoint deletes a checkpoint from the cluster state.
func (cluster *ClusterState) DeleteCheckpoint(namespaceName types.NamespacedName) {
	cluster.checkpointLock.Lock()
	defer cluster.checkpointLock.Unlock()

	checkpoint, ok := cluster.namespacedNameToCheckpoints[namespaceName]
	if !ok {
		return
	}

	delete(cluster.namespacedNameToCheckpoints, namespaceName)
	delete(cluster.aggregateKeyToCheckpoints, MakeAggregateStateKey(PodID{
		Namespace: checkpoint.Spec.Namespace,
		PodName:   checkpoint.Spec.PodName,
	}, checkpoint.Spec.ContainerName))
}

// GetCheckpoint returns the checkpoint for the given aggregate key.
func (cluster *ClusterState) GetCheckpoint(aggregateKey AggregateStateKey) *colocationv1.ContainerCheckpoint {
	cluster.checkpointLock.RLock()
	defer cluster.checkpointLock.RUnlock()

	ckpt, ok := cluster.aggregateKeyToCheckpoints[aggregateKey]
	if !ok {
		return nil
	}
	return ckpt
}

// GetAllCheckpoint returns all checkpoints in the cluster state.
func (cluster *ClusterState) GetAllCheckpoint() []*colocationv1.ContainerCheckpoint {
	cluster.checkpointLock.RLock()
	defer cluster.checkpointLock.RUnlock()

	ckpts := make([]*colocationv1.ContainerCheckpoint, 0)

	for _, ckpt := range cluster.namespacedNameToCheckpoints {
		ckpts = append(ckpts, ckpt)
	}
	return ckpts
}

// MergeInitialAggregateContainerState merges the initial aggregate state into the given aggregate state.
func (cluster *ClusterState) MergeInitialAggregateContainerState(aggregateKey AggregateStateKey,
	aggregateState *AggregateContainerState) {
	cluster.initialAggregateStateMapLock.RLock()
	defer cluster.initialAggregateStateMapLock.RUnlock()

	initialState, ok := cluster.initialAggregateStateMap[aggregateKey]
	if ok {
		aggregateState.MergeContainerState(initialState)
		delete(cluster.initialAggregateStateMap, aggregateKey)
	}
}
