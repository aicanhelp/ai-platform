/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Original file: https://github.com/kubernetes/autoscaler/tree/cluster-autoscaler-chart-1.0.2/vertical-pod-autoscaler/pkg/recommender/model/cluster.go
Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
Modifications:
Add function UpdateNode GetNode
Modified function AddOrUpdateContainer StateMapSize AddOrUpdatePod garbageCollectAggregateCollectionStates
Modified struct NodeState PodState AggretaContainerStatesMap

*/

package aggregate

import (
	"context"
	"fmt"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	colocationv1 "openfuyao.com/colocation-management/pkg/apis/v1"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

// NodeState holds all runtime information about the cluster required for the
// VPA operations, i.e. configuration of resources (pods, containers,
// VPA objects), aggregated utilization of compute resources (CPU, memory) and
// events (container OOMs).
// All input to the VPA Recommender algorithm lives in this structure.
type NodeState struct {
	ctx context.Context

	Node *corev1.Node

	// pods in the node.
	pods    map[PodID]*PodState
	podLock sync.RWMutex

	// All container aggregations where the usage samples are stored.
	aggregateStateMap     AggregateContainerStatesMap
	aggregateStateMapLock sync.RWMutex
	checkpointWritten     time.Time

	lastAggregateContainerStateGC time.Time
	aggregateStateGCInterval      time.Duration

	mergeInitialAggregateStateFunc func(key AggregateStateKey, state *AggregateContainerState)
}

// StateMapSize is the number of pods being tracked by the VPA
func (node *NodeState) StateMapSize() int {
	node.aggregateStateMapLock.RLock()
	defer node.aggregateStateMapLock.RUnlock()

	return len(node.aggregateStateMap)
}

// UpdateNode updates the state of the node with a given Node object
func (node *NodeState) UpdateNode(n *corev1.Node) {
	node.Node = n
}

// GetNode returns the node object of the node state.
func (node *NodeState) GetNode() *corev1.Node {
	return node.Node
}

// AggregateContainerStatesMap is a map from AggregateStateKey to AggregateContainerState.
type AggregateContainerStatesMap map[AggregateStateKey]*AggregateContainerState

// PodState holds runtime information about a single Pod.
type PodState struct {
	// Unique id of the Pod.
	ID PodID
	// Containers that belong to the Pod, keyed by the container name.
	Containers map[string]*ContainerState
	// PodPhase describing current life cycle phase of the Pod.
	Phase corev1.PodPhase
}

// NewNodeState returns a new NodeState with no pods.
func NewNodeState(
	ctx context.Context,
	node *corev1.Node,
	config *apps.Configuration,
	mergeInitialAggregateStateFunc func(AggregateStateKey, *AggregateContainerState),
) *NodeState {
	return &NodeState{
		ctx:                            ctx,
		Node:                           node,
		pods:                           make(map[PodID]*PodState),
		aggregateStateMap:              make(AggregateContainerStatesMap),
		lastAggregateContainerStateGC:  time.Unix(0, 0),
		aggregateStateGCInterval:       config.AggregateStateGCInterval,
		mergeInitialAggregateStateFunc: mergeInitialAggregateStateFunc,
	}
}

// ContainerUsageSampleWithKey holds a ContainerUsageSample together with the
// ID of the container it belongs to.
type ContainerUsageSampleWithKey struct {
	ContainerUsageSample
	Container ContainerID
}

// AddOrUpdatePodAndContainer updates the state of the pod with a given PodID
func (node *NodeState) AddOrUpdatePodAndContainer(pod *corev1.Pod) {
	podID := PodID{
		Namespace: pod.Namespace,
		PodName:   pod.Name,
	}
	node.AddOrUpdatePod(podID, pod.Status.Phase)
	klog.V(common.TraceDebugLog).Infof("add or updated a pod:%+v to nodeState done", podID)

	for _, container := range pod.Spec.InitContainers {
		containerID := ContainerID{
			PodID:         podID,
			ContainerName: container.Name,
		}
		if err := node.AddOrUpdateContainer(containerID, calculateRequestedResources(container)); err != nil {
			klog.Warningf("add or updated a init container %+v failed. Reason: %+v", containerID, err)
		}
		klog.V(common.TraceDebugLog).Infof("add or updated a init container:%+v to nodeState done", containerID)
	}
	for _, container := range pod.Spec.Containers {
		containerID := ContainerID{
			PodID:         podID,
			ContainerName: container.Name,
		}
		if err := node.AddOrUpdateContainer(containerID, calculateRequestedResources(container)); err != nil {
			klog.Warningf("add or updated a container %+v failed. Reason: %+v", containerID, err)
		}
		klog.V(common.TraceDebugLog).Infof("add or updated a container:%+v to nodeState done", containerID)
	}
}

// AddOrUpdatePod updates the state of the pod with a given PodID, if it is
// present in the cluster object. Otherwise a new pod is created and added to
// the Cluster object.
// If the labels of the pod have changed, it updates the links between the containers
// and the aggregations.
func (node *NodeState) AddOrUpdatePod(podID PodID, phase corev1.PodPhase) {
	node.podLock.Lock()
	defer node.podLock.Unlock()

	pod, podExists := node.pods[podID]
	if !podExists {
		pod = newPod(podID)
		node.pods[podID] = pod
	}

	if !podExists {
		// Set the links between the containers and aggregations based on the current pod labels.
		for containerName, container := range pod.Containers {
			containerID := ContainerID{PodID: podID, ContainerName: containerName}
			container.aggregator = node.findOrCreateAggregateContainerState(containerID)
		}
	}
	pod.Phase = phase
}

// GetContainer returns the ContainerState object for a given ContainerID or
// null if it's not present in the model.
func (node *NodeState) GetContainer(containerID ContainerID) *ContainerState {
	node.podLock.RLock()
	defer node.podLock.RUnlock()

	pod, podExists := node.pods[containerID.PodID]
	if podExists {
		container, containerExists := pod.Containers[containerID.ContainerName]
		if containerExists {
			return container
		}
	}
	return nil
}

// DeletePod removes an existing pod from the node.
func (node *NodeState) DeletePod(podID PodID) {
	node.podLock.Lock()
	defer node.podLock.Unlock()
	delete(node.pods, podID)
}

// AddOrUpdateContainer creates a new container with the given ContainerID and
// adds it to the parent pod in the NodeState object, if not yet present.
// Requires the pod to be added to the NodeState first. Otherwise an error is
// returned.
func (node *NodeState) AddOrUpdateContainer(containerID ContainerID, request Resources) error {
	node.podLock.Lock()
	defer node.podLock.Unlock()

	pod, podExists := node.pods[containerID.PodID]
	if !podExists {
		return NewKeyError(containerID.PodID)
	}
	if container, containerExists := pod.Containers[containerID.ContainerName]; !containerExists {
		node.findOrCreateAggregateContainerState(containerID)
		pod.Containers[containerID.ContainerName] = NewContainerState(request,
			NewContainerStateAggregatorProxy(node, containerID))
	} else {
		// Container aleady exists. Possibly update the request.
		container.Request = request
	}
	return nil
}

// AddSample adds a new usage sample to the proper container in the NodeState
// object. Requires the container as well as the parent pod to be added to the
// NodeState first. Otherwise an error is returned.
func (node *NodeState) AddSample(sample *ContainerUsageSampleWithKey) error {
	node.podLock.RLock()
	defer node.podLock.RUnlock()

	pod, podExists := node.pods[sample.Container.PodID]
	if !podExists {
		return NewKeyError(sample.Container.PodID)
	}
	containerState, containerExists := pod.Containers[sample.Container.ContainerName]
	if !containerExists {
		return NewKeyError(sample.Container)
	}
	if !containerState.AddSample(&sample.ContainerUsageSample) {
		return fmt.Errorf("sample discarded (invalid or out of order)")
	}
	return nil
}

func newPod(id PodID) *PodState {
	return &PodState{
		ID:         id,
		Containers: make(map[string]*ContainerState),
	}
}

// The pod with the corresponding PodID must already be present in the NodeState.
// aggregateStateKeyForContainerID returns the AggregateStateKey for the ContainerID.
func (node *NodeState) aggregateStateKeyForContainerID(containerID ContainerID) AggregateStateKey {

	pod, podExists := node.pods[containerID.PodID]
	if !podExists {
		panic(fmt.Sprintf("Pod not present in the NodeState: %s/%s", containerID.PodID.Namespace, containerID.PodID.PodName))
	}
	return MakeAggregateStateKey(pod.ID, containerID.ContainerName)
}

// findOrCreateAggregateContainerState returns (possibly newly created) AggregateContainerState
// that should be used to aggregate usage samples from container with a given ID.
// The pod with the corresponding PodID must already be present in the NodeState.
func (node *NodeState) findOrCreateAggregateContainerState(containerID ContainerID) *AggregateContainerState {
	node.aggregateStateMapLock.Lock()
	defer node.aggregateStateMapLock.Unlock()

	aggregateStateKey := node.aggregateStateKeyForContainerID(containerID)
	aggregateContainerState, aggregateStateExists := node.aggregateStateMap[aggregateStateKey]
	if !aggregateStateExists {
		aggregateContainerState = NewAggregateContainerState()
		node.mergeInitialAggregateStateFunc(aggregateStateKey, aggregateContainerState)
		node.aggregateStateMap[aggregateStateKey] = aggregateContainerState
	}
	return aggregateContainerState
}

// garbageCollectAggregateCollectionStates removes obsolete AggregateCollectionStates from the NodeState.
// AggregateCollectionState is obsolete in following situations:
// 1) It has no samples and there are no more contributive pods - a pod is contributive in any of following situations:
//
//	a) It is in an active state - i.e. not PodSucceeded nor PodFailed.
//
// 2) The last sample is too old to give meaningful recommendation (>8 days),
// 3) There are no samples and the aggregate state was created >8 days ago.
func (node *NodeState) garbageCollectAggregateCollectionStates(now time.Time) {
	keysToDelete := make([]AggregateStateKey, 0)
	contributiveKeys := node.getContributiveAggregateStateKeys()

	node.aggregateStateMapLock.Lock()
	defer node.aggregateStateMapLock.Unlock()

	for key, aggregateContainerState := range node.aggregateStateMap {
		isKeyContributive, found := contributiveKeys[key]
		if !found {
			keysToDelete = append(keysToDelete, key)
			klog.V(common.GeneralDebugLog).Infof("Remove AggregateState if it's corresponding pod is not exists for %+v", key)
			continue
		}
		if !isKeyContributive && aggregateContainerState.isEmpty() {
			keysToDelete = append(keysToDelete, key)
			klog.V(common.GeneralDebugLog).Infof("Removing empty and not contributive AggregateCollectionState for %+v", key)
			continue
		}
		if aggregateContainerState.isExpired(now) {
			keysToDelete = append(keysToDelete, key)
			klog.V(common.GeneralDebugLog).Infof("Removing expired AggregateCollectionState for %+v", key)
		}
	}
	for _, key := range keysToDelete {
		delete(node.aggregateStateMap, key)
	}
	klog.V(common.VerboseDebugLog).InfoS("Garbage collection of AggregateStates done.", "node", node.Node.Name,
		"deleted count", len(keysToDelete))
}

// RateLimitedGarbageCollectAggregateCollectionStates removes obsolete AggregateCollectionStates from the NodeState.
// It performs clean up only if more than `aggregateStateGCInterval` passed since the last time it performed a cleanup.
// AggregateCollectionState is obsolete in following situations:
// 1) It has no samples and there are no more contributive pods - a pod is contributive in any of following situations:
//
//	a) It is in an active state - i.e. not PodSucceeded nor PodFailed.
//
// 2) The last sample is too old to give meaningful recommendation (>8 days),
// 3) There are no samples and the aggregate state was created >8 days ago.
func (node *NodeState) RateLimitedGarbageCollectAggregateCollectionStates(now time.Time) {
	if now.Sub(node.lastAggregateContainerStateGC) < node.aggregateStateGCInterval {
		return
	}
	node.garbageCollectAggregateCollectionStates(now)
	node.lastAggregateContainerStateGC = now
}

// remove controllerFetcher
// because usage of pod that associated controller is not exists should be collected in this case
func (node *NodeState) getContributiveAggregateStateKeys() map[AggregateStateKey]bool {
	node.podLock.RLock()
	defer node.podLock.RUnlock()

	contributiveKeys := map[AggregateStateKey]bool{}
	for _, pod := range node.pods {
		// Pod is considered contributive in any of following situations:
		// 1) It is in active state - i.e. not PodSucceeded nor PodFailed.
		if utils.ContributivePod(pod.Phase) {
			for container := range pod.Containers {
				contributiveKeys[MakeAggregateStateKey(pod.ID, container)] = true
			}
		}
	}
	return contributiveKeys
}

// GetTotalResourceEstimation returns total resource estimation for all containers in the NodeState.
func (node *NodeState) GetTotalResourceEstimation(estimateFunc func(s *AggregateContainerState) Resources) Resources {
	node.aggregateStateMapLock.RLock()
	defer node.aggregateStateMapLock.RUnlock()

	totalRes := make(Resources)
	for _, aggregateContainerState := range node.aggregateStateMap {
		containerResources := estimateFunc(aggregateContainerState)
		totalRes.Add(containerResources)
	}
	return totalRes
}

// SaveCheckpoints returns all checkpoints for all containers in the NodeState.
func (node *NodeState) SaveCheckpoints() ([]AggregateStateKey, []*colocationv1.ContainerCheckpointStatus) {
	node.aggregateStateMapLock.RLock()
	defer node.aggregateStateMapLock.RUnlock()

	keys := make([]AggregateStateKey, 0)
	ckpts := make([]*colocationv1.ContainerCheckpointStatus, 0)
	for aggregateKey, aggregateContainerState := range node.aggregateStateMap {
		ckpt, err := aggregateContainerState.SaveToCheckpoint()
		if err != nil {
			klog.ErrorS(err, "save checkpoint failed.", "nodeState", node.Node.Name, "aggregateKey", aggregateKey)
			continue
		}
		node.checkpointWritten = time.Now()
		keys = append(keys, aggregateKey)
		ckpts = append(ckpts, ckpt)
	}
	return keys, ckpts
}
