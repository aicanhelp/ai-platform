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

package recommend

import (
	"context"
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

var (
	zeroCPU = resource.NewMilliQuantity(0, resource.DecimalSI)
	zeroMem = resource.NewQuantity(0, resource.BinarySI)
	zeroRes = v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(0, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(0, resource.BinarySI),
	}
)

const (
	// milliScale is the scale of millicores in a resource.Quantity.
	milliScale = -3
)

// ResourceRecommender is the resource recommender.
type ResourceRecommender struct {
	ctx context.Context
	client.Client
	config          *apps.Configuration
	clusterState    *aggregate.ClusterState
	nodeRecommender NodeResourceRecommender

	nodeToggleState              map[string]bool
	nodeLastUpdateRecommendation map[string]time.Time
}

// NewResourceRecommender creates a new resource recommender.
func NewResourceRecommender(ctx context.Context, mgr controllerruntime.Manager,
	config *apps.Configuration, clusterState *aggregate.ClusterState) *ResourceRecommender {
	return &ResourceRecommender{
		ctx:                          ctx,
		clusterState:                 clusterState,
		Client:                       mgr.GetClient(),
		config:                       config,
		nodeRecommender:              NewNodeResourceRecommender(config),
		nodeToggleState:              make(map[string]bool),
		nodeLastUpdateRecommendation: make(map[string]time.Time),
	}
}

// Run starts the resource recommender.
func (r *ResourceRecommender) Run() {
	ticker := time.NewTicker(r.config.BERecommendPeriod)
	defer ticker.Stop()

	startTime := time.Now()
	r.doUpdateNodeRecommendation()
	klog.V(common.VerboseDebugLog).Infof("ResourceRecommender: Update node recommendation done. "+
		"elapsed: %v", time.Since(startTime))
	for {
		select {
		case <-r.ctx.Done():
			return
		case startTime := <-ticker.C:
			r.doUpdateNodeRecommendation()
			klog.V(common.VerboseDebugLog).Infof("ResourceRecommender: Update node recommendation done."+
				" elapsed: %v", time.Since(startTime))
		}
	}
}

func (r *ResourceRecommender) doUpdateNodeRecommendation() {
	allStates := r.clusterState.GetAllNodeStates()
	if len(allStates) == 0 {
		return
	}
	klog.V(common.AdvanceDebugLog).Infof("ResourceRecommender:"+
		" Get total count of all nodeState is %v", len(allStates))

	for nodeName, state := range allStates {
		if state == nil {
			continue
		}
		r.processNode(state, nodeName)
	}
}
func (r *ResourceRecommender) processNode(state *aggregate.NodeState, nodeName string) {
	klog.V(common.TraceDebugLog).Infof("ResourceRecommender: Node:%v "+
		"have count:%v aggregateKey", nodeName, state.StateMapSize())
	if !r.config.RecommendToggleEnabled {
		toggleState, ok := r.nodeToggleState[nodeName]
		if !ok || !toggleState {
			if err := r.updateNodeStatus(state.GetNode(), zeroRes, false, false); err != nil {
				klog.ErrorS(err, "clean recommended resources failed "+
					"when recommand-toggle is enabled.", "node", nodeName)
				return
			}
			r.nodeToggleState[nodeName] = true
			klog.InfoS("clean recommended resources successful "+
				"when recommand-toggle is enabled.", "node", nodeName)
		}
		return
	}

	// 去除HLS POD请求量
	hlsPods, err := listHlsPodOnNode(r.ctx, r.Client, nodeName)
	if err != nil {
		klog.ErrorS(err, "list hls pod on node failed.", "node", nodeName)
		return
	}
	hlsCpuReqTotal, hlsMemoryReqTotal, err := CalculateHlsPodsUsage(hlsPods)
	if err != nil {
		klog.ErrorS(err, "calculate hls pods usage failed.", "node", nodeName)
		return
	}
	err = OverHeadNodeAllocate(state, hlsCpuReqTotal, hlsMemoryReqTotal)
	if err != nil {
		klog.ErrorS(err, "scale node allocate failed.", "node", nodeName)
		return
	}

	var skipCPUUpdate = false
	var skipMemUpdate = false

	recommendedRes := r.estimateNodeRes(state)
	for resName, resQuantity := range recommendedRes {
		switch resName {
		case v1.ResourceCPU:
			skipCPUUpdate = r.handleResUpdateJitter(state.GetNode(), resName, resQuantity)
		case v1.ResourceMemory:
			skipMemUpdate = r.handleResUpdateJitter(state.GetNode(), resName, resQuantity)
		}
	}
	err = r.updateNodeStatus(state.GetNode(), recommendedRes, skipCPUUpdate, skipMemUpdate)
	if err != nil {
		klog.ErrorS(err, "update node status failed.", "node", nodeName)
		return
	}
}
func (r *ResourceRecommender) estimateNodeRes(
	state *aggregate.NodeState,
) v1.ResourceList {
	estimatedResAmount := r.nodeRecommender.GetNodeResourceEstimation(state)
	estimatedResQuantity := aggregate.ResourcesAsResourceList(estimatedResAmount)

	var (
		allocatableResQuantity = v1.ResourceList{}
		recommendedRes         = v1.ResourceList{}
	)
	for resName, resQuantity := range state.GetNode().Status.Allocatable {
		var zeroQuantity *resource.Quantity
		switch resName {
		case v1.ResourceCPU:
			zeroQuantity = zeroCPU
		case v1.ResourceMemory:
			zeroQuantity = zeroMem
		default:
			continue
		}

		allocatableResQuantity[resName] = resQuantity
		estimated, ok := estimatedResQuantity[resName]
		if ok {
			resQuantity.Sub(estimated)
			if resQuantity.Cmp(*zeroQuantity) == -1 {
				resQuantity = *zeroQuantity
			}
		}
		recommendedRes[resName] = resQuantity
	}
	klog.V(common.VerboseDebugLog).InfoS("ResourceRecommender: resource statement:",
		"node", state.GetNode().Name,
		"allocatable", allocatableResQuantity,
		"estimated usage", estimatedResQuantity,
		"recommended", recommendedRes)
	return recommendedRes
}

func (r *ResourceRecommender) handleResUpdateJitter(
	node *v1.Node,
	resName v1.ResourceName,
	resQuantity resource.Quantity,
) bool {
	var (
		skipUpdate            = false
		diffRes               resource.Quantity
		skipExtendedResName   v1.ResourceName
		referenceMinThreshold resource.Quantity
	)

	extendedResName, recommendMinThreshold := r.getResourceConfig(resName)
	oldRecommendedRes, found := node.Status.Allocatable[extendedResName]
	if !found {
		return false
	}

	if extendedResName == common.ExtenderResourceCPU {
		oldRecommendedRes.SetScaled(oldRecommendedRes.Value(), milliScale)
	}
	klog.V(common.AdvanceDebugLog).InfoS("handleResUpdateJitter: before compare", "resName", resName,
		"resQuantity", resQuantity, "oldRecommendedRes", oldRecommendedRes)

	skipUpdate, diffRes = r.shouldSkipUpdate(resQuantity, oldRecommendedRes, recommendMinThreshold)

	if skipUpdate {
		klog.V(common.VerboseDebugLog).InfoS("handleResUpdateJitter:",
			"skipUpdate", skipUpdate,
			"skipExtendedResName", skipExtendedResName,
			"diffRes", diffRes,
			"referenceMinThreshold", referenceMinThreshold)
	} else {
		klog.V(common.AdvanceDebugLog).InfoS("handleResUpdateJitter: after compare", "extendedResName", extendedResName,
			"resQuantity", resQuantity, "oldRecommendedRes", oldRecommendedRes)
	}
	return skipUpdate
}

func (r *ResourceRecommender) getResourceConfig(resName v1.ResourceName) (v1.ResourceName, resource.Quantity) {
	switch resName {
	case v1.ResourceCPU:
		return common.ExtenderResourceCPU,
			*resource.NewMilliQuantity(r.config.BERecommendMinCPUMillicores, resource.DecimalSI)
	case v1.ResourceMemory:
		return common.ExtenderResourceMemory,
			*resource.NewQuantity(r.config.BERecommendMinMemoryMb, resource.BinarySI)
	default:
		return "", resource.Quantity{}
	}
}

// shouldSkipUpdate
// compare whether |newRecommendation - oldAllocatable| <= minThreshold, then skip updating node status
func (r *ResourceRecommender) shouldSkipUpdate(newRes, oldRes, threshold resource.Quantity) (bool, resource.Quantity) {
	comparison := newRes.Cmp(oldRes)

	switch comparison {
	case 0:
		return true, resource.Quantity{}

	case 1:
		diff := newRes.DeepCopy()
		diff.Sub(oldRes)
		return diff.Cmp(threshold) == -1, diff

	case -1:
		diff := oldRes.DeepCopy()
		diff.Sub(newRes)
		return diff.Cmp(threshold) == -1, diff
	}

	return false, resource.Quantity{}
}

func (r *ResourceRecommender) updateNodeStatus(
	node *v1.Node,
	recommendedRes v1.ResourceList,
	skipCPUUpdate bool,
	skipMemoryUpdate bool,
) error {
	if skipCPUUpdate && skipMemoryUpdate {
		return nil
	}

	nodeCopy := node.DeepCopy()
	for resName, resQuantity := range recommendedRes {
		var (
			extendedResName v1.ResourceName
		)
		switch resName {
		case v1.ResourceCPU:
			if skipCPUUpdate {
				break
			}
			extendedResName = common.ExtenderResourceCPU
			resQuantity.SetScaled(resQuantity.MilliValue(), 0)
		case v1.ResourceMemory:
			if skipMemoryUpdate {
				break
			}
			extendedResName = common.ExtenderResourceMemory
		default:
			continue
		}

		if extendedResName != "" {
			nodeCopy.Status.Capacity[extendedResName] = resQuantity
			nodeCopy.Status.Allocatable[extendedResName] = resQuantity
		}
	}
	if err := r.Client.Status().Update(r.ctx, nodeCopy); err != nil {
		klog.ErrorS(err,
			"ResourceRecommender: Update best-effort resource of node status with recommended allocatable and capacity fail.",
			"node", nodeCopy.Name)
		return err
	}

	klog.V(common.VerboseDebugLog).InfoS(
		"ResourceRecommender: Update best-effort resource of node status with recommended allocatable and capacity done.",
		"node", nodeCopy.Name)
	return nil
}

// NodeResourceRecommender computes resource estimation
type NodeResourceRecommender interface {
	GetNodeResourceEstimation(nodeState *aggregate.NodeState) aggregate.Resources
}

type nodeResourceRecommender struct {
	targetEstimator ResourceEstimator
}

// NewNodeResourceRecommender returns the primary recommender.
func NewNodeResourceRecommender(config *apps.Configuration) NodeResourceRecommender {
	targetEstimator := NewPercentileEstimator(config.EstimateCPUPercentile, config.EstimateMemoryPercentile)

	return &nodeResourceRecommender{targetEstimator: targetEstimator}
}

// GetNodeResourceEstimation returns the estimation of node resource usage.
func (r *nodeResourceRecommender) GetNodeResourceEstimation(nodeState *aggregate.NodeState) aggregate.Resources {
	if nodeState == nil {
		return nil
	}
	return nodeState.GetTotalResourceEstimation(r.targetEstimator.GetResourceEstimation)
}

// FilterControlledResources returns estimations from 'estimation' only for resources present in 'controlledResources'.
func FilterControlledResources(estimation aggregate.Resources,
	controlledResources []aggregate.ResourceName) aggregate.Resources {
	result := make(aggregate.Resources)
	for _, resource := range controlledResources {
		if value, ok := estimation[resource]; ok {
			result[resource] = value
		}
	}
	return result
}

// listHlsPodOnNode 获取指定节点上的HLS Pods
func listHlsPodOnNode(ctx context.Context, cl client.Client, nodeName string) ([]v1.Pod, error) {
	podList := &v1.PodList{}
	if err := cl.List(ctx, podList); err != nil {
		klog.ErrorS(err, "listHlsPodOnNode: List pods fail.")
		return nil, err
	}

	var hlsPodsOnNode []v1.Pod
	for _, pod := range podList.Items {
		if utils.HLSPod(&pod) && pod.Spec.NodeName == nodeName {
			hlsPodsOnNode = append(hlsPodsOnNode, pod)
		}
	}
	return hlsPodsOnNode, nil
}

// CalculateHlsPodsUsage 计算HLS Pods的request使用情况
func CalculateHlsPodsUsage(hlsPodsOnNode []v1.Pod) (*resource.Quantity, *resource.Quantity, error) {
	totalCPU := *resource.NewQuantity(0, resource.DecimalSI)
	totalMemory := *resource.NewQuantity(0, resource.BinarySI)

	for _, pod := range hlsPodsOnNode {
		calculateContainerResources(&totalCPU, &totalMemory, pod.Spec.Containers)
		calculateInitContainerResources(&totalCPU, &totalMemory, pod.Spec.InitContainers)
	}

	return &totalCPU, &totalMemory, nil
}

// calculateContainerResources 处理常规容器资源
func calculateContainerResources(totalCPU, totalMemory *resource.Quantity, containers []v1.Container) {
	for _, container := range containers {
		addResourceIfExists(totalCPU, container.Resources.Requests, v1.ResourceCPU)
		addResourceIfExists(totalMemory, container.Resources.Requests, v1.ResourceMemory)
	}
}

// calculateInitContainerResources 处理Init容器资源
func calculateInitContainerResources(totalCPU, totalMemory *resource.Quantity, initContainers []v1.Container) {
	for _, container := range initContainers {
		maxResourceIfGreater(totalCPU, container.Resources.Requests, v1.ResourceCPU)
		maxResourceIfGreater(totalMemory, container.Resources.Requests, v1.ResourceMemory)
	}
}

// addResourceIfExists 累加资源
func addResourceIfExists(total *resource.Quantity, requests v1.ResourceList, resourceName v1.ResourceName) {
	if request, ok := requests[resourceName]; ok {
		total.Add(request)
	}
}

// maxResourceIfGreater 取最大资源请求
func maxResourceIfGreater(total *resource.Quantity, requests v1.ResourceList, resourceName v1.ResourceName) {
	if request, ok := requests[resourceName]; ok && request.Cmp(*total) > 0 {
		*total = request.DeepCopy()
	}
}

// OverHeadNodeAllocate 对节点当前资源进行预留和去掉HLS Pods的资源
func OverHeadNodeAllocate(state *aggregate.NodeState, cpuReqTotal, memoryReqTotal *resource.Quantity) error {
	if state == nil || state.GetNode() == nil {
		return fmt.Errorf("nodeAllocate: state or state.node is nil")
	}
	if cpuReqTotal == nil || memoryReqTotal == nil {
		return fmt.Errorf("nodeAllocate: cpuReqTotal or memoryReqTotal is nil")
	}
	allocatable := state.GetNode().Status.Allocatable
	if allocatable == nil {
		return fmt.Errorf("node allocatable resources are nil")
	}
	// 当前预留系数 用于额外开销
	OverHead := 1 - common.OverheadFactor

	if cpuAlloc, exists := allocatable[v1.ResourceCPU]; exists {
		if cpuAlloc.Cmp(*cpuReqTotal) > 0 {
			scaledCPU := scaleQuantityCpu(cpuAlloc, OverHead)
			diff := scaledCPU.DeepCopy()
			// 预留后整机资源如果不满足HLS资源请求量 后续BE资源依然会设置为0 在日志中触发告警
			diff.Sub(*cpuReqTotal)
			if diff.Sign() < 0 {
				klog.Warningf("nodeAllocate: remaining cpu request is less than hls cpu request")
			}
			allocatable[v1.ResourceCPU] = diff
		} else if cpuAlloc.Cmp(*cpuReqTotal) == 0 {
			diff := cpuAlloc.DeepCopy()
			diff.Sub(*cpuReqTotal)
			// 如果HLS类Pod cpu请求量 与整机资源量一致 不再进行预留 达成尽可能使用的目标
			allocatable[v1.ResourceCPU] = diff
		} else {
			// HLS类Pod cpu请求量 大于整机资源量 属于异常情况
			return fmt.Errorf("hls cpu request is greater than node allocatable cpu resources")
		}
	} else {
		return fmt.Errorf("node allocatable cpu resources are nil")
	}

	if memAlloc, exists := allocatable[v1.ResourceMemory]; exists {
		if memAlloc.Cmp(*memoryReqTotal) > 0 {
			scaledMemory := scaleQuantityMeomry(memAlloc, OverHead)
			diff := scaledMemory.DeepCopy()
			// 预留后整机资源如果不满足HLS资源请求量 后续BE资源会设置为0 在日志中触发告警
			diff.Sub(*memoryReqTotal)
			if diff.Sign() < 0 {
				klog.Warningf("nodeAllocate: remaining memory request is less than hls memory request")
			}
			allocatable[v1.ResourceMemory] = diff
		} else if memAlloc.Cmp(*memoryReqTotal) == 0 {
			// 如果HLS类Pod memory请求量 与整机资源量一致 不再进行预留 达成尽可能使用的目标
			diff := memAlloc.DeepCopy()
			diff.Sub(*memoryReqTotal)
			allocatable[v1.ResourceMemory] = diff
		} else {
			// HLS类Pod memory请求量 大于整机资源量 属于异常情况
			return fmt.Errorf("hls memory request is greater than node allocatable memory resources")
		}
	} else {
		return fmt.Errorf("node allocatable memory resources are nil")
	}

	return nil
}

// scaleQuantityCpu 根据资源类型进行缩放
func scaleQuantityCpu(q resource.Quantity, factor float64) resource.Quantity {
	scaled := q.DeepCopy()
	scaled.SetMilli(int64(float64(scaled.MilliValue()) * factor))
	return scaled
}

// scaleQuantityMeomry 根据资源类型进行缩放
func scaleQuantityMeomry(q resource.Quantity, factor float64) resource.Quantity {
	scaled := q.DeepCopy()
	scaled.Set(int64(float64(scaled.Value()) * factor))
	return scaled
}
