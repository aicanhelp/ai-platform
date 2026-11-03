/*
// Copyright 2018 The Kubernetes Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


Original file: https://github.com/kubernetes-sigs/metrics-server/tree/helm-chart-3.6.0/pkg/storage/pod.go
Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
Modifications:
Added PodStorage interface
*/

package metric

import (
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"sigs.k8s.io/metrics-server/pkg/api"

	"openfuyao.com/colocation-management/pkg/common"
)

// PodStorage is an interface for storing and retrieving pod metrics.
type PodStorage interface {
	Store(podsMetrics *MetricsBatch)
	GetLastMetrics(pods ...*metav1.PartialObjectMetadata) (*NodeUsageReport, error)
}

type podStorage struct {
	// last stores pod metric points from last scrape
	last map[PodRef]ContanersMetricsPoint
	// prev stores pod metric points from scrape preceding the last one.
	// Points timestamp should be after last point's and share same start time (no restart).
	prev map[PodRef]ContanersMetricsPoint
	// scrape period of metrics server
	metricsPeriod time.Duration
	mu            sync.RWMutex

	node string
}

// NewPodStorage creates a new PodStorage instance.
func NewPodStorage(node string, metricsPeriod time.Duration) PodStorage {
	return &podStorage{
		last:          make(map[PodRef]ContanersMetricsPoint),
		prev:          make(map[PodRef]ContanersMetricsPoint),
		metricsPeriod: metricsPeriod,
		node:          node,
	}
}

// Store stores new metrics points.
func (s *podStorage) Store(newPods *MetricsBatch) {
	s.mu.Lock()
	defer s.mu.Unlock()
	lastPods := make(map[PodRef]ContanersMetricsPoint, len(newPods.Pods))
	prevPods := make(map[PodRef]ContanersMetricsPoint, len(newPods.Pods))
	var containerCount int
	for podRef, newPod := range newPods.Containers {
		if _, found := lastPods[podRef]; found {
			klog.ErrorS(nil, "Got duplicate pod point", "pod", podRef)
			continue
		}

		newLastPod := ContanersMetricsPoint{Containers: make(map[string]MetricsPoint, len(newPod.Containers))}
		newPrevPod := ContanersMetricsPoint{Containers: make(map[string]MetricsPoint, len(newPod.Containers))}
		for containerName, newPoint := range newPod.Containers {
			if _, exists := newLastPod.Containers[containerName]; exists {
				klog.ErrorS(nil, "Got duplicate Container point", "container", containerName, "pod", podRef)
				continue
			}
			newLastPod.Containers[containerName] = newPoint
			if s.handlePoint(podRef, containerName, newPoint, &newPrevPod) {
				containerCount++
			}
		}
		if len(newPrevPod.Containers) > 0 {
			prevPods[podRef] = newPrevPod
		}
		lastPods[podRef] = newLastPod
	}
	s.last, s.prev = lastPods, prevPods
}

// GetLastMetrics returns the last metrics points for the given pods.
func (s *podStorage) GetLastMetrics(pods ...*metav1.PartialObjectMetadata) (*NodeUsageReport, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	nmr := &NodeUsageReport{
		Node:               s.node,
		Pods:               make(map[PodRef]corev1.ResourceList),
		Containers:         make(map[PodRef]map[string]corev1.ResourceList),
		PodsFromContainers: make(map[PodRef]corev1.ResourceList),
	}
	for _, pod := range pods {
		singlePod := s.processSinglePod(pod, nmr)
		if !singlePod {
			continue
		}
	}
	return nmr, nil
}
func (s *podStorage) processSinglePod(pod *metav1.PartialObjectMetadata, nmr *NodeUsageReport) bool {
	podRef := NewPodRef(pod.Namespace, pod.Name)

	lastPod, found := s.last[podRef]
	if !found {
		klog.V(common.TraceDebugLog).Infof("lastPod not found for %v", podRef)
		return false
	}

	prevPod, found := s.prev[podRef]
	if !found {
		klog.V(common.TraceDebugLog).Infof("prevPod not found for %v", podRef)
		return false
	}

	containerProcessResult, allContainersPresent := s.processContainers(pod, lastPod, prevPod)

	if allContainersPresent {
		nmr.Timestamp = metav1.NewTime(containerProcessResult.earliestTimeInfo.Timestamp)
		nmr.Window = metav1.Duration{Duration: containerProcessResult.earliestTimeInfo.Window}
		nmr.Containers[podRef] = containerProcessResult.cmr
		nmr.PodsFromContainers[podRef] = containerProcessResult.pr
		klog.V(common.TraceDebugLog).InfoS("GetLastMetrics:", "pod.namespace", pod.Namespace, "pod.name", pod.Name,
			"Timestamp", nmr.Timestamp, "containers.len", len(nmr.Containers), "pod.usage", containerProcessResult.pr)
	}
	return true
}

// ContainerProcessResult 封装容器处理结果
type ContainerProcessResult struct {
	cmr              map[string]corev1.ResourceList
	pr               corev1.ResourceList
	earliestTimeInfo api.TimeInfo
}

func (s *podStorage) processContainers(
	pod *metav1.PartialObjectMetadata,
	lastPod, prevPod ContanersMetricsPoint,
) (ContainerProcessResult, bool) {
	cmr := make(map[string]corev1.ResourceList)
	pr := corev1.ResourceList{}
	var earliestTimeInfo api.TimeInfo

	for container, lastContainer := range lastPod.Containers {
		prevContainer, found := prevPod.Containers[container]
		if !found {
			return ContainerProcessResult{cmr: nil, pr: nil, earliestTimeInfo: api.TimeInfo{}}, false
		}

		usage, ti, err := resourceUsage(lastContainer, prevContainer)
		if err != nil {
			klog.ErrorS(err, "Skipping container usage metric",
				"container", container, "pod", klog.KRef(pod.Namespace, pod.Name))
			continue
		}

		cmr[container] = usage
		pr = SumResourceList(pr, usage)

		if earliestTimeInfo.Timestamp.IsZero() || earliestTimeInfo.Timestamp.After(ti.Timestamp) {
			earliestTimeInfo = ti
		}

		klog.V(common.TraceDebugLog).InfoS("GetLastMetrics:", "pod.namespace", pod.Namespace, "pod.name", pod.Name,
			"container.name", container, "container.usage", usage)
	}
	return ContainerProcessResult{cmr: cmr, pr: pr, earliestTimeInfo: earliestTimeInfo}, true

}
func isValidPoint(newPoint MetricsPoint, period, minResolution time.Duration) bool {
	return newPoint.StartTime.Before(newPoint.Timestamp) &&
		newPoint.Timestamp.Sub(newPoint.StartTime) < period &&
		newPoint.Timestamp.Sub(newPoint.StartTime) >= minResolution
}

// 提取出来的函数
func handlePrevPointComparison(podRef PodRef, containerName string, newPoint MetricsPoint,
	newPrevPod *ContanersMetricsPoint, prev map[PodRef]ContanersMetricsPoint) {

	prevPod, found := prev[podRef]
	if !found {
		return
	}

	prevContainer, found := prevPod.Containers[containerName]
	if !found {
		return
	}

	if prevContainer.Timestamp.Before(newPoint.Timestamp) {
		newPrevPod.Containers[containerName] = prevContainer
	} else {
		klog.V(common.GeneralDebugLog).InfoS("New metrics point is older than stored previous, dropping previous",
			"container", containerName,
			"pod", podRef,
			"previousTimestamp", prevContainer.Timestamp,
			"timestamp", newPoint.Timestamp)
	}
}

func (s *podStorage) handlePoint(podRef PodRef, containerName string,
	newPoint MetricsPoint, newPrevPod *ContanersMetricsPoint) bool {
	// 是否为有效Point
	if isValidPoint(newPoint, s.metricsPeriod, common.FreshContainerMinMetricsResolution) {
		copied := newPoint
		copied.Timestamp, copied.CumulativeCpuUsed = newPoint.StartTime, 0
		newPrevPod.Containers[containerName] = copied
		return true
	}

	lastPod, found := s.last[podRef]
	if !found {
		// 无效 且不存在上一次的记录 丢弃
		klog.V(common.TraceDebugLog).InfoS("Invalid new point", "pod", podRef, "container", containerName)
		return false
	}

	// Keep previous metric point if newPoint has not restarted (new metric start time < stored timestamp)
	lastContainer, found := lastPod.Containers[containerName]
	if !found || !newPoint.StartTime.Before(lastContainer.Timestamp) {
		klog.V(common.TraceDebugLog).InfoS("Found new containerName metrics point is older than stored, drop")
		return false
	}

	// If new point is different then one already stored
	if newPoint.Timestamp.After(lastContainer.Timestamp) {
		// Move stored point to previous
		newPrevPod.Containers[containerName] = lastContainer
		return true
	}
	prevPod, found := s.prev[podRef]
	if !found {
		return false
	}
	if prevContainer := prevPod.Containers[containerName]; prevContainer.Timestamp.Before(newPoint.Timestamp) {
		// Keep previous point
		newPrevPod.Containers[containerName] = prevContainer
		return true
	} else {
		klog.V(common.GeneralDebugLog).InfoS("Found new containerName metrics point is older than stored previous"+
			"drop previous", "containerName", containerName,
			"pod", podRef,
			"previousTimestamp", prevPod.Containers[containerName].Timestamp,
			"timestamp", newPoint.Timestamp)
	}
	return false
}
