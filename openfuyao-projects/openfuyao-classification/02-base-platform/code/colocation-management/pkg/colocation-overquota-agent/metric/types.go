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


Original file: https://github.com/kubernetes-sigs/metrics-server/tree/helm-chart-3.6.0/pkg/storage/types.go
Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
Modifications:
Add struct PodRef
Add function  NewPodRef
Modified struct MetricsBatch ContanersMetricsPoint NodeUsageReport
*/

package metric

import (
	"fmt"
	"math"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"sigs.k8s.io/metrics-server/pkg/api"

	"openfuyao.com/colocation-management/pkg/common"
)

type PodRef string

func NewPodRef(Namespace, Name string) PodRef {
	return PodRef(fmt.Sprintf("%s%s%s", Namespace, "/", Name))
}

func (r PodRef) NamespaceAndName() (string, string, error) {
	s := strings.Split(string(r), "/")
	if len(s) != 2 {
		return "", "", fmt.Errorf("invlid podRef: %v", r)
	}
	if s[0] == "" || s[1] == "" {
		return "", "", fmt.Errorf("invlid podRef: %v", r)
	}
	return s[0], s[1], nil
}

// MetricsBatch is a single batch of pod, container, and node metrics from some source.
type MetricsBatch struct {
	Node       MetricsPoint
	Pods       map[PodRef]MetricsPoint
	Containers map[PodRef]ContanersMetricsPoint
}

// ContanersMetricsPoint contains the metrics for some pod's containers.
type ContanersMetricsPoint struct {
	Containers map[string]MetricsPoint
}

// MetricsPoint represents the a set of specific metrics at some point in time.
type MetricsPoint struct {
	// StartTime is the start time of container/node. Cumulative CPU usage at that moment should be equal zero.
	StartTime time.Time
	// Timestamp is the time when metric point was measured. If CPU and Memory was measured at different time it should equal CPU time to allow accurate CPU calculation.
	Timestamp time.Time
	// CumulativeCpuUsed is the cumulative cpu used at Timestamp from the StartTime of container/node. Unit: nano core * seconds.
	CumulativeCpuUsed uint64
	// MemoryUsage is the working set size. Unit: bytes.
	MemoryUsage uint64
}

type NodeUsageReport struct {
	Node string `json:"node"`
	// The following fields define time interval from which metrics were
	// collected from the interval [Timestamp-Window, Timestamp].
	Timestamp metav1.Time     `json:"timestamp"`
	Window    metav1.Duration `json:"window"`

	// key: namespace + / +  pod name
	Pods               map[PodRef]corev1.ResourceList `json:"pods"` // only for LS pods
	PodsFromContainers map[PodRef]corev1.ResourceList `json:"pods_from_containers"`
	// key: namespace + / +  pod name and container name
	Containers map[PodRef]map[string]corev1.ResourceList `json:"containers"`
}

type NodeUsageResponse[T metav1.Time] struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data T      `json:"data"`
}

func resourceUsage(last, prev MetricsPoint) (corev1.ResourceList, api.TimeInfo, error) {
	if last.StartTime.Before(prev.StartTime) {
		return corev1.ResourceList{}, api.TimeInfo{}, fmt.Errorf("unexpected decrease in startTime of node/container")
	}
	if last.CumulativeCpuUsed < prev.CumulativeCpuUsed {
		return corev1.ResourceList{}, api.TimeInfo{}, fmt.Errorf("unexpected decrease in cumulative CPU usage value")
	}
	window := last.Timestamp.Sub(prev.Timestamp)
	cpuUsage := float64(last.CumulativeCpuUsed-prev.CumulativeCpuUsed) / window.Seconds()
	return corev1.ResourceList{
			corev1.ResourceCPU:    uint64Quantity(uint64(cpuUsage), resource.DecimalSI, -9),
			corev1.ResourceMemory: uint64Quantity(last.MemoryUsage, resource.BinarySI, 0),
		}, api.TimeInfo{
			Timestamp: last.Timestamp,
			Window:    window,
		}, nil
}

// uint64Quantity converts a uint64 into a Quantity, which only has constructors
// that work with int64 (except for parse, which requires costly round-trips to string).
// We lose precision until we fit in an int64 if greater than the max int64 value.
func uint64Quantity(val uint64, format resource.Format, scale resource.Scale) resource.Quantity {
	q := *resource.NewScaledQuantity(int64(val), scale)
	if val > math.MaxInt64 {
		// lose an decimal order-of-magnitude precision,
		// so we can fit into a scaled quantity
		klog.V(common.GeneralDebugLog).InfoS("Found unexpectedly large resource value, losing precision to fit in scaled resource.Quantity", "value", val)
		q = *resource.NewScaledQuantity(int64(val/10), resource.Scale(1)+scale)
	}
	q.Format = format
	return q
}

func PrintNamespaceNamesForContainers(m map[PodRef]ContanersMetricsPoint) (r string) {
	if len(m) != 0 {
		for k := range m {
			r += fmt.Sprintf("Namespace/Name:%v\n", k)
		}
	}
	return
}

func PrintNamespaceNamesForPods(m map[PodRef]MetricsPoint) (r string) {
	if len(m) != 0 {
		for k := range m {
			r += fmt.Sprintf("Namespace/Name:%v\n", k)
		}
	}
	return
}

func PrintNamespaceNamesForResList(m map[PodRef]map[string]corev1.ResourceList) (r string) {
	if len(m) != 0 {
		for k := range m {
			r += fmt.Sprintf("Namespace/Name:%v\n", k)
		}
	}
	return
}

func PrintNamespaceNamesForResList2(m map[PodRef]corev1.ResourceList) (r string) {
	if len(m) != 0 {
		for k := range m {
			r += fmt.Sprintf("Namespace/Name:%v\n", k)
		}
	}
	return
}

func PrintObjMeta(s []*metav1.PartialObjectMetadata) (r string) {
	if len(s) != 0 {
		for _, m := range s {
			r += fmt.Sprintf("Namespace:%v, Name:%v\n", m.Namespace, m.Name)
		}
	}
	return
}

func SumResourceList(res1, res2 corev1.ResourceList) corev1.ResourceList {
	if res1 == nil {
		return res2
	}
	if res2 == nil {
		return res1
	}
	result := corev1.ResourceList{}
	for resourceName, quantity := range res1 {
		if r2, ok := res2[resourceName]; ok {
			var r resource.Quantity = quantity
			r.Add(r2)
			result[resourceName] = r
		} else {
			result[resourceName] = quantity
		}
	}
	for resourceName, quantity := range res2 {
		if r1, ok := res1[resourceName]; ok {
			var r resource.Quantity = quantity
			r.Add(r1)
			result[resourceName] = r
		} else {
			result[resourceName] = quantity
		}
	}
	return result
}
