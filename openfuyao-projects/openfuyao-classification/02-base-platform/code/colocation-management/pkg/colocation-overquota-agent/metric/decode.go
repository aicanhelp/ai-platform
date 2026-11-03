/*
// Copyright 2021 The Kubernetes Authors.
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


Original file: https://github.com/kubernetes-sigs/metrics-server/tree/0.6.0/scraper/client/resource/decode.go
Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
Modifications:
Added var podCpuUsageMetricName podMemUsageMetricName
Added struct DataMetrics
Modified function decodeBatch parsePodCpuMetrics parsePodLabels
Added function checkContainerMetrics checkPodMetrics parsePodCpuMetrics parsePodMemMetrics

*/

package metric

import (
	"bytes"
	"fmt"
	"io"
	"time"

	"github.com/prometheus/prometheus/model/textparse"
	"github.com/prometheus/prometheus/model/timestamp"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/common"
)

var (
	nodeCpuUsageMetricName       = []byte("node_cpu_usage_seconds_total")
	nodeMemUsageMetricName       = []byte("node_memory_working_set_bytes")
	podCpuUsageMetricName        = []byte("pod_cpu_usage_seconds_total")
	podMemUsageMetricName        = []byte("pod_memory_working_set_bytes")
	containerCpuUsageMetricName  = []byte("container_cpu_usage_seconds_total")
	containerMemUsageMetricName  = []byte("container_memory_working_set_bytes")
	containerStartTimeMetricName = []byte("container_start_time_seconds")
	nanosecondsPerSecond         = 1e9
	millisToNanos                = int64(1e6)
)

// DataMetrics encapsulates all metrics data in a single struct
type DataMetrics struct {
	ClusterMetrics   *MetricsPoint
	PodMetrics       map[PodRef]MetricsPoint
	ContainerMetrics map[PodRef]ContanersMetricsPoint
}

func decodeBatch(b []byte, defaultTime time.Time, nodeName string) (*MetricsBatch, error) {
	res := &MetricsBatch{
		Pods:       make(map[PodRef]MetricsPoint),
		Containers: make(map[PodRef]ContanersMetricsPoint),
	}
	parser, err := textparse.New(b, "", false)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Prometheus parser: %w", err)
	}
	metricsData, err := parseMetrics(parser, timestamp.FromTime(defaultTime))
	if err != nil {
		return nil, fmt.Errorf("failed parsing metrics: %w", err)
	}
	node := metricsData.ClusterMetrics
	if node.Timestamp.IsZero() || node.CumulativeCpuUsed == 0 || node.MemoryUsage == 0 {
		klog.V(common.BasicDebugLog).InfoS("Failed getting complete node metric", "node", nodeName, "metric", node)
		node = nil
	} else {
		res.Node = *node
	}

	res.Pods = checkPodMetrics(metricsData.PodMetrics)

	for podRef, containersMetric := range metricsData.ContainerMetrics {
		if len(containersMetric.Containers) != 0 {
			// drop container metrics when Timestamp is zero
			pm := ContanersMetricsPoint{
				Containers: checkContainerMetrics(containersMetric),
			}
			if pm.Containers == nil {
				klog.V(common.BasicDebugLog).InfoS("Failed getting complete Pod metric", "pod", podRef)
			} else {
				res.Containers[podRef] = pm
			}
		}
	}
	return res, nil
}

func parseMetrics(parser textparse.Parser,
	defaultTimestamp int64) (*DataMetrics, error) {
	node := &MetricsPoint{}
	pods := make(map[PodRef]MetricsPoint)
	containers := make(map[PodRef]ContanersMetricsPoint)
	var et textparse.Entry
	var err error

	for {
		if et, err = parser.Next(); err != nil {
			if err == io.EOF {
				break
			} else {
				return nil, fmt.Errorf("failed parsing metrics: %w", err)
			}
		}

		if et != textparse.EntrySeries {
			continue
		}
		timeseries, maybeTimestamp, value := parser.Series()
		if maybeTimestamp == nil {
			maybeTimestamp = &defaultTimestamp
		}
		timeseriesData := TimeseriesData{
			timeseries: timeseries,
			timestamp:  *maybeTimestamp,
			value:      value,
		}
		processed := processTimeseries(timeseriesData, node, pods, containers)
		if !processed {
			continue
		}
	}
	return &DataMetrics{
		ClusterMetrics:   node,
		PodMetrics:       pods,
		ContainerMetrics: containers,
	}, nil
}

// TimeseriesData 封装时间序列数据
type TimeseriesData struct {
	timeseries []byte
	timestamp  int64
	value      float64
}

func processTimeseries(td TimeseriesData, node *MetricsPoint,
	pods map[PodRef]MetricsPoint, containers map[PodRef]ContanersMetricsPoint) bool {
	switch {
	case timeseriesMatchesName(td.timeseries, nodeCpuUsageMetricName):
		parseNodeCpuUsageMetrics(td.timestamp, td.value, node)
		return true
	case timeseriesMatchesName(td.timeseries, nodeMemUsageMetricName):
		parseNodeMemUsageMetrics(td.timestamp, td.value, node)
		return true
	case timeseriesMatchesName(td.timeseries, podCpuUsageMetricName):
		parsePodCpuMetrics(parsePodLabels(td.timeseries[len(podCpuUsageMetricName):]), td.timestamp, td.value, pods)
		return true
	case timeseriesMatchesName(td.timeseries, podMemUsageMetricName):
		parsePodMemMetrics(parsePodLabels(td.timeseries[len(podMemUsageMetricName):]), td.timestamp, td.value, pods)
		return true
	case timeseriesMatchesName(td.timeseries, containerCpuUsageMetricName):
		namespaceName, containerName := parseContainerLabels(td.timeseries[len(containerCpuUsageMetricName):])
		parseContainerCpuMetrics(namespaceName, containerName, td.timestamp, td.value, containers)
		return true
	case timeseriesMatchesName(td.timeseries, containerMemUsageMetricName):
		namespaceName, containerName := parseContainerLabels(td.timeseries[len(containerMemUsageMetricName):])
		parseContainerMemMetrics(namespaceName, containerName, td.timestamp, td.value, containers)
		return true
	case timeseriesMatchesName(td.timeseries, containerStartTimeMetricName):
		namespaceName, containerName := parseContainerLabels(td.timeseries[len(containerStartTimeMetricName):])
		parseContainerStartTimeMetrics(namespaceName, containerName, td.timestamp, td.value, containers)
		return true
	default:
		return false
	}
}

func timeseriesMatchesName(ts, name []byte) bool {
	return bytes.HasPrefix(ts, name) && (len(ts) == len(name) || ts[len(name)] == '{')
}

func parseNodeCpuUsageMetrics(timestamp int64, value float64, node *MetricsPoint) {
	node.CumulativeCpuUsed = uint64(value * nanosecondsPerSecond)
	// unit of timestamp is millisecond, need to convert to nanosecond
	node.Timestamp = time.Unix(0, timestamp*millisToNanos)
}

func parseNodeMemUsageMetrics(timestamp int64, value float64, node *MetricsPoint) {
	node.MemoryUsage = uint64(value)
	// unit of timestamp is millisecond, need to convert to nanosecond
	node.Timestamp = time.Unix(0, timestamp*millisToNanos)
}

func parsePodCpuMetrics(namespaceName PodRef, timestamp int64, value float64, pods map[PodRef]MetricsPoint) {
	if _, findPod := pods[namespaceName]; !findPod {
		pods[namespaceName] = MetricsPoint{}
	}
	point := pods[namespaceName]
	point.CumulativeCpuUsed = uint64(value * nanosecondsPerSecond)
	point.Timestamp = time.Unix(0, timestamp*millisToNanos)
	pods[namespaceName] = point
}

func parsePodMemMetrics(namespaceName PodRef, timestamp int64, value float64, pods map[PodRef]MetricsPoint) {
	if _, findPod := pods[namespaceName]; !findPod {
		pods[namespaceName] = MetricsPoint{}
	}
	point := pods[namespaceName]
	point.MemoryUsage = uint64(value)
	point.Timestamp = time.Unix(0, timestamp*millisToNanos)
	pods[namespaceName] = point
}

func parseContainerCpuMetrics(namespaceName PodRef, containerName string,
	timestamp int64, value float64, pods map[PodRef]ContanersMetricsPoint) {
	if _, findPod := pods[namespaceName]; !findPod {
		pods[namespaceName] = ContanersMetricsPoint{Containers: make(map[string]MetricsPoint)}
	}
	if _, findContainer := pods[namespaceName].Containers[containerName]; !findContainer {
		pods[namespaceName].Containers[containerName] = MetricsPoint{}
	}
	// unit of node_cpu_usage_seconds_total is second, need to convert to nanosecond
	containerMetrics := pods[namespaceName].Containers[containerName]
	containerMetrics.CumulativeCpuUsed = uint64(value * nanosecondsPerSecond)
	// unit of timestamp is millisecond, need to convert to nanosecond
	containerMetrics.Timestamp = time.Unix(0, timestamp*millisToNanos)
	pods[namespaceName].Containers[containerName] = containerMetrics
}

func parseContainerMemMetrics(namespaceName PodRef, containerName string,
	timestamp int64, value float64, pods map[PodRef]ContanersMetricsPoint) {
	if _, findPod := pods[namespaceName]; !findPod {
		pods[namespaceName] = ContanersMetricsPoint{Containers: make(map[string]MetricsPoint)}
	}
	if _, findContainer := pods[namespaceName].Containers[containerName]; !findContainer {
		pods[namespaceName].Containers[containerName] = MetricsPoint{}
	}
	containerMetrics := pods[namespaceName].Containers[containerName]
	containerMetrics.MemoryUsage = uint64(value)
	// unit of timestamp is millisecond, need to convert to nanosecond
	containerMetrics.Timestamp = time.Unix(0, timestamp*millisToNanos)
	pods[namespaceName].Containers[containerName] = containerMetrics
}

func parseContainerStartTimeMetrics(namespaceName PodRef, containerName string,
	timestamp int64, value float64, pods map[PodRef]ContanersMetricsPoint) {
	if _, findPod := pods[namespaceName]; !findPod {
		pods[namespaceName] = ContanersMetricsPoint{Containers: make(map[string]MetricsPoint)}
	}
	if _, findContainer := pods[namespaceName].Containers[containerName]; !findContainer {
		pods[namespaceName].Containers[containerName] = MetricsPoint{}
	}
	containerMetrics := pods[namespaceName].Containers[containerName]
	containerMetrics.StartTime = time.Unix(0, int64(value*nanosecondsPerSecond))
	pods[namespaceName].Containers[containerName] = containerMetrics
}

var (
	containerNameTag = []byte(`container="`)
	podNameTag       = []byte(`pod="`)
	namespaceTag     = []byte(`namespace="`)
)

func parsePodLabels(labels []byte) PodRef {
	i := bytes.Index(labels, podNameTag) + len(podNameTag)
	j := bytes.IndexByte(labels[i:], '"')
	Name := string(labels[i : i+j])
	i = bytes.Index(labels, namespaceTag) + len(namespaceTag)
	j = bytes.IndexByte(labels[i:], '"')
	Namespace := string(labels[i : i+j])
	return NewPodRef(Namespace, Name)
}

func parseContainerLabels(labels []byte) (PodRef, string) {
	i := bytes.Index(labels, containerNameTag) + len(containerNameTag)
	j := bytes.IndexByte(labels[i:], '"')
	containerName := string(labels[i : i+j])
	i = bytes.Index(labels, podNameTag) + len(podNameTag)
	j = bytes.IndexByte(labels[i:], '"')
	Name := string(labels[i : i+j])
	i = bytes.Index(labels, namespaceTag) + len(namespaceTag)
	j = bytes.IndexByte(labels[i:], '"')
	Namespace := string(labels[i : i+j])

	return NewPodRef(Namespace, Name), containerName
}

func checkPodMetrics(podsMetric map[PodRef]MetricsPoint) map[PodRef]MetricsPoint {
	result := make(map[PodRef]MetricsPoint)
	for podRef, podMetric := range podsMetric {
		if podMetric.CumulativeCpuUsed == 0 || podMetric.MemoryUsage == 0 {
			klog.V(common.BasicDebugLog).InfoS("Failed getting complete pod metric", "podRef", podRef, "podMetric", podMetric)
			continue
		}
		result[podRef] = podMetric
	}
	return result
}

func checkContainerMetrics(podMetric ContanersMetricsPoint) map[string]MetricsPoint {
	podMetrics := make(map[string]MetricsPoint)
	for containerName, containerMetric := range podMetric.Containers {
		if containerMetric != (MetricsPoint{}) {
			// drop metrics when CumulativeCpuUsed or MemoryUsage is zero
			if containerMetric.CumulativeCpuUsed == 0 || containerMetric.MemoryUsage == 0 {
				klog.V(common.BasicDebugLog).InfoS("Failed getting complete container metric",
					"containerName", containerName, "containerMetric", containerMetric)
				return nil
			} else {
				podMetrics[containerName] = containerMetric
			}
		}
	}
	return podMetrics
}
