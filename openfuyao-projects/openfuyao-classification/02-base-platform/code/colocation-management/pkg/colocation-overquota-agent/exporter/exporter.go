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

// Package exporter 用于监控指标暴露
package exporter

import (
	"fmt"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/common"
)

// CPU 结构体
type CPU struct {
	Time                 string `json:"time"`
	UsageNanoCores       uint64 `json:"usageNanoCores"`
	UsageCoreNanoSeconds uint64 `json:"usageCoreNanoSeconds"`
}

// Memory 结构体
type Memory struct {
	Time            string `json:"time"`
	UsageBytes      uint64 `json:"usageBytes"`
	WorkingSetBytes uint64 `json:"workingSetBytes"`
}

// KubeletStatsSummary 用于接收kubelet /stats/summary返回
type KubeletStatsSummary struct {
	Node struct {
		CPU    CPU    `json:"cpu"`
		Memory Memory `json:"memory"`
	} `json:"node"`
	Pods []struct {
		PodRef struct {
			Name      string `json:"name"`
			Namespace string `json:"namespace"`
		} `json:"podRef"`
		CPU        CPU    `json:"cpu"`
		Memory     Memory `json:"memory"`
		Containers []struct {
			Name   string `json:"name"`
			CPU    CPU    `json:"cpu"`
			Memory Memory `json:"memory"`
		} `json:"containers"`
	} `json:"pods"`
}

// PodExport 用于存储pod指标需要暴露的信息
type PodExport struct {
	Namespace       string  `json:"namespace"`
	Name            string  `json:"name"`
	NodeName        string  `json:"nodeName"`
	BeCpuRequest    float64 `json:"beCpuRequest"`
	BeMemoryRequest float64 `json:"beMemoryRequest"`
	CpuUsage        float64 `json:"cpuUsage"`
	MemoryUsage     float64 `json:"memoryUsage"`
	QosLevel        string  `json:"qosLevel"`
}

var (
	// 定义要暴露的指标
	nodeBeCpuAllocatable = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "node_overquota_cpu_allocatable",
			Help: "The allocatable BE-CPU for a node",
		},
		[]string{"node_name"},
	)

	// 定义要暴露的指标
	nodeBeMemoryAllocatable = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "node_overquota_memory_allocatable",
			Help: "The allocatable BE-Memory for a node",
		},
		[]string{"node_name"},
	)

	podBeCpuRequest = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "pod_overquota_cpu_request",
			Help: "The requested BE-CPU for a pod",
		},
		[]string{"namespace", "pod_name", "node"},
	)
	podBeMemoryRequest = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "pod_overquota_memory_request",
			Help: "The requested BE-Memory (in bytes) for a pod",
		},
		[]string{"namespace", "pod_name", "node"},
	)
	podCpuUsage = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "pod_cpu_usage_by_colocation",
			Help: "The actual CPU usage (in cores) for a pod",
		},
		[]string{"namespace", "pod_name", "node", "type"},
	)
	podMemoryUsage = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "pod_memory_usage_by_colocation",
			Help: "The actual Memory usage (in bytes) for a pod",
		},
		[]string{"namespace", "pod_name", "node", "type"},
	)
)

func init() {
	// 注册指标
	prometheus.MustRegister(nodeBeCpuAllocatable)
	prometheus.MustRegister(nodeBeMemoryAllocatable)
	prometheus.MustRegister(podBeCpuRequest)
	prometheus.MustRegister(podBeMemoryRequest)
	prometheus.MustRegister(podCpuUsage)
	prometheus.MustRegister(podMemoryUsage)
}

// NewIncludeExporter 暴露metrics服务
func NewIncludeExporter(exporterPort int) (*http.Server, error) {
	if exporterPort == 0 {
		exporterPort = 8900
	}

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())

	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", exporterPort),
		Handler: mux,
	}

	go func() {
		klog.Infof("Starting metrics exporter on port %d", exporterPort)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			klog.Errorf("Error starting server: %v", err)
		}
	}()
	return server, nil
}

// CollectPodMetrics 采集pod指标
func CollectPodMetrics(podExporter []*PodExport) {
	for _, exp := range podExporter {
		klog.V(common.TraceDebugLog).InfoS("CollectPodMetrics", "pod-name", exp.Name, "qos-level", exp.QosLevel,
			"cpu-usage", exp.CpuUsage, "mem-usage", exp.MemoryUsage)
		switch exp.QosLevel {
		case string(common.QosBe):
			podBeCpuRequest.WithLabelValues(exp.Namespace, exp.Name, exp.NodeName).Set(exp.BeCpuRequest)
			podBeMemoryRequest.WithLabelValues(exp.Namespace, exp.Name, exp.NodeName).Set(exp.BeMemoryRequest)
			podCpuUsage.WithLabelValues(exp.Namespace, exp.Name, exp.NodeName, exp.QosLevel).Set(exp.CpuUsage)
			podMemoryUsage.WithLabelValues(exp.Namespace, exp.Name, exp.NodeName, exp.QosLevel).Set(exp.MemoryUsage)
		case string(common.QosLs):
			podCpuUsage.WithLabelValues(exp.Namespace, exp.Name, exp.NodeName, exp.QosLevel).Set(exp.CpuUsage)
			podMemoryUsage.WithLabelValues(exp.Namespace, exp.Name, exp.NodeName, exp.QosLevel).Set(exp.MemoryUsage)
		case string(common.QosHLS):
			podCpuUsage.WithLabelValues(exp.Namespace, exp.Name, exp.NodeName, exp.QosLevel).Set(exp.CpuUsage)
			podMemoryUsage.WithLabelValues(exp.Namespace, exp.Name, exp.NodeName, exp.QosLevel).Set(exp.MemoryUsage)
		case string(common.QosNone):
			podCpuUsage.WithLabelValues(exp.Namespace, exp.Name, exp.NodeName, exp.QosLevel).Set(exp.CpuUsage)
			podMemoryUsage.WithLabelValues(exp.Namespace, exp.Name, exp.NodeName, exp.QosLevel).Set(exp.MemoryUsage)
		default:
			klog.Infof("unknown qos level:%v", exp.QosLevel)
		}
	}
}

// CollectNodeMetrics 采集node指标
func CollectNodeMetrics(node *v1.Node) {
	if beCpuAllocatable, found := node.Status.Allocatable[common.ExtenderResourceCPU]; found {
		// 默认上报时已经是 Millivalue，此处不再取 Millivalue
		klog.V(common.TraceDebugLog).InfoS("CollectNodeMetrics", "node-name", node.Name, "be-cpu-allocatable",
			float64(beCpuAllocatable.Value()))
		nodeBeCpuAllocatable.WithLabelValues(node.Name).Set(float64(beCpuAllocatable.Value()))
	}
	if beMemoryAllocatable, found := node.Status.Allocatable[common.ExtenderResourceMemory]; found {
		klog.V(common.TraceDebugLog).InfoS("CollectNodeMetrics", "node-name", node.Name, "be-memory-allocatable",
			float64(beMemoryAllocatable.Value()))
		nodeBeMemoryAllocatable.WithLabelValues(node.Name).Set(float64(beMemoryAllocatable.Value()))

	}
}

// DeletePodMetrics 删除pod指标
func DeletePodMetrics(pe *PodExport) {
	switch pe.QosLevel {
	case string(common.QosBe):
		podBeCpuRequest.DeleteLabelValues(pe.Namespace, pe.Name, pe.NodeName)
		podBeMemoryRequest.DeleteLabelValues(pe.Namespace, pe.Name, pe.NodeName)
		podCpuUsage.DeleteLabelValues(pe.Namespace, pe.Name, pe.NodeName, pe.QosLevel)
		podMemoryUsage.DeleteLabelValues(pe.Namespace, pe.Name, pe.NodeName, pe.QosLevel)
	case string(common.QosLs):
		podCpuUsage.DeleteLabelValues(pe.Namespace, pe.Name, pe.NodeName, pe.QosLevel)
		podMemoryUsage.DeleteLabelValues(pe.Namespace, pe.Name, pe.NodeName, pe.QosLevel)
	case string(common.QosHLS):
		podCpuUsage.DeleteLabelValues(pe.Namespace, pe.Name, pe.NodeName, pe.QosLevel)
		podMemoryUsage.DeleteLabelValues(pe.Namespace, pe.Name, pe.NodeName, pe.QosLevel)
	case string(common.QosNone):
		podCpuUsage.DeleteLabelValues(pe.Namespace, pe.Name, pe.NodeName, pe.QosLevel)
		podMemoryUsage.DeleteLabelValues(pe.Namespace, pe.Name, pe.NodeName, pe.QosLevel)
	default:
		klog.Infof("unknown qos level:%v", pe.QosLevel)
	}
}
