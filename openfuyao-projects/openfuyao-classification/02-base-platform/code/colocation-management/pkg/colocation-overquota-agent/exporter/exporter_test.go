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
* Date: 2025-09-05
 */

package exporter

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/prometheus/client_golang/prometheus/testutil"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/colocation-management/pkg/common"
)

func TestCollectPodMetrics(t *testing.T) {
	podBeCpuRequest.Reset()
	podBeMemoryRequest.Reset()
	podCpuUsage.Reset()
	podMemoryUsage.Reset()

	tests := []struct {
		name       string
		podExports []*PodExport
		expected   map[string]float64
	}{
		{
			name: "BE QoS pod metrics",
			podExports: []*PodExport{
				{
					Namespace:       "default",
					Name:            "test-pod-be",
					NodeName:        "node-1",
					BeCpuRequest:    2.5,
					BeMemoryRequest: 1024.0,
					CpuUsage:        1.8,
					MemoryUsage:     512.0,
					QosLevel:        string(common.QosBe),
				},
			},
			expected: map[string]float64{
				"pod_overquota_cpu_request":      2.5,
				"pod_overquota_memory_request":   1024.0,
				"pod_cpu_usage_by_colocation":    1.8,
				"pod_memory_usage_by_colocation": 512.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podBeCpuRequest.Reset()
			podBeMemoryRequest.Reset()
			podCpuUsage.Reset()
			podMemoryUsage.Reset()

			CollectPodMetrics(tt.podExports)

			if tt.podExports[0].QosLevel == string(common.QosBe) {
				cpuRequestValue := testutil.ToFloat64(podBeCpuRequest.WithLabelValues(
					tt.podExports[0].Namespace, tt.podExports[0].Name, tt.podExports[0].NodeName))
				memoryRequestValue := testutil.ToFloat64(podBeMemoryRequest.WithLabelValues(
					tt.podExports[0].Namespace, tt.podExports[0].Name, tt.podExports[0].NodeName))

				if cpuRequestValue != tt.expected["pod_overquota_cpu_request"] {
					t.Errorf("BE CPU request: expected %f, got %f", tt.expected["pod_overquota_cpu_request"], cpuRequestValue)
				}
				if memoryRequestValue != tt.expected["pod_overquota_memory_request"] {
					t.Errorf("BE Memory request: expected %f, got %f", tt.expected["pod_overquota_memory_request"], memoryRequestValue)
				}
			}

			cpuUsageValue := testutil.ToFloat64(podCpuUsage.WithLabelValues(
				tt.podExports[0].Namespace, tt.podExports[0].Name, tt.podExports[0].NodeName, tt.podExports[0].QosLevel))
			memoryUsageValue := testutil.ToFloat64(podMemoryUsage.WithLabelValues(
				tt.podExports[0].Namespace, tt.podExports[0].Name, tt.podExports[0].NodeName, tt.podExports[0].QosLevel))

			if cpuUsageValue != tt.expected["pod_cpu_usage_by_colocation"] {
				t.Errorf("CPU usage: expected %f, got %f", tt.expected["pod_cpu_usage_by_colocation"], cpuUsageValue)
			}
			if memoryUsageValue != tt.expected["pod_memory_usage_by_colocation"] {
				t.Errorf("Memory usage: expected %f, got %f", tt.expected["pod_memory_usage_by_colocation"], memoryUsageValue)
			}
		})
	}
}

func TestCollectNodeMetrics(t *testing.T) {
	nodeBeCpuAllocatable.Reset()
	nodeBeMemoryAllocatable.Reset()

	tests := []struct {
		name     string
		node     *v1.Node
		expected map[string]float64
	}{
		{
			name: "node with BE resources",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						v1.ResourceName(common.ExtenderResourceCPU):    resource.MustParse("4000m"),
						v1.ResourceName(common.ExtenderResourceMemory): resource.MustParse("8Gi"),
					},
				},
			},
			expected: map[string]float64{
				"node_overquota_cpu_allocatable":    4.0,
				"node_overquota_memory_allocatable": 8589934592.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodeBeCpuAllocatable.Reset()
			nodeBeMemoryAllocatable.Reset()

			CollectNodeMetrics(tt.node)

			cpuValue := testutil.ToFloat64(nodeBeCpuAllocatable.WithLabelValues(tt.node.Name))
			if cpuValue != tt.expected["node_overquota_cpu_allocatable"] {
				t.Errorf("CPU allocatable: expected %f, got %f", tt.expected["node_overquota_cpu_allocatable"], cpuValue)
			}

			memoryValue := testutil.ToFloat64(nodeBeMemoryAllocatable.WithLabelValues(tt.node.Name))
			if memoryValue != tt.expected["node_overquota_memory_allocatable"] {
				t.Errorf("Memory allocatable: expected %f, got %f", tt.expected["node_overquota_memory_allocatable"], memoryValue)
			}
		})
	}
}

func TestDeletePodMetrics(t *testing.T) {
	tests := []struct {
		name        string
		setupPod    *PodExport
		deletePod   *PodExport
		shouldExist bool
	}{
		{
			name: "delete BE pod metrics",
			setupPod: &PodExport{
				Namespace:       "default",
				Name:            "test-pod",
				NodeName:        "node-1",
				BeCpuRequest:    2.0,
				BeMemoryRequest: 1024.0,
				CpuUsage:        1.5,
				MemoryUsage:     512.0,
				QosLevel:        string(common.QosBe),
			},
			deletePod: &PodExport{
				Namespace: "default",
				Name:      "test-pod",
				NodeName:  "node-1",
				QosLevel:  string(common.QosBe),
			},
			shouldExist: false,
		},
		{
			name: "delete LS pod metrics",
			setupPod: &PodExport{
				Namespace:   "kube-system",
				Name:        "test-pod-ls",
				NodeName:    "node-2",
				CpuUsage:    0.8,
				MemoryUsage: 256.0,
				QosLevel:    string(common.QosLs),
			},
			deletePod: &PodExport{
				Namespace: "kube-system",
				Name:      "test-pod-ls",
				NodeName:  "node-2",
				QosLevel:  string(common.QosLs),
			},
			shouldExist: false,
		},
		{
			name: "delete HLS pod metrics",
			setupPod: &PodExport{
				Namespace:   "default",
				Name:        "test-pod-hls",
				NodeName:    "node-hls",
				CpuUsage:    2.2,
				MemoryUsage: 2048.0,
				QosLevel:    string(common.QosHLS),
			},
			deletePod: &PodExport{
				Namespace: "default",
				Name:      "test-pod-hls",
				NodeName:  "node-hls",
				QosLevel:  string(common.QosHLS),
			},
			shouldExist: false,
		},
		{
			name: "delete None QoS pod metrics",
			setupPod: &PodExport{
				Namespace:   "default",
				Name:        "test-pod-none",
				NodeName:    "node-none",
				CpuUsage:    3.3,
				MemoryUsage: 3072.0,
				QosLevel:    string(common.QosNone),
			},
			deletePod: &PodExport{
				Namespace: "default",
				Name:      "test-pod-none",
				NodeName:  "node-none",
				QosLevel:  string(common.QosNone),
			},
			shouldExist: false,
		},
		{
			name: "delete unknown QoS pod metrics (default branch)",
			setupPod: &PodExport{
				Namespace: "default",
				Name:      "test-pod-unknown",
				NodeName:  "node-unknown",
				QosLevel:  "bogus",
			},
			deletePod: &PodExport{
				Namespace: "default",
				Name:      "test-pod-unknown",
				NodeName:  "node-unknown",
				QosLevel:  "bogus",
			},
			shouldExist: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podBeCpuRequest.Reset()
			podBeMemoryRequest.Reset()
			podCpuUsage.Reset()
			podMemoryUsage.Reset()

			CollectPodMetrics([]*PodExport{tt.setupPod})

			DeletePodMetrics(tt.deletePod)

			if tt.setupPod.QosLevel == string(common.QosBe) {
				cpuRequestValue := testutil.ToFloat64(podBeCpuRequest.WithLabelValues(
					tt.setupPod.Namespace, tt.setupPod.Name, tt.setupPod.NodeName))
				memoryRequestValue := testutil.ToFloat64(podBeMemoryRequest.WithLabelValues(
					tt.setupPod.Namespace, tt.setupPod.Name, tt.setupPod.NodeName))

				if (cpuRequestValue != 0) != tt.shouldExist {
					t.Errorf("BE CPU request metric existence: expected %v, got %v", tt.shouldExist, cpuRequestValue != 0)
				}
				if (memoryRequestValue != 0) != tt.shouldExist {
					t.Errorf("BE Memory request metric existence: expected %v, got %v", tt.shouldExist, memoryRequestValue != 0)
				}
			}

			cpuUsageValue := testutil.ToFloat64(podCpuUsage.WithLabelValues(
				tt.setupPod.Namespace, tt.setupPod.Name, tt.setupPod.NodeName, tt.setupPod.QosLevel))
			memoryUsageValue := testutil.ToFloat64(podMemoryUsage.WithLabelValues(
				tt.setupPod.Namespace, tt.setupPod.Name, tt.setupPod.NodeName, tt.setupPod.QosLevel))

			if (cpuUsageValue != 0) != tt.shouldExist {
				t.Errorf("CPU usage metric existence: expected %v, got %v", tt.shouldExist, cpuUsageValue != 0)
			}
			if (memoryUsageValue != 0) != tt.shouldExist {
				t.Errorf("Memory usage metric existence: expected %v, got %v", tt.shouldExist, memoryUsageValue != 0)
			}
		})
	}
}

func TestNewIncludeExporter(t *testing.T) {
	tests := []struct {
		name         string
		exporterPort int
		expectedPort string
	}{
		{
			name:         "custom port",
			exporterPort: 9090,
			expectedPort: ":9090",
		},
		{
			name:         "default port",
			exporterPort: 0,
			expectedPort: ":8900",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, err := NewIncludeExporter(tt.exporterPort)
			if err != nil {
				t.Fatalf("Failed to create exporter: %v", err)
			}
			defer server.Close()

			if server.Addr != tt.expectedPort {
				t.Errorf("Expected server address %s, got %s", tt.expectedPort, server.Addr)
			}

			if server.Handler == nil {
				t.Error("Expected non-nil handler")
			}
		})
	}
}

func TestMetricsEndpoint(t *testing.T) {
	server := httptest.NewServer(promhttp.Handler())
	defer server.Close()

	podBeCpuRequest.WithLabelValues("test-ns", "test-pod", "test-node").Set(2.5)
	podCpuUsage.WithLabelValues("test-ns", "test-pod", "test-node", "be").Set(1.8)

	resp, err := http.Get(server.URL + "/metrics")
	if err != nil {
		t.Fatalf("Failed to get metrics: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if len(body) == 0 {
		t.Error("Expected non-empty metrics response")
	}
}

func TestNewIncludeExporterServesMetrics(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to get free port: %v", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatalf("Failed to release free port: %v", err)
	}

	server, err := NewIncludeExporter(port)
	if err != nil {
		t.Fatalf("Failed to create exporter: %v", err)
	}
	defer server.Close()

	url := fmt.Sprintf("http://127.0.0.1:%d/metrics", port)
	var resp *http.Response
	for i := 0; i < 10; i++ {
		resp, err = http.Get(url)
		if err == nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("Failed to request metrics endpoint: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Expected status 200, got %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if !strings.Contains(string(body), "pod_overquota_cpu_request") {
		t.Fatalf("Expected metrics output to contain pod_overquota_cpu_request")
	}
}

func TestCollectPodMetricsAllQosLevels(t *testing.T) {
	// Cover the HLS, None and default (unknown) QoS branches of CollectPodMetrics,
	// which set only podCpuUsage/podMemoryUsage (or nothing for unknown qos).
	tests := []struct {
		name      string
		podExport *PodExport
		expectCpu bool // whether the cpu/memory usage metrics are expected to be set
	}{
		{
			name: "HLS QoS pod metrics",
			podExport: &PodExport{
				Namespace:   "ns-hls",
				Name:        "pod-hls",
				NodeName:    "node-hls",
				CpuUsage:    3.3,
				MemoryUsage: 330.0,
				QosLevel:    string(common.QosHLS),
			},
			expectCpu: true,
		},
		{
			name: "None QoS pod metrics",
			podExport: &PodExport{
				Namespace:   "ns-none",
				Name:        "pod-none",
				NodeName:    "node-none",
				CpuUsage:    4.4,
				MemoryUsage: 440.0,
				QosLevel:    string(common.QosNone),
			},
			expectCpu: true,
		},
		{
			name: "unknown QoS pod metrics (default branch)",
			podExport: &PodExport{
				Namespace:   "ns-unknown",
				Name:        "pod-unknown",
				NodeName:    "node-unknown",
				CpuUsage:    5.5,
				MemoryUsage: 550.0,
				QosLevel:    "INVALID_QOS",
			},
			expectCpu: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podBeCpuRequest.Reset()
			podBeMemoryRequest.Reset()
			podCpuUsage.Reset()
			podMemoryUsage.Reset()

			CollectPodMetrics([]*PodExport{tt.podExport})

			cpuUsageValue := testutil.ToFloat64(podCpuUsage.WithLabelValues(
				tt.podExport.Namespace, tt.podExport.Name, tt.podExport.NodeName, tt.podExport.QosLevel))
			memoryUsageValue := testutil.ToFloat64(podMemoryUsage.WithLabelValues(
				tt.podExport.Namespace, tt.podExport.Name, tt.podExport.NodeName, tt.podExport.QosLevel))

			if tt.expectCpu {
				if cpuUsageValue != tt.podExport.CpuUsage {
					t.Errorf("CPU usage: expected %f, got %f", tt.podExport.CpuUsage, cpuUsageValue)
				}
				if memoryUsageValue != tt.podExport.MemoryUsage {
					t.Errorf("Memory usage: expected %f, got %f", tt.podExport.MemoryUsage, memoryUsageValue)
				}
			} else {
				// default branch: no usage metrics recorded for the (non-existent) label set
				if cpuUsageValue != 0 {
					t.Errorf("unknown QoS should not set CPU usage, got %f", cpuUsageValue)
				}
				if memoryUsageValue != 0 {
					t.Errorf("unknown QoS should not set memory usage, got %f", memoryUsageValue)
				}
			}
		})
	}
}

func TestCollectPodMetricsMultipleMixedQos(t *testing.T) {
	// A single CollectPodMetrics call iterating over pods of several QoS levels
	// exercises the loop plus every switch case together.
	podBeCpuRequest.Reset()
	podBeMemoryRequest.Reset()
	podCpuUsage.Reset()
	podMemoryUsage.Reset()

	pods := []*PodExport{
		{Namespace: "ns", Name: "p-be", NodeName: "n", BeCpuRequest: 1.0, BeMemoryRequest: 10.0, CpuUsage: 0.5, MemoryUsage: 5.0, QosLevel: string(common.QosBe)},
		{Namespace: "ns", Name: "p-ls", NodeName: "n", CpuUsage: 0.6, MemoryUsage: 6.0, QosLevel: string(common.QosLs)},
		{Namespace: "ns", Name: "p-hls", NodeName: "n", CpuUsage: 0.7, MemoryUsage: 7.0, QosLevel: string(common.QosHLS)},
		{Namespace: "ns", Name: "p-none", NodeName: "n", CpuUsage: 0.8, MemoryUsage: 8.0, QosLevel: string(common.QosNone)},
		{Namespace: "ns", Name: "p-unknown", NodeName: "n", CpuUsage: 0.9, MemoryUsage: 9.0, QosLevel: "bogus"},
	}

	CollectPodMetrics(pods)

	beCpu := testutil.ToFloat64(podBeCpuRequest.WithLabelValues("ns", "p-be", "n"))
	if beCpu != 1.0 {
		t.Errorf("BE cpu request: expected 1.0, got %f", beCpu)
	}
	// LS / HLS / None all set cpu usage; verify each.
	for _, p := range pods[:4] {
		v := testutil.ToFloat64(podCpuUsage.WithLabelValues(p.Namespace, p.Name, p.NodeName, p.QosLevel))
		if v != p.CpuUsage {
			t.Errorf("cpu usage for %s: expected %f, got %f", p.Name, p.CpuUsage, v)
		}
	}
	// unknown qos must not produce usage metrics
	if got := testutil.ToFloat64(podCpuUsage.WithLabelValues("ns", "p-unknown", "n", "bogus")); got != 0 {
		t.Errorf("unknown qos cpu usage expected 0, got %f", got)
	}
}

func TestEdgeCases(t *testing.T) {
	t.Run("empty pod exports", func(t *testing.T) {
		podBeCpuRequest.Reset()
		podBeMemoryRequest.Reset()
		podCpuUsage.Reset()
		podMemoryUsage.Reset()

		CollectPodMetrics([]*PodExport{})

		if testutil.CollectAndCount(podCpuUsage) > 0 {
			t.Error("Expected no metrics for empty pod exports")
		}
	})

	t.Run("nil pod exports", func(t *testing.T) {
		podBeCpuRequest.Reset()
		podBeMemoryRequest.Reset()
		podCpuUsage.Reset()
		podMemoryUsage.Reset()

		CollectPodMetrics(nil)

		if testutil.CollectAndCount(podCpuUsage) > 0 {
			t.Error("Expected no metrics for nil pod exports")
		}
	})

}

func BenchmarkCollectPodMetrics(b *testing.B) {
	podExports := make([]*PodExport, 1000)
	for i := 0; i < 1000; i++ {
		podExports[i] = &PodExport{
			Namespace:       fmt.Sprintf("ns-%d", i),
			Name:            fmt.Sprintf("pod-%d", i),
			NodeName:        "node-1",
			BeCpuRequest:    float64(i % 10),
			BeMemoryRequest: float64(i * 1024),
			CpuUsage:        float64(i%10) * 0.1,
			MemoryUsage:     float64(i * 512),
			QosLevel:        string(common.QosBe),
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CollectPodMetrics(podExports)
	}
}

func BenchmarkCollectNodeMetrics(b *testing.B) {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node",
		},
		Status: v1.NodeStatus{
			Allocatable: v1.ResourceList{
				v1.ResourceName(common.ExtenderResourceCPU):    resource.MustParse("4000m"),
				v1.ResourceName(common.ExtenderResourceMemory): resource.MustParse("8Gi"),
			},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CollectNodeMetrics(node)
	}
}
