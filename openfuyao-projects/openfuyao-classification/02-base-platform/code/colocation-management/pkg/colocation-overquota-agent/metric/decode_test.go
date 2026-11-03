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
* Date: 2025-08-18
 */

package metric

import (
	"testing"
	"time"

	"github.com/prometheus/prometheus/model/textparse"
	"github.com/prometheus/prometheus/model/timestamp"
	"github.com/stretchr/testify/assert"
)

// MockPrometheusData 生成测试用的Prometheus指标数据
func MockPrometheusData() []byte {
	return []byte(`
# HELP node_cpu_usage_seconds_total Total CPU usage in seconds
# TYPE node_cpu_usage_seconds_total counter
node_cpu_usage_seconds_total{node="test-node"} 123.456 1609459200000

# HELP node_memory_working_set_bytes Memory working set in bytes
# TYPE node_memory_working_set_bytes gauge
node_memory_working_set_bytes{node="test-node"} 1073741824 1609459200000

# HELP pod_cpu_usage_seconds_total Total CPU usage in seconds
# TYPE pod_cpu_usage_seconds_total counter
pod_cpu_usage_seconds_total{namespace="default",pod="test-pod"} 45.678 1609459200000

# HELP pod_memory_working_set_bytes Memory working set in bytes
# TYPE pod_memory_working_set_bytes gauge
pod_memory_working_set_bytes{namespace="default",pod="test-pod"} 536870912 1609459200000

# HELP container_cpu_usage_seconds_total Total CPU usage in seconds
# TYPE container_cpu_usage_seconds_total counter
container_cpu_usage_seconds_total{namespace="default",pod="test-pod",container="app"} 12.345 1609459200000

# HELP container_memory_working_set_bytes Memory working set in bytes
# TYPE container_memory_working_set_bytes gauge
container_memory_working_set_bytes{namespace="default",pod="test-pod",container="app"} 268435456 1609459200000

# HELP container_start_time_seconds Start time of the container
# TYPE container_start_time_seconds gauge
container_start_time_seconds{namespace="default",pod="test-pod",container="app"} 1609459100.0 1609459200000
`)
}

// MockIncompletePrometheusData 生成不完整的指标数据
func MockIncompletePrometheusData() []byte {
	return []byte(`
node_cpu_usage_seconds_total{node="test-node"} 123.456 1609459200000
# Missing memory metric intentionally
`)
}

// MockMalformedPrometheusData 生成格式错误的指标数据
func MockMalformedPrometheusData() []byte {
	return []byte(`
invalid metric line
node_cpu_usage_seconds_total{node="test-node"} invalid_value 1609459200000
`)
}

// TestDecodeBatch
func TestDecodeBatch(t *testing.T) {

	tests := []struct {
		name           string
		inputData      []byte
		defaultTime    time.Time
		nodeName       string
		expectedError  bool
		expectedResult *MetricsBatch
	}{
		{
			name:          "complete metrics data",
			inputData:     MockPrometheusData(),
			defaultTime:   time.Unix(0, 0),
			nodeName:      "test-node",
			expectedError: false,
			expectedResult: &MetricsBatch{
				Node: MetricsPoint{
					CumulativeCpuUsed: 123456000000,
					MemoryUsage:       1073741824,
					Timestamp:         time.Unix(0, 1609459200000*1000000),
				},
				Pods: map[PodRef]MetricsPoint{
					NewPodRef("default", "test-pod"): {
						CumulativeCpuUsed: 45678000000,
						MemoryUsage:       536870912,
						Timestamp:         time.Unix(0, 1609459200000*1000000),
					},
				},
				Containers: map[PodRef]ContanersMetricsPoint{
					NewPodRef("default", "test-pod"): {
						Containers: map[string]MetricsPoint{
							"app": {
								CumulativeCpuUsed: 12345000000,
								MemoryUsage:       268435456,
								StartTime:         time.Unix(0, 1609459100*1000000000),
								Timestamp:         time.Unix(0, 1609459200000*1000000),
							},
						},
					},
				},
			},
		},
		{
			name:          "incomplete metrics data",
			inputData:     MockIncompletePrometheusData(),
			defaultTime:   time.Unix(0, 0),
			nodeName:      "test-node",
			expectedError: false,
			expectedResult: &MetricsBatch{
				Node:       MetricsPoint{},
				Pods:       make(map[PodRef]MetricsPoint),
				Containers: make(map[PodRef]ContanersMetricsPoint),
			},
		},
		{
			name:           "malformed metrics data",
			inputData:      MockMalformedPrometheusData(),
			defaultTime:    time.Unix(0, 0),
			nodeName:       "test-node",
			expectedError:  true,
			expectedResult: nil,
		},
		{
			name:          "empty data",
			inputData:     []byte{},
			defaultTime:   time.Unix(0, 0),
			nodeName:      "test-node",
			expectedError: false,
			expectedResult: &MetricsBatch{
				Node:       MetricsPoint{},
				Pods:       make(map[PodRef]MetricsPoint),
				Containers: make(map[PodRef]ContanersMetricsPoint),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := decodeBatch(tt.inputData, tt.defaultTime, tt.nodeName)

			if tt.expectedError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if tt.expectedResult.Node.CumulativeCpuUsed != 0 {
				if result.Node.CumulativeCpuUsed != tt.expectedResult.Node.CumulativeCpuUsed {
					t.Errorf("Node CPU usage mismatch: expected %d, got %d",
						tt.expectedResult.Node.CumulativeCpuUsed, result.Node.CumulativeCpuUsed)
				}
				if result.Node.MemoryUsage != tt.expectedResult.Node.MemoryUsage {
					t.Errorf("Node memory usage mismatch: expected %d, got %d",
						tt.expectedResult.Node.MemoryUsage, result.Node.MemoryUsage)
				}
			}

			if len(tt.expectedResult.Pods) > 0 {
				if len(result.Pods) != len(tt.expectedResult.Pods) {
					t.Errorf("Expected %d pods, got %d", len(tt.expectedResult.Pods), len(result.Pods))
				}

				for podRef, expectedPod := range tt.expectedResult.Pods {
					if actualPod, exists := result.Pods[podRef]; exists {
						if actualPod.CumulativeCpuUsed != expectedPod.CumulativeCpuUsed {
							t.Errorf("Pod %s CPU usage mismatch: expected %d, got %d",
								podRef, expectedPod.CumulativeCpuUsed, actualPod.CumulativeCpuUsed)
						}
					} else {
						t.Errorf("Expected pod %s not found", podRef)
					}
				}
			}

			if len(tt.expectedResult.Containers) > 0 {
				if len(result.Containers) != len(tt.expectedResult.Containers) {
					t.Errorf("Expected %d container groups, got %d",
						len(tt.expectedResult.Containers), len(result.Containers))
				}

				for podRef, expectedContainers := range tt.expectedResult.Containers {
					if actualContainers, exists := result.Containers[podRef]; exists {
						if len(actualContainers.Containers) != len(expectedContainers.Containers) {
							t.Errorf("Expected %d containers for pod %s, got %d",
								len(expectedContainers.Containers), podRef, len(actualContainers.Containers))
						}

						for containerName, expectedContainer := range expectedContainers.Containers {
							if actualContainer, exists := actualContainers.Containers[containerName]; exists {
								if actualContainer.CumulativeCpuUsed != expectedContainer.CumulativeCpuUsed {
									t.Errorf("Container %s CPU usage mismatch: expected %d, got %d",
										containerName, expectedContainer.CumulativeCpuUsed, actualContainer.CumulativeCpuUsed)
								}
							} else {
								t.Errorf("Expected container %s not found in pod %s", containerName, podRef)
							}
						}
					} else {
						t.Errorf("Expected container group for pod %s not found", podRef)
					}
				}
			}

		})
	}
}

// TestTimeseriesMatchesName 测试timeseriesMatchesName函数
func TestTimeseriesMatchesName(t *testing.T) {
	tests := []struct {
		name     string
		ts       []byte
		match    []byte
		expected bool
	}{
		{
			name:     "exact match",
			ts:       []byte("node_cpu_usage_seconds_total"),
			match:    []byte("node_cpu_usage_seconds_total"),
			expected: true,
		},
		{
			name:     "match with labels",
			ts:       []byte(`node_cpu_usage_seconds_total{node="test"}`),
			match:    []byte("node_cpu_usage_seconds_total"),
			expected: true,
		},
		{
			name:     "no match",
			ts:       []byte("different_metric"),
			match:    []byte("node_cpu_usage_seconds_total"),
			expected: false,
		},
		{
			name:     "prefix but no brace",
			ts:       []byte("node_cpu_usage_seconds_total_extra"),
			match:    []byte("node_cpu_usage_seconds_total"),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := timeseriesMatchesName(tt.ts, tt.match)
			if result != tt.expected {
				t.Errorf("timeseriesMatchesName(%s, %s) = %v, expected %v",
					string(tt.ts), string(tt.match), result, tt.expected)
			}
		})
	}
}

// TestParsePodLabels 测试parsePodLabels函数
func TestParsePodLabels(t *testing.T) {
	tests := []struct {
		name     string
		labels   []byte
		expected PodRef
	}{
		{
			name:     "standard labels",
			labels:   []byte(`{namespace="default",pod="test-pod"}`),
			expected: NewPodRef("default", "test-pod"),
		},
		{
			name:     "labels with extra fields",
			labels:   []byte(`{namespace="kube-system",pod="coredns",instance="10.0.0.1"}`),
			expected: NewPodRef("kube-system", "coredns"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parsePodLabels(tt.labels)
			if result != tt.expected {
				t.Errorf("parsePodLabels(%s) = %v, expected %v",
					string(tt.labels), result, tt.expected)
			}
		})
	}
}

// TestParseContainerLabels 测试parseContainerLabels函数
func TestParseContainerLabels(t *testing.T) {
	tests := []struct {
		name              string
		labels            []byte
		expectedPod       PodRef
		expectedContainer string
	}{
		{
			name:              "standard labels",
			labels:            []byte(`{namespace="default",pod="test-pod",container="app"}`),
			expectedPod:       NewPodRef("default", "test-pod"),
			expectedContainer: "app",
		},
		{
			name:              "labels with extra fields",
			labels:            []byte(`{namespace="kube-system",pod="coredns",container="sidecar",image="alpine"}`),
			expectedPod:       NewPodRef("kube-system", "coredns"),
			expectedContainer: "sidecar",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podRef, containerName := parseContainerLabels(tt.labels)
			if podRef != tt.expectedPod {
				t.Errorf("PodRef mismatch: expected %v, got %v", tt.expectedPod, podRef)
			}
			if containerName != tt.expectedContainer {
				t.Errorf("Container name mismatch: expected %s, got %s", tt.expectedContainer, containerName)
			}
		})
	}
}

// TestCheckPodMetrics 测试checkPodMetrics函数
func TestCheckPodMetrics(t *testing.T) {
	tests := []struct {
		name     string
		input    map[PodRef]MetricsPoint
		expected map[PodRef]MetricsPoint
	}{
		{
			name: "complete metrics",
			input: map[PodRef]MetricsPoint{
				NewPodRef("default", "pod1"): {
					CumulativeCpuUsed: 1000,
					MemoryUsage:       1024,
				},
			},
			expected: map[PodRef]MetricsPoint{
				NewPodRef("default", "pod1"): {
					CumulativeCpuUsed: 1000,
					MemoryUsage:       1024,
				},
			},
		},
		{
			name: "incomplete metrics - missing cpu",
			input: map[PodRef]MetricsPoint{
				NewPodRef("default", "pod1"): {
					MemoryUsage: 1024,
				},
			},
			expected: map[PodRef]MetricsPoint{},
		},
		{
			name: "incomplete metrics - missing memory",
			input: map[PodRef]MetricsPoint{
				NewPodRef("default", "pod1"): {
					CumulativeCpuUsed: 1000,
				},
			},
			expected: map[PodRef]MetricsPoint{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := checkPodMetrics(tt.input)

			if len(result) != len(tt.expected) {
				t.Errorf("Expected %d pods, got %d", len(tt.expected), len(result))
			}

			for podRef, expected := range tt.expected {
				if actual, exists := result[podRef]; exists {
					if actual.CumulativeCpuUsed != expected.CumulativeCpuUsed {
						t.Errorf("CPU usage mismatch for pod %v", podRef)
					}
				} else {
					t.Errorf("Expected pod %v not found", podRef)
				}
			}
		})
	}
}

// TestCheckContainerMetrics 测试checkContainerMetrics函数
func TestCheckContainerMetrics(t *testing.T) {
	tests := []struct {
		name     string
		input    ContanersMetricsPoint
		expected map[string]MetricsPoint
	}{
		{
			name: "complete metrics",
			input: ContanersMetricsPoint{
				Containers: map[string]MetricsPoint{
					"container1": {
						CumulativeCpuUsed: 1000,
						MemoryUsage:       1024,
					},
				},
			},
			expected: map[string]MetricsPoint{
				"container1": {
					CumulativeCpuUsed: 1000,
					MemoryUsage:       1024,
				},
			},
		},
		{
			name: "incomplete metrics in one container",
			input: ContanersMetricsPoint{
				Containers: map[string]MetricsPoint{
					"container1": {
						CumulativeCpuUsed: 1000,
						MemoryUsage:       1024,
					},
					"container2": {
						CumulativeCpuUsed: 500,
					},
				},
			},
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := checkContainerMetrics(tt.input)

			if tt.expected == nil {
				if result != nil {
					t.Error("Expected nil result but got non-nil")
				}
				return
			}

			if len(result) != len(tt.expected) {
				t.Errorf("Expected %d containers, got %d", len(tt.expected), len(result))
			}
		})
	}
}

// TestParsePodMemMetrics covers the pod-already-present and pod-not-present branches.
func TestParsePodMemMetrics(t *testing.T) {
	podRef := NewPodRef("default", "pod1")

	t.Run("pod not present - creates entry", func(t *testing.T) {
		pods := make(map[PodRef]MetricsPoint)
		parsePodMemMetrics(podRef, 1609459200000, 536870912, pods)
		got, ok := pods[podRef]
		assert.True(t, ok)
		assert.Equal(t, uint64(536870912), got.MemoryUsage)
		assert.False(t, got.Timestamp.IsZero())
	})

	t.Run("pod already present - updates memory only", func(t *testing.T) {
		pods := map[PodRef]MetricsPoint{
			podRef: {CumulativeCpuUsed: 100},
		}
		parsePodMemMetrics(podRef, 1609459200000, 1024, pods)
		got := pods[podRef]
		assert.Equal(t, uint64(100), got.CumulativeCpuUsed, "existing cpu should be preserved")
		assert.Equal(t, uint64(1024), got.MemoryUsage)
	})
}

// TestParseContainerMemMetrics covers all three branches.
func TestParseContainerMemMetrics(t *testing.T) {
	podRef := NewPodRef("default", "pod1")

	t.Run("pod not present - creates pod and container", func(t *testing.T) {
		pods := make(map[PodRef]ContanersMetricsPoint)
		parseContainerMemMetrics(podRef, "c1", 1609459200000, 2048, pods)
		got, ok := pods[podRef]
		assert.True(t, ok)
		c, ok := got.Containers["c1"]
		assert.True(t, ok)
		assert.Equal(t, uint64(2048), c.MemoryUsage)
	})

	t.Run("pod present but container not present - creates container", func(t *testing.T) {
		pods := map[PodRef]ContanersMetricsPoint{
			podRef: {Containers: map[string]MetricsPoint{
				"other": {CumulativeCpuUsed: 50},
			}},
		}
		parseContainerMemMetrics(podRef, "c1", 1609459200000, 4096, pods)
		c, ok := pods[podRef].Containers["c1"]
		assert.True(t, ok)
		assert.Equal(t, uint64(4096), c.MemoryUsage)
		// existing container preserved
		assert.Equal(t, uint64(50), pods[podRef].Containers["other"].CumulativeCpuUsed)
	})

	t.Run("pod and container present - updates memory", func(t *testing.T) {
		pods := map[PodRef]ContanersMetricsPoint{
			podRef: {Containers: map[string]MetricsPoint{
				"c1": {CumulativeCpuUsed: 99},
			}},
		}
		parseContainerMemMetrics(podRef, "c1", 1609459200000, 8192, pods)
		c := pods[podRef].Containers["c1"]
		assert.Equal(t, uint64(99), c.CumulativeCpuUsed, "existing cpu preserved")
		assert.Equal(t, uint64(8192), c.MemoryUsage)
	})
}

// TestParseContainerStartTimeMetrics covers all three branches.
func TestParseContainerStartTimeMetrics(t *testing.T) {
	podRef := NewPodRef("default", "pod1")

	t.Run("pod not present - creates pod and container", func(t *testing.T) {
		pods := make(map[PodRef]ContanersMetricsPoint)
		parseContainerStartTimeMetrics(podRef, "c1", 1609459200000, 1609459100.0, pods)
		got, ok := pods[podRef]
		assert.True(t, ok)
		c, ok := got.Containers["c1"]
		assert.True(t, ok)
		assert.False(t, c.StartTime.IsZero())
	})

	t.Run("pod present but container not present - creates container", func(t *testing.T) {
		pods := map[PodRef]ContanersMetricsPoint{
			podRef: {Containers: map[string]MetricsPoint{
				"other": {MemoryUsage: 50},
			}},
		}
		parseContainerStartTimeMetrics(podRef, "c1", 1609459200000, 1609459100.0, pods)
		c, ok := pods[podRef].Containers["c1"]
		assert.True(t, ok)
		assert.False(t, c.StartTime.IsZero())
		// existing container preserved
		assert.Equal(t, uint64(50), pods[podRef].Containers["other"].MemoryUsage)
	})

	t.Run("pod and container present - updates startTime", func(t *testing.T) {
		pods := map[PodRef]ContanersMetricsPoint{
			podRef: {Containers: map[string]MetricsPoint{
				"c1": {CumulativeCpuUsed: 99, MemoryUsage: 7},
			}},
		}
		parseContainerStartTimeMetrics(podRef, "c1", 1609459200000, 1609459100.0, pods)
		c := pods[podRef].Containers["c1"]
		assert.Equal(t, uint64(99), c.CumulativeCpuUsed, "existing cpu preserved")
		assert.Equal(t, uint64(7), c.MemoryUsage, "existing mem preserved")
		assert.False(t, c.StartTime.IsZero())
	})
}

// TestDecodeBatchNilContainersMetrics covers the branch in decodeBatch where
// checkContainerMetrics returns nil (incomplete container metric), triggering
// the "Failed getting complete Pod metric" log path.
func TestDecodeBatchNilContainersMetrics(t *testing.T) {
	// container with cpu but no memory -> checkContainerMetrics returns nil
	data := []byte(`
container_cpu_usage_seconds_total{namespace="default",pod="pod1",container="app"} 12.345 1609459200000
container_start_time_seconds{namespace="default",pod="pod1",container="app"} 1609459100.0 1609459200000
`)
	res, err := decodeBatch(data, time.Unix(0, 0), "test-node")
	assert.NoError(t, err)
	// container metrics dropped because incomplete -> Containers map empty
	assert.Equal(t, 0, len(res.Containers))
}

// TestDecodeBatchParserError covers the parser-initialization error path of
// decodeBatch (textparse.New with an invalid content type name).
func TestDecodeBatchParserError(t *testing.T) {
	// An empty/whitespace input parses fine; trigger a real parse error with
	// malformed input that the prometheus parser rejects mid-stream.
	data := []byte(`{invalid`)
	_, err := decodeBatch(data, time.Unix(0, 0), "test-node")
	// The prometheus parser rejects "{" as an invalid start token, so this is
	// a guaranteed, deterministic parse error.
	assert.Error(t, err)
}

// TestParseMetricsDirect exercises parseMetrics directly including EOF break.
func TestParseMetricsDirect(t *testing.T) {
	parser, err := textparse.New(MockPrometheusData(), "", false)
	assert.NoError(t, err)
	dm, err := parseMetrics(parser, timestamp.FromTime(time.Unix(0, 0)))
	assert.NoError(t, err)
	assert.NotNil(t, dm)
	assert.NotNil(t, dm.ClusterMetrics)
	assert.Equal(t, 1, len(dm.PodMetrics))
	assert.Equal(t, 1, len(dm.ContainerMetrics))
}

// TestParseMetricsNoExplicitTimestamp covers the branch where a series has no
// explicit timestamp (maybeTimestamp == nil), so the default is applied.
func TestParseMetricsNoExplicitTimestamp(t *testing.T) {
	// Prometheus series without a trailing timestamp cause maybeTimestamp==nil.
	data := []byte(`node_cpu_usage_seconds_total{node="n"} 10.0
node_memory_working_set_bytes{node="n"} 2048
`)
	parser, err := textparse.New(data, "", false)
	assert.NoError(t, err)
	defaultTS := timestamp.FromTime(time.Unix(1600000000, 0))
	dm, err := parseMetrics(parser, defaultTS)
	assert.NoError(t, err)
	assert.NotNil(t, dm)
	// node timestamp should equal the default since no explicit timestamp was provided
	assert.False(t, dm.ClusterMetrics.Timestamp.IsZero())
}

// TestParseMetricsUnmatchedMetric covers the branch where processTimeseries
// returns false (unmatched metric) within parseMetrics' loop (the !processed continue).
func TestParseMetricsUnmatchedMetric(t *testing.T) {
	data := []byte(`some_unknown_metric_total{node="n"} 5.0 1609459200000
node_cpu_usage_seconds_total{node="n"} 10.0 1609459200000
`)
	parser, err := textparse.New(data, "", false)
	assert.NoError(t, err)
	dm, err := parseMetrics(parser, timestamp.FromTime(time.Unix(0, 0)))
	assert.NoError(t, err)
	assert.NotNil(t, dm)
	// unknown metric was skipped, known one processed
	assert.Equal(t, uint64(10)*uint64(1e9), dm.ClusterMetrics.CumulativeCpuUsed)
}

// TestProcessTimeseriesDefault covers the default (unmatched) return-false branch.
func TestProcessTimeseriesDefault(t *testing.T) {
	node := &MetricsPoint{}
	pods := make(map[PodRef]MetricsPoint)
	containers := make(map[PodRef]ContanersMetricsPoint)
	td := TimeseriesData{
		timeseries: []byte(`unknown_metric_total{node="x"}`),
		timestamp:  1609459200000,
		value:      1.0,
	}
	processed := processTimeseries(td, node, pods, containers)
	assert.False(t, processed)
	assert.Equal(t, 0, len(pods))
	assert.Equal(t, 0, len(containers))
}
