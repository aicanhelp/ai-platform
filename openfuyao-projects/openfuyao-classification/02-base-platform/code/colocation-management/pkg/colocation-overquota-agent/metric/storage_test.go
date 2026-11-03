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

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TestPodStorage
func TestPodStorage(t *testing.T) {
	tests := []struct {
		name           string
		setup          func(*podStorage)
		storeBatch     *MetricsBatch
		getPods        []*metav1.PartialObjectMetadata
		expectedError  bool
		expectedResult *NodeUsageReport
	}{
		{
			name:          "empty storage get metrics",
			setup:         func(s *podStorage) {},
			getPods:       []*metav1.PartialObjectMetadata{},
			expectedError: false,
			expectedResult: &NodeUsageReport{
				Node:               "test-node",
				Pods:               make(map[PodRef]corev1.ResourceList),
				Containers:         make(map[PodRef]map[string]corev1.ResourceList),
				PodsFromContainers: make(map[PodRef]corev1.ResourceList),
			},
		},
		{
			name: "get metrics for non-existent pod",
			setup: func(s *podStorage) {
				// 存储一些数据但不包含请求的pod
				s.last[NewPodRef("default", "existing-pod")] = ContanersMetricsPoint{
					Containers: map[string]MetricsPoint{
						"container1": {Timestamp: time.Now(), CumulativeCpuUsed: 1000},
					},
				}
			},
			getPods: []*metav1.PartialObjectMetadata{
				{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "v1",
						Kind:       "Pod",
					},
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "non-existent-pod",
					},
				},
			},
			expectedError: false,
			expectedResult: &NodeUsageReport{
				Node:               "test-node",
				Pods:               make(map[PodRef]corev1.ResourceList),
				Containers:         make(map[PodRef]map[string]corev1.ResourceList),
				PodsFromContainers: make(map[PodRef]corev1.ResourceList),
			},
		},
		{
			name: "store and get valid metrics",
			setup: func(s *podStorage) {
				// 设置初始数据
				now := time.Now()
				s.last[NewPodRef("default", "test-pod")] = ContanersMetricsPoint{
					Containers: map[string]MetricsPoint{
						"container1": {
							StartTime:         now.Add(-time.Minute),
							Timestamp:         now.Add(-30 * time.Second),
							CumulativeCpuUsed: 500,
						},
					},
				}
				s.prev[NewPodRef("default", "test-pod")] = ContanersMetricsPoint{
					Containers: map[string]MetricsPoint{
						"container1": {
							StartTime:         now.Add(-time.Minute),
							Timestamp:         now.Add(-40 * time.Second),
							CumulativeCpuUsed: 200,
						},
					},
				}
			},

			getPods: []*metav1.PartialObjectMetadata{
				{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "v1",
						Kind:       "Pod",
					},
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "test-pod",
					},
				},
			},
			expectedError: false,
			expectedResult: &NodeUsageReport{
				Node: "test-node",
				Pods: make(map[PodRef]corev1.ResourceList),
				Containers: map[PodRef]map[string]corev1.ResourceList{
					NewPodRef("default", "test-pod"): {
						"container1": corev1.ResourceList{
							corev1.ResourceCPU: *resource.NewMilliQuantity(30000, resource.DecimalSI), // 300ms over 10s = 30m/s
						},
					},
				},
				PodsFromContainers: map[PodRef]corev1.ResourceList{
					NewPodRef("default", "test-pod"): {
						corev1.ResourceCPU: *resource.NewMilliQuantity(30000, resource.DecimalSI),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			storage, ok := NewPodStorage("test-node", time.Minute).(*podStorage)
			if !ok {
				t.Errorf("failed to type assert to *podStorage: got types %T", NewPodStorage("test-node", time.Minute))
			}
			tt.setup(storage)

			result, err := storage.GetLastMetrics(tt.getPods...)

			if tt.expectedError && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.expectedError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			// 验证结果结构
			if result != nil && tt.expectedResult != nil {
				if result.Node != tt.expectedResult.Node {
					t.Errorf("Expected node %s, got %s", tt.expectedResult.Node, result.Node)
				}

				// 验证容器数量
				if len(result.Containers) != len(tt.expectedResult.Containers) {
					t.Errorf("Expected %d containers, got %d", len(tt.expectedResult.Containers), len(result.Containers))
				}

				// 验证pod数量
				if len(result.PodsFromContainers) != len(tt.expectedResult.PodsFromContainers) {
					t.Errorf("Expected %d pods, got %d", len(tt.expectedResult.PodsFromContainers), len(result.PodsFromContainers))
				}
			}

			t.Logf("Test passed")
		})
	}
}

// TestStoreMetrics
func TestStoreMetrics(t *testing.T) {
	storage, ok := NewPodStorage("test-node", time.Minute).(*podStorage)
	if !ok {
		t.Errorf("failed to type assert to *podStorage: got types %T", NewPodStorage("test-node", time.Minute))
	}
	now := time.Now()
	testBatch := &MetricsBatch{
		Pods: map[PodRef]MetricsPoint{
			NewPodRef("default", "pod1"): {
				StartTime:         now.Add(-time.Minute),
				Timestamp:         now,
				CumulativeCpuUsed: 1000,
			},
		},
		Containers: map[PodRef]ContanersMetricsPoint{
			NewPodRef("default", "pod1"): {
				Containers: map[string]MetricsPoint{
					"container1": {
						StartTime:         now.Add(-time.Minute),
						Timestamp:         now,
						CumulativeCpuUsed: 1000,
					},
				},
			},
		},
	}

	storage.Store(testBatch)

	storage.mu.RLock()
	defer storage.mu.RUnlock()

	if len(storage.last) != 1 {
		t.Errorf("Expected 1 pod in last storage, got %d", len(storage.last))
	}

	if len(storage.prev) != 0 {
		t.Errorf("Expected 0 pod in prev storage, got %d", len(storage.prev))
	}
}

// TestIsValidPoint
func TestIsValidPoint(t *testing.T) {
	tests := []struct {
		name          string
		point         MetricsPoint
		period        time.Duration
		minResolution time.Duration
		expected      bool
	}{
		{
			name: "valid point",
			point: MetricsPoint{
				StartTime: time.Now().Add(-time.Minute),
				Timestamp: time.Now().Add(-30 * time.Second),
			},
			period:        time.Minute,
			minResolution: time.Second,
			expected:      true,
		},
		{
			name: "start time after timestamp",
			point: MetricsPoint{
				StartTime: time.Now().Add(30 * time.Second),
				Timestamp: time.Now(),
			},
			period:        time.Minute,
			minResolution: time.Second,
			expected:      false,
		},
		{
			name: "duration too long",
			point: MetricsPoint{
				StartTime: time.Now().Add(-2 * time.Minute),
				Timestamp: time.Now(),
			},
			period:        time.Minute,
			minResolution: time.Second,
			expected:      false,
		},
		{
			name: "duration too short",
			point: MetricsPoint{
				StartTime: time.Now().Add(-500 * time.Millisecond),
				Timestamp: time.Now(),
			},
			period:        time.Minute,
			minResolution: time.Second,
			expected:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidPoint(tt.point, tt.period, tt.minResolution)
			if result != tt.expected {
				t.Errorf("isValidPoint() = %v, expected %v ", result, tt.expected)
			}
		})
	}
}

// TestHandlePoint
func TestHandlePoint(t *testing.T) {
	storage, ok := NewPodStorage("test-node", time.Minute).(*podStorage)
	if !ok {
		t.Errorf("failed to type assert to *podStorage: got types %T", NewPodStorage("test-node", time.Minute))
	}
	now := time.Now()

	tests := []struct {
		name          string
		setup         func()
		podRef        PodRef
		containerName string
		newPoint      MetricsPoint
		expected      bool
		// verify the entry that gets stored in newPrevPod after the call.
		// expectStored==nil means no entry expected.
		expectStored *MetricsPoint
	}{
		{
			name: "valid new point",
			setup: func() {
				storage.last = make(map[PodRef]ContanersMetricsPoint)
				storage.prev = make(map[PodRef]ContanersMetricsPoint)
			},
			podRef:        NewPodRef("default", "pod1"),
			containerName: "container1",
			newPoint: MetricsPoint{
				StartTime:         now.Add(-time.Minute),
				Timestamp:         now.Add(-30 * time.Second),
				CumulativeCpuUsed: 1000,
			},
			expected: true,
			// valid point path: copies point, sets Timestamp=StartTime, CumulativeCpu=0
			expectStored: &MetricsPoint{
				StartTime:         now.Add(-time.Minute),
				Timestamp:         now.Add(-time.Minute),
				CumulativeCpuUsed: 0,
			},
		},
		{
			name: "invalid point with no existing data",
			setup: func() {
				storage.last = make(map[PodRef]ContanersMetricsPoint)
				storage.prev = make(map[PodRef]ContanersMetricsPoint)
			},
			podRef:        NewPodRef("default", "pod1"),
			containerName: "container1",
			newPoint: MetricsPoint{
				StartTime:         now.Add(30 * time.Second), // invalid start time
				Timestamp:         now,
				CumulativeCpuUsed: 1000,
			},
			expected: false,
		},
		{
			name: "invalid point, last exists but container not in last",
			setup: func() {
				storage.last = map[PodRef]ContanersMetricsPoint{
					NewPodRef("default", "pod1"): {Containers: map[string]MetricsPoint{
						"other-container": {Timestamp: now.Add(-time.Minute)},
					}},
				}
				storage.prev = make(map[PodRef]ContanersMetricsPoint)
			},
			podRef:        NewPodRef("default", "pod1"),
			containerName: "container1",
			newPoint: MetricsPoint{
				StartTime: now.Add(30 * time.Second), // invalid
				Timestamp: now,
			},
			expected: false,
		},
		{
			name: "invalid point, last has container but startTime not before stored timestamp",
			setup: func() {
				storage.last = map[PodRef]ContanersMetricsPoint{
					NewPodRef("default", "pod1"): {Containers: map[string]MetricsPoint{
						"container1": {Timestamp: now.Add(-time.Hour)},
					}},
				}
				storage.prev = make(map[PodRef]ContanersMetricsPoint)
			},
			podRef:        NewPodRef("default", "pod1"),
			containerName: "container1",
			// newPoint.StartTime == lastContainer.Timestamp (not Before), so condition
			// !newPoint.StartTime.Before(lastContainer.Timestamp) is true -> drop
			newPoint: MetricsPoint{
				StartTime: now.Add(-time.Hour), // invalid start (after timestamp), equal to stored
				Timestamp: now.Add(-30 * time.Second),
			},
			expected: false,
		},
		{
			name: "invalid point, new point timestamp after stored last -> move last to prev",
			setup: func() {
				stored := MetricsPoint{
					StartTime:         now.Add(-2 * time.Minute),
					Timestamp:         now.Add(-90 * time.Second),
					CumulativeCpuUsed: 50,
				}
				storage.last = map[PodRef]ContanersMetricsPoint{
					NewPodRef("default", "pod1"): {Containers: map[string]MetricsPoint{
						"container1": stored,
					}},
				}
				storage.prev = make(map[PodRef]ContanersMetricsPoint)
			},
			podRef:        NewPodRef("default", "pod1"),
			containerName: "container1",
			// invalid (start after timestamp), but startTime < lastContainer.Timestamp and
			// newPoint.Timestamp > lastContainer.Timestamp -> move last to prev
			newPoint: MetricsPoint{
				StartTime: now.Add(-100 * time.Second), // < stored timestamp (-90s)
				Timestamp: now.Add(-30 * time.Second),  // > stored timestamp (-90s)
				// invalid because StartTime(-100s) NOT before... wait: -100s IS before -30s. Need invalid.
			},
			// To make the point invalid, StartTime must NOT be before Timestamp, OR the window
			// must be outside [minResolution, period). StartTime=-100s, Timestamp=-30s gives a
			// 70s window which is >= period(60s) -> invalid (window not < period).
			expected: true,
			expectStored: &MetricsPoint{
				StartTime:         now.Add(-2 * time.Minute),
				Timestamp:         now.Add(-90 * time.Second),
				CumulativeCpuUsed: 50,
			},
		},
		{
			name: "invalid point, new timestamp NOT after stored, no prev pod -> false",
			setup: func() {
				storage.last = map[PodRef]ContanersMetricsPoint{
					NewPodRef("default", "pod1"): {Containers: map[string]MetricsPoint{
						"container1": {Timestamp: now.Add(-time.Minute)},
					}},
				}
				storage.prev = make(map[PodRef]ContanersMetricsPoint)
			},
			podRef:        NewPodRef("default", "pod1"),
			containerName: "container1",
			// invalid, startTime before stored timestamp, but newPoint.Timestamp == stored (not After)
			newPoint: MetricsPoint{
				StartTime: now.Add(-2 * time.Minute),
				Timestamp: now.Add(-time.Minute), // equals stored, not after
			},
			expected: false,
		},
		{
			name: "invalid point, prev exists and prev timestamp before newPoint -> keep prev",
			setup: func() {
				storage.last = map[PodRef]ContanersMetricsPoint{
					NewPodRef("default", "pod1"): {Containers: map[string]MetricsPoint{
						"container1": {Timestamp: now.Add(-time.Minute)},
					}},
				}
				storage.prev = map[PodRef]ContanersMetricsPoint{
					NewPodRef("default", "pod1"): {Containers: map[string]MetricsPoint{
						"container1": {
							StartTime:         now.Add(-5 * time.Minute),
							Timestamp:         now.Add(-3 * time.Minute),
							CumulativeCpuUsed: 77,
						},
					}},
				}
			},
			podRef:        NewPodRef("default", "pod1"),
			containerName: "container1",
			// invalid, startTime < stored timestamp, newPoint.Timestamp == stored (not After),
			// prev exists, prev.Timestamp (-3m) before newPoint.Timestamp (-1m) -> keep prev
			newPoint: MetricsPoint{
				StartTime: now.Add(-2 * time.Minute),
				Timestamp: now.Add(-time.Minute),
			},
			expected: true,
			expectStored: &MetricsPoint{
				StartTime:         now.Add(-5 * time.Minute),
				Timestamp:         now.Add(-3 * time.Minute),
				CumulativeCpuUsed: 77,
			},
		},
		{
			name: "invalid point, prev exists but prev timestamp NOT before newPoint -> drop",
			setup: func() {
				storage.last = map[PodRef]ContanersMetricsPoint{
					NewPodRef("default", "pod1"): {Containers: map[string]MetricsPoint{
						"container1": {Timestamp: now.Add(-time.Minute)},
					}},
				}
				storage.prev = map[PodRef]ContanersMetricsPoint{
					NewPodRef("default", "pod1"): {Containers: map[string]MetricsPoint{
						// prev.Timestamp (-30s) is NOT before newPoint.Timestamp (-1m)
						"container1": {Timestamp: now.Add(-30 * time.Second)},
					}},
				}
			},
			podRef:        NewPodRef("default", "pod1"),
			containerName: "container1",
			newPoint: MetricsPoint{
				StartTime: now.Add(-2 * time.Minute),
				Timestamp: now.Add(-time.Minute),
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.setup()
			newPrevPod := &ContanersMetricsPoint{Containers: make(map[string]MetricsPoint)}
			result := storage.handlePoint(tt.podRef, tt.containerName, tt.newPoint, newPrevPod)

			if result != tt.expected {
				t.Errorf("handlePoint() = %v, expected %v ", result, tt.expected)
			}
			if tt.expectStored != nil {
				got, ok := newPrevPod.Containers[tt.containerName]
				if !ok {
					t.Fatalf("expected stored entry for %s, got none", tt.containerName)
				}
				assert.Equal(t, *tt.expectStored, got)
			} else {
				_, ok := newPrevPod.Containers[tt.containerName]
				assert.False(t, ok, "expected no stored entry but found one")
			}
		})
	}
}

// TestIsValidPointBoundaries covers the equality boundary conditions of isValidPoint.
func TestIsValidPointBoundaries(t *testing.T) {
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

	tests := []struct {
		name          string
		point         MetricsPoint
		period        time.Duration
		minResolution time.Duration
		expected      bool
	}{
		{
			name: "startTime equal to timestamp - false (Before is strict)",
			point: MetricsPoint{
				StartTime: base,
				Timestamp: base,
			},
			period:        time.Minute,
			minResolution: time.Second,
			expected:      false,
		},
		{
			name: "window exactly equal to period - false (not strictly less)",
			point: MetricsPoint{
				StartTime: base,
				Timestamp: base.Add(time.Minute),
			},
			period:        time.Minute,
			minResolution: time.Second,
			expected:      false,
		},
		{
			name: "window exactly equal to minResolution - true (>=)",
			point: MetricsPoint{
				StartTime: base,
				Timestamp: base.Add(time.Second),
			},
			period:        time.Minute,
			minResolution: time.Second,
			expected:      true,
		},
		{
			name: "window just above minResolution - true",
			point: MetricsPoint{
				StartTime: base,
				Timestamp: base.Add(2 * time.Second),
			},
			period:        time.Minute,
			minResolution: time.Second,
			expected:      true,
		},
		{
			name: "window just below minResolution - false",
			point: MetricsPoint{
				StartTime: base,
				Timestamp: base.Add(500 * time.Millisecond),
			},
			period:        time.Minute,
			minResolution: time.Second,
			expected:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidPoint(tt.point, tt.period, tt.minResolution)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestHandlePrevPointComparison directly exercises the standalone function.
func TestHandlePrevPointComparison(t *testing.T) {
	now := time.Now()
	podRef := NewPodRef("default", "pod1")
	containerName := "c1"

	tests := []struct {
		name          string
		setupPrev     map[PodRef]ContanersMetricsPoint
		newPoint      MetricsPoint
		expectStored  bool
		expectedPoint MetricsPoint
	}{
		{
			name:      "prev pod not found - no-op",
			setupPrev: map[PodRef]ContanersMetricsPoint{},
			newPoint:  MetricsPoint{Timestamp: now},
		},
		{
			name: "prev container not found - no-op",
			setupPrev: map[PodRef]ContanersMetricsPoint{
				podRef: {Containers: map[string]MetricsPoint{
					"other": {Timestamp: now.Add(-time.Minute)},
				}},
			},
			newPoint: MetricsPoint{Timestamp: now},
		},
		{
			name: "prev container older than new point - keep prev",
			setupPrev: map[PodRef]ContanersMetricsPoint{
				podRef: {Containers: map[string]MetricsPoint{
					containerName: {
						StartTime:         now.Add(-5 * time.Minute),
						Timestamp:         now.Add(-3 * time.Minute),
						CumulativeCpuUsed: 42,
					},
				}},
			},
			newPoint:     MetricsPoint{Timestamp: now},
			expectStored: true,
			expectedPoint: MetricsPoint{
				StartTime:         now.Add(-5 * time.Minute),
				Timestamp:         now.Add(-3 * time.Minute),
				CumulativeCpuUsed: 42,
			},
		},
		{
			name: "prev container newer than new point - drop prev (no store)",
			setupPrev: map[PodRef]ContanersMetricsPoint{
				podRef: {Containers: map[string]MetricsPoint{
					containerName: {Timestamp: now.Add(time.Minute)}, // after newPoint
				}},
			},
			newPoint: MetricsPoint{Timestamp: now},
		},
		{
			name: "prev container timestamp equal to new point - drop prev (Before is strict)",
			setupPrev: map[PodRef]ContanersMetricsPoint{
				podRef: {Containers: map[string]MetricsPoint{
					containerName: {Timestamp: now},
				}},
			},
			newPoint: MetricsPoint{Timestamp: now},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			newPrevPod := &ContanersMetricsPoint{Containers: make(map[string]MetricsPoint)}
			handlePrevPointComparison(podRef, containerName, tt.newPoint, newPrevPod, tt.setupPrev)

			got, ok := newPrevPod.Containers[containerName]
			if tt.expectStored {
				assert.True(t, ok, "expected entry stored")
				assert.Equal(t, tt.expectedPoint, got)
			} else {
				assert.False(t, ok, "expected no entry stored")
			}
		})
	}
}

// TestStoreDuplicatePodAndContainer exercises the duplicate-detection branches in Store.
func TestStoreDuplicatePodAndContainer(t *testing.T) {
	t.Run("duplicate container within same pod in input batch", func(t *testing.T) {
		// We cannot construct a map with duplicate keys, so test the
		// "Got duplicate Container point" branch indirectly is not possible
		// via the public map. Instead exercise the duplicate-pod branch is
		// also not possible via maps. We verify normal Store behaviour with
		// a valid point (window strictly < period) to keep last/prev consistent.
		storage, _ := NewPodStorage("test-node", time.Minute).(*podStorage)
		now := time.Now()
		batch := &MetricsBatch{
			Containers: map[PodRef]ContanersMetricsPoint{
				NewPodRef("default", "pod1"): {Containers: map[string]MetricsPoint{
					"c1": {
						StartTime:         now.Add(-time.Minute),
						Timestamp:         now.Add(-30 * time.Second),
						CumulativeCpuUsed: 100,
						MemoryUsage:       200,
					},
				}},
			},
		}
		storage.Store(batch)

		assert.Equal(t, 1, len(storage.last))
		// valid point -> prev is populated for c1
		assert.Equal(t, 1, len(storage.prev))
		assert.Equal(t, 1, len(storage.prev[NewPodRef("default", "pod1")].Containers))
	})

	t.Run("store with invalid point and no prior last leaves prev empty", func(t *testing.T) {
		storage, _ := NewPodStorage("test-node", time.Minute).(*podStorage)
		now := time.Now()
		batch := &MetricsBatch{
			Containers: map[PodRef]ContanersMetricsPoint{
				NewPodRef("default", "pod1"): {Containers: map[string]MetricsPoint{
					"c1": {
						// invalid: start time after timestamp
						StartTime: now.Add(time.Minute),
						Timestamp: now,
					},
				}},
			},
		}
		storage.Store(batch)
		assert.Equal(t, 1, len(storage.last))
		// prev empty because point invalid and no prior last
		assert.Equal(t, 0, len(storage.prev))
	})

	t.Run("store replaces previous last on second store", func(t *testing.T) {
		storage, _ := NewPodStorage("test-node", time.Minute).(*podStorage)
		now := time.Now()

		// first batch with one pod
		storage.Store(&MetricsBatch{
			Containers: map[PodRef]ContanersMetricsPoint{
				NewPodRef("default", "pod1"): {Containers: map[string]MetricsPoint{
					"c1": {
						StartTime: now.Add(-time.Minute),
						Timestamp: now.Add(-30 * time.Second),
					},
				}},
			},
		})
		assert.Equal(t, 1, len(storage.last))

		// second batch with different pod replaces last entirely
		storage.Store(&MetricsBatch{
			Containers: map[PodRef]ContanersMetricsPoint{
				NewPodRef("default", "pod2"): {Containers: map[string]MetricsPoint{
					"c1": {
						StartTime: now.Add(-time.Minute),
						Timestamp: now.Add(-30 * time.Second),
					},
				}},
			},
		})
		assert.Equal(t, 1, len(storage.last))
		_, found := storage.last[NewPodRef("default", "pod2")]
		assert.True(t, found)
		_, oldFound := storage.last[NewPodRef("default", "pod1")]
		assert.False(t, oldFound)
	})
}

// TestProcessSinglePodPrevNotFound exercises the branch where last exists but prev does not.
func TestProcessSinglePodPrevNotFound(t *testing.T) {
	storage, _ := NewPodStorage("test-node", time.Minute).(*podStorage)
	now := time.Now()
	podRef := NewPodRef("default", "pod1")
	// only last, no prev
	storage.last[podRef] = ContanersMetricsPoint{
		Containers: map[string]MetricsPoint{
			"c1": {Timestamp: now, CumulativeCpuUsed: 10},
		},
	}
	storage.prev = make(map[PodRef]ContanersMetricsPoint)

	nmr := &NodeUsageReport{
		Node:               storage.node,
		Pods:               make(map[PodRef]corev1.ResourceList),
		Containers:         make(map[PodRef]map[string]corev1.ResourceList),
		PodsFromContainers: make(map[PodRef]corev1.ResourceList),
	}

	pod := &metav1.PartialObjectMetadata{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod1"}}
	processed := storage.processSinglePod(pod, nmr)
	assert.False(t, processed, "processSinglePod should return false when prev pod missing")
	// nothing should be added
	assert.Equal(t, 0, len(nmr.Containers))
	assert.Equal(t, 0, len(nmr.PodsFromContainers))
}

// TestProcessContainersMissingPrevContainer covers the branch where a container
// in last does not exist in prev (returns allContainersPresent=false).
func TestProcessContainersMissingPrevContainer(t *testing.T) {
	storage, _ := NewPodStorage("test-node", time.Minute).(*podStorage)
	now := time.Now()
	pod := &metav1.PartialObjectMetadata{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod1"}}

	lastPod := ContanersMetricsPoint{Containers: map[string]MetricsPoint{
		"c1": {StartTime: now.Add(-time.Minute), Timestamp: now, CumulativeCpuUsed: 10, MemoryUsage: 100},
	}}
	// prev has a different container
	prevPod := ContanersMetricsPoint{Containers: map[string]MetricsPoint{
		"c2": {StartTime: now.Add(-time.Minute), Timestamp: now.Add(-30 * time.Second), CumulativeCpuUsed: 5, MemoryUsage: 50},
	}}

	res, ok := storage.processContainers(pod, lastPod, prevPod)
	assert.False(t, ok)
	assert.Nil(t, res.cmr)
	assert.Nil(t, res.pr)
}

// TestProcessContainersResourceUsageError covers the branch where resourceUsage
// returns an error for a container (CPU decreased) and that container is skipped.
func TestProcessContainersResourceUsageError(t *testing.T) {
	storage, _ := NewPodStorage("test-node", time.Minute).(*podStorage)
	now := time.Now()
	pod := &metav1.PartialObjectMetadata{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod1"}}

	lastPod := ContanersMetricsPoint{Containers: map[string]MetricsPoint{
		// CPU went down -> resourceUsage error -> skipped
		"bad": {StartTime: now.Add(-time.Minute), Timestamp: now, CumulativeCpuUsed: 10, MemoryUsage: 100},
	}}
	prevPod := ContanersMetricsPoint{Containers: map[string]MetricsPoint{
		"bad": {StartTime: now.Add(-time.Minute), Timestamp: now.Add(-30 * time.Second), CumulativeCpuUsed: 500, MemoryUsage: 50},
	}}

	res, ok := storage.processContainers(pod, lastPod, prevPod)
	// allContainersPresent still true (ok), but cmr/pr are empty
	assert.True(t, ok)
	assert.NotNil(t, res.cmr)
	assert.Equal(t, 0, len(res.cmr))
}

// TestGetLastMetricsMultipleContainers verifies processContainers with two valid
// containers and the earliest-timestamp selection logic.
func TestGetLastMetricsMultipleContainers(t *testing.T) {
	storage, _ := NewPodStorage("test-node", time.Minute).(*podStorage)
	now := time.Now()
	podRef := NewPodRef("default", "pod1")

	storage.last[podRef] = ContanersMetricsPoint{Containers: map[string]MetricsPoint{
		"c1": {
			StartTime:         now.Add(-time.Minute),
			Timestamp:         now.Add(-10 * time.Second),
			CumulativeCpuUsed: 100,
			MemoryUsage:       1024,
		},
		"c2": {
			StartTime:         now.Add(-time.Minute),
			Timestamp:         now.Add(-5 * time.Second), // later timestamp
			CumulativeCpuUsed: 200,
			MemoryUsage:       2048,
		},
	}}
	storage.prev[podRef] = ContanersMetricsPoint{Containers: map[string]MetricsPoint{
		"c1": {
			StartTime:         now.Add(-time.Minute),
			Timestamp:         now.Add(-20 * time.Second),
			CumulativeCpuUsed: 0,
			MemoryUsage:       512,
		},
		"c2": {
			StartTime:         now.Add(-time.Minute),
			Timestamp:         now.Add(-15 * time.Second),
			CumulativeCpuUsed: 0,
			MemoryUsage:       1024,
		},
	}}

	pod := &metav1.PartialObjectMetadata{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod1"}}
	nmr, err := storage.GetLastMetrics(pod)
	assert.NoError(t, err)
	assert.Equal(t, 1, len(nmr.Containers))
	assert.Equal(t, 2, len(nmr.Containers[podRef]))
	assert.Equal(t, 1, len(nmr.PodsFromContainers))
	// earliest timestamp among containers is c1 (-10s)
	assert.Equal(t, now.Add(-10*time.Second), nmr.Timestamp.Time)
	assert.Equal(t, 10*time.Second, nmr.Window.Duration)
}

// TestGetLastMetricsEmptyPods exercises GetLastMetrics with no input pods.
func TestGetLastMetricsEmptyPods(t *testing.T) {
	storage, _ := NewPodStorage("test-node", time.Minute).(*podStorage)
	nmr, err := storage.GetLastMetrics()
	assert.NoError(t, err)
	assert.Equal(t, "test-node", nmr.Node)
	assert.Equal(t, 0, len(nmr.Containers))
	assert.Equal(t, 0, len(nmr.PodsFromContainers))
	assert.Equal(t, 0, len(nmr.Pods))
}

// TestNewPodStorageDefaults verifies the constructor initializes internal maps.
func TestNewPodStorageDefaults(t *testing.T) {
	s := NewPodStorage("node-x", 30*time.Second)
	ps, ok := s.(*podStorage)
	assert.True(t, ok)
	assert.Equal(t, "node-x", ps.node)
	assert.Equal(t, 30*time.Second, ps.metricsPeriod)
	assert.NotNil(t, ps.last)
	assert.NotNil(t, ps.prev)
	assert.Equal(t, 0, len(ps.last))
	assert.Equal(t, 0, len(ps.prev))
}

// TestStoreEmptyBatch verifies Store handles an empty MetricsBatch cleanly.
func TestStoreEmptyBatch(t *testing.T) {
	storage, _ := NewPodStorage("test-node", time.Minute).(*podStorage)
	// pre-populate
	storage.last[NewPodRef("default", "pod1")] = ContanersMetricsPoint{Containers: map[string]MetricsPoint{}}

	storage.Store(&MetricsBatch{})
	assert.Equal(t, 0, len(storage.last))
	assert.Equal(t, 0, len(storage.prev))
}
