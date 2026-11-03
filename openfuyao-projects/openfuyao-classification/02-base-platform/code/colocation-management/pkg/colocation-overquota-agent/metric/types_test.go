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

package metric

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/json"
)

func TestAddResourceLists(t *testing.T) {
	empty := corev1.ResourceList{}
	res1 := corev1.ResourceList{
		"CPU": *resource.NewQuantity(100, resource.DecimalSI),
	}
	fmt.Printf("res1: %v\n", SumResourceList(empty, res1))

	res2 := corev1.ResourceList{
		"MEM": *resource.NewQuantity(200, resource.DecimalSI),
	}
	fmt.Printf("res2: %v\n", SumResourceList(res2, empty))

	fmt.Printf("sum of res1 and res2: %v\n", SumResourceList(res1, res2))
}

func TestMarshalNodeMetricsReport(t *testing.T) {
	inputs := []NodeUsageReport{
		{
			Node:      "192.168.0.10",
			Timestamp: metav1.Date(2024, 1, 2, 12, 13, 14, 0, time.Local),
			Window:    metav1.Duration{Duration: 10 * time.Second},
			Pods: map[PodRef]corev1.ResourceList{
				NewPodRef("ns1", "pod1"): corev1.ResourceList{
					corev1.ResourceCPU:    uint64Quantity(uint64(30), resource.DecimalSI, -9),
					corev1.ResourceMemory: uint64Quantity(50, resource.BinarySI, 0),
				},
			},
		},
		{
			Node:      "192.168.0.11",
			Timestamp: metav1.Date(2025, 11, 12, 22, 23, 24, 0, time.Local),
			Window:    metav1.Duration{Duration: 20 * time.Second},
		},
	}
	reqData, err := json.Marshal(inputs)
	assert.Nil(t, err)

	outputs := make([]NodeUsageReport, 0)
	err = json.Unmarshal(reqData, &outputs)
	assert.Nil(t, err)

	assert.Equal(t, "192.168.0.10", inputs[0].Node)
	assert.Equal(t, "192.168.0.11", inputs[1].Node)
	assert.Equal(t, metav1.Date(2024, 1, 2, 12, 13, 14, 0, time.Local), inputs[0].Timestamp)
	assert.Equal(t, metav1.Date(2025, 11, 12, 22, 23, 24, 0, time.Local), inputs[1].Timestamp)
	assert.Equal(t, metav1.Duration{Duration: 10 * time.Second}, inputs[0].Window)
	assert.Equal(t, metav1.Duration{Duration: 20 * time.Second}, inputs[1].Window)
}

func TestUint64Quantity(t *testing.T) {
	tests := []struct {
		name     string
		val      uint64
		format   resource.Format
		scale    resource.Scale
		expected int64
	}{
		{
			name:     "normal value within int64 range",
			val:      1000,
			format:   resource.DecimalSI,
			scale:    0,
			expected: 1000,
		},
		{
			name:     "zero value",
			val:      0,
			format:   resource.DecimalSI,
			scale:    3, // milli scale
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := uint64Quantity(tt.val, tt.format, tt.scale)

			if result.Value() != tt.expected {
				t.Errorf("Expected value %d, got %d", tt.expected, result.Value())
			}

			if result.Format != tt.format {
				t.Errorf("Expected format %v, got %v", tt.format, result.Format)
			}
		})
	}
}

func TestSumResourceList(t *testing.T) {
	tests := []struct {
		name     string
		res1     corev1.ResourceList
		res2     corev1.ResourceList
		expected corev1.ResourceList
	}{
		{
			name:     "both nil",
			res1:     nil,
			res2:     nil,
			expected: corev1.ResourceList{},
		},
		{
			name:     "res1 nil",
			res1:     nil,
			res2:     corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1")},
			expected: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1")},
		},
		{
			name:     "res2 nil",
			res1:     corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("1Gi")},
			res2:     nil,
			expected: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("1Gi")},
		},
		{
			name: "no overlapping resources",
			res1: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1"),
			},
			res2: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1"),
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
		},
		{
			name: "overlapping resources with addition",
			res1: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1"),
				corev1.ResourceMemory: resource.MustParse("512Mi"),
			},
			res2: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("2"),
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("3"),
				corev1.ResourceMemory: resource.MustParse("1536Mi"), // 512Mi + 1024Mi
			},
		},
		{
			name:     "empty resource lists",
			res1:     corev1.ResourceList{},
			res2:     corev1.ResourceList{},
			expected: corev1.ResourceList{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SumResourceList(tt.res1, tt.res2)

			// 检查结果长度
			if len(result) != len(tt.expected) {
				t.Errorf("Expected %d resources, got %d", len(tt.expected), len(result))
			}

			// 检查每个资源的值
			for resourceName, expectedQuantity := range tt.expected {
				actualQuantity, exists := result[resourceName]
				if !exists {
					t.Errorf("Expected resource %s not found in result", resourceName)
					continue
				}

				if actualQuantity.Cmp(expectedQuantity) != 0 {
					t.Errorf("For resource %s: expected %s, got %s", resourceName, expectedQuantity.String(), actualQuantity.String())
				}
			}

			// 检查没有多余的资源
			for resourceName := range result {
				if _, exists := tt.expected[resourceName]; !exists {
					t.Errorf("Unexpected resource %s found in result", resourceName)
				}
			}
		})
	}
}

func TestPrintObjMeta(t *testing.T) {
	tests := []struct {
		name     string
		input    []*metav1.PartialObjectMetadata
		expected string
	}{
		{
			name:     "empty slice",
			input:    []*metav1.PartialObjectMetadata{},
			expected: "",
		},
		{
			name:     "nil slice",
			input:    nil,
			expected: "",
		},
		{
			name: "single item with namespace",
			input: []*metav1.PartialObjectMetadata{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "pod-1",
					},
				},
			},
			expected: "Namespace:default, Name:pod-1\n",
		},
		{
			name: "single item without namespace",
			input: []*metav1.PartialObjectMetadata{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "cluster-scoped-resource",
					},
				},
			},
			expected: "Namespace:, Name:cluster-scoped-resource\n",
		},
		{
			name: "multiple items",
			input: []*metav1.PartialObjectMetadata{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "pod-1",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "pod-2",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "cluster-role",
					},
				},
			},
			expected: "Namespace:default, Name:pod-1\nNamespace:kube-system, Name:pod-2\nNamespace:, Name:cluster-role\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := PrintObjMeta(tt.input)
			if result != tt.expected {
				t.Errorf("Expected:\n%q\n\nGot:\n%q", tt.expected, result)
			}
		})
	}
}

func TestPrintNamespaceNamesForContainers(t *testing.T) {
	tests := []struct {
		name     string
		input    map[PodRef]ContanersMetricsPoint
		expected string
	}{
		{
			name:     "empty map",
			input:    map[PodRef]ContanersMetricsPoint{},
			expected: "",
		},
		{
			name:     "nil map",
			input:    nil,
			expected: "",
		},
		{
			name: "single item",
			input: map[PodRef]ContanersMetricsPoint{
				NewPodRef("default", "pod-1"): {},
			},
			expected: "Namespace/Name:default/pod-1\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := PrintNamespaceNamesForContainers(tt.input)
			if result != tt.expected {
				t.Errorf("Expected:\n%q\nGot:\n%q", tt.expected, result)
			}
		})
	}
}

func TestPrintNamespaceNamesForPods(t *testing.T) {
	tests := []struct {
		name     string
		input    map[PodRef]MetricsPoint
		expected string
	}{
		{
			name:     "empty map",
			input:    map[PodRef]MetricsPoint{},
			expected: "",
		},
		{
			name:     "nil map",
			input:    nil,
			expected: "",
		},
		{
			name: "single item",
			input: map[PodRef]MetricsPoint{
				NewPodRef("default", "pod-1"): {},
			},
			expected: "Namespace/Name:default/pod-1\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := PrintNamespaceNamesForPods(tt.input)
			if result != tt.expected {
				t.Errorf("Expected:\n%q\nGot:\n%q", tt.expected, result)
			}
		})
	}
}

func TestPrintNamespaceNamesForResList(t *testing.T) {
	tests := []struct {
		name     string
		input    map[PodRef]map[string]corev1.ResourceList
		expected string
	}{
		{
			name:     "empty map",
			input:    map[PodRef]map[string]corev1.ResourceList{},
			expected: "",
		},
		{
			name:     "nil map",
			input:    nil,
			expected: "",
		},
		{
			name: "single item",
			input: map[PodRef]map[string]corev1.ResourceList{
				NewPodRef("default", "pod-1"): {
					"container-1": {
						corev1.ResourceCPU:    resource.MustParse("100m"),
						corev1.ResourceMemory: resource.MustParse("128Mi"),
					},
				},
			},
			expected: "Namespace/Name:default/pod-1\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := PrintNamespaceNamesForResList(tt.input)
			if result != tt.expected {
				t.Errorf("Expected:\n%q\nGot:\n%q", tt.expected, result)
			}
		})
	}
}

func TestPrintNamespaceNamesForResList2(t *testing.T) {
	tests := []struct {
		name     string
		input    map[PodRef]corev1.ResourceList
		expected string
	}{
		{
			name:     "empty map",
			input:    map[PodRef]corev1.ResourceList{},
			expected: "",
		},
		{
			name:     "nil map",
			input:    nil,
			expected: "",
		},
		{
			name: "single item",
			input: map[PodRef]corev1.ResourceList{
				NewPodRef("default", "pod-1"): {
					corev1.ResourceCPU:    resource.MustParse("100m"),
					corev1.ResourceMemory: resource.MustParse("128Mi"),
				},
			},
			expected: "Namespace/Name:default/pod-1\n",
		},
		{
			name: "multiple items",
			input: map[PodRef]corev1.ResourceList{
				NewPodRef("ns1", "pod-a"): {corev1.ResourceCPU: resource.MustParse("1")},
				NewPodRef("ns2", "pod-b"): {corev1.ResourceMemory: resource.MustParse("1Gi")},
			},
			expected: "Namespace/Name:ns1/pod-a\nNamespace/Name:ns2/pod-b\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := PrintNamespaceNamesForResList2(tt.input)
			// for multi-item cases order is not deterministic; only verify exact for 0/1-item cases
			if len(tt.input) <= 1 {
				if result != tt.expected {
					t.Errorf("Expected:\n%q\nGot:\n%q", tt.expected, result)
				}
			} else {
				// verify both lines are present (order-independent)
				if strings.Count(result, "\n") != 2 {
					t.Errorf("Expected 2 lines, got %q", result)
				}
				if !strings.Contains(result, "Namespace/Name:ns1/pod-a\n") {
					t.Errorf("missing ns1/pod-a in %q", result)
				}
				if !strings.Contains(result, "Namespace/Name:ns2/pod-b\n") {
					t.Errorf("missing ns2/pod-b in %q", result)
				}
			}
		})
	}
}

func TestNewPodRef(t *testing.T) {
	tests := []struct {
		name      string
		namespace string
		podName   string
		expected  PodRef
	}{
		{name: "normal", namespace: "default", podName: "pod-1", expected: "default/pod-1"},
		{name: "kube-system", namespace: "kube-system", podName: "coredns-abc", expected: "kube-system/coredns-abc"},
		{name: "empty values", namespace: "", podName: "", expected: "/"},
		{name: "empty namespace", namespace: "", podName: "pod-1", expected: "/pod-1"},
		{name: "empty name", namespace: "default", podName: "", expected: "default/"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewPodRef(tt.namespace, tt.podName)
			assert.Equal(t, tt.expected, got)
		})
	}
}

func TestNamespaceAndName(t *testing.T) {
	tests := []struct {
		name        string
		ref         PodRef
		wantNS      string
		wantName    string
		wantErr     bool
		errContains string
	}{
		{name: "valid ref", ref: NewPodRef("default", "pod-1"), wantNS: "default", wantName: "pod-1", wantErr: false},
		{name: "kube-system ref", ref: NewPodRef("kube-system", "coredns-abc"), wantNS: "kube-system", wantName: "coredns-abc", wantErr: false},
		{name: "no slash", ref: PodRef("malformed"), wantErr: true, errContains: "invlid podRef"},
		{name: "empty ref", ref: PodRef(""), wantErr: true, errContains: "invlid podRef"},
		{name: "two slashes", ref: PodRef("a/b/c"), wantErr: true, errContains: "invlid podRef"},
		{name: "empty namespace", ref: PodRef("/pod-1"), wantErr: true, errContains: "invlid podRef"},
		{name: "empty name", ref: PodRef("default/"), wantErr: true, errContains: "invlid podRef"},
		{name: "trailing slash only", ref: PodRef("/"), wantErr: true, errContains: "invlid podRef"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ns, name, err := tt.ref.NamespaceAndName()
			if tt.wantErr {
				assert.Error(t, err)
				assert.Equal(t, "", ns)
				assert.Equal(t, "", name)
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.wantNS, ns)
				assert.Equal(t, tt.wantName, name)
			}
		})
	}
}

func TestResourceUsage(t *testing.T) {
	baseStart := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

	tests := []struct {
		name        string
		last        MetricsPoint
		prev        MetricsPoint
		wantErr     bool
		errContains string
		// expected checks (only when no error)
		wantCPU     string
		wantMem     string
		wantWindow  time.Duration
		wantTS      time.Time
	}{
		{
			name: "valid usage delta over 10s window",
			last: MetricsPoint{
				StartTime:         baseStart,
				Timestamp:         baseStart.Add(20 * time.Second),
				CumulativeCpuUsed: 1000,
				MemoryUsage:       2048,
			},
			prev: MetricsPoint{
				StartTime:         baseStart,
				Timestamp:         baseStart.Add(10 * time.Second),
				CumulativeCpuUsed: 400,
				MemoryUsage:       1024,
			},
			wantErr:    false,
			wantCPU:    "60n", // (1000-400)/10s = 60 nanoCores
			wantMem:    "2Ki",
			wantWindow: 10 * time.Second,
			wantTS:     baseStart.Add(20 * time.Second),
		},
		{
			name: "equal start times - no error",
			last: MetricsPoint{
				StartTime:         baseStart,
				Timestamp:         baseStart.Add(20 * time.Second),
				CumulativeCpuUsed: 1000,
				MemoryUsage:       2048,
			},
			prev: MetricsPoint{
				StartTime:         baseStart,
				Timestamp:         baseStart.Add(10 * time.Second),
				CumulativeCpuUsed: 400,
				MemoryUsage:       1024,
			},
			wantErr: false,
		},
		{
			name: "last start before prev start - error",
			last: MetricsPoint{
				StartTime:         baseStart,
				Timestamp:         baseStart.Add(20 * time.Second),
				CumulativeCpuUsed: 1000,
				MemoryUsage:       2048,
			},
			prev: MetricsPoint{
				StartTime:         baseStart.Add(time.Second),
				Timestamp:         baseStart.Add(10 * time.Second),
				CumulativeCpuUsed: 400,
				MemoryUsage:       1024,
			},
			wantErr:     true,
			errContains: "unexpected decrease in startTime",
		},
		{
			name: "cumulative cpu decreased - error",
			last: MetricsPoint{
				StartTime:         baseStart,
				Timestamp:         baseStart.Add(20 * time.Second),
				CumulativeCpuUsed: 100,
				MemoryUsage:       2048,
			},
			prev: MetricsPoint{
				StartTime:         baseStart,
				Timestamp:         baseStart.Add(10 * time.Second),
				CumulativeCpuUsed: 400,
				MemoryUsage:       1024,
			},
			wantErr:     true,
			errContains: "unexpected decrease in cumulative CPU usage value",
		},
		{
			name: "zero delta cpu and memory usage present",
			last: MetricsPoint{
				StartTime:         baseStart,
				Timestamp:         baseStart.Add(10 * time.Second),
				CumulativeCpuUsed: 500,
				MemoryUsage:       4096,
			},
			prev: MetricsPoint{
				StartTime:         baseStart,
				Timestamp:         baseStart.Add(5 * time.Second),
				CumulativeCpuUsed: 500,
				MemoryUsage:       0,
			},
			wantErr:    false,
			wantCPU:    "0",
			wantMem:    "4Ki",
			wantWindow: 5 * time.Second,
			wantTS:     baseStart.Add(10 * time.Second),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage, ti, err := resourceUsage(tt.last, tt.prev)
			if tt.wantErr {
				assert.Error(t, err)
				// error path returns an empty (allocated) ResourceList, not nil
				assert.Equal(t, corev1.ResourceList{}, usage)
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}
				return
			}
			assert.NoError(t, err)
			assert.NotNil(t, usage)
			if tt.wantCPU != "" {
				cpu := usage[corev1.ResourceCPU]
				assert.Equal(t, tt.wantCPU, (&cpu).String())
			}
			if tt.wantMem != "" {
				mem := usage[corev1.ResourceMemory]
				assert.Equal(t, tt.wantMem, (&mem).String())
			}
			if tt.wantWindow != 0 {
				assert.Equal(t, tt.wantWindow, ti.Window)
			}
			if !tt.wantTS.IsZero() {
				assert.Equal(t, tt.wantTS, ti.Timestamp)
			}
		})
	}
}

func TestUint64QuantityExtended(t *testing.T) {
	tests := []struct {
		name       string
		val        uint64
		format     resource.Format
		scale      resource.Scale
		wantFormat resource.Format
	}{
		{name: "small decimal", val: 1000, format: resource.DecimalSI, scale: 0, wantFormat: resource.DecimalSI},
		{name: "small binary", val: 2048, format: resource.BinarySI, scale: 0, wantFormat: resource.BinarySI},
		{name: "nano scale", val: 5, format: resource.DecimalSI, scale: -9, wantFormat: resource.DecimalSI},
		{name: "zero value", val: 0, format: resource.DecimalSI, scale: 3, wantFormat: resource.DecimalSI},
		{name: "max int64 boundary", val: math.MaxInt64, format: resource.DecimalSI, scale: 0, wantFormat: resource.DecimalSI},
		{name: "exceeds max int64 - precision loss branch", val: math.MaxInt64 + 1, format: resource.BinarySI, scale: 0, wantFormat: resource.BinarySI},
		{name: "large exceeds max int64 milli scale", val: math.MaxInt64 + uint64(1e9), format: resource.DecimalSI, scale: -3, wantFormat: resource.DecimalSI},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := uint64Quantity(tt.val, tt.format, tt.scale)
			assert.Equal(t, tt.wantFormat, q.Format, "format mismatch")
			// value should be positive for all positive inputs (and zero for zero)
			if tt.val == 0 {
				assert.Equal(t, int64(0), q.Value())
			} else {
				assert.True(t, q.Value() > 0, "expected positive value for val=%d, got %d", tt.val, q.Value())
			}
		})
	}
}

// TestNodeUsageResponseDecoding covers the generic NodeUsageResponse[T] type's
// JSON marshal/unmarshal round-trip (the generic decoding path used by the
// report package).
func TestNodeUsageResponseDecoding(t *testing.T) {
	// valid payload with data
	validJSON := `{"code":200,"msg":"ok","data":"2024-01-02T03:04:05Z"}`
	resp := &NodeUsageResponse[metav1.Time]{}
	err := json.Unmarshal([]byte(validJSON), resp)
	assert.Nil(t, err)
	assert.Equal(t, 200, resp.Code)
	assert.Equal(t, "ok", resp.Msg)
	assert.False(t, resp.Data.IsZero())

	// round-trip marshal
	out, err := json.Marshal(resp)
	assert.Nil(t, err)
	assert.Contains(t, string(out), `"code":200`)

	// malformed JSON -> error
	bad := &NodeUsageResponse[metav1.Time]{}
	err = json.Unmarshal([]byte(`{not-json`), bad)
	assert.Error(t, err)

	// empty data
	emptyData := `{"code":404,"msg":"not found"}`
	resp2 := &NodeUsageResponse[metav1.Time]{}
	err = json.Unmarshal([]byte(emptyData), resp2)
	assert.Nil(t, err)
	assert.Equal(t, 404, resp2.Code)
	assert.True(t, resp2.Data.IsZero())
}

func TestSumResourceListOverlappingOnly(t *testing.T) {
	// Force the second-loop branch where res2 keys already exist in res1.
	res1 := corev1.ResourceList{
		corev1.ResourceCPU:    resource.MustParse("1"),
		corev1.ResourceMemory: resource.MustParse("512Mi"),
	}
	res2 := corev1.ResourceList{
		corev1.ResourceCPU:    resource.MustParse("2"),
		corev1.ResourceMemory: resource.MustParse("512Mi"),
	}
	result := SumResourceList(res1, res2)
	cpu := result[corev1.ResourceCPU]
	mem := result[corev1.ResourceMemory]
	assert.Equal(t, "3", (&cpu).String())
	assert.Equal(t, "1Gi", (&mem).String())
}

func TestPrintNamespaceNamesForContainersMulti(t *testing.T) {
	// Exercise the non-empty branch with several keys (order-independent check).
	m := map[PodRef]ContanersMetricsPoint{
		NewPodRef("ns1", "a"): {},
		NewPodRef("ns2", "b"): {},
	}
	out := PrintNamespaceNamesForContainers(m)
	assert.Contains(t, out, "Namespace/Name:ns1/a\n")
	assert.Contains(t, out, "Namespace/Name:ns2/b\n")
	assert.Equal(t, 2, strings.Count(out, "\n"))
}

func TestPrintNamespaceNamesForPodsMulti(t *testing.T) {
	m := map[PodRef]MetricsPoint{
		NewPodRef("ns1", "a"): {},
		NewPodRef("ns2", "b"): {},
	}
	out := PrintNamespaceNamesForPods(m)
	assert.Contains(t, out, "Namespace/Name:ns1/a\n")
	assert.Contains(t, out, "Namespace/Name:ns2/b\n")
	assert.Equal(t, 2, strings.Count(out, "\n"))
}

func TestPrintNamespaceNamesForResListMulti(t *testing.T) {
	m := map[PodRef]map[string]corev1.ResourceList{
		NewPodRef("ns1", "a"): {"c1": {}},
		NewPodRef("ns2", "b"): {"c1": {}},
	}
	out := PrintNamespaceNamesForResList(m)
	assert.Contains(t, out, "Namespace/Name:ns1/a\n")
	assert.Contains(t, out, "Namespace/Name:ns2/b\n")
	assert.Equal(t, 2, strings.Count(out, "\n"))
}
