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

package aggregate

import (
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// TestResourcesAdd 测试Resources的Add方法
func TestResourcesAdd(t *testing.T) {
	tests := []struct {
		name     string
		a        Resources
		b        Resources
		expected Resources
	}{
		{
			name:     "empty resources",
			a:        Resources{},
			b:        Resources{},
			expected: Resources{},
		},
		{
			name: "add to empty",
			a:    Resources{},
			b: Resources{
				ResourceCPU:    1000,
				ResourceMemory: 1024 * 1024,
			},
			expected: Resources{
				ResourceCPU:    1000,
				ResourceMemory: 1024 * 1024,
			},
		},
		{
			name: "add existing resources",
			a: Resources{
				ResourceCPU:    500,
				ResourceMemory: 512 * 1024,
			},
			b: Resources{
				ResourceCPU:    300,
				ResourceMemory: 256 * 1024,
			},
			expected: Resources{
				ResourceCPU:    800,
				ResourceMemory: 768 * 1024,
			},
		},
		{
			name: "add new resource type",
			a: Resources{
				ResourceCPU: 500,
			},
			b: Resources{
				ResourceMemory: 1024 * 1024,
			},
			expected: Resources{
				ResourceCPU:    500,
				ResourceMemory: 1024 * 1024,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.a.Add(tt.b)

			if len(tt.a) != len(tt.expected) {
				t.Errorf("Expected %d resources, got %d", len(tt.expected), len(tt.a))
			}

			for key, expectedAmount := range tt.expected {
				if actualAmount, exists := tt.a[key]; !exists || actualAmount != expectedAmount {
					t.Errorf("Resource %s: expected %d, got %d", key, expectedAmount, actualAmount)
				}
			}
		})
	}
}

// TestCPUAmountConversions 测试CPU数量转换
func TestCPUAmountConversions(t *testing.T) {
	tests := []struct {
		name     string
		cores    float64
		expected ResourceAmount
	}{
		{"zero cores", 0.0, 0},
		{"integer cores", 2.0, 2000},
		{"fractional cores", 1.5, 1500},
		{"small cores", 0.001, 1},
		{"negative cores", -1.0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CPUAmountFromCores(tt.cores)
			if result != tt.expected {
				t.Errorf("CPUAmountFromCores(%f) = %d, expected %d", tt.cores, result, tt.expected)
			}

			// 测试反向转换
			if tt.cores >= 0 {
				backToCores := CoresFromCPUAmount(result)
				expectedCores := tt.cores
				if tt.cores < 0 {
					expectedCores = 0
				}
				if backToCores != expectedCores {
					t.Errorf("CoresFromCPUAmount(%d) = %f, expected %f", result, backToCores, expectedCores)
				}
			}
		})
	}
}

// TestMemoryAmountConversions 测试内存数量转换
func TestMemoryAmountConversions(t *testing.T) {
	tests := []struct {
		name     string
		bytes    float64
		expected ResourceAmount
	}{
		{"zero bytes", 0.0, 0},
		{"kilobyte", 1024.0, 1024},
		{"megabyte", 1024 * 1024.0, 1024 * 1024},
		{"gigabyte", 1024 * 1024 * 1024.0, 1024 * 1024 * 1024},
		{"negative bytes", -1024.0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MemoryAmountFromBytes(tt.bytes)
			if result != tt.expected {
				t.Errorf("MemoryAmountFromBytes(%f) = %d, expected %d", tt.bytes, result, tt.expected)
			}

			// 测试反向转换
			if tt.bytes >= 0 {
				backToBytes := BytesFromMemoryAmount(result)
				expectedBytes := tt.bytes
				if tt.bytes < 0 {
					expectedBytes = 0
				}
				if backToBytes != expectedBytes {
					t.Errorf("BytesFromMemoryAmount(%d) = %f, expected %f", result, backToBytes, expectedBytes)
				}
			}
		})
	}
}

// TestQuantityConversions 测试Kubernetes Quantity转换
func TestQuantityConversions(t *testing.T) {
	tests := []struct {
		name   string
		cpu    ResourceAmount
		memory ResourceAmount
	}{
		{"zero values", 0, 0},
		{"standard values", 1500, 1024 * 1024 * 1024},
		{"large values", 10000, 16 * 1024 * 1024 * 1024},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// CPU转换测试
			cpuQuantity := QuantityFromCPUAmount(tt.cpu)
			expectedMilli := int64(tt.cpu)
			if cpuQuantity.MilliValue() != expectedMilli {
				t.Errorf("QuantityFromCPUAmount(%d).MilliValue() = %d, expected %d",
					tt.cpu, cpuQuantity.MilliValue(), expectedMilli)
			}

			// 内存转换测试
			memoryQuantity := QuantityFromMemoryAmount(tt.memory)
			expectedBytes := int64(tt.memory)
			if memoryQuantity.Value() != expectedBytes {
				t.Errorf("QuantityFromMemoryAmount(%d).Value() = %d, expected %d",
					tt.memory, memoryQuantity.Value(), expectedBytes)
			}
		})
	}
}

// TestScaleResource 测试资源缩放
func TestScaleResource(t *testing.T) {
	tests := []struct {
		name     string
		amount   ResourceAmount
		factor   float64
		expected ResourceAmount
	}{
		{"scale by 1", 1000, 1.0, 1000},
		{"scale by 2", 1000, 2.0, 2000},
		{"scale by 0.5", 1000, 0.5, 500},
		{"scale by 0", 1000, 0.0, 0},
		{"scale negative factor", 1000, -1.0, 0},
		{"scale large number", 1000, 10.0, 10000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ScaleResource(tt.amount, tt.factor)
			if result != tt.expected {
				t.Errorf("ScaleResource(%d, %f) = %d, expected %d",
					tt.amount, tt.factor, result, tt.expected)
			}
		})
	}
}

// TestResourcesAsResourceList 测试Resources到ResourceList的转换
func TestResourcesAsResourceList(t *testing.T) {

	tests := []struct {
		name     string
		input    Resources
		expected v1.ResourceList
		warnings int
	}{
		{
			name:     "empty resources",
			input:    Resources{},
			expected: v1.ResourceList{},
			warnings: 0,
		},
		{
			name: "CPU and memory",
			input: Resources{
				ResourceCPU:    1500,
				ResourceMemory: 1024 * 1024 * 1024,
			},
			expected: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(1500, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(1024*1024*1024, resource.BinarySI),
			},
			warnings: 0,
		},
		{
			name: "unknown resource",
			input: Resources{
				ResourceCPU:             1000,
				ResourceName("unknown"): 500,
			},
			expected: v1.ResourceList{
				v1.ResourceCPU: *resource.NewMilliQuantity(1000, resource.DecimalSI),
			},
			warnings: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			result := ResourcesAsResourceList(tt.input)

			if len(result) != len(tt.expected) {
				t.Errorf("Expected %d resources, got %d", len(tt.expected), len(result))
			}

			for key, expectedQuantity := range tt.expected {
				if actualQuantity, exists := result[key]; !exists {
					t.Errorf("Expected resource %s not found", key)
				} else if !actualQuantity.Equal(expectedQuantity) {
					t.Errorf("Resource %s quantity mismatch: expected %v, got %v",
						key, expectedQuantity, actualQuantity)
				}
			}

		})
	}
}

// TestResourceNamesApiToModel 测试API资源名到模型资源名的转换
func TestResourceNamesApiToModel(t *testing.T) {

	tests := []struct {
		name     string
		input    []v1.ResourceName
		expected []ResourceName
		warnings int
	}{
		{
			name:     "empty array",
			input:    []v1.ResourceName{},
			expected: []ResourceName{},
			warnings: 0,
		},
		{
			name:     "CPU and memory",
			input:    []v1.ResourceName{v1.ResourceCPU, v1.ResourceMemory},
			expected: []ResourceName{ResourceCPU, ResourceMemory},
			warnings: 0,
		},
		{
			name:     "unknown resource",
			input:    []v1.ResourceName{v1.ResourceCPU, "unknown"},
			expected: []ResourceName{ResourceCPU},
			warnings: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			result := ResourceNamesApiToModel(tt.input)

			if len(*result) != len(tt.expected) {
				t.Errorf("Expected %d resources, got %d", len(tt.expected), len(*result))
			}

			for i, expected := range tt.expected {
				if (*result)[i] != expected {
					t.Errorf("Resource name mismatch at index %d: expected %s, got %s",
						i, expected, (*result)[i])
				}
			}

		})
	}
}

// TestRoundResourceAmount 测试资源数量舍入
func TestRoundResourceAmount(t *testing.T) {
	tests := []struct {
		name     string
		amount   ResourceAmount
		unit     ResourceAmount
		expected ResourceAmount
	}{
		{"round to 100", 123, 100, 100},
		{"round to 50", 123, 50, 100},
		{"round to 10", 123, 10, 120},
		{"round to 1", 123, 1, 123},
		{"round zero", 0, 100, 0},
		{"round smaller than unit", 50, 100, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := RoundResourceAmount(tt.amount, tt.unit)
			if result != tt.expected {
				t.Errorf("RoundResourceAmount(%d, %d) = %d, expected %d",
					tt.amount, tt.unit, result, tt.expected)
			}
		})
	}
}

// TestResourceAmountMax 测试资源数量最大值
func TestResourceAmountMax(t *testing.T) {
	tests := []struct {
		name     string
		amount1  ResourceAmount
		amount2  ResourceAmount
		expected ResourceAmount
	}{
		{"first larger", 100, 50, 100},
		{"second larger", 50, 100, 100},
		{"equal values", 100, 100, 100},
		{"zero values", 0, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ResourceAmountMax(tt.amount1, tt.amount2)
			if result != tt.expected {
				t.Errorf("ResourceAmountMax(%d, %d) = %d, expected %d",
					tt.amount1, tt.amount2, result, tt.expected)
			}
		})
	}
}

// TestResourceAmountFromFloat 测试浮点数到资源数量的转换
func TestResourceAmountFromFloat(t *testing.T) {
	tests := []struct {
		name     string
		input    float64
		expected ResourceAmount
	}{
		{"positive integer", 1000.0, 1000},
		{"positive fractional", 1000.5, 1000},
		{"zero", 0.0, 0},
		{"negative", -1000.0, 0},
		{"very large", 1e18, MaxResourceAmount},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := resourceAmountFromFloat(tt.input)
			if result != tt.expected {
				t.Errorf("resourceAmountFromFloat(%f) = %d, expected %d",
					tt.input, result, tt.expected)
			}
		})
	}
}

// TestAggregateStateKey 测试聚合状态键
func TestAggregateStateKey(t *testing.T) {
	podID := PodID{
		Namespace: "default",
		PodName:   "test-pod",
	}

	key := MakeAggregateStateKey(podID, "app")

	if key.Namespace() != "default" {
		t.Errorf("Expected namespace 'default', got '%s'", key.Namespace())
	}

	if key.PodName() != "test-pod" {
		t.Errorf("Expected pod name 'test-pod', got '%s'", key.PodName())
	}

	if key.ContainerName() != "app" {
		t.Errorf("Expected container name 'app', got '%s'", key.ContainerName())
	}
}
