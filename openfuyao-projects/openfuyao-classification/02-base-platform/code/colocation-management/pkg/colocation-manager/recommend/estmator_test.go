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
* Date: 2025-09-08
 */

package recommend

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
)

// TestGetResourceEstimation GetResourceEstimation
func TestGetResourceEstimation(t *testing.T) {
	tests := []struct {
		name      string
		resources aggregate.Resources
		expected  aggregate.Resources
	}{
		{
			name: "constant CPU and memory",
			resources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(2.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(1024 * 1024 * 1024),
			},
			expected: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(2.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(1024 * 1024 * 1024),
			},
		},
		{
			name: "zero resources",
			resources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(0),
			},
			expected: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(0),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			estimator := NewConstEstimator(tt.resources)

			result := estimator.GetResourceEstimation(nil)

			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestWithMarginGetResourceEstimation WithMargin
func TestWithMarginGetResourceEstimation(t *testing.T) {
	tests := []struct {
		name           string
		marginFraction float64
		baseResources  aggregate.Resources
		expected       aggregate.Resources
	}{
		{
			name:           "10% margin on CPU and memory",
			marginFraction: 0.1,
			baseResources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(2.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(1024 * 1024 * 1024),
			},
			expected: aggregate.Resources{
				// 2.0 + 0.2
				aggregate.ResourceCPU: aggregate.CPUAmountFromCores(2.2),
				// 1024 MB + (1024 * 0.1) = 1126.4 MB
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(1126.4 * 1024 * 1024),
			},
		},
		{
			name:           "20% margin",
			marginFraction: 0.2,
			baseResources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(1.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(512 * 1024 * 1024),
			},
			expected: aggregate.Resources{
				// 1.0 + 0.2
				aggregate.ResourceCPU: aggregate.CPUAmountFromCores(1.2),
				// 512 + 102.4 MB
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(614.4 * 1024 * 1024),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			baseEstimator := NewConstEstimator(tt.baseResources)
			estimator := WithMargin(tt.marginFraction, baseEstimator)

			result := estimator.GetResourceEstimation(nil)

			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestMinGetResourceEstimation minResourcesEstimator
func TestMinGetResourceEstimation(t *testing.T) {
	tests := []struct {
		name          string
		minResources  aggregate.Resources
		baseResources aggregate.Resources
		expected      aggregate.Resources
	}{
		{
			name: "base above minimum",
			minResources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(1.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(512 * 1024 * 1024),
			},
			baseResources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(2.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(1024 * 1024 * 1024),
			},
			expected: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(2.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(1024 * 1024 * 1024),
			},
		},
		{
			name: "base below minimum",
			minResources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(1.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(512 * 1024 * 1024),
			},
			baseResources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(0.5),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(256 * 1024 * 1024),
			},
			expected: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(1.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(512 * 1024 * 1024),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			baseEstimator := NewConstEstimator(tt.baseResources)
			estimator := WithMinResources(tt.minResources, baseEstimator)

			result := estimator.GetResourceEstimation(nil)

			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestEstimatorComposition
func TestEstimatorComposition(t *testing.T) {
	tests := []struct {
		name           string
		baseResources  aggregate.Resources
		minResources   aggregate.Resources
		marginFraction float64
		expected       aggregate.Resources
	}{
		{
			name: "margin with min resources - base below min",
			baseResources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(0.8),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(400 * 1024 * 1024),
			},
			minResources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(1.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(512 * 1024 * 1024),
			},
			marginFraction: 0.1,
			expected: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(1.0),                  // min(0.88, 1.0) = 1.0
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(512 * 1024 * 1024), // min(440, 512) = 512
			},
		},
		{
			name: "margin with min resources - base above min",
			baseResources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(2.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(1024 * 1024 * 1024),
			},
			minResources: aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(1.0),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(512 * 1024 * 1024),
			},
			marginFraction: 0.1,
			expected: aggregate.Resources{
				// 2.0 + 0.2
				aggregate.ResourceCPU: aggregate.CPUAmountFromCores(2.2),
				// 1024 MB + (1024 * 0.1) = 1126.4 MB
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(1126.4 * 1024 * 1024),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			baseEstimator := NewConstEstimator(tt.baseResources)
			withMargin := WithMargin(tt.marginFraction, baseEstimator)
			withMin := WithMinResources(tt.minResources, withMargin)

			result := withMin.GetResourceEstimation(nil)

			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestConfidenceMultiplier
func TestConfidenceMultiplier(t *testing.T) {
	tests := []struct {
		name       string
		multiplier float64
		exponent   float64
		baseCPU    float64
		baseMemory float64
	}{
		{
			name:       "basic confidence multiplier",
			multiplier: 1.0,
			exponent:   1.0,
			baseCPU:    2.0,
			baseMemory: 1024 * 1024 * 1024,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			baseResources := aggregate.Resources{
				aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(tt.baseCPU),
				aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(tt.baseMemory),
			}

			baseEstimator := NewConstEstimator(baseResources)
			estimator := WithConfidenceMultiplier(tt.multiplier, tt.exponent, baseEstimator)

			state := &aggregate.AggregateContainerState{
				TotalSamplesCount: 1000,
				FirstSampleStart:  time.Now().Add(-24 * time.Hour),
				LastSampleStart:   time.Now(),
			}

			result := estimator.GetResourceEstimation(state)

			assert.NotNil(t, result)
			assert.Contains(t, result, aggregate.ResourceCPU)
			assert.Contains(t, result, aggregate.ResourceMemory)
			assert.True(t, result[aggregate.ResourceCPU] > 0)
			assert.True(t, result[aggregate.ResourceMemory] > 0)
		})
	}
}

// TestPercentileEstimator 百分位预测
func TestPercentileEstimator(t *testing.T) {
	t.Run("percentile estimator creation", func(t *testing.T) {
		estimator := NewPercentileEstimator(0.9, 0.9)
		assert.NotNil(t, estimator)

		// 创建一个空的state来测试不会panic
		state := &aggregate.AggregateContainerState{}
		result := estimator.GetResourceEstimation(state)

		// 验证返回了Resources对象  空的histogram应该返回0.0
		assert.NotNil(t, result)
		assert.Contains(t, result, aggregate.ResourceCPU)
		assert.Contains(t, result, aggregate.ResourceMemory)
	})
}
