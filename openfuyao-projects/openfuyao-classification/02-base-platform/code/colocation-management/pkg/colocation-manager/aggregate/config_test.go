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

package aggregate

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// TestNewAggregationsConfig
func TestNewAggregationsConfig(t *testing.T) {
	params := AggregationsConfigParams{
		memoryAggregationInterval:      2 * time.Hour,
		memoryAggregationIntervalCount: 4,
		memoryHistogramDecayHalfLife:   12 * time.Hour,
		cpuHistogramDecayHalfLife:      12 * time.Hour,
		oomBumpUpRatio:                 1.3,
		oomMinBumpUp:                   200 * 1024 * 1024,
	}

	config := NewAggregationsConfig(params)

	assert.NotNil(t, config)
	assert.Equal(t, 2*time.Hour, config.MemoryAggregationInterval)
	assert.Equal(t, int64(4), config.MemoryAggregationIntervalCount)
	assert.Equal(t, 12*time.Hour, config.MemoryHistogramDecayHalfLife)
	assert.Equal(t, 12*time.Hour, config.CPUHistogramDecayHalfLife)
	assert.Equal(t, 1.3, config.OOMBumpUpRatio)
	assert.Equal(t, 200*1024*1024.0, config.OOMMinBumpUp)
	assert.Equal(t, DefaultHistogramBucketSizeGrowth, config.HistogramBucketSizeGrowth)

	assert.NotNil(t, config.CPUHistogramOptions)
	assert.NotNil(t, config.MemoryHistogramOptions)
}

// TestGetMemoryAggregationWindowLength
func TestGetMemoryAggregationWindowLength(t *testing.T) {
	params := AggregationsConfigParams{
		memoryAggregationInterval:      time.Hour,
		memoryAggregationIntervalCount: 8,
		memoryHistogramDecayHalfLife:   DefaultMemoryHistogramDecayHalfLife,
		cpuHistogramDecayHalfLife:      DefaultCPUHistogramDecayHalfLife,
		oomBumpUpRatio:                 DefaultOOMBumpUpRatio,
		oomMinBumpUp:                   DefaultOOMMinBumpUp,
	}

	config := NewAggregationsConfig(params)

	windowLength := config.GetMemoryAggregationWindowLength()
	expected := time.Hour * 8
	assert.Equal(t, expected, windowLength)

	params.memoryAggregationInterval = 30 * time.Minute
	params.memoryAggregationIntervalCount = 12
	config2 := NewAggregationsConfig(params)
	windowLength2 := config2.GetMemoryAggregationWindowLength()
	expected2 := 30 * time.Minute * 12
	assert.Equal(t, expected2, windowLength2)
}

// TestCPUHistogramOptions
func TestCPUHistogramOptions(t *testing.T) {
	params := AggregationsConfigParams{
		memoryAggregationInterval:      DefaultMemoryAggregationInterval,
		memoryAggregationIntervalCount: DefaultMemoryAggregationIntervalCount,
		memoryHistogramDecayHalfLife:   DefaultMemoryHistogramDecayHalfLife,
		cpuHistogramDecayHalfLife:      DefaultCPUHistogramDecayHalfLife,
		oomBumpUpRatio:                 DefaultOOMBumpUpRatio,
		oomMinBumpUp:                   DefaultOOMMinBumpUp,
	}

	config := NewAggregationsConfig(params)
	options := config.cpuHistogramOptions()

	assert.NotNil(t, options)

	assert.True(t, options.Epsilon() > 0)
}

// TestMemoryHistogramOptions
func TestMemoryHistogramOptions(t *testing.T) {
	params := AggregationsConfigParams{
		memoryAggregationInterval:      DefaultMemoryAggregationInterval,
		memoryAggregationIntervalCount: DefaultMemoryAggregationIntervalCount,
		memoryHistogramDecayHalfLife:   DefaultMemoryHistogramDecayHalfLife,
		cpuHistogramDecayHalfLife:      DefaultCPUHistogramDecayHalfLife,
		oomBumpUpRatio:                 DefaultOOMBumpUpRatio,
		oomMinBumpUp:                   DefaultOOMMinBumpUp,
	}

	config := NewAggregationsConfig(params)
	options := config.memoryHistogramOptions()

	assert.NotNil(t, options)
	assert.True(t, options.Epsilon() > 0)
}

// TestGetAggregationsConfig
func TestGetAggregationsConfig(t *testing.T) {
	config1 := GetAggregationsConfig()
	assert.NotNil(t, config1)

	config2 := GetAggregationsConfig()
	assert.Equal(t, config1, config2)

	assert.Equal(t, DefaultMemoryAggregationInterval, config1.MemoryAggregationInterval)
	assert.Equal(t, int64(DefaultMemoryAggregationIntervalCount), config1.MemoryAggregationIntervalCount)
	assert.Equal(t, DefaultMemoryHistogramDecayHalfLife, config1.MemoryHistogramDecayHalfLife)
	assert.Equal(t, DefaultCPUHistogramDecayHalfLife, config1.CPUHistogramDecayHalfLife)
	assert.Equal(t, DefaultOOMBumpUpRatio, config1.OOMBumpUpRatio)
	assert.Equal(t, DefaultOOMMinBumpUp, config1.OOMMinBumpUp)
	assert.Equal(t, DefaultHistogramBucketSizeGrowth, config1.HistogramBucketSizeGrowth)
}

// TestInitializeAggregationsConfig
func TestInitializeAggregationsConfig(t *testing.T) {
	aggregationsConfig = nil

	customConfig := &AggregationsConfig{
		MemoryAggregationInterval:      1 * time.Hour,
		MemoryAggregationIntervalCount: 6,
		HistogramBucketSizeGrowth:      0.1,
		MemoryHistogramDecayHalfLife:   6 * time.Hour,
		CPUHistogramDecayHalfLife:      6 * time.Hour,
		OOMBumpUpRatio:                 1.5,
		OOMMinBumpUp:                   50 * 1024 * 1024,
	}

	InitializeAggregationsConfig(customConfig)

	retrievedConfig := GetAggregationsConfig()
	assert.Equal(t, customConfig, retrievedConfig)
	assert.Equal(t, 1*time.Hour, retrievedConfig.MemoryAggregationInterval)
	assert.Equal(t, int64(6), retrievedConfig.MemoryAggregationIntervalCount)
	assert.Equal(t, 0.1, retrievedConfig.HistogramBucketSizeGrowth)
	assert.Equal(t, 1.5, retrievedConfig.OOMBumpUpRatio)
}

// TestDefaultValues
func TestDefaultValues(t *testing.T) {
	assert.Equal(t, time.Hour*24, DefaultMemoryAggregationInterval)
	assert.Equal(t, 8, DefaultMemoryAggregationIntervalCount)
	assert.Equal(t, 0.05, DefaultHistogramBucketSizeGrowth)
	assert.Equal(t, time.Hour*24, DefaultMemoryHistogramDecayHalfLife)
	assert.Equal(t, time.Hour*24, DefaultCPUHistogramDecayHalfLife)
	assert.Equal(t, 1.2, DefaultOOMBumpUpRatio)
	assert.Equal(t, 100*1024*1024.0, DefaultOOMMinBumpUp)
}

// TestHistogramOptionsCreation
func TestHistogramOptionsCreation(t *testing.T) {
	params := AggregationsConfigParams{
		memoryAggregationInterval:      DefaultMemoryAggregationInterval,
		memoryAggregationIntervalCount: DefaultMemoryAggregationIntervalCount,
		memoryHistogramDecayHalfLife:   DefaultMemoryHistogramDecayHalfLife,
		cpuHistogramDecayHalfLife:      DefaultCPUHistogramDecayHalfLife,
		oomBumpUpRatio:                 DefaultOOMBumpUpRatio,
		oomMinBumpUp:                   DefaultOOMMinBumpUp,
	}

	config := NewAggregationsConfig(params)

	cpuOptions := config.cpuHistogramOptions()
	memoryOptions := config.memoryHistogramOptions()

	assert.NotNil(t, cpuOptions)
	assert.NotNil(t, memoryOptions)

	config.HistogramBucketSizeGrowth = 0.01
	cpuOptions2 := config.cpuHistogramOptions()
	assert.NotNil(t, cpuOptions2)

	config.HistogramBucketSizeGrowth = 0.2
	memoryOptions2 := config.memoryHistogramOptions()
	assert.NotNil(t, memoryOptions2)
}

// TestConfigEdgeCases Test edge cases
func TestConfigEdgeCases(t *testing.T) {
	params := AggregationsConfigParams{
		memoryAggregationInterval:      0,
		memoryAggregationIntervalCount: 0,
		memoryHistogramDecayHalfLife:   0,
		cpuHistogramDecayHalfLife:      0,
		oomBumpUpRatio:                 0,
		oomMinBumpUp:                   0,
	}

	config := NewAggregationsConfig(params)
	assert.NotNil(t, config)

	assert.Equal(t, time.Duration(0), config.GetMemoryAggregationWindowLength())

	assert.NotNil(t, config.cpuHistogramOptions())
	assert.NotNil(t, config.memoryHistogramOptions())
}

// TestNegativeValues
func TestNegativeValues(t *testing.T) {
	params := AggregationsConfigParams{
		memoryAggregationInterval:      -time.Hour,
		memoryAggregationIntervalCount: -5,
		memoryHistogramDecayHalfLife:   -time.Hour,
		cpuHistogramDecayHalfLife:      -time.Hour,
		oomBumpUpRatio:                 -1.0,
		oomMinBumpUp:                   -1000000,
	}

	config := NewAggregationsConfig(params)
	assert.NotNil(t, config)

	windowLength := config.GetMemoryAggregationWindowLength()
	assert.Equal(t, time.Duration(5)*time.Hour, windowLength)
}

// TestOOMBumpUpValues
func TestOOMBumpUpValues(t *testing.T) {
	params := AggregationsConfigParams{
		memoryAggregationInterval:      DefaultMemoryAggregationInterval,
		memoryAggregationIntervalCount: DefaultMemoryAggregationIntervalCount,
		memoryHistogramDecayHalfLife:   DefaultMemoryHistogramDecayHalfLife,
		cpuHistogramDecayHalfLife:      DefaultCPUHistogramDecayHalfLife,
		oomBumpUpRatio:                 1.0,
		oomMinBumpUp:                   0,
	}

	config := NewAggregationsConfig(params)
	assert.Equal(t, 1.0, config.OOMBumpUpRatio)
	assert.Equal(t, 0.0, config.OOMMinBumpUp)

	params.oomBumpUpRatio = 10.0
	params.oomMinBumpUp = 1e12
	config2 := NewAggregationsConfig(params)
	assert.Equal(t, 10.0, config2.OOMBumpUpRatio)
	assert.Equal(t, 1e12, config2.OOMMinBumpUp)
}

// TestCPUHistogramOptionsPanic covers the defensive panic branch of
// cpuHistogramOptions when the histogram options are invalid (ratio <= 1.0).
func TestCPUHistogramOptionsPanic(t *testing.T) {
	// HistogramBucketSizeGrowth is added to coreBucketSizeRatioBase (1.0).
	// A growth of -0.5 yields ratio 0.5 <= 1.0 -> NewExponentialHistogramOptions errors -> panic.
	config := &AggregationsConfig{HistogramBucketSizeGrowth: -0.5}
	assert.Panics(t, func() {
		config.cpuHistogramOptions()
	})
}

// TestMemoryHistogramOptionsPanic covers the defensive panic branch of
// memoryHistogramOptions when the histogram options are invalid.
func TestMemoryHistogramOptionsPanic(t *testing.T) {
	config := &AggregationsConfig{HistogramBucketSizeGrowth: -0.5}
	assert.Panics(t, func() {
		config.memoryHistogramOptions()
	})
}

// BenchmarkGetAggregationsConfig Benchmark tests
func BenchmarkGetAggregationsConfig(b *testing.B) {
	aggregationsConfig = nil

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GetAggregationsConfig()
	}
}

// BenchmarkNewAggregationsConfig
func BenchmarkNewAggregationsConfig(b *testing.B) {
	params := AggregationsConfigParams{
		memoryAggregationInterval:      DefaultMemoryAggregationInterval,
		memoryAggregationIntervalCount: DefaultMemoryAggregationIntervalCount,
		memoryHistogramDecayHalfLife:   DefaultMemoryHistogramDecayHalfLife,
		cpuHistogramDecayHalfLife:      DefaultCPUHistogramDecayHalfLife,
		oomBumpUpRatio:                 DefaultOOMBumpUpRatio,
		oomMinBumpUp:                   DefaultOOMMinBumpUp,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewAggregationsConfig(params)
	}
}

// BenchmarkGetMemoryAggregationWindowLength
func BenchmarkGetMemoryAggregationWindowLength(b *testing.B) {
	config := GetAggregationsConfig()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		config.GetMemoryAggregationWindowLength()
	}
}
