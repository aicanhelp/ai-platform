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
	"github.com/stretchr/testify/mock"
)

type MockContainerStateAggregator struct {
	mock.Mock
}

func (m *MockContainerStateAggregator) AddSample(sample *ContainerUsageSample) {
	m.Called(sample)
}

func (m *MockContainerStateAggregator) SubtractSample(sample *ContainerUsageSample) {
	m.Called(sample)
}

// // TestSampleIsValid 测试容器使用样本的有效性验证
func TestSampleIsValid(t *testing.T) {
	tests := []struct {
		name             string
		sample           ContainerUsageSample
		expectedResource ResourceName
		expectedValid    bool
	}{
		{
			name: "valid CPU sample",
			sample: ContainerUsageSample{
				Usage: 1000, Resource: ResourceCPU,
			},
			expectedResource: ResourceCPU, expectedValid: true,
		},
		{
			name: "valid Memory sample",
			sample: ContainerUsageSample{
				Usage: 1024 * 1024, Resource: ResourceMemory,
			},
			expectedResource: ResourceMemory, expectedValid: true,
		},
		{
			name: "negative usage",
			sample: ContainerUsageSample{
				Usage: -100, Resource: ResourceCPU,
			},
			expectedResource: ResourceCPU, expectedValid: false,
		},
		{
			name: "wrong resource type",
			sample: ContainerUsageSample{
				Usage: 1000, Resource: ResourceCPU,
			},
			expectedResource: ResourceMemory, expectedValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.sample.isValid(tt.expectedResource)
			assert.Equal(t, tt.expectedValid, result)
		})
	}
}

// TestNewContainerState 测试创建新的容器状态
func TestNewContainerState(t *testing.T) {
	request := Resources{ResourceCPU: 1000, ResourceMemory: 1024 * 1024}
	mockAggregator := &MockContainerStateAggregator{}

	containerState := NewContainerState(request, mockAggregator)

	assert.NotNil(t, containerState)
	assert.Equal(t, request, containerState.Request)
	assert.Equal(t, mockAggregator, containerState.aggregator)
	assert.True(t, containerState.LastCPUSampleStart.IsZero())
	assert.True(t, containerState.WindowEnd.IsZero())
	assert.Equal(t, ResourceAmount(0), containerState.memoryPeak)
	assert.Equal(t, ResourceAmount(0), containerState.oomPeak)
}

// TestAddCPUSample 测试添加CPU使用样本
func TestAddSampleCPU(t *testing.T) {
	mockAggregator := &MockContainerStateAggregator{}
	request := Resources{ResourceCPU: 2000}
	containerState := NewContainerState(request, mockAggregator)

	// 使用固定时间而不是 time.Now()
	fixedTime := time.Date(2023, 1, 1, 10, 0, 0, 0, time.UTC)
	sample1 := &ContainerUsageSample{
		MeasureStart: fixedTime, Usage: 1000, Request: 2000, Resource: ResourceCPU,
	}

	mockAggregator.On("AddSample", sample1).Once()
	result := containerState.AddSample(sample1)

	assert.True(t, result)
	assert.Equal(t, fixedTime, containerState.LastCPUSampleStart)
	mockAggregator.AssertExpectations(t)

	earlierTime := fixedTime.Add(-time.Minute)
	sample2 := &ContainerUsageSample{
		MeasureStart: earlierTime, Usage: 1500, Request: 2000, Resource: ResourceCPU,
	}

	result = containerState.AddSample(sample2)
	assert.False(t, result)
	assert.Equal(t, fixedTime, containerState.LastCPUSampleStart)
}

// TestAddMemorySample 测试添加内存使用样本
func TestAddSampleMemory(t *testing.T) {
	mockAggregator := &MockContainerStateAggregator{}
	request := Resources{ResourceMemory: 1024 * 1024}
	containerState := NewContainerState(request, mockAggregator)

	originalConfig := aggregationsConfig
	aggregationsConfig = &AggregationsConfig{MemoryAggregationInterval: time.Minute}
	defer func() { aggregationsConfig = originalConfig }()

	now := time.Now()
	sample1 := &ContainerUsageSample{
		MeasureStart: now, Usage: 512 * 1024, Request: 1024 * 1024, Resource: ResourceMemory,
	}

	var capturedPeak *ContainerUsageSample
	mockAggregator.On("AddSample", mock.Anything).Run(func(args mock.Arguments) {
		if sample, ok := args.Get(0).(*ContainerUsageSample); ok {
			capturedPeak = sample
		} else {
			t.Fatalf("capturedPeak is not a ContainerUsageSample")
		}

	}).Once()

	result := containerState.AddSample(sample1)

	assert.True(t, result)
	assert.Equal(t, now, containerState.lastMemorySampleStart)

	expectedWindowEnd := now.Add(time.Minute)
	assert.Equal(t, expectedWindowEnd, containerState.WindowEnd)

	assert.Equal(t, ResourceAmount(512*1024), containerState.memoryPeak)

	assert.NotNil(t, capturedPeak)
	assert.Equal(t, expectedWindowEnd, capturedPeak.MeasureStart)
	assert.Equal(t, sample1.Usage, capturedPeak.Usage)
	assert.Equal(t, sample1.Request, capturedPeak.Request)
	assert.Equal(t, sample1.Resource, capturedPeak.Resource)

	mockAggregator.AssertExpectations(t)
}

// TestAddInvalidResource 测试添加无效资源类型的样本
func TestAddSampleInvalidResource(t *testing.T) {
	mockAggregator := &MockContainerStateAggregator{}
	containerState := NewContainerState(Resources{}, mockAggregator)

	sample := &ContainerUsageSample{
		MeasureStart: time.Now(), Usage: 1000, Resource: "invalid-resource",
	}

	result := containerState.AddSample(sample)
	assert.False(t, result)
}

// TestGetMemoryPeak 测试获取内存峰值
func TestGetMaxMemoryPeak(t *testing.T) {
	containerState := &ContainerState{}

	containerState.memoryPeak = 500
	containerState.oomPeak = 0
	assert.Equal(t, ResourceAmount(500), containerState.GetMaxMemoryPeak())

	containerState.memoryPeak = 0
	containerState.oomPeak = 800
	assert.Equal(t, ResourceAmount(800), containerState.GetMaxMemoryPeak())

	containerState.memoryPeak = 500
	containerState.oomPeak = 800
	assert.Equal(t, ResourceAmount(800), containerState.GetMaxMemoryPeak())

	containerState.memoryPeak = 900
	containerState.oomPeak = 800
	assert.Equal(t, ResourceAmount(900), containerState.GetMaxMemoryPeak())
}

// TestMemoryWindowManagement 测试内存窗口管理
func TestMemoryWindowManagement(t *testing.T) {
	mockAggregator := &MockContainerStateAggregator{}
	request := Resources{ResourceMemory: 1024 * 1024}
	containerState := NewContainerState(request, mockAggregator)

	originalConfig := aggregationsConfig
	aggregationsConfig = &AggregationsConfig{MemoryAggregationInterval: time.Hour}
	defer func() { aggregationsConfig = originalConfig }()

	var addedSamples, subtractedSamples []*ContainerUsageSample
	mockAggregator.On("AddSample", mock.Anything).Run(func(args mock.Arguments) {
		addedSamples = append(addedSamples, args.Get(0).(*ContainerUsageSample))
	})
	mockAggregator.On("SubtractSample", mock.Anything).Run(func(args mock.Arguments) {
		subtractedSamples = append(subtractedSamples, args.Get(0).(*ContainerUsageSample))
	})

	startTime := time.Date(2023, 1, 1, 10, 0, 0, 0, time.UTC)
	sample1 := &ContainerUsageSample{
		MeasureStart: startTime, Usage: 500 * 1024, Request: 1024 * 1024, Resource: ResourceMemory,
	}

	containerState.addMemorySample(sample1, false)
	// 第一个窗口结束时间应该是 11:00 (10:00 + 1h)
	expectedFirstWindowEnd := time.Date(2023, 1, 1, 11, 0, 0, 0, time.UTC)
	assert.Equal(t, expectedFirstWindowEnd, containerState.WindowEnd)
	assert.Equal(t, 1, len(addedSamples))
	assert.Equal(t, 0, len(subtractedSamples))

	sample2 := &ContainerUsageSample{
		MeasureStart: startTime.Add(30 * time.Minute), Usage: 800 * 1024, Request: 1024 * 1024, Resource: ResourceMemory,
	}

	containerState.addMemorySample(sample2, false)
	assert.Equal(t, ResourceAmount(800*1024), containerState.memoryPeak)
	assert.Equal(t, 2, len(addedSamples))
	assert.Equal(t, 1, len(subtractedSamples))
	// WindowEnd 应该仍然是 11:00
	assert.Equal(t, expectedFirstWindowEnd, containerState.WindowEnd)

	sample3 := &ContainerUsageSample{
		MeasureStart: startTime.Add(90 * time.Minute), // 11:30
		Usage:        600 * 1024, Request: 1024 * 1024, Resource: ResourceMemory,
	}

	containerState.addMemorySample(sample3, false)
	// 新窗口结束时间应该是 12:00 (按小时对齐)
	expectedNewWindowEnd := time.Date(2023, 1, 1, 12, 0, 0, 0, time.UTC)
	assert.Equal(t, expectedNewWindowEnd, containerState.WindowEnd)
	assert.Equal(t, ResourceAmount(600*1024), containerState.memoryPeak)
	assert.Equal(t, 3, len(addedSamples))
	assert.Equal(t, 1, len(subtractedSamples))
}

// TestOutOfOrderSample 测试乱序样本处理
func TestOutOfOrderSample(t *testing.T) {
	mockAggregator := &MockContainerStateAggregator{}
	containerState := NewContainerState(Resources{}, mockAggregator)

	now := time.Now()
	sample1 := &ContainerUsageSample{
		MeasureStart: now, Usage: 1000, Request: 2000, Resource: ResourceMemory,
	}

	mockAggregator.On("AddSample", mock.Anything).Once()
	containerState.addMemorySample(sample1, false)

	earlierSample := &ContainerUsageSample{
		MeasureStart: now.Add(-time.Minute), Usage: 1500, Request: 2000, Resource: ResourceMemory,
	}

	result := containerState.addMemorySample(earlierSample, false)
	assert.False(t, result)
	mockAggregator.AssertNumberOfCalls(t, "AddSample", 1)
}

// TestLowerPeakSample 测试低峰值样本处理
func TestLowerPeakSample(t *testing.T) {
	mockAggregator := &MockContainerStateAggregator{}
	containerState := NewContainerState(Resources{}, mockAggregator)

	now := time.Now()
	sample1 := &ContainerUsageSample{
		MeasureStart: now, Usage: 1000, Request: 2000, Resource: ResourceMemory,
	}

	mockAggregator.On("AddSample", mock.Anything).Once()
	containerState.addMemorySample(sample1, false)

	sample2 := &ContainerUsageSample{
		MeasureStart: now.Add(time.Minute), Usage: 500, Request: 2000, Resource: ResourceMemory,
	}

	result := containerState.addMemorySample(sample2, false)
	assert.True(t, result)
	mockAggregator.AssertNumberOfCalls(t, "AddSample", 1)
}

// TestAddMemorySampleInvalid 测试无效内存样本处理
func TestAddMemorySampleInvalid(t *testing.T) {
	mockAggregator := &MockContainerStateAggregator{}
	containerState := NewContainerState(Resources{}, mockAggregator)

	invalidSample := &ContainerUsageSample{
		MeasureStart: time.Now(), Usage: -100, Request: 2000, Resource: ResourceMemory,
	}

	result := containerState.addMemorySample(invalidSample, false)
	assert.False(t, result)

	wrongResourceSample := &ContainerUsageSample{
		MeasureStart: time.Now(), Usage: 1000, Request: 2000, Resource: ResourceCPU,
	}

	result = containerState.addMemorySample(wrongResourceSample, false)
	assert.False(t, result)
}

// TestAddMemorySampleOOM covers the isOOM=true branch of addMemorySample,
// which sets the oomPeak field instead of the memoryPeak field.
func TestAddMemorySampleOOM(t *testing.T) {
	mockAggregator := &MockContainerStateAggregator{}
	containerState := NewContainerState(Resources{ResourceMemory: 1024 * 1024}, mockAggregator)

	originalConfig := aggregationsConfig
	aggregationsConfig = &AggregationsConfig{MemoryAggregationInterval: time.Minute}
	defer func() { aggregationsConfig = originalConfig }()

	var addedSamples []*ContainerUsageSample
	var subtractedSamples []*ContainerUsageSample
	mockAggregator.On("AddSample", mock.Anything).Run(func(args mock.Arguments) {
		addedSamples = append(addedSamples, args.Get(0).(*ContainerUsageSample))
	})
	mockAggregator.On("SubtractSample", mock.Anything).Run(func(args mock.Arguments) {
		subtractedSamples = append(subtractedSamples, args.Get(0).(*ContainerUsageSample))
	})

	now := time.Now()
	oomSample := &ContainerUsageSample{
		MeasureStart: now,
		Usage:        900 * 1024,
		Request:      1024 * 1024,
		Resource:     ResourceMemory,
	}

	// OOM samples are always processed (regardless of order) and recorded via oomPeak.
	result := containerState.addMemorySample(oomSample, true)
	assert.True(t, result)
	assert.Equal(t, ResourceAmount(900*1024), containerState.oomPeak)
	assert.Equal(t, ResourceAmount(0), containerState.memoryPeak)
	assert.Equal(t, 1, len(addedSamples))
	assert.Equal(t, now, containerState.lastMemorySampleStart)

	// GetMaxMemoryPeak should reflect the oomPeak.
	assert.Equal(t, ResourceAmount(900*1024), containerState.GetMaxMemoryPeak())

	// A second OOM sample within the same window with a larger usage triggers
	// a peak replacement (subtract old, add new) and updates oomPeak.
	largerOOM := &ContainerUsageSample{
		MeasureStart: now.Add(30 * time.Second), // still within WindowEnd (now + 1m)
		Usage:        1200 * 1024,
		Request:      1024 * 1024,
		Resource:     ResourceMemory,
	}
	result = containerState.addMemorySample(largerOOM, true)
	assert.True(t, result)
	assert.Equal(t, ResourceAmount(1200*1024), containerState.oomPeak)
	// one subtract (old peak) and two adds (initial + replacement)
	assert.Equal(t, 2, len(addedSamples))
	assert.Equal(t, 1, len(subtractedSamples))
}

// BenchmarkAddSampleCPU 测试CPU样本添加性能
func BenchmarkAddSampleCPU(b *testing.B) {
	mockAggregator := &MockContainerStateAggregator{}
	mockAggregator.On("AddSample", mock.Anything).Return()
	containerState := NewContainerState(Resources{ResourceCPU: 2000}, mockAggregator)

	sample := &ContainerUsageSample{
		MeasureStart: time.Now(), Usage: 1000, Request: 2000, Resource: ResourceCPU,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		containerState.AddSample(sample)
	}
}

// BenchmarkAddSampleMemory 测试内存样本添加性能
func BenchmarkAddSampleMemory(b *testing.B) {
	mockAggregator := &MockContainerStateAggregator{}
	mockAggregator.On("AddSample", mock.Anything).Return()
	containerState := NewContainerState(Resources{ResourceMemory: 1024 * 1024}, mockAggregator)

	sample := &ContainerUsageSample{
		MeasureStart: time.Now(), Usage: 512 * 1024, Request: 1024 * 1024, Resource: ResourceMemory,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		containerState.AddSample(sample)
	}
}

// BenchmarkGetMaxMemoryPeak 测试获取最大内存峰值性能
func BenchmarkGetMaxMemoryPeak(b *testing.B) {
	containerState := &ContainerState{memoryPeak: 500, oomPeak: 800}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = containerState.GetMaxMemoryPeak()
	}
}
