/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original file: https://github.com/kubernetes/autoscaler/tree/cluster-autoscaler-chart-1.0.2/vertical-pod-autoscaler/pkg/recommender/model/aggregate_container_state_test.go
Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.

*/

package aggregate

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	vpav1 "k8s.io/autoscaler/vertical-pod-autoscaler/pkg/apis/autoscaling.k8s.io/v1"

	colocationv1 "openfuyao.com/colocation-management/pkg/apis/v1"
)

var (
	testTimestamp = time.Date(2024, time.January, 1, 12, 0, 0, 0, time.UTC)
	testPodID     = PodID{Namespace: "test-namespace", PodName: "test-pod"}
	testContainer = "test-container"
)

func TestNewAggregateContainerState(t *testing.T) {
	aggregationsConfig = nil
	state := NewAggregateContainerState()
	assert.NotNil(t, state)
	assert.NotNil(t, state.AggregateCPUUsage)
	assert.NotNil(t, state.AggregateMemoryPeaks)
	assert.True(t, state.AggregateCPUUsage.IsEmpty())
	assert.True(t, state.AggregateMemoryPeaks.IsEmpty())
	assert.Zero(t, state.TotalSamplesCount)
	assert.True(t, state.CreationTime.After(testTimestamp.Add(-time.Minute)))
}

func TestMergeContainerState(t *testing.T) {
	aggregationsConfig = nil
	state1 := NewAggregateContainerState()
	state2 := NewAggregateContainerState()

	// Add samples to both states
	sampleTime1 := testTimestamp
	sampleTime2 := testTimestamp.Add(time.Hour)

	state1.AddSample(&ContainerUsageSample{
		MeasureStart: sampleTime1,
		Usage:        CPUAmountFromCores(1.0),
		Request:      CPUAmountFromCores(2.0),
		Resource:     ResourceCPU,
	})

	state2.AddSample(&ContainerUsageSample{
		MeasureStart: sampleTime2,
		Usage:        CPUAmountFromCores(2.0),
		Request:      CPUAmountFromCores(3.0),
		Resource:     ResourceCPU,
	})

	state2.AddSample(&ContainerUsageSample{
		MeasureStart: sampleTime2,
		Usage:        MemoryAmountFromBytes(1e9),
		Request:      MemoryAmountFromBytes(2e9),
		Resource:     ResourceMemory,
	})

	// Merge state2 into state1
	state1.MergeContainerState(state2)

	assert.Equal(t, 2, state1.TotalSamplesCount)
	assert.Equal(t, sampleTime1, state1.FirstSampleStart)
	assert.Equal(t, sampleTime2, state1.LastSampleStart)
	assert.False(t, state1.AggregateCPUUsage.IsEmpty())
	assert.False(t, state1.AggregateMemoryPeaks.IsEmpty())
}

func TestContainerStateAddSampleCPU(t *testing.T) {
	aggregationsConfig = nil
	state := NewAggregateContainerState()
	sampleTime := testTimestamp

	sample := &ContainerUsageSample{
		MeasureStart: sampleTime,
		Usage:        CPUAmountFromCores(1.5),
		Request:      CPUAmountFromCores(2.0),
		Resource:     ResourceCPU,
	}

	state.AddSample(sample)

	assert.Equal(t, 1, state.TotalSamplesCount)
	assert.Equal(t, sampleTime, state.FirstSampleStart)
	assert.Equal(t, sampleTime, state.LastSampleStart)
	assert.False(t, state.AggregateCPUUsage.IsEmpty())
}

func TestContainerStateAddSampleMemory(t *testing.T) {
	aggregationsConfig = nil
	state := NewAggregateContainerState()
	sampleTime := testTimestamp

	sample := &ContainerUsageSample{
		MeasureStart: sampleTime,
		Usage:        MemoryAmountFromBytes(512 * 1024 * 1024),      // 512MB
		Request:      MemoryAmountFromBytes(1 * 1024 * 1024 * 1024), // 1GB
		Resource:     ResourceMemory,
	}

	state.AddSample(sample)

	assert.Equal(t, 0, state.TotalSamplesCount) // Memory samples don't count towards total
	assert.False(t, state.AggregateMemoryPeaks.IsEmpty())
}

func TestSubtractSampleMemory(t *testing.T) {
	aggregationsConfig = nil
	state := NewAggregateContainerState()
	sampleTime := testTimestamp

	// First add a sample
	addSample := &ContainerUsageSample{
		MeasureStart: sampleTime,
		Usage:        MemoryAmountFromBytes(512 * 1024 * 1024),
		Request:      MemoryAmountFromBytes(1 * 1024 * 1024 * 1024),
		Resource:     ResourceMemory,
	}
	state.AddSample(addSample)

	// Then subtract the same sample
	subtractSample := &ContainerUsageSample{
		MeasureStart: sampleTime,
		Usage:        MemoryAmountFromBytes(512 * 1024 * 1024),
		Request:      MemoryAmountFromBytes(1 * 1024 * 1024 * 1024),
		Resource:     ResourceMemory,
	}
	state.SubtractSample(subtractSample)

	assert.True(t, state.AggregateMemoryPeaks.IsEmpty())
}

func TestSaveToCheckpoint(t *testing.T) {
	aggregationsConfig = nil
	state := NewAggregateContainerState()
	sampleTime := testTimestamp

	// Add some samples
	state.AddSample(&ContainerUsageSample{
		MeasureStart: sampleTime,
		Usage:        CPUAmountFromCores(1.0),
		Request:      CPUAmountFromCores(2.0),
		Resource:     ResourceCPU,
	})

	state.AddSample(&ContainerUsageSample{
		MeasureStart: sampleTime,
		Usage:        MemoryAmountFromBytes(1e9),
		Request:      MemoryAmountFromBytes(2e9),
		Resource:     ResourceMemory,
	})

	checkpoint, err := state.SaveToCheckpoint()
	assert.NoError(t, err)

	assert.Equal(t, 1, checkpoint.TotalSamplesCount)
	assert.Equal(t, sampleTime, checkpoint.FirstSampleStart.Time)
	assert.Equal(t, sampleTime, checkpoint.LastSampleStart.Time)
	assert.Equal(t, SupportedCheckpointVersion, checkpoint.Version)
	assert.NotNil(t, checkpoint.CPUHistogram)
	assert.NotNil(t, checkpoint.MemoryHistogram)
}

func TestLoadFromCheckpoint(t *testing.T) {
	aggregationsConfig = nil
	state := NewAggregateContainerState()
	checkpointTime := testTimestamp

	checkpoint := &colocationv1.ContainerCheckpointStatus{
		Version:           SupportedCheckpointVersion,
		LastUpdateTime:    metav1.NewTime(time.Now()),
		FirstSampleStart:  metav1.NewTime(checkpointTime),
		LastSampleStart:   metav1.NewTime(checkpointTime.Add(time.Hour)),
		TotalSamplesCount: 5,
	}

	err := state.LoadFromCheckpoint(checkpoint)
	assert.NoError(t, err)
	assert.Equal(t, 5, state.TotalSamplesCount)
	assert.Equal(t, checkpointTime, state.FirstSampleStart)
	assert.Equal(t, checkpointTime.Add(time.Hour), state.LastSampleStart)
}

func TestLoadFromCheckpointVersionMismatch(t *testing.T) {
	aggregationsConfig = nil
	state := NewAggregateContainerState()
	checkpoint := &colocationv1.ContainerCheckpointStatus{
		Version: "unsupported-version",
	}

	err := state.LoadFromCheckpoint(checkpoint)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported checkpoint version")
}

func TestIsExpired(t *testing.T) {
	aggregationsConfig = nil
	cs := NewAggregateContainerState()
	cs.LastSampleStart = testTimestamp
	cs.TotalSamplesCount = 1
	assert.False(t, cs.isExpired(testTimestamp.Add(7*24*time.Hour)))
	assert.True(t, cs.isExpired(testTimestamp.Add(8*24*time.Hour)))

	csEmpty := NewAggregateContainerState()
	csEmpty.TotalSamplesCount = 0
	csEmpty.CreationTime = testTimestamp
	assert.False(t, csEmpty.isExpired(testTimestamp.Add(7*24*time.Hour)))
	assert.True(t, csEmpty.isExpired(testTimestamp.Add(8*24*time.Hour)))
}

func TestIsEmpty(t *testing.T) {
	aggregationsConfig = nil
	state := NewAggregateContainerState()
	assert.True(t, state.isEmpty())

	state.TotalSamplesCount = 1
	assert.False(t, state.isEmpty())
}

// TestAddSampleUnsupportedResourcePanic verifies the default branch of
// AggregateContainerState.AddSample panics for an unsupported resource type.
func TestAddSampleUnsupportedResourcePanic(t *testing.T) {
	aggregationsConfig = nil
	state := NewAggregateContainerState()
	sample := &ContainerUsageSample{
		MeasureStart: testTimestamp,
		Usage:        100,
		Request:      100,
		Resource:     ResourceName("bogus-resource"),
	}
	assert.Panics(t, func() {
		state.AddSample(sample)
	})
}

// TestSubtractSampleUnsupportedResourcePanic verifies the default branch of
// AggregateContainerState.SubtractSample panics for a non-memory resource.
func TestSubtractSampleUnsupportedResourcePanic(t *testing.T) {
	aggregationsConfig = nil
	state := NewAggregateContainerState()
	sample := &ContainerUsageSample{
		MeasureStart: testTimestamp,
		Usage:        100,
		Request:      100,
		Resource:     ResourceCPU,
	}
	assert.Panics(t, func() {
		state.SubtractSample(sample)
	})
}

// TestLoadFromCheckpointHistogramError verifies the error paths of
// LoadFromCheckpoint when the underlying histograms fail to load. The decaying
// histogram's LoadFromCheckpoint forwards to the inner histogram, which errors
// on a negative TotalWeight.
func TestLoadFromCheckpointHistogramError(t *testing.T) {
	aggregationsConfig = nil

	t.Run("corrupted memory histogram", func(t *testing.T) {
		state := NewAggregateContainerState()
		// A memory histogram with a negative total weight triggers an error.
		checkpoint := &colocationv1.ContainerCheckpointStatus{
			Version:          SupportedCheckpointVersion,
			FirstSampleStart: metav1.NewTime(testTimestamp),
			LastSampleStart:  metav1.NewTime(testTimestamp.Add(time.Hour)),
			MemoryHistogram:  vpav1.HistogramCheckpoint{TotalWeight: -1.0},
		}
		err := state.LoadFromCheckpoint(checkpoint)
		assert.Error(t, err)
	})

	t.Run("corrupted cpu histogram", func(t *testing.T) {
		state := NewAggregateContainerState()
		// Memory side loads fine (valid empty checkpoint), but the CPU
		// histogram has a negative total weight -> error.
		checkpoint := &colocationv1.ContainerCheckpointStatus{
			Version:          SupportedCheckpointVersion,
			FirstSampleStart: metav1.NewTime(testTimestamp),
			LastSampleStart:  metav1.NewTime(testTimestamp.Add(time.Hour)),
			CPUHistogram:     vpav1.HistogramCheckpoint{TotalWeight: -1.0},
		}
		err := state.LoadFromCheckpoint(checkpoint)
		assert.Error(t, err)
	})
}

// TestContainerStateAggregatorProxy covers the proxy AddSample/SubtractSample
// forwarding methods, including the previously-uncovered SubtractSample.
func TestContainerStateAggregatorProxy(t *testing.T) {
	aggregationsConfig = nil
	ctx := context.Background()
	node := createTestNode("proxy-node")
	nodeState := NewNodeState(ctx, node, getTestConfig(),
		func(key AggregateStateKey, state *AggregateContainerState) {})

	podID := PodID{Namespace: "default", PodName: "proxy-pod"}
	nodeState.AddOrUpdatePod(podID, corev1.PodRunning)
	containerID := ContainerID{PodID: podID, ContainerName: "proxy-container"}

	proxy := NewContainerStateAggregatorProxy(nodeState, containerID)
	assert.NotNil(t, proxy)

	memSample := &ContainerUsageSample{
		MeasureStart: testTimestamp,
		Usage:        MemoryAmountFromBytes(512 * 1024 * 1024),
		Request:      MemoryAmountFromBytes(1 * 1024 * 1024 * 1024),
		Resource:     ResourceMemory,
	}
	// AddSample forwards to the aggregate state.
	proxy.AddSample(memSample)
	assert.False(t, nodeState.aggregateStateMap[
		MakeAggregateStateKey(podID, "proxy-container")].AggregateMemoryPeaks.IsEmpty())

	// SubtractSample forwards and removes the previously-added peak.
	proxy.SubtractSample(memSample)
	assert.True(t, nodeState.aggregateStateMap[
		MakeAggregateStateKey(podID, "proxy-container")].AggregateMemoryPeaks.IsEmpty())

	cpuSample := &ContainerUsageSample{
		MeasureStart: testTimestamp.Add(time.Second),
		Usage:        CPUAmountFromCores(1.0),
		Request:      CPUAmountFromCores(1.0),
		Resource:     ResourceCPU,
	}
	proxy.AddSample(cpuSample)
	assert.Equal(t, 1, nodeState.aggregateStateMap[
		MakeAggregateStateKey(podID, "proxy-container")].TotalSamplesCount)
}
