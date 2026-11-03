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

Original file: https://github.com/kubernetes/autoscaler/tree/cluster-autoscaler-chart-1.0.2/vertical-pod-autoscaler/pkg/recommender/model/aggregate_container_state.go
Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
Modifications:
Modified function SaveToCheckpoint
Modified struct ContainerStateAggregatorProxy

*/

package aggregate

import (
	"fmt"
	"math"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/autoscaler/vertical-pod-autoscaler/pkg/recommender/util"

	colocationv1 "openfuyao.com/colocation-management/pkg/apis/v1"
)

// ContainerNameToAggregateStateMap maps a container name to AggregateContainerState
// that aggregates state of containers with that name.
type ContainerNameToAggregateStateMap map[string]*AggregateContainerState

const (
	// SupportedCheckpointVersion is the tag of the supported version of serialized checkpoints.
	// Version id should be incremented on every non incompatible change, i.e. if the new
	// version of the recommender binary can't initialize from the old checkpoint format or the
	// previous version of the recommender binary can't initialize from the new checkpoint format.
	SupportedCheckpointVersion = "v1"
)

var (
	// DefaultControlledResources is a default value of Spec.ResourcePolicy.ContainerPolicies[].ControlledResources.
	defaultControlledResources = []ResourceName{ResourceCPU, ResourceMemory}
)

// ContainerStateAggregator is an interface for objects that consume and
// aggregate container usage samples.
type ContainerStateAggregator interface {
	// AddSample aggregates a single usage sample.
	AddSample(sample *ContainerUsageSample)
	// SubtractSample removes a single usage sample. The subtracted sample
	// should be equal to some sample that was aggregated with AddSample()
	// in the past.
	SubtractSample(sample *ContainerUsageSample)
}

// AggregateContainerState It can be used as an input to compute the recommendation.
// The CPU and memory distributions use decaying histograms by default
// (see NewAggregateContainerState()).
// Implements ContainerStateAggregator interface.
// AggregateContainerState holds input signals aggregated from a set of containers.
type AggregateContainerState struct {
	// AggregateCPUUsage is a distribution of all CPU samples.
	AggregateCPUUsage util.Histogram
	// AggregateMemoryPeaks is a distribution of memory peaks from all containers:
	// each container should add one peak per memory aggregation interval (e.g. once every 24h).
	AggregateMemoryPeaks util.Histogram
	// Note: first/last sample timestamps as well as the sample count are based only on CPU samples.
	FirstSampleStart  time.Time
	LastSampleStart   time.Time
	TotalSamplesCount int
	CreationTime      time.Time
}

// MergeContainerState merges two AggregateContainerStates.
func (a *AggregateContainerState) MergeContainerState(other *AggregateContainerState) {
	a.AggregateCPUUsage.Merge(other.AggregateCPUUsage)
	a.AggregateMemoryPeaks.Merge(other.AggregateMemoryPeaks)

	if a.FirstSampleStart.IsZero() ||
		(!other.FirstSampleStart.IsZero() && other.FirstSampleStart.Before(a.FirstSampleStart)) {
		a.FirstSampleStart = other.FirstSampleStart
	}
	if other.LastSampleStart.After(a.LastSampleStart) {
		a.LastSampleStart = other.LastSampleStart
	}
	a.TotalSamplesCount += other.TotalSamplesCount
}

// NewAggregateContainerState returns a new, empty AggregateContainerState.
func NewAggregateContainerState() *AggregateContainerState {
	config := GetAggregationsConfig()
	return &AggregateContainerState{
		AggregateCPUUsage:    util.NewDecayingHistogram(config.CPUHistogramOptions, config.CPUHistogramDecayHalfLife),
		AggregateMemoryPeaks: util.NewDecayingHistogram(config.MemoryHistogramOptions, config.MemoryHistogramDecayHalfLife),
		CreationTime:         time.Now(),
	}
}

// AddSample aggregates a single usage sample.
func (a *AggregateContainerState) AddSample(sample *ContainerUsageSample) {
	switch sample.Resource {
	case ResourceCPU:
		a.addCPUSample(sample)
	case ResourceMemory:
		a.AggregateMemoryPeaks.AddSample(BytesFromMemoryAmount(sample.Usage), 1.0, sample.MeasureStart)
	default:
		panic(fmt.Sprintf("AddSample doesn't support resource '%s'", sample.Resource))
	}
}

// SubtractSample removes a single usage sample from an aggregation.
// The subtracted sample should be equal to some sample that was aggregated with
// AddSample() in the past.
// Only memory samples can be subtracted at the moment. Support for CPU could be
// added if necessary.
func (a *AggregateContainerState) SubtractSample(sample *ContainerUsageSample) {
	switch sample.Resource {
	case ResourceMemory:
		a.AggregateMemoryPeaks.SubtractSample(BytesFromMemoryAmount(sample.Usage), 1.0, sample.MeasureStart)
	default:
		panic(fmt.Sprintf("SubtractSample doesn't support resource '%s'", sample.Resource))
	}
}

func (a *AggregateContainerState) addCPUSample(sample *ContainerUsageSample) {
	cpuUsageCores := CoresFromCPUAmount(sample.Usage)
	cpuRequestCores := CoresFromCPUAmount(sample.Request)
	// Samples are added with the weight equal to the current request. This means that
	// whenever the request is increased, the history accumulated so far effectively decays,
	// which helps react quickly to CPU starvation.
	a.AggregateCPUUsage.AddSample(
		cpuUsageCores, math.Max(cpuRequestCores, minSampleWeight), sample.MeasureStart)
	if sample.MeasureStart.After(a.LastSampleStart) {
		a.LastSampleStart = sample.MeasureStart
	}
	if a.FirstSampleStart.IsZero() || sample.MeasureStart.Before(a.FirstSampleStart) {
		a.FirstSampleStart = sample.MeasureStart
	}
	a.TotalSamplesCount++
}

// SaveToCheckpoint serializes AggregateContainerState as VerticalPodAutoscalerCheckpointStatus.
// The serialization may result in loss of precission of the histograms.
func (a *AggregateContainerState) SaveToCheckpoint() (*colocationv1.ContainerCheckpointStatus, error) {
	memory, err := a.AggregateMemoryPeaks.SaveToChekpoint()
	if err != nil {
		return nil, err
	}
	cpu, err := a.AggregateCPUUsage.SaveToChekpoint()
	if err != nil {
		return nil, err
	}
	return &colocationv1.ContainerCheckpointStatus{
		LastUpdateTime:    metav1.NewTime(time.Now()),
		FirstSampleStart:  metav1.NewTime(a.FirstSampleStart),
		LastSampleStart:   metav1.NewTime(a.LastSampleStart),
		TotalSamplesCount: a.TotalSamplesCount,
		MemoryHistogram:   *memory,
		CPUHistogram:      *cpu,
		Version:           SupportedCheckpointVersion,
	}, nil
}

// LoadFromCheckpoint deserializes data from VerticalPodAutoscalerCheckpointStatus
// into the AggregateContainerState.
func (a *AggregateContainerState) LoadFromCheckpoint(checkpoint *colocationv1.ContainerCheckpointStatus) error {
	if checkpoint.Version != SupportedCheckpointVersion {
		return fmt.Errorf("unsupported checkpoint version %s", checkpoint.Version)
	}
	a.TotalSamplesCount = checkpoint.TotalSamplesCount
	a.FirstSampleStart = checkpoint.FirstSampleStart.Time
	a.LastSampleStart = checkpoint.LastSampleStart.Time
	err := a.AggregateMemoryPeaks.LoadFromCheckpoint(&checkpoint.MemoryHistogram)
	if err != nil {
		return err
	}
	err = a.AggregateCPUUsage.LoadFromCheckpoint(&checkpoint.CPUHistogram)
	if err != nil {
		return err
	}
	return nil
}

func (a *AggregateContainerState) isExpired(now time.Time) bool {
	if a.isEmpty() {
		return now.Sub(a.CreationTime) >= GetAggregationsConfig().GetMemoryAggregationWindowLength()
	}
	return now.Sub(a.LastSampleStart) >= GetAggregationsConfig().GetMemoryAggregationWindowLength()
}

func (a *AggregateContainerState) isEmpty() bool {
	return a.TotalSamplesCount == 0
}

// ContainerStateAggregatorProxy is a wrapper for ContainerStateAggregator
// that creates ContainerStateAgregator for container if it is no longer
// present in the cluster state.
type ContainerStateAggregatorProxy struct {
	containerID ContainerID
	cluster     *NodeState
}

// NewContainerStateAggregatorProxy creates a ContainerStateAggregatorProxy
// pointing to the cluster state.
func NewContainerStateAggregatorProxy(cluster *NodeState, containerID ContainerID) ContainerStateAggregator {
	return &ContainerStateAggregatorProxy{
		containerID: containerID,
		cluster:     cluster}
}

// AddSample adds a container sample to the aggregator.
func (p *ContainerStateAggregatorProxy) AddSample(sample *ContainerUsageSample) {
	aggregator := p.cluster.findOrCreateAggregateContainerState(p.containerID)
	aggregator.AddSample(sample)
}

// SubtractSample subtracts a container sample from the aggregator.
func (p *ContainerStateAggregatorProxy) SubtractSample(sample *ContainerUsageSample) {
	aggregator := p.cluster.findOrCreateAggregateContainerState(p.containerID)
	aggregator.SubtractSample(sample)
}
