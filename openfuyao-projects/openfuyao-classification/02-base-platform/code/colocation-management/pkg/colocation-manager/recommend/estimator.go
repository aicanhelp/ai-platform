/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original file: https://github.com/kubernetes/autoscaler/tree/cluster-autoscaler-1.12.0-beta.1/vertical-pod-autoscaler/pkg/recommender/logic/estimator.go
*/

// Package recommend
package recommend

import (
	"math"
	"reflect"
	"time"

	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
)

var (
	houresPerDay      = 24
	minutesPerHour    = 60
	adjustedBaseValue = 1.0 // 桶大小的增长比例

)

// ResourceEstimator is a function from AggregateContainerState to
// aggregate.Resources, e.g. a prediction of resources needed by a group of
// containers.
type ResourceEstimator interface {
	GetResourceEstimation(s *aggregate.AggregateContainerState) aggregate.Resources
}

// Implementation of ResourceEstimator that returns constant amount of
// resources. This can be used as by a fake recommender for test purposes.
type constEstimator struct {
	resources aggregate.Resources
}

// Simple implementation of the ResourceEstimator interface. It returns specific
// percentiles of CPU usage distribution and memory peaks distribution.
type percentileEstimator struct {
	cpuPercentile    float64
	memoryPercentile float64
}

type marginEstimator struct {
	marginFraction float64
	baseEstimator  ResourceEstimator
}

type minResourcesEstimator struct {
	minResources  aggregate.Resources
	baseEstimator ResourceEstimator
}

type confidenceMultiplier struct {
	multiplier    float64
	exponent      float64
	baseEstimator ResourceEstimator
}

// NewConstEstimator returns a new constEstimator with given resources.
func NewConstEstimator(resources aggregate.Resources) ResourceEstimator {
	return &constEstimator{resources: resources}
}

// NewPercentileEstimator returns a new percentileEstimator that uses provided percentiles.
func NewPercentileEstimator(cpuPercentile float64, memoryPercentile float64) ResourceEstimator {
	return &percentileEstimator{cpuPercentile: cpuPercentile, memoryPercentile: memoryPercentile}
}

// WithMargin returns a given ResourceEstimator with margin applied.
// The returned resources are equal to the original resources plus (originalResource * marginFraction)
func WithMargin(marginFraction float64, baseEstimator ResourceEstimator) ResourceEstimator {
	return &marginEstimator{marginFraction: marginFraction, baseEstimator: baseEstimator}
}

// WithMinResources returns a given ResourceEstimator with minResources applied.
// The returned resources are equal to the max(original resources, minResources)
func WithMinResources(minResources aggregate.Resources, baseEstimator ResourceEstimator) ResourceEstimator {
	return &minResourcesEstimator{minResources: minResources, baseEstimator: baseEstimator}
}

// WithConfidenceMultiplier returns a given ResourceEstimator with confidenceMultiplier applied.
func WithConfidenceMultiplier(multiplier, exponent float64, baseEstimator ResourceEstimator) ResourceEstimator {
	return &confidenceMultiplier{multiplier: multiplier, exponent: exponent, baseEstimator: baseEstimator}
}

// GetResourceEstimation Returns a constant amount of resources.
func (e *constEstimator) GetResourceEstimation(s *aggregate.AggregateContainerState) aggregate.Resources {
	return e.resources
}

// GetResourceEstimation  Returns specific percentiles of CPU and memory peaks distributions.
func (e *percentileEstimator) GetResourceEstimation(s *aggregate.AggregateContainerState) aggregate.Resources {
	if reflect.DeepEqual(s, &aggregate.AggregateContainerState{}) {
		klog.Errorf("AggregateContainerState is nil")
		return aggregate.Resources{
			aggregate.ResourceCPU:    aggregate.CPUAmountFromCores(0.0),
			aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(0.0),
		}
	}
	return aggregate.Resources{
		aggregate.ResourceCPU: aggregate.CPUAmountFromCores(
			s.AggregateCPUUsage.Percentile(e.cpuPercentile)),
		aggregate.ResourceMemory: aggregate.MemoryAmountFromBytes(
			s.AggregateMemoryPeaks.Percentile(e.memoryPercentile)),
	}
}

// Returns a non-negative real number that heuristically measures how much
// confidence the history aggregated in the AggregateContainerState provides.
// For a workload producing a steady stream of samples over N days at the rate
// of 1 sample per minute, this metric is equal to N.
// This implementation is a very simple heuristic which looks at the total count
// of samples and the time between the first and the last sample.
func getConfidence(s *aggregate.AggregateContainerState) float64 {
	// Distance between the first and the last observed sample time, measured in days.
	lifespanInDays := float64(s.LastSampleStart.Sub(s.FirstSampleStart)) / float64(time.Hour*time.Duration(houresPerDay))
	// Total count of samples normalized such that it equals the number of days for
	// frequency of 1 sample/minute.
	samplesAmount := float64(s.TotalSamplesCount) / (float64(minutesPerHour * houresPerDay))
	return math.Min(lifespanInDays, samplesAmount)
}

// Returns resources computed by the underlying estimator, scaled based on the
// confidence metric, which depends on the amount of available historical data.
// Each resource is transformed as follows:
//
//	scaledResource = originalResource * (1 + 1/confidence)^exponent.
//
// This can be used to widen or narrow the gap between the lower and upper bound
// estimators depending on how much input data is available to the estimators.
func (e *confidenceMultiplier) GetResourceEstimation(s *aggregate.AggregateContainerState) aggregate.Resources {
	confidence := getConfidence(s)
	originalResources := e.baseEstimator.GetResourceEstimation(s)
	scaledResources := make(aggregate.Resources)
	for resource, resourceAmount := range originalResources {
		scaledResources[resource] = aggregate.ScaleResource(
			resourceAmount, math.Pow(adjustedBaseValue+e.multiplier/confidence, e.exponent))
	}
	return scaledResources
}

func (e *marginEstimator) GetResourceEstimation(s *aggregate.AggregateContainerState) aggregate.Resources {
	originalResources := e.baseEstimator.GetResourceEstimation(s)
	newResources := make(aggregate.Resources)
	for resource, resourceAmount := range originalResources {
		margin := aggregate.ScaleResource(resourceAmount, e.marginFraction)
		newResources[resource] = originalResources[resource] + margin
	}
	return newResources
}

func (e *minResourcesEstimator) GetResourceEstimation(s *aggregate.AggregateContainerState) aggregate.Resources {
	originalResources := e.baseEstimator.GetResourceEstimation(s)
	newResources := make(aggregate.Resources)
	for resource, resourceAmount := range originalResources {
		if resourceAmount < e.minResources[resource] {
			resourceAmount = e.minResources[resource]
		}
		newResources[resource] = resourceAmount
	}
	return newResources
}
