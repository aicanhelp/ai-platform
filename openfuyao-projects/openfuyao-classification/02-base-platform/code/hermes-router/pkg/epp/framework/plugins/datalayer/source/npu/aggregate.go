/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package npu provides NPU data-layer source, extractor, and attribute support.
package npu

import (
	"reflect"
	"sort"
	"time"

	dto "github.com/prometheus/client_model/go"
)

// Fixed Prometheus family names consumed by this datasource. Configurable
// metric selection was deliberately removed from the contract: the three
// utilization gauges below are the only signals routing requires.
const (
	familyAICoreUtilization       = "container_npu_utilization"
	familyHBMUtilization          = "npu_chip_info_hbm_utilization"
	familyHBMBandwidthUtilization = "npu_chip_info_hbm_bandwidth_utilization"

	deviceLabelKey = "id"
)

// PodType returns the reflect.Type of Pod, used by the GIE runtime to validate
// extractor compatibility against this source's OutputType.
func PodType() reflect.Type {
	return reflect.TypeFor[Pod]()
}

// Device is a single NPU device's per-poll utilization snapshot. Target is
// the exporter URL the sample was scraped from; deviceID is exporter-local
// (typically the per-node NPU index starting at 0), so the same ID can
// legitimately appear under different targets and refer to different
// physical cards. Identity is therefore (Target, ID), not ID alone.
type Device struct {
	Target                  string
	ID                      string
	AICoreUtilization       float64
	HBMUtilization          float64
	HBMBandwidthUtilization float64
}

// Pod is the per-endpoint datasource output. Matched=false means the scrape
// produced no samples attributable to this pod; the extractor must no-op.
type Pod struct {
	Namespace                  string
	PodName                    string
	Matched                    bool
	ObservedAt                 time.Time
	Devices                    []Device
	DeviceCount                int
	AICoreUtilizationAvg       float64
	AICoreUtilizationMax       float64
	HBMUtilizationAvg          float64
	HBMUtilizationMax          float64
	HBMBandwidthUtilizationAvg float64
	HBMBandwidthUtilizationMax float64
}

// targetResult is the per-target output of a scrape, including the parsed
// payload. Failed targets are not represented here; the caller drops them
// after logging. target carries the exporter URL so that snapshot.buildPod
// can scope device identity per target.
type targetResult struct {
	target    string
	families  map[string]*dto.MetricFamily
	fetchedAt time.Time
}

// buildPod folds successful per-target scrape results into a single Pod
// snapshot for the given namespace/podName. Device identity is
// (target, deviceID): the same deviceID seen across two different targets
// represents two distinct physical cards on two different nodes and both
// contribute to pod-level aggregates. Within one target, a duplicate
// (family, deviceID) sample is treated as a malformed-but-harmless repeat
// and the first occurrence wins.
func buildPod(params Parameters, namespace, podName string, results []targetResult) Pod {
	pod := Pod{Namespace: namespace, PodName: podName}

	deviceMap := map[deviceKey]*Device{}
	var maxSampleTS time.Time
	var maxFetchedAt time.Time

	for _, result := range results {
		if result.fetchedAt.After(maxFetchedAt) {
			maxFetchedAt = result.fetchedAt
		}
		// Per-target dedup scope: prevents a single malformed scrape from
		// double-counting one (family, deviceID), without collapsing the
		// same deviceID coming from two different exporter targets.
		seenSamples := map[string]struct{}{}
		context := sampleApplyContext{
			deviceMap:   deviceMap,
			seenSamples: seenSamples,
			maxSampleTS: &maxSampleTS,
			target:      result.target,
			families:    result.families,
			params:      params,
			namespace:   namespace,
			podName:     podName,
		}
		applySamples(context, familyAICoreUtilization, func(d *Device, v float64) { d.AICoreUtilization = v })
		applySamples(context, familyHBMUtilization, func(d *Device, v float64) { d.HBMUtilization = v })
		applySamples(context, familyHBMBandwidthUtilization, func(d *Device, v float64) {
			d.HBMBandwidthUtilization = v
		})
	}

	pod.Devices = sortedDevices(deviceMap)
	pod.DeviceCount = len(pod.Devices)
	pod.Matched = pod.DeviceCount > 0
	pod.ObservedAt = pickObservedAt(maxSampleTS, maxFetchedAt)
	computePodAggregates(&pod)
	return pod
}

// deviceKey is the composite identity for a physical NPU card observed
// through one specific exporter target.
type deviceKey struct {
	target string
	id     string
}

// applySamples walks one Prometheus family, filters samples down to the
// target pod, and folds each unique (family, deviceID) sample for the given
// exporter target into deviceMap via the assign callback.
type sampleApplyContext struct {
	deviceMap   map[deviceKey]*Device
	seenSamples map[string]struct{}
	maxSampleTS *time.Time
	target      string
	families    map[string]*dto.MetricFamily
	params      Parameters
	namespace   string
	podName     string
}

func applySamples(context sampleApplyContext, family string, assign func(*Device, float64)) {
	if !context.ready(assign) {
		return
	}
	mf := context.families[family]
	if mf == nil || mf.GetType() != dto.MetricType_GAUGE {
		return
	}
	for _, metric := range mf.GetMetric() {
		ns, pod, deviceID := requiredLabels(metric.GetLabel(), context.params.NamespaceLabelKey, context.params.PodLabelKey)
		if ns != context.namespace || pod != context.podName {
			continue
		}
		if deviceID == "" || metric.GetGauge() == nil {
			continue
		}
		dedupeKey := family + "\x00" + deviceID
		if _, ok := context.seenSamples[dedupeKey]; ok {
			continue
		}
		context.seenSamples[dedupeKey] = struct{}{}

		key := deviceKey{target: context.target, id: deviceID}
		device, ok := context.deviceMap[key]
		if !ok {
			device = &Device{Target: context.target, ID: deviceID}
			context.deviceMap[key] = device
		}
		assign(device, metric.GetGauge().GetValue())

		if ts := metric.GetTimestampMs(); ts > 0 {
			observedAt := time.UnixMilli(ts)
			if observedAt.After(*context.maxSampleTS) {
				*context.maxSampleTS = observedAt
			}
		}
	}
}

func (context sampleApplyContext) ready(assign func(*Device, float64)) bool {
	return context.deviceMap != nil &&
		context.seenSamples != nil &&
		context.maxSampleTS != nil &&
		assign != nil
}

// requiredLabels does a single linear scan of a sample's label list and
// returns the namespace, pod and device-id values. It avoids allocating a
// label map per metric, which is hot when many endpoints share one exporter
// target. Absent labels surface as empty strings, and the caller applies its
// own equality check.
func requiredLabels(labels []*dto.LabelPair, namespaceKey, podKey string) (namespace, pod, deviceID string) {
	for _, label := range labels {
		switch label.GetName() {
		case namespaceKey:
			namespace = label.GetValue()
		case podKey:
			pod = label.GetValue()
		case deviceLabelKey:
			deviceID = label.GetValue()
		default:
			continue
		}
	}
	return namespace, pod, deviceID
}

// sortedDevices returns devices in deterministic (Target, ID) order so that
// downstream consumers and tests see a stable layout regardless of map
// iteration order or the goroutine ordering of scrapeAllTargets.
func sortedDevices(deviceMap map[deviceKey]*Device) []Device {
	keys := make([]deviceKey, 0, len(deviceMap))
	for key := range deviceMap {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].target != keys[j].target {
			return keys[i].target < keys[j].target
		}
		return keys[i].id < keys[j].id
	})
	out := make([]Device, 0, len(keys))
	for _, key := range keys {
		out = append(out, *deviceMap[key])
	}
	return out
}

// pickObservedAt prefers per-sample exporter timestamps when available, since
// those describe when the device value was actually measured. It falls back
// to the latest fetch time across targets when the exporter omits timestamps.
func pickObservedAt(maxSampleTS, maxFetchedAt time.Time) time.Time {
	if !maxSampleTS.IsZero() {
		return maxSampleTS
	}
	return maxFetchedAt
}

func computePodAggregates(pod *Pod) {
	if len(pod.Devices) == 0 {
		return
	}
	var aiSum, hbmSum, bwSum float64
	for _, device := range pod.Devices {
		aiSum += device.AICoreUtilization
		hbmSum += device.HBMUtilization
		bwSum += device.HBMBandwidthUtilization
		if device.AICoreUtilization > pod.AICoreUtilizationMax {
			pod.AICoreUtilizationMax = device.AICoreUtilization
		}
		if device.HBMUtilization > pod.HBMUtilizationMax {
			pod.HBMUtilizationMax = device.HBMUtilization
		}
		if device.HBMBandwidthUtilization > pod.HBMBandwidthUtilizationMax {
			pod.HBMBandwidthUtilizationMax = device.HBMBandwidthUtilization
		}
	}
	count := float64(len(pod.Devices))
	pod.AICoreUtilizationAvg = aiSum / count
	pod.HBMUtilizationAvg = hbmSum / count
	pod.HBMBandwidthUtilizationAvg = bwSum / count
}
