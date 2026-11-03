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

package npu

import (
	"strings"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
)

const sampleMetrics = `# TYPE container_npu_utilization gauge
container_npu_utilization{id="0",namespace="ns",pod_name="pod-a"} 51 1777454292000
container_npu_utilization{id="1",namespace="ns",pod_name="pod-a"} 47 1777454292500
container_npu_utilization{id="0",namespace="ns",pod_name="pod-b"} 9 1777454292000
# TYPE npu_chip_info_hbm_utilization gauge
npu_chip_info_hbm_utilization{id="0",namespace="ns",pod_name="pod-a"} 30
npu_chip_info_hbm_utilization{id="1",namespace="ns",pod_name="pod-a"} 50
# TYPE npu_chip_info_hbm_bandwidth_utilization gauge
npu_chip_info_hbm_bandwidth_utilization{id="0",namespace="ns",pod_name="pod-a"} 20
npu_chip_info_hbm_bandwidth_utilization{id="1",namespace="ns",pod_name="pod-a"} 60
`

func parseFixture(t *testing.T, body string) map[string]*dto.MetricFamily {
	t.Helper()
	parser := expfmt.NewTextParser(model.LegacyValidation)
	families, err := parser.TextToMetricFamilies(strings.NewReader(body))
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return families
}

func defaultParamsForTest() Parameters {
	p := Parameters{}
	p.applyDefaults()
	return p
}

func TestBuildPodAggregatesAcrossDevices(t *testing.T) {
	families := parseFixture(t, sampleMetrics)
	pod := buildPod(defaultParamsForTest(), "ns", "pod-a", []targetResult{{
		families: families, fetchedAt: time.Unix(200, 0),
	}})

	if !pod.Matched {
		t.Fatalf("expected matched pod")
	}
	if pod.DeviceCount != 2 || len(pod.Devices) != 2 {
		t.Fatalf("expected 2 devices, got %+v", pod)
	}
	if pod.Devices[0].ID != "0" || pod.Devices[1].ID != "1" {
		t.Fatalf("expected sorted device IDs, got %+v", pod.Devices)
	}
	if pod.AICoreUtilizationMax != 51 || pod.AICoreUtilizationAvg != 49 {
		t.Fatalf("ai core agg wrong: avg=%v max=%v", pod.AICoreUtilizationAvg, pod.AICoreUtilizationMax)
	}
	if pod.HBMUtilizationMax != 50 || pod.HBMUtilizationAvg != 40 {
		t.Fatalf("hbm agg wrong: avg=%v max=%v", pod.HBMUtilizationAvg, pod.HBMUtilizationMax)
	}
	if pod.HBMBandwidthUtilizationMax != 60 || pod.HBMBandwidthUtilizationAvg != 40 {
		t.Fatalf("hbm bw agg wrong: avg=%v max=%v", pod.HBMBandwidthUtilizationAvg, pod.HBMBandwidthUtilizationMax)
	}
	if got := pod.ObservedAt.UnixMilli(); got != 1777454292500 {
		t.Fatalf("expected ObservedAt from latest sample timestamp, got %d", got)
	}
}

func TestBuildPodFiltersOtherPods(t *testing.T) {
	families := parseFixture(t, sampleMetrics)
	pod := buildPod(defaultParamsForTest(), "ns", "pod-b", []targetResult{{
		families: families, fetchedAt: time.Unix(200, 0),
	}})
	if !pod.Matched || pod.DeviceCount != 1 {
		t.Fatalf("expected pod-b to match exactly 1 device, got %+v", pod)
	}
}

func TestBuildPodNoMatchProducesUnmatched(t *testing.T) {
	families := parseFixture(t, sampleMetrics)
	pod := buildPod(defaultParamsForTest(), "ns", "missing", []targetResult{{
		families: families, fetchedAt: time.Unix(200, 0),
	}})
	if pod.Matched || pod.DeviceCount != 0 {
		t.Fatalf("expected unmatched, got %+v", pod)
	}
}

func TestBuildPodDedupsWithinSingleTarget(t *testing.T) {
	// Same parsed families seen twice under the same target URL — defends
	// against malformed exporter output repeating a (family, deviceID).
	families := parseFixture(t, sampleMetrics)
	pod := buildPod(defaultParamsForTest(), "ns", "pod-a", []targetResult{
		{target: "http://node-a:8080/metrics", families: families, fetchedAt: time.Unix(200, 0)},
		{target: "http://node-a:8080/metrics", families: families, fetchedAt: time.Unix(300, 0)},
	})
	if pod.DeviceCount != 2 {
		t.Fatalf("expected per-target dedup to keep 2 devices, got %d", pod.DeviceCount)
	}
}

func TestBuildPodTreatsSameDeviceIDOnDifferentTargetsAsDistinctCards(t *testing.T) {
	// Models a pod that spans two nodes; each node's exporter reports the
	// pod under its own local id=0. These are two physical cards and must
	// both contribute to pod-level aggregates.
	const targetA = `# TYPE container_npu_utilization gauge
container_npu_utilization{id="0",namespace="ns",pod_name="pod-a"} 40
# TYPE npu_chip_info_hbm_utilization gauge
npu_chip_info_hbm_utilization{id="0",namespace="ns",pod_name="pod-a"} 20
# TYPE npu_chip_info_hbm_bandwidth_utilization gauge
npu_chip_info_hbm_bandwidth_utilization{id="0",namespace="ns",pod_name="pod-a"} 10
`
	const targetB = `# TYPE container_npu_utilization gauge
container_npu_utilization{id="0",namespace="ns",pod_name="pod-a"} 60
# TYPE npu_chip_info_hbm_utilization gauge
npu_chip_info_hbm_utilization{id="0",namespace="ns",pod_name="pod-a"} 80
# TYPE npu_chip_info_hbm_bandwidth_utilization gauge
npu_chip_info_hbm_bandwidth_utilization{id="0",namespace="ns",pod_name="pod-a"} 30
`
	pod := buildPod(defaultParamsForTest(), "ns", "pod-a", []targetResult{
		{target: "http://node-a:8080/metrics", families: parseFixture(t, targetA), fetchedAt: time.Unix(200, 0)},
		{target: "http://node-b:8080/metrics", families: parseFixture(t, targetB), fetchedAt: time.Unix(200, 0)},
	})
	if pod.DeviceCount != 2 {
		t.Fatalf("expected 2 distinct devices across targets, got %d (%+v)", pod.DeviceCount, pod.Devices)
	}
	if pod.Devices[0].Target != "http://node-a:8080/metrics" || pod.Devices[1].Target != "http://node-b:8080/metrics" {
		t.Fatalf("expected sort by (Target, ID), got %+v", pod.Devices)
	}
	if pod.AICoreUtilizationAvg != 50 || pod.AICoreUtilizationMax != 60 {
		t.Fatalf("ai core agg wrong: avg=%v max=%v", pod.AICoreUtilizationAvg, pod.AICoreUtilizationMax)
	}
	if pod.HBMUtilizationAvg != 50 || pod.HBMUtilizationMax != 80 {
		t.Fatalf("hbm agg wrong: avg=%v max=%v", pod.HBMUtilizationAvg, pod.HBMUtilizationMax)
	}
	if pod.HBMBandwidthUtilizationAvg != 20 || pod.HBMBandwidthUtilizationMax != 30 {
		t.Fatalf("hbm bw agg wrong: avg=%v max=%v", pod.HBMBandwidthUtilizationAvg, pod.HBMBandwidthUtilizationMax)
	}
}

func TestBuildPodFallsBackToFetchedAtWhenNoSampleTimestamp(t *testing.T) {
	const body = `# TYPE container_npu_utilization gauge
container_npu_utilization{id="0",namespace="ns",pod_name="pod-a"} 10
`
	families := parseFixture(t, body)
	fetchedAt := time.Unix(500, 0)
	pod := buildPod(defaultParamsForTest(), "ns", "pod-a", []targetResult{{
		families: families, fetchedAt: fetchedAt,
	}})
	if !pod.ObservedAt.Equal(fetchedAt) {
		t.Fatalf("expected fallback to fetchedAt, got %s", pod.ObservedAt)
	}
}
