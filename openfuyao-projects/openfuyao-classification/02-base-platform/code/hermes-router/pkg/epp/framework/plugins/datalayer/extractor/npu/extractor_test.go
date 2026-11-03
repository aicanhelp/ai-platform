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
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	attrnpu "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/npu"
	npusrc "hermes-router/pkg/epp/framework/plugins/datalayer/source/npu"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

const (
	testExtractorName = "npu-extractor"
	testAttributeKey  = "test-npu-state"
	testStaleAfter    = 5 * time.Second
)

func newTestEndpoint() fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{}, nil)
}

func validExtractorRaw(t *testing.T) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(map[string]string{
		"attributeKey": testAttributeKey,
		"staleAfter":   testStaleAfter.String(),
	})
	if err != nil {
		t.Fatalf("marshal raw params: %v", err)
	}
	return raw
}

func TestFactoryHappyPath(t *testing.T) {
	plugin, err := Factory(testExtractorName, validExtractorRaw(t), nil)
	if err != nil {
		t.Fatalf("Factory returned error: %v", err)
	}
	extractor, ok := plugin.(*Extractor)
	if !ok {
		t.Fatalf("Factory returned %T, want *Extractor", plugin)
	}
	if got := extractor.TypedName().Type; got != NPUExporterExtractorType {
		t.Fatalf("TypedName().Type = %q, want %q", got, NPUExporterExtractorType)
	}
	if got := extractor.TypedName().Name; got != testExtractorName {
		t.Fatalf("TypedName().Name = %q, want %q", got, testExtractorName)
	}
}

func TestFactoryEmptyRawUsesDefaultKeyButRequiresStaleAfter(t *testing.T) {
	_, err := Factory(testExtractorName, nil, nil)
	if err == nil {
		t.Fatalf("expected error because staleAfter is missing, got nil")
	}
	if !strings.Contains(err.Error(), "staleAfter") {
		t.Fatalf("error %q should mention staleAfter", err.Error())
	}
}

func TestFactoryValidationErrors(t *testing.T) {
	cases := map[string]struct {
		raw      string
		contains string
	}{
		"empty attributeKey": {
			raw:      `{"attributeKey":"","staleAfter":"5s"}`,
			contains: "attributeKey",
		},
		"zero staleAfter": {
			raw:      `{"attributeKey":"k","staleAfter":"0s"}`,
			contains: "staleAfter",
		},
		"negative staleAfter": {
			raw:      `{"attributeKey":"k","staleAfter":"-1s"}`,
			contains: "staleAfter",
		},
		"unknown field": {
			raw:      `{"attributeKey":"k","staleAfter":"5s","unexpected":true}`,
			contains: "unexpected",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := Factory(testExtractorName, json.RawMessage(tc.raw), nil)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.contains)
			}
			if !strings.Contains(err.Error(), tc.contains) {
				t.Fatalf("error %q should contain %q", err.Error(), tc.contains)
			}
		})
	}
}

func TestExpectedInputType(t *testing.T) {
	e := New(testExtractorName, testAttributeKey, testStaleAfter)
	if got := e.ExpectedInputType(); got != npusrc.PodType() {
		t.Fatalf("ExpectedInputType() = %v, want %v", got, npusrc.PodType())
	}
}

func TestExtractRejectsWrongInputType(t *testing.T) {
	e := New(testExtractorName, testAttributeKey, testStaleAfter)
	err := e.Extract(context.Background(), "not a pod", newTestEndpoint())
	if err == nil {
		t.Fatalf("expected error on wrong input type")
	}
}

func TestExtractUnmatchedPodIsNoOp(t *testing.T) {
	e := New(testExtractorName, testAttributeKey, testStaleAfter)
	ep := newTestEndpoint()
	pod := npusrc.Pod{Matched: false}

	if err := e.Extract(context.Background(), pod, ep); err != nil {
		t.Fatalf("Extract returned error: %v", err)
	}
	if keys := ep.GetAttributes().Keys(); len(keys) != 0 {
		t.Fatalf("expected no attribute Put, got keys=%v", keys)
	}
}

func TestExtractUnmatchedAgesStoredStateToStale(t *testing.T) {
	observedAt := time.Unix(1_000_000, 0)
	freshPod := npusrc.Pod{
		Matched:     true,
		ObservedAt:  observedAt,
		DeviceCount: 1,
		Devices:     []npusrc.Device{{ID: "0", AICoreUtilization: 10}},
	}

	t.Run("ages to stale once observation is older than staleAfter", func(t *testing.T) {
		e := New(testExtractorName, testAttributeKey, testStaleAfter)
		now := observedAt
		e.nowFn = func() time.Time { return now }

		ep := newTestEndpoint()
		if err := e.Extract(context.Background(), freshPod, ep); err != nil {
			t.Fatalf("seed Extract: %v", err)
		}
		if got := extractedState(t, ep).DataStatus; got != attrnpu.DataPresent {
			t.Fatalf("seeded DataStatus = %q, want Present", got)
		}

		// Advance past staleAfter and feed an unmatched pod (e.g. exporter
		// failure or pod no longer present in scrape). The previously stored
		// state must transition to Stale rather than remain Present forever.
		now = observedAt.Add(testStaleAfter + time.Second)
		if err := e.Extract(context.Background(), npusrc.Pod{Matched: false}, ep); err != nil {
			t.Fatalf("unmatched Extract: %v", err)
		}
		if got := extractedState(t, ep).DataStatus; got != attrnpu.DataStale {
			t.Fatalf("aged DataStatus = %q, want Stale", got)
		}
	})

	t.Run("does not age while still within staleAfter", func(t *testing.T) {
		e := New(testExtractorName, testAttributeKey, testStaleAfter)
		now := observedAt
		e.nowFn = func() time.Time { return now }

		ep := newTestEndpoint()
		if err := e.Extract(context.Background(), freshPod, ep); err != nil {
			t.Fatalf("seed Extract: %v", err)
		}
		now = observedAt.Add(testStaleAfter) // boundary still counts as fresh.
		if err := e.Extract(context.Background(), npusrc.Pod{Matched: false}, ep); err != nil {
			t.Fatalf("unmatched Extract: %v", err)
		}
		if got := extractedState(t, ep).DataStatus; got != attrnpu.DataPresent {
			t.Fatalf("DataStatus = %q, want Present at boundary", got)
		}
	})
}

func TestExtractMatchedPodOverwritesStaleState(t *testing.T) {
	// Verifies the Present -> Stale -> Present recovery path on a single
	// long-lived endpoint: after aging to Stale, a fresh matched scrape must
	// fully replace the stored state (no leftover devices or aggregates from
	// the previous generation).
	observedAt := time.Unix(1_000_000, 0)
	firstPod := npusrc.Pod{
		Matched:                    true,
		ObservedAt:                 observedAt,
		DeviceCount:                2,
		AICoreUtilizationAvg:       30,
		AICoreUtilizationMax:       40,
		HBMUtilizationAvg:          20,
		HBMUtilizationMax:          25,
		HBMBandwidthUtilizationAvg: 10,
		HBMBandwidthUtilizationMax: 15,
		Devices: []npusrc.Device{
			{ID: "0", AICoreUtilization: 20},
			{ID: "1", AICoreUtilization: 40},
		},
	}
	refreshObservedAt := observedAt.Add(10 * testStaleAfter)
	secondPod := npusrc.Pod{
		Matched:                    true,
		ObservedAt:                 refreshObservedAt,
		DeviceCount:                1,
		AICoreUtilizationAvg:       80,
		AICoreUtilizationMax:       80,
		HBMUtilizationAvg:          70,
		HBMUtilizationMax:          70,
		HBMBandwidthUtilizationAvg: 60,
		HBMBandwidthUtilizationMax: 60,
		Devices: []npusrc.Device{
			{ID: "7", AICoreUtilization: 80, HBMUtilization: 70, HBMBandwidthUtilization: 60},
		},
	}

	e := New(testExtractorName, testAttributeKey, testStaleAfter)
	now := observedAt
	e.nowFn = func() time.Time { return now }
	ep := newTestEndpoint()

	// 1. Seed Present.
	if err := e.Extract(context.Background(), firstPod, ep); err != nil {
		t.Fatalf("seed Extract: %v", err)
	}
	if got := extractedState(t, ep).DataStatus; got != attrnpu.DataPresent {
		t.Fatalf("seeded DataStatus = %q, want Present", got)
	}

	// 2. Age past staleAfter via an unmatched scrape -> Stale.
	now = observedAt.Add(testStaleAfter + time.Second)
	if err := e.Extract(context.Background(), npusrc.Pod{Matched: false}, ep); err != nil {
		t.Fatalf("age Extract: %v", err)
	}
	if got := extractedState(t, ep).DataStatus; got != attrnpu.DataStale {
		t.Fatalf("aged DataStatus = %q, want Stale", got)
	}

	// 3. New matched scrape with a different device set must fully replace the
	//    stored state. No leftover devices, aggregates, or ObservationTime.
	now = refreshObservedAt
	if err := e.Extract(context.Background(), secondPod, ep); err != nil {
		t.Fatalf("refresh Extract: %v", err)
	}
	state := extractedState(t, ep)
	if state.DataStatus != attrnpu.DataPresent {
		t.Fatalf("refreshed DataStatus = %q, want Present", state.DataStatus)
	}
	if !state.ObservationTime.Equal(refreshObservedAt) {
		t.Fatalf("ObservationTime = %v, want %v", state.ObservationTime, refreshObservedAt)
	}
	if state.DeviceCount != secondPod.DeviceCount {
		t.Fatalf("DeviceCount = %d, want %d", state.DeviceCount, secondPod.DeviceCount)
	}
	if len(state.Devices) != 1 || state.Devices[0].DeviceID != "7" {
		t.Fatalf("Devices = %+v, want only id=7", state.Devices)
	}
	if state.AICoreUtilizationAvg != secondPod.AICoreUtilizationAvg ||
		state.AICoreUtilizationMax != secondPod.AICoreUtilizationMax ||
		state.HBMUtilizationAvg != secondPod.HBMUtilizationAvg ||
		state.HBMUtilizationMax != secondPod.HBMUtilizationMax ||
		state.HBMBandwidthUtilizationAvg != secondPod.HBMBandwidthUtilizationAvg ||
		state.HBMBandwidthUtilizationMax != secondPod.HBMBandwidthUtilizationMax {
		t.Fatalf("aggregates not refreshed: got %+v want from %+v", state, secondPod)
	}
}

func extractedState(t *testing.T, ep fwkdl.Endpoint) *attrnpu.State {
	t.Helper()
	value, ok := ep.GetAttributes().Get(testAttributeKey)
	if !ok {
		t.Fatalf("expected attribute %q to be set", testAttributeKey)
	}
	state, ok := value.(*attrnpu.State)
	if !ok || state == nil {
		t.Fatalf("attribute %q is %T, want non-nil *State", testAttributeKey, value)
	}
	return state
}

func TestExtractMatchedPodPublishesState(t *testing.T) {
	observedAt := time.Unix(1_000_000, 0)
	pod := npusrc.Pod{
		Matched:                    true,
		ObservedAt:                 observedAt,
		DeviceCount:                2,
		AICoreUtilizationAvg:       49,
		AICoreUtilizationMax:       51,
		HBMUtilizationAvg:          40,
		HBMUtilizationMax:          50,
		HBMBandwidthUtilizationAvg: 40,
		HBMBandwidthUtilizationMax: 60,
		Devices: []npusrc.Device{
			{ID: "0", AICoreUtilization: 47, HBMUtilization: 30, HBMBandwidthUtilization: 20},
			{ID: "1", AICoreUtilization: 51, HBMUtilization: 50, HBMBandwidthUtilization: 60},
		},
	}

	t.Run("present when fresh", func(t *testing.T) {
		e := New(testExtractorName, testAttributeKey, testStaleAfter)
		e.nowFn = func() time.Time { return observedAt.Add(testStaleAfter) }

		ep := newTestEndpoint()
		if err := e.Extract(context.Background(), pod, ep); err != nil {
			t.Fatalf("Extract: %v", err)
		}
		state := extractedState(t, ep)

		if state.DataStatus != attrnpu.DataPresent {
			t.Fatalf("DataStatus = %q, want Present", state.DataStatus)
		}
		if state.DeviceCount != pod.DeviceCount {
			t.Fatalf("DeviceCount = %d, want %d", state.DeviceCount, pod.DeviceCount)
		}
		if state.AICoreUtilizationAvg != pod.AICoreUtilizationAvg ||
			state.AICoreUtilizationMax != pod.AICoreUtilizationMax ||
			state.HBMUtilizationAvg != pod.HBMUtilizationAvg ||
			state.HBMUtilizationMax != pod.HBMUtilizationMax ||
			state.HBMBandwidthUtilizationAvg != pod.HBMBandwidthUtilizationAvg ||
			state.HBMBandwidthUtilizationMax != pod.HBMBandwidthUtilizationMax {
			t.Fatalf("aggregates not propagated: %+v", state)
		}
		if len(state.Devices) != len(pod.Devices) {
			t.Fatalf("Devices len = %d, want %d", len(state.Devices), len(pod.Devices))
		}
		for i, device := range state.Devices {
			if device.DeviceID != pod.Devices[i].ID ||
				device.AICoreUtilization != pod.Devices[i].AICoreUtilization ||
				device.HBMUtilization != pod.Devices[i].HBMUtilization ||
				device.HBMBandwidthUtilization != pod.Devices[i].HBMBandwidthUtilization {
				t.Fatalf("Devices[%d] mismatch: got %+v want source %+v", i, device, pod.Devices[i])
			}
		}
	})

	t.Run("stale when older than staleAfter", func(t *testing.T) {
		e := New(testExtractorName, testAttributeKey, testStaleAfter)
		e.nowFn = func() time.Time { return observedAt.Add(testStaleAfter + time.Nanosecond) }

		ep := newTestEndpoint()
		if err := e.Extract(context.Background(), pod, ep); err != nil {
			t.Fatalf("Extract: %v", err)
		}
		state := extractedState(t, ep)
		if state.DataStatus != attrnpu.DataStale {
			t.Fatalf("DataStatus = %q, want Stale", state.DataStatus)
		}
	})
}

func TestExtractClampsFutureObservationTime(t *testing.T) {
	e := New(testExtractorName, testAttributeKey, testStaleAfter)
	now := time.Unix(1_000_000, 0)
	e.nowFn = func() time.Time { return now }

	futurePod := npusrc.Pod{
		Matched:     true,
		ObservedAt:  now.Add(3 * time.Hour),
		DeviceCount: 1,
		Devices:     []npusrc.Device{{ID: "0"}},
	}
	ep := newTestEndpoint()
	if err := e.Extract(context.Background(), futurePod, ep); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	state := extractedState(t, ep)
	if !state.ObservationTime.Equal(now) {
		t.Fatalf("expected ObservationTime clamped to now (%v), got %v", now, state.ObservationTime)
	}
	if state.DataStatus != attrnpu.DataPresent {
		t.Fatalf("clamped state should be Present, got %s", state.DataStatus)
	}

	// Aging baseline should be the clamped value, not the raw future time.
	e.nowFn = func() time.Time { return now.Add(testStaleAfter + time.Second) }
	if err := e.Extract(context.Background(), npusrc.Pod{Matched: false}, ep); err != nil {
		t.Fatalf("age Extract: %v", err)
	}
	if got := extractedState(t, ep).DataStatus; got != attrnpu.DataStale {
		t.Fatalf("expected aged to Stale, got %s", got)
	}
}

func TestExtractAllocatesFreshStatePerCall(t *testing.T) {
	e := New(testExtractorName, testAttributeKey, testStaleAfter)
	e.nowFn = func() time.Time { return time.Unix(0, 0) }
	pod := npusrc.Pod{Matched: true, DeviceCount: 1, Devices: []npusrc.Device{{ID: "0"}}}

	ep1 := newTestEndpoint()
	ep2 := newTestEndpoint()

	if err := e.Extract(context.Background(), pod, ep1); err != nil {
		t.Fatalf("first Extract: %v", err)
	}
	if err := e.Extract(context.Background(), pod, ep2); err != nil {
		t.Fatalf("second Extract: %v", err)
	}
	s1 := extractedState(t, ep1)
	s2 := extractedState(t, ep2)
	if s1 == s2 {
		t.Fatalf("expected distinct *State pointers, got identical %p", s1)
	}
}

func TestConvertDevicesFiltersEmptyIDs(t *testing.T) {
	e := New(testExtractorName, testAttributeKey, testStaleAfter)
	e.nowFn = func() time.Time { return time.Unix(0, 0) }
	pod := npusrc.Pod{
		Matched:     true,
		DeviceCount: 3,
		Devices: []npusrc.Device{
			{ID: "0", AICoreUtilization: 10},
			{ID: "", AICoreUtilization: 99},
			{ID: "2", AICoreUtilization: 30},
		},
	}
	ep := newTestEndpoint()
	if err := e.Extract(context.Background(), pod, ep); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	state := extractedState(t, ep)
	if len(state.Devices) != 2 {
		t.Fatalf("expected 2 devices after filtering, got %d", len(state.Devices))
	}
	if state.DeviceCount != len(state.Devices) {
		t.Fatalf("DeviceCount = %d, want len(Devices) = %d", state.DeviceCount, len(state.Devices))
	}
	for _, device := range state.Devices {
		if device.DeviceID == "" {
			t.Fatalf("empty DeviceID should have been filtered: %+v", state.Devices)
		}
	}
}
