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

package lifecycle

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	requestcontrol "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencyattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	npuattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/npu"
	predictioninputattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/predictioninput"
	inflightproducer "hermes-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflight"
	internalinflight "hermes-router/pkg/epp/internal/inflight"
	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/predictiondata"
)

const (
	testPrimaryProfileName = "default"
	testDecodeProfileName  = "decode"
)

func TestPreRequestTracksSelectedPrimaryProfileEndpoint(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-pre-request")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	decode := newTestEndpoint("decode-a", "10.0.20.3")
	fallback := newTestEndpoint("decode-b", "10.0.20.4")
	decode.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(2, 4, 16))
	decode.Put(latencyattr.LatencyPredictionInfoKey, latencyattr.NewLatencyPredictionInfo(true, true, 10, 5, 20, 2))
	request := newTestRequest("req-1", []uint32{1, 2, 3, 4})
	tracker.now = func() time.Time { return time.Unix(100, 0) }

	tracker.PreRequest(context.Background(), request, newSchedulingResult(decode, fallback))

	assertLoad(t, store.Snapshot(decode.GetMetadata()), 4, 1)
	assertLoad(t, store.Snapshot(fallback.GetMetadata()), 0, 0)

	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active request metadata")
	}
	if active.MetaData.SelectedEndpoint != "pod://default/decode-a" {
		t.Fatalf("expected selected endpoint metadata, got %q", active.MetaData.SelectedEndpoint)
	}
	if active.MetaData.PrefixCacheScore != 0.5 {
		t.Fatalf("expected prefix cache score 0.5, got %v", active.MetaData.PrefixCacheScore)
	}
	if active.MetaData.Prediction.TTFTMillis != 20 || active.MetaData.Prediction.TPOTMillis != 2 {
		t.Fatalf("unexpected latency prediction metadata: %+v", active.MetaData.Prediction)
	}
}

func TestPreRequestTracksSelectedPrefillAndDecodePods(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-pd-pre-request")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	leader := newTestEndpoint("leader-pd", "10.0.22.1")
	prefill := newTestEndpoint("prefill-pd", "10.0.22.2")
	decode := newTestEndpoint("decode-pd", "10.0.22.3")
	setSelectedPDRoute(leader, prefill, decode)
	request := newTestRequest("req-pd-pre", []uint32{1, 2, 3, 4})
	tracker.now = func() time.Time { return time.Unix(120, 0) }

	tracker.PreRequest(context.Background(), request, newSchedulingResult(leader))

	assertLoad(t, store.Snapshot(leader.GetMetadata()), 0, 0)
	assertLoad(t, store.Snapshot(prefill.GetMetadata()), 4, 1)
	assertLoad(t, store.Snapshot(decode.GetMetadata()), 4, 1)

	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active request metadata")
	}
	if active.MetaData.SelectedEndpoint != "pod://default/leader-pd" {
		t.Fatalf("expected primary selected endpoint metadata to remain leader, got %q", active.MetaData.SelectedEndpoint)
	}
}

func TestDecodeRouteTracksAndReleasesOnlySelectedDecodePod(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-decode-only")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	leader := newTestEndpoint("leader-d", "10.0.24.1")
	prefill := newTestEndpoint("prefill-unused", "10.0.24.2")
	decode := newTestEndpoint("decode-d", "10.0.24.3")
	decode.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(3, 4, 16))
	decode.Put(latencyattr.LatencyPredictionInfoKey, latencyattr.NewLatencyPredictionInfo(true, true, 10, 5, 20, 2))
	setSelectedDecodeRoute(leader, decode)
	request := newTestRequest("req-decode-only", []uint32{1, 2, 3, 4})
	tracker.now = func() time.Time { return time.Unix(130, 0) }

	tracker.PreRequest(context.Background(), request, newSchedulingResultForProfile(testDecodeProfileName, leader))

	assertLoad(t, store.Snapshot(leader.GetMetadata()), 0, 0)
	assertLoad(t, store.Snapshot(prefill.GetMetadata()), 0, 0)
	assertLoad(t, store.Snapshot(decode.GetMetadata()), 4, 1)
	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active Decode-only request metadata")
	}
	if active.MetaData.PrefixCacheScore != 0.75 {
		t.Fatalf("expected Decode prefix cache score 0.75, got %v", active.MetaData.PrefixCacheScore)
	}
	if active.MetaData.SelectedRoute == nil || active.MetaData.SelectedRoute.Decode == nil {
		t.Fatalf("expected selected Decode route snapshot, got %+v", active.MetaData.SelectedRoute)
	}
	if active.MetaData.SelectedRoute.Prefill != nil {
		t.Fatalf("expected Decode-only snapshot without Prefill, got %+v", active.MetaData.SelectedRoute.Prefill)
	}

	tracker.ResponseBody(context.Background(), request, &requestcontrol.Response{EndOfStream: true}, decode.GetMetadata())
	assertLoad(t, store.Snapshot(decode.GetMetadata()), 0, 0)
}

func TestPreRequestCapturesAggregateSelectedRouteSnapshot(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-aggregate-selected-route")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	leader := newTestEndpoint("aggregate-route", "10.0.28.1")
	seedEndpointMetrics(leader, 5, 4, 67.5, time.Unix(95, 0))
	seedEndpointNPUState(leader, time.Unix(96, 0), "node-a", "0")
	store.StartRequest(internalinflight.RequestStart{
		RequestID:        "req-existing-aggregate",
		SelectedEndpoint: leader.GetMetadata(),
		InputTokenLength: 6,
		StartedAt:        time.Unix(90, 0),
	})
	request := newTestRequest("req-aggregate-selected-route", []uint32{1, 2, 3, 4})
	tracker.now = func() time.Time { return time.Unix(100, 0) }

	tracker.PreRequest(context.Background(), request, newSchedulingResult(leader))

	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active request metadata")
	}
	if active.MetaData.SelectedRoute == nil {
		t.Fatal("expected selected route to be stored")
	}
	if active.MetaData.SelectedRoute.Leader == nil {
		t.Fatal("expected leader snapshot to be stored")
	}
	if active.MetaData.SelectedRoute.Leader.ID != "pod://default/aggregate-route" {
		t.Fatalf("expected aggregate leader id, got %q", active.MetaData.SelectedRoute.Leader.ID)
	}
	if active.MetaData.SelectedRoute.Prefill != nil || active.MetaData.SelectedRoute.Decode != nil {
		t.Fatalf("expected aggregate route to omit prefill/decode snapshots, got %+v", active.MetaData.SelectedRoute)
	}
	if active.MetaData.SelectedRoute.Leader.Metrics == nil {
		t.Fatal("expected aggregate leader metrics snapshot")
	}
	if active.MetaData.SelectedRoute.Leader.Metrics.WaitingQueueSize != 5 {
		t.Fatalf("expected waiting queue size 5, got %d", active.MetaData.SelectedRoute.Leader.Metrics.WaitingQueueSize)
	}
	if active.MetaData.SelectedRoute.Leader.Metrics.KVCacheUsagePercent != 67.5 {
		t.Fatalf("expected kv cache usage 67.5, got %v", active.MetaData.SelectedRoute.Leader.Metrics.KVCacheUsagePercent)
	}
	if active.MetaData.SelectedRoute.Leader.Inflight == nil {
		t.Fatal("expected aggregate leader inflight snapshot")
	}
	if active.MetaData.SelectedRoute.Leader.Inflight.Tokens != 6 || active.MetaData.SelectedRoute.Leader.Inflight.Requests != 1 {
		t.Fatalf("expected pre-dispatch inflight snapshot, got %+v", active.MetaData.SelectedRoute.Leader.Inflight)
	}
	if active.MetaData.SelectedRoute.Leader.NPU == nil {
		t.Fatal("expected aggregate leader npu snapshot")
	}
	if active.MetaData.SelectedRoute.Leader.NPU.DeviceCount != 1 {
		t.Fatalf("expected one npu device, got %+v", active.MetaData.SelectedRoute.Leader.NPU)
	}
	if len(active.MetaData.SelectedRoute.Leader.NPU.Devices) != 1 || active.MetaData.SelectedRoute.Leader.NPU.Devices[0].DeviceID != "0" {
		t.Fatalf("expected npu device 0 in snapshot, got %+v", active.MetaData.SelectedRoute.Leader.NPU.Devices)
	}
}

func TestPreRequestCapturesPDRoleSnapshotsAndStoresLeaderAsIdentityOnly(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-pd-selected-route")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	leader := newTestEndpoint("leader-selected-route", "10.0.29.1")
	prefill := newTestEndpoint("prefill-selected-route", "10.0.29.2")
	decode := newTestEndpoint("decode-selected-route", "10.0.29.3")
	seedEndpointMetrics(leader, 9, 8, 77.7, time.Unix(95, 0))
	seedEndpointNPUState(leader, time.Unix(96, 0), "node-leader", "9")
	seedEndpointMetrics(prefill, 1, 2, 21.5, time.Unix(97, 0))
	seedEndpointNPUState(prefill, time.Unix(98, 0), "node-prefill", "1")
	seedEndpointMetrics(decode, 3, 4, 42.5, time.Unix(99, 0))
	seedEndpointNPUState(decode, time.Unix(100, 0), "node-decode", "2")
	setSelectedPDRoute(leader, prefill, decode)
	store.StartRequest(internalinflight.RequestStart{
		RequestID:        "req-existing-pd",
		SelectedEndpoint: leader.GetMetadata(),
		TrackedEndpoints: []*fwkdl.EndpointMetadata{prefill.GetMetadata(), decode.GetMetadata()},
		InputTokenLength: 5,
		StartedAt:        time.Unix(90, 0),
	})
	request := newTestRequest("req-pd-selected-route", []uint32{1, 2, 3, 4})
	tracker.now = func() time.Time { return time.Unix(110, 0) }

	tracker.PreRequest(context.Background(), request, newSchedulingResult(leader))

	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active request metadata")
	}
	if active.MetaData.SelectedRoute == nil {
		t.Fatal("expected selected route to be stored")
	}
	if active.MetaData.SelectedRoute.Leader == nil {
		t.Fatal("expected leader identity to be stored")
	}
	if active.MetaData.SelectedRoute.Leader.ID != "pod://default/leader-selected-route" {
		t.Fatalf("expected leader id, got %q", active.MetaData.SelectedRoute.Leader.ID)
	}
	if active.MetaData.SelectedRoute.Leader.Metrics != nil || active.MetaData.SelectedRoute.Leader.Inflight != nil || active.MetaData.SelectedRoute.Leader.NPU != nil {
		t.Fatalf("expected pd leader snapshot to be identity only, got %+v", active.MetaData.SelectedRoute.Leader)
	}
	if active.MetaData.SelectedRoute.Prefill == nil || active.MetaData.SelectedRoute.Decode == nil {
		t.Fatalf("expected pd route snapshots for prefill/decode, got %+v", active.MetaData.SelectedRoute)
	}
	if active.MetaData.SelectedRoute.Prefill.ID != "pod://default/prefill-selected-route" {
		t.Fatalf("expected prefill id, got %q", active.MetaData.SelectedRoute.Prefill.ID)
	}
	if active.MetaData.SelectedRoute.Decode.ID != "pod://default/decode-selected-route" {
		t.Fatalf("expected decode id, got %q", active.MetaData.SelectedRoute.Decode.ID)
	}
	if active.MetaData.SelectedRoute.Prefill.Metrics == nil || active.MetaData.SelectedRoute.Prefill.Metrics.WaitingQueueSize != 1 {
		t.Fatalf("expected prefill metrics snapshot, got %+v", active.MetaData.SelectedRoute.Prefill.Metrics)
	}
	if active.MetaData.SelectedRoute.Decode.Metrics == nil || active.MetaData.SelectedRoute.Decode.Metrics.RunningRequestsSize != 4 {
		t.Fatalf("expected decode metrics snapshot, got %+v", active.MetaData.SelectedRoute.Decode.Metrics)
	}
	if active.MetaData.SelectedRoute.Prefill.Inflight == nil || active.MetaData.SelectedRoute.Prefill.Inflight.Tokens != 5 || active.MetaData.SelectedRoute.Prefill.Inflight.Requests != 1 {
		t.Fatalf("expected prefill pre-dispatch inflight snapshot, got %+v", active.MetaData.SelectedRoute.Prefill.Inflight)
	}
	if active.MetaData.SelectedRoute.Decode.Inflight == nil || active.MetaData.SelectedRoute.Decode.Inflight.Tokens != 5 || active.MetaData.SelectedRoute.Decode.Inflight.Requests != 1 {
		t.Fatalf("expected decode pre-dispatch inflight snapshot, got %+v", active.MetaData.SelectedRoute.Decode.Inflight)
	}
	if active.MetaData.SelectedRoute.Prefill.NPU == nil || active.MetaData.SelectedRoute.Prefill.NPU.DeviceCount != 1 {
		t.Fatalf("expected prefill npu snapshot, got %+v", active.MetaData.SelectedRoute.Prefill.NPU)
	}
	if active.MetaData.SelectedRoute.Decode.NPU == nil || active.MetaData.SelectedRoute.Decode.NPU.DeviceCount != 1 {
		t.Fatalf("expected decode npu snapshot, got %+v", active.MetaData.SelectedRoute.Decode.NPU)
	}
}

func TestPreRequestPersistsAggregatePredictionInputAtDispatchTime(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-aggregate-prediction-input")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	leader := newTestEndpoint("aggregate-prediction-input", "10.0.30.1")
	dispatchInput := predictiondata.Input{InputID: 0, PodType: predictiondata.PodTypeAggregate, InputTokenLength: 4, ServerIP: "10.0.30.1"}
	leader.Put(predictioninputattr.PredictionInputKey, &predictioninputattr.InputSnapshot{Input: dispatchInput})
	request := newTestRequest("req-aggregate-prediction-input", []uint32{1, 2, 3, 4})
	now := time.Unix(100, 0)
	tracker.now = func() time.Time { return now }

	tracker.PreRequest(context.Background(), request, newSchedulingResult(leader))
	leader.Put(predictioninputattr.PredictionInputKey, &predictioninputattr.InputSnapshot{Input: predictiondata.Input{InputID: 99, PodType: predictiondata.PodTypeAggregate, InputTokenLength: 99, ServerIP: "drifted"}})
	now = time.Unix(101, 0)
	tracker.ResponseBody(context.Background(), request, &requestcontrol.Response{EndOfStream: true, Usage: requestcontrol.Usage{CompletionTokens: 2}}, leader.GetMetadata())

	completed := store.DrainCompleted(10)
	if len(completed) != 1 {
		t.Fatalf("expected one completed record, got %d", len(completed))
	}
	if completed[0].PredictionInputs[predictiondata.RoleLeader] != dispatchInput {
		t.Fatalf("expected dispatch-time prediction input, got %+v", completed[0].PredictionInputs)
	}
}

func TestPreRequestPersistsPDPredictionInputs(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-pd-prediction-input")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	leader := newTestEndpoint("leader-prediction-input", "10.0.31.1")
	prefill := newTestEndpoint("prefill-prediction-input", "10.0.31.2")
	decode := newTestEndpoint("decode-prediction-input", "10.0.31.3")
	prefillInput := predictiondata.Input{InputID: 1, PodType: predictiondata.PodTypePrefill, InputTokenLength: 4, ServerIP: "10.0.31.2"}
	decodeInput := predictiondata.Input{InputID: 2, PodType: predictiondata.PodTypeDecode, InputTokenLength: 4, ServerIP: "10.0.31.3"}
	prefill.Put(predictioninputattr.PredictionInputKey, &predictioninputattr.InputSnapshot{Input: prefillInput})
	decode.Put(predictioninputattr.PredictionInputKey, &predictioninputattr.InputSnapshot{Input: decodeInput})
	setSelectedPDRoute(leader, prefill, decode)
	request := newTestRequest("req-pd-prediction-input", []uint32{1, 2, 3, 4})
	tracker.now = func() time.Time { return time.Unix(100, 0) }

	tracker.PreRequest(context.Background(), request, newSchedulingResult(leader))

	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active request metadata")
	}
	if active.MetaData.PredictionInputs[predictiondata.RolePrefill] != prefillInput {
		t.Fatalf("expected prefill prediction input, got %+v", active.MetaData.PredictionInputs)
	}
	if active.MetaData.PredictionInputs[predictiondata.RoleDecode] != decodeInput {
		t.Fatalf("expected decode prediction input, got %+v", active.MetaData.PredictionInputs)
	}
}

func TestPreRequestPersistsSelectedMissingFeatureReasons(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-prediction-missing-reasons")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	leader := newTestEndpoint("leader-missing-reason", "10.0.32.1")
	leader.Put(predictioninputattr.MissingFeatureReasonKey, &predictioninputattr.MissingFeatureReason{Reason: "missing npu state"})
	request := newTestRequest("req-missing-reason", []uint32{1, 2})

	tracker.PreRequest(context.Background(), request, newSchedulingResult(leader))

	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active request metadata")
	}
	if active.MetaData.MissingFeatureReasons[predictiondata.RoleLeader] != "missing npu state" {
		t.Fatalf("expected leader missing feature reason, got %+v", active.MetaData.MissingFeatureReasons)
	}
}

func TestPreRequestRecordsPDAwareScoresAndPredictions(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-pd-scores")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	leader := newTestEndpoint("leader-score", "10.0.27.1")
	prefill := newTestEndpoint("prefill-score", "10.0.27.2")
	decode := newTestEndpoint("decode-score", "10.0.27.3")
	prefill.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(3, 4, 16))
	prefill.Put(latencyattr.LatencyPredictionInfoKey, latencyattr.NewLatencyPredictionInfo(true, true, 0, 0, 30, 0))
	decode.Put(latencyattr.LatencyPredictionInfoKey, latencyattr.NewLatencyPredictionInfo(true, true, 0, 0, 0, 5))
	setSelectedPDRoute(leader, prefill, decode)
	request := newTestRequest("req-pd-score", []uint32{1, 2, 3, 4})

	tracker.PreRequest(context.Background(), request, newSchedulingResult(leader))

	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active request metadata")
	}
	if active.MetaData.PrefixCacheScore != 0.75 {
		t.Fatalf("expected prefix cache score 0.75 from prefill, got %v", active.MetaData.PrefixCacheScore)
	}
	if active.MetaData.Prediction.TTFTMillis != 30 {
		t.Fatalf("expected TTFT 30ms from prefill, got %v", active.MetaData.Prediction.TTFTMillis)
	}
	if active.MetaData.Prediction.TPOTMillis != 5 {
		t.Fatalf("expected TPOT 5ms from decode, got %v", active.MetaData.Prediction.TPOTMillis)
	}
}

func TestPreRequestFailOpenWhenPDRouteLeaderIsNil(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-pd-nil-leader")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	request := newTestRequest("req-pd-nil-leader", []uint32{1, 2, 3, 4})

	tracker.PreRequest(context.Background(), request, newSchedulingResult(nil))

	if active := store.ActiveRequest(request.RequestId); active != nil {
		t.Fatalf("expected nil active request for nil-leader, got %+v", active)
	}
	if got := len(store.DrainCompleted(10)); got != 0 {
		t.Fatalf("expected no completed records for nil-leader, got %d", got)
	}
}

func TestPreRequestFailOpenWhenPDRouteMissingPrefill(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-pd-missing-prefill")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	leader := newTestEndpoint("leader-pd", "10.0.25.1")
	decode := newTestEndpoint("decode-pd", "10.0.25.3")
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:    "group-a",
		DecodePods: []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0}},
	})
	request := newTestRequest("req-pd-missing-prefill", []uint32{1, 2, 3, 4})

	tracker.PreRequest(context.Background(), request, newSchedulingResult(leader))

	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active request with incomplete Info to still be tracked against leader")
	}
	assertLoad(t, store.Snapshot(leader.GetMetadata()), 4, 1)
	assertLoad(t, store.Snapshot(decode.GetMetadata()), 0, 0)
}

func TestPreRequestFailOpenWhenPDRouteMissingDecode(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-pd-missing-decode")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	leader := newTestEndpoint("leader-pd", "10.0.26.1")
	prefill := newTestEndpoint("prefill-pd", "10.0.26.2")
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0}},
	})
	request := newTestRequest("req-pd-missing-decode", []uint32{1, 2, 3, 4})

	tracker.PreRequest(context.Background(), request, newSchedulingResult(leader))

	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active request with incomplete Info to still be tracked against leader")
	}
	assertLoad(t, store.Snapshot(leader.GetMetadata()), 4, 1)
	assertLoad(t, store.Snapshot(prefill.GetMetadata()), 0, 0)
}

func TestResponseBodyMarksFirstChunkAndReleasesTrackedStateOnEndOfStream(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-release")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	decode := newTestEndpoint("decode-c", "10.0.21.3")
	request := newTestRequest("req-2", []uint32{7, 8, 9})
	now := time.Unix(200, 0)
	tracker.now = func() time.Time { return now }

	tracker.PreRequest(context.Background(), request, newSchedulingResult(decode))
	now = time.Unix(201, 0)
	tracker.ResponseBody(context.Background(), request, &requestcontrol.Response{EndOfStream: false}, decode.GetMetadata())
	assertLoad(t, store.Snapshot(decode.GetMetadata()), 3, 1)

	now = time.Unix(206, 0)
	tracker.ResponseBody(context.Background(), request, &requestcontrol.Response{
		EndOfStream: true,
		Usage:       requestcontrol.Usage{CompletionTokens: 12},
	}, decode.GetMetadata())

	assertLoad(t, store.Snapshot(decode.GetMetadata()), 0, 0)

	completed := store.DrainCompleted(10)
	if len(completed) != 1 {
		t.Fatalf("expected one completed record, got %d", len(completed))
	}
	if completed[0].TTFTMillis != 1000 {
		t.Fatalf("expected TTFTMillis=1000, got %v", completed[0].TTFTMillis)
	}
	if completed[0].TotalLatencyMillis != 6000 {
		t.Fatalf("expected TotalLatencyMillis=6000, got %v", completed[0].TotalLatencyMillis)
	}
	if completed[0].SelectedEndpoint != "pod://default/decode-c" {
		t.Fatalf("expected completed selected endpoint to match response target, got %q", completed[0].SelectedEndpoint)
	}
	if completed[0].OutputLength != 12 {
		t.Fatalf("expected OutputLength=12 from response usage, got %d", completed[0].OutputLength)
	}
}

func TestResponseBodyReleasesSelectedPrefillAndDecodePodsOnEndOfStream(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-pd-release")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	leader := newTestEndpoint("leader-release", "10.0.23.1")
	prefill := newTestEndpoint("prefill-release", "10.0.23.2")
	decode := newTestEndpoint("decode-release", "10.0.23.3")
	setSelectedPDRoute(leader, prefill, decode)
	request := newTestRequest("req-pd-release", []uint32{7, 8, 9})
	now := time.Unix(220, 0)
	tracker.now = func() time.Time { return now }

	tracker.PreRequest(context.Background(), request, newSchedulingResult(leader))
	now = time.Unix(221, 0)
	tracker.ResponseBody(context.Background(), request, &requestcontrol.Response{EndOfStream: false}, decode.GetMetadata())
	assertLoad(t, store.Snapshot(prefill.GetMetadata()), 3, 1)
	assertLoad(t, store.Snapshot(decode.GetMetadata()), 3, 1)

	now = time.Unix(226, 0)
	tracker.ResponseBody(context.Background(), request, &requestcontrol.Response{
		EndOfStream: true,
		Usage:       requestcontrol.Usage{CompletionTokens: 12},
	}, decode.GetMetadata())

	assertLoad(t, store.Snapshot(leader.GetMetadata()), 0, 0)
	assertLoad(t, store.Snapshot(prefill.GetMetadata()), 0, 0)
	assertLoad(t, store.Snapshot(decode.GetMetadata()), 0, 0)

	completed := store.DrainCompleted(10)
	if len(completed) != 1 {
		t.Fatalf("expected one completed record, got %d", len(completed))
	}
	if completed[0].SelectedEndpoint != "pod://default/leader-release" {
		t.Fatalf("expected completed metadata to preserve primary selected endpoint, got %q", completed[0].SelectedEndpoint)
	}
	if completed[0].TTFTMillis != 1000 {
		t.Fatalf("expected TTFTMillis=1000, got %v", completed[0].TTFTMillis)
	}
	if completed[0].TotalLatencyMillis != 6000 {
		t.Fatalf("expected TotalLatencyMillis=6000, got %v", completed[0].TotalLatencyMillis)
	}
	if completed[0].OutputLength != 12 {
		t.Fatalf("expected OutputLength=12 from response usage, got %d", completed[0].OutputLength)
	}
}

func TestFactoryFlushesCompletedRecordsOnHandleContextDone(t *testing.T) {
	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "shutdown.jsonl")
	ctx, cancel := context.WithCancel(context.Background())
	handle := fwkplugin.NewEppHandle(ctx, func() []metav1types.NamespacedName { return nil })
	params, err := json.Marshal(map[string]any{
		"storeName": "lifecycle-shutdown",
		"persistence": map[string]any{
			"enabled":        true,
			"flushThreshold": 10,
			"outputPath":     outputPath,
		},
	})
	if err != nil {
		t.Fatalf("expected test parameters to marshal: %v", err)
	}

	plugin, err := Factory("tracker", json.RawMessage(params), handle)
	if err != nil {
		t.Fatalf("expected factory to succeed, got %v", err)
	}
	tracker, ok := plugin.(*RequestLifecycleTracker)
	if !ok {
		t.Fatalf("expected lifecycle tracker plugin, got %T", plugin)
	}
	store := internalinflight.StoreForName("lifecycle-shutdown")
	t.Cleanup(func() { _ = store.Close() })
	decode := newTestEndpoint("decode-shutdown", "10.0.21.9")
	request := newTestRequest("req-shutdown", []uint32{1, 2, 3})
	now := time.Unix(300, 0)
	tracker.now = func() time.Time { return now }

	tracker.PreRequest(context.Background(), request, newSchedulingResult(decode))
	now = time.Unix(301, 0)
	tracker.ResponseBody(context.Background(), request, &requestcontrol.Response{EndOfStream: true}, decode.GetMetadata())

	cancel()
	waitForCondition(t, func() bool {
		content, readErr := os.ReadFile(outputPath)
		return readErr == nil && len(splitNonEmptyLines(string(content))) == 1
	}, "expected shutdown to flush completed records to disk")
	var reloaded *internalinflight.Store
	waitForCondition(t, func() bool {
		reloaded = internalinflight.StoreForName("lifecycle-shutdown")
		return reloaded != store
	}, "expected shutdown close to remove the shared store from registry")
	t.Cleanup(func() { _ = reloaded.Close() })
}

func TestParseParametersAcceptsFlushIntervalDuration(t *testing.T) {
	raw, err := json.Marshal(map[string]any{
		"storeName": "lifecycle-flush-interval",
		"persistence": map[string]any{
			"enabled":        true,
			"flushThreshold": 10,
			"flushInterval":  "1m",
			"outputPath":     "/tmp/lifecycle-flush-interval.jsonl",
		},
	})
	if err != nil {
		t.Fatalf("expected test parameters to marshal: %v", err)
	}

	params, err := parseParameters(json.RawMessage(raw))
	if err != nil {
		t.Fatalf("expected parameters to parse, got %v", err)
	}
	if params.Persistence.FlushInterval != time.Minute {
		t.Fatalf("expected flush interval 1m, got %v", params.Persistence.FlushInterval)
	}
}

func TestParseParametersRejectsInvalidFlushInterval(t *testing.T) {
	raw, err := json.Marshal(map[string]any{
		"persistence": map[string]any{
			"enabled":       true,
			"flushInterval": "later",
			"outputPath":    "/tmp/lifecycle-flush-interval.jsonl",
		},
	})
	if err != nil {
		t.Fatalf("expected test parameters to marshal: %v", err)
	}

	_, err = parseParameters(json.RawMessage(raw))
	if err == nil {
		t.Fatal("expected invalid flush interval to be rejected")
	}
	if !strings.Contains(err.Error(), "persistence.flushInterval") {
		t.Fatalf("expected flush interval validation error, got %v", err)
	}
}

func TestPreRequestFailOpenOnNilOrIncompleteInputs(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-prerequest-fail-open")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	ctx := context.Background()

	tracker.PreRequest(ctx, nil, nil)
	tracker.PreRequest(ctx, &fwkscheduling.LLMRequest{}, nil)
	tracker.PreRequest(ctx, newTestRequest("req-missing-result", []uint32{1}), nil)
	tracker.PreRequest(ctx, newTestRequest("req-no-target", []uint32{1}), newSchedulingResult())
	tracker.PreRequest(ctx, newTestRequest("req-nil-target", []uint32{1}), newSchedulingResult(nil))

	if got := len(store.DrainCompleted(10)); got != 0 {
		t.Fatalf("expected no completed records for fail-open prerequest paths, got %d", got)
	}
	if active := store.ActiveRequest("req-missing-result"); active != nil {
		t.Fatalf("expected nil active request for fail-open path, got %+v", active)
	}
	if active := store.ActiveRequest("req-no-target"); active != nil {
		t.Fatalf("expected nil active request for missing target path, got %+v", active)
	}
	if active := store.ActiveRequest("req-nil-target"); active != nil {
		t.Fatalf("expected nil active request for nil target path, got %+v", active)
	}
	if got := store.Snapshot(newTestEndpoint("unused", "10.0.99.1").GetMetadata()); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected fail-open prerequest paths to leave store empty, got %+v", got)
	}
}

func TestResponseBodyFailOpenOnNilOrIncompleteInputs(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-response-fail-open")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	endpoint := newTestEndpoint("decode-fail-open", "10.0.21.10")

	tracker.ResponseBody(context.Background(), nil, &requestcontrol.Response{EndOfStream: true}, endpoint.GetMetadata())
	tracker.ResponseBody(context.Background(), &fwkscheduling.LLMRequest{}, &requestcontrol.Response{EndOfStream: true}, endpoint.GetMetadata())
	tracker.ResponseBody(context.Background(), newTestRequest("req-nil-response", []uint32{1}), nil, endpoint.GetMetadata())

	if got := store.Snapshot(endpoint.GetMetadata()); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected fail-open response paths to leave endpoint load unchanged, got %+v", got)
	}
	if completed := store.DrainCompleted(10); len(completed) != 0 {
		t.Fatalf("expected fail-open response paths to avoid completion records, got %d", len(completed))
	}
}

func TestInflightProducerAndLifecycleShareLoadLifecycle(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-integration")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	producer := inflightproducer.NewLoadProducer("producer", store, internalinflight.DefaultAttributeKey)
	endpoint := newTestEndpoint("decode-integration", "10.0.21.11")
	request := newTestRequest("req-integration", []uint32{1, 2, 3, 4})
	now := time.Unix(500, 0)
	tracker.now = func() time.Time { return now }

	if err := producer.PrepareRequestData(context.Background(), request, []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("expected initial prepare to succeed, got %v", err)
	}
	assertEndpointAttributeLoad(t, endpoint, 0, 0)

	tracker.PreRequest(context.Background(), request, newSchedulingResult(endpoint))
	if err := producer.PrepareRequestData(context.Background(), request, []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("expected prepare after prerequest to succeed, got %v", err)
	}
	assertEndpointAttributeLoad(t, endpoint, 4, 1)

	now = time.Unix(501, 0)
	tracker.ResponseBody(context.Background(), request, &requestcontrol.Response{EndOfStream: true}, endpoint.GetMetadata())
	if err := producer.PrepareRequestData(context.Background(), request, []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("expected prepare after completion to succeed, got %v", err)
	}
	assertEndpointAttributeLoad(t, endpoint, 0, 0)
}

func TestPreRequestRecordsActiveModeWhenOnlyActiveKeyPresent(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-mode-active-only")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	endpoint := newTestEndpoint("only-active", "10.0.40.1")
	endpoint.Put(latencyattr.LatencyPredictionInfoKey, latencyattr.NewLatencyPredictionInfo(true, true, 0, 0, 25, 3))
	request := newTestRequest("req-active-only", []uint32{1, 2, 3, 4})

	tracker.PreRequest(context.Background(), request, newSchedulingResult(endpoint))

	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active request metadata")
	}
	if active.MetaData.PredictionMode != internalinflight.PredictionModeActive {
		t.Fatalf("expected mode %q, got %q", internalinflight.PredictionModeActive, active.MetaData.PredictionMode)
	}
	if active.MetaData.Prediction.TTFTMillis != 25 || active.MetaData.Prediction.TPOTMillis != 3 {
		t.Fatalf("unexpected active prediction: %+v", active.MetaData.Prediction)
	}
}

func TestPreRequestRecordsEmptyModeWhenNeitherKeyPresent(t *testing.T) {
	store := internalinflight.StoreForName("lifecycle-mode-off")
	t.Cleanup(func() { _ = store.Close() })
	tracker := NewRequestLifecycleTracker("tracker", store)
	endpoint := newTestEndpoint("no-prediction", "10.0.40.4")
	request := newTestRequest("req-off", []uint32{1, 2, 3, 4})

	tracker.PreRequest(context.Background(), request, newSchedulingResult(endpoint))

	active := store.ActiveRequest(request.RequestId)
	if active == nil {
		t.Fatal("expected active request metadata")
	}
	if active.MetaData.PredictionMode != "" {
		t.Fatalf("expected empty mode for data-collection-only path, got %q", active.MetaData.PredictionMode)
	}
	if active.MetaData.Prediction != (internalinflight.LatencyPrediction{}) {
		t.Fatalf("expected zero active prediction, got %+v", active.MetaData.Prediction)
	}
}

func newTestRequest(requestID string, tokenIDs []uint32) *fwkscheduling.LLMRequest {
	return &fwkscheduling.LLMRequest{
		RequestId: requestID,
		Body: &fwkscheduling.LLMRequestBody{
			Completions: &fwkscheduling.CompletionsRequest{Prompt: fwkscheduling.Prompt{Raw: "hello world"}},
		},
		TokenizedPrompt: &fwkscheduling.TokenizedPrompt{TokenIDs: tokenIDs},
	}
}

func newSchedulingResult(targets ...fwkscheduling.Endpoint) *fwkscheduling.SchedulingResult {
	return newSchedulingResultForProfile(testPrimaryProfileName, targets...)
}

func newSchedulingResultForProfile(
	profileName string,
	targets ...fwkscheduling.Endpoint,
) *fwkscheduling.SchedulingResult {
	return &fwkscheduling.SchedulingResult{
		PrimaryProfileName: profileName,
		ProfileResults: map[string]*fwkscheduling.ProfileRunResult{
			profileName: {TargetEndpoints: targets},
		},
	}
}

func newTestEndpoint(name, address string) fwkscheduling.Endpoint {
	return fwkscheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
			PodName:        name,
			Address:        address,
			Port:           "8000",
		},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
}

func setSelectedPDRoute(leader, prefill, decode fwkscheduling.Endpoint) {
	prefillSelection := fwkscheduling.ScoredEndpoint{Endpoint: prefill, Score: 0}
	decodeSelection := fwkscheduling.ScoredEndpoint{Endpoint: decode, Score: 0}
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:            "test-group",
		PrefillPods:        []fwkscheduling.ScoredEndpoint{prefillSelection},
		DecodePods:         []fwkscheduling.ScoredEndpoint{decodeSelection},
		SelectedPrefillPod: &prefillSelection,
		SelectedDecodePod:  &decodeSelection,
	})
}

func setSelectedDecodeRoute(leader, decode fwkscheduling.Endpoint) {
	decodeSelection := fwkscheduling.ScoredEndpoint{Endpoint: decode, Score: 0}
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:           "test-group",
		DecodePods:        []fwkscheduling.ScoredEndpoint{decodeSelection},
		SelectedDecodePod: &decodeSelection,
	})
}

func seedEndpointMetrics(endpoint fwkscheduling.Endpoint, waiting, running int, kvUsagePercent float64, updateTime time.Time) {
	metrics := endpoint.GetMetrics()
	metrics.ActiveModels = map[string]int{"Qwen/Qwen3-8B": 1}
	metrics.WaitingModels = map[string]int{"Qwen/Qwen3-8B": waiting}
	metrics.MaxActiveModels = 2
	metrics.RunningRequestsSize = running
	metrics.WaitingQueueSize = waiting
	metrics.KVCacheUsagePercent = kvUsagePercent
	metrics.KvCacheMaxTokenCapacity = 131072
	metrics.CacheBlockSize = 16
	metrics.CacheNumBlocks = 8192
	metrics.UpdateTime = updateTime
}

func seedEndpointNPUState(endpoint fwkscheduling.Endpoint, observedAt time.Time, target, deviceID string) {
	endpoint.Put(npuattr.DefaultStateAttributeKey, &npuattr.State{
		ObservationTime:            observedAt,
		DataStatus:                 npuattr.DataPresent,
		DeviceCount:                1,
		AICoreUtilizationAvg:       40,
		AICoreUtilizationMax:       60,
		HBMUtilizationAvg:          50,
		HBMUtilizationMax:          70,
		HBMBandwidthUtilizationAvg: 30,
		HBMBandwidthUtilizationMax: 45,
		Devices: []npuattr.DeviceState{{
			Target:                  target,
			DeviceID:                deviceID,
			AICoreUtilization:       60,
			HBMUtilization:          70,
			HBMBandwidthUtilization: 45,
		}},
	})
}

func assertLoad(t *testing.T, load internalinflight.Load, wantTokens, wantRequests int) {
	t.Helper()
	if load.Tokens != wantTokens || load.Requests != wantRequests {
		t.Fatalf("unexpected inflight load: got %+v, want tokens=%d requests=%d", load, wantTokens, wantRequests)
	}
}

func assertEndpointAttributeLoad(t *testing.T, endpoint fwkscheduling.Endpoint, wantTokens, wantRequests int) {
	t.Helper()
	value, ok := endpoint.Get(internalinflight.DefaultAttributeKey)
	if !ok {
		t.Fatalf("expected attribute %q to be present", internalinflight.DefaultAttributeKey)
	}
	load, ok := value.(*internalinflight.Load)
	if !ok || load == nil {
		t.Fatalf("expected *inflight.Load attribute, got %T", value)
	}
	assertLoad(t, *load, wantTokens, wantRequests)
}

func waitForCondition(t *testing.T, condition func() bool, failureMessage string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal(failureMessage)
}

func splitNonEmptyLines(content string) []string {
	lines := make([]string, 0)
	start := 0
	for index := 0; index < len(content); index++ {
		if content[index] != '\n' {
			continue
		}
		if start != index {
			lines = append(lines, content[start:index])
		}
		start = index + 1
	}
	if start < len(content) {
		lines = append(lines, content[start:])
	}
	return lines
}
