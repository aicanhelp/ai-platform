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

package inflight

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"

	"hermes-router/pkg/epp/internal/predictiondata"
)

func TestStartRequestStoresSelectedEndpointMetadata(t *testing.T) {
	store := newStore()
	selected := newTestEndpointMetadata("decode-a", "10.0.0.8")

	store.StartRequest(RequestStart{
		RequestID:        "req-1",
		Model:            "demo",
		InputTokenLength: 8,
		SelectedEndpoint: selected,
		DispatchInflight: Load{Tokens: 4, Requests: 1},
		PrefixCacheScore: 0.75,
		Prediction: LatencyPrediction{
			TTFTMillis:         20,
			TotalLatencyMillis: 90,
		},
		StartedAt: time.Unix(100, 0),
	})

	active := store.ActiveRequest("req-1")
	if active == nil {
		t.Fatal("expected active request to exist")
	}
	if active.MetaData.SelectedEndpoint != "pod://default/decode-a" {
		t.Fatalf("expected selected endpoint to be stored, got %q", active.MetaData.SelectedEndpoint)
	}
	if active.MetaData.PrefixCacheScore != 0.75 {
		t.Fatalf("expected prefix cache score to be stored, got %v", active.MetaData.PrefixCacheScore)
	}
	if got := store.Snapshot(selected); got.Tokens != 8 || got.Requests != 1 {
		t.Fatalf("expected selected endpoint load to increase, got %+v", got)
	}
}

func TestStartRequestTracksMultipleEndpointKeys(t *testing.T) {
	store := newStore()
	leader := newTestEndpointMetadata("leader-a", "10.0.0.30")
	prefill := newTestEndpointMetadata("prefill-a", "10.0.0.31")
	decode := newTestEndpointMetadata("decode-a", "10.0.0.32")

	store.StartRequest(RequestStart{
		RequestID:        "req-multi",
		SelectedEndpoint: leader,
		TrackedEndpoints: []*fwkdl.EndpointMetadata{prefill, decode},
		InputTokenLength: 4,
		StartedAt:        time.Unix(100, 0),
	})

	active := store.ActiveRequest("req-multi")
	if active == nil {
		t.Fatal("expected active request to exist")
	}
	if active.MetaData.SelectedEndpoint != "pod://default/leader-a" {
		t.Fatalf("expected selected endpoint metadata to remain leader, got %q", active.MetaData.SelectedEndpoint)
	}
	if got := store.Snapshot(leader); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected leader endpoint to remain untracked for load accounting, got %+v", got)
	}
	if got := store.Snapshot(prefill); got.Tokens != 4 || got.Requests != 1 {
		t.Fatalf("expected prefill endpoint load to increase, got %+v", got)
	}
	if got := store.Snapshot(decode); got.Tokens != 4 || got.Requests != 1 {
		t.Fatalf("expected decode endpoint load to increase, got %+v", got)
	}

	state, ok := store.requests["req-multi"]
	if !ok {
		t.Fatal("expected internal request state to exist")
	}
	if len(state.endpointKeys) != 2 {
		t.Fatalf("expected two tracked endpoint keys, got %d", len(state.endpointKeys))
	}
}

func TestCompleteRequestReleasesAllTrackedEndpoints(t *testing.T) {
	store := newStore()
	leader := newTestEndpointMetadata("leader-release", "10.0.0.33")
	prefill := newTestEndpointMetadata("prefill-release", "10.0.0.34")
	decode := newTestEndpointMetadata("decode-release", "10.0.0.35")

	store.StartRequest(RequestStart{
		RequestID:        "req-release",
		SelectedEndpoint: leader,
		TrackedEndpoints: []*fwkdl.EndpointMetadata{prefill, decode},
		InputTokenLength: 6,
		StartedAt:        time.Unix(200, 0),
	})
	store.CompleteRequest(RequestCompletion{
		RequestID:        "req-release",
		CompletedAt:      time.Unix(205, 0),
		SelectedEndpoint: decode,
	})

	if got := store.Snapshot(prefill); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected prefill load to be released, got %+v", got)
	}
	if got := store.Snapshot(decode); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected decode load to be released, got %+v", got)
	}
	completed := store.DrainCompleted(10)
	if len(completed) != 1 {
		t.Fatalf("expected one completed record, got %d", len(completed))
	}
	if completed[0].SelectedEndpoint != "pod://default/leader-release" {
		t.Fatalf("expected completed metadata to preserve the primary selected endpoint, got %q", completed[0].SelectedEndpoint)
	}
}

func TestStartRequestDeduplicatesTrackedEndpointKeys(t *testing.T) {
	store := newStore()
	leader := newTestEndpointMetadata("leader-dedupe", "10.0.0.36")
	prefill := newTestEndpointMetadata("shared-dedupe", "10.0.0.37")
	decode := newTestEndpointMetadata("decode-dedupe", "10.0.0.38")

	store.StartRequest(RequestStart{
		RequestID:        "req-dedupe",
		SelectedEndpoint: leader,
		TrackedEndpoints: []*fwkdl.EndpointMetadata{prefill, prefill, decode, prefill},
		InputTokenLength: 5,
		StartedAt:        time.Unix(300, 0),
	})

	if got := store.Snapshot(prefill); got.Tokens != 5 || got.Requests != 1 {
		t.Fatalf("expected duplicate tracked endpoints to count once, got %+v", got)
	}
	if got := store.Snapshot(decode); got.Tokens != 5 || got.Requests != 1 {
		t.Fatalf("expected decode endpoint to count once, got %+v", got)
	}
	state, ok := store.requests["req-dedupe"]
	if !ok {
		t.Fatal("expected internal request state to exist")
	}
	if len(state.endpointKeys) != 2 {
		t.Fatalf("expected tracked endpoint keys to be deduplicated, got %d", len(state.endpointKeys))
	}
}

func TestStartRequestFailsOpenWhenAnyTrackedEndpointIsInvalid(t *testing.T) {
	store := newStore()
	selected := newTestEndpointMetadata("leader-invalid-tracked", "10.0.0.42")
	validTracked := newTestEndpointMetadata("decode-valid-tracked", "10.0.0.43")
	invalidTracked := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: "invalid-tracked"},
		Address:        "10.0.0.44",
	}

	store.StartRequest(RequestStart{
		RequestID:        "req-invalid-tracked",
		SelectedEndpoint: selected,
		TrackedEndpoints: []*fwkdl.EndpointMetadata{validTracked, invalidTracked},
		InputTokenLength: 7,
		StartedAt:        time.Unix(325, 0),
	})

	if active := store.ActiveRequest("req-invalid-tracked"); active != nil {
		t.Fatalf("expected invalid tracked endpoint to fail open, got %+v", active)
	}
	if len(store.requests) != 0 {
		t.Fatalf("expected no tracked requests, got %d", len(store.requests))
	}
	if got := store.Snapshot(selected); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected selected endpoint load to remain zero, got %+v", got)
	}
	if got := store.Snapshot(validTracked); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected valid tracked endpoint load to remain zero, got %+v", got)
	}
}

func TestStartRequestFailsOpenWhenSelectedEndpointMissingOrInvalid(t *testing.T) {
	tests := []struct {
		name             string
		selectedEndpoint *fwkdl.EndpointMetadata
	}{
		{
			name:             "missing selected endpoint",
			selectedEndpoint: nil,
		},
		{
			name: "invalid selected endpoint",
			selectedEndpoint: &fwkdl.EndpointMetadata{
				NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: "invalid-selected"},
				Address:        "10.0.0.39",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := newStore()
			trackedA := newTestEndpointMetadata("tracked-a", "10.0.0.40")
			trackedB := newTestEndpointMetadata("tracked-b", "10.0.0.41")

			store.StartRequest(RequestStart{
				RequestID:        "req-incomplete",
				SelectedEndpoint: tt.selectedEndpoint,
				TrackedEndpoints: []*fwkdl.EndpointMetadata{trackedA, trackedB},
				InputTokenLength: 9,
				StartedAt:        time.Unix(350, 0),
			})

			if active := store.ActiveRequest("req-incomplete"); active != nil {
				t.Fatalf("expected incomplete request start to be ignored, got %+v", active)
			}
			if len(store.requests) != 0 {
				t.Fatalf("expected no tracked requests, got %d", len(store.requests))
			}
			if got := store.Snapshot(trackedA); got.Tokens != 0 || got.Requests != 0 {
				t.Fatalf("expected tracked endpoint A load to remain zero, got %+v", got)
			}
			if got := store.Snapshot(trackedB); got.Tokens != 0 || got.Requests != 0 {
				t.Fatalf("expected tracked endpoint B load to remain zero, got %+v", got)
			}
		})
	}
}

func TestFlushWritesCompletedRecordsBelowThreshold(t *testing.T) {
	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "flush.jsonl")
	store := newStoreWithPersistence(PersistenceConfig{
		Enabled:        true,
		FlushThreshold: 10,
		OutputPath:     outputPath,
	})

	store.StartRequest(RequestStart{RequestID: "req-flush", SelectedEndpoint: newTestEndpointMetadata("decode-flush", "10.0.0.10"), StartedAt: time.Unix(10, 0)})
	store.CompleteRequest(RequestCompletion{RequestID: "req-flush", CompletedAt: time.Unix(11, 0)})

	if err := store.Flush(); err != nil {
		t.Fatalf("expected flush to succeed, got %v", err)
	}
	if drained := store.DrainCompleted(10); len(drained) != 0 {
		t.Fatalf("expected flush to clear in-memory completed entries, got %d", len(drained))
	}
	content, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("expected flush file to exist, got %v", err)
	}
	if len(splitNonEmptyLines(string(content))) != 1 {
		t.Fatalf("expected one flushed record, got %q", string(content))
	}
}

func TestFlushWritesSelectedRouteWithoutDuplicateTopLevelFields(t *testing.T) {
	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "selected-route.jsonl")
	store := newStoreWithPersistence(PersistenceConfig{
		Enabled:        true,
		FlushThreshold: 1,
		OutputPath:     outputPath,
	})

	store.StartRequest(RequestStart{
		RequestID:        "req-selected-route-flush",
		Model:            "demo",
		InputTokenLength: 9,
		SelectedEndpoint: newTestEndpointMetadata("decode-selected-route", "10.0.0.45"),
		DispatchInflight: Load{Tokens: 7, Requests: 2},
		SelectedRoute: &SelectedRoute{
			Leader: &RouteEndpointSnapshot{
				ID:       "pod://default/decode-selected-route",
				Metrics:  &EndpointMetricsSnapshot{WaitingQueueSize: 3, RunningRequestsSize: 4},
				Inflight: &Load{Tokens: 7, Requests: 2},
				NPU: &NPUStateSnapshot{
					ObservationTime: time.Unix(8, 0),
					DataStatus:      "Present",
					DeviceCount:     1,
					Devices: []NPUDeviceSnapshot{{
						Target:   "node-a",
						DeviceID: "0",
					}},
				},
			},
		},
		StartedAt: time.Unix(10, 0),
	})
	store.CompleteRequest(RequestCompletion{RequestID: "req-selected-route-flush", CompletedAt: time.Unix(11, 0)})

	lines := waitForOutputLines(t, outputPath, 1)
	if len(lines) != 1 {
		t.Fatalf("expected one flushed record, got %d", len(lines))
	}
	line := lines[0]
	if strings.Contains(line, "\"selectedEndpoint\"") {
		t.Fatalf("expected selectedEndpoint to be omitted from json, got %s", line)
	}
	if strings.Contains(line, "\"dispatchInflight\"") {
		t.Fatalf("expected dispatchInflight to be omitted from json, got %s", line)
	}

	var got RequestMetaData
	if err := json.Unmarshal([]byte(line), &got); err != nil {
		t.Fatalf("expected valid request metadata json, got %v", err)
	}
	if got.SelectedRoute == nil || got.SelectedRoute.Leader == nil {
		t.Fatalf("expected selected route leader to be present, got %+v", got.SelectedRoute)
	}
	if got.SelectedRoute.Leader.ID != "pod://default/decode-selected-route" {
		t.Fatalf("expected selected route leader id, got %+v", got.SelectedRoute.Leader)
	}
	if got.SelectedRoute.Leader.Inflight == nil || got.SelectedRoute.Leader.Inflight.Tokens != 7 || got.SelectedRoute.Leader.Inflight.Requests != 2 {
		t.Fatalf("expected selected route inflight snapshot, got %+v", got.SelectedRoute.Leader.Inflight)
	}
}

func TestCompletedRecordIncludesPredictionInputsAndSchemaFields(t *testing.T) {
	store := newStore()
	selected := newTestEndpointMetadata("decode-prediction-input", "10.0.0.46")
	input := predictiondata.Input{
		InputID:          0,
		PodType:          predictiondata.PodTypeAggregate,
		InputTokenLength: 8,
		ServerIP:         "10.0.0.46",
	}

	store.StartRequest(RequestStart{
		RequestID:        "req-prediction-input",
		Model:            "test-model",
		SelectedEndpoint: selected,
		InputTokenLength: 8,
		PredictionInputs: map[string]predictiondata.Input{
			predictiondata.RoleLeader: input,
		},
		MissingFeatureReasons: map[string]string{
			predictiondata.RoleDecode: "missing npu state",
		},
		StartedAt: time.Unix(10, 0),
	})
	completedAt := time.Unix(12, 0).UTC()
	store.CompleteRequest(RequestCompletion{RequestID: "req-prediction-input", CompletedAt: completedAt, OutputLength: 4})

	completed := store.DrainCompleted(10)
	if len(completed) != 1 {
		t.Fatalf("expected one completed record, got %d", len(completed))
	}
	got := completed[0]
	if got.SchemaVersion != CompletedRecordSchemaV1 {
		t.Fatalf("expected schemaVersion %d, got %d", CompletedRecordSchemaV1, got.SchemaVersion)
	}
	if !got.CompletedAt.Equal(completedAt) {
		t.Fatalf("expected completedAt %v, got %v", completedAt, got.CompletedAt)
	}
	if got.PredictionInputs[predictiondata.RoleLeader] != input {
		t.Fatalf("expected prediction input to be copied, got %+v", got.PredictionInputs)
	}
	if got.MissingFeatureReasons[predictiondata.RoleDecode] != "missing npu state" {
		t.Fatalf("expected missing feature reason, got %+v", got.MissingFeatureReasons)
	}

	raw, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal completed record: %v", err)
	}
	line := string(raw)
	for _, want := range []string{`"schemaVersion":1`, `"completedAt":`, `"predictionInputs":`, `"prefixCacheScore":0`, `"numRequestWaiting":0`, `"missingFeatureReasons":`} {
		if !strings.Contains(line, want) {
			t.Fatalf("expected JSON to contain %s, got %s", want, line)
		}
	}
}

func TestCompletedBufferIsBoundedWithoutPersistence(t *testing.T) {
	store := newStore()
	for index := 0; index < defaultCompletedBufferLimit+5; index++ {
		requestID := "req-buffer-" + time.Unix(int64(index), 0).UTC().Format("150405")
		store.StartRequest(RequestStart{RequestID: requestID, SelectedEndpoint: newTestEndpointMetadata("decode-buffer", "10.0.0.11"), StartedAt: time.Unix(int64(index), 0)})
		store.CompleteRequest(RequestCompletion{RequestID: requestID, CompletedAt: time.Unix(int64(index+1), 0)})
	}

	drained := store.DrainCompleted(defaultCompletedBufferLimit + 10)
	if len(drained) != defaultCompletedBufferLimit {
		t.Fatalf("expected completed buffer to be capped at %d, got %d", defaultCompletedBufferLimit, len(drained))
	}
	if store.DroppedCompletedCount() != 5 {
		t.Fatalf("expected dropped completed count to be 5, got %d", store.DroppedCompletedCount())
	}
}

func TestFailedAutoFlushRetainsBoundedRecordsAndSurfacesError(t *testing.T) {
	outputDir := t.TempDir()
	blockedParent := filepath.Join(outputDir, "blocked")
	if err := os.WriteFile(blockedParent, []byte("x"), 0o644); err != nil {
		t.Fatalf("expected to create blocking file, got %v", err)
	}
	store := newStoreWithPersistence(PersistenceConfig{
		Enabled:        true,
		FlushThreshold: 1,
		OutputPath:     filepath.Join(blockedParent, "requests.jsonl"),
	})

	store.StartRequest(RequestStart{RequestID: "req-fail", SelectedEndpoint: newTestEndpointMetadata("decode-fail", "10.0.0.15"), StartedAt: time.Unix(20, 0)})
	store.CompleteRequest(RequestCompletion{RequestID: "req-fail", CompletedAt: time.Unix(21, 0)})

	waitForCondition(t, func() bool {
		return store.LastFlushError() != nil
	}, "auto flush failure to be recorded")
	if store.LastFlushError() == nil {
		t.Fatal("expected failed auto flush to expose the last flush error")
	}
	if store.FlushFailureCount() == 0 {
		t.Fatal("expected failed auto flush to increment failure count")
	}
	drained := store.DrainCompleted(10)
	if len(drained) != 1 {
		t.Fatalf("expected failed auto flush to retain completed records, got %d", len(drained))
	}
}

func TestFlushRetainsOnlyUnwrittenRecordsAfterPartialWriteFailure(t *testing.T) {
	store := newStoreWithPersistence(PersistenceConfig{Enabled: true, FlushThreshold: 1, OutputPath: filepath.Join(t.TempDir(), "requests.jsonl")})
	store.mu.Lock()
	store.flushWriter = failingBatchWriter{retryFrom: 2, err: errors.New("write failed")}
	store.completed = []RequestMetaData{
		{RequestID: "req-1"},
		{RequestID: "req-2"},
		{RequestID: "req-3"},
	}
	store.mu.Unlock()

	err := store.Flush()
	if err == nil {
		t.Fatal("expected partial flush to fail")
	}
	drained := store.DrainCompleted(10)
	if len(drained) != 1 {
		t.Fatalf("expected only unwritten records to remain, got %d", len(drained))
	}
	if drained[0].RequestID != "req-3" {
		t.Fatalf("expected req-3 to remain buffered, got %+v", drained)
	}
	if store.LastFlushError() == nil {
		t.Fatal("expected partial flush failure to be observable")
	}
	if store.FlushFailureCount() == 0 {
		t.Fatal("expected partial flush failure to be counted")
	}
}

func TestApplyEndpointEventDeleteRemovesIdleEndpoint(t *testing.T) {
	store := newStore()
	endpoint := newTestSchedulingEndpoint("decode-event", "10.0.0.12")
	metadata := endpoint.GetMetadata()

	store.ApplyEndpointEvent(fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint}, time.Unix(1, 0))
	store.StartRequest(RequestStart{RequestID: "req-event", SelectedEndpoint: metadata, StartedAt: time.Unix(2, 0)})
	store.CompleteRequest(RequestCompletion{RequestID: "req-event", CompletedAt: time.Unix(3, 0)})
	if len(store.endpoints) != 1 {
		t.Fatalf("expected active endpoint state to remain until delete event, got %d endpoints", len(store.endpoints))
	}

	store.ApplyEndpointEvent(fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: endpoint}, time.Unix(4, 0))
	if len(store.endpoints) != 0 {
		t.Fatalf("expected delete event to remove idle endpoint state, got %d endpoints", len(store.endpoints))
	}
}

func TestApplyEndpointEventDeleteDropsActiveEndpointLoadImmediately(t *testing.T) {
	store := newStore()
	endpoint := newTestSchedulingEndpoint("decode-delete-active", "10.0.0.60")
	metadata := endpoint.GetMetadata()

	store.ApplyEndpointEvent(fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint}, time.Unix(1, 0))
	store.StartRequest(RequestStart{RequestID: "req-delete-active", SelectedEndpoint: metadata, InputTokenLength: 5, StartedAt: time.Unix(2, 0)})

	if got := store.Snapshot(metadata); got.Tokens != 5 || got.Requests != 1 {
		t.Fatalf("expected active request load before delete, got %+v", got)
	}

	store.ApplyEndpointEvent(fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: endpoint}, time.Unix(3, 0))

	if got := store.Snapshot(metadata); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected delete event to drop active endpoint load immediately, got %+v", got)
	}
	state, ok := store.requests["req-delete-active"]
	if !ok {
		t.Fatal("expected active request metadata to remain for later completion")
	}
	if len(state.endpointKeys) != 0 {
		t.Fatalf("expected deleted endpoint key to be pruned from active request, got %d keys", len(state.endpointKeys))
	}

	store.CompleteRequest(RequestCompletion{RequestID: "req-delete-active", CompletedAt: time.Unix(4, 0), SelectedEndpoint: metadata})
	completed := store.DrainCompleted(10)
	if len(completed) != 1 {
		t.Fatalf("expected completion metadata to survive delete cleanup, got %d records", len(completed))
	}
	if got := store.Snapshot(metadata); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected completed deleted endpoint to remain at zero load, got %+v", got)
	}
	if _, exists := store.endpoints[EndpointKey("pod://default/decode-delete-active")]; exists {
		t.Fatal("expected deleted endpoint state to be fully removed after completion")
	}
}

func TestApplyEndpointEventDeletePrunesOnlyDeletedTrackedEndpointKey(t *testing.T) {
	store := newStore()
	leader := newTestEndpointMetadata("leader-delete-tracked", "10.0.0.61")
	prefill := newTestSchedulingEndpoint("prefill-delete-tracked", "10.0.0.62")
	decode := newTestSchedulingEndpoint("decode-delete-tracked", "10.0.0.63")

	store.ApplyEndpointEvent(fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: prefill}, time.Unix(1, 0))
	store.ApplyEndpointEvent(fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: decode}, time.Unix(1, 0))
	store.StartRequest(RequestStart{
		RequestID:        "req-delete-tracked",
		SelectedEndpoint: leader,
		TrackedEndpoints: []*fwkdl.EndpointMetadata{prefill.GetMetadata(), decode.GetMetadata()},
		InputTokenLength: 6,
		StartedAt:        time.Unix(2, 0),
	})

	store.ApplyEndpointEvent(fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: decode}, time.Unix(3, 0))

	if got := store.Snapshot(prefill.GetMetadata()); got.Tokens != 6 || got.Requests != 1 {
		t.Fatalf("expected surviving tracked endpoint to keep load, got %+v", got)
	}
	if got := store.Snapshot(decode.GetMetadata()); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected deleted tracked endpoint to drop load immediately, got %+v", got)
	}
	state, ok := store.requests["req-delete-tracked"]
	if !ok {
		t.Fatal("expected active tracked request to remain")
	}
	if len(state.endpointKeys) != 1 || state.endpointKeys[0] != EndpointKey("pod://default/prefill-delete-tracked") {
		t.Fatalf("expected only surviving tracked endpoint to remain, got %+v", state.endpointKeys)
	}

	store.CompleteRequest(RequestCompletion{RequestID: "req-delete-tracked", CompletedAt: time.Unix(4, 0), SelectedEndpoint: decode.GetMetadata()})
	if got := store.Snapshot(prefill.GetMetadata()); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected surviving tracked endpoint load to be released on completion, got %+v", got)
	}
	if got := store.Snapshot(decode.GetMetadata()); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected deleted tracked endpoint load to stay at zero, got %+v", got)
	}
}

func TestApplyEndpointEventAddOrUpdateCreatesActiveEndpoint(t *testing.T) {
	store := newStore()
	endpoint := newTestSchedulingEndpoint("decode-add", "10.0.0.16")
	metadata := endpoint.GetMetadata()
	key, ok := endpointKey(metadata)
	if !ok {
		t.Fatal("expected endpoint metadata to produce a key")
	}

	store.ApplyEndpointEvent(fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint}, time.Unix(10, 0))

	state, exists := store.endpoints[key]
	if !exists {
		t.Fatal("expected add/update event to create endpoint state")
	}
	if !state.Active {
		t.Fatal("expected add/update event to mark endpoint active")
	}
	if !state.LastSeenAt.Equal(time.Unix(10, 0)) {
		t.Fatalf("expected LastSeenAt to be updated, got %v", state.LastSeenAt)
	}
}

func TestStartRequestReplacesExistingLoadContribution(t *testing.T) {
	store := newStore()
	first := newTestEndpointMetadata("decode-old", "10.0.0.17")
	second := newTestEndpointMetadata("decode-new", "10.0.0.18")

	store.StartRequest(RequestStart{RequestID: "req-replace", SelectedEndpoint: first, InputTokenLength: 3, StartedAt: time.Unix(1, 0)})
	store.StartRequest(RequestStart{RequestID: "req-replace", SelectedEndpoint: second, InputTokenLength: 7, StartedAt: time.Unix(2, 0)})

	if got := store.Snapshot(first); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected original endpoint load to be removed, got %+v", got)
	}
	if got := store.Snapshot(second); got.Tokens != 7 || got.Requests != 1 {
		t.Fatalf("expected replacement endpoint load to reflect latest request, got %+v", got)
	}
	active := store.ActiveRequest("req-replace")
	if active == nil || active.MetaData.SelectedEndpoint != "pod://default/decode-new" {
		t.Fatalf("expected active request metadata to be replaced, got %+v", active)
	}
}

func TestDuplicateCompleteRequestDoesNotMakeLoadNegative(t *testing.T) {
	store := newStore()
	selected := newTestEndpointMetadata("decode-complete", "10.0.0.19")

	store.StartRequest(RequestStart{RequestID: "req-complete", SelectedEndpoint: selected, InputTokenLength: 5, StartedAt: time.Unix(1, 0)})
	store.CompleteRequest(RequestCompletion{RequestID: "req-complete", CompletedAt: time.Unix(2, 0)})
	store.CompleteRequest(RequestCompletion{RequestID: "req-complete", CompletedAt: time.Unix(3, 0)})

	if got := store.Snapshot(selected); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected duplicate completion to keep load at zero, got %+v", got)
	}
	completed := store.DrainCompleted(10)
	if len(completed) != 1 {
		t.Fatalf("expected duplicate completion to record only one completion, got %d", len(completed))
	}
}

func TestExpireRequestsReleasesTimedOutLoad(t *testing.T) {
	store := newStore()
	selected := newTestEndpointMetadata("decode-expire", "10.0.0.20")
	store.requestTimeout = time.Minute

	store.StartRequest(RequestStart{RequestID: "req-expire", SelectedEndpoint: selected, InputTokenLength: 6, StartedAt: time.Unix(0, 0)})
	store.expireRequests(time.Unix(61, 0))

	if got := store.Snapshot(selected); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected expired request load to be released, got %+v", got)
	}
	if active := store.ActiveRequest("req-expire"); active != nil {
		t.Fatalf("expected expired request to be removed, got %+v", active)
	}
}

func TestCloseRemovesStoreFromRegistry(t *testing.T) {
	store := StoreForName("close-registry")
	if err := store.Close(); err != nil {
		t.Fatalf("expected close to succeed, got %v", err)
	}
	reloaded := StoreForName("close-registry")
	if reloaded == store {
		t.Fatal("expected closed store to be removed from registry")
	}
}

func TestEndpointKeyUsesPrefixedFormats(t *testing.T) {
	podKey, ok := endpointKey(newTestEndpointMetadata("decode-key", "10.0.0.13"))
	if !ok {
		t.Fatal("expected pod-backed metadata to produce a key")
	}
	if podKey != "pod://default/decode-key" {
		t.Fatalf("expected prefixed pod key, got %q", podKey)
	}

	addrKey, ok := endpointKey(&fwkdl.EndpointMetadata{Address: "10.0.0.14", Port: "9000"})
	if !ok {
		t.Fatal("expected address-backed metadata to produce a key")
	}
	if addrKey != "addr://10.0.0.14:9000" {
		t.Fatalf("expected prefixed address key, got %q", addrKey)
	}
}

func TestMarkFirstChunkOnlyUsesFirstChunkTimeAndCompletes(t *testing.T) {
	store := newStore()
	selected := newTestEndpointMetadata("decode-b", "10.0.0.9")

	store.StartRequest(RequestStart{
		RequestID:        "req-2",
		InputTokenLength: 3,
		SelectedEndpoint: selected,
		StartedAt:        time.Unix(200, 0),
	})

	store.MarkFirstChunk("req-2", time.Unix(201, 0))
	store.MarkFirstChunk("req-2", time.Unix(205, 0))
	store.CompleteRequest(RequestCompletion{
		RequestID:    "req-2",
		CompletedAt:  time.Unix(206, 0),
		OutputLength: 128,
	})

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
	if completed[0].OutputLength != 128 {
		t.Fatalf("expected OutputLength=128, got %d", completed[0].OutputLength)
	}
	if got := store.Snapshot(selected); got.Tokens != 0 || got.Requests != 0 {
		t.Fatalf("expected selected endpoint load to be released, got %+v", got)
	}
}

func TestCompleteRequestFlushesCompletedBatchWhenThresholdReached(t *testing.T) {
	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "requests.jsonl")
	store := newStoreWithPersistence(PersistenceConfig{
		Enabled:        true,
		FlushThreshold: 2,
		OutputPath:     outputPath,
	})

	store.StartRequest(RequestStart{RequestID: "req-3", SelectedEndpoint: newTestEndpointMetadata("decode-c", "10.0.1.1"), StartedAt: time.Unix(300, 0)})
	store.MarkFirstChunk("req-3", time.Unix(301, 0))
	store.CompleteRequest(RequestCompletion{RequestID: "req-3", CompletedAt: time.Unix(302, 0), OutputLength: 64})

	if _, err := os.Stat(outputPath); !os.IsNotExist(err) {
		t.Fatalf("expected no file flush before threshold, got err=%v", err)
	}

	store.StartRequest(RequestStart{RequestID: "req-4", SelectedEndpoint: newTestEndpointMetadata("decode-d", "10.0.1.2"), StartedAt: time.Unix(400, 0)})
	store.MarkFirstChunk("req-4", time.Unix(401, 0))
	store.CompleteRequest(RequestCompletion{RequestID: "req-4", CompletedAt: time.Unix(402, 0), OutputLength: 32})

	lines := waitForOutputLines(t, outputPath, 2)
	if len(lines) != 2 {
		t.Fatalf("expected 2 flushed records, got %d", len(lines))
	}
	for _, line := range lines {
		var got RequestMetaData
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("expected valid json line, got %v", err)
		}
	}
	if drained := store.DrainCompleted(10); len(drained) != 0 {
		t.Fatalf("expected flushed records to be cleared from memory, got %d", len(drained))
	}
}

func TestCompleteRequestFlushesCompletedBatchWhenIntervalElapsed(t *testing.T) {
	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "requests.jsonl")
	store := newStoreWithPersistence(PersistenceConfig{
		Enabled:        true,
		FlushThreshold: 100,
		FlushInterval:  20 * time.Millisecond,
		OutputPath:     outputPath,
	})
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
	})

	store.StartRequest(RequestStart{RequestID: "req-interval", SelectedEndpoint: newTestEndpointMetadata("decode-interval", "10.0.1.7"), StartedAt: time.Unix(700, 0)})
	store.CompleteRequest(RequestCompletion{RequestID: "req-interval", CompletedAt: time.Unix(701, 0), OutputLength: 16})

	lines := waitForOutputLines(t, outputPath, 1)
	if len(lines) != 1 {
		t.Fatalf("expected 1 flushed record, got %d", len(lines))
	}
	if drained := store.DrainCompleted(10); len(drained) != 0 {
		t.Fatalf("expected interval flush to clear in-memory completed entries, got %d", len(drained))
	}
}

func TestAutoFlushRunsOutsideStoreMutex(t *testing.T) {
	writer := &blockingBatchWriter{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	store := newStoreWithPersistence(PersistenceConfig{
		Enabled:        true,
		FlushThreshold: 1,
		OutputPath:     filepath.Join(t.TempDir(), "requests.jsonl"),
	})
	store.mu.Lock()
	store.flushWriter = writer
	store.mu.Unlock()

	store.StartRequest(RequestStart{RequestID: "req-blocking-flush", SelectedEndpoint: newTestEndpointMetadata("decode-blocking", "10.0.1.3"), StartedAt: time.Unix(500, 0)})

	completeDone := make(chan struct{})
	go func() {
		store.CompleteRequest(RequestCompletion{RequestID: "req-blocking-flush", CompletedAt: time.Unix(501, 0)})
		close(completeDone)
	}()

	select {
	case <-completeDone:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("CompleteRequest blocked behind async flush")
	}
	select {
	case <-writer.started:
	case <-time.After(time.Second):
		t.Fatal("expected async flush writer to start")
	}

	startDone := make(chan struct{})
	go func() {
		store.StartRequest(RequestStart{RequestID: "req-after-blocking-flush", SelectedEndpoint: newTestEndpointMetadata("decode-after-blocking", "10.0.1.4"), StartedAt: time.Unix(502, 0)})
		close(startDone)
	}()

	select {
	case <-startDone:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("StartRequest blocked while async flush writer was blocked")
	}

	close(writer.release)
	if err := store.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestAutoFlushKeepsCompletedBufferBoundedWhenWriterStalls(t *testing.T) {
	writer := &blockingBatchWriter{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	store := newStoreWithPersistence(PersistenceConfig{
		Enabled:        true,
		FlushThreshold: 1,
		OutputPath:     filepath.Join(t.TempDir(), "requests.jsonl"),
	})
	store.mu.Lock()
	store.flushWriter = writer
	store.mu.Unlock()
	defer func() {
		close(writer.release)
		if err := store.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}()

	store.StartRequest(RequestStart{RequestID: "req-block-buffer-seed", SelectedEndpoint: newTestEndpointMetadata("decode-block-buffer-seed", "10.0.1.5"), StartedAt: time.Unix(600, 0)})
	store.CompleteRequest(RequestCompletion{RequestID: "req-block-buffer-seed", CompletedAt: time.Unix(601, 0)})
	select {
	case <-writer.started:
	case <-time.After(time.Second):
		t.Fatal("expected async flush writer to start")
	}

	for index := 0; index < defaultCompletedBufferLimit+5; index++ {
		requestID := "req-block-buffer-" + time.Unix(int64(index), 0).UTC().Format("150405")
		store.StartRequest(RequestStart{RequestID: requestID, SelectedEndpoint: newTestEndpointMetadata("decode-block-buffer", "10.0.1.6"), StartedAt: time.Unix(int64(index), 0)})
		store.CompleteRequest(RequestCompletion{RequestID: requestID, CompletedAt: time.Unix(int64(index+1), 0)})
	}

	drained := store.DrainCompleted(defaultCompletedBufferLimit + 10)
	if len(drained) != defaultCompletedBufferLimit {
		t.Fatalf("expected completed buffer to be capped at %d, got %d", defaultCompletedBufferLimit, len(drained))
	}
	if got := store.DroppedCompletedCount(); got != 5 {
		t.Fatalf("expected dropped completed count to be 5, got %d", got)
	}
}

func newTestEndpointMetadata(name, address string) *fwkdl.EndpointMetadata {
	return &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
		PodName:        name,
		Address:        address,
		Port:           "8000",
	}
}

func newTestSchedulingEndpoint(name, address string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(newTestEndpointMetadata(name, address), fwkdl.NewMetrics())
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

func waitForOutputLines(t *testing.T, outputPath string, want int) []string {
	t.Helper()
	var lastErr error
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		content, err := os.ReadFile(outputPath)
		if err == nil {
			lines := splitNonEmptyLines(string(content))
			if len(lines) >= want {
				return lines
			}
		} else {
			lastErr = err
		}
		time.Sleep(10 * time.Millisecond)
	}
	if lastErr != nil {
		t.Fatalf("timed out waiting for %d output lines in %s: %v", want, outputPath, lastErr)
	}
	t.Fatalf("timed out waiting for %d output lines in %s", want, outputPath)
	return nil
}

func waitForCondition(t *testing.T, condition func() bool, description string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", description)
}

type failingBatchWriter struct {
	retryFrom int
	err       error
}

func (w failingBatchWriter) WriteCompletedBatch(_ PersistenceConfig, records []RequestMetaData) (int, error) {
	if w.err != nil {
		return w.retryFrom, w.err
	}
	return len(records), nil
}

type blockingBatchWriter struct {
	started chan struct{}
	release chan struct{}
}

func (w *blockingBatchWriter) WriteCompletedBatch(_ PersistenceConfig, records []RequestMetaData) (int, error) {
	select {
	case <-w.started:
	default:
		close(w.started)
	}
	<-w.release
	return len(records), nil
}
