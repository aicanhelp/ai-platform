/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *         http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package testutil

import (
	"errors"
	"testing"

	"github.com/go-logr/logr"
)

func TestRecordingSinkRecordsInfoAndErrorEntries(t *testing.T) {
	_, sink := ContextWithRecordingLogger()
	sink.Init(logr.RuntimeInfo{})
	if !sink.Enabled(0) {
		t.Fatal("expected sink to be enabled")
	}

	sink.Info(0, "processing request", "model", "qwen", "requestId", "r-1")
	sink.Info(0, "request finished", "model", "qwen")
	sink.Error(errors.New("boom"), "request failed", "requestId", "r-2")

	info := sink.LastInfo("request finished")
	if info.Message != "request finished" {
		t.Fatalf("expected last info entry, got %+v", info)
	}
	if info.Keys["model"] != "qwen" {
		t.Fatalf("expected model key on info entry, got %+v", info.Keys)
	}

	errEntry := sink.LastError("request failed")
	if errEntry.Error != "boom" {
		t.Fatalf("expected error string on error entry, got %+v", errEntry)
	}
	if errEntry.Keys["requestId"] != "r-2" {
		t.Fatalf("expected requestId key on error entry, got %+v", errEntry.Keys)
	}

	sink.Error(nil, "error without cause")
	if got := sink.LastError("error without cause").Error; got != "" {
		t.Fatalf("expected empty error string, got %q", got)
	}
}

func TestRecordingSinkLastEntriesReturnZeroValuesWhenMissing(t *testing.T) {
	_, sink := ContextWithRecordingLogger()
	if got := sink.LastInfo("missing"); got.Message != "" {
		t.Fatalf("expected empty info entry, got %+v", got)
	}
	if got := sink.LastError("missing"); got.Message != "" {
		t.Fatalf("expected empty error entry, got %+v", got)
	}
}

func TestRecordingSinkWithValuesAndWithNameReturnSameSink(t *testing.T) {
	_, sink := ContextWithRecordingLogger()
	if sink.WithValues("k", "v") != sink {
		t.Fatal("expected WithValues to return the same sink")
	}
	if sink.WithName("child") != sink {
		t.Fatal("expected WithName to return the same sink")
	}
}

func TestNewHandleContextIsBackground(t *testing.T) {
	handle := NewHandle()
	if handle == nil {
		t.Fatal("expected non-nil handle")
	}
}

func TestNewEndpointCopiesLabels(t *testing.T) {
	labels := map[string]string{"app": "hermes"}
	endpoint := NewEndpoint("pod-a", "10.0.60.1", TestInferencePort, labels)
	labels["app"] = "mutated"

	metadata := endpoint.GetMetadata()
	if metadata.PodName != "pod-a" {
		t.Fatalf("expected pod name pod-a, got %q", metadata.PodName)
	}
	if metadata.Address != "10.0.60.1" {
		t.Fatalf("expected address 10.0.60.1, got %q", metadata.Address)
	}
	if metadata.Port != TestInferencePort {
		t.Fatalf("expected port %q, got %q", TestInferencePort, metadata.Port)
	}
	if got := metadata.Labels["app"]; got != "hermes" {
		t.Fatalf("expected cloned label to be hermes, got %q", got)
	}

	nilLabelEndpoint := NewEndpoint("pod-b", "10.0.60.2", TestInferencePort, nil)
	if nilLabelEndpoint.GetMetadata().Labels == nil {
		t.Fatal("expected non-nil label map for nil labels input")
	}
}

func TestNewScoredEndpointPopulatesMetricsAndScore(t *testing.T) {
	scored := NewScoredEndpoint("pod-c", "10.0.61.1", TestInferencePort, ScoredEndpointMetrics{
		Score:   0.75,
		Waiting: 3,
		KVUsage: 0.4,
		Running: 7,
	}, map[string]string{"role": "prefill"})

	if scored.Score != 0.75 {
		t.Fatalf("expected score 0.75, got %v", scored.Score)
	}
	metrics := scored.Endpoint.GetMetrics()
	if metrics.WaitingQueueSize != 3 {
		t.Fatalf("expected waiting queue size 3, got %d", metrics.WaitingQueueSize)
	}
	if metrics.KVCacheUsagePercent != 0.4 {
		t.Fatalf("expected kv cache usage 0.4, got %v", metrics.KVCacheUsagePercent)
	}
	if metrics.RunningRequestsSize != 7 {
		t.Fatalf("expected running requests size 7, got %d", metrics.RunningRequestsSize)
	}
	if got := scored.Endpoint.GetMetadata().Labels["role"]; got != "prefill" {
		t.Fatalf("expected role label prefill, got %q", got)
	}
}

func TestNewCompletionsRequestCarriesPrompt(t *testing.T) {
	request := NewCompletionsRequest("r-1", "qwen", "hello world")
	if request.RequestId != "r-1" {
		t.Fatalf("expected request id r-1, got %q", request.RequestId)
	}
	if request.TargetModel != "qwen" {
		t.Fatalf("expected target model qwen, got %q", request.TargetModel)
	}
	if request.Body.Completions.Prompt.Raw != "hello world" {
		t.Fatalf("expected raw prompt, got %q", request.Body.Completions.Prompt.Raw)
	}
}

func TestNewChatRequestCarriesMessages(t *testing.T) {
	request := NewChatRequest("r-2", "qwen", "first", "second")
	messages := request.Body.ChatCompletions.Messages
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}
	if messages[0].Content.Raw != "first" || messages[1].Content.Raw != "second" {
		t.Fatalf("unexpected message contents: %+v", messages)
	}
}

func TestMustSplitHostPort(t *testing.T) {
	hostname, port := MustSplitHostPort(t, "http://10.0.62.1:9090/metrics")
	if hostname != "10.0.62.1" {
		t.Fatalf("expected hostname 10.0.62.1, got %q", hostname)
	}
	if port != 9090 {
		t.Fatalf("expected port 9090, got %d", port)
	}

	hostname, port = MustSplitHostPort(t, "http://10.0.62.2/metrics")
	if hostname != "10.0.62.2" {
		t.Fatalf("expected hostname 10.0.62.2, got %q", hostname)
	}
	if port != 0 {
		t.Fatalf("expected default port 0, got %d", port)
	}
}

func TestNewSchedulingResultExposesPrimaryTargets(t *testing.T) {
	target := NewEndpoint("pod-d", "10.0.63.1", TestInferencePort, nil)
	result := NewSchedulingResult(target)
	if result.PrimaryProfileName != "primary" {
		t.Fatalf("expected primary profile, got %q", result.PrimaryProfileName)
	}
	profile, ok := result.ProfileResults["primary"]
	if !ok {
		t.Fatal("expected primary profile result")
	}
	if len(profile.TargetEndpoints) != 1 || profile.TargetEndpoints[0].GetMetadata().PodName != "pod-d" {
		t.Fatalf("unexpected target endpoints: %+v", profile.TargetEndpoints)
	}
}

func TestEndOfStreamResponse(t *testing.T) {
	response := EndOfStreamResponse()
	if !response.EndOfStream {
		t.Fatal("expected end-of-stream response")
	}
}

func TestAssertFloatClose(t *testing.T) {
	AssertFloatClose(t, 0.1+0.2, 0.3)
}

func TestContextWithRecordingLoggerReturnsUsableContext(t *testing.T) {
	ctx, sink := ContextWithRecordingLogger()
	if sink == nil {
		t.Fatal("expected non-nil sink")
	}
	if err := ctx.Err(); err != nil {
		t.Fatalf("expected live context, got %v", err)
	}
}
