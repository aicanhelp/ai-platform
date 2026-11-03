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

package score

import (
	"math"
	"testing"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	internalinflight "hermes-router/pkg/epp/internal/inflight"
	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/testutil"
)

func TestAggregateFlipsBadnessIntoScore(t *testing.T) {
	endpoint := testutil.NewEndpoint("aggregate", "10.0.50.1", testutil.TestInferencePort, nil)
	endpoint.GetMetrics().KVCacheUsagePercent = 0.5
	endpoint.GetMetrics().WaitingQueueSize = 5

	// Only the XPU cache usage signal is active: badness 0.5 -> score 0.5.
	weights := AggregateWeights{XPUCacheUsageWeight: 1}
	if got := Aggregate(endpoint, 10, 100, weights); !floatClose(got, 0.5) {
		t.Fatalf("expected aggregate score 0.5, got %v", got)
	}

	// All signals active with equal weights: prefix cache miss 1 (no token IDs),
	// kv usage 0.5, queue 5 of 10 and inflight 0 of 100 average to badness 0.5.
	allWeights := AggregateWeights{
		KVCacheNotHitRateWeight: 1,
		XPUCacheUsageWeight:     1,
		WaitingRequestWeight:    1,
		InflightWeight:          1,
	}
	got := Aggregate(endpoint, 10, 100, allWeights)
	if !floatClose(got, 0.5) {
		t.Fatalf("expected aggregate score 0.5, got %v", got)
	}
}

func TestAggregateWithNoActiveWeightsIsBestScore(t *testing.T) {
	endpoint := testutil.NewEndpoint("aggregate-empty", "10.0.50.2", testutil.TestInferencePort, nil)
	if got := Aggregate(endpoint, 10, 100, AggregateWeights{}); !floatClose(got, 1) {
		t.Fatalf("expected best score when no weight is active, got %v", got)
	}
}

func TestPDGroupScoresGroupAndSelectsMembers(t *testing.T) {
	prefill := testutil.NewScoredEndpoint("prefill-a", "10.0.51.1", testutil.TestInferencePort,
		testutil.ScoredEndpointMetrics{Waiting: 2, KVUsage: 0.2}, nil)
	decode := testutil.NewScoredEndpoint("decode-a", "10.0.52.1", testutil.TestInferencePort,
		testutil.ScoredEndpointMetrics{Waiting: 1, Running: 3, KVUsage: 0.1}, nil)
	info := &pdgroup.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{prefill},
		DecodePods:  []fwkscheduling.ScoredEndpoint{decode},
	}

	got, updated, ok := PDGroup(info, PDScorer{
		Prefill: PrefillWeights{KVWeight: 1, PrefixWeight: 1, QueueWeight: 1, InflightWeight: 1},
		Decode:  DecodeWeights{RunningWeight: 1, KVWeight: 1, QueueWeight: 1, InflightWeight: 1},
		Group:   GroupWeights{PrefillScoreWeight: 1, DecodeScoreWeight: 1},
		Limits: PDLimits{
			Prefill: PrefillLimits{MaxQueue: 4, MaxInflight: 100},
			Decode:  DecodeLimits{MaxRunning: 6, MaxQueue: 4, MaxInflight: 100},
		},
		NextRandIntn: func(int) int { return 0 },
	})
	if !ok {
		t.Fatal("expected PD group to score successfully")
	}
	if got <= 0 || got > 1 {
		t.Fatalf("expected score in (0, 1], got %v", got)
	}
	if updated.SelectedPrefillPod == nil || updated.SelectedPrefillPod.GetMetadata().PodName != "prefill-a" {
		t.Fatalf("expected prefill-a to be selected, got %+v", updated.SelectedPrefillPod)
	}
	if updated.SelectedDecodePod == nil || updated.SelectedDecodePod.GetMetadata().PodName != "decode-a" {
		t.Fatalf("expected decode-a to be selected, got %+v", updated.SelectedDecodePod)
	}
	if updated == info {
		t.Fatal("expected PDGroup to return a cloned info")
	}
}

func TestPDGroupRejectsInvalidInfo(t *testing.T) {
	for _, info := range []*pdgroup.Info{
		nil,
		{GroupID: "", PrefillPods: []fwkscheduling.ScoredEndpoint{{}}, DecodePods: []fwkscheduling.ScoredEndpoint{{}}},
		{GroupID: "group-a"},
		{GroupID: "group-a", PrefillPods: []fwkscheduling.ScoredEndpoint{{}}},
	} {
		if _, _, ok := PDGroup(info, PDScorer{}); ok {
			t.Fatalf("expected info %+v to be rejected", info)
		}
	}
}

func TestMaxWaitingQueue(t *testing.T) {
	pods := []fwkscheduling.Endpoint{
		testutil.NewEndpoint("queue-a", "10.0.53.1", testutil.TestInferencePort, nil),
		testutil.NewEndpoint("queue-b", "10.0.53.2", testutil.TestInferencePort, nil),
		testutil.NewEndpoint("queue-c", "10.0.53.3", testutil.TestInferencePort, nil),
	}
	pods[0].GetMetrics().WaitingQueueSize = 3
	pods[1].GetMetrics().WaitingQueueSize = 9
	pods[2].GetMetrics().WaitingQueueSize = 5

	if got := MaxWaitingQueue(pods); got != 9 {
		t.Fatalf("expected max waiting queue 9, got %d", got)
	}
	if got := MaxWaitingQueue(nil); got != 0 {
		t.Fatalf("expected 0 for empty pods, got %d", got)
	}
}

func TestLatencyBadnessFromMillis(t *testing.T) {
	cases := []struct {
		latencyMillis float64
		want          float64
	}{
		{0, 0},
		{-5, 0},
		{math.NaN(), 0},
		{math.Inf(1), 1},
		{1000, 0.5},
		{3000, 0.75},
	}
	for _, tc := range cases {
		if got := LatencyBadnessFromMillis(tc.latencyMillis); !floatClose(got, tc.want) {
			t.Fatalf("latency %v: expected badness %v, got %v", tc.latencyMillis, tc.want, got)
		}
	}
}

func endpointWithInflightTokens(name string, tokens int) fwkscheduling.Endpoint {
	endpoint := testutil.NewEndpoint(name, "10.0.54.1", testutil.TestInferencePort, nil)
	endpoint.Put(internalinflight.DefaultAttributeKey, &internalinflight.Load{Tokens: tokens, Requests: 1})
	return endpoint
}

func TestMaxInflightTokens(t *testing.T) {
	pods := []fwkscheduling.Endpoint{
		endpointWithInflightTokens("inflight-a", 10),
		endpointWithInflightTokens("inflight-b", 70),
		endpointWithInflightTokens("inflight-c", 40),
	}
	if got := MaxInflightTokens(pods); !floatClose(got, 70) {
		t.Fatalf("expected max inflight tokens 70, got %v", got)
	}
	if got := MaxInflightTokens(nil); got != 0 {
		t.Fatalf("expected 0 for empty pods, got %v", got)
	}

	scored := []fwkscheduling.ScoredEndpoint{
		{Endpoint: pods[0]},
		{Endpoint: pods[1]},
		{Endpoint: pods[2]},
	}
	if got := MaxInflightTokensFromScored(scored); !floatClose(got, 70) {
		t.Fatalf("expected max inflight tokens 70 from scored pods, got %v", got)
	}

	// Endpoints without an inflight attribute contribute nothing.
	bare := testutil.NewEndpoint("bare", "10.0.54.2", testutil.TestInferencePort, nil)
	if got := InflightTokens(bare); got != 0 {
		t.Fatalf("expected 0 inflight tokens for bare endpoint, got %v", got)
	}
	if got := InflightTokens(nil); got != 0 {
		t.Fatalf("expected 0 inflight tokens for nil endpoint, got %v", got)
	}
}

func TestHasNegativeWeight(t *testing.T) {
	if HasNegativeWeight() {
		t.Fatal("expected no negative weight for empty input")
	}
	if HasNegativeWeight(0, 1, 2) {
		t.Fatal("expected no negative weight for non-negative values")
	}
	if !HasNegativeWeight(1, -0.1, 2) {
		t.Fatal("expected negative weight to be detected")
	}
}

func floatClose(got, want float64) bool {
	return math.Abs(got-want) < 1e-9
}
