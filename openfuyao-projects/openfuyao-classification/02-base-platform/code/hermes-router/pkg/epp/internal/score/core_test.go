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

package score

import (
	"math"
	"math/rand"
	"testing"

	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/testutil"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestClamp01(t *testing.T) {
	cases := []struct {
		name string
		in   float64
		want float64
	}{
		{"below zero clamps to zero", -0.5, 0},
		{"zero stays zero", 0, 0},
		{"middle stays middle", 0.42, 0.42},
		{"one stays one", 1, 1},
		{"above one clamps to one", 1.7, 1},
		{"NaN clamps to zero", math.NaN(), 0},
		{"+Inf clamps to one", math.Inf(1), 1},
		{"-Inf clamps to zero", math.Inf(-1), 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Clamp01(tc.in); got != tc.want {
				t.Fatalf("Clamp01(%v) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestFromBadnessLinearFlip(t *testing.T) {
	cases := []struct {
		name string
		in   float64
		want float64
	}{
		{"zero badness yields max score", 0, 1},
		{"half badness yields half score", 0.5, 0.5},
		{"full badness yields zero score", 1, 0},
		{"out of range clamps", 1.5, 0},
		{"negative badness clamps", -0.2, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testutil.AssertFloatClose(t, FromBadness(tc.in), tc.want)
		})
	}
}

func TestKVCacheUsageBadnessUsesGIEFraction(t *testing.T) {
	endpoint := endpointWithKVCacheUsage(0.75)

	testutil.AssertFloatClose(t, AggregateBadness(endpoint, 0, 0, AggregateWeights{XPUCacheUsageWeight: 1}), 0.75)
	testutil.AssertFloatClose(t, PrefillBadness(endpoint, 0, 0, PrefillWeights{KVWeight: 1}), 0.75)
	testutil.AssertFloatClose(t, DecodeBadness(endpoint, 0, 0, 0, DecodeWeights{KVWeight: 1}), 0.75)
}

func TestWeightedBadnessAveragesActiveSignals(t *testing.T) {
	signals := []Signal{
		{Name: "kv", Weight: 1, Badness: 0.4},
		{Name: "queue", Weight: 3, Badness: 0.8},
	}
	got, ok := WeightedBadness(signals)
	if !ok {
		t.Fatal("expected weighted badness to be available")
	}
	// (1*0.4 + 3*0.8) / (1+3) = 2.8/4 = 0.7
	testutil.AssertFloatClose(t, got, 0.7)
}

func TestWeightedBadnessSkipsNonPositiveWeights(t *testing.T) {
	signals := []Signal{
		{Name: "kv", Weight: 0, Badness: 1},     // skipped
		{Name: "queue", Weight: -1, Badness: 1}, // skipped
		{Name: "inflight", Weight: 2, Badness: 0.3},
	}
	got, ok := WeightedBadness(signals)
	if !ok {
		t.Fatal("expected weighted badness to be available")
	}
	testutil.AssertFloatClose(t, got, 0.3)
}

func TestWeightedBadnessReturnsFalseWhenAllSkipped(t *testing.T) {
	signals := []Signal{
		{Name: "kv", Weight: 0, Badness: 1},
		{Name: "queue", Weight: 0, Badness: 1},
	}
	if got, ok := WeightedBadness(signals); ok || got != 0 {
		t.Fatalf("expected (0, false), got (%v, %v)", got, ok)
	}
}

func TestWeightedBadnessClampsPerSignalBadness(t *testing.T) {
	signals := []Signal{
		{Name: "out-of-range", Weight: 1, Badness: 1.5}, // clamped to 1
		{Name: "negative", Weight: 1, Badness: -0.2},    // clamped to 0
	}
	got, ok := WeightedBadness(signals)
	if !ok {
		t.Fatal("expected weighted badness to be available")
	}
	// (1*1 + 1*0) / 2 = 0.5
	testutil.AssertFloatClose(t, got, 0.5)
}

func TestWeightGroupSumAndHasNegative(t *testing.T) {
	prefill := PrefillWeights{KVWeight: 1, PrefixWeight: 2, QueueWeight: 3, InflightWeight: 4}
	if got := prefill.Sum(); got != 10 {
		t.Fatalf("PrefillWeights.Sum() = %v, want 10", got)
	}
	if prefill.HasNegative() {
		t.Fatal("expected non-negative PrefillWeights")
	}
	if !(PrefillWeights{QueueWeight: -0.1}).HasNegative() {
		t.Fatal("expected PrefillWeights with a negative field to report HasNegative")
	}

	decode := DecodeWeights{PrefixWeight: 1, RunningWeight: 1, KVWeight: 2, QueueWeight: 3, InflightWeight: 4}
	if got := decode.Sum(); got != 11 {
		t.Fatalf("DecodeWeights.Sum() = %v, want 11", got)
	}
	if decode.HasNegative() {
		t.Fatal("expected non-negative DecodeWeights")
	}
	if !(DecodeWeights{PrefixWeight: -0.1}).HasNegative() {
		t.Fatal("expected DecodeWeights with a negative PrefixWeight to report HasNegative")
	}

	group := GroupWeights{PrefillScoreWeight: 0.3, DecodeScoreWeight: 0.2}
	if got := group.Sum(); got != 0.5 {
		t.Fatalf("GroupWeights.Sum() = %v, want 0.5", got)
	}
	if group.HasNegative() {
		t.Fatal("expected non-negative GroupWeights")
	}

	aggregate := AggregateWeights{
		KVCacheNotHitRateWeight: 1,
		XPUCacheUsageWeight:     2,
		WaitingRequestWeight:    3,
		InflightWeight:          4,
	}
	if got := aggregate.Sum(); got != 10 {
		t.Fatalf("AggregateWeights.Sum() = %v, want 10", got)
	}
	if aggregate.HasNegative() {
		t.Fatal("expected non-negative AggregateWeights")
	}
}

func TestWeightGroupFillDefaultsFillsEmptyAndKeepsConfigured(t *testing.T) {
	var empty PrefillWeights
	empty.FillDefaults(1)
	wantPrefill := PrefillWeights{KVWeight: 1, PrefixWeight: 1, QueueWeight: 1, InflightWeight: 1}
	if empty != wantPrefill {
		t.Fatalf("FillDefaults empty prefill = %+v, want %+v", empty, wantPrefill)
	}

	configured := PrefillWeights{KVWeight: 2}
	configured.FillDefaults(1)
	if configured != (PrefillWeights{KVWeight: 2}) {
		t.Fatalf("FillDefaults must not rewrite a group whose sum is already positive, got %+v", configured)
	}

	var emptyDecode DecodeWeights
	emptyDecode.FillDefaults(1)
	wantDecode := DecodeWeights{
		PrefixWeight:   1,
		RunningWeight:  1,
		KVWeight:       1,
		QueueWeight:    1,
		InflightWeight: 1,
	}
	if emptyDecode != wantDecode {
		t.Fatalf("FillDefaults empty decode = %+v, want %+v", emptyDecode, wantDecode)
	}

	var emptyGroup GroupWeights
	emptyGroup.FillDefaults(0.5)
	wantGroup := GroupWeights{PrefillScoreWeight: 0.5, DecodeScoreWeight: 0.5}
	if emptyGroup != wantGroup {
		t.Fatalf("FillDefaults empty group = %+v, want %+v", emptyGroup, wantGroup)
	}

	var nilPrefill *PrefillWeights
	nilPrefill.FillDefaults(1)
}

func TestResolvePDWeightDefaultsFillsEmptyGroups(t *testing.T) {
	prefill := PrefillWeights{}
	decode := DecodeWeights{}
	group := GroupWeights{}

	ResolvePDWeightDefaults(&prefill, &decode, &group, 1, 0.5)

	wantPrefill := PrefillWeights{KVWeight: 1, PrefixWeight: 1, QueueWeight: 1, InflightWeight: 1}
	if prefill != wantPrefill {
		t.Fatalf("expected empty prefill group to default to 1, got %+v", prefill)
	}
	wantDecode := DecodeWeights{
		PrefixWeight:   1,
		RunningWeight:  1,
		KVWeight:       1,
		QueueWeight:    1,
		InflightWeight: 1,
	}
	if decode != wantDecode {
		t.Fatalf("expected empty decode group to default to 1, got %+v", decode)
	}
	wantGroup := GroupWeights{PrefillScoreWeight: 0.5, DecodeScoreWeight: 0.5}
	if group != wantGroup {
		t.Fatalf("expected empty group weights to default to 0.5, got %+v", group)
	}
}

func TestResolvePDWeightDefaultsKeepsConfiguredGroups(t *testing.T) {
	prefill := PrefillWeights{KVWeight: 2}
	decode := DecodeWeights{KVWeight: 0.75}
	group := GroupWeights{PrefillScoreWeight: 0.3}

	ResolvePDWeightDefaults(&prefill, &decode, &group, 1, 0.5)

	if prefill != (PrefillWeights{KVWeight: 2}) {
		t.Fatalf("expected prefill weights to stay configured, got %+v", prefill)
	}
	if decode != (DecodeWeights{KVWeight: 0.75}) {
		t.Fatalf("expected decode weights to stay configured, got %+v", decode)
	}
	if group != (GroupWeights{PrefillScoreWeight: 0.3}) {
		t.Fatalf("expected group weights to stay configured, got %+v", group)
	}
}

func TestSelectBestPrefillRandomizesEqualBadness(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	weights := PrefillWeights{KVWeight: 1, PrefixWeight: 1, QueueWeight: 1, InflightWeight: 1}
	first := testutil.NewScoredEndpoint("prefill-a", "10.0.31.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 2, KVUsage: 0.2, Running: 0}, nil)
	second := testutil.NewScoredEndpoint("prefill-b", "10.0.31.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 2, KVUsage: 0.2, Running: 0}, nil)

	selected, badness := SelectBestPrefill([]fwkscheduling.ScoredEndpoint{first, second}, 2, 0, weights, rng.Intn)

	if selected == nil || selected.Endpoint == nil {
		t.Fatal("expected a selected prefill endpoint")
	}
	if got := selected.Endpoint.GetMetadata().PodName; got != "prefill-b" {
		t.Fatalf("expected second tied prefill endpoint, got %q", got)
	}
	if want := PrefillBadness(second.Endpoint, 2, 0, weights); badness != want {
		t.Fatalf("expected badness %v, got %v", want, badness)
	}
}

func TestSelectBestDecodeRandomizesEqualBadness(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	weights := DecodeWeights{RunningWeight: 1, KVWeight: 1, QueueWeight: 1, InflightWeight: 1}
	first := testutil.NewScoredEndpoint("decode-a", "10.0.32.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 3, KVUsage: 0.3, Running: 4}, nil)
	second := testutil.NewScoredEndpoint("decode-b", "10.0.32.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 3, KVUsage: 0.3, Running: 4}, nil)

	selected, badness := SelectBestDecode(
		[]fwkscheduling.ScoredEndpoint{first, second},
		DecodeLimits{MaxRunning: 4, MaxQueue: 3},
		weights,
		rng.Intn,
	)

	if selected == nil || selected.Endpoint == nil {
		t.Fatal("expected a selected decode endpoint")
	}
	if got := selected.Endpoint.GetMetadata().PodName; got != "decode-b" {
		t.Fatalf("expected second tied decode endpoint, got %q", got)
	}
	if want := DecodeBadness(second.Endpoint, 4, 3, 0, weights); badness != want {
		t.Fatalf("expected badness %v, got %v", want, badness)
	}
}

func TestCollectPDLimitsUsesPoolWideMaxima(t *testing.T) {
	leaderA := testutil.NewEndpoint("leader-a", "10.0.40.1", testutil.TestInferencePort, nil)
	leaderB := testutil.NewEndpoint("leader-b", "10.0.40.2", testutil.TestInferencePort, nil)
	bare := testutil.NewEndpoint("bare", "10.0.40.3", testutil.TestInferencePort, nil)

	prefillA := testutil.NewScoredEndpoint("prefill-a", "10.0.41.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Waiting: 2}, nil)
	decodeA := testutil.NewScoredEndpoint("decode-a", "10.0.42.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Waiting: 4, Running: 3}, nil)
	prefillB := testutil.NewScoredEndpoint("prefill-b", "10.0.41.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Waiting: 10}, nil)
	decodeB := testutil.NewScoredEndpoint("decode-b", "10.0.42.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Waiting: 1, Running: 7}, nil)

	pdgroup.Set(leaderA, &pdgroup.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{prefillA},
		DecodePods:  []fwkscheduling.ScoredEndpoint{decodeA},
	})
	pdgroup.Set(leaderB, &pdgroup.Info{
		GroupID:     "group-b",
		PrefillPods: []fwkscheduling.ScoredEndpoint{prefillB},
		DecodePods:  []fwkscheduling.ScoredEndpoint{decodeB},
	})

	// The bare endpoint carries no group metadata and must be skipped.
	limits := CollectPDLimits([]fwkscheduling.Endpoint{leaderA, bare, leaderB})

	if limits.Prefill.MaxQueue != 10 {
		t.Fatalf("expected pool-wide prefill max queue 10, got %d", limits.Prefill.MaxQueue)
	}
	if limits.Decode.MaxQueue != 4 {
		t.Fatalf("expected pool-wide decode max queue 4, got %d", limits.Decode.MaxQueue)
	}
	if limits.Decode.MaxRunning != 7 {
		t.Fatalf("expected pool-wide decode max running 7, got %d", limits.Decode.MaxRunning)
	}
}

func endpointWithKVCacheUsage(kvUsage float64) fwkscheduling.Endpoint {
	endpoint := testutil.NewEndpoint("kv-cache-usage", "10.0.30.1", testutil.TestInferencePort, nil)
	endpoint.GetMetrics().KVCacheUsagePercent = kvUsage
	return endpoint
}
