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

package utils

import (
	"encoding/json"
	"math/rand"
	"testing"
	"time"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/testutil"
)

func TestDurationUnmarshalJSONAcceptsDurationStrings(t *testing.T) {
	cases := []struct {
		raw  string
		want time.Duration
	}{
		{`"100ms"`, 100 * time.Millisecond},
		{`"1.5s"`, 1500 * time.Millisecond},
		{`"2h"`, 2 * time.Hour},
	}
	for _, tc := range cases {
		var got Duration
		if err := json.Unmarshal([]byte(tc.raw), &got); err != nil {
			t.Fatalf("unmarshal %s: %v", tc.raw, err)
		}
		if got.Std() != tc.want {
			t.Fatalf("unmarshal %s: expected %v, got %v", tc.raw, tc.want, got.Std())
		}
	}
}

func TestDurationUnmarshalJSONAcceptsNumericNanoseconds(t *testing.T) {
	var got Duration
	if err := json.Unmarshal([]byte("1500000000"), &got); err != nil {
		t.Fatalf("unmarshal numeric duration: %v", err)
	}
	if got.Std() != 1500*time.Millisecond {
		t.Fatalf("expected 1.5s, got %v", got.Std())
	}
}

func TestDurationUnmarshalJSONRejectsInvalidValues(t *testing.T) {
	for _, raw := range []string{`"not-a-duration"`, `true`, `{}`} {
		var got Duration
		if err := json.Unmarshal([]byte(raw), &got); err == nil {
			t.Fatalf("expected %s to be rejected", raw)
		}
	}
}

func TestDurationStdConvertsZeroAndNegativeValues(t *testing.T) {
	if got := Duration(0).Std(); got != 0 {
		t.Fatalf("expected zero duration, got %v", got)
	}
	if got := Duration(-time.Second).Std(); got != -time.Second {
		t.Fatalf("expected negative duration, got %v", got)
	}
}

func TestDecodePluginParameters(t *testing.T) {
	var got struct {
		Model string `json:"model"`
	}
	if err := DecodePluginParameters(nil, &got); err != nil {
		t.Fatalf("expected empty raw message to be accepted: %v", err)
	}
	if err := DecodePluginParameters([]byte(`{"model":"m-1"}`), &got); err != nil {
		t.Fatalf("expected valid parameters to decode: %v", err)
	}
	if got.Model != "m-1" {
		t.Fatalf("expected model m-1, got %q", got.Model)
	}
	if err := DecodePluginParameters([]byte(`{"model":"m-1","unknown":1}`), &got); err == nil {
		t.Fatal("expected unknown field to be rejected")
	}
	if err := DecodePluginParameters([]byte(`{invalid`), &got); err == nil {
		t.Fatal("expected invalid JSON to be rejected")
	}
}

func TestValidateEmptyPluginParameters(t *testing.T) {
	valid := [][]byte{nil, []byte(""), []byte("   "), []byte("{}"), []byte(" { } ")}
	for _, raw := range valid {
		if err := ValidateEmptyPluginParameters(raw); err != nil {
			t.Fatalf("expected %q to be accepted: %v", raw, err)
		}
	}

	invalid := [][]byte{[]byte("null"), []byte(`{"extra":1}`), []byte("bad-json")}
	for _, raw := range invalid {
		if err := ValidateEmptyPluginParameters(raw); err == nil {
			t.Fatalf("expected %q to be rejected", raw)
		}
	}
}

func newScoredCandidate(name string, score float64) *fwkscheduling.ScoredEndpoint {
	endpoint := testutil.NewScoredEndpoint(name, "10.0.0.1", "8000", testutil.ScoredEndpointMetrics{Score: score}, nil)
	return &endpoint
}

func TestSelectExtremeScoreResultEmptyCandidates(t *testing.T) {
	result := SelectExtremeScoreResult(nil, rand.New(rand.NewSource(1)), func(candidate, current float64) bool {
		return candidate > current
	})
	if result == nil || len(result.TargetEndpoints) != 0 {
		t.Fatalf("expected empty result for no candidates, got %+v", result)
	}
}

func TestSelectExtremeScoreResultNilEndpointWinner(t *testing.T) {
	candidates := []*fwkscheduling.ScoredEndpoint{{Endpoint: nil, Score: 1}}
	result := SelectExtremeScoreResult(candidates, rand.New(rand.NewSource(1)), func(candidate, current float64) bool {
		return candidate > current
	})
	if result == nil || len(result.TargetEndpoints) != 0 {
		t.Fatalf("expected empty result for nil endpoint winner, got %+v", result)
	}
}

func TestSelectExtremeScoreResultPicksMaximum(t *testing.T) {
	candidates := []*fwkscheduling.ScoredEndpoint{
		newScoredCandidate("pod-a", 0.2),
		newScoredCandidate("pod-b", 0.9),
		newScoredCandidate("pod-c", 0.5),
	}
	result := SelectExtremeScoreResult(candidates, rand.New(rand.NewSource(1)), func(candidate, current float64) bool {
		return candidate > current
	})
	if len(result.TargetEndpoints) != 1 {
		t.Fatalf("expected single target endpoint, got %d", len(result.TargetEndpoints))
	}
	if got := result.TargetEndpoints[0].GetMetadata().PodName; got != "pod-b" {
		t.Fatalf("expected pod-b to win, got %q", got)
	}
}

func TestSelectExtremeScoreResultPicksMinimum(t *testing.T) {
	candidates := []*fwkscheduling.ScoredEndpoint{
		newScoredCandidate("pod-a", 0.8),
		newScoredCandidate("pod-b", 0.1),
		newScoredCandidate("pod-c", 0.4),
	}
	result := SelectExtremeScoreResult(candidates, rand.New(rand.NewSource(1)), func(candidate, current float64) bool {
		return candidate < current
	})
	if len(result.TargetEndpoints) != 1 {
		t.Fatalf("expected single target endpoint, got %d", len(result.TargetEndpoints))
	}
	if got := result.TargetEndpoints[0].GetMetadata().PodName; got != "pod-b" {
		t.Fatalf("expected pod-b to win, got %q", got)
	}
}

func TestSelectExtremeScoreResultBreaksTiesAmongWinners(t *testing.T) {
	candidates := []*fwkscheduling.ScoredEndpoint{
		newScoredCandidate("pod-a", 0.5),
		newScoredCandidate("pod-b", 0.5),
		newScoredCandidate("pod-c", 0.1),
	}
	// Any seeded RNG must pick exactly one of the tied winners.
	result := SelectExtremeScoreResult(candidates, rand.New(rand.NewSource(7)), func(candidate, current float64) bool {
		return candidate > current
	})
	if len(result.TargetEndpoints) != 1 {
		t.Fatalf("expected single target endpoint, got %d", len(result.TargetEndpoints))
	}
	podName := result.TargetEndpoints[0].GetMetadata().PodName
	if podName != "pod-a" && podName != "pod-b" {
		t.Fatalf("expected a tied winner, got %q", podName)
	}
}

func TestPickTiedIndex(t *testing.T) {
	if got := PickTiedIndex([]int{42}, nil); got != 42 {
		t.Fatalf("expected 42 for single index, got %d", got)
	}
	if got := PickTiedIndex([]int{1, 2, 3}, nil); got != 1 {
		t.Fatalf("expected first index when nextRandIntn is nil, got %d", got)
	}
	if got := PickTiedIndex([]int{10, 20, 30}, func(n int) int { return n - 1 }); got != 30 {
		t.Fatalf("expected last index from stub RNG, got %d", got)
	}
	if got := PickTiedIndex([]int{10, 20, 30}, func(n int) int { return 0 }); got != 10 {
		t.Fatalf("expected first index from stub RNG, got %d", got)
	}
}
