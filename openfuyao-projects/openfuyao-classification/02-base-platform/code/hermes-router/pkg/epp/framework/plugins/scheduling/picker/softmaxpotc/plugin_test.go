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

package softmaxpotc

import (
	"context"
	"encoding/json"
	"math"
	"math/rand"
	"strings"
	"sync"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/pdgroup"
)

func TestPickReturnsSingleCandidateDirectly(t *testing.T) {
	picker := New("picker", Config{Temperature: 1.0})
	only := newScoredEndpoint("only", "10.0.80.2", 0.4)

	result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), []*fwkscheduling.ScoredEndpoint{only})

	assertPickedEndpoint(t, result, only.Endpoint)
}

func TestPickReturnsEmptyResultForEmptyInput(t *testing.T) {
	picker := New("picker", Config{Temperature: 1.0})

	result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), nil)

	if result == nil || len(result.TargetEndpoints) != 0 {
		t.Fatalf("expected empty result, got %+v", result)
	}
}

func TestPickWithTwoCandidatesReturnsHigherScore(t *testing.T) {
	picker := New("picker", Config{Temperature: 1.0})
	picker.randFloat = rand.New(rand.NewSource(1)).Float64
	higher := newScoredEndpoint("higher", "10.0.81.2", 0.9)
	lower := newScoredEndpoint("lower", "10.0.81.3", 0.1)

	result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), []*fwkscheduling.ScoredEndpoint{lower, higher})

	assertPickedEndpoint(t, result, higher.Endpoint)
}

func TestPickClampsNaNAndNegativeScores(t *testing.T) {
	tests := []struct {
		name     string
		badScore float64
	}{
		{name: "nan", badScore: math.NaN()},
		{name: "negative", badScore: -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			picker := New("picker", Config{Temperature: 1.0})
			picker.randFloat = rand.New(rand.NewSource(2)).Float64
			good := newScoredEndpoint("good", "10.0.82.2", 0.5)
			bad := newScoredEndpoint("bad", "10.0.82.3", tt.badScore)

			result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), []*fwkscheduling.ScoredEndpoint{good, bad})

			assertPickedEndpoint(t, result, good.Endpoint)
		})
	}
}

func TestFactoryValidatesTemperature(t *testing.T) {
	_, err := Factory("picker", json.RawMessage(`{"temperature":0}`), nil)
	if err == nil || !strings.Contains(err.Error(), "temperature must be greater than 0") {
		t.Fatalf("expected temperature validation error, got %v", err)
	}
}

func TestFactoryRejectsNullParameters(t *testing.T) {
	_, err := Factory("picker", json.RawMessage(`null`), nil)
	if err == nil || !strings.Contains(err.Error(), "explicit null") {
		t.Fatalf("expected explicit null parameters error, got %v", err)
	}
}

func TestLowTemperatureConcentratesOnBestEndpoint(t *testing.T) {
	picker := New("picker", Config{Temperature: 0.01})
	picker.randFloat = rand.New(rand.NewSource(7)).Float64
	best := newScoredEndpoint("best", "10.0.83.2", 0.9)
	middle := newScoredEndpoint("middle", "10.0.83.3", 0.5)
	worst := newScoredEndpoint("worst", "10.0.83.4", 0.1)

	bestCount := 0
	for i := 0; i < 2000; i++ {
		result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), []*fwkscheduling.ScoredEndpoint{best, middle, worst})
		if result != nil && len(result.TargetEndpoints) == 1 && fwkscheduling.EndpointComparer(result.TargetEndpoints[0], best.Endpoint) {
			bestCount++
		}
	}
	if bestCount < 1900 {
		t.Fatalf("expected best endpoint to dominate at low temperature, selected %d times", bestCount)
	}
}

func TestPickReturnsOneEndpointWhenScoresAreEqual(t *testing.T) {
	picker := New("picker", Config{Temperature: 1.0})
	picker.randFloat = rand.New(rand.NewSource(9)).Float64
	left := newScoredEndpoint("left", "10.0.84.2", 0.5)
	right := newScoredEndpoint("right", "10.0.84.3", 0.5)

	result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), []*fwkscheduling.ScoredEndpoint{left, right})

	if result == nil || len(result.TargetEndpoints) != 1 {
		t.Fatalf("expected exactly one endpoint, got %+v", result)
	}
}

func TestHighTemperatureApproachesUniformSoftmaxWeights(t *testing.T) {
	first := newScoredEndpoint("first", "10.0.85.2", 0.9)
	second := newScoredEndpoint("second", "10.0.85.3", 0.5)
	third := newScoredEndpoint("third", "10.0.85.4", 0.1)
	weights := softmaxWeights([]*fwkscheduling.ScoredEndpoint{first, second, third}, 100.0)
	for index, weight := range weights {
		if math.Abs(weight-(1.0/3.0)) > 0.01 {
			t.Fatalf("expected weight %d near 1/3, got %v", index, weight)
		}
	}
}

func TestAllZeroScoresProduceUniformSoftmaxWeights(t *testing.T) {
	first := newScoredEndpoint("first", "10.0.86.2", 0)
	second := newScoredEndpoint("second", "10.0.86.3", 0)
	third := newScoredEndpoint("third", "10.0.86.4", 0)
	weights := softmaxWeights([]*fwkscheduling.ScoredEndpoint{first, second, third}, 1.0)

	for index, weight := range weights {
		if math.Abs(weight-(1.0/3.0)) > 1e-9 {
			t.Fatalf("expected weight %d to equal 1/3, got %v", index, weight)
		}
	}
}

func TestPickDistributionMatchesSoftmaxPOTC(t *testing.T) {
	high := newScoredEndpoint("high", "10.0.90.2", 0.9)
	mid := newScoredEndpoint("mid", "10.0.90.3", 0.5)
	low := newScoredEndpoint("low", "10.0.90.4", 0.1)
	candidates := []*fwkscheduling.ScoredEndpoint{high, mid, low}

	picker := New("picker", Config{Temperature: 1.0})

	counts := map[string]int{"high": 0, "mid": 0, "low": 0}
	const trials = 10000

	for i := 0; i < trials; i++ {
		result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), candidates)
		name := result.TargetEndpoints[0].GetMetadata().PodName
		counts[name]++
	}

	t.Logf("distribution: high=%d mid=%d low=%d", counts["high"], counts["mid"], counts["low"])

	if counts["low"] != 0 {
		t.Fatalf("expected low to never win in POTC comparison, got %d selections", counts["low"])
	}
	if counts["mid"] == 0 {
		t.Fatal("expected mid-score endpoint to be picked at least once (beats low)")
	}
	if counts["high"] < counts["mid"] {
		t.Fatal("expected high > mid in selection count")
	}
}

func TestPickSupportsConcurrentUse(t *testing.T) {
	picker := New("picker", Config{Temperature: 1.0})
	first := newScoredEndpoint("first", "10.0.86.5", 0.9)
	second := newScoredEndpoint("second", "10.0.86.6", 0.5)
	third := newScoredEndpoint("third", "10.0.86.7", 0.1)
	inputs := []*fwkscheduling.ScoredEndpoint{first, second, third}

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), inputs)
				if result == nil || len(result.TargetEndpoints) != 1 {
					t.Error("expected one target endpoint per concurrent pick")
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestPickUsesReadyPDLeadersOnly(t *testing.T) {
	picker := New("picker", Config{Temperature: 1.0})
	picker.randFloat = rand.New(rand.NewSource(1)).Float64

	pendingLeader := newScoredEndpoint("pending", "10.0.87.2", 0.99)
	pdgroup.Set(pendingLeader.Endpoint, &pdgroup.Info{
		GroupID:     "group-pending",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: newEndpoint("prefill-pending", "10.0.87.3"), Score: 0.1}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: newEndpoint("decode-pending", "10.0.87.4"), Score: 0.2}},
	})

	readyLeader := newScoredEndpoint("ready", "10.0.87.5", 0.5)
	pdgroup.Set(readyLeader.Endpoint, &pdgroup.Info{
		GroupID:     "group-ready",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: newEndpoint("prefill-ready", "10.0.87.6"), Score: 0.1}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: newEndpoint("decode-ready", "10.0.87.7"), Score: 0.2}},
		SelectedPrefillPod: &fwkscheduling.ScoredEndpoint{
			Endpoint: newEndpoint("prefill-selected", "10.0.87.8"),
			Score:    0.1,
		},
		SelectedDecodePod: &fwkscheduling.ScoredEndpoint{
			Endpoint: newEndpoint("decode-selected", "10.0.87.9"),
			Score:    0.2,
		},
	})

	result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), []*fwkscheduling.ScoredEndpoint{pendingLeader, readyLeader})
	assertPickedEndpoint(t, result, readyLeader.Endpoint)
}

func newScoredEndpoint(name, address string, score float64) *fwkscheduling.ScoredEndpoint {
	return &fwkscheduling.ScoredEndpoint{
		Endpoint: newEndpoint(name, address),
		Score:    score,
	}
}

func newEndpoint(name, address string) fwkscheduling.Endpoint {
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

func assertPickedEndpoint(t *testing.T, result *fwkscheduling.ProfileRunResult, want fwkscheduling.Endpoint) {
	t.Helper()
	if result == nil || len(result.TargetEndpoints) != 1 {
		t.Fatalf("expected exactly one picked endpoint, got %+v", result)
	}
	if !fwkscheduling.EndpointComparer(result.TargetEndpoints[0], want) {
		t.Fatalf("unexpected picked endpoint: got %v want %v", result.TargetEndpoints[0], want)
	}
}
