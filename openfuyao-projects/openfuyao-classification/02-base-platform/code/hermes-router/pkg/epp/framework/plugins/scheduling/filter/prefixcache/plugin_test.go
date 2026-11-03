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

package prefixcache

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"sync"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencyattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	endpointmeta "hermes-router/pkg/epp/internal/endpointmeta"
	"hermes-router/pkg/epp/internal/pdgroup"
)

func TestFilterKeepsAboveThresholdEndpoints(t *testing.T) {
	filter := New("filter", Config{Threshold: 0.5, PFilter: 0.8, MaxTTFTPenaltyMs: 5000})
	filter.randFloat = func() float64 { return 0.1 }
	stickyA := newTestEndpoint("sticky-a", "10.0.60.2")
	stickyB := newTestEndpoint("sticky-b", "10.0.60.3")
	setPrefixScore(stickyA, 3, 4)
	setPrefixScore(stickyB, 2, 3)

	filtered := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{stickyA, stickyB})

	assertEndpointsEqual(t, filtered, []fwkscheduling.Endpoint{stickyA, stickyB})
}

func TestFilterFailOpensWhenEverythingIsFiltered(t *testing.T) {
	filter := New("filter", Config{Threshold: 0.5, PFilter: 1.0, MaxTTFTPenaltyMs: 0})
	filter.randFloat = func() float64 { return 0.0 }
	endpointA := newTestEndpoint("cold-a", "10.0.61.2")
	endpointB := newTestEndpoint("cold-b", "10.0.61.3")

	filtered := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{endpointA, endpointB})

	assertEndpointsEqual(t, filtered, []fwkscheduling.Endpoint{endpointA, endpointB})
}

func TestFilterBreaksStickinessWhenTTFTPenaltyTooHigh(t *testing.T) {
	filter := New("filter", Config{Threshold: 0.5, PFilter: 1.0, MaxTTFTPenaltyMs: 5000})
	filter.randFloat = func() float64 { return 0.0 }
	sticky := newTestEndpoint("sticky", "10.0.62.2")
	nonSticky := newTestEndpoint("cold", "10.0.62.3")
	setPrefixScore(sticky, 4, 4)
	setPrediction(sticky, 12000, 1)
	setPrediction(nonSticky, 1000, 1)

	filtered := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{sticky, nonSticky})

	assertEndpointsEqual(t, filtered, []fwkscheduling.Endpoint{sticky, nonSticky})
}

func TestFilterKeepsStickyAndProbabilisticallyFiltersNonSticky(t *testing.T) {
	filter := New("filter", Config{Threshold: 0.5, PFilter: 0.8, MaxTTFTPenaltyMs: 5000})
	randValues := []float64{0.2, 0.95}
	filter.randFloat = func() float64 {
		value := randValues[0]
		randValues = randValues[1:]
		return value
	}
	sticky := newTestEndpoint("sticky", "10.0.62.6")
	nonStickyFiltered := newTestEndpoint("cold-filtered", "10.0.62.7")
	nonStickyKept := newTestEndpoint("cold-kept", "10.0.62.8")
	setPrefixScore(sticky, 4, 4)

	filtered := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{sticky, nonStickyFiltered, nonStickyKept})

	assertEndpointsEqual(t, filtered, []fwkscheduling.Endpoint{sticky, nonStickyKept})
}

func TestFilterDoesNotBreakStickinessWhenTTFTGateDisabled(t *testing.T) {
	filter := New("filter", Config{Threshold: 0.5, PFilter: 1.0, MaxTTFTPenaltyMs: 0})
	filter.randFloat = func() float64 { return 0.0 }
	sticky := newTestEndpoint("sticky", "10.0.62.4")
	nonSticky := newTestEndpoint("cold", "10.0.62.5")
	setPrefixScore(sticky, 4, 4)
	setPrediction(sticky, 12000, 1)
	setPrediction(nonSticky, 1000, 1)

	filtered := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{sticky, nonSticky})

	assertEndpointsEqual(t, filtered, []fwkscheduling.Endpoint{sticky})
}

func TestFilterKeepsStickinessWhenTTFTPenaltyBelowThreshold(t *testing.T) {
	filter := New("filter", Config{Threshold: 0.5, PFilter: 1.0, MaxTTFTPenaltyMs: 5000})
	filter.randFloat = func() float64 { return 0.0 }
	sticky := newTestEndpoint("sticky", "10.0.62.9")
	nonSticky := newTestEndpoint("cold", "10.0.62.10")
	setPrefixScore(sticky, 4, 4)
	setPrediction(sticky, 4500, 1)
	setPrediction(nonSticky, 1000, 1)

	filtered := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{sticky, nonSticky})

	assertEndpointsEqual(t, filtered, []fwkscheduling.Endpoint{sticky})
}

func TestFilterSkipsTTFTGateWhenPredictionsMissing(t *testing.T) {
	filter := New("filter", Config{Threshold: 0.5, PFilter: 1.0, MaxTTFTPenaltyMs: 5000})
	filter.randFloat = func() float64 { return 0.0 }
	sticky := newTestEndpoint("sticky", "10.0.63.2")
	nonSticky := newTestEndpoint("cold", "10.0.63.3")
	setPrefixScore(sticky, 4, 4)

	filtered := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{sticky, nonSticky})

	assertEndpointsEqual(t, filtered, []fwkscheduling.Endpoint{sticky})
}

func TestFactoryValidatesParameters(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantErr    string
	}{
		{name: "threshold out of range", parameters: `{"threshold":1.5}`, wantErr: "threshold must be between 0 and 1"},
		{name: "pFilter out of range", parameters: `{"pFilter":-0.1}`, wantErr: "pFilter must be between 0 and 1"},
		{name: "negative ttft penalty", parameters: `{"maxTTFTPenaltyMs":-1}`, wantErr: "maxTTFTPenaltyMs must be greater than or equal to 0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Factory("filter", json.RawMessage(tt.parameters), nil)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestFactoryPreservesZeroTTFTPenalty(t *testing.T) {
	plugin, err := Factory("filter", json.RawMessage(`{"threshold":0.5,"pFilter":1.0,"maxTTFTPenaltyMs":0}`), nil)
	if err != nil {
		t.Fatalf("Factory returned error: %v", err)
	}
	filter, ok := plugin.(*Filter)
	if !ok {
		t.Fatalf("expected *Filter, got %T", plugin)
	}
	if filter.maxTTFTPenaltyMs != 0 {
		t.Fatalf("expected maxTTFTPenaltyMs to remain 0, got %v", filter.maxTTFTPenaltyMs)
	}
}

func TestFilterStatisticallyMatchesPFilter(t *testing.T) {
	const runs = 10000
	const pFilter = 0.8
	filter := New("filter", Config{Threshold: 0.5, PFilter: pFilter, MaxTTFTPenaltyMs: 0})
	seed := uint64(1)
	filter.randFloat = func() float64 {
		seed = seed*6364136223846793005 + 1
		return float64(seed>>11) / float64(uint64(1)<<53)
	}
	sticky := newTestEndpoint("sticky", "10.0.62.11")
	setPrefixScore(sticky, 4, 4)
	nonSticky := newTestEndpoint("cold", "10.0.62.11")
	filteredCount := 0
	for i := 0; i < runs; i++ {
		result := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{sticky, nonSticky})
		if len(result) == 1 {
			filteredCount++
		}
	}
	got := float64(filteredCount) / runs
	if math.Abs(got-pFilter) > 0.03 {
		t.Fatalf("expected filtered probability close to %v, got %v", pFilter, got)
	}
}

func TestFilterSupportsConcurrentUse(t *testing.T) {
	filter := New("filter", Config{Threshold: 0.5, PFilter: 0.5, MaxTTFTPenaltyMs: 0})
	sticky := newTestEndpoint("sticky", "10.0.62.12")
	nonSticky := newTestEndpoint("cold", "10.0.62.13")
	setPrefixScore(sticky, 4, 4)

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				_ = filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{sticky, nonSticky})
			}
		}()
	}
	wg.Wait()
}

func TestFilterUsesBestPrefillPrefixScoreForPDLeaders(t *testing.T) {
	filter := New("filter", Config{Threshold: 0.5, PFilter: 1.0, MaxTTFTPenaltyMs: 0})
	leaderSticky := newTestEndpoint("leader-sticky", "10.0.64.2")
	leaderCold := newTestEndpoint("leader-cold", "10.0.64.3")

	prefillHot := newTestEndpoint("prefill-hot", "10.0.64.4")
	prefillCold := newTestEndpoint("prefill-cold", "10.0.64.5")
	decodeSticky := newTestEndpoint("decode-sticky", "10.0.64.6")
	decodeCold := newTestEndpoint("decode-cold", "10.0.64.7")
	setPrefixScore(prefillHot, 8, 10)
	setPrefixScore(prefillCold, 1, 10)

	pdgroup.Set(leaderSticky, &pdgroup.Info{
		GroupID: "group-sticky",
		PrefillPods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: prefillHot, Score: 0.1},
			{Endpoint: prefillCold, Score: 0.2},
		},
		DecodePods: []fwkscheduling.ScoredEndpoint{{Endpoint: decodeSticky, Score: 0.3}},
	})
	pdgroup.Set(leaderCold, &pdgroup.Info{
		GroupID:     "group-cold",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: newTestEndpoint("prefill-cold-2", "10.0.64.8"), Score: 0.1}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decodeCold, Score: 0.3}},
	})

	filtered := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leaderSticky, leaderCold})
	assertEndpointPodNamesEqual(t, filtered, []string{"leader-sticky"})
}

func TestFilterUsesBestPrefillTTFTForPDLeaders(t *testing.T) {
	filter := New("filter", Config{Threshold: 0.5, PFilter: 1.0, MaxTTFTPenaltyMs: 5000})
	leaderSticky := newTestEndpoint("leader-sticky", "10.0.65.2")
	leaderCold := newTestEndpoint("leader-cold", "10.0.65.3")

	prefillSticky := newTestEndpoint("prefill-sticky", "10.0.65.4")
	prefillCold := newTestEndpoint("prefill-cold", "10.0.65.5")
	decodeSticky := newTestEndpoint("decode-sticky", "10.0.65.6")
	decodeCold := newTestEndpoint("decode-cold", "10.0.65.7")

	setPrefixScore(prefillSticky, 9, 10)
	setPrediction(prefillSticky, 12000, 1)
	setPrediction(prefillCold, 1000, 1)

	pdgroup.Set(leaderSticky, &pdgroup.Info{
		GroupID:     "group-sticky",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefillSticky, Score: 0.1}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decodeSticky, Score: 0.2}},
	})
	pdgroup.Set(leaderCold, &pdgroup.Info{
		GroupID:     "group-cold",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefillCold, Score: 0.1}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decodeCold, Score: 0.2}},
	})

	filtered := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leaderSticky, leaderCold})
	assertEndpointPodNamesEqual(t, filtered, []string{"leader-sticky", "leader-cold"})
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

func setPrefixScore(endpoint fwkscheduling.Endpoint, matchBlocks, totalBlocks int) {
	endpoint.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(matchBlocks, totalBlocks, 16))
}

func setPrediction(endpoint fwkscheduling.Endpoint, ttft, tpot float64) {
	endpoint.Put(latencyattr.LatencyPredictionInfoKey, latencyattr.NewLatencyPredictionInfo(true, true, 0, 0, ttft, tpot))
}

func assertEndpointsEqual(t *testing.T, got, want []fwkscheduling.Endpoint) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("expected %d endpoints, got %d", len(want), len(got))
	}
	for i := range want {
		if !fwkscheduling.EndpointComparer(got[i], want[i]) {
			t.Fatalf("unexpected endpoint at index %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func assertEndpointPodNamesEqual(t *testing.T, got []fwkscheduling.Endpoint, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("expected %d endpoints, got %d", len(want), len(got))
	}
	for i := range want {
		if podName := endpointmeta.EndpointPodName(got[i]); podName != want[i] {
			t.Fatalf("unexpected endpoint at index %d: got %q want %q", i, podName, want[i])
		}
	}
}
