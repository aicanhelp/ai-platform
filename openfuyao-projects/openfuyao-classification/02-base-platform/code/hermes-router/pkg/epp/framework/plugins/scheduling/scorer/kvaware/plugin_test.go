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

package kvaware

import (
	"context"
	"encoding/json"
	"testing"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/testutil"
)

func TestFactoryDefaultsAndName(t *testing.T) {
	plugin, err := Factory("agg-kv", json.RawMessage(`{}`), fwkplugin.NewEppHandle(context.Background(), nil))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	scorer, ok := plugin.(*KVCacheAwareScorer)
	if !ok {
		t.Fatalf("expected *KVCacheAwareScorer, got %T", plugin)
	}
	if scorer.typedName.Name != "agg-kv" {
		t.Fatalf("expected name agg-kv, got %q", scorer.typedName.Name)
	}
	if scorer.params.KVCacheNotHitRateWeight != defaultKVCacheNotHitRateWeight {
		t.Fatalf("expected default KV weight %v, got %v", defaultKVCacheNotHitRateWeight, scorer.params.KVCacheNotHitRateWeight)
	}
}

func TestScoreUsesPrefixCacheAndNormalizedMetrics(t *testing.T) {
	scorer := New("agg-kv", Parameters{
		KVCacheNotHitRateWeight: 1.0,
		XPUCacheUsageWeight:     2.0,
		WaitingRequestWeight:    3.0,
	})
	podA := testutil.NewScoredEndpoint("pod-a", "10.0.0.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 1, KVUsage: 0.2, Running: 0}, nil).Endpoint
	podB := testutil.NewScoredEndpoint("pod-b", "10.0.0.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 4, KVUsage: 0.4, Running: 0}, nil).Endpoint
	setPrefixScore(podA, 8, 10)
	setPrefixScore(podB, 5, 10)

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), testutil.NewCompletionsRequest("req-1", "model", "hello world"), []fwkscheduling.Endpoint{podA, podB})

	// Weighted average badness over 4 signals (inflight weight 0 → skipped):
	//   podA: (1·0.2 + 2·0.2 + 3·0.25) / 6 = 0.225 → score 0.775
	//   podB: (1·0.5 + 2·0.4 + 3·1.0) / 6 ≈ 0.7167 → score ≈ 0.2833
	testutil.AssertFloatClose(t, scores[podA], 0.775)
	testutil.AssertFloatClose(t, scores[podB], 1.0-4.3/6.0)
	if scores[podA] <= scores[podB] {
		t.Fatalf("expected pod-a score %v to be greater than pod-b score %v", scores[podA], scores[podB])
	}
}

func TestScoreBypassesPDSeparatedEndpoints(t *testing.T) {
	scorer := New("agg-kv", Parameters{
		KVCacheNotHitRateWeight: 1.0,
		XPUCacheUsageWeight:     0.0,
		WaitingRequestWeight:    0.0,
	})

	aggregate := testutil.NewScoredEndpoint("aggregate", "10.0.0.9", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	leader := testutil.NewScoredEndpoint("leader", "10.0.0.10", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	prefill := testutil.NewScoredEndpoint("prefill-hot", "10.0.0.12", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	decode := testutil.NewScoredEndpoint("decode-hot", "10.0.0.14", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	setPrefixScore(aggregate, 8, 10)
	setPrefixScore(prefill, 9, 10)

	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:     "group-hot",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0}},
	})

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{aggregate, leader})

	if len(scores) != 1 {
		t.Fatalf("expected only aggregate endpoint to be scored, got %d entries", len(scores))
	}
	if _, ok := scores[leader]; ok {
		t.Fatal("expected PD-separated endpoint to be bypassed")
	}
	// Only KV weight is active; badness = 0.2 / 1 = 0.2; score = 1 - 0.2 = 0.8.
	testutil.AssertFloatClose(t, scores[aggregate], 0.8)
}

func TestScoreTreatsMissingPrefixInfoAsColdCache(t *testing.T) {
	scorer := New("agg-kv", defaultParameters())
	warm := testutil.NewScoredEndpoint("warm", "10.0.0.3", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	cold := testutil.NewScoredEndpoint("cold", "10.0.0.4", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	setPrefixScore(warm, 10, 10)

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{warm, cold})

	// All weights = 1. Warm pod has prefix=1.0 so all signals 0 → score 1.
	// Cold pod has prefix=0 so kv-not-hit signal contributes 1, others 0;
	// badness = 1/4 = 0.25 → score = 0.75.
	testutil.AssertFloatClose(t, scores[warm], 1.0)
	testutil.AssertFloatClose(t, scores[cold], 0.75)
}

func setPrefixScore(endpoint fwkscheduling.Endpoint, matchBlocks, totalBlocks int) {
	endpoint.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(matchBlocks, totalBlocks, 16))
}
