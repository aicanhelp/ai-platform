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

package pdkvaware

import (
	"context"
	"encoding/json"
	"math/rand"
	"testing"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/score"
	"hermes-router/pkg/epp/internal/testutil"
)

func TestFactoryRejectsInvalidWeights(t *testing.T) {
	_, err := Factory("pd-kv", json.RawMessage(`{"prefillScoreWeight":0,"decodeScoreWeight":0}`), nil)
	if err == nil {
		t.Fatal("expected validation error")
	}
}

func TestFactoryAcceptsKVUsageWeightNames(t *testing.T) {
	plugin, err := Factory("pd-kv", json.RawMessage(`{
		"prefillKVUsageWeight":1,
		"prefillPrefixWeight":1,
		"prefillQueueWeight":1,
		"prefillInflightWeight":1,
		"decodeKVUsageWeight":1,
		"decodeQueueWeight":1,
		"decodeInflightWeight":1,
		"prefillScoreWeight":1,
		"decodeScoreWeight":1
	}`), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	scorer, ok := plugin.(*PDKVCacheAwareScorer)
	if !ok {
		t.Fatalf("expected *PDKVCacheAwareScorer, got %T", plugin)
	}
	if scorer.config.Prefill.KVWeight != 1 || scorer.config.Decode.KVWeight != 1 {
		t.Fatalf("expected KV usage weights to be applied, got %+v", scorer.config)
	}
}

func TestFactoryKeepsOmittedDecodeRunningWeightDisabled(t *testing.T) {
	plugin, err := Factory("pd-kv", json.RawMessage(`{
		"prefillKVUsageWeight":1,
		"prefillPrefixWeight":1,
		"prefillQueueWeight":1,
		"prefillInflightWeight":1,
		"decodeKVUsageWeight":1,
		"decodeQueueWeight":1,
		"decodeInflightWeight":1,
		"prefillScoreWeight":1,
		"decodeScoreWeight":1
	}`), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	scorer, ok := plugin.(*PDKVCacheAwareScorer)
	if !ok {
		t.Fatalf("expected *PDKVCacheAwareScorer, got %T", plugin)
	}
	if scorer.config.Decode.RunningWeight != 0 {
		t.Fatalf("expected omitted decodeRunningWeight to stay disabled, got %v", scorer.config.Decode.RunningWeight)
	}
	if scorer.config.Decode.KVWeight != 1 || scorer.config.Decode.QueueWeight != 1 || scorer.config.Decode.InflightWeight != 1 {
		t.Fatalf("expected decode KV/queue/inflight weights to stay enabled, got %+v", scorer.config)
	}
}

func TestScoreSelectsBestPrefillAndDecode(t *testing.T) {
	leader := newTestEndpoint("leader", 0, 0, 0)
	prefillFast := newTestEndpoint("prefill-fast", 1, 10, 1)
	prefillSlow := newTestEndpoint("prefill-slow", 5, 70, 2)
	decodeFast := newTestEndpoint("decode-fast", 1, 10, 1)
	decodeSlow := newTestEndpoint("decode-slow", 6, 80, 8)
	prefillFast.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(9, 10, 16))
	prefillSlow.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(1, 10, 16))
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID: "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: prefillFast, Score: 0},
			{Endpoint: prefillSlow, Score: 0},
		},
		DecodePods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: decodeFast, Score: 0},
			{Endpoint: decodeSlow, Score: 0},
		},
	})

	scorer := New("pd-kv", Config{})
	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leader})
	if len(scores) != 1 {
		t.Fatalf("expected one scored endpoint, got %d", len(scores))
	}
	if scores[leader] <= 0 {
		t.Fatalf("expected positive leader score, got %v", scores[leader])
	}
	info, ok := pdgroup.Get(leader)
	if !ok {
		t.Fatal("expected leader Info after scoring")
	}
	if info.SelectedPrefillPod == nil || !fwkscheduling.EndpointComparer(info.SelectedPrefillPod.Endpoint, prefillFast) {
		t.Fatal("expected fastest prefill pod to be selected")
	}
	if info.SelectedDecodePod == nil || !fwkscheduling.EndpointComparer(info.SelectedDecodePod.Endpoint, decodeFast) {
		t.Fatal("expected fastest decode pod to be selected")
	}
}

func TestScoreBypassesAggregateAndInvalidPDEndpointsIndividually(t *testing.T) {
	leader := newTestEndpoint("leader", 0, 0, 0)
	prefill := newTestEndpoint("prefill", 1, 10, 1)
	decode := newTestEndpoint("decode", 1, 10, 1)
	aggregate := newTestEndpoint("aggregate", 3, 30, 2)
	invalidLeader := newTestEndpoint("invalid", 0, 0, 0)
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0}},
	})
	pdgroup.Set(invalidLeader, &pdgroup.Info{GroupID: "broken"})

	scorer := New("pd-kv", Config{})
	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leader, aggregate, invalidLeader})
	if len(scores) != 1 {
		t.Fatalf("expected only the valid PD leader to be scored, got %d entries", len(scores))
	}
	if _, ok := scores[aggregate]; ok {
		t.Fatal("expected aggregate endpoint to be bypassed")
	}
	if _, ok := scores[invalidLeader]; ok {
		t.Fatal("expected invalid PD endpoint to be bypassed")
	}
	if _, ok := scores[leader]; !ok {
		t.Fatal("expected PD leader score to be present")
	}
}

func TestScoreCalculatesExpectedGroupScore(t *testing.T) {
	leader := newTestEndpoint("leader", 0, 0, 0)
	prefillFast := newTestEndpoint("prefill-fast", 1, 0, 0)
	prefillSlow := newTestEndpoint("prefill-slow", 4, 0, 0)
	decodeFast := newTestEndpoint("decode-fast", 1, 0, 0)
	decodeSlow := newTestEndpoint("decode-slow", 4, 0, 0)
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID: "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: prefillFast, Score: 0},
			{Endpoint: prefillSlow, Score: 0},
		},
		DecodePods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: decodeFast, Score: 0},
			{Endpoint: decodeSlow, Score: 0},
		},
	})

	scorer := New("pd-kv", Config{
		Prefill: score.PrefillWeights{QueueWeight: 1},
		Decode:  score.DecodeWeights{QueueWeight: 1},
		Group:   score.GroupWeights{PrefillScoreWeight: 1, DecodeScoreWeight: 1},
	})
	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leader})
	// Only queue weights active. PrefillPods queue badnesses: 1/4=0.25 and 4/4=1.
	// Best prefill badness = 0.25 → score = 0.75. Same for decode.
	// Group badness = (0.5·0.25 + 0.5·0.25) / 1 = 0.25 → score = 0.75.
	testutil.AssertFloatClose(t, scores[leader], 0.75)

	info, ok := pdgroup.Get(leader)
	if !ok {
		t.Fatal("expected leader Info after scoring")
	}
	testutil.AssertFloatClose(t, info.PrefillPods[0].Score, 0.75)
	testutil.AssertFloatClose(t, info.PrefillPods[1].Score, 0.0)
	testutil.AssertFloatClose(t, info.DecodePods[0].Score, 0.75)
	testutil.AssertFloatClose(t, info.DecodePods[1].Score, 0.0)
	if info.SelectedPrefillPod == nil || !fwkscheduling.EndpointComparer(info.SelectedPrefillPod.Endpoint, prefillFast) {
		t.Fatal("expected fastest prefill pod to be selected")
	}
	if info.SelectedDecodePod == nil || !fwkscheduling.EndpointComparer(info.SelectedDecodePod.Endpoint, decodeFast) {
		t.Fatal("expected fastest decode pod to be selected")
	}
}

// TestScoreNormalizesQueueAcrossGroups guards against per-group queue
// normalization. With one prefill pod per group (1P1D), a group-local max makes
// every busy pod its own max, so queue badness collapses to 1.0 and two groups
// with different queue depths become indistinguishable. Pool-wide normalization
// keeps them comparable: the shallower queue must score strictly better.
func TestScoreNormalizesQueueAcrossGroups(t *testing.T) {
	leaderShallow := newTestEndpoint("leader-shallow", 0, 0, 0)
	leaderDeep := newTestEndpoint("leader-deep", 0, 0, 0)
	prefillShallow := newTestEndpoint("prefill-shallow", 2, 0, 0)
	prefillDeep := newTestEndpoint("prefill-deep", 10, 0, 0)
	decodeShallow := newTestEndpoint("decode-shallow", 0, 0, 0)
	decodeDeep := newTestEndpoint("decode-deep", 0, 0, 0)
	pdgroup.Set(leaderShallow, &pdgroup.Info{
		GroupID:     "group-shallow",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefillShallow, Score: 0}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decodeShallow, Score: 0}},
	})
	pdgroup.Set(leaderDeep, &pdgroup.Info{
		GroupID:     "group-deep",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefillDeep, Score: 0}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decodeDeep, Score: 0}},
	})

	// Only the prefill queue signal drives the group score (decode score weight 0).
	scorer := New("pd-kv", Config{
		Prefill: score.PrefillWeights{QueueWeight: 1},
		Decode:  score.DecodeWeights{QueueWeight: 1},
		Group:   score.GroupWeights{PrefillScoreWeight: 1, DecodeScoreWeight: 0},
	})
	scores := scorer.Score(
		context.Background(),
		fwkscheduling.NewCycleState(),
		nil,
		[]fwkscheduling.Endpoint{leaderShallow, leaderDeep},
	)
	// Pool-wide max queue = 10. Shallow badness = 2/10 = 0.2 → score 0.8.
	// Deep badness = 10/10 = 1.0 → score 0.0. Per-group normalization would tie
	// both at 1.0 → score 0.0.
	testutil.AssertFloatClose(t, scores[leaderShallow], 0.8)
	testutil.AssertFloatClose(t, scores[leaderDeep], 0.0)
	if scores[leaderShallow] <= scores[leaderDeep] {
		t.Fatalf("expected shallow-queue group to outscore deep-queue group, got %v vs %v",
			scores[leaderShallow], scores[leaderDeep])
	}
}

func TestScoreRandomizesEqualPDMembers(t *testing.T) {
	leader := newTestEndpoint("leader-equal", 0, 0, 0)
	prefillFirst := newTestEndpoint("prefill-first", 2, 20, 1)
	prefillSecond := newTestEndpoint("prefill-second", 2, 20, 1)
	decodeFirst := newTestEndpoint("decode-first", 3, 30, 4)
	decodeSecond := newTestEndpoint("decode-second", 3, 30, 4)
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID: "group-equal",
		PrefillPods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: prefillFirst, Score: 0},
			{Endpoint: prefillSecond, Score: 0},
		},
		DecodePods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: decodeFirst, Score: 0},
			{Endpoint: decodeSecond, Score: 0},
		},
	})

	scorer := New("pd-kv", Config{})
	scorer.rng = rand.New(rand.NewSource(1))
	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leader})
	if len(scores) != 1 {
		t.Fatalf("expected one scored endpoint, got %d", len(scores))
	}

	info, ok := pdgroup.Get(leader)
	if !ok {
		t.Fatal("expected leader Info after scoring")
	}
	if info.SelectedPrefillPod == nil || !fwkscheduling.EndpointComparer(info.SelectedPrefillPod.Endpoint, prefillSecond) {
		t.Fatal("expected second tied prefill pod to be selected")
	}
	if info.SelectedDecodePod == nil || !fwkscheduling.EndpointComparer(info.SelectedDecodePod.Endpoint, decodeSecond) {
		t.Fatal("expected second tied decode pod to be selected")
	}
}

func newTestEndpoint(name string, waiting int, kvUsage float64, running int) fwkscheduling.Endpoint {
	return testutil.NewScoredEndpoint(name, name+".svc.cluster.local", "8080", testutil.ScoredEndpointMetrics{Score: 0, Waiting: waiting, KVUsage: kvUsage, Running: running}, nil).Endpoint
}
